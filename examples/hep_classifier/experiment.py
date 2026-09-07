"""The HEP classifier study: reusable profiled-D_s bins, evaluated out of sample.

The arc, in the order the walkthrough tells it:

* an **in-sample study** on the whole cross-fitted score table: the finite
  profiled-\\(D_s\\) partition at the headline budget, its certified
  `efficient_score_bound` ceiling, a sweep across bin budgets, the
  classifier-output baselines a physicist would reach for, and a three-point
  `delta` convergence check. Every labeling is scored both ways, full-D and
  profiled D_s, because that disagreement is the point;
* a **cross-evaluation**: the events are split once into two halves, every
  *reusable* rule is built on one half and applied with `predict_scores`
  (or with edges chosen on that half) to the other, in both directions. The
  held-out profiled retention carries a percentile bootstrap interval and is
  certified by the evaluation half's own ceiling;
* a **downstream signal-strength fit**: each reusable rule is applied to the
  events' `tes`-shifted copies, giving yield templates
  ``N_b(mu, nu, tes) = mu S_b(tes) + nu B_b(tes)``, and the Asimov binned
  Poisson likelihood then gives the expected relative uncertainty on the
  signal strength with the background rate and the energy scale floating.

The central claim is a *prediction*: measure it, and report whatever the run
produces, including a small, zero, or reversed gap.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import scorequant as sq
from examples._env import example_scale

from .data import TES_POINTS, HepData, load_fixture
from .scores import (
    INTEREST,
    SCHEMA,
    SignalBackgroundOOF,
    TesOOF,
    assemble_score_sample,
    event_folds,
    fit_signal_background_oof,
    fit_tes_oof,
    out_of_fold_scores,
    tes_score_reliability,
)

CELLS_FIGURE_PATH = Path("docs/examples/assets/hep-cells.png")
BUDGET_FIGURE_PATH = Path("docs/examples/assets/hep-budget.png")
METRICS_PATH = Path("docs/examples/assets/hep-classifier.json")

#: Bin budget of the headline comparison.
HEADLINE_BINS = 6
#: Bin budgets swept against the certified profiled ceiling.
BUDGET_SWEEP = (3, 4, 6, 8)
#: Headline finite-difference half-offset and the three-point convergence
#: study around it, matching the fixture's committed tes points.
HEADLINE_DELTA = 0.05
DELTA_SWEEP = (0.025, 0.05, 0.10)
#: Seed shared by every finite-D and soft solver in the study.
SOLVER_SEED = 11
#: Base seed for the deterministic event-fold assignment.
FOLD_SEED = 2026
#: Seed of the one stratified half/half event split the cross-evaluation uses.
SPLIT_SEED = 7
#: Seed and size of the percentile bootstrap over evaluation events.
BOOTSTRAP_SEED = 2027
BOOTSTRAP_REPLICATES = 200
#: Fewest simulated background events an interval of the classifier-output
#: baseline may hold. Without it the interval search isolates background-free
#: pockets of a handful of signal events, which look infinitely informative
#: and are a Monte Carlo artefact; requiring a minimum simulated background
#: count per bin is the usual analysis practice.
MIN_BACKGROUND_EVENTS = 10
#: Width of the Gaussian `tes` constraint recorded next to the unconstrained
#: downstream uncertainty. Nothing in the criterion or the retention numbers
#: uses it; it exists so the page can say what a physical constraint would
#: change here.
TES_CONSTRAINT_WIDTH = 0.01

type MetricRow = dict[str, object]


# --------------------------------------------------------------------------- #
# Scoring one labeling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LabelingScore:
    """Both retention numbers a labeling earns, and the binned profiled information.

    Attributes
    ----------
    full_retention
        Geometric-mean retention of the whole three-parameter Fisher matrix.
    profiled_retention
        Geometric-mean retention of the profiled information of
        `mu_htautau` alone.
    profiled_information
        The scalar binned Schur complement itself.
    """

    full_retention: float
    profiled_retention: float
    profiled_information: float


def score_labeling(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, *, n_bins: int
) -> LabelingScore:
    """Score one hard labeling on both the full and the profiled criterion."""
    full = sq.information_report(scores, labels, weights, n_bins=n_bins)
    profiled = sq.profiled_information_report(
        scores, labels, interest=INTEREST, weights=weights, n_bins=n_bins
    )
    return LabelingScore(
        full_retention=float(full.geometric_mean_retention),
        profiled_retention=float(profiled.geometric_mean_retention),
        profiled_information=float(np.asarray(profiled.schur_binned)[0, 0]),
    )


def try_score_labeling(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, *, n_bins: int
) -> LabelingScore | None:
    """Score a labeling, or return ``None`` when the binned model is not identified.

    A labeling with fewer occupied cells than parameters cannot determine
    the nuisance block, and the library refuses to profile a singular one.
    On the evaluation half that refusal is a finding about the rule, not an
    error in the study, so it is recorded rather than raised.
    """
    try:
        return score_labeling(scores, labels, weights, n_bins=n_bins)
    except sq.ContractError:
        return None


def unbinned_profiled_information(scores: np.ndarray, weights: np.ndarray) -> float:
    """Return the unbinned profiled information of `mu_htautau`.

    This is the certified ceiling's reference denominator.
    """
    information = np.asarray(sq.fisher_information(scores, weights))
    nuisance = [index for index in range(information.shape[0]) if index not in set(INTEREST)]
    interest_indices = list(INTEREST)
    block = information[np.ix_(interest_indices, interest_indices)]
    cross = information[np.ix_(interest_indices, nuisance)]
    nuisance_block = information[np.ix_(nuisance, nuisance)]
    schur = block - cross @ np.linalg.solve(nuisance_block, cross.T)
    return float(schur[0, 0])


def ceiling_retention(scores: np.ndarray, weights: np.ndarray, *, n_bins: int) -> float:
    """Return the certified profiled ceiling of a sample as a retention."""
    bound = sq.efficient_score_bound(scores, interest=INTEREST, weights=weights, n_bins=n_bins)
    reference = unbinned_profiled_information(scores, weights)
    return float(np.exp(bound.upper_bound - np.log(reference)))


# --------------------------------------------------------------------------- #
# Rules: anything that labels events it has never seen
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EventTable:
    """One set of events with everything a rule or a scorer needs.

    Attributes
    ----------
    scores
        Out-of-fold score table, shape ``[N, 3]`` in `SCHEMA` order.
    posterior
        Calibrated out-of-fold signal posterior, shape ``[N]``.
    weights
        Monte Carlo weights, shape ``[N]``.
    is_signal
        Signal label, shape ``[N]``.
    """

    scores: np.ndarray
    posterior: np.ndarray
    weights: np.ndarray
    is_signal: np.ndarray

    @property
    def n_events(self) -> int:
        """Number of events in the table."""
        return int(self.weights.shape[0])

    def subset(self, mask: np.ndarray) -> EventTable:
        """Return the rows selected by a boolean mask."""
        return EventTable(
            self.scores[mask], self.posterior[mask], self.weights[mask], self.is_signal[mask]
        )

    def sample(self) -> sq.ScoreSample:
        """Wrap the table as a `ScoreSample` in the study's schema."""
        return sq.ScoreSample(
            self.scores,
            self.weights,
            schema=SCHEMA,
            provenance=sq.ScoreProvenance(kind="estimated_ratio"),
        )


@dataclass(frozen=True, slots=True)
class BinRule:
    """A reusable labeling rule: a name, a budget, and a function of new events.

    Attributes
    ----------
    key, label
        Stable identifier and published name.
    kind
        ``"scorequant"`` for a rule built by the library, ``"classifier"``
        for a binning of the signal classifier's output, ``"conventional"``
        for the two-bin significance cut.
    n_bins
        Number of cells the rule can emit.
    assign
        Maps a table of new events to integer labels in ``[0, n_bins)``.
    """

    key: str
    label: str
    kind: str
    n_bins: int
    assign: Callable[[EventTable], np.ndarray] = field(repr=False)


def _logit(posterior: np.ndarray) -> np.ndarray:
    clipped = np.clip(posterior, 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def quantile_rule(reference: EventTable, n_bins: int) -> BinRule:
    """Equal-count cells of the signal posterior, edges chosen on the reference events.

    "Bin the network output" -- the first thing a physicist reaches for.
    The cells hold equal numbers of simulated events, not equal weighted
    yield.
    """
    edges = np.quantile(reference.posterior, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return BinRule(
        "classifier_quantile",
        "Classifier quantile bins",
        "classifier",
        n_bins,
        lambda table: np.digitize(table.posterior, edges),
    )


def logit_equal_width_rule(reference: EventTable, n_bins: int) -> BinRule:
    """Equal-width cells of the posterior's logit, edges chosen on the reference events.

    Equal-width cells in the posterior itself waste most of their range,
    because a calibrated posterior on a 0.1% signal fraction piles up near
    zero; the logit spreads that pile out.
    """
    logit = _logit(reference.posterior)
    edges = np.linspace(logit.min(), logit.max(), n_bins + 1)[1:-1]
    return BinRule(
        "classifier_logit_equal_width",
        "Classifier logit, equal width",
        "classifier",
        n_bins,
        lambda table: np.digitize(_logit(table.posterior), edges),
    )


def significance_interval_edges(
    posterior: np.ndarray,
    weights: np.ndarray,
    is_signal: np.ndarray,
    n_bins: int,
    *,
    min_background_events: int = MIN_BACKGROUND_EVENTS,
) -> np.ndarray:
    """Choose contiguous posterior intervals maximizing ``sum_b S_b^2 / (S_b + B_b)``.

    This is the strongest classifier-only binning a physicist would build:
    the objective is the Asimov Fisher information for the signal strength
    when every nuisance is held fixed, and the exact optimum over ordered
    intervals is found by dynamic programming over the sorted events, under
    the usual requirement that every interval holds at least
    ``min_background_events`` simulated background events.

    Parameters
    ----------
    posterior, weights, is_signal
        The reference events.
    n_bins
        Number of contiguous intervals.
    min_background_events
        Fewest simulated background events per interval.

    Returns
    -------
    numpy.ndarray
        Interior edges in posterior units, shape ``[n_bins - 1]``, placed
        midway between the neighbouring reference events.
    """
    order = np.argsort(posterior, kind="stable")
    sorted_posterior = posterior[order]
    signal = np.concatenate([[0.0], np.cumsum(np.where(is_signal[order], weights[order], 0.0))])
    total = np.concatenate([[0.0], np.cumsum(weights[order])])
    background_events = np.concatenate([[0], np.cumsum(~is_signal[order])])
    n_events = posterior.shape[0]
    # gain[i, j]: information of the interval covering sorted events i..j-1.
    signal_gain = signal[None, :] - signal[:, None]
    total_gain = total[None, :] - total[:, None]
    enough_background = (
        background_events[None, :] - background_events[:, None]
    ) >= min_background_events
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where((total_gain > 0) & enough_background, signal_gain**2 / total_gain, -np.inf)
    best = np.full((n_bins + 1, n_events + 1), -np.inf)
    argument = np.zeros((n_bins + 1, n_events + 1), dtype=np.int64)
    best[0, 0] = 0.0
    for cells in range(1, n_bins + 1):
        candidates = best[cells - 1][:, None] + gain
        argument[cells] = np.argmax(candidates, axis=0)
        best[cells] = candidates[argument[cells], np.arange(n_events + 1)]
    cuts: list[int] = []
    position = n_events
    for cells in range(n_bins, 0, -1):
        position = int(argument[cells, position])
        cuts.append(position)
    if not np.isfinite(best[n_bins, n_events]):
        raise ValueError("no interval binning satisfies the minimum background count")
    cuts = sorted(cut for cut in cuts if 0 < cut < n_events)
    return np.array(
        [0.5 * (sorted_posterior[cut - 1] + sorted_posterior[cut]) for cut in cuts], dtype=float
    )


def significance_interval_rule(reference: EventTable, n_bins: int) -> BinRule:
    """Contiguous posterior intervals maximizing ``sum S^2/(S+B)`` on the reference events."""
    edges = significance_interval_edges(
        reference.posterior, reference.weights, reference.is_signal, n_bins
    )
    return BinRule(
        "classifier_significance_intervals",
        "Classifier intervals, max S²/(S+B)",
        "classifier",
        n_bins,
        lambda table: np.digitize(table.posterior, edges),
    )


def threshold_cut_rule(reference: EventTable) -> BinRule:
    """The two-bin signal-region cut maximizing weighted S/sqrt(B) on the reference events.

    The most recognizable analysis there is, kept as a conventional
    reference. Two cells cannot identify three parameters, so its profiled
    retention is not a number but a statement that the model is not
    identified.
    """
    order = np.argsort(reference.posterior)
    sorted_posterior = reference.posterior[order]
    signal_weight = np.where(reference.is_signal[order], reference.weights[order], 0.0)
    background_weight = np.where(~reference.is_signal[order], reference.weights[order], 0.0)
    cumulative_signal = np.cumsum(signal_weight[::-1])[::-1]
    cumulative_background = np.cumsum(background_weight[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        significance = np.where(
            cumulative_background > 0,
            cumulative_signal / np.sqrt(cumulative_background),
            0.0,
        )
    threshold = float(sorted_posterior[int(np.argmax(significance))])
    return BinRule(
        "threshold_cut",
        "Significance cut, two bins",
        "conventional",
        2,
        lambda table: (table.posterior >= threshold).astype(np.int64),
    )


def fit_ds_rule(
    reference: EventTable,
    *,
    n_bins: int,
    soft_steps: int,
    validation: EventTable | None = None,
) -> tuple[BinRule, sq.QuantizerResult]:
    """Fit the reusable profiled-D_s rule with `fit_quantizer` on the reference events.

    Finite profiled-D_s labels have no compile bridge, so the deployable
    profiled rule is fitted as one: a soft Voronoi rule in score space,
    hardened at the end. A validation table, when given, is diagnostic only
    and never touches the fit.
    """
    result = sq.fit_quantizer(
        reference.sample(),
        validation=None if validation is None else validation.sample(),
        n_bins=n_bins,
        criterion=sq.ProfiledDOptimality(("mu_htautau",)),
        config=sq.SoftVoronoiConfig(
            seed=SOLVER_SEED,
            initializer_restarts=8,
            max_steps=soft_steps,
            record_every=max(soft_steps // 8, 1),
        ),
    )
    rule = BinRule(
        "ds_rule",
        "Profiled Ds rule",
        "scorequant",
        n_bins,
        lambda table: np.asarray(result.predict_scores(table.scores)),
    )
    return rule, result


def fit_d_rule(reference: EventTable, *, n_bins: int) -> tuple[BinRule, sq.PartitionResult]:
    """Fit the plain-D exchange partition and compile it into its Voronoi rule."""
    partition = sq.optimize_partition(
        reference.sample(),
        n_bins=n_bins,
        criterion=sq.DOptimality(),
        config=sq.DExchangeConfig(seed=SOLVER_SEED),
    )
    quantizer = partition.compile_quantizer()
    rule = BinRule(
        "d_rule",
        "Plain D rule",
        "scorequant",
        n_bins,
        lambda table: np.asarray(quantizer.predict_scores(table.scores)),
    )
    return rule, partition


def finite_ds_partition(reference: EventTable, *, n_bins: int) -> tuple[sq.PartitionResult, float]:
    """Optimize the finite profiled-D_s labels of a table, seeded by the ceiling's labels.

    Returns the partition and the certified ceiling (as a retention) on the
    same table. This is an in-sample object: it labels these rows and no
    others.
    """
    bound = sq.efficient_score_bound(
        reference.scores, interest=INTEREST, weights=reference.weights, n_bins=n_bins
    )
    partition = sq.optimize_partition(
        reference.sample(),
        n_bins=n_bins,
        criterion=sq.ProfiledDOptimality(("mu_htautau",)),
        config=sq.DExchangeConfig(seed=SOLVER_SEED),
        initial_labels=bound.labels,
    )
    reference_information = unbinned_profiled_information(reference.scores, reference.weights)
    ceiling = float(np.exp(bound.upper_bound - np.log(reference_information)))
    return partition, ceiling


# --------------------------------------------------------------------------- #
# Evaluating a rule on events it never saw
# --------------------------------------------------------------------------- #


def bin_occupancy(table: EventTable, labels: np.ndarray, *, n_bins: int) -> list[MetricRow]:
    """Describe each cell of a labeling: events, yield, signal share, mean `tes` score."""
    rows: list[MetricRow] = []
    for cell in range(n_bins):
        mask = labels == cell
        events = int(np.count_nonzero(mask))
        mass = float(np.sum(table.weights[mask]))
        signal_mass = float(np.sum(table.weights[mask & table.is_signal]))
        rows.append(
            {
                "cell": cell,
                "events": events,
                "signal_events": int(np.count_nonzero(mask & table.is_signal)),
                "weighted_yield": mass,
                "signal_fraction": signal_mass / mass if mass > 0 else None,
                "mean_tes_score": (
                    float(np.average(table.scores[mask, 2], weights=table.weights[mask]))
                    if mass > 0
                    else None
                ),
                "mean_posterior": (
                    float(np.average(table.posterior[mask], weights=table.weights[mask]))
                    if mass > 0
                    else None
                ),
            }
        )
    return rows


def bootstrap_profiled_retention(
    table: EventTable,
    labels: np.ndarray,
    *,
    n_bins: int,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Percentile bootstrap of a fixed labeling's profiled retention over events.

    Rows are resampled with replacement and the labels travel with them, so
    the interval reflects the Monte Carlo sample's own size and nothing
    about how the rule was fitted. Replicates whose binned nuisance block is
    singular are counted and dropped.
    """
    generator = np.random.default_rng(seed)
    values: list[float] = []
    dropped = 0
    for _ in range(replicates):
        index = generator.integers(0, table.n_events, table.n_events)
        scored = try_score_labeling(
            table.scores[index], labels[index], table.weights[index], n_bins=n_bins
        )
        if scored is None:
            dropped += 1
        else:
            values.append(scored.profiled_retention)
    if not values:
        return {"replicates": replicates, "dropped": dropped}
    array = np.asarray(values)
    return {
        "replicates": replicates,
        "dropped": dropped,
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def evaluate_rule(
    rule: BinRule,
    reference: EventTable,
    evaluation: EventTable,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> MetricRow:
    """Score a reusable rule on the events it was built from and on events it never saw."""
    reference_labels = rule.assign(reference)
    evaluation_labels = rule.assign(evaluation)
    reference_score = try_score_labeling(
        reference.scores, reference_labels, reference.weights, n_bins=rule.n_bins
    )
    evaluation_score = try_score_labeling(
        evaluation.scores, evaluation_labels, evaluation.weights, n_bins=rule.n_bins
    )
    occupied = int(len(np.unique(evaluation_labels)))
    identified = evaluation_score is not None and rule.n_bins > len(SCHEMA.parameters) - 1
    row: MetricRow = {
        "key": rule.key,
        "label": rule.label,
        "kind": rule.kind,
        "n_bins": rule.n_bins,
        "identified": identified,
        "reference_full_retention": None
        if reference_score is None
        else reference_score.full_retention,
        "reference_profiled_retention": None
        if reference_score is None
        else reference_score.profiled_retention,
        "evaluation_full_retention": None
        if evaluation_score is None
        else evaluation_score.full_retention,
        "evaluation_profiled_retention": None
        if evaluation_score is None or not identified
        else evaluation_score.profiled_retention,
        "evaluation_occupied_bins": occupied,
        "evaluation_min_bin_events": int(
            np.min(np.bincount(evaluation_labels, minlength=rule.n_bins))
        ),
        "evaluation_occupancy": bin_occupancy(evaluation, evaluation_labels, n_bins=rule.n_bins),
    }
    if identified:
        row["evaluation_bootstrap"] = bootstrap_profiled_retention(
            evaluation,
            evaluation_labels,
            n_bins=rule.n_bins,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
    return row


# --------------------------------------------------------------------------- #
# The event split and the two directions
# --------------------------------------------------------------------------- #


def event_split(is_signal: np.ndarray, *, seed: int = SPLIT_SEED) -> np.ndarray:
    """Split the events once into two stratified halves, returning a 0/1 half id."""
    return event_folds(is_signal, n_folds=2, seed=seed)


def cross_fitted_table(
    data: HepData, *, max_iter: int | None = None, n_folds: int | None = None
) -> EventTable:
    """Build the leakage-free score table of every event with one call.

    This is the walkthrough's entry point: the two classifiers are
    cross-fitted out of fold with one fold id per event, and the result is
    the three-column score table plus the calibrated signal posterior.

    Parameters
    ----------
    data
        The loaded fixture.
    max_iter, n_folds
        Classifier budget and fold count; default to the study's own
        (``example_scale(300, 60)`` and ``example_scale(5, 3)``).

    Returns
    -------
    EventTable
        Scores, posterior, weights and labels of every event.
    """
    max_iter = example_scale(300, 60) if max_iter is None else max_iter
    n_folds = example_scale(5, 3) if n_folds is None else n_folds
    fold_ids = event_folds(data.is_signal, n_folds=n_folds, seed=FOLD_SEED)
    sigbg = fit_signal_background_oof(
        data, fold_ids=fold_ids, max_iter=max_iter, seed=FOLD_SEED + 100
    )
    tes = fit_tes_oof(
        data, delta=HEADLINE_DELTA, fold_ids=fold_ids, max_iter=max_iter, seed=FOLD_SEED + 500
    )
    sample = assemble_score_sample(data, sigbg, tes)
    return EventTable(
        np.asarray(sample.scores), sigbg.probabilities[:, 1], data.weights, data.is_signal
    )


def build_rules(
    reference: EventTable,
    evaluation: EventTable | None,
    *,
    n_bins: int,
    soft_steps: int,
) -> tuple[list[BinRule], sq.QuantizerResult]:
    """Build every reusable rule on the reference events, in published order."""
    ds_rule, ds_result = fit_ds_rule(
        reference, n_bins=n_bins, soft_steps=soft_steps, validation=evaluation
    )
    d_rule, _ = fit_d_rule(reference, n_bins=n_bins)
    rules = [
        ds_rule,
        d_rule,
        significance_interval_rule(reference, n_bins),
        logit_equal_width_rule(reference, n_bins),
        quantile_rule(reference, n_bins),
        threshold_cut_rule(reference),
    ]
    return rules, ds_result


def cross_evaluate_direction(
    name: str,
    reference: EventTable,
    evaluation: EventTable,
    *,
    n_bins: int,
    soft_steps: int,
    bootstrap_replicates: int,
) -> tuple[MetricRow, dict[str, BinRule]]:
    """Build every rule on one half and score it on the other."""
    rules, ds_result = build_rules(reference, evaluation, n_bins=n_bins, soft_steps=soft_steps)
    partition, reference_ceiling = finite_ds_partition(reference, n_bins=n_bins)
    partition_score = score_labeling(
        reference.scores, np.asarray(partition.labels), reference.weights, n_bins=n_bins
    )
    rows: dict[str, MetricRow] = {}
    for index, rule in enumerate(rules):
        rows[rule.key] = evaluate_rule(
            rule,
            reference,
            evaluation,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=BOOTSTRAP_SEED + 10 * index,
        )
    validation_report = ds_result.validation_profiled_report
    direction: MetricRow = {
        "name": name,
        "reference_events": reference.n_events,
        "evaluation_events": evaluation.n_events,
        "reference_ceiling_retention": reference_ceiling,
        "evaluation_ceiling_retention": ceiling_retention(
            evaluation.scores, evaluation.weights, n_bins=n_bins
        ),
        "ds_partition_reference_profiled_retention": partition_score.profiled_retention,
        "ds_partition_reference_full_retention": partition_score.full_retention,
        "ds_rule_validation_report_retention": (
            None if validation_report is None else float(validation_report.geometric_mean_retention)
        ),
        "ds_rule_hardening_gap": ds_result.hardening_gap,
        "rules": rows,
    }
    return direction, {rule.key: rule for rule in rules}


def _mean_over_directions(directions: list[MetricRow]) -> dict[str, MetricRow]:
    """Average each rule's held-out numbers over the two directions."""
    keys = list(directions[0]["rules"])  # type: ignore[arg-type]
    summary: dict[str, MetricRow] = {}
    for key in keys:
        rows = [direction["rules"][key] for direction in directions]  # type: ignore[index]
        first = rows[0]
        entry: MetricRow = {
            "label": first["label"],
            "kind": first["kind"],
            "n_bins": first["n_bins"],
            "identified": all(bool(row["identified"]) for row in rows),
        }
        for metric in (
            "reference_profiled_retention",
            "evaluation_profiled_retention",
            "evaluation_full_retention",
        ):
            values = [row[metric] for row in rows]
            entry[metric] = (
                None if any(value is None for value in values) else float(np.mean(values))  # type: ignore[arg-type]
            )
        lows = [row.get("evaluation_bootstrap", {}).get("p05") for row in rows]  # type: ignore[union-attr]
        highs = [row.get("evaluation_bootstrap", {}).get("p95") for row in rows]  # type: ignore[union-attr]
        entry["evaluation_p05_min"] = None if any(v is None for v in lows) else float(min(lows))  # type: ignore[type-var]
        entry["evaluation_p95_max"] = None if any(v is None for v in highs) else float(max(highs))  # type: ignore[type-var]
        entry["evaluation_min_bin_events"] = int(
            min(int(row["evaluation_min_bin_events"]) for row in rows)  # type: ignore[call-overload]
        )
        summary[key] = entry
    return summary


# --------------------------------------------------------------------------- #
# Downstream: the signal-strength fit on yield templates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ShiftedTables:
    """The out-of-fold score table and posterior of every event at every `tes` point."""

    points: tuple[float, ...]
    tables: dict[float, EventTable]

    def at(self, tes_point: float) -> EventTable:
        """The table at one committed `tes` point."""
        return self.tables[tes_point]

    def subset(self, mask: np.ndarray) -> ShiftedTables:
        """Restrict every point's table to the same rows."""
        return ShiftedTables(
            self.points, {t: table.subset(mask) for t, table in self.tables.items()}
        )


def shifted_tables(data: HepData, sigbg: SignalBackgroundOOF, tes: TesOOF) -> ShiftedTables:
    """Score every committed `tes` copy of the events with the out-of-fold models."""
    tables: dict[float, EventTable] = {}
    for point in TES_POINTS:
        scores = out_of_fold_scores(data, sigbg, tes, tes_point=point)
        posterior = sigbg.predict(data.features_at(point))[:, 1]
        tables[point] = EventTable(scores, posterior, data.weights, data.is_signal)
    return ShiftedTables(TES_POINTS, tables)


def yield_templates(
    rule: BinRule, shifted: ShiftedTables
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``S[t, b]``, ``B[t, b]`` and the nominal ``sum w^2`` per cell.

    Each event's copy at each `tes` point is labelled by the rule; the
    signal and background yields per cell are the Monte Carlo weights that
    landed there. The last array is what the Barlow-Beeston-lite inflation
    needs: the sum of squared weights per cell at the nominal point.
    """
    signal = np.zeros((len(shifted.points), rule.n_bins))
    background = np.zeros_like(signal)
    squared = np.zeros(rule.n_bins)
    for row, point in enumerate(shifted.points):
        table = shifted.at(point)
        labels = rule.assign(table)
        signal[row] = np.bincount(
            labels, weights=np.where(table.is_signal, table.weights, 0.0), minlength=rule.n_bins
        )
        background[row] = np.bincount(
            labels, weights=np.where(~table.is_signal, table.weights, 0.0), minlength=rule.n_bins
        )
        if abs(point - 1.0) < 1e-9:
            squared = np.bincount(labels, weights=table.weights**2, minlength=rule.n_bins)
    return signal, background, squared


def _sigma_from_fisher(information: np.ndarray) -> float | None:
    eigenvalues = np.linalg.eigvalsh(information)
    if eigenvalues[0] <= 1e-9 * eigenvalues[-1]:
        return None
    return float(np.sqrt(np.linalg.inv(information)[0, 0]))


def asimov_uncertainty(
    signal: np.ndarray,
    background: np.ndarray,
    squared_weights: np.ndarray,
    points: tuple[float, ...],
    *,
    constraint_width: float,
) -> MetricRow:
    """Expected relative uncertainty on the signal strength from binned yield templates.

    The Asimov binned-Poisson Fisher matrix at ``(mu, nu, tes) = (1, 1, 1)``
    is ``sum_b g_b g_b^T / N_b`` with ``g_b = (S_b, B_b, dN_b/dtes)``. The
    `tes` derivative is the slope of a least-squares line through the seven
    committed points, because a central difference on ~500 events is
    dominated by single events crossing a boundary. The Monte Carlo
    statistical inflation replaces ``N_b`` by ``N_b + sum_{i in b} w_i^2``.
    """
    nominal = int(np.argmin(np.abs(np.asarray(points) - 1.0)))
    total = signal + background
    t = np.asarray(points)
    centered = t - t.mean()
    slope = centered @ (total - total.mean(axis=0)) / float(centered @ centered)
    minus = int(np.argmin(np.abs(t - 0.95)))
    plus = int(np.argmin(np.abs(t - 1.05)))
    central = (total[plus] - total[minus]) / (t[plus] - t[minus])
    yields = total[nominal]
    occupied = yields > 0
    gradient = np.stack([signal[nominal], background[nominal], slope], axis=1)[occupied]
    information = gradient.T @ (gradient / yields[occupied, None])
    inflated = gradient.T @ (gradient / (yields[occupied] + squared_weights[occupied])[:, None])
    constrained = information.copy()
    constrained[2, 2] += 1.0 / constraint_width**2
    gradient_central = np.stack([signal[nominal], background[nominal], central], axis=1)[occupied]
    information_central = gradient_central.T @ (gradient_central / yields[occupied, None])
    two_parameter = information[:2, :2]
    return {
        "sigma_mu": _sigma_from_fisher(information),
        "sigma_mu_mc_inflated": _sigma_from_fisher(inflated),
        "sigma_mu_tes_constrained": _sigma_from_fisher(constrained),
        "sigma_mu_tes_fixed": _sigma_from_fisher(two_parameter),
        "sigma_mu_central_difference": _sigma_from_fisher(information_central),
        "occupied_bins": int(np.count_nonzero(occupied)),
        "nominal_yield": [float(value) for value in yields],
        "signal_yield": [float(value) for value in signal[nominal]],
        "tes_slope": [float(value) for value in slope],
    }


def unbinned_proxy_uncertainty(table: EventTable, signal_fraction: float) -> MetricRow:
    """Expected relative uncertainty on the signal strength from the proxy score itself.

    The extended-likelihood Fisher matrix in the intensity coefficients is
    ``sum_i w_i s_i s_i^T``; the signal coefficient's reference value is the
    physical signal fraction, so dividing by it gives the same relative
    uncertainty the templates report. This is the reference no binning can
    beat -- and it is a proxy, because no unbinned likelihood exists.
    """
    information = table.scores.T @ (table.scores * table.weights[:, None])
    sigma = _sigma_from_fisher(information)
    fixed = _sigma_from_fisher(information[:2, :2])
    return {
        "sigma_mu": None if sigma is None else sigma / signal_fraction,
        "sigma_mu_tes_fixed": None if fixed is None else fixed / signal_fraction,
    }


def downstream_uncertainty(
    rules: dict[str, BinRule],
    shifted: ShiftedTables,
    *,
    signal_fraction: float,
) -> MetricRow:
    """Run the downstream fit for every rule on one set of events."""
    nominal = shifted.at(1.0)
    rows: dict[str, MetricRow] = {}
    for key, rule in rules.items():
        signal, background, squared = yield_templates(rule, shifted)
        row = asimov_uncertainty(
            signal, background, squared, shifted.points, constraint_width=TES_CONSTRAINT_WIDTH
        )
        # The proxy-score version of the same binned model, for comparison:
        # the binned score Fisher restated as a relative uncertainty.
        labels = rule.assign(nominal)
        proxy = None
        try:
            report = sq.information_report(
                nominal.scores, labels, nominal.weights, n_bins=rule.n_bins
            )
            binned = np.asarray(report.fisher_binned)
            proxy_sigma = _sigma_from_fisher(binned)
            proxy = None if proxy_sigma is None else proxy_sigma / signal_fraction
        except sq.ContractError:
            proxy = None
        row["label"] = rule.label
        row["kind"] = rule.kind
        row["n_bins"] = rule.n_bins
        row["sigma_mu_proxy_score"] = proxy
        rows[key] = row
    return {
        "events": nominal.n_events,
        "weighted_yield": float(np.sum(nominal.weights)),
        "unbinned_proxy": unbinned_proxy_uncertainty(nominal, signal_fraction),
        "rules": rows,
    }


# --------------------------------------------------------------------------- #
# In-sample study on the whole table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PartitionRow:
    """One labeling of the whole sample, scored both ways.

    Attributes
    ----------
    key, label, criterion
        Stable identifier, published name, and the criterion (or baseline
        recipe) that produced the labels.
    n_bins
        Number of bins the labeling actually uses.
    full_retention, profiled_retention
        The two retention numbers from `score_labeling`; ``None`` when the
        binned model is not identified.
    """

    key: str
    label: str
    criterion: str
    n_bins: int
    full_retention: float | None
    profiled_retention: float | None


def _row(
    key: str, label: str, criterion: str, table: EventTable, labels: np.ndarray, n_bins: int
) -> PartitionRow:
    scored = try_score_labeling(table.scores, labels, table.weights, n_bins=n_bins)
    identified = scored is not None and n_bins > len(SCHEMA.parameters) - 1
    return PartitionRow(
        key,
        label,
        criterion,
        n_bins,
        None if scored is None else scored.full_retention,
        scored.profiled_retention if identified and scored is not None else None,
    )


@dataclass(frozen=True, slots=True)
class InSampleStudy:
    """The whole-sample partitions, baselines and ceiling of the in-sample study."""

    rows: list[PartitionRow]
    d_labels: np.ndarray = field(repr=False)
    ds_labels: np.ndarray = field(repr=False)
    ds_partition: sq.PartitionResult = field(repr=False)
    ceiling: sq.EfficientScoreBound = field(repr=False)
    rules: dict[str, BinRule] = field(repr=False)


def in_sample_study(table: EventTable, *, n_bins: int, soft_steps: int) -> InSampleStudy:
    """Optimize the finite partitions on all events and score every baseline in sample."""
    ceiling = sq.efficient_score_bound(
        table.scores, interest=INTEREST, weights=table.weights, n_bins=n_bins
    )
    d_partition = sq.optimize_partition(
        table.sample(),
        n_bins=n_bins,
        criterion=sq.DOptimality(),
        config=sq.DExchangeConfig(seed=SOLVER_SEED),
    )
    ds_partition = sq.optimize_partition(
        table.sample(),
        n_bins=n_bins,
        criterion=sq.ProfiledDOptimality(("mu_htautau",)),
        config=sq.DExchangeConfig(seed=SOLVER_SEED),
        initial_labels=ceiling.labels,
    )
    rules, _ = build_rules(table, None, n_bins=n_bins, soft_steps=soft_steps)
    by_key = {rule.key: rule for rule in rules}
    rows = [
        _row(
            "ds_partition",
            "Profiled Ds partition",
            "ProfiledDOptimality",
            table,
            np.asarray(ds_partition.labels),
            n_bins,
        ),
        _row(
            "d_partition",
            "Plain D partition",
            "DOptimality",
            table,
            np.asarray(d_partition.labels),
            n_bins,
        ),
    ]
    for rule in rules:
        rows.append(_row(rule.key, rule.label, rule.kind, table, rule.assign(table), rule.n_bins))
    return InSampleStudy(
        rows,
        np.asarray(d_partition.labels),
        np.asarray(ds_partition.labels),
        ds_partition,
        ceiling,
        by_key,
    )


def ceiling_sweep(table: EventTable, budgets: tuple[int, ...]) -> list[MetricRow]:
    """Sweep the bin budget in sample: profiled D_s, the quantile baseline, and the ceiling."""
    reference = unbinned_profiled_information(table.scores, table.weights)
    rows: list[MetricRow] = []
    for n_bins in budgets:
        bound = sq.efficient_score_bound(
            table.scores, interest=INTEREST, weights=table.weights, n_bins=n_bins
        )
        ds_partition = sq.optimize_partition(
            table.sample(),
            n_bins=n_bins,
            criterion=sq.ProfiledDOptimality(("mu_htautau",)),
            config=sq.DExchangeConfig(seed=SOLVER_SEED),
            initial_labels=bound.labels,
        )
        ds_score = score_labeling(
            table.scores, np.asarray(ds_partition.labels), table.weights, n_bins=n_bins
        )
        quantile = quantile_rule(table, n_bins)
        quantile_score = try_score_labeling(
            table.scores, quantile.assign(table), table.weights, n_bins=n_bins
        )
        rows.append(
            {
                "n_bins": float(n_bins),
                "ds_profiled_retention": ds_score.profiled_retention,
                "classifier_quantile_profiled_retention": (
                    None if quantile_score is None else quantile_score.profiled_retention
                ),
                "ceiling_retention": float(np.exp(bound.upper_bound - np.log(reference))),
                "gap": float(bound.gap_to(ds_partition)),
                "scans": int(ds_partition.scans),
                "accepted_moves": int(ds_partition.accepted_moves),
            }
        )
    return rows


def held_out_budget_sweep(
    halves: tuple[EventTable, EventTable], budgets: tuple[int, ...], *, soft_steps: int
) -> list[MetricRow]:
    """The reusable profiled-D_s rule's held-out retention at every swept budget."""
    rows: list[MetricRow] = []
    for n_bins in budgets:
        values: list[float | None] = []
        for reference, evaluation in (halves, halves[::-1]):
            rule, _ = fit_ds_rule(reference, n_bins=n_bins, soft_steps=soft_steps)
            scored = try_score_labeling(
                evaluation.scores, rule.assign(evaluation), evaluation.weights, n_bins=n_bins
            )
            values.append(None if scored is None else scored.profiled_retention)
        rows.append(
            {
                "n_bins": float(n_bins),
                "ds_rule_evaluation_profiled_retention": (
                    None if any(v is None for v in values) else float(np.mean(values))  # type: ignore[arg-type]
                ),
                "directions": values,
            }
        )
    return rows


@dataclass(frozen=True, slots=True)
class DeltaRow:
    """One point of the delta convergence study.

    Attributes
    ----------
    delta
        The finite-difference half-offset.
    minus_plus_auc
        Weighted out-of-fold AUC of the minus/plus classification task itself.
    near_half_fraction
        Fraction of events within `NEAR_HALF_TOLERANCE` of an
        undecided (0.5) posterior at the nominal point.
    ds_profiled_retention
        Profiled D_s retention at `HEADLINE_BINS` under this delta's score.
    ceiling_retention
        The certified ceiling at `HEADLINE_BINS` under this delta's score.
    """

    delta: float
    minus_plus_auc: float
    near_half_fraction: float
    ds_profiled_retention: float
    ceiling_retention: float


def delta_convergence_study(
    data: HepData, sigbg: SignalBackgroundOOF, fold_ids: np.ndarray, *, max_iter: int
) -> tuple[list[DeltaRow], dict[str, float]]:
    """Recompute the tes score at each swept delta and report the agreement.

    Returns the three-point table, and an agreement summary between the
    headline delta and delta/2 -- "a disagreement is a result to report,
    not a parameter to tune away".
    """
    rows: list[DeltaRow] = []
    tes_columns: dict[float, np.ndarray] = {}
    for delta in DELTA_SWEEP:
        tes = fit_tes_oof(
            data, delta=delta, fold_ids=fold_ids, max_iter=max_iter, seed=FOLD_SEED + 500
        )
        sample = assemble_score_sample(data, sigbg, tes)
        table = EventTable(
            np.asarray(sample.scores), sigbg.probabilities[:, 1], data.weights, data.is_signal
        )
        tes_columns[delta] = table.scores[:, 2]
        partition, ceiling = finite_ds_partition(table, n_bins=HEADLINE_BINS)
        scored = score_labeling(
            table.scores, np.asarray(partition.labels), table.weights, n_bins=HEADLINE_BINS
        )
        rows.append(
            DeltaRow(
                delta=delta,
                minus_plus_auc=tes.minus_plus_auc,
                near_half_fraction=tes.near_half_fraction,
                ds_profiled_retention=scored.profiled_retention,
                ceiling_retention=ceiling,
            )
        )
    half = HEADLINE_DELTA / 2.0
    headline_column = tes_columns[HEADLINE_DELTA]
    half_column = next(value for delta, value in tes_columns.items() if abs(delta - half) < 1e-9)
    correlation = float(np.corrcoef(headline_column, half_column)[0, 1])
    by_delta = {row.delta: row for row in rows}
    agreement = {
        "headline_delta": HEADLINE_DELTA,
        "half_delta": half,
        "score_correlation": correlation,
        "retention_gap": abs(
            by_delta[HEADLINE_DELTA].ds_profiled_retention - by_delta[half].ds_profiled_retention
        ),
    }
    return rows, agreement


# --------------------------------------------------------------------------- #
# The whole run
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Study:
    """Everything the doc page, the figures, and the tests need from one run."""

    metrics: dict[str, object]
    table: EventTable = field(repr=False)
    ds_partition_labels: np.ndarray = field(repr=False)
    ds_rule_labels: np.ndarray = field(repr=False)
    logit_labels: np.ndarray = field(repr=False)
    ds_rule_occupancy: list[MetricRow] = field(repr=False)
    logit_occupancy: list[MetricRow] = field(repr=False)


def _split_summary(reference: EventTable, evaluation: EventTable, name: str) -> MetricRow:
    return {
        "name": name,
        "reference_events": reference.n_events,
        "reference_signal_events": int(np.count_nonzero(reference.is_signal)),
        "evaluation_events": evaluation.n_events,
        "evaluation_signal_events": int(np.count_nonzero(evaluation.is_signal)),
    }


def run_study(
    *,
    max_iter: int | None = None,
    n_folds: int | None = None,
    soft_steps: int | None = None,
    budgets: tuple[int, ...] | None = None,
    bootstrap_replicates: int | None = None,
) -> Study:
    """Run the whole HEP classifier study and return its metrics and arrays.

    Parameters
    ----------
    max_iter
        Boosting round budget for every classifier fit. Defaults to
        `examples._env.example_scale(300, 60)`.
    n_folds
        Number of stratified event folds. Defaults to ``example_scale(5, 3)``.
    soft_steps
        Adam step budget of every soft-Voronoi rule. Defaults to
        ``example_scale(400, 80)``.
    budgets
        Bin budgets swept against the certified ceiling and held out.
        Defaults to ``example_scale(BUDGET_SWEEP, (3, 6))``.
    bootstrap_replicates
        Percentile-bootstrap replicates per held-out rule. Defaults to
        ``example_scale(200, 20)``.

    Returns
    -------
    Study
        The exact structure written to
        ``docs/examples/assets/hep-classifier.json``, together with the
        arrays the figures draw.
    """
    max_iter = example_scale(300, 60) if max_iter is None else max_iter
    n_folds = example_scale(5, 3) if n_folds is None else n_folds
    soft_steps = example_scale(400, 80) if soft_steps is None else soft_steps
    budgets = example_scale(BUDGET_SWEEP, (3, 6)) if budgets is None else budgets
    bootstrap_replicates = (
        example_scale(BOOTSTRAP_REPLICATES, 20)
        if bootstrap_replicates is None
        else bootstrap_replicates
    )
    timings: dict[str, float] = {}

    data = load_fixture()
    fold_ids = event_folds(data.is_signal, n_folds=n_folds, seed=FOLD_SEED)

    start = time.perf_counter()
    sigbg = fit_signal_background_oof(
        data, fold_ids=fold_ids, max_iter=max_iter, seed=FOLD_SEED + 100
    )
    timings["signal_classifier"] = time.perf_counter() - start
    start = time.perf_counter()
    tes = fit_tes_oof(
        data, delta=HEADLINE_DELTA, fold_ids=fold_ids, max_iter=max_iter, seed=FOLD_SEED + 500
    )
    timings["tes_classifier"] = time.perf_counter() - start
    start = time.perf_counter()
    reliability = tes_score_reliability(
        data, delta=HEADLINE_DELTA, max_iter=max_iter, seed=FOLD_SEED + 700
    )
    timings["tes_reliability"] = time.perf_counter() - start

    sample = assemble_score_sample(data, sigbg, tes)
    table = EventTable(
        np.asarray(sample.scores), sigbg.probabilities[:, 1], data.weights, data.is_signal
    )
    shifted = shifted_tables(data, sigbg, tes)
    if not np.allclose(shifted.at(1.0).scores, table.scores):
        raise ValueError("the nominal shifted table must reproduce the study sample")

    # In sample, on all events.
    start = time.perf_counter()
    whole = in_sample_study(table, n_bins=HEADLINE_BINS, soft_steps=soft_steps)
    sweep = ceiling_sweep(table, budgets)
    delta_rows, delta_agreement = delta_convergence_study(data, sigbg, fold_ids, max_iter=max_iter)
    timings["in_sample"] = time.perf_counter() - start

    # Cross-evaluation in both directions of one split.
    start = time.perf_counter()
    half_ids = event_split(data.is_signal)
    half_a = table.subset(half_ids == 0)
    half_b = table.subset(half_ids == 1)
    direction_ab, rules_ab = cross_evaluate_direction(
        "a_to_b",
        half_a,
        half_b,
        n_bins=HEADLINE_BINS,
        soft_steps=soft_steps,
        bootstrap_replicates=bootstrap_replicates,
    )
    direction_ba, rules_ba = cross_evaluate_direction(
        "b_to_a",
        half_b,
        half_a,
        n_bins=HEADLINE_BINS,
        soft_steps=soft_steps,
        bootstrap_replicates=bootstrap_replicates,
    )
    directions = [direction_ab, direction_ba]
    held_out_sweep = held_out_budget_sweep((half_a, half_b), budgets, soft_steps=soft_steps)
    timings["cross_evaluation"] = time.perf_counter() - start

    # Downstream, held out per direction and in sample on all events.
    start = time.perf_counter()
    downstream_held_out = [
        downstream_uncertainty(
            rules_ab, shifted.subset(half_ids == 1), signal_fraction=sigbg.signal_fraction
        ),
        downstream_uncertainty(
            rules_ba, shifted.subset(half_ids == 0), signal_fraction=sigbg.signal_fraction
        ),
    ]
    downstream_all = downstream_uncertainty(
        whole.rules, shifted, signal_fraction=sigbg.signal_fraction
    )
    timings["downstream"] = time.perf_counter() - start

    reference = unbinned_profiled_information(table.scores, table.weights)
    by_key = {row.key: row for row in whole.rows}
    ceiling_value = float(np.exp(whole.ceiling.upper_bound - np.log(reference)))
    ds_profiled = by_key["ds_partition"].profiled_retention
    d_profiled = by_key["d_partition"].profiled_retention
    ds_full = by_key["ds_partition"].full_retention
    d_full = by_key["d_partition"].full_retention
    if ds_profiled is None or d_profiled is None or ds_full is None or d_full is None:
        raise ValueError("the in-sample ScoreQuant partitions must be identified")
    classifier_keys = (
        "classifier_significance_intervals",
        "classifier_logit_equal_width",
        "classifier_quantile",
    )
    classifier_profiled = {key: by_key[key].profiled_retention for key in classifier_keys}
    if any(value is None for value in classifier_profiled.values()):
        raise ValueError("every six-bin classifier baseline must be identified in sample")
    best_key = max(classifier_keys, key=lambda key: float(classifier_profiled[key]))  # type: ignore[arg-type]

    mean_directions = _mean_over_directions(directions)
    ds_rule_labels = whole.rules["ds_rule"].assign(table)
    logit_labels = whole.rules["classifier_logit_equal_width"].assign(table)

    metrics: dict[str, object] = {
        "fixture": {
            "n_events": data.n_events,
            "weight_sum": float(np.sum(data.weights)),
            "effective_events": float(np.sum(data.weights) ** 2 / np.sum(data.weights**2)),
            "signal_events": int(np.count_nonzero(data.is_signal)),
            "background_events": int(np.count_nonzero(~data.is_signal)),
        },
        "n_bins": HEADLINE_BINS,
        "n_parameters": len(SCHEMA.parameters),
        "interest": list(INTEREST),
        "schema": list(SCHEMA.parameters),
        "delta": HEADLINE_DELTA,
        "n_folds": n_folds,
        "classifier_max_iter": max_iter,
        "soft_steps": soft_steps,
        "tes_constraint_width": TES_CONSTRAINT_WIDTH,
        "classifiers": {
            "signal_weighted_auc": sigbg.weighted_auc,
            "signal_temperature": sigbg.temperature,
            "signal_fraction": sigbg.signal_fraction,
            "tes_minus_plus_auc": tes.minus_plus_auc,
            "tes_temperature": tes.temperature,
            "tes_near_half_fraction": tes.near_half_fraction,
            "tes_reliability": reliability,
        },
        "in_sample": {
            "partitions": [asdict(row) for row in whole.rows],
            "by_key": {row.key: asdict(row) for row in whole.rows},
            "criterion_trade": {
                "full_retention_given_up": d_full - ds_full,
                "profiled_retention_gained": ds_profiled - d_profiled,
            },
            "scorequant_vs_classifier_binning": {
                "best_baseline_key": best_key,
                "best_baseline_label": by_key[best_key].label,
                "profiled_retention_gap": ds_profiled - float(classifier_profiled[best_key]),  # type: ignore[arg-type]
                "profiled_retention_gap_to_equal_frequency": (
                    ds_profiled - float(classifier_profiled["classifier_quantile"])  # type: ignore[arg-type]
                ),
                "baseline_spread": (
                    max(float(v) for v in classifier_profiled.values())  # type: ignore[arg-type]
                    - min(float(v) for v in classifier_profiled.values())  # type: ignore[arg-type]
                ),
            },
            "ceiling": {
                "upper_bound": float(whole.ceiling.upper_bound),
                "ceiling_retention": ceiling_value,
                "gap_to_ds_partition_nats": float(whole.ceiling.gap_to(whole.ds_partition)),
                "gap_to_ds_partition_retention": ceiling_value - ds_profiled,
            },
            "ceiling_sweep": sweep,
            "delta_convergence": {
                "rows": [asdict(row) for row in delta_rows],
                "agreement": delta_agreement,
            },
        },
        "split": {
            "seed": SPLIT_SEED,
            "halves": [
                _split_summary(half_a, half_b, "a_to_b"),
                _split_summary(half_b, half_a, "b_to_a"),
            ],
        },
        "cross_evaluation": {
            "directions": directions,
            "mean": mean_directions,
            "held_out_budget_sweep": held_out_sweep,
            "bootstrap_replicates": bootstrap_replicates,
        },
        "downstream": {
            "tes_constraint_width": TES_CONSTRAINT_WIDTH,
            "held_out": downstream_held_out,
            "all_events": downstream_all,
        },
        "timings_seconds": timings,
    }
    ds_rule_occupancy = bin_occupancy(table, ds_rule_labels, n_bins=HEADLINE_BINS)
    logit_occupancy = bin_occupancy(table, logit_labels, n_bins=HEADLINE_BINS)
    metrics["in_sample"]["occupancy"] = {  # type: ignore[index]
        "ds_rule": ds_rule_occupancy,
        "classifier_logit_equal_width": logit_occupancy,
    }
    return Study(
        metrics=metrics,
        table=table,
        ds_partition_labels=whole.ds_labels,
        ds_rule_labels=ds_rule_labels,
        logit_labels=logit_labels,
        ds_rule_occupancy=ds_rule_occupancy,
        logit_occupancy=logit_occupancy,
    )


def main() -> None:
    """Run the study and write the committed JSON and figures."""
    import jax
    import matplotlib.pyplot as plt

    from .figures import make_budget_figure, make_cells_figure

    jax.config.update("jax_enable_x64", True)

    study = run_study()
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as stream:
        json.dump(study.metrics, stream, indent=2)
        stream.write("\n")
    for path, figure in (
        (CELLS_FIGURE_PATH, make_cells_figure(study)),
        (BUDGET_FIGURE_PATH, make_budget_figure(study)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    main()
