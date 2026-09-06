"""A Michelson interferometer phase: an analytic ScoreFunction against a profiled nuisance.

This script is the deterministic generator behind
`docs/examples/michelson-phase.md`. It is the first example to exercise both
the analytic `ScoreFunction` route (a score callback against a bounded
`IntegrationSource`, rather than a precomputed score table or a linear
component model) and the NumPy backend end to end, on a model exact enough
that the library's own retained-information numbers are also a check on the
mathematics rather than only an illustration of it.

The model: a photon's fringe phase `u` over `FRINGES` whole fringes, with the
phase `phi` the parameter of interest and the fractional fringe-frequency
error `epsilon` a nuisance that a short-baseline interferometer cannot avoid
confounding with `phi`. It runs

* the closed-form conditional score and its two closed-form Fisher entries,
  asserted to floating-point precision;
* the profiled ceiling that phase information faces once `epsilon` is
  profiled out, which every phase-retention number in this study is stated
  against;
* three finite partitions per bin budget on the same score table -- naive
  equal-width detector segments, `DOptimality`, and `ProfiledDOptimality`
  seeded from the certified efficient-score bound -- and the certified gap
  each profiled partition leaves on the table;
* the compile bridge for the D-optimal partition, and the refusal a profiled
  partition gives instead;
* two reusable rules fitted directly from the `IntegrationSource` route --
  the one this example exists to cover -- one compiled and one soft; and
* the "comb": the compiled six-bin rule predicted on a fine grid of `u`,
  which is a segmentation the aperture itself cannot express contiguously.

and writes `docs/examples/assets/michelson-phase.json` plus
`docs/examples/assets/michelson-phase.png`.

Run it with::

    JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run python -m examples.michelson_phase
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

import scorequant as sq
from examples._env import example_scale

FIGURE_PATH = Path("docs/examples/assets/michelson-phase.png")
METRICS_PATH = Path("docs/examples/assets/michelson-phase.json")

#: Where the portal's committed figure files live. Unlike
#: `website/static/walkthrough-figures/`, which `generate_walkthroughs.py`
#: refills from `docs/examples/assets/` and `.gitignore` excludes, this
#: directory is committed: the walkthrough reads these three panels straight
#: from it, so a file written here has to be committed alongside the code that
#: produced it.
SCORE_FIGURE_DIR = Path("website/static/figures")

#: Every call in this study runs on the portable NumPy backend at float64, the
#: one combination `docs/examples` had no example for before this one.
EXECUTION = sq.ExecutionConfig(backend="numpy", precision="float64", device="cpu")

#: Fringe visibility, phase reference, and fringe count. `u` ranges over
#: `[0, U_MAX)` with `U_MAX = 2 pi FRINGES`.
V0 = 0.6
PHI0 = 0.0
FRINGES = 4
U_MAX = 2.0 * np.pi * FRINGES

#: Solver seed shared by every finite fit in the study.
SEED = 4
#: Bin budget of the headline comparison and its compile/refusal demonstration.
HEADLINE_BINS = 6
#: Bin budgets swept against the certified profiled ceiling.
BIN_SWEEP = (4, 6, 8, 10)
#: Tensor Gauss-Legendre order of the `IntegrationSource` route -- the input
#: route this example exists to cover, used for the reusable-rule fits.
GAUSS_LEGENDRE_ORDER = 256
#: Grid resolution of the "comb" prediction check.
COMB_GRID_POINTS = 4_001

#: Names of the two score columns, in column order.
SCHEMA = sq.ScoreSchema(("phase", "fringe_frequency"))
#: `phase` is column 0, the parameter of interest for every profiled fit below.
INTEREST: tuple[int, ...] = (0,)


def fringe_density(observations: np.ndarray) -> np.ndarray:
    """Return the normalized one-photon fringe-phase density at the reference point.

    `1 + V0 cos(u)` integrates to `U_MAX` over `FRINGES` whole periods, so
    dividing by `U_MAX` makes this a proper density: a source built from it
    carries unit total mass, and `fisher_information` on that source is
    therefore the *per-photon* Fisher information the closed forms describe.
    """
    u = np.asarray(observations)[:, 0]
    return (1.0 + V0 * np.cos(u)) / U_MAX


def michelson_score(observations: np.ndarray) -> np.ndarray:
    """Return the exact conditional score `(s_phi, s_epsilon)` at `(phi0, epsilon0) = (0, 0)`.

    `s_epsilon = u * s_phi - V0`; the `-V0` is the normalizer derivative
    `d/d_epsilon log Z`, not a centering convenience -- dropping it would leave
    `E[s_epsilon] = V0 != 0`, violating the library's never-center-scores
    invariant. Both components are bounded because `1 + V0 cos(u) >= 1 - V0 >
    0` on this visibility, so the `ScoreFunction` finiteness contract holds by
    construction.
    """
    u = np.asarray(observations)[:, 0]
    denominator = 1.0 + V0 * np.cos(u)
    s_phi = -V0 * np.sin(u) / denominator
    s_epsilon = u * s_phi - V0
    return np.column_stack([s_phi, s_epsilon])


def build_provider() -> sq.ScoreFunction:
    """Return the analytic `ScoreFunction` provider, with schema and exact provenance."""
    return sq.ScoreFunction(
        michelson_score,
        provenance=sq.ScoreProvenance(kind="exact", reference_point=(PHI0, 0.0)),
        schema=SCHEMA,
    )


def build_integration_source() -> sq.IntegrationSource:
    """Return the bounded `IntegrationSource` reference measure over one photon's phase.

    This, paired with `build_provider`, is the one input route the library had
    no example for: an analytic score callback against a bounded quadrature
    measure rather than a precomputed score table or a linear component model.
    """
    return sq.IntegrationSource(
        [[0.0, U_MAX]],
        density=fringe_density,
        quadrature=sq.GaussLegendreConfig(order=GAUSS_LEGENDRE_ORDER),
    )


@dataclass(frozen=True, slots=True)
class TrainSample:
    """The finite weighted score table every finite partition in this study fits on.

    Attributes
    ----------
    observations
        Fringe-phase nodes with shape ``[N, 1]``.
    scores
        `(s_phi, s_epsilon)` evaluated at `observations`, shape ``[N, 2]``.
    weights
        Deterministic midpoint-quadrature measure at each node.
    """

    observations: np.ndarray = field(repr=False)
    scores: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)


def build_train_sample(provider: sq.ScoreFunction, *, n_nodes: int) -> TrainSample:
    """Build the deterministic midpoint-quadrature score table.

    `fringe_density` and `michelson_score` are both `2 pi`-periodic in `u` up
    to the explicit `u * s_phi` term, so midpoint quadrature of the periodic
    part converges exponentially rather than at the usual second order --
    which is what lets `fisher_information` on this table reproduce the
    closed forms to machine precision at a few thousand nodes.

    Parameters
    ----------
    provider
        The analytic score provider.
    n_nodes
        Number of midpoint-quadrature nodes over `[0, U_MAX)`.

    Returns
    -------
    TrainSample
        Observations, scores, and quadrature weights.
    """
    step = U_MAX / n_nodes
    u = (np.arange(n_nodes) + 0.5) * step
    observations = u[:, None]
    weights = fringe_density(observations) * step
    scores = np.asarray(provider.score(observations, execution=EXECUTION))
    return TrainSample(observations=observations, scores=scores, weights=weights)


def closed_form_information() -> dict[str, float]:
    """Return the two closed-form unbinned Fisher entries at `V0`.

    Returns
    -------
    dict of float
        ``i_phiphi = 1 - sqrt(1 - V0**2)`` and
        ``i_phieps = i_phiphi * U_MAX / 2``.
    """
    i_phiphi = 1.0 - np.sqrt(1.0 - V0**2)
    i_phieps = i_phiphi * U_MAX / 2.0
    return {"i_phiphi": float(i_phiphi), "i_phieps": float(i_phieps)}


def unbinned_profiled_information(scores: np.ndarray, weights: np.ndarray) -> float:
    """Return the unbinned profiled (Schur-complemented) phase information.

    This is the ceiling every phase-retention number in this study is stated
    against -- never `I_phiphi`, which is not available to an analyst who does
    not know `epsilon`.
    """
    information = np.asarray(sq.fisher_information(scores, weights, execution=EXECUTION))
    nuisance = [index for index in range(information.shape[0]) if index not in set(INTEREST)]
    interest_indices = list(INTEREST)
    block = information[np.ix_(interest_indices, interest_indices)]
    cross = information[np.ix_(interest_indices, nuisance)]
    nuisance_block = information[np.ix_(nuisance, nuisance)]
    schur = block - cross @ np.linalg.solve(nuisance_block, cross.T)
    return float(schur[0, 0])


def profiled_retention(sample: TrainSample, labels: np.ndarray, n_bins: int) -> float:
    """Return one labeling's profiled phase-information retention.

    `profiled_information_report` Schur-completes the nuisance block out of
    both the unbinned and the binned Fisher matrix of *this* sample, so the
    denominator is automatically the unbinned profiled ceiling of the same
    weighted score table -- this is what makes every retention number in this
    study stated against that ceiling rather than against `I_phiphi`.
    """
    return float(
        sq.profiled_information_report(
            sample.scores,
            labels,
            interest=INTEREST,
            weights=sample.weights,
            n_bins=n_bins,
            schema=SCHEMA,
            execution=EXECUTION,
        ).geometric_mean_retention
    )


def equal_width_labels(observations: np.ndarray, n_bins: int) -> np.ndarray:
    """Return the naive equal-width detector-segment labeling.

    Parameters
    ----------
    observations
        Fringe-phase nodes with shape ``[N, 1]``.
    n_bins
        Number of equal-width segments over `[0, U_MAX)`.

    Returns
    -------
    numpy.ndarray
        Integer segment index per row.
    """
    u = np.asarray(observations)[:, 0]
    edges = np.linspace(0.0, U_MAX, n_bins + 1)
    return np.digitize(u, edges[1:-1])


def label_runs(observations: np.ndarray, labels: np.ndarray, *, u_max: float) -> list[list[float]]:
    """Run-length encode a labeling along increasing `u` into aperture-edge triples.

    Parameters
    ----------
    observations
        Fringe-phase nodes with shape ``[N, 1]``, already sorted by `u` --
        the midpoint-quadrature nodes `build_train_sample` returns are.
    labels
        Integer label per row, aligned with `observations`.
    u_max
        Upper edge of the aperture, `U_MAX` for this study.

    Returns
    -------
    list of list
        ``[u_start, u_end, label]`` triples, one per maximal constant-label
        run, in increasing `u` order. Node `i` of `n` midpoint-quadrature
        nodes spans ``[i * u_max / n, (i + 1) * u_max / n]``, so a run's
        bounds are the outer edges of its first and last node: consecutive
        runs share an edge and together tile ``[0, u_max]`` exactly.
    """
    u = np.asarray(observations)[:, 0]
    labels = np.asarray(labels)
    n_nodes = u.shape[0]
    assert np.all(np.diff(u) > 0.0), "observations must be sorted by increasing u"
    step = u_max / n_nodes
    edges = np.flatnonzero(np.diff(labels) != 0)
    starts = np.concatenate([[0], edges + 1])
    stops = np.concatenate([edges + 1, [n_nodes]])
    return [
        [float(start * step), float(stop * step), int(labels[start])]
        for start, stop in zip(starts, stops, strict=True)
    ]


@dataclass(frozen=True, slots=True)
class SweepRow:
    """One bin budget's three retentions, the certified ceiling, and its gap.

    Attributes
    ----------
    n_bins
        Bin budget of this row.
    equal_width_retention, d_optimal_retention, profiled_retention_value
        Profiled phase-information retention of the three labelings, stated
        against the unbinned profiled ceiling.
    ceiling_retention
        The certified efficient-score ceiling at this budget, on the same
        retention scale.
    bound_gap
        `bound.gap_to(profiled_partition)`: the certified log-scale slack
        between the ceiling and the profiled-D_s partition's own objective.
    """

    n_bins: int
    equal_width_retention: float
    d_optimal_retention: float
    profiled_retention_value: float
    ceiling_retention: float
    bound_gap: float
    equal_width_labels: np.ndarray = field(repr=False)
    d_labels: np.ndarray = field(repr=False)
    profiled_labels: np.ndarray = field(repr=False)
    d_partition: sq.PartitionResult = field(repr=False)
    profiled_partition: sq.PartitionResult = field(repr=False)


def sweep_bin_budget(
    sample: TrainSample, reference: float, budgets: tuple[int, ...] = BIN_SWEEP
) -> list[SweepRow]:
    """Fit and score all three labelings at every swept bin budget.

    Parameters
    ----------
    sample
        The finite weighted score table.
    reference
        Unbinned profiled phase information, from `unbinned_profiled_information`.
    budgets
        Bin budgets to evaluate.

    Returns
    -------
    list of SweepRow
        One row per budget, in `budgets` order.
    """
    rows: list[SweepRow] = []
    for n_bins in budgets:
        equal_labels = equal_width_labels(sample.observations, n_bins)
        d_partition = sq.optimize_partition(
            sample.scores,
            weights=sample.weights,
            n_bins=n_bins,
            criterion=sq.DOptimality(),
            config=sq.DExchangeConfig(seed=SEED),
            execution=EXECUTION,
        )
        bound = sq.efficient_score_bound(
            sample.scores,
            interest=INTEREST,
            weights=sample.weights,
            n_bins=n_bins,
            execution=EXECUTION,
        )
        profiled_partition = sq.optimize_partition(
            sample.scores,
            weights=sample.weights,
            n_bins=n_bins,
            criterion=sq.ProfiledDOptimality(interest=INTEREST),
            config=sq.DExchangeConfig(seed=SEED),
            initial_labels=bound.labels,
            execution=EXECUTION,
        )
        d_labels = np.asarray(d_partition.labels)
        profiled_labels = np.asarray(profiled_partition.labels)
        rows.append(
            SweepRow(
                n_bins=n_bins,
                equal_width_retention=profiled_retention(sample, equal_labels, n_bins),
                d_optimal_retention=profiled_retention(sample, d_labels, n_bins),
                profiled_retention_value=profiled_retention(sample, profiled_labels, n_bins),
                ceiling_retention=float(np.exp(bound.upper_bound - np.log(reference))),
                bound_gap=float(bound.gap_to(profiled_partition)),
                equal_width_labels=equal_labels,
                d_labels=d_labels,
                profiled_labels=profiled_labels,
                d_partition=d_partition,
                profiled_partition=profiled_partition,
            )
        )
    return rows


@dataclass(frozen=True, slots=True)
class CompileBridge:
    """The compile bridge for the headline D-optimal partition, and the profiled refusal.

    Attributes
    ----------
    compiled_test_retention
        Overall D-efficiency of the compiled rule, evaluated on a fresh
        quadrature sample.
    exchange_stable
        Whether the D-optimal partition the rule was compiled from is
        exchange-stable.
    refusal_message
        `str` of the `RefusalError` `compile_quantizer()` raises on the
        profiled partition.
    """

    compiled_test_retention: float
    exchange_stable: bool
    refusal_message: str


def compile_bridge(
    headline: SweepRow, test_sample: TrainSample, *, n_bins: int
) -> tuple[CompileBridge, sq.Quantizer]:
    """Compile the headline D-optimal partition and record the profiled refusal.

    Parameters
    ----------
    headline
        The `SweepRow` at the headline bin budget.
    test_sample
        A held-out quadrature sample to evaluate the compiled rule on.
    n_bins
        The headline bin budget, for the evaluation's own report.

    Returns
    -------
    tuple
        The `CompileBridge` record and the compiled `Quantizer`, kept for the
        comb check and the figure.
    """
    compiled = headline.d_partition.compile_quantizer(execution=EXECUTION)
    compiled_report = compiled.evaluate_scores(
        test_sample.scores, test_sample.weights, execution=EXECUTION
    )
    try:
        headline.profiled_partition.compile_quantizer(execution=EXECUTION)
        raise AssertionError("a profiled-D partition must refuse compile_quantizer()")
    except sq.RefusalError as error:
        refusal_message = str(error)
    return (
        CompileBridge(
            compiled_test_retention=float(compiled_report.geometric_mean_retention),
            exchange_stable=bool(headline.d_partition.exchange_stable),
            refusal_message=refusal_message,
        ),
        compiled,
    )


@dataclass(frozen=True, slots=True)
class RuleRow:
    """One reusable rule fitted directly from the `IntegrationSource` route.

    Attributes
    ----------
    key, label, criterion, solver
        Identity of the fit.
    profiled_retention
        Profiled phase-information retention of the rule's own labels on the
        training sample, from `profiled_retention` -- the *same* quantity and
        the same ceiling as every column of the bin-budget sweep. Both rules
        report it, so the two rows are directly comparable.
    criterion_efficiency
        What each rule scores on the criterion it actually optimized:
        `train_report` for `DOptimality`, `train_profiled_report` for
        `ProfiledDOptimality`. Not comparable across rows -- the denominators
        differ -- which is exactly why `profiled_retention` exists beside it.
    hardening_gap
        Soft-to-hard retention gap; `0.0` for the compiled route.
    """

    key: str
    label: str
    criterion: str
    solver: str
    profiled_retention: float
    criterion_efficiency: float
    hardening_gap: float


def reusable_rules(
    provider: sq.ScoreFunction,
    source: sq.IntegrationSource,
    sample: TrainSample,
    *,
    n_bins: int,
    soft_steps: int,
) -> list[RuleRow]:
    """Fit a reusable rule under each criterion directly from the missing route.

    Parameters
    ----------
    provider
        The analytic score provider.
    source
        The bounded `IntegrationSource` reference measure.
    sample
        The training sample each rule is scored against, so both rows report
        profiled phase retention on one common ceiling.
    n_bins
        Bin budget of both fits.
    soft_steps
        Adam step budget of the soft profiled fit.

    Returns
    -------
    list of RuleRow
        One row for `DOptimality` (compiled exchange) and one for
        `ProfiledDOptimality` (soft Voronoi) -- the latter is the only route to
        a *reusable* profiled rule, since finite profiled-D labels have no
        compile bridge.
    """
    d_rule = sq.fit_quantizer(
        source,
        provider=provider,
        n_bins=n_bins,
        criterion=sq.DOptimality(),
        config=sq.DExchangeConfig(seed=SEED),
        execution=EXECUTION,
    )
    ds_rule = sq.fit_quantizer(
        source,
        provider=provider,
        n_bins=n_bins,
        criterion=sq.ProfiledDOptimality(interest=("phase",)),
        config=sq.SoftVoronoiConfig(
            seed=SEED,
            initializer_restarts=8,
            max_steps=soft_steps,
            record_every=max(soft_steps // 8, 1),
        ),
        execution=EXECUTION,
    )
    assert ds_rule.train_profiled_report is not None
    return [
        RuleRow(
            key="d_rule",
            label="Plain D, compiled exchange",
            criterion="DOptimality",
            solver="DExchangeConfig",
            profiled_retention=profiled_retention(
                sample, d_rule.predict_scores(sample.scores), n_bins
            ),
            criterion_efficiency=float(d_rule.train_report.geometric_mean_retention),
            hardening_gap=float(d_rule.hardening_gap or 0.0),
        ),
        RuleRow(
            key="ds_rule",
            label="Profiled D_s, soft Voronoi",
            criterion="ProfiledDOptimality",
            solver="SoftVoronoiConfig",
            profiled_retention=profiled_retention(
                sample, ds_rule.predict_scores(sample.scores), n_bins
            ),
            criterion_efficiency=float(ds_rule.train_profiled_report.geometric_mean_retention),
            hardening_gap=float(ds_rule.hardening_gap or 0.0),
        ),
    ]


def comb_runs(
    compiled: sq.Quantizer, provider: sq.ScoreFunction
) -> tuple[np.ndarray, np.ndarray, int]:
    """Predict the compiled rule on a fine grid of `u` and count contiguous runs.

    Because the score depends on `u` only through the fringe phase, a
    score-space cell pulls back to one interval per fringe: the
    information-optimal detector segmentation is a comb, not a contiguous
    segmentation the aperture alone can express.

    Parameters
    ----------
    compiled
        The compiled six-bin rule from `compile_bridge`.
    provider
        The analytic score provider.

    Returns
    -------
    tuple
        The grid `u` values, the predicted labels, and the number of maximal
        constant-label runs.
    """
    grid = np.linspace(0.0, U_MAX, COMB_GRID_POINTS)[:, None]
    grid_scores = np.asarray(provider.score(grid, execution=EXECUTION))
    labels = np.asarray(compiled.predict_scores(grid_scores, execution=EXECUTION))
    runs = 1 + int(np.sum(np.diff(labels) != 0))
    # `u = 0` and `u = U_MAX` are the same physical fringe phase, so a run that
    # touches both ends of the linear grid is one comb tooth, not two.
    if runs > 1 and labels[0] == labels[-1]:
        runs -= 1
    return grid[:, 0], labels, runs


@dataclass(frozen=True, slots=True)
class Study:
    """Everything the page and the figure need from one deterministic run."""

    metrics: dict[str, object]
    sample: TrainSample = field(repr=False)
    sweep: list[SweepRow] = field(repr=False)
    compiled: sq.Quantizer = field(repr=False)
    comb_u: np.ndarray = field(repr=False)
    comb_labels: np.ndarray = field(repr=False)


def run_study(
    *,
    n_nodes: int | None = None,
    soft_steps: int | None = None,
    budgets: tuple[int, ...] = BIN_SWEEP,
) -> Study:
    """Run the whole Michelson-phase study and return its metrics and arrays.

    Parameters
    ----------
    n_nodes
        Midpoint-quadrature node count of the finite score table.
    soft_steps
        Adam step budget of the soft profiled reusable-rule fit.
    budgets
        Bin budgets swept against the certified ceiling.

    Returns
    -------
    Study
        The exact structure written to
        ``docs/examples/assets/michelson-phase.json``, together with the
        arrays the figure draws.
    """
    n_nodes = example_scale(8_000, 2_000) if n_nodes is None else n_nodes
    soft_steps = example_scale(300, 80) if soft_steps is None else soft_steps

    provider = build_provider()
    source = build_integration_source()
    sample = build_train_sample(provider, n_nodes=n_nodes)
    test_sample = build_train_sample(provider, n_nodes=n_nodes + 1)

    closed_form = closed_form_information()
    information = np.asarray(
        sq.fisher_information(sample.scores, sample.weights, execution=EXECUTION)
    )
    i_phiphi, i_phieps, i_epseps = (
        float(information[0, 0]),
        float(information[0, 1]),
        float(information[1, 1]),
    )
    assert abs(i_phiphi - closed_form["i_phiphi"]) < 1e-12
    assert abs(i_phieps - closed_form["i_phieps"]) < 1e-12

    reference = unbinned_profiled_information(sample.scores, sample.weights)
    correlation = i_phieps / np.sqrt(i_phiphi * i_epseps)
    cost_of_profiling = 1.0 - reference / i_phiphi

    sweep = sweep_bin_budget(sample, reference, budgets)
    by_bins = {row.n_bins: row for row in sweep}
    headline = by_bins[HEADLINE_BINS]

    bridge, compiled = compile_bridge(headline, test_sample, n_bins=HEADLINE_BINS)
    comb_u, comb_labels, runs = comb_runs(compiled, provider)
    rules = reusable_rules(provider, source, sample, n_bins=HEADLINE_BINS, soft_steps=soft_steps)

    metrics: dict[str, object] = {
        "problem": "michelson_phase",
        "v0": V0,
        "phi0": PHI0,
        "fringes": FRINGES,
        "u_max": U_MAX,
        "n_nodes": n_nodes,
        "headline_bins": HEADLINE_BINS,
        "closed_form": {
            "i_phiphi": i_phiphi,
            "i_phieps": i_phieps,
            "i_epseps": i_epseps,
            "correlation": float(correlation),
        },
        "profiled_ceiling": reference,
        "cost_of_profiling": float(cost_of_profiling),
        "sweep": [
            {
                "n_bins": row.n_bins,
                "equal_width_retention": row.equal_width_retention,
                "d_optimal_retention": row.d_optimal_retention,
                "profiled_retention": row.profiled_retention_value,
                "ceiling_retention": row.ceiling_retention,
                "bound_gap": row.bound_gap,
                "equal_width_runs": label_runs(
                    sample.observations, row.equal_width_labels, u_max=U_MAX
                ),
                "d_runs": label_runs(sample.observations, row.d_labels, u_max=U_MAX),
                "profiled_runs": label_runs(sample.observations, row.profiled_labels, u_max=U_MAX),
            }
            for row in sweep
        ],
        "compile_bridge": asdict(bridge),
        "rules": [asdict(row) for row in rules],
        "comb": {"n_grid": COMB_GRID_POINTS, "n_runs": runs},
    }
    return Study(
        metrics=metrics,
        sample=sample,
        sweep=sweep,
        compiled=compiled,
        comb_u=comb_u,
        comb_labels=comb_labels,
    )


def make_figure(study: Study) -> Figure:
    """Render the two-panel Michelson-phase dashboard.

    Left: the score trajectory `(s_phi, s_epsilon)` -- a single curve, since
    `u` is one-dimensional -- colored by the compiled six-bin rule's
    prediction. Right: the same labels drawn back onto `u` over the four
    fringes, above the equal-width segmentation for contrast, so the comb and
    the aliasing are visible in one look.

    Parameters
    ----------
    study
        The object returned by `run_study`.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.
    """
    colors = [
        "#38618c",
        "#c0563c",
        "#4f9d69",
        "#b8860b",
        "#7b5ea7",
        "#3c8f9c",
        "#a8577e",
        "#666666",
    ]
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)

    u = study.comb_u
    labels = study.comb_labels
    s_phi = -V0 * np.sin(u) / (1.0 + V0 * np.cos(u))
    s_eps = u * s_phi - V0
    left.scatter(s_phi, s_eps, c=[colors[label % len(colors)] for label in labels], s=3)
    left.set(
        xlabel=r"$s_\varphi$",
        ylabel=r"$s_\epsilon$",
        title="Score trajectory, colored by the compiled six-bin rule",
    )

    headline = next(row for row in study.sweep if row.n_bins == HEADLINE_BINS)
    order = np.argsort(u, kind="stable")
    sorted_u = u[order]

    def bands(row_labels: np.ndarray, row: float) -> None:
        sorted_labels = row_labels[order]
        edges = np.flatnonzero(np.diff(sorted_labels) != 0)
        starts = np.concatenate([[0], edges + 1])
        stops = np.concatenate([edges + 1, [len(sorted_u)]])
        for start, stop in zip(starts, stops, strict=True):
            left_edge = float(sorted_u[start])
            right_edge = float(sorted_u[stop - 1])
            right.broken_barh(
                [(left_edge, max(right_edge - left_edge, 1e-3))],
                (row - 0.4, 0.8),
                facecolors=colors[int(sorted_labels[start]) % len(colors)],
            )

    equal_labels_on_grid = equal_width_labels(u[:, None], HEADLINE_BINS)
    bands(equal_labels_on_grid, 0.0)
    bands(labels, 1.0)
    right.set(
        xlim=(0.0, U_MAX),
        ylim=(-0.8, 1.8),
        yticks=[0.0, 1.0],
        yticklabels=["equal-width segments", "compiled D-optimal (comb)"],
        xlabel="fringe phase $u$",
        title=f"{headline.n_bins}-bin comb versus equal-width segmentation",
    )
    figure.suptitle("Michelson interferometer phase, profiled against fringe frequency")
    return figure


def make_score_figures() -> dict[Path, Figure]:
    """Return the walkthrough's three analytic-score panels, keyed by output path.

    These illustrate the closed-form score of the fringe model, so none of them
    depends on a fit: they are `s_phi` and `s_eps` at the reference point, drawn
    once along the detector and once against each other. The walkthrough places
    them beside the paragraphs that derive each expression.

    They are written as SVG rather than PNG because they are line art, and they
    go to `SCORE_FIGURE_DIR` rather than beside the study's own PNG because the
    portal serves them directly.

    Returns
    -------
    dict[pathlib.Path, matplotlib.figure.Figure]
        Output path to figure, for the caller to write and close.
    """
    curve = "#38618c"
    u = np.linspace(0.0, U_MAX, 2_001)
    s_phi = -V0 * np.sin(u) / (1.0 + V0 * np.cos(u))
    s_eps = u * s_phi - V0

    along, along_axes = plt.subplots(figsize=(8.8, 3.6), constrained_layout=True)
    along_axes.plot(u / np.pi, s_phi, color=curve, linewidth=1.4)
    along_axes.axhline(0.0, color="#666666", linewidth=0.8)
    along_axes.set(
        xlim=(0.0, U_MAX / np.pi),
        xlabel=r"Detector coordinate $u/\pi$",
        ylabel=r"$s_\varphi(u)$",
        title="Phase score along the detector",
    )

    growth, growth_axes = plt.subplots(figsize=(8.8, 3.6), constrained_layout=True)
    growth_axes.plot(u / np.pi, s_eps, color=curve, linewidth=1.4)
    growth_axes.axhline(0.0, color="#666666", linewidth=0.8)
    growth_axes.set(
        xlim=(0.0, U_MAX / np.pi),
        xlabel=r"Detector coordinate $u/\pi$",
        ylabel=r"$s_\epsilon(u)$",
        title="Fringe-frequency score along the detector",
    )

    space, space_axes = plt.subplots(figsize=(6.1, 5.1), constrained_layout=True)
    space_axes.plot(s_phi, s_eps, color=curve, linewidth=1.0)
    space_axes.axhline(0.0, color="#666666", linewidth=0.8)
    space_axes.axvline(0.0, color="#666666", linewidth=0.8)
    space_axes.set(
        xlabel=r"$s_\varphi$",
        ylabel=r"$s_\epsilon$",
        title="Michelson model in score space",
    )

    return {
        SCORE_FIGURE_DIR / "michelson-phase-score.svg": along,
        SCORE_FIGURE_DIR / "michelson-frequency-score.svg": growth,
        SCORE_FIGURE_DIR / "michelson-score-space.svg": space,
    }


def main() -> None:
    """Run the study, then write the committed JSON and every committed figure."""
    study = run_study()
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as stream:
        json.dump(study.metrics, stream, indent=2)
        stream.write("\n")
    figure = make_figure(study)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)

    SCORE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path, panel in make_score_figures().items():
        panel.savefig(path)
        plt.close(panel)


if __name__ == "__main__":
    main()
