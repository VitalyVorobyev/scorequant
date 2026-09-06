"""A Michelson interferometer phase: D-optimal quantization, then profiled D_s.

This script is the deterministic generator behind
`docs/examples/michelson-phase.md` and the portal walkthrough
`website/walkthroughs/michelson.mdx`. It exercises the analytic
`ScoreFunction` route (a score callback against a bounded
`IntegrationSource`) on the NumPy backend, on a model exact enough that the
library's own retained-information numbers are a check on the mathematics.

The model: a photon's position `u` along a detector spanning `N_FRINGES`
whole fringes, with the phase `phi` the parameter of interest and the
fractional fringe-frequency error `epsilon` a nuisance that can imitate a
phase shift. The study tells a two-act story:

* **Act one, ordinary D-optimality.** The two-dimensional score is quantized
  into `N_BINS` cells by maximizing `log det` of the binned Fisher matrix.
  The exchange-stable finite partition is a Mahalanobis-Voronoi partition of
  score space (book chapter 8, Theorem 3), so it compiles into a reusable
  rule; the study records that rule's geometry, its pull-back onto the
  detector, and its predictions for new positions.
* **Act two, profiled D_s.** Only the phase is reported, so the objective
  becomes the Schur complement `I_phiphi - I_phieps^2 / I_epseps`. The study
  fits the finite profiled partition from the certified efficient-score
  bound, diagnoses how fragmented its detector pull-back is and how much of
  that fragmentation the exchange moves created, and fits the reusable
  soft-Voronoi profiled rule that a profiled partition cannot compile into.

It writes `docs/examples/assets/michelson-phase.json`, two study figures
beside it, and the two closed-form score panels the walkthrough shows.

Run it with::

    JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run python -m examples.michelson_phase
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.figure import Figure

import scorequant as sq
from examples._env import example_scale

ASSET_DIR = Path("docs/examples/assets")
METRICS_PATH = ASSET_DIR / "michelson-phase.json"
D_GEOMETRY_FIGURE_PATH = ASSET_DIR / "michelson-d-geometry.png"
PROFILED_FIGURE_PATH = ASSET_DIR / "michelson-profiled-ds.png"

#: Where the portal's committed figure files live (ADR 0032). Unlike
#: `website/static/walkthrough-figures/`, which `generate_walkthroughs.py`
#: refills from `docs/examples/assets/` and `.gitignore` excludes, this
#: directory is committed: the walkthrough reads the closed-form score panels
#: straight from it, so a file written here is committed alongside the code.
SCORE_FIGURE_DIR = Path("website/static/figures")

#: Every call in this study runs on the portable NumPy backend at float64.
EXECUTION = sq.ExecutionConfig(backend="numpy", precision="float64", device="cpu")

#: Fixed fringe visibility, phase reference point, and fringe count. The
#: detector coordinate `u` is dimensionless, scaled so one fringe period has
#: length `2 pi`; it ranges over `[0, U_MAX)` with `U_MAX = 2 pi N_FRINGES`.
VISIBILITY = 0.60
PHI0 = 0.0
N_FRINGES = 4
U_MAX = 2.0 * np.pi * N_FRINGES

#: Solver seed shared by every fit in the study.
SEED = 4
#: Bin budget of the headline comparison, the compile bridge and the diagnostics.
N_BINS = 6
#: Bin budgets swept against the certified profiled ceiling.
BIN_SWEEP = (4, 6, 8, 10)
#: Tensor Gauss-Legendre order of the `IntegrationSource` route used for the
#: reusable-rule fits.
GAUSS_LEGENDRE_ORDER = 256
#: Adam step budget of the soft-Voronoi profiled fit at full scale.
SOFT_STEPS = 1000
#: Three fresh detector positions every reusable rule is asked to classify.
NEW_OBSERVATIONS = np.array([[1.0], [7.0], [15.0]])
#: A detector run narrower than this (about four percent of a fringe period)
#: counts as a fragment in the diagnostics below.
NARROW_RUN_WIDTH = 0.25
#: Widths below which runs are absorbed into a neighbour, one row of the
#: smoothing ladder each. A diagnostic of what the fragments are worth, not a
#: regularizer: nothing in the fits uses these numbers.
SMOOTHING_THRESHOLDS = (0.15, 0.25, 0.5, 1.0)

#: Names of the two score columns, in column order, and the interest column.
SCHEMA = sq.ScoreSchema(("phase", "fringe_frequency"))
INTEREST = SCHEMA.select("phase")

#: The portal's counter palette (`website/src/components/ApertureStrip.tsx`,
#: light theme), so counter `k` is the same colour in the study figures and in
#: the interactive strips.
COUNTER_COLORS = (
    "#276be8",
    "#bc7413",
    "#bb3d3d",
    "#29405d",
    "#178a8b",
    "#189136",
    "#7b6ce7",
    "#c937de",
    "#1f84c0",
    "#dd2db1",
)
CURVE_COLOR = "#38618c"
NEUTRAL_COLOR = "#666666"


def counter_color(label: int) -> str:
    """Return the palette colour of counter ``label``, wrapping past the palette."""
    return COUNTER_COLORS[int(label) % len(COUNTER_COLORS)]


def fringe_density(observations: np.ndarray) -> np.ndarray:
    """Return the normalized one-photon density along the detector at the reference point.

    `1 + VISIBILITY cos(u)` integrates to `U_MAX` over `N_FRINGES` whole
    periods, so dividing by `U_MAX` makes this a proper density: a source
    built from it carries unit total mass, and `fisher_information` on that
    source is therefore the *per-photon* Fisher information the closed forms
    describe.
    """
    u = np.asarray(observations)[:, 0]
    return (1.0 + VISIBILITY * np.cos(u)) / U_MAX


def michelson_score(observations: np.ndarray) -> np.ndarray:
    """Return the exact score `(s_phi, s_epsilon)` at `(phi0, epsilon0) = (0, 0)`.

    `s_epsilon = u * s_phi - VISIBILITY`; the constant is the normalizer
    derivative `d/d_epsilon log Z`, not a centering convenience -- dropping it
    would leave `E[s_epsilon] = VISIBILITY != 0`, violating the library's
    never-center-scores invariant. Both components are bounded because
    `1 + VISIBILITY cos(u) >= 1 - VISIBILITY > 0`, so the `ScoreFunction`
    finiteness contract holds by construction.
    """
    u = np.asarray(observations)[:, 0]
    denominator = 1.0 + VISIBILITY * np.cos(u)
    s_phi = -VISIBILITY * np.sin(u) / denominator
    s_epsilon = u * s_phi - VISIBILITY
    return np.column_stack([s_phi, s_epsilon])


def build_provider() -> sq.ScoreFunction:
    """Return the analytic `ScoreFunction` provider, with schema and exact provenance."""
    return sq.ScoreFunction(
        michelson_score,
        provenance=sq.ScoreProvenance(kind="exact", reference_point=(PHI0, 0.0)),
        schema=SCHEMA,
    )


def build_integration_source() -> sq.IntegrationSource:
    """Return the bounded `IntegrationSource` reference measure over one photon's position."""
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
        Detector positions with shape ``[N, 1]``, increasing.
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
    """Return the two closed-form unbinned Fisher entries at `VISIBILITY`.

    Returns
    -------
    dict of float
        ``i_phiphi = 1 - sqrt(1 - VISIBILITY**2)`` and
        ``i_phieps = i_phiphi * U_MAX / 2``.
    """
    i_phiphi = 1.0 - np.sqrt(1.0 - VISIBILITY**2)
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
    weighted score table.
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
        Detector positions with shape ``[N, 1]``.
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
    """Run-length encode a labeling along increasing `u` into detector-edge triples.

    Parameters
    ----------
    observations
        Detector positions with shape ``[N, 1]``, already sorted by `u` --
        the midpoint-quadrature nodes `build_train_sample` returns are.
    labels
        Integer label per row, aligned with `observations`.
    u_max
        Upper edge of the detector, `U_MAX` for this study.

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


def periodic_runs(
    observations: np.ndarray, labels: np.ndarray, *, u_max: float
) -> list[list[float]]:
    """Run-length encode a labeling with the two detector ends joined.

    `s(0)` and `s(u_max)` are the same score vector, so a run that touches
    both ends of the linear detector is one piece of one score-space cell,
    not two. The joined run is reported once, starting at a negative `u`.
    """
    runs = label_runs(observations, labels, u_max=u_max)
    if len(runs) > 1 and runs[0][2] == runs[-1][2]:
        first, last = runs[0], runs[-1]
        runs = [[last[0] - u_max, first[1], first[2]], *runs[1:-1]]
    return runs


@dataclass(frozen=True, slots=True)
class RunDiagnostics:
    """How a labeling of the detector fragments, with the two detector ends joined.

    Attributes
    ----------
    n_runs
        Number of maximal constant-label runs.
    runs_per_bin
        Number of runs carrying each label, in label order.
    narrow_runs
        Number of runs narrower than `NARROW_RUN_WIDTH`.
    min_run_width
        Width in `u` of the narrowest run.
    min_run_mass
        Probability mass of the narrowest run.
    """

    n_runs: int
    runs_per_bin: list[int]
    narrow_runs: int
    min_run_width: float
    min_run_mass: float


def run_diagnostics(sample: TrainSample, labels: np.ndarray, *, n_bins: int) -> RunDiagnostics:
    """Measure the fragmentation of one labeling of the training sample.

    Parameters
    ----------
    sample
        The finite weighted score table, sorted by `u`.
    labels
        Integer label per row.
    n_bins
        Bin budget of the labeling, for the per-bin run counts.

    Returns
    -------
    RunDiagnostics
        Run counts and the narrowest run's width and mass.
    """
    labels = np.asarray(labels)
    runs = periodic_runs(sample.observations, labels, u_max=U_MAX)
    n_nodes = labels.shape[0]
    step = U_MAX / n_nodes
    widths = np.array([stop - start for start, stop, _ in runs])
    masses = []
    for start, stop, _ in runs:
        lo, hi = round(start / step), round(stop / step)
        rows = np.arange(lo, hi) % n_nodes
        masses.append(float(np.sum(sample.weights[rows])))
    narrowest = int(np.argmin(widths))
    return RunDiagnostics(
        n_runs=len(runs),
        runs_per_bin=np.bincount([label for _, _, label in runs], minlength=n_bins).tolist(),
        narrow_runs=int(np.sum(widths < NARROW_RUN_WIDTH)),
        min_run_width=float(widths[narrowest]),
        min_run_mass=masses[narrowest],
    )


def absorb_narrow_runs(
    observations: np.ndarray, labels: np.ndarray, *, min_width: float, u_max: float
) -> np.ndarray:
    """Relabel every run narrower than ``min_width`` into its wider neighbour.

    The narrowest run is absorbed first and the runs recomputed, until none is
    narrower than the threshold. Neighbours wrap around the detector ends.
    This is a diagnostic transformation of a finished labeling, never part of
    a fit.

    Parameters
    ----------
    observations
        Detector positions with shape ``[N, 1]``, increasing.
    labels
        Integer label per row.
    min_width
        Runs narrower than this, in `u`, are absorbed.
    u_max
        Upper edge of the detector.

    Returns
    -------
    numpy.ndarray
        The smoothed labeling.
    """
    labels = np.array(labels, copy=True)
    n_nodes = labels.shape[0]
    step = u_max / n_nodes
    while True:
        runs = periodic_runs(observations, labels, u_max=u_max)
        narrow = [
            (stop - start, index)
            for index, (start, stop, _) in enumerate(runs)
            if stop - start < min_width - 1e-12
        ]
        if not narrow or len(runs) == 1:
            return labels
        _, index = min(narrow)
        start, stop, _ = runs[index]
        left = runs[index - 1]
        right = runs[(index + 1) % len(runs)]
        donor = left if (left[1] - left[0]) >= (right[1] - right[0]) else right
        # A run joined across the seam starts at a negative `u`; the modulo
        # maps its node range back onto the linear table.
        nodes = np.arange(round(start / step), round(stop / step)) % n_nodes
        labels[nodes] = donor[2]


@dataclass(frozen=True, slots=True)
class SmoothingRow:
    """One row of the smoothing ladder: what a labeling is worth without its fragments.

    Attributes
    ----------
    min_width
        Absorption threshold in `u`.
    n_runs
        Runs left after absorption, with the detector ends joined.
    bins_used
        Distinct labels left after absorption.
    retention
        Profiled phase-information retention of the smoothed labeling.
    """

    min_width: float
    n_runs: int
    bins_used: int
    retention: float


def smoothing_ladder(
    sample: TrainSample,
    labels: np.ndarray,
    *,
    n_bins: int,
    thresholds: tuple[float, ...] = SMOOTHING_THRESHOLDS,
) -> list[SmoothingRow]:
    """Absorb ever wider runs and record what each absorption costs.

    Parameters
    ----------
    sample
        The finite weighted score table.
    labels
        The labeling to smooth, typically the finite profiled partition.
    n_bins
        Bin budget of the labeling.
    thresholds
        Absorption thresholds, one row each, in increasing order.

    Returns
    -------
    list of SmoothingRow
        One row per threshold.
    """
    rows: list[SmoothingRow] = []
    for min_width in thresholds:
        smoothed = absorb_narrow_runs(sample.observations, labels, min_width=min_width, u_max=U_MAX)
        rows.append(
            SmoothingRow(
                min_width=float(min_width),
                n_runs=len(periodic_runs(sample.observations, smoothed, u_max=U_MAX)),
                bins_used=int(np.unique(smoothed).shape[0]),
                retention=profiled_retention(sample, smoothed, n_bins),
            )
        )
    return rows


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
    bound: sq.EfficientScoreBound = field(repr=False)


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
                bound=bound,
            )
        )
    return rows


def predictions(
    rule: sq.Quantizer | sq.QuantizerResult, provider: sq.ScoreFunction
) -> list[dict[str, float | int]]:
    """Classify `NEW_OBSERVATIONS` with a reusable rule, observation to score to bin.

    Parameters
    ----------
    rule
        A compiled `Quantizer` or a fitted `QuantizerResult`; both expose
        `predict_scores`.
    provider
        The analytic score provider that turns a position into a score.

    Returns
    -------
    list of dict
        ``{"u": position, "bin": label}`` per new observation.
    """
    new_scores = np.asarray(provider.score(NEW_OBSERVATIONS, execution=EXECUTION))
    labels = np.asarray(rule.predict_scores(new_scores, execution=EXECUTION))
    return [
        {"u": float(u), "bin": int(label)}
        for u, label in zip(NEW_OBSERVATIONS[:, 0], labels, strict=True)
    ]


@dataclass(frozen=True, slots=True)
class CompileBridge:
    """The compile bridge for the headline D-optimal partition, and the profiled refusal.

    Attributes
    ----------
    exchange_stable
        Whether the D-optimal partition the rule was compiled from is
        exchange-stable.
    voronoi_consistent
        Whether the partition's own `GeometryReport` certifies the
        Mahalanobis-Voronoi geometry the compile bridge relies on.
    compiled_test_retention
        Overall D-efficiency of the compiled rule on a fresh quadrature
        sample. A D-efficiency, not a profiled number: never compared with
        the phase-retention columns.
    reproduces_training_labels
        Whether the compiled rule reproduces every training label.
    predictions
        The compiled rule's bin for each of `NEW_OBSERVATIONS`.
    refusal_message
        `str` of the `RefusalError` `compile_quantizer()` raises on the
        profiled partition.
    """

    exchange_stable: bool
    voronoi_consistent: bool
    compiled_test_retention: float
    reproduces_training_labels: bool
    predictions: list[dict[str, float | int]]
    refusal_message: str


def compile_bridge(
    headline: SweepRow, sample: TrainSample, test_sample: TrainSample, provider: sq.ScoreFunction
) -> tuple[CompileBridge, sq.Quantizer]:
    """Compile the headline D-optimal partition and record the profiled refusal.

    Parameters
    ----------
    headline
        The `SweepRow` at the headline bin budget.
    sample
        The training sample, to check the compiled rule reproduces its labels.
    test_sample
        A held-out quadrature sample to evaluate the compiled rule on.
    provider
        The analytic score provider, for the prediction demonstration.

    Returns
    -------
    tuple
        The `CompileBridge` record and the compiled `Quantizer`, kept for the
        geometry figure.
    """
    compiled = headline.d_partition.compile_quantizer(execution=EXECUTION)
    compiled_report = compiled.evaluate_scores(
        test_sample.scores, test_sample.weights, execution=EXECUTION
    )
    reproduced = np.asarray(compiled.predict_scores(sample.scores, execution=EXECUTION))
    geometry = headline.d_partition.geometry
    try:
        headline.profiled_partition.compile_quantizer(execution=EXECUTION)
        raise AssertionError("a profiled-D partition must refuse compile_quantizer()")
    except sq.RefusalError as error:
        refusal_message = str(error)
    return (
        CompileBridge(
            exchange_stable=bool(headline.d_partition.exchange_stable),
            voronoi_consistent=bool(geometry is not None and geometry.voronoi_consistent),
            compiled_test_retention=float(compiled_report.geometric_mean_retention),
            reproduces_training_labels=bool(np.array_equal(reproduced, headline.d_labels)),
            predictions=predictions(compiled, provider),
            refusal_message=refusal_message,
        ),
        compiled,
    )


@dataclass(frozen=True, slots=True)
class ProfiledTrace:
    """What the exchange did to the efficient-score initializer at the headline budget.

    Attributes
    ----------
    initial
        Fragmentation of `bound.labels`, the certified interval solution in
        the efficient score, pulled back to the detector.
    initial_retention
        Profiled retention of that initializer.
    final
        Fragmentation of the exchange-stable profiled partition.
    final_retention
        Its profiled retention.
    accepted_moves, scans
        Exchange work: rows relocated, and full candidate scans.
    exchange_stable
        Whether the solver ended exchange-stable.
    objective_gain
        `objective_history[-1] - objective_history[0]`, in log units of the
        profiled objective.
    retention_gain
        `final_retention - initial_retention`.
    relabelled_fraction
        Fraction of nodes whose label the exchange changed.
    """

    initial: RunDiagnostics
    initial_retention: float
    final: RunDiagnostics
    final_retention: float
    accepted_moves: int
    scans: int
    exchange_stable: bool
    objective_gain: float
    retention_gain: float
    relabelled_fraction: float


def profiled_trace(sample: TrainSample, headline: SweepRow) -> ProfiledTrace:
    """Compare the profiled partition with the certified initializer it started from.

    Parameters
    ----------
    sample
        The finite weighted score table.
    headline
        The `SweepRow` at the headline bin budget.

    Returns
    -------
    ProfiledTrace
        Fragmentation and retention before and after the exchange.
    """
    n_bins = headline.n_bins
    initial_labels = np.asarray(headline.bound.labels)
    final_labels = headline.profiled_labels
    history = np.asarray(headline.profiled_partition.objective_history)
    initial_retention = profiled_retention(sample, initial_labels, n_bins)
    return ProfiledTrace(
        initial=run_diagnostics(sample, initial_labels, n_bins=n_bins),
        initial_retention=initial_retention,
        final=run_diagnostics(sample, final_labels, n_bins=n_bins),
        final_retention=headline.profiled_retention_value,
        accepted_moves=int(headline.profiled_partition.accepted_moves),
        scans=int(headline.profiled_partition.scans),
        exchange_stable=bool(headline.profiled_partition.exchange_stable),
        objective_gain=float(history[-1] - history[0]),
        retention_gain=float(headline.profiled_retention_value - initial_retention),
        relabelled_fraction=float(np.mean(initial_labels != final_labels)),
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
        the same ceiling as every column of the bin-budget sweep.
    criterion_efficiency
        What each rule scores on the criterion it actually optimized:
        `train_report` for `DOptimality`, `train_profiled_report` for
        `ProfiledDOptimality`. Not comparable across rows -- the denominators
        differ -- which is exactly why `profiled_retention` exists beside it.
    hardening_gap
        Soft-to-hard retention gap; `0.0` for the compiled route.
    n_runs
        Detector runs of the rule's labels on the training sample, with the
        detector ends joined.
    predictions
        The rule's bin for each of `NEW_OBSERVATIONS`.
    """

    key: str
    label: str
    criterion: str
    solver: str
    profiled_retention: float
    criterion_efficiency: float
    hardening_gap: float
    n_runs: int
    predictions: list[dict[str, float | int]]


@dataclass(frozen=True, slots=True)
class ReusableRules:
    """The two reusable rules and their comparison rows."""

    rows: list[RuleRow]
    d_rule: sq.QuantizerResult = field(repr=False)
    ds_rule: sq.QuantizerResult = field(repr=False)


def reusable_rules(
    provider: sq.ScoreFunction,
    source: sq.IntegrationSource,
    sample: TrainSample,
    *,
    n_bins: int,
    soft_steps: int,
) -> ReusableRules:
    """Fit a reusable rule under each criterion directly from the integration source.

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
    ReusableRules
        One row for `DOptimality` (compiled exchange) and one for
        `ProfiledDOptimality` (soft Voronoi) -- the latter is the only route to
        a *reusable* profiled rule, since finite profiled-D labels have no
        compile bridge -- plus both fitted results.
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
        criterion=sq.ProfiledDOptimality(interest=INTEREST),
        config=sq.SoftVoronoiConfig(
            seed=SEED,
            initializer_restarts=8,
            max_steps=soft_steps,
            record_every=max(soft_steps // 8, 1),
        ),
        execution=EXECUTION,
    )
    assert ds_rule.train_profiled_report is not None
    d_labels = np.asarray(d_rule.predict_scores(sample.scores, execution=EXECUTION))
    ds_labels = np.asarray(ds_rule.predict_scores(sample.scores, execution=EXECUTION))
    rows = [
        RuleRow(
            key="d_rule",
            label="Plain D, compiled exchange",
            criterion="DOptimality",
            solver="DExchangeConfig",
            profiled_retention=profiled_retention(sample, d_labels, n_bins),
            criterion_efficiency=float(d_rule.train_report.geometric_mean_retention),
            hardening_gap=float(d_rule.hardening_gap or 0.0),
            n_runs=run_diagnostics(sample, d_labels, n_bins=n_bins).n_runs,
            predictions=predictions(d_rule, provider),
        ),
        RuleRow(
            key="ds_rule",
            label="Profiled D_s, soft Voronoi",
            criterion="ProfiledDOptimality",
            solver="SoftVoronoiConfig",
            profiled_retention=profiled_retention(sample, ds_labels, n_bins),
            criterion_efficiency=float(ds_rule.train_profiled_report.geometric_mean_retention),
            hardening_gap=float(ds_rule.hardening_gap or 0.0),
            n_runs=run_diagnostics(sample, ds_labels, n_bins=n_bins).n_runs,
            predictions=predictions(ds_rule, provider),
        ),
    ]
    return ReusableRules(rows=rows, d_rule=d_rule, ds_rule=ds_rule)


@dataclass(frozen=True, slots=True)
class Study:
    """Everything the pages and the figures need from one deterministic run."""

    metrics: dict[str, object]
    sample: TrainSample = field(repr=False)
    sweep: list[SweepRow] = field(repr=False)
    compiled: sq.Quantizer = field(repr=False)
    rules: ReusableRules = field(repr=False)


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
        Bin budgets swept against the certified ceiling; must include `N_BINS`.

    Returns
    -------
    Study
        The exact structure written to
        ``docs/examples/assets/michelson-phase.json``, together with the
        objects the figures draw.
    """
    n_nodes = example_scale(8_000, 2_000) if n_nodes is None else n_nodes
    soft_steps = example_scale(SOFT_STEPS, 80) if soft_steps is None else soft_steps

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
    headline = by_bins[N_BINS]

    bridge, compiled = compile_bridge(headline, sample, test_sample, provider)
    d_geometry = run_diagnostics(sample, headline.d_labels, n_bins=N_BINS)
    trace = profiled_trace(sample, headline)
    smoothing = smoothing_ladder(sample, headline.profiled_labels, n_bins=N_BINS)
    rules = reusable_rules(provider, source, sample, n_bins=N_BINS, soft_steps=soft_steps)

    metrics: dict[str, object] = {
        "problem": "michelson_phase",
        "visibility": VISIBILITY,
        "phi0": PHI0,
        "fringes": N_FRINGES,
        "u_max": U_MAX,
        "n_nodes": n_nodes,
        "headline_bins": N_BINS,
        "narrow_run_width": NARROW_RUN_WIDTH,
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
        "d_geometry": asdict(d_geometry),
        "compile_bridge": asdict(bridge),
        "profiled_diagnostics": {
            **asdict(trace),
            "smoothing": [asdict(row) for row in smoothing],
        },
        "rules": [asdict(row) for row in rules.rows],
    }
    return Study(metrics=metrics, sample=sample, sweep=sweep, compiled=compiled, rules=rules)


def _draw_bands(axes: plt.Axes, sample: TrainSample, bands: list[tuple[str, np.ndarray]]) -> None:
    """Draw labelings as rows of coloured intervals along `u`, bottom row first."""
    for row, (_, labels) in enumerate(bands):
        for start, stop, label in label_runs(sample.observations, labels, u_max=U_MAX):
            axes.broken_barh(
                [(start, stop - start)], (row - 0.4, 0.8), facecolors=counter_color(label)
            )
    fringe_edges = np.arange(N_FRINGES + 1) * 2.0 * np.pi
    for edge in fringe_edges[1:-1]:
        axes.axvline(edge, color="white", linewidth=0.8)
    axes.set(
        xlim=(0.0, U_MAX),
        ylim=(-0.7, len(bands) - 0.3),
        yticks=list(range(len(bands))),
        yticklabels=[name for name, _ in bands],
        xticks=fringe_edges,
        xticklabels=["0"] + [f"{2 * k}π" for k in range(1, N_FRINGES + 1)],
        xlabel="detector coordinate $u$",
    )


def _draw_trajectory(axes: plt.Axes, sample: TrainSample, labels: np.ndarray) -> None:
    """Draw the score trajectory coloured by label, on axes with the origin marked."""
    s_phi = sample.scores[:, 0]
    s_eps = sample.scores[:, 1]
    axes.scatter(s_phi, s_eps, c=[counter_color(label) for label in labels], s=4, linewidths=0)
    axes.axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.6)
    axes.axvline(0.0, color=NEUTRAL_COLOR, linewidth=0.6)
    axes.set(xlabel=r"$s_\varphi$", ylabel=r"$s_\epsilon$")


def make_d_geometry_figure(study: Study, *, grid_points: int = 400) -> Figure:
    """Render the D-optimal partition in score space and along the detector.

    Top: the raw score plane, shaded by the compiled rule's own Mahalanobis-
    Voronoi cells (`compiled.predict_scores` on a dense grid of raw scores,
    so the cells drawn are the cells the library computes), the cell mean
    scores, and the trajectory coloured by the finite partition's labels.
    Bottom: the same labels laid back along the detector above the
    equal-width segments. A connected cell that the folded trajectory crosses
    several times is what makes the bottom row a comb.

    Parameters
    ----------
    study
        The object returned by `run_study`.
    grid_points
        Resolution of the shaded grid per axis.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.
    """
    headline = next(row for row in study.sweep if row.n_bins == N_BINS)
    sample = study.sample
    s_phi = sample.scores[:, 0]
    s_eps = sample.scores[:, 1]
    phi_extent = 1.1 * float(np.max(np.abs(s_phi)))
    eps_extent = 1.1 * float(np.max(np.abs(s_eps)))
    phi_grid = np.linspace(-phi_extent, phi_extent, grid_points)
    eps_grid = np.linspace(-eps_extent, eps_extent, grid_points)
    mesh_phi, mesh_eps = np.meshgrid(phi_grid, eps_grid)
    grid_scores = np.column_stack([mesh_phi.ravel(), mesh_eps.ravel()])
    cells = np.asarray(study.compiled.predict_scores(grid_scores, execution=EXECUTION))
    cells = cells.reshape(mesh_phi.shape)

    figure = plt.figure(figsize=(9.0, 9.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=[3.2, 1.0])
    plane = figure.add_subplot(grid[0, 0])
    strip = figure.add_subplot(grid[1, 0])

    tint = ListedColormap(
        [to_rgba(counter_color(label), alpha=0.18) for label in range(headline.n_bins)]
    )
    plane.pcolormesh(
        phi_grid,
        eps_grid,
        cells,
        cmap=tint,
        vmin=-0.5,
        vmax=headline.n_bins - 0.5,
        shading="nearest",
        rasterized=True,
    )
    _draw_trajectory(plane, sample, headline.d_labels)
    means = headline.d_partition.cell_score_means
    plane.scatter(
        means[:, 0],
        means[:, 1],
        c=[counter_color(label) for label in range(means.shape[0])],
        s=80,
        marker="o",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
    )
    for label, (mean_phi, mean_eps) in enumerate(means):
        plane.annotate(
            str(label),
            (mean_phi, mean_eps),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=9,
            color="black",
        )
    plane.set(
        xlim=(-phi_extent, phi_extent),
        ylim=(-eps_extent, eps_extent),
        title=(
            f"D-optimal partition, {headline.n_bins} cells\n"
            "cells of the compiled Mahalanobis-Voronoi rule, cell means, "
            "and the detector trajectory"
        ),
    )

    _draw_bands(
        strip,
        sample,
        [("equal-width", headline.equal_width_labels), ("D-optimal", headline.d_labels)],
    )
    strip.set_title(
        f"The same labels along the detector: {study.metrics['d_geometry']['n_runs']} runs"  # type: ignore[index]
        f" from {headline.n_bins} cells"
    )
    return figure


def make_profiled_figure(study: Study) -> Figure:
    """Render the finite profiled-D_s partition, designed to expose its fragmentation.

    Top: the trajectory coloured by the finite D_s labels, with no cell
    shading -- a profiled partition has no canonical Voronoi rule to shade.
    Middle: the efficient score along the detector, coloured by the same
    labels; the certified initializer is an interval partition of this
    curve, and its thin detector pieces sit where the curve is steep.
    Bottom: three strips along the detector at true width -- the
    efficient-score initializer, the exchange-stable profiled partition, and
    the reusable soft-Voronoi rule.

    Parameters
    ----------
    study
        The object returned by `run_study`.

    Returns
    -------
    matplotlib.figure.Figure
        The three-panel figure.
    """
    headline = next(row for row in study.sweep if row.n_bins == N_BINS)
    sample = study.sample
    u = sample.observations[:, 0]
    efficient = np.asarray(headline.bound.efficient_scores)[:, 0]
    initial_labels = np.asarray(headline.bound.labels)
    rule_labels = np.asarray(study.rules.ds_rule.predict_scores(sample.scores, execution=EXECUTION))
    diagnostics = study.metrics["profiled_diagnostics"]
    assert isinstance(diagnostics, dict)

    figure = plt.figure(figsize=(9.0, 11.0), constrained_layout=True)
    grid = figure.add_gridspec(3, 1, height_ratios=[2.6, 1.2, 1.2])
    plane = figure.add_subplot(grid[0, 0])
    curve = figure.add_subplot(grid[1, 0])
    strip = figure.add_subplot(grid[2, 0], sharex=curve)

    _draw_trajectory(plane, sample, headline.profiled_labels)
    plane.set_title(
        f"Profiled $D_s$ partition, {headline.n_bins} cells\n"
        "the trajectory coloured by the finite labels; no Voronoi cells to shade"
    )

    for start, stop, label in label_runs(
        sample.observations, headline.profiled_labels, u_max=U_MAX
    ):
        mask = (u >= start) & (u <= stop)
        curve.plot(u[mask], efficient[mask], color=counter_color(label), linewidth=1.6)
    curve.axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.6)
    curve.set(
        ylabel=r"efficient score $\hat s(u)$",
        title="The efficient score along the detector, coloured by the same labels",
    )
    curve.tick_params(labelbottom=False)

    _draw_bands(
        strip,
        sample,
        [
            ("soft-Voronoi rule", rule_labels),
            ("profiled $D_s$", headline.profiled_labels),
            ("efficient-score start", initial_labels),
        ],
    )
    strip.set_title(
        f"Along the detector: initializer {diagnostics['initial']['n_runs']} runs, "
        f"profiled $D_s$ {diagnostics['final']['n_runs']} runs, "
        f"reusable rule {study.rules.rows[1].n_runs} runs"
    )
    return figure


def make_score_figures() -> dict[Path, Figure]:
    """Return the walkthrough's two closed-form score panels, keyed by output path.

    Neither depends on a fit: the phase score along the detector, and the
    trajectory `(s_phi, s_epsilon)` in score space. They are written as SVG
    because they are line art, and they go to `SCORE_FIGURE_DIR` because the
    portal serves them directly.

    Returns
    -------
    dict[pathlib.Path, matplotlib.figure.Figure]
        Output path to figure, for the caller to write and close.
    """
    u = np.linspace(0.0, U_MAX, 2_001)
    s_phi = -VISIBILITY * np.sin(u) / (1.0 + VISIBILITY * np.cos(u))
    s_eps = u * s_phi - VISIBILITY

    along, along_axes = plt.subplots(figsize=(8.8, 3.6), constrained_layout=True)
    along_axes.plot(u / np.pi, s_phi, color=CURVE_COLOR, linewidth=1.4)
    along_axes.axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.8)
    along_axes.set(
        xlim=(0.0, U_MAX / np.pi),
        xlabel=r"Detector coordinate $u/\pi$",
        ylabel=r"$s_\varphi(u)$",
        title="Phase score along the detector",
    )

    space, space_axes = plt.subplots(figsize=(6.1, 5.1), constrained_layout=True)
    space_axes.plot(s_phi, s_eps, color=CURVE_COLOR, linewidth=1.0)
    space_axes.axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.8)
    space_axes.axvline(0.0, color=NEUTRAL_COLOR, linewidth=0.8)
    space_axes.set(
        xlabel=r"$s_\varphi$",
        ylabel=r"$s_\epsilon$",
        title="Michelson model in score space",
    )

    return {
        SCORE_FIGURE_DIR / "michelson-phase-score.svg": along,
        SCORE_FIGURE_DIR / "michelson-score-space.svg": space,
    }


def main() -> None:
    """Run the study, then write the committed JSON and every committed figure."""
    study = run_study()
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as stream:
        json.dump(study.metrics, stream, indent=2)
        stream.write("\n")
    for path, figure in (
        (D_GEOMETRY_FIGURE_PATH, make_d_geometry_figure(study)),
        (PROFILED_FIGURE_PATH, make_profiled_figure(study)),
    ):
        figure.savefig(path, dpi=160)
        plt.close(figure)

    # The SVGs are committed (ADR 0032), so they have to regenerate byte for
    # byte or every rerun shows up as a diff of nothing. Matplotlib stamps a
    # wall-clock `dc:date` and derives every `<defs>` element id from a random
    # salt; a fixed salt and a suppressed date make the file a pure function
    # of the data, the way the PNGs already are.
    plt.rcParams["svg.hashsalt"] = "scorequant-michelson-phase"
    SCORE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path, panel in make_score_figures().items():
        panel.savefig(path, metadata={"Date": None})
        plt.close(panel)


if __name__ == "__main__":
    main()
