"""The unguarded batch Mahalanobis-Lloyd step, and what the guard does instead.

This script is the deterministic generator behind the Lloyd counterexample
used in the paper and the explainers. It runs

* the committed eight-row counterexample: one frozen-metric batch relabeling
  that lowers `log det` of the binned information while improving the
  frozen-metric distortion it actually minimizes, together with the unguarded
  iteration continued to its own fixed point;
* the same eight rows under `scorequant.MahalanobisLloydConfig` with
  `guard="reject"` and with `guard="exchange"`, which is the acceptance trace;
* a failure-mode ledger: many unguarded runs from random starting labels across
  three synthetic problems, three sample sizes, and two bin budgets, counting
  how many of them ever step downhill and how many vacate a cell entirely;
* one large guarded run from random starting labels, so the batch phase has
  real work to do and the handoff to the exact exchange is visible;

and writes `examples/outputs/lloyd-nonmonotone.json` plus
`examples/outputs/lloyd-nonmonotone.png`.

Run it with::

    JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run python -m examples.lloyd_nonmonotone
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

import scorequant as sq
from examples._env import example_scale
from examples.synthetic_problems import (
    SyntheticDataset,
    signal_background_shape,
    spatial_sources,
    spectral_templates,
)

FIGURE_PATH = Path("examples/outputs/lloyd-nonmonotone.png")
METRICS_PATH = Path("examples/outputs/lloyd-nonmonotone.json")

#: The committed eight-row score table of the counterexample. Two parameters,
#: three cells, equal weights: the smallest configuration on which one frozen
#: metric batch relabeling can lower the determinant criterion.
COUNTEREXAMPLE_SCORES = np.array(
    [
        [0.1116, 0.4427],
        [-0.2932, 0.6537],
        [-0.5995, -1.2685],
        [-0.6848, -1.5456],
        [0.4810, 0.9521],
        [1.6707, 0.9370],
        [0.1689, 1.7090],
        [-0.8548, -1.8805],
    ]
)
#: The starting labels from which one batch step loses information.
COUNTEREXAMPLE_LABELS = np.array([1, 0, 0, 1, 2, 2, 2, 1])
#: Cells of the counterexample.
COUNTEREXAMPLE_BINS = 3

#: Synthetic problems the failure-mode ledger sweeps.
LEDGER_PROBLEMS = ("signal_background_shape", "spectral_templates", "spatial_sources")
#: Sample sizes of the ledger sweep.
LEDGER_SIZES = (60, 250, 1_000)
#: Bin budgets of the ledger sweep.
LEDGER_BUDGETS = (4, 6)
#: Random starting labelings tried per ledger configuration.
LEDGER_STARTS = 24
#: Iteration cap of one unguarded run.
UNGUARDED_MAX_ITER = 100

#: Sample size of the single large guarded run.
CLIMB_SIZE = 4_000
#: Bin budget of that run.
CLIMB_BINS = 6
#: Seed of its random starting labeling and of its solvers.
CLIMB_SEED = 0


def ledger_split(name: str, n_rows: int) -> SyntheticDataset:
    """Return the training split of one problem the failure ledger sweeps.

    Parameters
    ----------
    name
        One of `LEDGER_PROBLEMS`.
    n_rows
        Training rows to generate.

    Returns
    -------
    SyntheticDataset
        Observations, exact scores, and reference weights.
    """
    sizes = (n_rows, 20, 20)
    if name == "signal_background_shape":
        return signal_background_shape(
            background_rates=(1.0, 4.0), n_bins=6, sizes=sizes, seed=50
        ).train
    if name == "spectral_templates":
        return spectral_templates(sizes=sizes).train
    if name == "spatial_sources":
        return spatial_sources(sizes=sizes).train
    raise ValueError(f"unknown ledger problem {name!r}")


def raw_log_determinant(
    scores: np.ndarray, weights: np.ndarray, labels: np.ndarray, *, n_bins: int
) -> float:
    """Return the log determinant of the binned information of one labeling.

    Parameters
    ----------
    scores
        Score matrix with shape ``[N, P]``.
    weights
        Nonnegative measure weights with shape ``[N]``.
    labels
        Integer bin label per row.
    n_bins
        Number of cells, including empty ones.

    Returns
    -------
    float
        The raw log determinant, in the untransformed score coordinates. It
        differs from `scorequant.PartitionResult.objective` by the
        rule-independent constant ``log det`` of the unbinned information.
    """
    information = np.asarray(sq.binned_fisher_information(scores, labels, weights, n_bins=n_bins))
    return float(np.linalg.slogdet(information)[1])


def cell_means(
    scores: np.ndarray, weights: np.ndarray, labels: np.ndarray, n_bins: int
) -> np.ndarray:
    """Return the weighted score mean of every cell."""
    return np.stack(
        [
            np.average(scores[labels == cell], axis=0, weights=weights[labels == cell])
            for cell in range(n_bins)
        ]
    )


def frozen_metric_proposal(
    scores: np.ndarray, weights: np.ndarray, labels: np.ndarray, *, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Send every row to its nearest cell mean in the current criterion metric.

    This is the batch step the determinant criterion's geometry suggests: freeze
    the metric that the current labeling induces, and relabel every row at once.

    Parameters
    ----------
    scores, weights, labels
        The weighted score table and its current labeling.
    n_bins
        Number of cells.

    Returns
    -------
    tuple
        The proposed labeling, the cell means it was measured against, and the
        frozen metric itself.
    """
    information = np.asarray(sq.binned_fisher_information(scores, labels, weights, n_bins=n_bins))
    means = cell_means(scores, weights, labels, n_bins)
    metric = np.linalg.inv(information)
    residuals = scores[:, None, :] - means[None, :, :]
    distances = np.einsum("nkp,pq,nkq->nk", residuals, metric, residuals)
    return np.argmin(distances, axis=1), means, metric


def frozen_metric_distortion(
    scores: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    means: np.ndarray,
    metric: np.ndarray,
) -> float:
    """Return the weighted within-cell distortion in a fixed metric.

    Parameters
    ----------
    scores, weights, labels
        The weighted score table and the labeling to score.
    means
        Cell means the distances are measured to.
    metric
        The frozen metric, held fixed while `labels` varies.

    Returns
    -------
    float
        Weighted mean squared Mahalanobis distance to the assigned cell mean.
        A nearest-centroid relabeling minimizes exactly this quantity, which is
        why it is the surrogate the batch step actually improves.
    """
    offsets = scores - means[labels]
    total = float(np.einsum("np,pq,nq->", weights[:, None] * offsets, metric, offsets))
    return total / float(np.sum(weights))


@dataclass(frozen=True, slots=True)
class UnguardedRun:
    """One unguarded batch iteration, run until it stops.

    Attributes
    ----------
    objectives
        Raw log determinant before the first step and after every step taken.
    moved
        Rows relabeled by each step.
    outcome
        ``"fixed"`` when the proposal repeats the current labeling,
        ``"emptied"`` when it vacates a cell, and ``"max_iter"`` when the cap
        stopped it.
    """

    objectives: list[float]
    moved: list[int]
    outcome: str

    @property
    def worst_step(self) -> float:
        """Return the most negative objective change of the run, or zero."""
        if len(self.objectives) < 2:
            return 0.0
        return float(min(np.diff(np.asarray(self.objectives)), default=0.0))

    @property
    def went_downhill(self) -> bool:
        """Return whether any step strictly lowered the objective."""
        return self.worst_step < -1e-12


def unguarded_trajectory(
    scores: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int,
    max_iter: int = UNGUARDED_MAX_ITER,
) -> UnguardedRun:
    """Iterate `frozen_metric_proposal` with no guard whatsoever.

    The library never does this: it exists here only so the page can measure
    what the guard is protecting against. The iteration stops at a fixed point,
    at a proposal that vacates a cell, or at `max_iter`.

    Parameters
    ----------
    scores, weights, labels
        The weighted score table and its starting labeling.
    n_bins
        Number of cells.
    max_iter
        Maximum number of batch relabelings.

    Returns
    -------
    UnguardedRun
        The objective trajectory, the per-step relabeling counts, and why the
        iteration stopped.
    """
    current = np.asarray(labels).copy()
    objectives = [raw_log_determinant(scores, weights, current, n_bins=n_bins)]
    moved: list[int] = []
    for _ in range(max_iter):
        proposal, _, _ = frozen_metric_proposal(scores, weights, current, n_bins=n_bins)
        if len(np.unique(proposal)) < n_bins:
            return UnguardedRun(objectives, moved, "emptied")
        moved.append(int(np.sum(proposal != current)))
        objectives.append(raw_log_determinant(scores, weights, proposal, n_bins=n_bins))
        if np.array_equal(proposal, current):
            return UnguardedRun(objectives, moved, "fixed")
        current = proposal
    return UnguardedRun(objectives, moved, "max_iter")


@dataclass(frozen=True, slots=True)
class Counterexample:
    """The eight-row fixture, measured both ways.

    Attributes
    ----------
    before, after, step
        Raw log determinant before and after one frozen-metric batch step, and
        their difference.
    distortion_before, distortion_after
        The surrogate the step minimizes, before and after it.
    tangent_change
        First-order change of the concave log determinant at the current point.
        Concavity forces it to exceed `step`.
    moved
        Rows the step relabels.
    unguarded
        The unguarded iteration continued past that first step.
    rejected_objective, rescued_objective, rescued_moves
        What the guarded solver reports under ``guard="reject"`` and under
        ``guard="exchange"``, both in raw log determinant units.
    rescued_history
        Raw objective after every accepted step of the rescued run.
    whitening_offset
        ``log det`` of the unbinned information, the constant separating raw
        log determinants from the library's whitened `objective`.
    """

    before: float
    after: float
    step: float
    distortion_before: float
    distortion_after: float
    tangent_change: float
    moved: int
    unguarded: list[float]
    unguarded_outcome: str
    rejected_objective: float
    rejected_iterations: int
    rejected_accepted: int
    rejected_stable: bool
    rescued_objective: float
    rescued_moves: int
    rescued_history: list[float] = field(default_factory=list)
    whitening_offset: float = 0.0


def counterexample_study() -> Counterexample:
    """Measure the committed eight-row fixture under both guards.

    Returns
    -------
    Counterexample
        Every number the page's counterexample section prints, including the
        unguarded trajectory continued to its own fixed point.
    """
    scores = COUNTEREXAMPLE_SCORES
    labels = COUNTEREXAMPLE_LABELS
    n_bins = COUNTEREXAMPLE_BINS
    weights = np.full(scores.shape[0], 1.0 / scores.shape[0])

    proposal, means, metric = frozen_metric_proposal(scores, weights, labels, n_bins=n_bins)
    before = raw_log_determinant(scores, weights, labels, n_bins=n_bins)
    after = raw_log_determinant(scores, weights, proposal, n_bins=n_bins)
    proposed_information = np.asarray(
        sq.binned_fisher_information(scores, proposal, weights, n_bins=n_bins)
    )
    tangent = float(np.trace(metric @ proposed_information)) - scores.shape[1]

    unguarded = unguarded_trajectory(scores, weights, labels, n_bins=n_bins)
    offset = float(np.linalg.slogdet(np.asarray(sq.fisher_information(scores, weights)))[1])

    rejected = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=n_bins,
        config=sq.MahalanobisLloydConfig(seed=0, guard="reject"),
        initial_labels=labels,
    )
    rescued = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=n_bins,
        config=sq.MahalanobisLloydConfig(seed=0, guard="exchange"),
        initial_labels=labels,
    )
    return Counterexample(
        before=before,
        after=after,
        step=after - before,
        distortion_before=frozen_metric_distortion(scores, weights, labels, means, metric),
        distortion_after=frozen_metric_distortion(scores, weights, proposal, means, metric),
        tangent_change=tangent,
        moved=int(np.sum(proposal != labels)),
        unguarded=list(unguarded.objectives),
        unguarded_outcome=unguarded.outcome,
        rejected_objective=float(rejected.objective) + offset,
        rejected_iterations=int(rejected.lloyd_iterations),
        rejected_accepted=int(rejected.accepted_lloyd_steps),
        rejected_stable=bool(rejected.exchange_stable),
        rescued_objective=float(rescued.objective) + offset,
        rescued_moves=int(rescued.accepted_moves),
        rescued_history=[float(value) + offset for value in np.asarray(rescued.objective_history)],
        whitening_offset=offset,
    )


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """How the unguarded iteration fails on one problem, size, and budget.

    Attributes
    ----------
    problem, n_rows, n_bins
        The configuration swept.
    runs
        Random starting labelings tried.
    downhill_runs
        Runs in which some unguarded step strictly lowered the objective.
    emptied_runs
        Runs stopped because a proposal vacated a cell, a state the criterion
        cannot represent at all.
    fixed_point_runs
        Runs that reached a fixed point without either failure.
    worst_step
        Most negative single-step objective change over the whole group.
    median_steps
        Median number of batch relabelings before the iteration stopped.
    """

    problem: str
    n_rows: int
    n_bins: int
    runs: int
    downhill_runs: int
    emptied_runs: int
    fixed_point_runs: int
    worst_step: float
    median_steps: float


def guard_ledger(
    *,
    problems: tuple[str, ...] = LEDGER_PROBLEMS,
    sizes: tuple[int, ...] = LEDGER_SIZES,
    budgets: tuple[int, ...] = LEDGER_BUDGETS,
    starts: int = LEDGER_STARTS,
) -> list[LedgerRow]:
    """Count how the unguarded iteration fails, over many random starts.

    Parameters
    ----------
    problems
        Names of the synthetic problems to sweep.
    sizes
        Training sample sizes.
    budgets
        Bin budgets.
    starts
        Random starting labelings per configuration.

    Returns
    -------
    list of LedgerRow
        One row per configuration. Two failure modes are counted separately
        because they are different: a downhill step is the concavity failure of
        the surrogate, and an emptied cell is an infeasible proposal.
    """
    rows: list[LedgerRow] = []
    for name in problems:
        for n_rows in sizes:
            split = ledger_split(name, n_rows)
            for n_bins in budgets:
                runs = 0
                downhill = 0
                emptied = 0
                worst = 0.0
                lengths: list[int] = []
                for seed in range(starts):
                    start = np.random.default_rng(seed).integers(0, n_bins, size=n_rows)
                    if len(np.unique(start)) < n_bins:
                        continue
                    run = unguarded_trajectory(split.scores, split.weights, start, n_bins=n_bins)
                    runs += 1
                    lengths.append(len(run.moved))
                    downhill += run.went_downhill
                    emptied += run.outcome == "emptied"
                    worst = min(worst, run.worst_step)
                rows.append(
                    LedgerRow(
                        problem=name,
                        n_rows=n_rows,
                        n_bins=n_bins,
                        runs=runs,
                        downhill_runs=downhill,
                        emptied_runs=emptied,
                        fixed_point_runs=runs - emptied,
                        worst_step=worst,
                        median_steps=float(np.median(lengths)) if lengths else 0.0,
                    )
                )
    return rows


@dataclass(frozen=True, slots=True)
class GuardedClimb:
    """One large guarded run started far from any sensible geometry.

    Attributes
    ----------
    problem, n_rows, n_bins
        The configuration.
    start_objective, final_objective
        Whitened objective of the random starting labeling and of the terminal
        state.
    lloyd_iterations, accepted_lloyd_steps
        Proposals built and proposals accepted by the guarded batch phase.
    scans, accepted_moves
        Exchange work after the handoff.
    exchange_objective, exchange_scans, exchange_moves
        The same problem solved by plain exact exchange from its own seeding.
    monotone
        Whether every recorded step strictly increased the objective.
    history
        The whole accepted-step trajectory.
    """

    problem: str
    n_rows: int
    n_bins: int
    start_objective: float
    final_objective: float
    lloyd_iterations: int
    accepted_lloyd_steps: int
    scans: int
    accepted_moves: int
    exchange_objective: float
    exchange_scans: int
    exchange_moves: int
    monotone: bool
    history: list[float] = field(repr=False, default_factory=list)


def guarded_climb(
    *, n_rows: int = CLIMB_SIZE, n_bins: int = CLIMB_BINS, seed: int = CLIMB_SEED
) -> GuardedClimb:
    """Run the guarded solver from a random labeling on a large sample.

    Random starting labels are the regime the guarded batch is for: every
    iteration reconsiders every row at once, so it crosses a bad initialization
    in a few full-data passes where single-row exchange would need many scans.

    Parameters
    ----------
    n_rows
        Training rows drawn from the signal-plus-backgrounds problem.
    n_bins
        Bin budget.
    seed
        Seed of the random starting labeling and of both solvers.

    Returns
    -------
    GuardedClimb
        The acceptance counters, the trajectory, and the plain-exchange
        comparison.
    """
    split = ledger_split("signal_background_shape", n_rows)
    start = np.random.default_rng(seed).integers(0, n_bins, size=n_rows)
    guarded = sq.optimize_partition(
        split.scores,
        weights=split.weights,
        n_bins=n_bins,
        config=sq.MahalanobisLloydConfig(seed=seed),
        initial_labels=start,
    )
    exchange = sq.optimize_partition(
        split.scores,
        weights=split.weights,
        n_bins=n_bins,
        config=sq.DExchangeConfig(seed=seed),
        initial_labels=start,
    )
    history = np.asarray(guarded.objective_history)
    return GuardedClimb(
        problem="signal_background_shape",
        n_rows=n_rows,
        n_bins=n_bins,
        start_objective=float(history[0]),
        final_objective=float(guarded.objective),
        lloyd_iterations=int(guarded.lloyd_iterations),
        accepted_lloyd_steps=int(guarded.accepted_lloyd_steps),
        scans=int(guarded.scans),
        accepted_moves=int(guarded.accepted_moves),
        exchange_objective=float(exchange.objective),
        exchange_scans=int(exchange.scans),
        exchange_moves=int(exchange.accepted_moves),
        monotone=bool(np.all(np.diff(history) > 0)),
        history=[float(value) for value in history],
    )


@dataclass(frozen=True, slots=True)
class Study:
    """Everything the page and the figure need from one deterministic run."""

    metrics: dict[str, object]
    counterexample: Counterexample = field(repr=False)
    ledger: list[LedgerRow] = field(repr=False)
    climb: GuardedClimb = field(repr=False)


def run_study(
    *,
    sizes: tuple[int, ...] | None = None,
    budgets: tuple[int, ...] = LEDGER_BUDGETS,
    starts: int | None = None,
    climb_rows: int | None = None,
) -> Study:
    """Run the counterexample, the failure ledger, and the large guarded climb.

    Parameters
    ----------
    sizes
        Sample sizes of the ledger sweep. Defaults to the fast-mode aware sizes.
    budgets
        Bin budgets of the ledger sweep.
    starts
        Random starting labelings per ledger configuration.
    climb_rows
        Sample size of the large guarded run.

    Returns
    -------
    Study
        The exact structure written to
        ``examples/outputs/lloyd-nonmonotone.json``, plus the objects the
        figure draws.
    """
    resolved_sizes = example_scale(LEDGER_SIZES, (60, 250)) if sizes is None else sizes
    resolved_starts = example_scale(LEDGER_STARTS, 6) if starts is None else starts
    resolved_rows = example_scale(CLIMB_SIZE, 800) if climb_rows is None else climb_rows

    counterexample = counterexample_study()
    ledger = guard_ledger(sizes=resolved_sizes, budgets=budgets, starts=resolved_starts)
    climb = guarded_climb(n_rows=resolved_rows)

    metrics: dict[str, object] = {
        "counterexample": asdict(counterexample),
        "ledger": [asdict(row) for row in ledger],
        "ledger_totals": {
            "runs": sum(row.runs for row in ledger),
            "downhill_runs": sum(row.downhill_runs for row in ledger),
            "emptied_runs": sum(row.emptied_runs for row in ledger),
            "worst_step": min((row.worst_step for row in ledger), default=0.0),
        },
        "climb": asdict(climb),
    }
    return Study(metrics=metrics, counterexample=counterexample, ledger=ledger, climb=climb)


def make_figure(study: Study) -> Figure:
    """Render the three-panel guarded-Lloyd dashboard.

    Parameters
    ----------
    study
        The object returned by `run_study`.

    Returns
    -------
    matplotlib.figure.Figure
        The counterexample trajectories with the rejected step marked, the
        large guarded climb with its handoff, and the failure-mode ledger.
    """
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)
    guarded_color = "#38618c"
    unguarded_color = "#c0563c"

    case = study.counterexample
    unguarded = np.asarray(case.unguarded)
    rescued = np.asarray(case.rescued_history)
    axes[0].plot(
        np.arange(unguarded.shape[0]),
        unguarded,
        marker="s",
        linestyle="--",
        color=unguarded_color,
        label="unguarded batch iteration",
    )
    axes[0].plot(
        np.arange(rescued.shape[0]),
        rescued,
        marker="o",
        color=guarded_color,
        label="guarded solver",
    )
    axes[0].scatter([1], [case.after], s=140, facecolors="none", edgecolors=unguarded_color)
    axes[0].annotate(
        f"rejected\n{case.step:+.6f} nat",
        xy=(1.05, case.after),
        xytext=(1.5, case.after - 0.12),
        color=unguarded_color,
        arrowprops={"arrowstyle": "->", "color": unguarded_color},
    )
    axes[0].axhline(case.before, color="#999999", linewidth=0.8, linestyle=":")
    axes[0].set(
        ylim=(case.after - 0.55, case.unguarded[-1] + 0.35),
        xlabel="accepted step",
        ylabel="$\\log\\det$ of the binned information",
        title="Eight rows: one batch step, downhill",
    )
    axes[0].legend(loc="upper left")

    climb = study.climb
    history = np.asarray(climb.history)
    handoff = 1 + climb.accepted_lloyd_steps
    shortfall = np.maximum(climb.final_objective - history, 1e-12)
    axes[1].plot(
        np.arange(handoff),
        shortfall[:handoff],
        marker="o",
        markersize=3,
        color=guarded_color,
        label=f"guarded batch, {climb.accepted_lloyd_steps} of "
        f"{climb.lloyd_iterations} proposals accepted",
    )
    axes[1].plot(
        np.arange(handoff - 1, history.shape[0] - 1),
        shortfall[handoff - 1 : -1],
        marker="^",
        markersize=3,
        color="#4f9d69",
        label=f"exact exchange, {climb.accepted_moves} relocations",
    )
    axes[1].axvline(handoff - 1, color="#999999", linewidth=0.8, linestyle=":")
    axes[1].set(
        yscale="log",
        xlabel="accepted step",
        ylabel="shortfall from the terminal objective (nat)",
        title=f"{climb.n_rows} events from random labels",
    )
    axes[1].legend(loc="upper right", fontsize=8)

    markers = {4: "o", 6: "s"}
    colors = {
        "signal_background_shape": unguarded_color,
        "spectral_templates": guarded_color,
        "spatial_sources": "#4f9d69",
    }
    for problem in dict.fromkeys(row.problem for row in study.ledger):
        for n_bins in dict.fromkeys(row.n_bins for row in study.ledger):
            group = [row for row in study.ledger if row.problem == problem and row.n_bins == n_bins]
            if not group:
                continue
            axes[2].plot(
                [row.n_rows for row in group],
                [row.emptied_runs / max(row.runs, 1) for row in group],
                marker=markers.get(n_bins, "^"),
                color=colors.get(problem, "#666666"),
                linestyle="-" if n_bins == 4 else "--",
                label=f"{problem.split('_')[0]}, {n_bins} cells",
            )
    totals = study.metrics["ledger_totals"]
    if isinstance(totals, dict):
        axes[2].text(
            0.03,
            0.06,
            f"{totals['downhill_runs']} of {totals['runs']} runs stepped downhill",
            transform=axes[2].transAxes,
            fontsize=8,
            color="#444444",
        )
    axes[2].set(
        xscale="log",
        ylim=(-0.05, 1.12),
        xlabel="events in the sample",
        ylabel="unguarded runs that vacated a cell",
        title="The other failure mode",
    )
    axes[2].legend(loc="upper right", fontsize=7)

    figure.suptitle("An unguarded batch step is not an improvement")
    return figure


def main() -> None:
    """Run the study, then write the committed JSON and figure."""
    jax.config.update("jax_enable_x64", True)

    study = run_study()
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as stream:
        json.dump(study.metrics, stream, indent=2)
        stream.write("\n")
    figure = make_figure(study)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
