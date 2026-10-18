"""Guarded batch Mahalanobis-Lloyd for both finite D criteria.

A batch iteration that freezes the criterion metric, relabels every row to its
nearest centroid in that metric, and recomputes the binned information is *not*
monotone: the tangent of the concave log determinant is an upper bound, not a
minorizer. Every batch proposal is therefore accepted only against the exactly
rebuilt objective, exactly as guarded batch exchange acceptance already is.
"""

from __future__ import annotations

import numpy as np

from scorequant._exchange import ExchangeObjective, ExchangeState, cell_statistics
from scorequant._execution import xp as jnp
from scorequant._predict import assign_nearest
from scorequant.config import DExchangeConfig, MahalanobisLloydConfig

from .exchange import ExchangeRun, gain_tolerance, kmeans_labels, run_exchange, scan, trial_state


def optimize_lloyd(
    *,
    points: jnp.ndarray,
    coordinates: jnp.ndarray,
    weights: jnp.ndarray,
    n_bins: int,
    objective: ExchangeObjective,
    config: MahalanobisLloydConfig,
    initial_labels: jnp.ndarray | None = None,
) -> ExchangeRun:
    """Iterate guarded nearest-centroid batches, then settle the exchange guard.

    Each iteration freezes the current criterion metric, relabels every row to
    its nearest centroid in that metric, and repairs any cell the proposal would
    empty. The proposal is adopted only when the objective of its exactly
    rebuilt state strictly improves, because the frozen-metric batch step is not
    monotone on its own. Iteration stops at the first non-improving or unchanged
    proposal, or at ``max_iter``; ``guard`` then decides whether the labels are
    handed to the exchange engine or merely certified by one final scan. Supplied
    labels replace the k-means seeding, so ``initializer_restarts`` then governs
    only the exchange handoff.
    """
    seeded = (
        kmeans_labels(
            coordinates,
            weights,
            n_bins,
            rank_rtol=config.rank_rtol,
            seed=config.seed,
            initializer_restarts=config.initializer_restarts,
        )
        if initial_labels is None
        else initial_labels
    )
    labels = np.asarray(seeded, dtype=np.int32).copy()
    state = objective.init_state(cell_statistics(points, weights, jnp.asarray(labels), n_bins))
    history = [state.objective]
    iterations = 0
    accepted = 0

    while iterations < config.max_iter:
        iterations += 1
        proposal = _lloyd_proposal(points, labels, state, objective, n_bins)
        if np.array_equal(proposal, labels):
            break
        trial = trial_state(points, weights, proposal, n_bins, objective)
        if trial is None or trial.objective <= state.objective + gain_tolerance(config):
            break
        state, labels = trial, proposal
        accepted += 1
        history.append(state.objective)

    return _settle(
        points,
        weights,
        labels,
        n_bins,
        state,
        objective,
        config,
        history=history,
        iterations=iterations,
        accepted=accepted,
    )


def _settle(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: np.ndarray,
    n_bins: int,
    state: ExchangeState,
    objective: ExchangeObjective,
    config: MahalanobisLloydConfig,
    *,
    history: list[float],
    iterations: int,
    accepted: int,
) -> ExchangeRun:
    """Hand the guarded batch result to the exchange engine or certify it once.

    A Lloyd fixed point need not be exchange-stable, so ``"exchange"``
    continues with exact positive-gain relocations from the same labels. The
    exchange rebuilds its own initial state from those labels, which reproduces
    the terminal batch objective exactly, so its first history entry is dropped
    rather than repeated.
    """
    exchange_config = DExchangeConfig(
        rank_rtol=config.rank_rtol,
        seed=config.seed,
        initializer_restarts=config.initializer_restarts,
        batch_moves=True,
        gain_tolerance=config.gain_tolerance,
    )
    if config.guard == "exchange":
        handoff = run_exchange(
            points, weights, jnp.asarray(labels), n_bins, objective, exchange_config
        )
        history.extend(handoff.objective_history[1:])
        return ExchangeRun(
            labels=handoff.labels,
            state=handoff.state,
            objective_history=tuple(history),
            accepted_moves=handoff.accepted_moves,
            scans=handoff.scans,
            exchange_stable=handoff.exchange_stable,
            best_remaining_gain=handoff.best_remaining_gain,
            lloyd_iterations=iterations,
            accepted_lloyd_steps=accepted,
        )
    best_remaining_gain = scan(
        points, weights, labels, state, objective, exchange_config, rows=False
    ).best_gain
    return ExchangeRun(
        labels=jnp.asarray(labels),
        state=state,
        objective_history=tuple(history),
        accepted_moves=0,
        scans=1,
        exchange_stable=best_remaining_gain <= gain_tolerance(config),
        best_remaining_gain=best_remaining_gain,
        lloyd_iterations=iterations,
        accepted_lloyd_steps=accepted,
    )


def _lloyd_proposal(
    points: jnp.ndarray,
    labels: np.ndarray,
    state: ExchangeState,
    objective: ExchangeObjective,
    n_bins: int,
) -> np.ndarray:
    """Propose the complete nearest-centroid relabeling of the current metric.

    A frozen-metric batch can vacate a cell entirely, which the exact criterion
    state cannot represent. The repair retains, for each emptied cell, the row
    it already holds that is closest to its own centroid, so the recovered cell
    is the one the criterion currently supports best. The retained rows are
    distinct because the current labels partition the sample; a repair that
    empties some other cell as collateral is rejected by the guard like any
    other infeasible proposal.
    """
    metric = objective.assignment_metric(state)
    proposal = np.array(assign_nearest(points, state.means, metric), dtype=np.int32)
    for cell in np.flatnonzero(np.bincount(proposal, minlength=n_bins)[:n_bins] == 0):
        members = np.flatnonzero(labels == cell)
        if members.size == 0:
            continue
        residuals = points[members] - state.means[cell]
        distances = np.asarray(jnp.einsum("nr,rs,ns->n", residuals, metric, residuals))
        proposal[members[int(np.argmin(distances))]] = int(cell)
    return proposal
