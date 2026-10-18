"""Exchange-stability certificate and terminal geometry diagnostics of finite partitions.

``exchange_stability_report`` runs one complete scan of the exchange engine and
nothing else, so a labeling of any origin can be checked before it is trusted.
The geometry reports measure how far a terminal state is from the
nearest-centroid rule of its own metric. Every row-wise quantity is evaluated
in memory-bounded row chunks, so no ``[N, B, P]`` tensor is materialized.
"""

from __future__ import annotations

import numpy as np

from ._chunking import chunk_rows_for
from ._errors import ContractError
from ._exchange import (
    DObjective,
    ExchangeObjective,
    ExchangeState,
    cell_statistics,
    chunk_gains,
    profiled_objective,
    profiled_semimetric,
)
from ._execution import canonicalize_public, execution_scope
from ._execution import xp as jnp
from ._typing import ArrayLike
from ._validation import _ValidatedSample, integer_labels, validate_sample
from .config import DExchangeConfig, ExecutionConfig, dtype_tolerance, resolve_gain_tolerance
from .criteria import DOptimality, ProfiledDOptimality
from .reports import GeometryReport, ProfiledGeometryReport, StabilityReport
from .solvers.exchange import scan
from .transforms import fisher_transform

# Relative slack allowed when certifying the leverage bound q_delta <= 1/W_a +
# 1/W_b. The bound is exact mathematics, so this only absorbs the rounding of
# one inverse and two quadratic forms; it never hides a real violation. This is
# the float64 value; other dtypes raise it to their noise floor.
_SEPARATION_RTOL = 1e-8
_SEPARATION_EPS_MULTIPLE = 100


@execution_scope
def exchange_stability_report(
    scores: ArrayLike,
    labels: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    criterion: DOptimality | ProfiledDOptimality | None = None,
    rank_rtol: float | None = None,
    gain_tolerance: float | None = None,
    execution: ExecutionConfig | None = None,
) -> StabilityReport:
    """Certify one supplied labeling against every single-row relocation.

    This runs exactly one complete exact scan of the same engine
    ``optimize_partition`` uses, on labels of any origin: a guarded
    Mahalanobis-Lloyd run stopped by ``guard="reject"``, an external tool, or a
    hand edit. Nothing is optimized and nothing is repaired; the report states
    whether an improving relocation exists and, when it does, names it.

    Rows are scanned as supplied. Identical score rows are deliberately not
    merged, because relocating one row of a repeated score atom is an ordinary
    single-point move that the certificate must consider.

    Parameters
    ----------
    scores
        Finite score matrix with shape ``[N, P]``.
    labels
        Integer labeling with shape ``[N]``. Cell count is taken from the
        labels, and every cell below it must hold positive weight. Labels of
        zero-weight rows carry no measure but still count toward the declared
        cell range.
    weights
        Optional finite, nonnegative weights with shape ``[N]``.
    criterion
        ``DOptimality`` by default; ``ProfiledDOptimality`` certifies the
        same-label profiled objective instead.
    rank_rtol
        Relative threshold of the informative Fisher subspace, matching the
        solver configuration that produced the labels.
    gain_tolerance
        Strict minimum gain that counts as an improvement. Match it to the
        ``gain_tolerance`` of the solver configuration that produced the
        labels: a labeling optimized at one tolerance is not certified at a
        smaller one, and the returned report records the tolerance it holds at.
        ``None`` (default) uses the solvers' dtype-aware default.

    Returns
    -------
    StabilityReport
        Stability verdict at ``gain_tolerance``, exact objective, best remaining
        gain, and the improving move in original row indexing when one exists.
    """
    del execution
    resolved = DOptimality() if criterion is None else criterion
    if not isinstance(resolved, (DOptimality, ProfiledDOptimality)):
        raise TypeError("exchange stability supports DOptimality or ProfiledDOptimality")
    sample = validate_sample(scores, weights)
    n_bins, effective_labels = _validate_labeling(labels, sample)
    points, objective = _stability_objective(sample, resolved, rank_rtol=rank_rtol)
    gain_tolerance = resolve_gain_tolerance(gain_tolerance, points.dtype)
    weights_array = sample.effective_weights
    state = objective.init_state(
        cell_statistics(points, weights_array, jnp.asarray(effective_labels), n_bins)
    )
    outcome = scan(
        points,
        weights_array,
        effective_labels,
        state,
        objective,
        DExchangeConfig(rank_rtol=rank_rtol, gain_tolerance=gain_tolerance),
        rows=False,
    )
    best_gain = outcome.best_gain
    stable = best_gain <= gain_tolerance
    best_move = None
    if not stable and outcome.best is not None:
        rows = np.flatnonzero(np.asarray(sample.positive_weight_mask))
        best_move = (int(rows[outcome.best.row]), int(outcome.best.destination))
    return canonicalize_public(
        StabilityReport(
            stable=stable,
            best_gain=best_gain,
            best_move=best_move,
            objective=state.objective,
            n_bins=n_bins,
            criterion=resolved,
            gain_tolerance=gain_tolerance,
        )
    )


def _validate_labeling(labels: ArrayLike, sample: _ValidatedSample) -> tuple[int, np.ndarray]:
    """Return the declared cell count and the positive-weight part of a labeling."""
    array = integer_labels(labels, int(sample.scores.shape[0]), "labels")
    declared = np.asarray(array, dtype=np.int64)
    if int(declared.min()) < 0:
        raise ContractError("labels must be nonnegative bin indices")
    n_bins = int(declared.max()) + 1
    effective = declared[np.asarray(sample.positive_weight_mask)]
    if int(np.bincount(effective, minlength=n_bins)[:n_bins].min()) == 0:
        raise ContractError(
            f"labels declare {n_bins} cells but at least one of them holds no "
            "positive-weight row; every declared cell must carry measure"
        )
    return n_bins, effective


def _stability_objective(
    sample: _ValidatedSample,
    criterion: DOptimality | ProfiledDOptimality,
    *,
    rank_rtol: float | None,
) -> tuple[jnp.ndarray, ExchangeObjective]:
    """Build the scan coordinates and criterion algebra of a supplied labeling.

    The D scan runs in Fisher-whitened coordinates and the profiled scan runs in
    the declared interest/nuisance parameterization, exactly as the two solvers
    do, so the reported objective matches ``PartitionResult.objective``.
    """
    scores = sample.effective_scores
    information = jnp.einsum("n,np,nq->pq", sample.effective_weights, scores, scores)
    transform = fisher_transform(information, whiten=True, rank_rtol=rank_rtol)
    if isinstance(criterion, DOptimality):
        return transform.apply(scores), DObjective()
    objective = profiled_objective(criterion.interest_indices, int(scores.shape[1]), transform.rank)
    return scores, objective


def voronoi_disagreement_gain(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: jnp.ndarray,
    geometric: np.ndarray,
    state: ExchangeState,
    objective: ExchangeObjective,
) -> float:
    """Return the largest exact gain over rows the terminal Voronoi rule relabels.

    This is the compile bridge measured in the units the solver optimizes. A
    terminal labeling and the nearest-centroid rule of its own metric may
    disagree on rows within the solver's stopping tolerance of a cell boundary;
    each admissible disagreement is tolerated when its individual relocation,
    priced against the original partition, is worth no more than that tolerance.
    This says nothing about simultaneous relabeling
    (CE-D-COMPILE-BATCH-TOLERANCE-001). Comparing the two labelings for equality
    instead verifies at tolerance zero and rejects such a state, which is what
    made a converged 10^6-row fit unusable.

    A row whose cell holds no other weight admits no relocation at all, so no
    tolerance can excuse a rule that sends it elsewhere; that case returns
    ``inf`` and is always rejected.
    """
    rows = np.flatnonzero(np.asarray(geometric) != np.asarray(labels))
    if rows.size == 0:
        return float("-inf")
    index = jnp.asarray(rows)
    row_weights = weights[index]
    sources = jnp.asarray(np.asarray(labels)[rows])
    if bool(np.asarray(jnp.any(state.weights[sources] <= row_weights))):
        return float("inf")
    gains = np.asarray(objective.chunk_gains(state, points[index], row_weights, sources))
    return float(np.max(gains[np.arange(rows.size), np.asarray(geometric)[rows]]))


def geometry_report(
    coordinates: jnp.ndarray,
    weights: jnp.ndarray,
    labels: jnp.ndarray,
    state: ExchangeState,
    *,
    gain_tolerance: float,
) -> GeometryReport:
    """Measure the self-consistent Voronoi geometry of a terminal D state.

    Distances are evaluated once per row chunk against every cell, so the same
    ``[chunk, B]`` table supplies the own-cell distance, the nearest competing
    distance, and the Voronoi-violating move set. The cell-separation lemma
    needs only the ``[B, B]`` table of centroid distances.

    The verdict is taken on the exact gain of the violating moves rather than on
    the sign of a distance gap: a distance gap is not the quantity the solver
    stops on, and comparing it to zero verifies at a tolerance no finite
    optimizer claims. ``gain_tolerance`` is the solver's own threshold, and the
    same per-row gain kernel the scan uses supplies the gains.
    """
    n_bins = int(state.weights.shape[0])
    n_rows = int(coordinates.shape[0])
    destinations = np.arange(n_bins)
    cell_weights = np.asarray(state.weights, dtype=np.float64)

    differences = state.means[:, None, :] - state.means[None, :, :]
    separations = np.asarray(
        jnp.einsum("abr,rs,abs->ab", differences, state.inverse, differences), dtype=np.float64
    )
    leverage = 1 / cell_weights[:, None] + 1 / cell_weights[None, :]
    pairs = np.triu(np.ones((n_bins, n_bins), dtype=bool), k=1)
    separation_residual = float(np.max(np.where(pairs, separations - leverage, -np.inf)))
    separation_scale = float(np.max(np.where(pairs, leverage, 0.0)))

    worst_violation = -np.inf
    guaranteed_gain = 0.0
    violation_gain = 0.0
    violating = 0
    evaluated = 0
    chunk_rows = chunk_rows_for(coordinates, n_bins)
    label_array = jnp.asarray(labels)
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        residuals = coordinates[start:stop, None, :] - state.means[None, :, :]
        distances = np.asarray(
            jnp.einsum("nbr,rs,nbs->nb", residuals, state.inverse, residuals), dtype=np.float64
        )
        sources = np.asarray(label_array[start:stop], dtype=np.int64)
        row_weights = np.asarray(weights[start:stop], dtype=np.float64)
        own = distances[np.arange(stop - start), sources]
        elsewhere = destinations[None, :] != sources[:, None]
        worst_violation = max(
            worst_violation, float(np.max(own - np.min(np.where(elsewhere, distances, np.inf), 1)))
        )

        source_weights = cell_weights[sources]
        shrinks = source_weights > row_weights
        admissible = shrinks[:, None] & elsewhere
        violates = admissible & (own[:, None] >= distances)
        alpha = row_weights * source_weights / np.where(shrinks, source_weights - row_weights, 1.0)
        beta = (
            row_weights[:, None]
            * cell_weights[None, :]
            / (cell_weights[None, :] + row_weights[:, None])
        )
        bounds = np.log1p(alpha[:, None] * beta * separations[sources] ** 2 / 4)
        guaranteed_gain = max(guaranteed_gain, float(np.max(np.where(violates, bounds, 0.0))))
        if violates.any():
            # A clean chunk contributes 0.0 either way, so the gain kernel runs
            # only where there is actually a violation to price.
            exact = np.asarray(
                chunk_gains(
                    coordinates[start:stop],
                    weights[start:stop],
                    label_array[start:stop],
                    state.weights,
                    state.means,
                    state.inverse,
                ),
                dtype=np.float64,
            )
            violation_gain = max(violation_gain, float(np.max(np.where(violates, exact, 0.0))))
        violating += int(np.count_nonzero(violates))
        evaluated += int(np.count_nonzero(admissible))

    return GeometryReport(
        maximum_voronoi_violation=worst_violation,
        guaranteed_violation_gain=guaranteed_gain,
        maximum_violation_gain=violation_gain,
        maximum_separation_residual=separation_residual,
        violating_moves=violating,
        evaluated_moves=evaluated,
        voronoi_consistent=violation_gain <= gain_tolerance,
        separation_certified=separation_residual
        <= dtype_tolerance(_SEPARATION_RTOL, coordinates.dtype, _SEPARATION_EPS_MULTIPLE)
        * separation_scale,
        gain_tolerance=gain_tolerance,
    )


def profiled_geometry_report(
    scores: jnp.ndarray,
    weights: jnp.ndarray,
    labels: jnp.ndarray,
    state: ExchangeState,
    *,
    nuisance: tuple[int, ...],
    exchange_stable: bool,
) -> ProfiledGeometryReport:
    """Diagnose the finite efficient-semimetric gap of a terminal profiled state.

    Each row chunk builds its own ``[chunk, B, P]`` destination residual; the
    reported maxima and counts are exact under any chunking.
    """
    metric = profiled_semimetric(state, nuisance)
    n_rows = int(scores.shape[0])
    n_bins = int(state.weights.shape[0])
    destinations = jnp.arange(n_bins)[None, :]
    violation = bound = residual = -np.inf
    violating = evaluated = 0
    chunk_rows = chunk_rows_for(scores, n_bins)
    for start in range(0, n_rows, chunk_rows):
        rows = slice(start, min(start + chunk_rows, n_rows))
        chunk_labels = labels[rows]
        chunk_weights = weights[rows]
        source_residuals = scores[rows] - state.means[chunk_labels]
        destination_residuals = scores[rows, None, :] - state.means[None, :, :]
        source_distances = jnp.einsum("nr,rs,ns->n", source_residuals, metric, source_residuals)
        destination_distances = jnp.einsum(
            "nbr,rs,nbs->nb", destination_residuals, metric, destination_residuals
        )
        violations = jnp.maximum(source_distances[:, None] - destination_distances, 0)
        q_source = jnp.einsum("nr,rs,ns->n", source_residuals, state.inverse, source_residuals)
        source_weights = state.weights[chunk_labels]
        bounds = (
            chunk_weights[:, None]
            * q_source[:, None]
            * (1 / source_weights[:, None] + 1 / state.weights[None, :])
        )
        admissible = (source_weights > chunk_weights)[:, None] & (
            destinations != chunk_labels[:, None]
        )
        violation = max(violation, float(np.asarray(jnp.max(jnp.where(admissible, violations, 0)))))
        bound = max(bound, float(np.asarray(jnp.max(jnp.where(admissible, bounds, 0)))))
        residual = max(
            residual,
            float(np.asarray(jnp.max(jnp.where(admissible, violations - bounds, -jnp.inf)))),
        )
        violating += int(np.asarray(jnp.sum(admissible & (violations > 0))))
        evaluated += int(np.asarray(jnp.sum(admissible)))
    return ProfiledGeometryReport(
        metric=metric,
        maximum_positive_violation=violation,
        maximum_theoretical_bound=bound,
        maximum_bound_residual=residual,
        violating_moves=violating,
        evaluated_moves=evaluated,
        bound_certified=exchange_stable,
    )
