"""Analytic-gradient soft-Voronoi optimization shared by all backends.

Every pass over the rows (soft moments, the centre gradient, checkpoints) runs
in memory-bounded row chunks through compiled per-chunk kernels, so the
``[N, B, R]`` residual tensor is never materialized beyond one chunk.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from scorequant._chunking import chunk_rows_for
from scorequant._errors import ContractError, RefusalError
from scorequant._execution import (
    AdamState,
    adam_update,
    backend_jit,
    create_adam,
    scatter_block_add,
)
from scorequant._execution import xp as jnp
from scorequant._predict import assign_nearest
from scorequant._validation import nuisance_complement
from scorequant.config import KMeansConfig, SoftVoronoiConfig
from scorequant.criteria import DOptimality, ProfiledDOptimality
from scorequant.transforms import fisher_transform, is_numerically_singular

from .common import QuantizerRun, _bin_weights, squared_distances
from .kmeans import weighted_kmeans


def _normalized_objective_scores(
    points: jnp.ndarray, weights: jnp.ndarray, rank_rtol: float | None
) -> jnp.ndarray:
    fisher = jnp.einsum("n,np,nq->pq", weights, points, points)
    return points @ fisher_transform(fisher, whiten=True, rank_rtol=rank_rtol).matrix


def soft_responsibilities(
    points: jnp.ndarray, centers: jnp.ndarray, temperature: float | jnp.ndarray
) -> jnp.ndarray:
    """Return stable soft nearest-center responsibilities."""
    logits = -squared_distances(points, centers) / (2 * jnp.asarray(temperature) ** 2)
    logits = logits - jnp.max(logits, axis=1, keepdims=True)
    exponentials = jnp.exp(logits)
    return exponentials / jnp.sum(exponentials, axis=1, keepdims=True)


type _ChunkKernel = Callable[..., tuple[jnp.ndarray, ...]]


def _chunk_totals(
    kernel: _ChunkKernel,
    rows: tuple[jnp.ndarray, ...],
    *shared: jnp.ndarray | float,
) -> tuple[jnp.ndarray, ...]:
    """Sum a per-chunk kernel over memory-bounded slices of the row arrays.

    ``rows[0]`` is the ``[N, R]`` point table and ``shared[0]`` the ``[B, R]``
    centres, which together size the chunk. One chunk calls the kernel on the
    full arrays, so a sample that fits is computed exactly as before chunking;
    more chunks change only the order of the final row sums.
    """
    n_rows = int(rows[0].shape[0])
    chunk_rows = chunk_rows_for(rows[0], int(np.shape(shared[0])[0]))
    if chunk_rows >= n_rows:
        return kernel(*rows, *shared)
    totals = kernel(*(array[:chunk_rows] for array in rows), *shared)
    for start in range(chunk_rows, n_rows, chunk_rows):
        parts = kernel(*(array[start : start + chunk_rows] for array in rows), *shared)
        totals = tuple(total + part for total, part in zip(totals, parts, strict=True))
    return totals


@backend_jit
def _soft_moment_chunk(
    points: jnp.ndarray,
    objective_scores: jnp.ndarray,
    weights: jnp.ndarray,
    centers: jnp.ndarray,
    temperature: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return one chunk's soft cell occupancies ``[B]`` and score moments ``[B, P]``."""
    weighted = weights[:, None] * soft_responsibilities(points, centers, temperature)
    return jnp.sum(weighted, axis=0), weighted.T @ objective_scores


def _soft_fisher(occupancies: jnp.ndarray, sums: jnp.ndarray) -> jnp.ndarray:
    means = sums / jnp.maximum(occupancies[:, None], jnp.finfo(sums.dtype).tiny)
    fisher = jnp.einsum("b,bp,bq->pq", occupancies, means, means)
    return 0.5 * (fisher + fisher.T)


def _soft_initial_centers(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    n_bins: int,
    config: SoftVoronoiConfig,
) -> jnp.ndarray:
    initializer_config = KMeansConfig(
        whiten=config.whiten,
        rank_rtol=config.rank_rtol,
        seed=config.seed,
        solver_restarts=config.initializer_restarts,
        max_iter=config.kmeans_max_iter,
        tolerance=config.kmeans_tolerance,
        record_every=config.kmeans_max_iter,
    )
    return weighted_kmeans(points, weights, n_bins, initializer_config).centers


def _soft_temperature_bounds(
    centers: jnp.ndarray,
    end_ratio: float,
) -> tuple[float, float]:
    n_bins = centers.shape[0]
    center_distances = squared_distances(centers, centers)
    center_distances = jnp.where(jnp.eye(n_bins, dtype=bool), jnp.inf, center_distances)
    nearest_separations = jnp.sqrt(jnp.min(center_distances, axis=1))
    start = float(np.asarray(jnp.median(nearest_separations)))
    if not np.isfinite(start) or start <= 0:
        raise ContractError("soft optimization requires distinct initial centers")
    return start, start * end_ratio


def criterion_objective_label(criterion: DOptimality | ProfiledDOptimality) -> str:
    """Name the units of a log-determinant objective recorded for one criterion."""
    return "profiled_logdet" if isinstance(criterion, ProfiledDOptimality) else "logdet_retained"


def _criterion_logdet(
    fisher: jnp.ndarray, criterion: DOptimality | ProfiledDOptimality
) -> tuple[bool, jnp.ndarray, int]:
    """Return whether the criterion is regular, its log determinant, and its dimension.

    Regularity is the library's rank rule, not a ``slogdet`` sign: on a
    deficient soft Fisher matrix the sign is rounding noise, and a positive one
    would hand the optimizer a finite but meaningless objective and gradient.
    The profiled criterion is regular exactly when the full matrix and its
    nuisance block are, the verdict the exact exchange reaches on hard cells.
    """
    if isinstance(criterion, DOptimality):
        _, value = jnp.linalg.slogdet(fisher)
        return not is_numerically_singular(fisher), value, fisher.shape[0]
    nuisance = nuisance_complement(criterion.interest_indices, int(fisher.shape[0]))
    interest_indices = jnp.asarray(criterion.interest)
    nuisance_indices = jnp.asarray(nuisance)
    interest_block = fisher[jnp.ix_(interest_indices, interest_indices)]
    cross_block = fisher[jnp.ix_(interest_indices, nuisance_indices)]
    nuisance_block = fisher[jnp.ix_(nuisance_indices, nuisance_indices)]
    if is_numerically_singular(nuisance_block, unit_diagonal=True):
        return False, jnp.asarray(-jnp.inf, dtype=fisher.dtype), len(criterion.interest_indices)
    schur = interest_block - cross_block @ jnp.linalg.solve(nuisance_block, cross_block.T)
    _, value = jnp.linalg.slogdet(0.5 * (schur + schur.T))
    regular = not is_numerically_singular(fisher, unit_diagonal=True)
    return regular, value, len(criterion.interest_indices)


def soft_objective_and_center_gradient(
    points: jnp.ndarray,
    objective_scores: jnp.ndarray,
    weights: jnp.ndarray,
    centers: jnp.ndarray,
    temperature: float | jnp.ndarray,
    criterion: DOptimality | ProfiledDOptimality,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Return the negative soft log determinant and its analytic center gradient.

    The derivative is shared by both execution backends. If ``G`` is the
    derivative of the negative criterion with respect to the soft Fisher
    matrix, differentiation first through each cell's weighted score moment
    and then through the softmax gives

    ``h_nb = r_nb (q_nb - sum_j r_nj q_nj)`` and
    ``grad(c_b) = sum_n h_nb (x_n - c_b) / temperature**2``.

    For ordinary D-optimality ``G = -F^-1``. For profiled D, the determinant
    identity ``logdet(S) = logdet(F) - logdet(C)`` adds the nuisance-block
    inverse back to that metric. No backend maintains a separate objective or
    asks autodiff to rediscover this equation at runtime. A soft Fisher matrix
    that is singular at the dtype's rank threshold has no finite objective: the loss is
    infinite and the gradient zero.

    The moments and the gradient are each one chunked pass over the rows; the
    rank verdict between them is the library's host-side rule, which is why
    the two passes are compiled separately rather than as one kernel.
    """
    occupancies, sums = _chunk_totals(
        _soft_moment_chunk, (points, objective_scores, weights), centers, temperature
    )
    safe = jnp.maximum(occupancies, jnp.finfo(weights.dtype).tiny)
    fisher = jnp.einsum("b,bp,bq->pq", 1 / safe, sums, sums)
    fisher = 0.5 * (fisher + fisher.T)
    regular, logdet, _ = _criterion_logdet(fisher, criterion)
    if not regular:
        return jnp.asarray(jnp.inf, dtype=points.dtype), jnp.zeros_like(centers)

    metric = -jnp.linalg.inv(fisher)
    if isinstance(criterion, ProfiledDOptimality):
        indices = jnp.asarray(nuisance_complement(criterion.interest_indices, int(fisher.shape[0])))
        nuisance_information = fisher[jnp.ix_(indices, indices)]
        metric = scatter_block_add(metric, indices, jnp.linalg.inv(nuisance_information))

    (gradient,) = _chunk_totals(
        _center_gradient_chunk,
        (points, objective_scores, weights),
        centers,
        temperature,
        metric,
        sums,
        safe,
    )
    return -logdet, gradient / (jnp.asarray(temperature) ** 2)


@backend_jit
def _center_gradient_chunk(
    points: jnp.ndarray,
    objective_scores: jnp.ndarray,
    weights: jnp.ndarray,
    centers: jnp.ndarray,
    temperature: float | jnp.ndarray,
    metric: jnp.ndarray,
    sums: jnp.ndarray,
    safe: jnp.ndarray,
) -> tuple[jnp.ndarray]:
    """Return one chunk's share of ``sum_n h_nb (x_n - c_b)``, before the temperature scale."""
    responsibilities = soft_responsibilities(points, centers, temperature)
    score_moment_cross = objective_scores @ metric @ sums.T
    moment_quadratic = jnp.einsum("bp,pq,bq->b", sums, metric, sums)
    response_gradient = weights[:, None] * (
        2 * score_moment_cross / safe[None, :] - moment_quadratic[None, :] / safe[None, :] ** 2
    )
    centered_response_gradient = responsibilities * (
        response_gradient - jnp.sum(responsibilities * response_gradient, axis=1, keepdims=True)
    )
    residuals = points[:, None, :] - centers[None, :, :]
    return (jnp.einsum("nb,nbr->br", centered_response_gradient, residuals),)


@dataclass(slots=True)
class _SoftHistory:
    points: jnp.ndarray
    weights: jnp.ndarray
    objective_scores: jnp.ndarray
    n_bins: int
    criterion: DOptimality | ProfiledDOptimality
    reference_objective: float
    objective_dimension: int
    steps: list[int] = field(default_factory=list)
    centers: list[jnp.ndarray] = field(default_factory=list)
    objectives: list[float] = field(default_factory=list)
    bin_weights: list[jnp.ndarray] = field(default_factory=list)
    retentions: list[float] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)

    def append(
        self,
        *,
        step: int,
        centers: jnp.ndarray,
        loss: jnp.ndarray,
        temperature: float,
        gradient_norm: jnp.ndarray,
    ) -> None:
        """Record one aggregate soft-optimization checkpoint."""
        soft_fisher = _soft_fisher(
            *_chunk_totals(
                _soft_moment_chunk,
                (self.points, self.objective_scores, self.weights),
                centers,
                temperature,
            )
        )
        labels = assign_nearest(self.points, centers)
        self.steps.append(step)
        self.centers.append(centers)
        self.objectives.append(float(np.asarray(-loss)))
        self.bin_weights.append(_bin_weights(labels, self.weights, self.n_bins))
        regular, value, _ = _criterion_logdet(soft_fisher, self.criterion)
        retention = (
            jnp.exp((value - self.reference_objective) / self.objective_dimension)
            if regular
            else 0.0
        )
        self.retentions.append(float(np.asarray(retention)))
        self.temperatures.append(float(temperature))
        self.gradient_norms.append(float(np.asarray(gradient_norm)))

    def finish(self, centers: jnp.ndarray) -> QuantizerRun:
        """Build the common quantizer result from recorded checkpoints."""
        return QuantizerRun(
            centers=centers,
            steps=self.steps,
            center_history=self.centers,
            objective_history=self.objectives,
            bin_weight_history=self.bin_weights,
            objective_label=criterion_objective_label(self.criterion),
            soft_retention_history=self.retentions,
            temperature_history=self.temperatures,
            gradient_norm_history=self.gradient_norms,
        )


def _soft_voronoi_single_cell(
    centers: jnp.ndarray, weights: jnp.ndarray, criterion: DOptimality | ProfiledDOptimality
) -> QuantizerRun:
    """Return the degenerate one-cell run: every row already shares the only center."""
    return QuantizerRun(
        centers=centers,
        steps=[0],
        center_history=[centers],
        objective_history=[0.0],
        bin_weight_history=[jnp.asarray([jnp.sum(weights)])],
        objective_label=criterion_objective_label(criterion),
        soft_retention_history=[1.0],
        temperature_history=[1.0],
        gradient_norm_history=[0.0],
    )


def _soft_voronoi_reference(
    objective_scores: jnp.ndarray,
    weights: jnp.ndarray,
    criterion: DOptimality | ProfiledDOptimality,
    *,
    effective_rank: int,
    rank_rtol: float | None,
) -> tuple[jnp.ndarray, float, int]:
    """Normalize D-optimal objective scores and check the full-data reference is regular."""
    if isinstance(criterion, DOptimality):
        objective_scores = _normalized_objective_scores(objective_scores, weights, rank_rtol)
    elif objective_scores.shape[1] != effective_rank:
        raise ContractError("profiled-D soft fitting requires full-rank supplied-score information")
    full_fisher = jnp.einsum("n,np,nq->pq", weights, objective_scores, objective_scores)
    reference_regular, reference_value, objective_dimension = _criterion_logdet(
        full_fisher, criterion
    )
    if not reference_regular:
        raise ContractError("soft fitting requires nonsingular criterion information")
    return objective_scores, float(np.asarray(reference_value)), objective_dimension


def _run_soft_schedule(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    objective_scores: jnp.ndarray,
    n_bins: int,
    criterion: DOptimality | ProfiledDOptimality,
    config: SoftVoronoiConfig,
    *,
    centers: jnp.ndarray,
    optimizer: AdamState,
    start_temperature: float,
    end_temperature: float,
    reference_objective: float,
    objective_dimension: int,
) -> QuantizerRun:
    """Run the annealed Adam schedule from prepared centers and return its history."""
    history = _SoftHistory(
        points,
        weights,
        objective_scores,
        n_bins,
        criterion,
        reference_objective,
        objective_dimension,
    )

    for step in range(config.max_iter + 1):
        fraction = step / config.max_iter
        temperature = start_temperature * (end_temperature / start_temperature) ** fraction
        loss, gradients = soft_objective_and_center_gradient(
            points, objective_scores, weights, centers, temperature, criterion
        )
        gradient_norm = jnp.asarray(0.0, dtype=points.dtype)
        if step > 0:
            centers, optimizer, gradient_norm_value = adam_update(centers, gradients, optimizer)
            centers = jnp.asarray(centers)
            gradient_norm = jnp.asarray(gradient_norm_value, dtype=points.dtype)
        if step % config.record_every == 0 or step == config.max_iter:
            history.append(
                step=step,
                centers=centers,
                loss=loss,
                temperature=temperature,
                gradient_norm=gradient_norm,
            )

    if not np.isfinite(float(np.asarray(loss))):
        # A singular soft Fisher matrix has no objective and a zero gradient, so
        # the schedule can only coast on momentum; ending there means the fit
        # stalled and its centres say nothing about the criterion. On an exactly
        # centred sample the soft moments sum to zero, so n_bins cells span at
        # most n_bins - 1 directions at every temperature.
        raise RefusalError(
            "soft optimization stalled on a singular soft Fisher matrix: the criterion is "
            "undefined at the end of the temperature schedule. n_bins cells of a centred "
            "score sample span at most n_bins - 1 directions; add a bin or use "
            "DExchangeConfig",
            "CE-DS-MARGINS-RANK-VACUITY-001",
        )
    return history.finish(centers)


def soft_voronoi(
    points: jnp.ndarray,
    objective_scores: jnp.ndarray,
    weights: jnp.ndarray,
    n_bins: int,
    effective_rank: int,
    criterion: DOptimality | ProfiledDOptimality,
    config: SoftVoronoiConfig,
) -> QuantizerRun:
    """Optimize soft Fisher retention, then return centers for hard assignment."""
    if n_bins < effective_rank:
        raise ContractError(
            "soft D-optimal fitting requires n_bins >= the effective Fisher rank; "
            "use k-means for smaller partitions"
        )
    centers = _soft_initial_centers(points, weights, n_bins, config)
    if n_bins == 1:
        return _soft_voronoi_single_cell(centers, weights, criterion)

    start_temperature, end_temperature = _soft_temperature_bounds(
        centers, config.temperature_end_ratio
    )
    objective_scores, reference_objective, objective_dimension = _soft_voronoi_reference(
        objective_scores,
        weights,
        criterion,
        effective_rank=effective_rank,
        rank_rtol=config.rank_rtol,
    )
    optimizer = create_adam(
        centers,
        learning_rate=config.learning_rate * start_temperature,
        gradient_clip=config.gradient_clip,
    )
    return _run_soft_schedule(
        points,
        weights,
        objective_scores,
        n_bins,
        criterion,
        config,
        centers=centers,
        optimizer=optimizer,
        start_temperature=start_temperature,
        end_temperature=end_temperature,
        reference_objective=reference_objective,
        objective_dimension=objective_dimension,
    )
