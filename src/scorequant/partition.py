"""Exact finite-sample D-optimal partition optimization.

This module prepares one fixed weighted score table (validation, duplicate
collapsing, Fisher whitening), routes it to the exchange engine or the guarded
Mahalanobis-Lloyd solver in ``solvers/``, extends the result to zero-weight
rows, and assembles the ``PartitionResult``. The criterion algebra is in
``_exchange.py`` and the stability and geometry diagnostics in
``diagnostics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ._binstats import scatter_bin_statistics
from ._errors import ContractError, RefusalError
from ._exchange import DObjective, ExchangeObjective, profiled_objective
from ._execution import current_execution, scatter_set
from ._execution import xp as jnp
from ._predict import assign_nearest
from ._typing import ArrayLike
from ._validation import (
    _ValidatedSample,
    collapse_duplicate_scores,
    integer_labels,
    resolve_collapse_duplicates,
    validate_n_bins,
    validate_sample,
)
from .config import MahalanobisLloydConfig, PartitionConfig, resolve_gain_tolerance
from .criteria import DOptimality, ProfiledDOptimality
from .diagnostics import geometry_report, profiled_geometry_report, voronoi_disagreement_gain
from .information import information_report, profiled_information_report
from .reports import GeometryReport, ProfiledGeometryReport, ProfiledInformationReport
from .result import PartitionResult
from .solvers.exchange import ExchangeRun, gain_tolerance, optimize_exchange
from .solvers.lloyd import optimize_lloyd
from .sources import ScoreProvenance, ScoreSchema
from .transforms import FisherTransform, fisher_transform


def optimize_d_partition(
    scores: ArrayLike,
    *,
    weights: ArrayLike | None,
    n_bins: int,
    config: PartitionConfig,
    provenance: ScoreProvenance,
    initial_labels: ArrayLike | None = None,
    schema: ScoreSchema | None = None,
) -> PartitionResult:
    """Optimize arbitrary labels of one fixed weighted score table."""
    prepared = _prepare_partition(scores, weights, n_bins=n_bins, config=config)
    config = _resolve_tolerance(config, prepared)
    _require_d_bin_budget(prepared, n_bins)
    run = _optimize_labels(
        points=prepared.coordinates,
        coordinates=prepared.coordinates,
        weights=prepared.weights,
        n_bins=n_bins,
        objective=DObjective(),
        config=config,
        initial_labels=_collapsed_initial_labels(prepared, initial_labels, n_bins),
    )
    state = run.state
    sample = prepared.sample

    # The D theorem supplies the only canonical labels for zero-measure rows.
    all_coordinates = prepared.transform.apply(sample.scores)
    compiled_labels = _int32_labels(assign_nearest(all_coordinates, state.means, state.inverse))
    effective_labels = run.labels[prepared.inverse_rows]
    compiled_labels = scatter_set(compiled_labels, sample.positive_weight_mask, effective_labels)
    if run.exchange_stable:
        geometric = np.asarray(assign_nearest(prepared.coordinates, state.means, state.inverse))
        disagreement = voronoi_disagreement_gain(
            prepared.coordinates,
            prepared.weights,
            run.labels,
            geometric,
            state,
            DObjective(),
        )
        if disagreement > gain_tolerance(config):
            raise RefusalError(
                "terminal D state is geometrically degenerate: relabeling a row by the "
                f"terminal Mahalanobis rule gains {disagreement:.6g}, above "
                f"gain_tolerance={gain_tolerance(config):g}; duplicate/tied score atoms "
                "must be merged or assigned consistently",
                "CE-D-UNMERGED-DUPLICATES-001",
            )

    return _partition_result(
        prepared,
        run,
        compiled_labels=compiled_labels,
        effective_labels=effective_labels,
        n_bins=n_bins,
        criterion=DOptimality(),
        config=config,
        provenance=provenance,
        transformed_centers=state.means,
        metric=state.inverse,
        geometry=geometry_report(
            prepared.coordinates,
            prepared.weights,
            run.labels,
            state,
            gain_tolerance=gain_tolerance(config),
        ),
        schema=schema,
    )


def _resolve_tolerance(config: PartitionConfig, prepared: _PreparedPartition) -> PartitionConfig:
    """Fix a default ``gain_tolerance`` to the working dtype before any solver reads it.

    The result records this resolved configuration, so its stability and
    geometry verdicts state the tolerance they actually hold at.
    """
    resolved = resolve_gain_tolerance(config.gain_tolerance, prepared.scores.dtype)
    return config if resolved == config.gain_tolerance else replace(config, gain_tolerance=resolved)


def _require_d_bin_budget(prepared: _PreparedPartition, n_bins: int) -> None:
    """Reject a cell budget that cannot make the binned D information regular."""
    if n_bins < prepared.transform.rank:
        raise ContractError(
            "D-optimality requires at least as many bins as informative directions; "
            "a normalized mean-zero score law generally requires one additional bin"
        )


def _int32_labels(labels: jnp.ndarray) -> jnp.ndarray:
    """Return assignment labels in the ``int32`` dtype finite partitions report."""
    return jnp.asarray(np.asarray(labels, dtype=np.int32))


def optimize_profiled_d_partition(
    scores: ArrayLike,
    *,
    weights: ArrayLike | None,
    n_bins: int,
    criterion: ProfiledDOptimality,
    config: PartitionConfig,
    provenance: ScoreProvenance,
    initial_labels: ArrayLike | None = None,
    schema: ScoreSchema | None = None,
) -> PartitionResult:
    """Optimize same-label profiled-D labels of one fixed score table."""
    prepared = _prepare_partition(scores, weights, n_bins=n_bins, config=config)
    config = _resolve_tolerance(config, prepared)
    dimension = int(prepared.scores.shape[1])
    objective = profiled_objective(criterion.interest_indices, dimension, prepared.transform.rank)
    # This bound is centering-agnostic and is the right one here. Binned
    # information has rank at most n_bins, so n_bins >= dimension is necessary
    # for any sample. An exactly centered sample needs one more bin than that,
    # because sum_b w_b m_b = 0 makes the bin means dependent and drops the
    # ceiling to n_bins - 1 (CE-DS-MARGINS-RANK-VACUITY-001, which is the
    # d_psi = 1 case of it). That extra bin is deliberately not required here:
    # scores away from the true reference point have a nonzero mean, and for
    # them n_bins == dimension is feasible and not vacuous. The centered case
    # is refused where it is detectable - at the singular initial state.
    if n_bins < dimension:
        raise ContractError("profiled D requires at least as many bins as score dimensions")
    run = _optimize_labels(
        points=prepared.scores,
        coordinates=prepared.coordinates,
        weights=prepared.weights,
        n_bins=n_bins,
        objective=objective,
        config=config,
        initial_labels=_collapsed_initial_labels(prepared, initial_labels, n_bins),
    )
    state = run.state
    sample = prepared.sample

    all_coordinates = prepared.transform.apply(sample.scores)
    extension_labels = _int32_labels(
        assign_nearest(all_coordinates, prepared.transform.apply(state.means))
    )
    effective_labels = run.labels[prepared.inverse_rows]
    compiled_labels = scatter_set(extension_labels, sample.positive_weight_mask, effective_labels)
    return _partition_result(
        prepared,
        run,
        compiled_labels=compiled_labels,
        effective_labels=effective_labels,
        n_bins=n_bins,
        criterion=criterion,
        config=config,
        provenance=provenance,
        profiled_report=profiled_information_report(
            sample.scores,
            compiled_labels,
            interest=criterion.interest_indices,
            weights=sample.weights,
            n_bins=n_bins,
            schema=schema,
        ),
        profiled_geometry=profiled_geometry_report(
            prepared.scores,
            prepared.weights,
            run.labels,
            state,
            nuisance=objective.nuisance,
            exchange_stable=run.exchange_stable,
        ),
        schema=schema,
    )


def _partition_result(
    prepared: _PreparedPartition,
    run: ExchangeRun,
    *,
    compiled_labels: jnp.ndarray,
    effective_labels: jnp.ndarray,
    n_bins: int,
    criterion: DOptimality | ProfiledDOptimality,
    config: PartitionConfig,
    provenance: ScoreProvenance,
    transformed_centers: jnp.ndarray | None = None,
    metric: jnp.ndarray | None = None,
    geometry: GeometryReport | None = None,
    profiled_report: ProfiledInformationReport | None = None,
    profiled_geometry: ProfiledGeometryReport | None = None,
    schema: ScoreSchema | None = None,
) -> PartitionResult:
    """Assemble the criterion-independent part of one finite partition result."""
    sample = prepared.sample
    raw = scatter_bin_statistics(
        effective_labels, sample.effective_weights, sample.effective_scores, n_bins
    )
    raw_weights, raw_sums, raw_means = raw.weights, raw.sums, raw.means
    return PartitionResult(
        labels=compiled_labels,
        training_scores=sample.scores,
        cell_weights=raw_weights,
        cell_score_sums=raw_sums,
        cell_score_means=raw_means,
        information_full=prepared.full_information,
        information_partitioned=jnp.einsum("b,bp,bq->pq", raw_weights, raw_means, raw_means),
        objective=run.state.objective,
        transform=prepared.transform,
        transformed_centers=transformed_centers,
        metric=metric,
        criterion=criterion,
        config=config,
        execution=current_execution(),
        train_report=information_report(
            sample.scores,
            compiled_labels,
            sample.weights,
            n_bins=n_bins,
            rank_rtol=config.rank_rtol,
        ),
        provenance=provenance,
        accepted_moves=run.accepted_moves,
        scans=run.scans,
        lloyd_iterations=run.lloyd_iterations,
        accepted_lloyd_steps=run.accepted_lloyd_steps,
        exchange_stable=run.exchange_stable,
        best_remaining_gain=run.best_remaining_gain,
        objective_history=jnp.asarray(run.objective_history),
        positive_weight_mask=sample.positive_weight_mask,
        geometry=geometry,
        profiled_report=profiled_report,
        profiled_geometry=profiled_geometry,
        schema=schema,
    )


@dataclass(frozen=True, slots=True)
class _PreparedPartition:
    """Validated, duplicate-collapsed inputs shared by both finite criteria."""

    sample: _ValidatedSample
    scores: jnp.ndarray
    weights: jnp.ndarray
    inverse_rows: jnp.ndarray
    full_information: jnp.ndarray
    transform: FisherTransform
    coordinates: jnp.ndarray


def _prepare_partition(
    scores: ArrayLike,
    weights: ArrayLike | None,
    *,
    n_bins: int,
    config: PartitionConfig,
) -> _PreparedPartition:
    sample = validate_sample(scores, weights)
    validate_n_bins(n_bins, sample.n_effective)
    effective_scores, effective_weights = sample.effective_scores, sample.effective_weights
    # Skipping the collapse keeps an identity row mapping, so expanding atom
    # results back to input rows is the same code either way; only the atom
    # count, and with it the search space, changes.
    inverse_rows = jnp.arange(effective_scores.shape[0])
    if resolve_collapse_duplicates(config.collapse_duplicates, int(effective_scores.shape[0])):
        effective_scores, effective_weights, inverse_rows = collapse_duplicate_scores(
            effective_scores, effective_weights
        )
    if n_bins > effective_scores.shape[0]:
        raise ContractError("n_bins exceeds distinct positive-weight score rows")
    full_information = jnp.einsum(
        "n,np,nq->pq", effective_weights, effective_scores, effective_scores
    )
    transform = fisher_transform(full_information, whiten=True, rank_rtol=config.rank_rtol)
    return _PreparedPartition(
        sample=sample,
        scores=effective_scores,
        weights=effective_weights,
        inverse_rows=inverse_rows,
        full_information=full_information,
        transform=transform,
        coordinates=transform.apply(effective_scores),
    )


def _collapsed_initial_labels(
    prepared: _PreparedPartition, initial_labels: ArrayLike | None, n_bins: int
) -> jnp.ndarray | None:
    """Reduce a caller-supplied ``[N]`` labeling to one label per distinct score atom.

    Zero-weight rows carry no measure and are dropped with the rest of the
    zero-measure sample, and identical score rows are collapsed before the
    solver runs, so they must already agree on their bin.
    """
    if initial_labels is None:
        return None
    labels = integer_labels(initial_labels, int(prepared.sample.scores.shape[0]), "initial_labels")
    effective = np.asarray(labels[prepared.sample.positive_weight_mask], dtype=np.int64)
    if effective.size and (effective.min() < 0 or effective.max() >= n_bins):
        raise ContractError("initial_labels contain a label outside [0, n_bins)")
    inverse = np.asarray(prepared.inverse_rows)
    collapsed = np.zeros(int(prepared.scores.shape[0]), dtype=np.int32)
    collapsed[inverse] = effective
    if not np.array_equal(collapsed[inverse], effective):
        raise ContractError("initial_labels must assign identical score rows to the same bin")
    if int(np.bincount(collapsed, minlength=n_bins)[:n_bins].min()) == 0:
        raise ContractError(
            "initial_labels must leave every one of the n_bins cells nonempty once "
            "zero-weight rows are dropped and identical score rows are merged"
        )
    return jnp.asarray(collapsed)


def _optimize_labels(
    *,
    points: jnp.ndarray,
    coordinates: jnp.ndarray,
    weights: jnp.ndarray,
    n_bins: int,
    objective: ExchangeObjective,
    config: PartitionConfig,
    initial_labels: jnp.ndarray | None = None,
) -> ExchangeRun:
    """Route one prepared finite problem to its configured solver."""
    if isinstance(config, MahalanobisLloydConfig):
        return optimize_lloyd(
            points=points,
            coordinates=coordinates,
            weights=weights,
            n_bins=n_bins,
            objective=objective,
            config=config,
            initial_labels=initial_labels,
        )
    return optimize_exchange(
        points=points,
        coordinates=coordinates,
        weights=weights,
        n_bins=n_bins,
        objective=objective,
        config=config,
        initial_labels=initial_labels,
    )
