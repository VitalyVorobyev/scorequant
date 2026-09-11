"""Exact finite-sample D-optimal partition optimization.

One unified exchange engine drives both supported finite criteria. A *scan* is
one complete evaluation of every admissible single-row relocation; a scan
either accepts work or certifies exchange stability. With ``batch_moves`` a
single scan may relocate many rows at once, so accepted moves and scans are
different quantities.

The same engine also drives the guarded Mahalanobis-Lloyd solver. A batch
iteration that freezes the criterion metric, relabels every row to its nearest
centroid in that metric, and recomputes the binned information is *not*
monotone: the tangent of the concave log determinant is an upper bound, not a
minorizer. Every batch proposal is therefore accepted only against the exactly
rebuilt objective, exactly as guarded batch exchange acceptance already is.

The same scan certifies labels the engine did not produce:
``exchange_stability_report`` runs one scan and nothing else, so a labeling of
any origin can be checked before it is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ._binstats import scatter_bin_statistics
from ._chunking import assignment_chunk_rows
from ._errors import ContractError, RefusalError
from ._execution import (
    backend_jit,
    canonicalize_public,
    current_execution,
    execution_scope,
    random_permutation,
    scatter_add,
    scatter_block_add,
    scatter_set,
)
from ._execution import (
    xp as jnp,
)
from ._typing import ArrayLike
from ._validation import (
    _ValidatedSample,
    collapse_duplicate_scores,
    resolve_collapse_duplicates,
    validate_n_bins,
    validate_sample,
)
from .config import (
    DExchangeConfig,
    ExecutionConfig,
    KMeansConfig,
    MahalanobisLloydConfig,
    PartitionConfig,
)
from .criteria import DOptimality, ProfiledDOptimality
from .information import (
    PROFILED_RANK_ADVICE,
    _nuisance_information,
    binned_information_is_degenerate,
    information_report,
    profiled_information_report,
)
from .reports import (
    GeometryReport,
    ProfiledGeometryReport,
    ProfiledInformationReport,
    StabilityReport,
)
from .result import PartitionResult
from .solvers.common import hard_assign
from .solvers.kmeans import weighted_kmeans
from .sources import ScoreProvenance, ScoreSchema
from .transforms import FisherTransform, fisher_transform

# Relative slack allowed when certifying the leverage bound q_delta <= 1/W_a +
# 1/W_b. The bound is exact mathematics, so this only absorbs the rounding of
# one inverse and two quadratic forms; it never hides a real violation.
_SEPARATION_RTOL = 1e-8

# Strict positive-gain acceptance cannot revisit a labeling, so the exchange
# terminates without any scan cap. This bound only stops a run that a numerical
# pathology would otherwise keep alive; it is never reached by a healthy fit.
_SAFETY_SCAN_LIMIT = 100_000

# A guarded batch halves its size on rejection, so this bounds the retries of a
# single scan for any sample that fits in memory.
_MAX_BATCH_SHRINKS = 32

# Smallest improving set worth a guarded batch, and the largest share of a
# cell's weight that one batch may move in or out of it.
_MIN_BATCH_ROWS = 8
_BATCH_MASS_FRACTION = 0.25

# Largest tolerated max-norm residual of I @ I^-1 - identity after two chained
# Sherman-Morrison rank-one updates, dtype-dependent because the accumulated
# rounding of the chained updates scales with machine epsilon. Above this the
# incremental inverse is judged to have drifted too far from the exactly
# rebuilt one, and _rank_two_inverse_update falls back to a fresh inversion.
_RANK_TWO_RESIDUAL_TOLERANCE_F64 = 1e-8
_RANK_TWO_RESIDUAL_TOLERANCE_F32 = 2e-3


@dataclass(frozen=True, slots=True)
class _CellStatistics:
    """Weighted occupancy, score sums, and score means of every requested cell."""

    weights: jnp.ndarray
    sums: jnp.ndarray
    means: jnp.ndarray


@dataclass(frozen=True, slots=True)
class _ExchangeState:
    """Cell statistics and criterion matrices of one labeling."""

    cells: _CellStatistics
    information: jnp.ndarray
    inverse: jnp.ndarray
    objective: float
    nuisance_information: jnp.ndarray | None = None
    nuisance_inverse: jnp.ndarray | None = None

    @property
    def weights(self) -> jnp.ndarray:
        """Return the weighted occupancy of every cell."""
        return self.cells.weights

    @property
    def means(self) -> jnp.ndarray:
        """Return the weighted score mean of every cell."""
        return self.cells.means


@dataclass(frozen=True, slots=True)
class _Move:
    """One candidate relocation of a row into another cell."""

    row: int
    destination: int
    gain: float


@dataclass(frozen=True, slots=True)
class _Relocation:
    """Exact rank-two relocation data of one accepted move."""

    source: int
    destination: int
    point: jnp.ndarray
    point_weight: jnp.ndarray
    source_residual: jnp.ndarray
    destination_residual: jnp.ndarray
    alpha: jnp.ndarray
    beta: jnp.ndarray


@dataclass(frozen=True, slots=True)
class _Candidates:
    """Vectorized relocation geometry of one memory-bounded row chunk."""

    source_residuals: jnp.ndarray
    destination_residuals: jnp.ndarray
    alpha: jnp.ndarray
    beta: jnp.ndarray
    admissible: jnp.ndarray


@dataclass(frozen=True, slots=True)
class _UpdatedBlock:
    """One symmetric information block after an exact rank-two relocation."""

    information: jnp.ndarray
    inverse: jnp.ndarray
    sign: float
    logdet: float


@dataclass(frozen=True, slots=True)
class _ScanOutcome:
    """Result of one complete candidate scan."""

    best: _Move | None
    row_gains: np.ndarray | None
    row_destinations: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _ExchangeRun:
    """Terminal state and diagnostics of one exchange or guarded-batch restart."""

    labels: jnp.ndarray
    state: _ExchangeState
    objective_history: tuple[float, ...]
    accepted_moves: int
    scans: int
    exchange_stable: bool
    best_remaining_gain: float
    lloyd_iterations: int = 0
    accepted_lloyd_steps: int = 0


class _ExchangeObjective(Protocol):
    """Criterion-specific algebra shared by every exchange run."""

    def init_state(self, cells: _CellStatistics) -> _ExchangeState:
        """Build criterion matrices and the exact objective of one labeling."""
        ...

    def chunk_gains(
        self,
        state: _ExchangeState,
        points: jnp.ndarray,
        weights: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return exact objective gains of every candidate in one row chunk."""
        ...

    def apply_move(self, state: _ExchangeState, relocation: _Relocation) -> _ExchangeState:
        """Return the exactly refreshed state after one accepted relocation."""
        ...

    def assignment_metric(self, state: _ExchangeState) -> jnp.ndarray:
        """Return the quadratic form whose nearest-centroid rule a batch proposes."""
        ...


@dataclass(frozen=True, slots=True)
class _DObjective:
    """Log-determinant algebra of Fisher-whitened D-optimal exchange."""

    def init_state(self, cells: _CellStatistics) -> _ExchangeState:
        """Build the cell information matrix, its inverse, and its log determinant."""
        information, inverse, sign, logdet = _cell_information_matrices(cells.weights, cells.means)
        if float(np.asarray(sign)) <= 0:
            raise ContractError(
                "initial D partition is singular; increase n_bins or use a different "
                "reference sample"
            )
        return _ExchangeState(
            cells=cells,
            information=information,
            inverse=inverse,
            objective=float(np.asarray(logdet)),
        )

    def chunk_gains(
        self,
        state: _ExchangeState,
        points: jnp.ndarray,
        weights: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return the exact rank-two log-determinant gain of every candidate."""
        return _d_chunk_gains(points, weights, labels, state.weights, state.means, state.inverse)

    def apply_move(self, state: _ExchangeState, relocation: _Relocation) -> _ExchangeState:
        """Apply the exact rank-two information and inverse update of one move."""
        block = _rank_two_block(state.information, state.inverse, relocation)
        if block.sign <= 0:
            raise FloatingPointError("an accepted D move produced singular information")
        return _ExchangeState(
            cells=_relocate_cells(state.cells, relocation),
            information=block.information,
            inverse=block.inverse,
            objective=block.logdet,
        )

    def assignment_metric(self, state: _ExchangeState) -> jnp.ndarray:
        r"""Return \(I^{-1}\), the metric of the D nearest-centroid rule."""
        return state.inverse


@dataclass(frozen=True, slots=True)
class _ProfiledDObjective:
    r"""Same-label profiled-\(D_s\) algebra: full determinant minus its nuisance block.

    Parameters
    ----------
    interest
        Score-column indices of the parameters of interest.
    nuisance
        Complementary score-column indices profiled out of the same labels.
    """

    interest: tuple[int, ...]
    nuisance: tuple[int, ...]

    def init_state(self, cells: _CellStatistics) -> _ExchangeState:
        """Build full and nuisance information with the guarded block algebra."""
        information, inverse, _full_sign, full_logdet = _cell_information_matrices(
            cells.weights, cells.means
        )
        # The whole-matrix rank verdict comes first, before the nuisance block is
        # touched. Bin means satisfy sum_b w_b m_b = mu, the overall weighted
        # mean, so the information they generate has rank at most n_bins - and at
        # most n_bins - 1 when mu is zero, because the means are then linearly
        # dependent. A profiled value is positive only at full rank, so on an
        # exactly centered sample every labeling into n_bins <= dimension cells is
        # degenerate no matter how large the sample
        # (CE-DS-MARGINS-RANK-VACUITY-001). Such a state also tends to leave the
        # nuisance sub-block borderline, and _nuisance_information would then
        # blame the nuisance parameterization for what is a bin-budget fact -
        # a message that sends users looking for a data problem that is not there.
        if binned_information_is_degenerate(information):
            raise RefusalError(
                f"initial profiled-D partition is singular: {int(cells.means.shape[0])} "
                f"bins cannot generate nonsingular {int(information.shape[0])}-dimensional "
                f"binned information. {PROFILED_RANK_ADVICE}",
                "CE-DS-MARGINS-RANK-VACUITY-001",
            )
        # information._nuisance_information owns the nonsingular-nuisance guard
        # and the index bookkeeping. The Schur block is deliberately not built:
        # the exchange gains telescope with the difference of the two log
        # determinants, so a rebuild on this hot path never needs it.
        nuisance_information, nuisance_logdet, _ = _nuisance_information(information, self.interest)
        return _ExchangeState(
            cells=cells,
            information=information,
            inverse=inverse,
            objective=float(np.asarray(full_logdet - nuisance_logdet)),
            nuisance_information=nuisance_information,
            nuisance_inverse=jnp.linalg.inv(nuisance_information),
        )

    def chunk_gains(
        self,
        state: _ExchangeState,
        points: jnp.ndarray,
        weights: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return the difference of the full and nuisance determinant-lemma gains."""
        return _profiled_chunk_gains(
            points,
            weights,
            labels,
            state.weights,
            state.means,
            state.inverse,
            _require_nuisance(state.nuisance_inverse),
            self.nuisance,
        )

    def apply_move(self, state: _ExchangeState, relocation: _Relocation) -> _ExchangeState:
        """Apply exact rank-two updates to the full and the nuisance block."""
        full = _rank_two_block(state.information, state.inverse, relocation)
        nuisance = _rank_two_block(
            _require_nuisance(state.nuisance_information),
            _require_nuisance(state.nuisance_inverse),
            relocation,
            block=jnp.asarray(self.nuisance),
        )
        if full.sign <= 0 or nuisance.sign <= 0:
            raise FloatingPointError("an accepted profiled-D move produced singular information")
        return _ExchangeState(
            cells=_relocate_cells(state.cells, relocation),
            information=full.information,
            inverse=full.inverse,
            objective=full.logdet - nuisance.logdet,
            nuisance_information=nuisance.information,
            nuisance_inverse=nuisance.inverse,
        )

    def assignment_metric(self, state: _ExchangeState) -> jnp.ndarray:
        r"""Return the efficient semimetric \(G_s=L^\top S^{-1}L\) of the current state.

        Subtracting the nuisance inverse from the nuisance block of the full
        inverse yields exactly the profiled gradient metric, so the batch
        proposal ranks destinations by the same quadratic forms the exchange
        gains difference. The form is singular by construction: it measures
        only the efficient-score directions.
        """
        return _profiled_semimetric(state, self.nuisance)


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
    _require_d_bin_budget(prepared, n_bins)
    run = _optimize_labels(
        points=prepared.coordinates,
        coordinates=prepared.coordinates,
        weights=prepared.weights,
        n_bins=n_bins,
        objective=_DObjective(),
        config=config,
        initial_labels=_collapsed_initial_labels(prepared, initial_labels, n_bins),
    )
    state = run.state
    sample = prepared.sample

    # The D theorem supplies the only canonical labels for zero-measure rows.
    all_coordinates = prepared.transform.apply(sample.scores)
    compiled_labels = jnp.asarray(_assign_nearest(all_coordinates, state.means, state.inverse))
    effective_labels = run.labels[prepared.inverse_rows]
    compiled_labels = scatter_set(compiled_labels, sample.positive_weight_mask, effective_labels)
    if run.exchange_stable:
        geometric = _assign_nearest(prepared.coordinates, state.means, state.inverse)
        disagreement = _voronoi_disagreement_gain(
            prepared.coordinates,
            prepared.weights,
            run.labels,
            geometric,
            state,
            _DObjective(),
        )
        if disagreement > config.gain_tolerance:
            raise RefusalError(
                "terminal D state is geometrically degenerate: relabeling a row by the "
                f"terminal Mahalanobis rule gains {disagreement:.6g}, above "
                f"gain_tolerance={config.gain_tolerance:g}; duplicate/tied score atoms "
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
        geometry=_geometry_report(
            prepared.coordinates,
            prepared.weights,
            run.labels,
            state,
            gain_tolerance=config.gain_tolerance,
        ),
        schema=schema,
    )


def _require_d_bin_budget(prepared: _PreparedPartition, n_bins: int) -> None:
    """Reject a cell budget that cannot make the binned D information regular."""
    if n_bins < prepared.transform.rank:
        raise ContractError(
            "D-optimality requires at least as many bins as informative directions; "
            "a normalized mean-zero score law generally requires one additional bin"
        )


def _voronoi_disagreement_gain(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: jnp.ndarray,
    geometric: np.ndarray,
    state: _ExchangeState,
    objective: _ExchangeObjective,
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
    dimension = prepared.scores.shape[1]
    if any(index >= dimension for index in criterion.interest_indices):
        raise ContractError(f"interest indices must be smaller than score dimension {dimension}")
    interest_set = set(criterion.interest_indices)
    nuisance = tuple(index for index in range(dimension) if index not in interest_set)
    if not nuisance:
        raise ContractError("profiled D requires a nuisance block; use DOptimality")
    if prepared.transform.rank != dimension:
        raise ContractError(
            "profiled D requires full-rank supplied-score information in the declared "
            "interest/nuisance parameterization"
        )
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
    objective = _ProfiledDObjective(interest=criterion.interest_indices, nuisance=nuisance)
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
    extension_labels = jnp.asarray(
        _assign_nearest(all_coordinates, prepared.transform.apply(state.means), None)
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
        profiled_geometry=_profiled_geometry_report(
            prepared.scores,
            prepared.weights,
            run.labels,
            state,
            nuisance=nuisance,
            exchange_stable=run.exchange_stable,
        ),
        schema=schema,
    )


@execution_scope
def exchange_stability_report(
    scores: ArrayLike,
    labels: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    criterion: DOptimality | ProfiledDOptimality | None = None,
    rank_rtol: float | None = None,
    gain_tolerance: float = 1e-10,
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
    weights_array = sample.effective_weights
    state = objective.init_state(
        _cell_statistics(points, weights_array, jnp.asarray(effective_labels), n_bins)
    )
    outcome = _scan(
        points,
        weights_array,
        effective_labels,
        state,
        objective,
        DExchangeConfig(rank_rtol=rank_rtol, gain_tolerance=gain_tolerance),
        rows=False,
    )
    best_gain = float("-inf") if outcome.best is None else outcome.best.gain
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
    array = jnp.asarray(labels)
    n_rows = int(sample.scores.shape[0])
    if array.shape != (n_rows,):
        raise ContractError(f"labels must have shape [{n_rows}], got {array.shape}")
    if not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError("labels must contain integer bin labels")
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
) -> tuple[jnp.ndarray, _ExchangeObjective]:
    """Build the scan coordinates and criterion algebra of a supplied labeling.

    The D scan runs in Fisher-whitened coordinates and the profiled scan runs in
    the declared interest/nuisance parameterization, exactly as the two solvers
    do, so the reported objective matches ``PartitionResult.objective``.
    """
    scores = sample.effective_scores
    information = jnp.einsum("n,np,nq->pq", sample.effective_weights, scores, scores)
    transform = fisher_transform(information, whiten=True, rank_rtol=rank_rtol)
    if isinstance(criterion, DOptimality):
        return transform.apply(scores), _DObjective()
    dimension = int(scores.shape[1])
    if any(index >= dimension for index in criterion.interest_indices):
        raise ContractError(f"interest indices must be smaller than score dimension {dimension}")
    interest = set(criterion.interest_indices)
    nuisance = tuple(index for index in range(dimension) if index not in interest)
    if not nuisance:
        raise ContractError("profiled D requires a nuisance block; use DOptimality")
    if transform.rank != dimension:
        raise ContractError(
            "profiled D requires full-rank supplied-score information in the declared "
            "interest/nuisance parameterization"
        )
    return scores, _ProfiledDObjective(interest=criterion.interest_indices, nuisance=nuisance)


def _partition_result(
    prepared: _PreparedPartition,
    run: _ExchangeRun,
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
    raw_weights, raw_sums, raw_means = _raw_cell_statistics(sample, effective_labels, n_bins)
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


def _maybe_collapse_duplicates(
    scores: jnp.ndarray, weights: jnp.ndarray, collapse_duplicates: bool | None
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply the configured duplicate-score-collapsing policy, or skip it.

    Skipping returns an identity row mapping, so downstream code that expands
    atom-level results back to input rows behaves the same either way; only
    the effective atom count, and therefore the search space the solver
    explores, changes.
    """
    if resolve_collapse_duplicates(collapse_duplicates, int(scores.shape[0])):
        return collapse_duplicate_scores(scores, weights)
    return scores, weights, jnp.arange(scores.shape[0])


def _prepare_partition(
    scores: ArrayLike,
    weights: ArrayLike | None,
    *,
    n_bins: int,
    config: PartitionConfig,
) -> _PreparedPartition:
    sample = validate_sample(scores, weights)
    validate_n_bins(n_bins, sample.n_effective)
    effective_scores, effective_weights, inverse_rows = _maybe_collapse_duplicates(
        sample.effective_scores, sample.effective_weights, config.collapse_duplicates
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
    labels = jnp.asarray(initial_labels)
    n_rows = int(prepared.sample.scores.shape[0])
    if labels.shape != (n_rows,):
        raise ContractError(f"initial_labels must have shape [{n_rows}], got {labels.shape}")
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise TypeError("initial_labels must contain integer bin labels")
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
    objective: _ExchangeObjective,
    config: PartitionConfig,
    initial_labels: jnp.ndarray | None = None,
) -> _ExchangeRun:
    """Route one prepared finite problem to its configured solver."""
    if isinstance(config, MahalanobisLloydConfig):
        return _optimize_lloyd(
            points=points,
            coordinates=coordinates,
            weights=weights,
            n_bins=n_bins,
            objective=objective,
            config=config,
            initial_labels=initial_labels,
        )
    return _optimize_exchange(
        points=points,
        coordinates=coordinates,
        weights=weights,
        n_bins=n_bins,
        objective=objective,
        config=config,
        initial_labels=initial_labels,
    )


def _optimize_exchange(
    *,
    points: jnp.ndarray,
    coordinates: jnp.ndarray,
    weights: jnp.ndarray,
    n_bins: int,
    objective: _ExchangeObjective,
    config: DExchangeConfig,
    initial_labels: jnp.ndarray | None = None,
) -> _ExchangeRun:
    """Run every seeded exchange restart and keep the best exact objective.

    Supplied labels replace the seeding of the first restart only, so ``init``
    and ``initializer_restarts`` still govern restarts one and above and an initializer can be
    compared against ordinary seeding inside a single call.
    """
    best: _ExchangeRun | None = None
    for restart in range(config.solver_restarts):
        labels = (
            initial_labels
            if restart == 0 and initial_labels is not None
            else _initial_labels(coordinates, weights, n_bins, config, restart)
        )
        run = _run_exchange(points, weights, labels, n_bins, objective, config)
        # Strict comparison makes the earliest restart win exact ties.
        if best is None or run.state.objective > best.state.objective:
            best = run
    if best is None:
        raise ContractError("solver_restarts must be at least one")
    return best


def _initial_labels(
    coordinates: jnp.ndarray,
    weights: jnp.ndarray,
    n_bins: int,
    config: DExchangeConfig,
    restart: int,
) -> jnp.ndarray:
    seed = config.seed + restart
    if config.init == "random":
        # A permuted balanced labeling keeps every requested cell nonempty.
        permutation = random_permutation(seed, coordinates.shape[0])
        return (jnp.arange(coordinates.shape[0]) % n_bins).astype(jnp.int32)[permutation]
    return _kmeans_labels(
        coordinates,
        weights,
        n_bins,
        rank_rtol=config.rank_rtol,
        seed=seed,
        initializer_restarts=config.initializer_restarts,
    )


def _kmeans_labels(
    coordinates: jnp.ndarray,
    weights: jnp.ndarray,
    n_bins: int,
    *,
    rank_rtol: float | None,
    seed: int,
    initializer_restarts: int,
) -> jnp.ndarray:
    """Seed one solver with deterministic weighted k-means++ labels."""
    initializer = weighted_kmeans(
        coordinates,
        weights,
        n_bins,
        KMeansConfig(
            whiten=False,
            rank_rtol=rank_rtol,
            seed=seed,
            solver_restarts=initializer_restarts,
            max_iter=100,
            tolerance=1e-8,
            record_every=100,
        ),
    )
    return jnp.asarray(_assign_nearest(coordinates, initializer.centers, None))


def _optimize_lloyd(
    *,
    points: jnp.ndarray,
    coordinates: jnp.ndarray,
    weights: jnp.ndarray,
    n_bins: int,
    objective: _ExchangeObjective,
    config: MahalanobisLloydConfig,
    initial_labels: jnp.ndarray | None = None,
) -> _ExchangeRun:
    """Iterate guarded nearest-centroid batches, then settle the exchange guard.

    Each iteration freezes the current criterion metric, relabels every row to
    its nearest centroid in that metric, and repairs any cell the proposal would
    empty. The proposal is adopted only when the objective of its exactly
    rebuilt state strictly improves, because the frozen-metric batch step is not
    monotone on its own. Iteration stops at the first non-improving or unchanged
    proposal, or at ``max_iter``; ``guard`` then decides whether the labels are
    handed to the exchange engine or merely certified by one final scan. Supplied
    labels replace the k-means seeding, so ``initializer_restarts`` then governs only the
    exchange handoff.
    """
    seeded = (
        _kmeans_labels(
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
    state = objective.init_state(_cell_statistics(points, weights, jnp.asarray(labels), n_bins))
    history = [state.objective]
    iterations = 0
    accepted = 0

    while iterations < config.max_iter:
        iterations += 1
        proposal = _lloyd_proposal(points, labels, state, objective, n_bins)
        if np.array_equal(proposal, labels):
            break
        trial = _trial_state(points, weights, proposal, n_bins, objective)
        if trial is None or trial.objective <= state.objective + config.gain_tolerance:
            break
        state, labels = trial, proposal
        accepted += 1
        history.append(state.objective)

    return _settle_lloyd(
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


def _settle_lloyd(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: np.ndarray,
    n_bins: int,
    state: _ExchangeState,
    objective: _ExchangeObjective,
    config: MahalanobisLloydConfig,
    *,
    history: list[float],
    iterations: int,
    accepted: int,
) -> _ExchangeRun:
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
        handoff = _run_exchange(
            points, weights, jnp.asarray(labels), n_bins, objective, exchange_config
        )
        history.extend(handoff.objective_history[1:])
        return _ExchangeRun(
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
    outcome = _scan(points, weights, labels, state, objective, exchange_config, rows=False)
    best_remaining_gain = float("-inf") if outcome.best is None else outcome.best.gain
    return _ExchangeRun(
        labels=jnp.asarray(labels),
        state=state,
        objective_history=tuple(history),
        accepted_moves=0,
        scans=1,
        exchange_stable=best_remaining_gain <= config.gain_tolerance,
        best_remaining_gain=best_remaining_gain,
        lloyd_iterations=iterations,
        accepted_lloyd_steps=accepted,
    )


def _lloyd_proposal(
    points: jnp.ndarray,
    labels: np.ndarray,
    state: _ExchangeState,
    objective: _ExchangeObjective,
    n_bins: int,
) -> np.ndarray:
    """Propose the complete nearest-centroid relabeling of the current metric."""
    metric = objective.assignment_metric(state)
    proposal = _assign_nearest(points, state.means, metric)
    return _repair_empty_cells(proposal, labels, points, state.means, metric, n_bins)


def _assign_nearest(
    points: jnp.ndarray, means: jnp.ndarray, metric: jnp.ndarray | None
) -> np.ndarray:
    """Assign every row to its nearest centroid in memory-bounded chunks.

    ``metric=None`` selects the Euclidean assignment (used to extend a
    profiled-D partition, whose objective has no criterion metric to reuse)
    instead of the Mahalanobis one; chunking rows changes nothing about the
    per-row arithmetic, so both are bit-identical to their unchunked form.
    """
    n_rows = int(points.shape[0])
    chunk_rows = _candidate_chunk_rows(points, int(means.shape[0]))
    labels = np.empty(n_rows, dtype=np.int32)
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        chunk_points = points[start:stop]
        chunk_labels = (
            hard_assign(chunk_points, means)
            if metric is None
            else _metric_assign(chunk_points, means, metric)
        )
        labels[start:stop] = np.asarray(chunk_labels)
    return labels


def _repair_empty_cells(
    proposal: np.ndarray,
    labels: np.ndarray,
    points: jnp.ndarray,
    means: jnp.ndarray,
    metric: jnp.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Keep the most representative current member of every emptied cell.

    A frozen-metric batch can vacate a cell entirely, which the exact criterion
    state cannot represent. The repair retains, for each emptied cell, the row
    it already holds that is closest to its own centroid, so the recovered cell
    is the one the criterion currently supports best. The retained rows are
    distinct because the current labels partition the sample; a repair that
    empties some other cell as collateral is rejected by the guard like any
    other infeasible proposal.
    """
    empty = np.flatnonzero(np.bincount(proposal, minlength=n_bins)[:n_bins] == 0)
    if empty.size == 0:
        return proposal
    repaired = proposal.copy()
    for cell in empty:
        members = np.flatnonzero(labels == cell)
        if members.size == 0:
            continue
        residuals = points[members] - means[cell]
        distances = np.asarray(jnp.einsum("nr,rs,ns->n", residuals, metric, residuals))
        repaired[members[int(np.argmin(distances))]] = int(cell)
    return repaired


def _run_exchange(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: jnp.ndarray,
    n_bins: int,
    objective: _ExchangeObjective,
    config: DExchangeConfig,
) -> _ExchangeRun:
    """Scan, accept exact positive-gain work, and stop at exchange stability."""
    label_array = np.asarray(labels, dtype=np.int32).copy()
    state = objective.init_state(_cell_statistics(points, weights, labels, n_bins))
    history = [state.objective]
    accepted_moves = 0
    scans = 0
    exchange_stable = False
    best_remaining_gain = float("-inf")
    # first_improvement deliberately stops each scan early, so it never produces
    # the complete per-row table a guarded batch needs.
    batched = config.batch_moves and not config.first_improvement
    batch_size = label_array.shape[0]
    limit = _SAFETY_SCAN_LIMIT if config.max_scans is None else config.max_scans

    while scans < limit:
        scans += 1
        outcome = _scan(points, weights, label_array, state, objective, config, rows=batched)
        best_remaining_gain = float("-inf") if outcome.best is None else outcome.best.gain
        if outcome.best is None or outcome.best.gain <= config.gain_tolerance:
            exchange_stable = True
            break
        accepted = (
            _accept_batch(
                points,
                weights,
                label_array,
                n_bins,
                state,
                objective,
                config,
                outcome=outcome,
                batch_size=batch_size,
            )
            if batched
            else None
        )
        if accepted is None:
            relocation = _relocation(points, weights, label_array, state, outcome.best)
            state = objective.apply_move(state, relocation)
            label_array[outcome.best.row] = outcome.best.destination
            accepted_moves += 1
        else:
            state, label_array, moved = accepted
            accepted_moves += moved
            batch_size = min(2 * moved, label_array.shape[0])
        history.append(state.objective)

    if not exchange_stable:
        scans += 1
        outcome = _scan(points, weights, label_array, state, objective, config, rows=False)
        best_remaining_gain = float("-inf") if outcome.best is None else outcome.best.gain
        exchange_stable = outcome.best is None or outcome.best.gain <= config.gain_tolerance

    return _ExchangeRun(
        labels=jnp.asarray(label_array),
        state=state,
        objective_history=tuple(history),
        accepted_moves=accepted_moves,
        scans=scans,
        exchange_stable=exchange_stable,
        best_remaining_gain=best_remaining_gain,
    )


def _scan(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: np.ndarray,
    state: _ExchangeState,
    objective: _ExchangeObjective,
    config: DExchangeConfig,
    *,
    rows: bool,
) -> _ScanOutcome:
    """Evaluate every admissible relocation in memory-bounded row chunks."""
    n_rows = points.shape[0]
    chunk_rows = _candidate_chunk_rows(points, state.weights.shape[0])
    label_array = jnp.asarray(labels)
    row_gains = np.full(n_rows, -np.inf) if rows else None
    row_destinations = np.zeros(n_rows, dtype=np.int64) if rows else None
    best: _Move | None = None
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        gains = np.asarray(
            objective.chunk_gains(
                state, points[start:stop], weights[start:stop], label_array[start:stop]
            )
        )
        if config.first_improvement:
            improving = gains > config.gain_tolerance
            improving_rows = np.flatnonzero(improving.any(axis=1))
            if improving_rows.size:
                row = int(improving_rows[0])
                destination = int(np.argmax(improving[row]))
                return _ScanOutcome(
                    _Move(start + row, destination, float(gains[row, destination])), None, None
                )
        destinations = np.argmax(gains, axis=1)
        best_gains = gains[np.arange(stop - start), destinations]
        if row_gains is not None and row_destinations is not None:
            row_gains[start:stop] = best_gains
            row_destinations[start:stop] = destinations
        row = int(np.argmax(best_gains))
        gain = float(best_gains[row])
        if np.isfinite(gain) and (best is None or gain > best.gain):
            best = _Move(start + row, int(destinations[row]), gain)
    return _ScanOutcome(best, row_gains, row_destinations)


def _accept_batch(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: np.ndarray,
    n_bins: int,
    state: _ExchangeState,
    objective: _ExchangeObjective,
    config: DExchangeConfig,
    *,
    outcome: _ScanOutcome,
    batch_size: int,
) -> tuple[_ExchangeState, np.ndarray, int] | None:
    """Relocate many rows at once, accepting only a verified exact improvement.

    Every candidate batch is scored by rebuilding the criterion state exactly
    from the proposed labels, so acceptance never relies on accumulated
    increments and the exchange stays strictly monotone. A rejected batch is
    halved by gain rank and retried; when nothing survives, the caller applies
    the exact single best relocation instead.
    """
    if outcome.row_gains is None or outcome.row_destinations is None:
        return None
    improving = np.flatnonzero(outcome.row_gains > config.gain_tolerance)
    # Below this floor a batch cannot amortize its exact rebuild, and the few
    # surviving rows are strongly interacting boundary atoms whose certified
    # greedy relocation is both cheaper and better conditioned.
    if improving.size <= _MIN_BATCH_ROWS:
        return None
    order = improving[np.argsort(-outcome.row_gains[improving], kind="stable")]
    destinations = outcome.row_destinations[order]
    size = min(
        batch_size,
        _mass_budget_cut(
            np.asarray(weights)[order],
            labels[order],
            destinations,
            np.asarray(state.weights),
        ),
    )
    for _ in range(_MAX_BATCH_SHRINKS):
        if size <= 1:
            # The single best relocation has an exact positive gain; the caller
            # applies it through the cheaper rank-two path.
            return None
        trial_labels = labels.copy()
        trial_labels[order[:size]] = destinations[:size]
        trial = _trial_state(points, weights, trial_labels, n_bins, objective)
        if trial is not None and trial.objective > state.objective + config.gain_tolerance:
            return trial, trial_labels, size
        size //= 2
    return None


def _mass_budget_cut(
    row_weights: np.ndarray,
    sources: np.ndarray,
    destinations: np.ndarray,
    cell_weights: np.ndarray,
) -> int:
    """Return how many gain-ranked rows a batch may relocate at once.

    The determinant lemma is exact for one relocation. A simultaneous batch
    displaces every touched cell mean in proportion to the mass it moves, so
    the batch is cut before any cell would gain or lose more than a fixed
    fraction of its weight. Without this cut the guarded batch still stays
    monotone, but it repeatedly overshoots into worse exchange-stable states.
    """
    cut = int(row_weights.shape[0])
    for cell in range(int(cell_weights.shape[0])):
        budget = _BATCH_MASS_FRACTION * float(cell_weights[cell])
        for touching in (sources == cell, destinations == cell):
            index = np.flatnonzero(touching)
            if index.size == 0:
                continue
            over = np.flatnonzero(np.cumsum(row_weights[index]) > budget)
            if over.size:
                cut = min(cut, int(index[over[0]]))
    return cut


def _trial_state(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: np.ndarray,
    n_bins: int,
    objective: _ExchangeObjective,
) -> _ExchangeState | None:
    """Rebuild the exact state of a proposed labeling, or reject an infeasible one."""
    try:
        return objective.init_state(_cell_statistics(points, weights, jnp.asarray(labels), n_bins))
    except (ContractError, RefusalError):
        # An emptied cell or a singular proposal is an ordinary batch rejection,
        # regardless of whether ``init_state`` reports it as a malformed input
        # (``ContractError``) or a theorem-backed refusal on this trial labeling
        # (``RefusalError``, e.g. a profiled trial whose binned information
        # would be degenerate): a discarded trial is never a request outcome,
        # so neither should surface past this internal rejection.
        return None


def _cell_statistics(
    points: jnp.ndarray, weights: jnp.ndarray, labels: jnp.ndarray, n_bins: int
) -> _CellStatistics:
    """Accumulate exact weighted cell occupancy, score sums, and score means."""
    statistics = scatter_bin_statistics(labels, weights, points, n_bins)
    if bool(np.asarray(jnp.any(statistics.weights <= 0))):
        raise ContractError("exact exchange requires exactly n_bins nonempty cells")
    return _CellStatistics(weights=statistics.weights, sums=statistics.sums, means=statistics.means)


@backend_jit
def _cell_information_matrices(
    cell_weights: jnp.ndarray, cell_means: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the cell information, its inverse, and its determinant sign and log.

    Every guarded batch acceptance rebuilds this from scratch to keep the
    exchange exactly monotone, so the rebuild runs far more often than a move
    does. On a rank-sized matrix each of these primitives costs more to
    dispatch eagerly than to execute, which is why they are compiled as one
    kernel; the inverse is returned unconditionally because a singular matrix
    yields non-finite entries rather than an error, and the caller's sign check
    still rejects it.
    """
    information = jnp.einsum("b,bp,bq->pq", cell_weights, cell_means, cell_means)
    information = 0.5 * (information + information.T)
    sign, logdet = jnp.linalg.slogdet(information)
    return information, jnp.linalg.inv(information), sign, logdet


def _candidates(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: jnp.ndarray,
    cell_weights: jnp.ndarray,
    cell_means: jnp.ndarray,
) -> _Candidates:
    """Build the exact rank-two geometry of every relocation in one chunk."""
    source_weights = cell_weights[labels]
    source_denominator = jnp.where(source_weights > weights, source_weights - weights, 1)
    destinations = jnp.arange(cell_weights.shape[0])[None, :]
    return _Candidates(
        source_residuals=points - cell_means[labels],
        destination_residuals=points[:, None, :] - cell_means[None, :, :],
        alpha=weights * source_weights / source_denominator,
        beta=weights[:, None] * cell_weights[None, :] / (cell_weights[None, :] + weights[:, None]),
        admissible=(source_weights > weights)[:, None] & (destinations != labels[:, None]),
    )


def _determinant_ratios(
    chunk: _Candidates,
    source: jnp.ndarray,
    destination: jnp.ndarray,
    inverse: jnp.ndarray,
) -> jnp.ndarray:
    """Return the exact rank-two determinant ratio of every candidate relocation."""
    q_source = jnp.einsum("nr,rs,ns->n", source, inverse, source)
    q_destination = jnp.einsum("nbr,rs,nbs->nb", destination, inverse, destination)
    q_cross = jnp.einsum("nr,rs,nbs->nb", source, inverse, destination)
    return (1 + chunk.alpha[:, None] * q_source[:, None]) * (
        1 - chunk.beta * q_destination
    ) + chunk.alpha[:, None] * chunk.beta * q_cross**2


@backend_jit
def _d_chunk_gains(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: jnp.ndarray,
    cell_weights: jnp.ndarray,
    cell_means: jnp.ndarray,
    inverse: jnp.ndarray,
) -> jnp.ndarray:
    """Return one chunk of exact D gains as a single compiled kernel.

    A scan evaluates this for every row of the sample, so it is the innermost
    hot loop of the whole engine. Compiling it lets XLA fuse the
    ``[chunk, n_bins, rank]`` destination residual into the contractions that
    consume it instead of materializing it, which is where the win comes from;
    the arithmetic is exactly the eager expression it replaces.
    """
    chunk = _candidates(points, weights, labels, cell_weights, cell_means)
    ratios = _determinant_ratios(
        chunk, chunk.source_residuals, chunk.destination_residuals, inverse
    )
    valid = chunk.admissible & (ratios > 0)
    return jnp.where(valid, jnp.log(jnp.where(valid, ratios, 1)), -jnp.inf)


@backend_jit(static_argnames=("nuisance",))
def _profiled_chunk_gains(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: jnp.ndarray,
    cell_weights: jnp.ndarray,
    cell_means: jnp.ndarray,
    inverse: jnp.ndarray,
    nuisance_inverse: jnp.ndarray,
    nuisance: tuple[int, ...],
) -> jnp.ndarray:
    """Return one chunk of exact profiled-``D_s`` gains as a single compiled kernel.

    ``nuisance`` is static so the score-column slice is a compile-time constant
    and the two determinant ratios fuse into one kernel.
    """
    chunk = _candidates(points, weights, labels, cell_weights, cell_means)
    indices = jnp.asarray(nuisance)
    full_ratios = _determinant_ratios(
        chunk, chunk.source_residuals, chunk.destination_residuals, inverse
    )
    nuisance_ratios = _determinant_ratios(
        chunk,
        chunk.source_residuals[:, indices],
        chunk.destination_residuals[:, :, indices],
        nuisance_inverse,
    )
    valid = chunk.admissible & (full_ratios > 0) & (nuisance_ratios > 0)
    return jnp.where(
        valid,
        jnp.log(jnp.where(valid, full_ratios, 1)) - jnp.log(jnp.where(valid, nuisance_ratios, 1)),
        -jnp.inf,
    )


def _relocation(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: np.ndarray,
    state: _ExchangeState,
    move: _Move,
) -> _Relocation:
    """Build the exact rank-two update data of one accepted relocation."""
    source = int(labels[move.row])
    point_weight = weights[move.row]
    point = points[move.row]
    source_weight = state.weights[source]
    destination_weight = state.weights[move.destination]
    return _Relocation(
        source=source,
        destination=move.destination,
        point=point,
        point_weight=point_weight,
        source_residual=point - state.means[source],
        destination_residual=point - state.means[move.destination],
        alpha=point_weight * source_weight / (source_weight - point_weight),
        beta=point_weight * destination_weight / (destination_weight + point_weight),
    )


def _relocate_cells(cells: _CellStatistics, relocation: _Relocation) -> _CellStatistics:
    weight = relocation.point_weight
    cell_weights = scatter_add(cells.weights, relocation.source, -weight)
    cell_weights = scatter_add(cell_weights, relocation.destination, weight)
    sums = scatter_add(cells.sums, relocation.source, -weight * relocation.point)
    sums = scatter_add(
        sums,
        relocation.destination,
        weight * relocation.point,
    )
    return _CellStatistics(weights=cell_weights, sums=sums, means=sums / cell_weights[:, None])


def _rank_two_block(
    information: jnp.ndarray,
    inverse: jnp.ndarray,
    relocation: _Relocation,
    *,
    block: jnp.ndarray | None = None,
) -> _UpdatedBlock:
    """Update one symmetric information block and its inverse by an exact relocation.

    ``block`` selects a score-column sub-block, which is how the profiled
    criterion maintains its nuisance information alongside the full matrix.
    """
    source = relocation.source_residual if block is None else relocation.source_residual[block]
    destination = (
        relocation.destination_residual if block is None else relocation.destination_residual[block]
    )
    updated = (
        information
        + relocation.alpha * jnp.outer(source, source)
        - relocation.beta * jnp.outer(destination, destination)
    )
    updated = 0.5 * (updated + updated.T)
    refreshed = _rank_two_inverse_update(
        inverse, source, destination, relocation.alpha, relocation.beta, updated
    )
    sign, logdet = jnp.linalg.slogdet(updated)
    return _UpdatedBlock(
        information=updated,
        inverse=refreshed,
        sign=float(np.asarray(sign)),
        logdet=float(np.asarray(logdet)),
    )


def _rank_two_inverse_update(
    inverse: jnp.ndarray,
    source_residual: jnp.ndarray,
    destination_residual: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    information: jnp.ndarray,
) -> jnp.ndarray:
    """Apply two Sherman-Morrison updates, refreshing exactly when they drift."""
    source_projection = inverse @ source_residual
    source_denominator = 1 + alpha * (source_residual @ source_projection)
    first = inverse - alpha * jnp.outer(source_projection, source_projection) / source_denominator
    destination_projection = first @ destination_residual
    destination_denominator = 1 - beta * (destination_residual @ destination_projection)
    updated = (
        first
        + beta * jnp.outer(destination_projection, destination_projection) / destination_denominator
    )
    updated = 0.5 * (updated + updated.T)
    identity = jnp.eye(information.shape[0], dtype=information.dtype)
    residual = jnp.max(jnp.abs(information @ updated - identity))
    tolerance = (
        _RANK_TWO_RESIDUAL_TOLERANCE_F64
        if information.dtype == jnp.float64
        else _RANK_TWO_RESIDUAL_TOLERANCE_F32
    )
    if (
        not bool(np.asarray(jnp.isfinite(destination_denominator)))
        or float(np.asarray(destination_denominator)) <= 0
        or not bool(np.asarray(jnp.isfinite(residual)))
        or float(np.asarray(residual)) > tolerance
    ):
        return jnp.linalg.inv(information)
    return updated


def _profiled_semimetric(state: _ExchangeState, nuisance: tuple[int, ...]) -> jnp.ndarray:
    r"""Return \(G_s=L^\top S^{-1}L\) of one profiled state, with \(L=[I,-BC^{-1}]\)."""
    indices = jnp.asarray(nuisance)
    metric = scatter_block_add(
        state.inverse,
        indices,
        -_require_nuisance(state.nuisance_inverse),
    )
    return 0.5 * (metric + metric.T)


def _require_nuisance(matrix: jnp.ndarray | None) -> jnp.ndarray:
    if matrix is None:
        raise ContractError("profiled-D exchange state is missing its nuisance block")
    return matrix


def _candidate_chunk_rows(scores: jnp.ndarray, n_bins: int) -> int:
    return assignment_chunk_rows(scores.dtype, scores.shape[0], n_bins, scores.shape[1])


def _geometry_report(
    coordinates: jnp.ndarray,
    weights: jnp.ndarray,
    labels: jnp.ndarray,
    state: _ExchangeState,
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
    chunk_rows = _candidate_chunk_rows(coordinates, n_bins)
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
                _d_chunk_gains(
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
        separation_certified=separation_residual <= _SEPARATION_RTOL * separation_scale,
        gain_tolerance=gain_tolerance,
    )


def _profiled_geometry_report(
    scores: jnp.ndarray,
    weights: jnp.ndarray,
    labels: jnp.ndarray,
    state: _ExchangeState,
    *,
    nuisance: tuple[int, ...],
    exchange_stable: bool,
) -> ProfiledGeometryReport:
    """Diagnose the finite efficient-semimetric gap of a terminal profiled state."""
    metric = _profiled_semimetric(state, nuisance)
    source_residuals = scores - state.means[labels]
    destination_residuals = scores[:, None, :] - state.means[None, :, :]
    source_distances = jnp.einsum("nr,rs,ns->n", source_residuals, metric, source_residuals)
    destination_distances = jnp.einsum(
        "nbr,rs,nbs->nb", destination_residuals, metric, destination_residuals
    )
    violations = jnp.maximum(source_distances[:, None] - destination_distances, 0)
    q_source = jnp.einsum("nr,rs,ns->n", source_residuals, state.inverse, source_residuals)
    source_weights = state.weights[labels]
    bounds = (
        weights[:, None]
        * q_source[:, None]
        * (1 / source_weights[:, None] + 1 / state.weights[None, :])
    )
    destinations = jnp.arange(state.weights.shape[0])[None, :]
    admissible = (source_weights > weights)[:, None] & (destinations != labels[:, None])
    evaluated_violations = jnp.where(admissible, violations, 0)
    evaluated_bounds = jnp.where(admissible, bounds, 0)
    residuals = jnp.where(admissible, violations - bounds, -jnp.inf)
    return ProfiledGeometryReport(
        metric=metric,
        maximum_positive_violation=float(np.asarray(jnp.max(evaluated_violations))),
        maximum_theoretical_bound=float(np.asarray(jnp.max(evaluated_bounds))),
        maximum_bound_residual=float(np.asarray(jnp.max(residuals))),
        violating_moves=int(np.asarray(jnp.sum(admissible & (violations > 0)))),
        evaluated_moves=int(np.asarray(jnp.sum(admissible))),
        bound_certified=exchange_stable,
    )


def _metric_assign(scores: jnp.ndarray, means: jnp.ndarray, inverse: jnp.ndarray) -> jnp.ndarray:
    residuals = scores[:, None, :] - means[None, :, :]
    distances = jnp.einsum("nbr,rs,nbs->nb", residuals, inverse, residuals)
    return jnp.argmin(distances, axis=1)


def _raw_cell_statistics(
    sample: _ValidatedSample, labels: jnp.ndarray, n_bins: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    statistics = scatter_bin_statistics(
        labels, sample.effective_weights, sample.effective_scores, n_bins
    )
    return statistics.weights, statistics.sums, statistics.means
