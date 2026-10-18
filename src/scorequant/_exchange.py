r"""Private exchange algebra: cell statistics, rank-two updates and objective state.

A relocation moves one weighted row from a source cell to a destination cell.
Its effect on the binned information is an exact rank-two update, so its gain
is a ratio of determinants (the matrix determinant lemma) and the inverse
follows by two Sherman-Morrison steps. D and profiled-\(D_s\) share this
algebra through the ``ExchangeObjective`` protocol; the exchange engine in
``solvers/exchange.py`` and the geometry diagnostics consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ._binstats import scatter_bin_statistics
from ._errors import ContractError, RefusalError
from ._execution import backend_jit, scatter_add, scatter_block_add
from ._execution import xp as jnp
from ._validation import nuisance_complement
from .information import (
    PROFILED_RANK_ADVICE,
    _nuisance_information,
    binned_information_is_degenerate,
)
from .transforms import is_numerically_singular

# Largest tolerated max-norm residual of I @ I^-1 - identity after two chained
# Sherman-Morrison rank-one updates, dtype-dependent because the accumulated
# rounding of the chained updates scales with machine epsilon. Above this the
# incremental inverse is judged to have drifted too far from the exactly
# rebuilt one, and _rank_two_inverse_update falls back to a fresh inversion.
_RANK_TWO_RESIDUAL_TOLERANCE_F64 = 1e-8
_RANK_TWO_RESIDUAL_TOLERANCE_F32 = 2e-3


@dataclass(frozen=True, slots=True)
class CellStatistics:
    """Weighted occupancy, score sums, and score means of every requested cell."""

    weights: jnp.ndarray
    sums: jnp.ndarray
    means: jnp.ndarray


@dataclass(frozen=True, slots=True)
class ExchangeState:
    """Cell statistics and criterion matrices of one labeling."""

    cells: CellStatistics
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
class Relocation:
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


class ExchangeObjective(Protocol):
    """Criterion-specific algebra shared by every exchange run."""

    def init_state(self, cells: CellStatistics) -> ExchangeState:
        """Build criterion matrices and the exact objective of one labeling."""
        ...

    def chunk_gains(
        self,
        state: ExchangeState,
        points: jnp.ndarray,
        weights: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return exact objective gains of every candidate in one row chunk."""
        ...

    def apply_move(self, state: ExchangeState, relocation: Relocation) -> ExchangeState:
        """Return the exactly refreshed state after one accepted relocation."""
        ...

    def assignment_metric(self, state: ExchangeState) -> jnp.ndarray:
        """Return the quadratic form whose nearest-centroid rule a batch proposes."""
        ...


@dataclass(frozen=True, slots=True)
class DObjective:
    """Log-determinant algebra of Fisher-whitened D-optimal exchange."""

    def init_state(self, cells: CellStatistics) -> ExchangeState:
        """Build the cell information matrix, its inverse, and its log determinant."""
        information, inverse, _sign, logdet = _cell_information_matrices(cells.weights, cells.means)
        # The library's rank rule, as information_report applies it to the same
        # whitened between-cell matrix: a positive log-determinant sign on a
        # deficient matrix is rounding noise, and a start judged regular here
        # but singular there would certify a partition the report calls empty.
        if is_numerically_singular(information):
            raise ContractError(
                "initial D partition is singular; increase n_bins or use a different "
                "reference sample"
            )
        return ExchangeState(
            cells=cells,
            information=information,
            inverse=inverse,
            objective=float(np.asarray(logdet)),
        )

    def chunk_gains(
        self,
        state: ExchangeState,
        points: jnp.ndarray,
        weights: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return the exact rank-two log-determinant gain of every candidate."""
        return chunk_gains(points, weights, labels, state.weights, state.means, state.inverse)

    def apply_move(self, state: ExchangeState, relocation: Relocation) -> ExchangeState:
        """Apply the exact rank-two information and inverse update of one move."""
        block = _rank_two_block(state.information, state.inverse, relocation)
        # Only a sanity check: an accepted move has a positive gain, so from a
        # state the rank rule called regular the determinant can only grow.
        if block.sign <= 0:
            raise FloatingPointError("an accepted D move produced singular information")
        return ExchangeState(
            cells=_relocate_cells(state.cells, relocation),
            information=block.information,
            inverse=block.inverse,
            objective=block.logdet,
        )

    def assignment_metric(self, state: ExchangeState) -> jnp.ndarray:
        r"""Return \(I^{-1}\), the metric of the D nearest-centroid rule."""
        return state.inverse


@dataclass(frozen=True, slots=True)
class ProfiledDObjective:
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

    def init_state(self, cells: CellStatistics) -> ExchangeState:
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
        return ExchangeState(
            cells=cells,
            information=information,
            inverse=inverse,
            objective=float(np.asarray(full_logdet - nuisance_logdet)),
            nuisance_information=nuisance_information,
            nuisance_inverse=jnp.linalg.inv(nuisance_information),
        )

    def chunk_gains(
        self,
        state: ExchangeState,
        points: jnp.ndarray,
        weights: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return the difference of the full and nuisance determinant-lemma gains."""
        return chunk_gains(
            points,
            weights,
            labels,
            state.weights,
            state.means,
            state.inverse,
            _require_nuisance(state.nuisance_inverse),
            self.nuisance,
        )

    def apply_move(self, state: ExchangeState, relocation: Relocation) -> ExchangeState:
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
        return ExchangeState(
            cells=_relocate_cells(state.cells, relocation),
            information=full.information,
            inverse=full.inverse,
            objective=full.logdet - nuisance.logdet,
            nuisance_information=nuisance.information,
            nuisance_inverse=nuisance.inverse,
        )

    def assignment_metric(self, state: ExchangeState) -> jnp.ndarray:
        r"""Return the efficient semimetric \(G_s=L^\top S^{-1}L\) of the current state.

        Subtracting the nuisance inverse from the nuisance block of the full
        inverse yields exactly the profiled gradient metric, so the batch
        proposal ranks destinations by the same quadratic forms the exchange
        gains difference. The form is singular by construction: it measures
        only the efficient-score directions.
        """
        return profiled_semimetric(state, self.nuisance)


def profiled_objective(interest: tuple[int, ...], dimension: int, rank: int) -> ProfiledDObjective:
    """Validate a profiled split of ``dimension`` score columns and build its algebra.

    The finite solver and the stability certificate share this check: the
    interest indices must name columns, a nuisance block must remain, and the
    supplied-score information must be full rank in the declared
    interest/nuisance parameterization.
    """
    nuisance = nuisance_complement(interest, dimension)
    if rank != dimension:
        raise ContractError(
            "profiled D requires full-rank supplied-score information in the declared "
            "interest/nuisance parameterization"
        )
    return ProfiledDObjective(interest=interest, nuisance=nuisance)


def cell_statistics(
    points: jnp.ndarray, weights: jnp.ndarray, labels: jnp.ndarray, n_bins: int
) -> CellStatistics:
    """Accumulate exact weighted cell occupancy, score sums, and score means."""
    statistics = scatter_bin_statistics(labels, weights, points, n_bins)
    if bool(np.asarray(jnp.any(statistics.weights <= 0))):
        raise ContractError("exact exchange requires exactly n_bins nonempty cells")
    return CellStatistics(weights=statistics.weights, sums=statistics.sums, means=statistics.means)


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


@backend_jit(static_argnames=("nuisance",))
def chunk_gains(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: jnp.ndarray,
    cell_weights: jnp.ndarray,
    cell_means: jnp.ndarray,
    inverse: jnp.ndarray,
    nuisance_inverse: jnp.ndarray | None = None,
    nuisance: tuple[int, ...] | None = None,
) -> jnp.ndarray:
    """Return one chunk of exact D or profiled-``D_s`` gains as a single compiled kernel.

    A scan evaluates this for every row of the sample, so it is the innermost
    hot loop of the whole engine. Compiling it lets XLA fuse the
    ``[chunk, n_bins, rank]`` destination residual into the contractions that
    consume it instead of materializing it; the arithmetic is exactly the eager
    expression. With ``nuisance`` (static, so the score-column slice is a
    compile-time constant) the nuisance-block log ratio is subtracted.
    """
    chunk = _candidates(points, weights, labels, cell_weights, cell_means)
    ratios = _determinant_ratios(
        chunk, chunk.source_residuals, chunk.destination_residuals, inverse
    )
    valid = chunk.admissible & (ratios > 0)
    gains = jnp.log(jnp.where(valid, ratios, 1))
    if nuisance is not None and nuisance_inverse is not None:
        indices = jnp.asarray(nuisance)
        nuisance_ratios = _determinant_ratios(
            chunk,
            chunk.source_residuals[:, indices],
            chunk.destination_residuals[:, :, indices],
            nuisance_inverse,
        )
        valid = valid & (nuisance_ratios > 0)
        gains = gains - jnp.log(jnp.where(valid, nuisance_ratios, 1))
    return jnp.where(valid, gains, -jnp.inf)


def relocation(
    points: jnp.ndarray,
    weights: jnp.ndarray,
    labels: np.ndarray,
    state: ExchangeState,
    row: int,
    destination: int,
) -> Relocation:
    """Build the exact rank-two update data of relocating ``row`` to ``destination``."""
    source = int(labels[row])
    point_weight = weights[row]
    point = points[row]
    source_weight = state.weights[source]
    destination_weight = state.weights[destination]
    return Relocation(
        source=source,
        destination=destination,
        point=point,
        point_weight=point_weight,
        source_residual=point - state.means[source],
        destination_residual=point - state.means[destination],
        alpha=point_weight * source_weight / (source_weight - point_weight),
        beta=point_weight * destination_weight / (destination_weight + point_weight),
    )


def _relocate_cells(cells: CellStatistics, relocation: Relocation) -> CellStatistics:
    weight = relocation.point_weight
    cell_weights = scatter_add(cells.weights, relocation.source, -weight)
    cell_weights = scatter_add(cell_weights, relocation.destination, weight)
    sums = scatter_add(cells.sums, relocation.source, -weight * relocation.point)
    sums = scatter_add(sums, relocation.destination, weight * relocation.point)
    return CellStatistics(weights=cell_weights, sums=sums, means=sums / cell_weights[:, None])


def _rank_two_block(
    information: jnp.ndarray,
    inverse: jnp.ndarray,
    relocation: Relocation,
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


def profiled_semimetric(state: ExchangeState, nuisance: tuple[int, ...]) -> jnp.ndarray:
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
