"""Fisher eigenspace projection and optional whitening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._errors import ContractError
from ._execution import canonical_array, canonicalize_public, execution_scope
from ._execution import xp as jnp
from ._json import json_ready
from ._typing import ArrayLike, JsonValue
from .config import ExecutionConfig, dtype_tolerance, validate_rank_rtol

_FLOAT64_RANK_RTOL = 1e-10


def _default_rank_rtol(dtype: jnp.dtype) -> float:
    """Return the relative rank threshold of the working dtype (1e-10 in float64)."""
    return dtype_tolerance(_FLOAT64_RANK_RTOL, dtype)


def numerical_rank(
    matrix: jnp.ndarray, rank_rtol: float | None = None, *, unit_diagonal: bool = False
) -> int:
    """Return the rank of a symmetric matrix at the library's relative threshold.

    This is the one rank rule: an eigenvalue counts only above ``rank_rtol``
    times the largest one, exactly as :func:`fisher_transform` selects
    informative directions. NumPy's ``matrix_rank`` default scales with
    ``max(M, N) * eps`` instead, and a ``slogdet`` sign on an exactly deficient
    matrix is decided by rounding noise, so neither may disagree with this rule
    about the same matrix.

    ``unit_diagonal`` first rescales to ``D^-1/2 M D^-1/2`` with ``D`` the
    diagonal. Rank is invariant under that congruence, so the verdict no longer
    depends on the units of each parameter; information in a declared
    interest/nuisance parameterization is judged this way, while whitened
    coordinates already have a common scale. A zero diagonal entry keeps its
    zero row and counts as deficient.
    """
    if unit_diagonal:
        diagonal = jnp.diagonal(matrix)
        scale = jnp.where(diagonal > 0, 1 / jnp.sqrt(jnp.where(diagonal > 0, diagonal, 1)), 0)
        matrix = matrix * scale[:, None] * scale[None, :]
    eigenvalues = jnp.linalg.eigvalsh(matrix)
    maximum = float(np.asarray(jnp.max(eigenvalues)))
    if maximum <= 0:
        return 0
    resolved_rtol = _default_rank_rtol(matrix.dtype) if rank_rtol is None else rank_rtol
    return int(np.asarray(jnp.sum(eigenvalues > resolved_rtol * maximum)))


def is_numerically_singular(
    matrix: jnp.ndarray, rank_rtol: float | None = None, *, unit_diagonal: bool = False
) -> bool:
    """Return whether a symmetric matrix is rank deficient at the library threshold.

    Shared by every singularity guard so that one state cannot be called
    singular on one platform and regular on another: on an exactly deficient
    matrix the smallest eigenvalue is rounding noise, and its sign is a property
    of the host LAPACK rather than of the data. The relative threshold is scale
    free, so it agrees everywhere.
    """
    return numerical_rank(matrix, rank_rtol, unit_diagonal=unit_diagonal) < int(matrix.shape[0])


@dataclass(frozen=True, slots=True)
class FisherTransform:
    """Projection onto informative Fisher directions, optionally whitened.

    Scores are never mean-centered. ``matrix`` maps raw score vectors from
    shape ``[..., P]`` to optimization coordinates ``[..., R]``.

    Attributes
    ----------
    matrix
        Projection or whitening matrix with shape ``[P, R]``.
    eigenvectors, eigenvalues
        Complete eigendecomposition of the symmetrized input Fisher matrix.
    retained_eigenvalues
        Eigenvalues above the numerical rank threshold.
    rank_rtol, threshold
        Relative and absolute rank thresholds.
    whiten
        Whether retained directions are scaled by inverse square root eigenvalues.
    """

    matrix: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    retained_eigenvalues: np.ndarray
    rank_rtol: float
    threshold: float
    whiten: bool

    @property
    def input_dim(self) -> int:
        """Return the raw score dimension ``P``."""
        return int(self.matrix.shape[0])

    @property
    def rank(self) -> int:
        """Return the retained informative rank ``R``."""
        return int(self.matrix.shape[1])

    @property
    def dropped_directions(self) -> int:
        """Return the number of projected-out score directions."""
        return self.input_dim - self.rank

    @execution_scope
    def apply(
        self,
        scores: ArrayLike,
        *,
        execution: ExecutionConfig | None = None,
    ) -> np.ndarray:
        """Map raw scores into the fitted informative coordinate system."""
        del execution
        array = jnp.asarray(scores, dtype=self.matrix.dtype)
        if array.ndim != 2 or array.shape[1] != self.input_dim:
            raise ContractError(f"scores must have shape [N, {self.input_dim}], got {array.shape}")
        if not bool(np.asarray(jnp.all(jnp.isfinite(array)))):
            raise ContractError("scores must be finite")
        return canonical_array(array @ self.matrix)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible representation."""
        return json_ready(
            {
                "matrix": self.matrix,
                "eigenvectors": self.eigenvectors,
                "eigenvalues": self.eigenvalues,
                "retained_eigenvalues": self.retained_eigenvalues,
                "rank_rtol": self.rank_rtol,
                "threshold": self.threshold,
                "whiten": self.whiten,
                "rank": self.rank,
                "dropped_directions": self.dropped_directions,
            }
        )


@execution_scope
def fisher_transform(
    fisher: ArrayLike,
    *,
    whiten: bool = True,
    rank_rtol: float | None = None,
    execution: ExecutionConfig | None = None,
) -> FisherTransform:
    """Construct an informative-subspace transform from a Fisher matrix.

    Parameters
    ----------
    fisher
        Finite non-empty square Fisher matrix.
    whiten
        Scale retained eigenvectors by inverse square root eigenvalues.
    rank_rtol
        Relative threshold applied to the largest eigenvalue. A dtype-aware
        default is used when omitted.

    Returns
    -------
    FisherTransform
        Projection metadata and the matrix mapping scores into optimization
        coordinates.
    """
    del execution
    matrix = jnp.asarray(fisher)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ContractError("fisher must be a non-empty square matrix")
    if not bool(np.asarray(jnp.all(jnp.isfinite(matrix)))):
        raise ContractError("fisher must be finite")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    maximum = float(np.asarray(jnp.max(eigenvalues)))
    if maximum <= 0:
        raise ContractError("Fisher information has no positive informative direction")
    validate_rank_rtol(rank_rtol)
    resolved_rtol = _default_rank_rtol(matrix.dtype) if rank_rtol is None else rank_rtol
    threshold = resolved_rtol * maximum
    keep = np.asarray(eigenvalues > threshold)
    if not np.any(keep):
        raise ContractError("rank threshold removes every Fisher direction")
    basis = eigenvectors[:, keep]
    retained = eigenvalues[keep]
    transform_matrix = basis / jnp.sqrt(retained)[None, :] if whiten else basis
    return canonicalize_public(
        FisherTransform(
            matrix=transform_matrix,
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
            retained_eigenvalues=retained,
            rank_rtol=float(resolved_rtol),
            threshold=float(threshold),
            whiten=whiten,
        )
    )
