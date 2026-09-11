r"""Explicit bounded global certification of finite D-optimal partitions.

A finite D exchange returns a locally exchange-stable labeling. Deciding
whether that labeling is globally optimal is a separate, exponential question,
so it is never answered implicitly: ``certify_partition`` is the only entry
point, it refuses inputs larger than its declared capacity, and it always
reports whether the search finished or ran out of budget.

The search is a depth-first branch and bound over restricted-growth labelings
with the *singleton-completion* bound. Assigning atoms in a fixed order, any
completion of the first ``t`` assignments is a coarsening of the current
partial cells together with one singleton cell per remaining atom, so

\[
    I_{\text{completion}} \preceq I_{\text{partial}} + R_t,
    \qquad R_t=\sum_{u\ge t} w_u\,y_u y_u^\top ,
\]

and Loewner monotonicity of \(\log\det\) makes
\(\log\det(I_{\text{partial}}+R_t)\) a valid ceiling that tightens with depth.
This implementation certifies D only. On a fixed regular block decomposition,
the profiled Schur complement is also Loewner-monotone; D-only support must not
be interpreted as a mathematical impossibility of profiled bounds. A profiled
search would need its own singular-block policy and validated objective bounds.

The tree search is sequential and its nodes are tiny, so the inner algebra is
plain NumPy in float64: dispatching each node through JAX would dominate the
cost, and float64 keeps the bound trustworthy under both x64 settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from ._errors import ContractError
from ._execution import canonicalize_public, execution_scope, scatter_set
from ._execution import xp as jnp
from ._json import json_ready
from ._typing import ArrayLike, JsonValue
from .config import DExchangeConfig, ExecutionConfig, _validate_finite, _validate_integer
from .criteria import DOptimality, ProfiledDOptimality
from .partition import (
    _cell_statistics,
    _collapsed_initial_labels,
    _DObjective,
    _metric_assign,
    _optimize_labels,
    _prepare_partition,
    _PreparedPartition,
    _require_d_bin_budget,
)
from .reports import PartitionCertificate

# The search recursion is one frame per atom, so the capacity guard also keeps
# the deepest supported instance far inside the interpreter recursion limit.
_MAX_SUPPORTED_ROWS = 512


@dataclass(frozen=True, slots=True)
class CertificationConfig:
    """Configure explicit branch-and-bound certification of a D partition.

    Parameters
    ----------
    max_nodes
        Maximum number of search nodes. Reaching it stops the search and
        downgrades the certificate to ``status="budget_exhausted"`` with a
        genuine outstanding upper bound; it never turns a partial search into a
        claim of optimality.
    max_rows
        Maximum number of distinct positive-weight score atoms. Global
        certification is exponential in this count, so the guard refuses an
        oversized instance by name instead of appearing to hang. It may not
        exceed 512, the depth the search recursion supports.
    gain_tolerance
        Slack used both to prune a subtree and to accept a new incumbent, so
        ``status="optimal"`` means no labeling beats the reported objective by
        more than this much.
    """

    method: Literal["branch_and_bound"] = field(default="branch_and_bound", init=False)
    max_nodes: int = 2_000_000
    max_rows: int = 64
    gain_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        """Validate the bounded-search capacity contract at construction time."""
        _validate_integer("max_nodes", self.max_nodes, minimum=1)
        _validate_integer("max_rows", self.max_rows, minimum=1)
        if self.max_rows > _MAX_SUPPORTED_ROWS:
            raise ContractError(f"max_rows must be at most {_MAX_SUPPORTED_ROWS}")
        _validate_finite("gain_tolerance", self.gain_tolerance, positive=False)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible configuration mapping."""
        return json_ready(asdict(self))


@execution_scope
def certify_partition(
    scores: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    n_bins: int,
    incumbent: ArrayLike | None = None,
    criterion: DOptimality | ProfiledDOptimality | None = None,
    rank_rtol: float | None = None,
    config: CertificationConfig | None = None,
    execution: ExecutionConfig | None = None,
) -> PartitionCertificate:
    """Prove or bound the global optimality of a finite D partition.

    The search starts from an incumbent so that pruning is effective from the
    first node. Supplying the labels of an exchange result therefore answers the
    practical question directly: did the exchange already find the global
    optimum of this weighted score table?

    Identical score rows are certified as one atom with their pooled weight,
    matching ``optimize_partition`` and the exhaustive test oracle. Their
    common label is what the certificate reports.

    Parameters
    ----------
    scores
        Finite score matrix with shape ``[N, P]``.
    weights
        Optional finite, nonnegative weights with shape ``[N]``.
    n_bins
        Number of cells to certify.
    incumbent
        Optional starting labeling with shape ``[N]`` and values in
        ``[0, n_bins)``, normally ``PartitionResult.labels``. When omitted, one
        default D exchange supplies it. Identical score rows must already agree
        on their bin, and every cell must stay nonempty.
    criterion
        ``DOptimality`` by default and the only supported value.
    rank_rtol
        Relative threshold of the informative Fisher subspace, matching the
        configuration that produced the incumbent.
    config
        Search capacity and tolerance. Defaults to ``CertificationConfig()``.

    Returns
    -------
    PartitionCertificate
        Best labeling found, the global upper bound at termination, the
        remaining gap, and whether the tree was exhausted.

    Raises
    ------
    ContractError
        When the criterion is profiled, when the instance exceeds ``max_rows``,
        or when the inputs cannot support a regular ``n_bins``-cell partition.
    """
    del execution
    resolved_config = CertificationConfig() if config is None else config
    if not isinstance(resolved_config, CertificationConfig):
        raise TypeError("certify_partition requires CertificationConfig")
    resolved_criterion = DOptimality() if criterion is None else criterion
    if not isinstance(resolved_criterion, DOptimality):
        raise ContractError(
            "global certification supports DOptimality only; profiled certification "
            "requires its own singular-block policy and validated objective bounds"
        )
    exchange_config = DExchangeConfig(
        rank_rtol=rank_rtol, gain_tolerance=resolved_config.gain_tolerance
    )
    prepared = _prepare_partition(scores, weights, n_bins=n_bins, config=exchange_config)
    _require_d_bin_budget(prepared, n_bins)
    n_atoms = int(prepared.scores.shape[0])
    if n_atoms > resolved_config.max_rows:
        raise ContractError(
            f"certify_partition received {n_atoms} distinct positive-weight score "
            f"atoms, exceeding max_rows={resolved_config.max_rows}; global "
            "certification is exponential and is not attempted beyond that capacity"
        )

    coordinates = np.asarray(prepared.coordinates, dtype=np.float64)
    atom_weights = np.asarray(prepared.weights, dtype=np.float64)
    start_labels = _incumbent_labels(prepared, incumbent, n_bins, exchange_config)
    order = np.argsort(-np.einsum("nr,nr->n", coordinates, coordinates), kind="stable")

    search = _Search(
        coordinates=coordinates[order],
        weights=atom_weights[order],
        n_bins=n_bins,
        max_nodes=resolved_config.max_nodes,
        tolerance=resolved_config.gain_tolerance,
        labels=start_labels[order],
    )
    search.run()

    certified = np.empty(n_atoms, dtype=np.int64)
    certified[order] = search.best_labels
    return canonicalize_public(_certificate(prepared, search, certified, n_bins))


def _incumbent_labels(
    prepared: _PreparedPartition,
    incumbent: ArrayLike | None,
    n_bins: int,
    config: DExchangeConfig,
) -> np.ndarray:
    """Return one atom labeling to seed the incumbent of the search."""
    if incumbent is not None:
        collapsed = _collapsed_initial_labels(prepared, incumbent, n_bins)
        if collapsed is None:  # pragma: no cover - defensive, incumbent is not None
            raise ContractError("incumbent labels could not be reduced to score atoms")
        return np.asarray(collapsed, dtype=np.int64)
    run = _optimize_labels(
        points=prepared.coordinates,
        coordinates=prepared.coordinates,
        weights=prepared.weights,
        n_bins=n_bins,
        objective=_DObjective(),
        config=config,
    )
    return np.asarray(run.labels, dtype=np.int64)


def _certificate(
    prepared: _PreparedPartition,
    search: _Search,
    certified: np.ndarray,
    n_bins: int,
) -> PartitionCertificate:
    """Expand certified atom labels to input rows and assemble the certificate."""
    if not np.isfinite(search.best_objective):
        raise ContractError(
            "certification found no regular n_bins-cell partition; the incumbent is "
            "singular and the node budget stopped the search before it found one"
        )
    state = _DObjective().init_state(
        _cell_statistics(prepared.coordinates, prepared.weights, jnp.asarray(certified), n_bins)
    )
    sample = prepared.sample
    labels = _metric_assign(prepared.transform.apply(sample.scores), state.means, state.inverse)
    labels = scatter_set(
        labels,
        sample.positive_weight_mask,
        jnp.asarray(certified)[prepared.inverse_rows],
    )
    objective = search.best_objective
    upper_bound = max(objective, search.outstanding) if search.capped else objective
    return PartitionCertificate(
        status="budget_exhausted" if search.capped else "optimal",
        objective=objective,
        labels=labels,
        upper_bound=upper_bound,
        gap=max(upper_bound - objective, 0.0),
        nodes_explored=search.nodes,
        incumbent_was_optimal=not search.capped and not search.improved,
    )


class _Search:
    """Depth-first branch and bound with the singleton-completion upper bound.

    Cell moments are updated in place and restored by assignment rather than by
    subtraction, so a deep search cannot accumulate the drift that repeated
    add-then-subtract updates would introduce into the bound.
    """

    __slots__ = (
        "_cell_sums",
        "_cell_weights",
        "_suffix",
        "_zero",
        "best_labels",
        "best_objective",
        "capped",
        "coordinates",
        "improved",
        "labels",
        "max_nodes",
        "n_bins",
        "nodes",
        "outstanding",
        "tolerance",
        "weights",
    )

    def __init__(
        self,
        *,
        coordinates: np.ndarray,
        weights: np.ndarray,
        n_bins: int,
        max_nodes: int,
        tolerance: float,
        labels: np.ndarray,
    ) -> None:
        self.coordinates = coordinates
        self.weights = weights
        self.n_bins = n_bins
        self.max_nodes = max_nodes
        self.tolerance = tolerance
        self.labels = labels
        self.nodes = 0
        self.capped = False
        self.improved = False
        self.outstanding = -np.inf

        n_rows, rank = coordinates.shape
        suffix = np.zeros((n_rows + 1, rank, rank))
        for position in range(n_rows - 1, -1, -1):
            point = coordinates[position]
            suffix[position] = suffix[position + 1] + weights[position] * np.outer(point, point)
        self._suffix = suffix
        self._cell_weights = np.zeros(n_bins)
        self._cell_sums = np.zeros((n_bins, rank))
        self._zero = np.zeros((rank, rank))
        self.best_labels = labels.copy()
        self.best_objective = _labels_objective(coordinates, weights, labels, n_bins)

    def run(self) -> None:
        """Explore the tree from the empty assignment."""
        self._explore(0, 0, self._bound(self._suffix[0]))

    def _partial_information(self) -> np.ndarray:
        """Return the binned information of the cells filled so far."""
        information = np.zeros_like(self._zero)
        for cell in range(self.n_bins):
            mass = self._cell_weights[cell]
            if mass > 0:
                moment = self._cell_sums[cell]
                information += np.outer(moment, moment) / mass
        return information

    def _bound(self, extra: np.ndarray) -> float:
        """Return the log determinant of the partial information plus a remainder."""
        sign, logdet = np.linalg.slogdet(self._partial_information() + extra)
        return float(logdet) if sign > 0 else -np.inf

    def _explore(self, position: int, used: int, bound: float) -> None:
        """Expand one node, or record its bound when the budget is gone."""
        self.nodes += 1
        if self.capped or self.nodes > self.max_nodes:
            self.capped = True
            self.outstanding = max(self.outstanding, bound)
            return
        if position == self.coordinates.shape[0]:
            if used == self.n_bins:
                objective = self._bound(self._zero)
                if objective > self.best_objective + self.tolerance:
                    self.best_objective = objective
                    self.best_labels = self.labels.copy()
                    self.improved = True
            return

        weight = self.weights[position]
        point = self.coordinates[position]
        children: list[tuple[float, int, int]] = []
        # Only the first unused cell can start a new one, which visits every
        # partition exactly once instead of once per relabeling of its cells.
        for cell in range(min(used + 1, self.n_bins)):
            opened = used + (1 if cell == used else 0)
            if self.n_bins - opened > self.coordinates.shape[0] - position - 1:
                continue
            restore = self._assign(cell, weight, point)
            child = self._bound(self._suffix[position + 1])
            self._restore(cell, restore)
            if child > self.best_objective + self.tolerance:
                children.append((child, cell, opened))

        # Best-first descent finds a strong incumbent early, which is what makes
        # the sibling re-check below prune most of the remaining tree.
        children.sort(reverse=True)
        for child, cell, opened in children:
            if self.capped:
                self.outstanding = max(self.outstanding, child)
                continue
            if child <= self.best_objective + self.tolerance:
                continue
            restore = self._assign(cell, weight, point)
            self.labels[position] = cell
            self._explore(position + 1, opened, child)
            self._restore(cell, restore)

    def _assign(self, cell: int, weight: float, point: np.ndarray) -> tuple[float, np.ndarray]:
        """Add one atom to a cell and return the exact state to restore."""
        previous = (float(self._cell_weights[cell]), self._cell_sums[cell].copy())
        self._cell_weights[cell] = previous[0] + weight
        self._cell_sums[cell] = previous[1] + weight * point
        return previous

    def _restore(self, cell: int, previous: tuple[float, np.ndarray]) -> None:
        """Undo one assignment exactly."""
        self._cell_weights[cell] = previous[0]
        self._cell_sums[cell] = previous[1]


def _labels_objective(
    coordinates: np.ndarray, weights: np.ndarray, labels: np.ndarray, n_bins: int
) -> float:
    """Return the exact whitened log determinant of one complete labeling."""
    masses = np.bincount(labels, weights=weights, minlength=n_bins)[:n_bins]
    if float(masses.min()) <= 0:
        return -np.inf
    sums = np.zeros((n_bins, coordinates.shape[1]))
    np.add.at(sums, labels, weights[:, None] * coordinates)
    scaled = sums / np.sqrt(masses)[:, None]
    sign, logdet = np.linalg.slogdet(scaled.T @ scaled)
    return float(logdet) if sign > 0 else -np.inf
