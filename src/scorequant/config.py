"""Public optimizer configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from ._errors import ContractError
from ._json import json_ready
from ._typing import DTypeLike, JsonValue

type BackendName = Literal["jax", "numpy"]
type PrecisionPolicy = Literal["preserve", "float32", "float64"]
type DeviceKind = Literal["default", "cpu", "gpu", "tpu"]


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Select the numerical runtime without changing statistical semantics.

    Parameters
    ----------
    backend
        Numerical implementation. ``"jax"`` remains the default; ``"numpy"``
        is the portable CPU implementation used by Pyodide.
    precision
        Input precision policy. ``"preserve"`` retains float32/float64 inputs
        while promoting integer and lower-precision inputs to float32. JAX
        retains float64 only when X64 is enabled; otherwise it computes in
        float32 and warns once per process. Default tolerances follow the
        working dtype: ``gain_tolerance`` is 1e-10 in float64 and about 1.2e-6
        in float32, the rank threshold 1e-10 and about 1.2e-5.
    device
        Requested device family. NumPy accepts only ``"default"`` and
        ``"cpu"``. JAX resolves the first available device of the requested
        family and never changes global JAX configuration.
    """

    backend: BackendName = "jax"
    precision: PrecisionPolicy = "preserve"
    device: DeviceKind = "default"

    def __post_init__(self) -> None:
        """Validate backend, precision, and device compatibility eagerly."""
        if self.backend not in ("jax", "numpy"):
            raise ContractError("backend must be 'jax' or 'numpy'")
        if self.precision not in ("preserve", "float32", "float64"):
            raise ContractError("precision must be 'preserve', 'float32', or 'float64'")
        if self.device not in ("default", "cpu", "gpu", "tpu"):
            raise ContractError("device must be 'default', 'cpu', 'gpu', or 'tpu'")
        if self.backend == "numpy" and self.device not in ("default", "cpu"):
            raise ContractError("the NumPy backend supports only the default CPU device")

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible execution record."""
        return json_ready(asdict(self))


def _validate_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _validate_integer(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ContractError(f"{name} must be at least {minimum}")


def _validate_finite(name: str, value: object, *, positive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value):
        raise ContractError(f"{name} must be finite")
    if positive and value <= 0:
        raise ContractError(f"{name} must be positive")
    if not positive and value < 0:
        raise ContractError(f"{name} must be nonnegative")


FLOAT64_GAIN_TOLERANCE = 1e-10
"""Default ``gain_tolerance`` of a float64 run; other dtypes raise it to their noise floor."""

GAIN_EPS_MULTIPLE = 10
"""Noise floor of a log-determinant gain, in units of the working epsilon.

A gain is the logarithm of a determinant ratio near one, so its rounding is a
few epsilon; below that a float32 gain rounds to exactly zero.
"""


def dtype_tolerance(float64_value: float, dtype: DTypeLike, eps_multiple: float) -> float:
    """Raise a float64 default tolerance to the noise floor of the working dtype.

    Returns ``max(float64_value, eps_multiple * eps)``. Float64 defaults sit far
    above that floor and come back unchanged; in float32 the floor decides, so a
    stability or geometry verdict is stated just above the precision the
    arithmetic has and no real gain above it is left unused.
    """
    return max(float64_value, eps_multiple * float(np.finfo(np.dtype(dtype)).eps))


def resolve_gain_tolerance(gain_tolerance: float | None, dtype: DTypeLike) -> float:
    """Return an explicit ``gain_tolerance`` unchanged, or the working dtype's default."""
    if gain_tolerance is None:
        return dtype_tolerance(FLOAT64_GAIN_TOLERANCE, dtype, GAIN_EPS_MULTIPLE)
    return gain_tolerance


def validate_rank_rtol(rank_rtol: float | None) -> float | None:
    """Validate a relative rank threshold: ``None`` (dtype default) or a finite value in [0, 1)."""
    if rank_rtol is not None:
        _validate_finite("rank_rtol", rank_rtol, positive=False)
        if rank_rtol >= 1:
            raise ContractError("rank_rtol must be less than one")
    return rank_rtol


def _validate_common_config(
    *,
    whiten: bool,
    rank_rtol: float | None,
    seed: int,
    restarts: int,
    restarts_field: str,
    tolerance: float,
    tolerance_field: str,
    record_every: int,
) -> None:
    _validate_bool("whiten", whiten)
    validate_rank_rtol(rank_rtol)
    _validate_integer("seed", seed, minimum=0)
    _validate_integer(restarts_field, restarts, minimum=1)
    _validate_finite(tolerance_field, tolerance, positive=False)
    _validate_integer("record_every", record_every, minimum=1)


@dataclass(frozen=True, slots=True)
class KMeansConfig:
    """Configure deterministic weighted score-space k-means.

    Parameters
    ----------
    whiten
        Whiten retained Fisher directions before computing distances.
    rank_rtol
        Relative eigenvalue threshold for informative-rank selection. ``None``
        uses the working dtype's default, 1e-10 in float64 and about 1.2e-5
        in float32.
    seed
        Nonnegative JAX random seed.
    solver_restarts
        Number of weighted k-means++ restarts after invariant row ordering.
        This is k-means' own restart count: the run with the lowest final
        objective wins.
    max_iter
        Maximum Lloyd iterations per restart.
    tolerance
        Relative objective-change convergence tolerance.
    record_every
        Number of Lloyd iterations between trace snapshots.
    """

    method: Literal["kmeans"] = field(default="kmeans", init=False)
    whiten: bool = True
    rank_rtol: float | None = None
    seed: int = 0
    solver_restarts: int = 8
    max_iter: int = 100
    tolerance: float = 1e-6
    record_every: int = 1

    def __post_init__(self) -> None:
        """Validate the complete configuration at construction time."""
        _validate_common_config(
            whiten=self.whiten,
            rank_rtol=self.rank_rtol,
            seed=self.seed,
            restarts=self.solver_restarts,
            restarts_field="solver_restarts",
            tolerance=self.tolerance,
            tolerance_field="tolerance",
            record_every=self.record_every,
        )
        _validate_integer("max_iter", self.max_iter, minimum=1)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible configuration mapping."""
        return json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class SoftVoronoiConfig:
    """Configure differentiable D-optimal soft Voronoi fitting.

    Parameters
    ----------
    whiten
        Whiten retained Fisher directions before computing distances.
    rank_rtol
        Relative eigenvalue threshold for informative-rank selection. ``None``
        uses the working dtype's default, 1e-10 in float64 and about 1.2e-5
        in float32.
    seed
        Nonnegative JAX random seed.
    initializer_restarts
        Number of k-means restarts used to initialize the centers.
    kmeans_max_iter
        Maximum Lloyd iterations for each initialization restart.
    kmeans_tolerance
        Relative objective-change convergence tolerance of each initialization
        restart.
    max_iter
        Number of Adam updates. The annealing schedule always runs all of
        them: its temperature reaches ``temperature_end_ratio`` at the last.
    learning_rate
        Adam learning rate before scaling by initial center separation.
    gradient_clip
        Global gradient-norm clipping threshold.
    temperature_end_ratio
        Final temperature divided by the initial temperature.
    record_every
        Number of Adam steps between trace snapshots.
    """

    method: Literal["soft_voronoi"] = field(default="soft_voronoi", init=False)
    whiten: bool = True
    rank_rtol: float | None = None
    seed: int = 0
    initializer_restarts: int = 8
    kmeans_max_iter: int = 100
    kmeans_tolerance: float = 1e-6
    max_iter: int = 1000
    learning_rate: float = 1e-2
    gradient_clip: float = 10.0
    temperature_end_ratio: float = 0.05
    record_every: int = 10

    def __post_init__(self) -> None:
        """Validate the complete configuration at construction time."""
        _validate_common_config(
            whiten=self.whiten,
            rank_rtol=self.rank_rtol,
            seed=self.seed,
            restarts=self.initializer_restarts,
            restarts_field="initializer_restarts",
            tolerance=self.kmeans_tolerance,
            tolerance_field="kmeans_tolerance",
            record_every=self.record_every,
        )
        _validate_integer("kmeans_max_iter", self.kmeans_max_iter, minimum=1)
        _validate_integer("max_iter", self.max_iter, minimum=1)
        _validate_finite("learning_rate", self.learning_rate, positive=True)
        _validate_finite("gradient_clip", self.gradient_clip, positive=True)
        _validate_finite("temperature_end_ratio", self.temperature_end_ratio, positive=True)
        if self.temperature_end_ratio > 1:
            raise ContractError("temperature_end_ratio must be at most one")

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible configuration mapping."""
        return json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class DExchangeConfig:
    """Configure exact positive-gain D-optimal point exchange.

    One *scan* is one complete evaluation of every admissible single-row
    relocation. A scan accepts either one move or, under ``batch_moves``, many
    relocations at once, so ``accepted_moves`` and ``scans`` are different
    quantities.

    Parameters
    ----------
    rank_rtol
        Relative threshold for the informative Fisher subspace. ``None`` uses the
        working dtype's default, 1e-10 in float64 and about 1.2e-5 in float32.
    seed
        Base seed of deterministic initialization. Exchange restart ``r`` uses
        ``seed + r``.
    initializer_restarts
        Number of k-means seeding restarts inside one exchange restart. This
        nests inside ``solver_restarts``: the total seeding work is the
        product of the two.
    max_iter
        Maximum number of scans, one complete candidate scan per iteration;
        ``PartitionResult.scans`` counts them. ``None`` runs until the
        exchange is stable, which strict positive-gain acceptance guarantees;
        a generous internal safety bound still stops a numerically pathological
        run and records ``best_remaining_gain``.
    batch_moves
        Relocate many positive-gain rows per scan. The batch is ranked by gain,
        truncated before it would displace too much of any cell's weight, and
        accepted only when the exactly recomputed objective strictly improves;
        a rejected batch is halved and retried, finally falling back to the
        single best move. Small improving sets always use that single exact
        move, so results match ``batch_moves=False`` on small samples. Ignored
        when ``first_improvement`` is set.
    solver_restarts
        Number of independent exchange restarts. The restart with the best
        exact final objective wins, ties resolving to the earliest restart, and
        every reported diagnostic describes that winning restart.
    init
        Initial labeling of each restart: weighted k-means++ seeding or a
        balanced random labeling.
    gain_tolerance
        Strict minimum accepted log-determinant gain. ``None`` (default) uses
        the working dtype's default, 1e-10 in float64 and about 1.2e-6 (ten
        epsilon) in float32, so exchange stability and the geometry certificate
        sit just above rounding noise; the result's ``config`` records the resolved value. An
        explicit value is used unchanged.
    first_improvement
        Accept the first improving move in deterministic row/bin order instead
        of the best move in a scan. This stops each scan early, so it forces
        single-move acceptance and disables ``batch_moves``.
    collapse_duplicates
        Merge identical score rows into one weighted atom before solving.
        ``None`` (default) collapses automatically below 100,000 effective
        rows and skips it above, since the O(N log N) merge only pays for
        itself against the O(N) exchange scan it speeds up on samples with
        many repeated score atoms. ``True``/``False`` always collapses or
        always skips, regardless of sample size.
    """

    method: Literal["d_exchange"] = field(default="d_exchange", init=False)
    rank_rtol: float | None = None
    seed: int = 0
    initializer_restarts: int = 8
    max_iter: int | None = None
    batch_moves: bool = True
    solver_restarts: int = 1
    init: Literal["kmeans++", "random"] = "kmeans++"
    gain_tolerance: float | None = None
    first_improvement: bool = False
    collapse_duplicates: bool | None = None

    def __post_init__(self) -> None:
        """Validate exchange settings at construction time."""
        validate_rank_rtol(self.rank_rtol)
        _validate_integer("seed", self.seed, minimum=0)
        _validate_integer("initializer_restarts", self.initializer_restarts, minimum=1)
        if self.max_iter is not None:
            _validate_integer("max_iter", self.max_iter, minimum=1)
        _validate_bool("batch_moves", self.batch_moves)
        _validate_integer("solver_restarts", self.solver_restarts, minimum=1)
        if self.init not in ("kmeans++", "random"):
            raise ContractError("init must be 'kmeans++' or 'random'")
        if self.gain_tolerance is not None:
            _validate_finite("gain_tolerance", self.gain_tolerance, positive=False)
        _validate_bool("first_improvement", self.first_improvement)
        if self.collapse_duplicates is not None:
            _validate_bool("collapse_duplicates", self.collapse_duplicates)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible configuration mapping."""
        return json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class MahalanobisLloydConfig:
    """Configure guarded batch Mahalanobis-Lloyd iteration.

    One *iteration* freezes the current criterion metric, proposes the complete
    nearest-centroid relabeling of every row in that metric, and accepts the
    proposal only when the exactly rebuilt objective strictly improves. The
    unguarded batch step is not monotone: the tangent of the concave log
    determinant is an upper bound, not a minorizer, and a committed eight-row
    fixture loses 0.136521 nat in one such step. The guard is therefore part of
    the solver contract, not an optional safeguard.

    Parameters
    ----------
    rank_rtol
        Relative threshold for the informative Fisher subspace. ``None`` uses the
        working dtype's default, 1e-10 in float64 and about 1.2e-5 in float32.
    seed
        Nonnegative seed of the deterministic k-means seeding.
    initializer_restarts
        Number of weighted k-means++ restarts used to seed the labels.
    max_iter
        Maximum number of guarded batch iterations.
    guard
        Behavior once a proposal stops improving. ``"exchange"`` hands the
        labels to the exact positive-gain exchange engine, so the reported
        state is exchange-stable and a nonsingular D result stays compilable.
        ``"reject"`` stops at the last accepted labeling and only certifies
        exchange stability with one final scan, which may report ``False``.
    gain_tolerance
        Strict minimum accepted objective gain, shared by the guarded batch and
        the exchange phase. ``None`` (default) uses the working dtype's
        default, as ``DExchangeConfig`` does; an explicit value is used
        unchanged.
    collapse_duplicates
        Merge identical score rows into one weighted atom before solving.
        ``None`` (default) collapses automatically below 100,000 effective
        rows and skips it above, since the O(N log N) merge only pays for
        itself against the O(N) work it speeds up on samples with many
        repeated score atoms. ``True``/``False`` always collapses or always
        skips, regardless of sample size.
    """

    method: Literal["mahalanobis_lloyd"] = field(default="mahalanobis_lloyd", init=False)
    rank_rtol: float | None = None
    seed: int = 0
    initializer_restarts: int = 8
    max_iter: int = 100
    guard: Literal["exchange", "reject"] = "exchange"
    gain_tolerance: float | None = None
    collapse_duplicates: bool | None = None

    def __post_init__(self) -> None:
        """Validate the guarded batch settings at construction time."""
        validate_rank_rtol(self.rank_rtol)
        _validate_integer("seed", self.seed, minimum=0)
        _validate_integer("initializer_restarts", self.initializer_restarts, minimum=1)
        _validate_integer("max_iter", self.max_iter, minimum=1)
        if self.guard not in ("exchange", "reject"):
            raise ContractError("guard must be 'exchange' or 'reject'")
        if self.gain_tolerance is not None:
            _validate_finite("gain_tolerance", self.gain_tolerance, positive=False)
        if self.collapse_duplicates is not None:
            _validate_bool("collapse_duplicates", self.collapse_duplicates)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible configuration mapping."""
        return json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class ScalarDPConfig:
    """Configure exact interval dynamic programming for one score coordinate.

    The solver is exact and deterministic: on a rank-one score law the optimal
    hard partition has ordered interval cells, and the weighted interval dynamic
    program returns the global optimum rather than a local one.

    Parameters
    ----------
    whiten
        Whiten the single retained Fisher direction before solving. Scalar
        whitening is a strictly positive rescaling, so it never changes the
        optimal interval labels; it fixes the units of the reported
        within-segment objective and matches the other score-space solvers.
    rank_rtol
        Relative threshold for the informative score direction. ``None`` uses the
        working dtype's default, 1e-10 in float64 and about 1.2e-5 in float32.
    max_rows
        Maximum number of distinct positive-weight score atoms. The exact
        dynamic program uses quadratic work in this count and is evaluated in
        memory-bounded stripes.
    """

    method: Literal["scalar_dp"] = field(default="scalar_dp", init=False)
    whiten: bool = True
    rank_rtol: float | None = None
    max_rows: int = 20_000

    def __post_init__(self) -> None:
        """Validate the exact-solver capacity contract."""
        _validate_bool("whiten", self.whiten)
        validate_rank_rtol(self.rank_rtol)
        _validate_integer("max_rows", self.max_rows, minimum=1)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible configuration mapping."""
        return json_ready(asdict(self))


type PartitionConfig = DExchangeConfig | MahalanobisLloydConfig
type QuantizerConfig = (
    KMeansConfig | SoftVoronoiConfig | DExchangeConfig | MahalanobisLloydConfig | ScalarDPConfig
)
