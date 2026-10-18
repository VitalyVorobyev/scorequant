"""Private execution resolution and backend primitive adapters.

Backend selection is scoped with :mod:`contextvars`, so nested public calls
inherit one resolved runtime without process-global mutable state. Numerical
modules import ``xp`` instead of a concrete array namespace; backend-name
conditionals stay in this module.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import cache, wraps
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast, overload

import numpy as np

from ._typing import ArrayLike
from .config import DeviceKind, ExecutionConfig


@dataclass(frozen=True, slots=True)
class _Runtime:
    config: ExecutionConfig
    namespace: ModuleType
    device: object | None


_ACTIVE_RUNTIME: ContextVar[_Runtime | None] = ContextVar("scorequant_active_runtime", default=None)


def _resolve_runtime(config: ExecutionConfig) -> _Runtime:
    if config.backend == "numpy":
        return _Runtime(
            config=replace(config, device="cpu"),
            namespace=np,
            device=None,
        )

    try:
        import jax
        import jax.numpy as jax_numpy
    except ImportError as error:  # pragma: no cover - exercised in isolated import smoke test
        raise RuntimeError(
            "the JAX backend is unavailable; install ScoreQuant on CPython with its default "
            "dependencies or select ExecutionConfig(backend='numpy')"
        ) from error

    if config.precision == "float64" and not bool(jax.config.x64_enabled):
        raise RuntimeError(
            "float64 was requested for the JAX backend, but JAX_ENABLE_X64 is disabled; "
            "enable X64 before importing JAX or choose precision='float32'"
        )
    requested_platform = None if config.device == "default" else config.device
    devices = jax.devices(requested_platform)
    if not devices:
        raise RuntimeError(f"no JAX {config.device!r} device is available")
    device = devices[0]
    resolved = replace(config, device=cast(DeviceKind, device.platform))
    return _Runtime(config=resolved, namespace=jax_numpy, device=device)


@contextmanager
def use_execution(execution: ExecutionConfig | None) -> Iterator[ExecutionConfig]:
    """Enter a validated execution scope, inheriting an existing scope if omitted."""
    active = _ACTIVE_RUNTIME.get()
    if execution is None and active is not None:
        yield active.config
        return

    runtime = _resolve_runtime(execution or ExecutionConfig())
    token = _ACTIVE_RUNTIME.set(runtime)
    device_scope = nullcontext()
    if runtime.config.backend == "jax":
        import jax

        device_scope = jax.default_device(runtime.device)
    try:
        with device_scope:
            yield runtime.config
    finally:
        _ACTIVE_RUNTIME.reset(token)


def current_execution() -> ExecutionConfig:
    """Return the active resolved execution, defaulting lazily to JAX."""
    runtime = _ACTIVE_RUNTIME.get()
    if runtime is None:
        return _resolve_runtime(ExecutionConfig()).config
    return runtime.config


def _current_runtime() -> _Runtime:
    runtime = _ACTIVE_RUNTIME.get()
    return _resolve_runtime(ExecutionConfig()) if runtime is None else runtime


class _NamespaceProxy:
    """Resolve NumPy-compatible namespace attributes from the active runtime."""

    def __getattr__(self, name: str) -> object:
        return getattr(_current_runtime().namespace, name)


if TYPE_CHECKING:
    # Shared kernels use NumPy's typed API. At runtime the proxy resolves the
    # equivalent primitive from the active NumPy or JAX namespace.
    import numpy as xp
else:
    xp = _NamespaceProxy()


@cache
def _warn_float64_truncation() -> None:
    # Once per process: the fallback is a property of the JAX configuration,
    # not of one call, and repeating it on every array would bury it.
    warnings.warn(
        "float64 scores were converted to float32 because JAX_ENABLE_X64 is disabled; "
        "results carry float32 precision and default tolerances are relaxed to match. Enable "
        "X64 before importing JAX, pass float32 scores, or select ExecutionConfig(backend='numpy')",
        UserWarning,
        stacklevel=3,
    )


def input_array(array: ArrayLike) -> np.ndarray:
    """Convert a public numerical input to the active backend's array type.

    JAX without X64 truncates float64 at this conversion, so this is the last
    point where the caller's dtype is still visible and the fallback can warn.
    """
    runtime = _current_runtime()
    if runtime.config.backend == "jax" and runtime.config.precision == "preserve":
        import jax

        if not bool(jax.config.x64_enabled):
            source = getattr(array, "dtype", None)
            if source is None:
                source = np.asarray(array).dtype
            if source == np.float64:
                _warn_float64_truncation()
    return cast(np.ndarray, runtime.namespace.asarray(array))


def apply_precision(array: ArrayLike) -> np.ndarray:
    """Convert a numerical input according to the active precision policy."""
    runtime = _current_runtime()
    values = runtime.namespace.asarray(array)
    dtype = values.dtype
    if runtime.config.precision == "float32":
        return cast(np.ndarray, values.astype(runtime.namespace.float32))
    if runtime.config.precision == "float64":
        return cast(np.ndarray, values.astype(runtime.namespace.float64))
    if not runtime.namespace.issubdtype(dtype, runtime.namespace.inexact):
        return cast(np.ndarray, values.astype(runtime.namespace.float32))
    if dtype in (runtime.namespace.float16, getattr(runtime.namespace, "bfloat16", object())):
        return cast(np.ndarray, values.astype(runtime.namespace.float32))
    return cast(np.ndarray, values)


class _FunctionalUpdate(Protocol):
    def add(self, values: object) -> object: ...

    def set(self, values: object) -> object: ...


class _ArrayAtIndexer(Protocol):
    def __getitem__(self, indices: object) -> _FunctionalUpdate: ...


class _FunctionalArray(Protocol):
    @property
    def at(self) -> _ArrayAtIndexer: ...


def scatter_add(base: np.ndarray, indices: object, values: object) -> np.ndarray:
    """Return ``base`` with repeated-index additions applied functionally."""
    if _current_runtime().config.backend == "jax":
        return cast(np.ndarray, cast(_FunctionalArray, base).at[indices].add(values))
    result = np.array(base, copy=True)
    np.add.at(result, np.asarray(indices), np.asarray(values))
    return result


def scatter_set(base: np.ndarray, indices: object, values: object) -> np.ndarray:
    """Return ``base`` with selected entries replaced functionally."""
    if _current_runtime().config.backend == "jax":
        return cast(np.ndarray, cast(_FunctionalArray, base).at[indices].set(values))
    result = np.array(base, copy=True)
    result[np.asarray(indices)] = np.asarray(values)
    return result


def scatter_block_add(base: np.ndarray, row_indices: object, values: object) -> np.ndarray:
    """Add a square block selected by one index vector."""
    if _current_runtime().config.backend == "jax":
        namespace = _current_runtime().namespace
        selection = namespace.ix_(row_indices, row_indices)
        return cast(np.ndarray, cast(_FunctionalArray, base).at[selection].add(values))
    result = np.array(base, copy=True)
    selection = np.ix_(np.asarray(row_indices), np.asarray(row_indices))
    result[selection] += np.asarray(values)
    return result


type RandomSeed = int | tuple[int, int]


def _key_seed(key: object) -> tuple[int, int]:
    values = np.asarray(key, dtype=np.uint32)
    return int(values[0]), int(values[1])


def split_seeds(seed: RandomSeed, count: int) -> tuple[RandomSeed, ...]:
    """Return deterministic independent seeds for the active backend."""
    if _current_runtime().config.backend == "jax":
        import jax

        key = (
            jax.random.PRNGKey(seed) if isinstance(seed, int) else xp.asarray(seed, dtype=xp.uint32)
        )
        keys = jax.random.split(key, count)
        return tuple(_key_seed(key) for key in keys)
    sequence = np.random.SeedSequence(seed)
    return tuple(
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in sequence.spawn(count)
    )


def weighted_choice(seed: RandomSeed, probabilities: object) -> int:
    """Draw one categorical index with a backend-local deterministic seed."""
    values = np.asarray(probabilities, dtype=np.float64)
    values = values / np.sum(values)
    if _current_runtime().config.backend == "jax":
        import jax

        key = (
            jax.random.PRNGKey(seed) if isinstance(seed, int) else xp.asarray(seed, dtype=xp.uint32)
        )
        return int(np.asarray(jax.random.choice(key, values.shape[0], p=xp.asarray(values))))
    return int(np.random.default_rng(seed).choice(values.shape[0], p=values))


def random_permutation(seed: int, size: int) -> np.ndarray:
    """Return one backend-local deterministic permutation."""
    if _current_runtime().config.backend == "jax":
        import jax

        return cast(np.ndarray, jax.random.permutation(jax.random.PRNGKey(seed), size))
    return np.random.default_rng(seed).permutation(size).astype(np.int32)


class _OptimizerTransformation(Protocol):
    def init(self, parameters: np.ndarray) -> object: ...

    def update(
        self,
        gradients: np.ndarray,
        state: object,
        parameters: np.ndarray,
    ) -> tuple[np.ndarray, object]: ...


@dataclass(slots=True)
class AdamState:
    """Opaque backend-owned Adam state shared by the solver orchestration."""

    learning_rate: float
    gradient_clip: float
    step: int = 0
    first_moment: np.ndarray | None = None
    second_moment: np.ndarray | None = None
    transformation: _OptimizerTransformation | None = None
    backend_state: object | None = None


def create_adam(
    parameters: np.ndarray,
    *,
    learning_rate: float,
    gradient_clip: float,
) -> AdamState:
    """Create the active backend's private Adam state."""
    if _current_runtime().config.backend == "jax":
        import optax

        transformation = cast(
            _OptimizerTransformation,
            optax.chain(
                optax.clip_by_global_norm(gradient_clip),
                optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8),
            ),
        )
        return AdamState(
            learning_rate=learning_rate,
            gradient_clip=gradient_clip,
            transformation=transformation,
            backend_state=transformation.init(parameters),
        )
    values = np.asarray(parameters)
    return AdamState(
        learning_rate=learning_rate,
        gradient_clip=gradient_clip,
        first_moment=np.zeros_like(values),
        second_moment=np.zeros_like(values),
    )


def adam_update(
    parameters: np.ndarray,
    gradients: np.ndarray,
    state: AdamState,
) -> tuple[np.ndarray, AdamState, float]:
    """Apply clipping and one bias-corrected Adam update."""
    gradient_norm = float(np.linalg.norm(np.asarray(gradients)))
    if _current_runtime().config.backend == "jax":
        import optax

        transformation = state.transformation
        if transformation is None:
            raise RuntimeError("JAX Adam state is missing its transformation")
        updates, backend_state = transformation.update(
            gradients,
            state.backend_state,
            parameters,
        )
        state.backend_state = backend_state
        state.step += 1
        return cast(np.ndarray, optax.apply_updates(parameters, updates)), state, gradient_norm

    gradient = np.asarray(gradients)
    if gradient_norm > state.gradient_clip:
        gradient = gradient * (state.gradient_clip / gradient_norm)
    if state.first_moment is None or state.second_moment is None:
        raise RuntimeError("NumPy Adam state is uninitialized")
    first = 0.9 * np.asarray(state.first_moment) + 0.1 * gradient
    second = 0.999 * np.asarray(state.second_moment) + 0.001 * gradient**2
    state.step += 1
    first_hat = first / (1 - 0.9**state.step)
    second_hat = second / (1 - 0.999**state.step)
    updated = np.asarray(parameters) - state.learning_rate * first_hat / (
        np.sqrt(second_hat) + 1e-8
    )
    state.first_moment = first
    state.second_moment = second
    return updated, state, gradient_norm


@cache
def _compiled[**P, R](function: Callable[P, R], static_argnames: tuple[str, ...]) -> Callable[P, R]:
    import jax

    return jax.jit(function, static_argnames=static_argnames)


@overload
def backend_jit[**P, R](function: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def backend_jit[**P, R](
    function: None = None, /, *, static_argnames: tuple[str, ...]
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def backend_jit[**P, R](
    function: Callable[P, R] | None = None,
    /,
    *,
    static_argnames: tuple[str, ...] = (),
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Compile one shared kernel under JAX and call it directly under NumPy."""

    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        @wraps(target)
        def dispatched(*args: P.args, **kwargs: P.kwargs) -> R:
            if _current_runtime().config.backend == "jax":
                return _compiled(target, static_argnames)(*args, **kwargs)
            return target(*args, **kwargs)

        return dispatched

    return decorate(function) if function is not None else decorate


def execution_scope[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Resolve the ``execution`` keyword for one public numerical operation."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        execution = kwargs.get("execution")
        if execution is not None and not isinstance(execution, ExecutionConfig):
            raise TypeError("execution must be an ExecutionConfig or None")
        with use_execution(execution):
            return function(*args, **kwargs)

    return wrapped


def canonical_array(value: object) -> np.ndarray:
    """Copy one backend array into the stable NumPy API representation."""
    return np.asarray(value).copy()


def backend_array(value: np.ndarray) -> np.ndarray:
    """Place one canonical NumPy array on the active backend.

    The inverse of :func:`canonical_array`, for kernels re-entered from the
    NumPy arrays a public result stores. ``jax.device_put`` is a transfer
    rather than a staged primitive, so unlike ``jnp.asarray`` it costs no XLA
    compilation on the first call for a given shape. The device is left
    implicit: :func:`use_execution` is already inside
    ``jax.default_device(runtime.device)``, and naming the device here commits
    the array through a resharding path two orders of magnitude slower.
    """
    runtime = _current_runtime()
    if runtime.config.backend == "jax":
        import jax

        return cast(np.ndarray, jax.device_put(value))
    return np.asarray(value)


def canonicalize_public[T](value: T) -> T:
    """Recursively replace arrays in a public dataclass tree with NumPy arrays."""
    if isinstance(value, np.ndarray):
        return value.copy()  # type: ignore[return-value]
    value_type = type(value)
    if value_type.__module__.startswith("jax") or value_type.__module__.startswith("numpy"):
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return cast(T, np.asarray(value).copy())
    if is_dataclass(value) and not isinstance(value, type):
        for record_field in fields(value):
            current = getattr(value, record_field.name)
            canonical = canonicalize_public(current)
            if canonical is not current:
                object.__setattr__(value, record_field.name, canonical)
        return value
    if isinstance(value, tuple):
        return cast(T, tuple(canonicalize_public(item) for item in value))
    if isinstance(value, list):
        return cast(T, [canonicalize_public(item) for item in value])
    return value
