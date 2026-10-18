"""Linear-intensity models: the component score provider and its score map."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ._errors import ContractError
from ._execution import canonical_array, execution_scope, input_array
from ._execution import xp as jnp
from ._typing import ArrayLike
from ._validation import promote_low_precision
from .config import ExecutionConfig
from .sources import ScoreProvenance, ScoreSchema

ComponentFunction = Callable[[np.ndarray], ArrayLike]


def _names(
    names: Sequence[str] | None,
    size: int,
    *,
    prefix: str,
) -> tuple[str, ...]:
    resolved = (
        tuple(f"{prefix}_{index}" for index in range(size)) if names is None else tuple(names)
    )
    if len(resolved) != size:
        raise ContractError(f"expected {size} {prefix} names, got {len(resolved)}")
    if any(not isinstance(name, str) or not name for name in resolved):
        raise ContractError(f"{prefix} names must be non-empty strings")
    if len(set(resolved)) != len(resolved):
        raise ContractError(f"{prefix} names must be unique")
    return resolved


@execution_scope
def _scores_from_components(
    components: ArrayLike,
    coefficients: ArrayLike,
    *,
    execution: ExecutionConfig | None = None,
) -> np.ndarray:
    """Construct scores for a linear intensity model.

    Parameters
    ----------
    components
        Finite component matrix with shape ``[N, M]``.
    coefficients
        Finite reference coefficients with shape ``[M]``.

    Returns
    -------
    numpy.ndarray
        Score matrix ``components / (components @ coefficients)[:, None]``
        with shape ``[N, M]``.

    Raises
    ------
    ContractError
        If shapes are incompatible, values are non-finite, or the reference
        intensity is not strictly positive at every row.

    Notes
    -----
    Components and coefficients may be signed and need not be normalized.
    They must be finite, and their resulting reference intensity must be
    strictly positive at every supplied integration point.

    The map is invariant under a common event-wise rescaling of a row: for
    any positive ``c(x)``, replacing ``components[i, :]`` by
    ``c(x_i) * components[i, :]`` leaves the scores unchanged. Component
    densities and component density *ratios* in any gauge therefore produce
    identical scores; absolute normalization is never required.
    """
    del execution
    component_array = input_array(components)
    if component_array.ndim != 2 or min(component_array.shape) == 0:
        raise ContractError("components must have non-empty shape [N, M]")
    component_array = promote_low_precision(component_array)
    coefficient_array = jnp.asarray(coefficients, dtype=component_array.dtype)
    if coefficient_array.shape != (component_array.shape[1],):
        raise ContractError(
            f"coefficients must have shape [{component_array.shape[1]}], "
            f"got {coefficient_array.shape}"
        )
    if not bool(np.asarray(jnp.all(jnp.isfinite(component_array)))):
        raise ContractError("components must be finite")
    if not bool(np.asarray(jnp.all(jnp.isfinite(coefficient_array)))):
        raise ContractError("coefficients must be finite")
    intensity = component_array @ coefficient_array
    if bool(np.asarray(jnp.any(~jnp.isfinite(intensity)))) or bool(
        np.asarray(jnp.any(intensity <= 0))
    ):
        raise ContractError("reference intensity must be finite and strictly positive at every row")
    return canonical_array(component_array / intensity[:, None])


@dataclass(frozen=True, slots=True, init=False)
class LinearComponentScore:
    """Score observations under a linear-intensity model of named components.

    The model is ``lambda(x; theta) = sum_k theta_k phi_k(x)``, and its score
    at the reference coefficients ``theta_0`` is
    ``phi_k(x) / lambda(x; theta_0)``, one column per component. Components
    and coefficients may be signed and need not be normalized; the reference
    intensity must be strictly positive at every evaluated row.

    Parameters
    ----------
    components
        Either an insertion-ordered mapping from names to callables or a
        sequence of callables. Each callable receives ``X`` with shape
        ``[N, K]`` and returns one finite value per row.
    coefficients
        Reference coefficients ``theta_0``: a mapping with exactly the
        component keys, or a sequence aligned with sequence components.
    variables
        Optional unique physical-variable names used to validate ``K``.
    provenance
        Score provenance. Exact by default, because the components are the
        model itself rather than an estimate of it.

    Notes
    -----
    Already evaluated component values are density ratios in the Lebesgue
    gauge, so ``scorequant.ratios.IntensityParameterization(coefficients)``
    maps them to the same scores.
    """

    components: tuple[ComponentFunction, ...]
    coefficients: tuple[float, ...]
    component_names: tuple[str, ...]
    variables: tuple[str, ...] | None
    provenance: ScoreProvenance

    def __init__(
        self,
        components: Mapping[str, ComponentFunction] | Sequence[ComponentFunction],
        coefficients: Mapping[str, float] | Sequence[float],
        *,
        variables: Sequence[str] | None = None,
        provenance: ScoreProvenance | None = None,
    ) -> None:
        """Validate and freeze component order, coefficients, and metadata."""
        if isinstance(components, Mapping):
            if not isinstance(coefficients, Mapping):
                raise TypeError("named components require a coefficient mapping")
            component_names = tuple(components.keys())
            if set(coefficients) != set(component_names):
                missing = sorted(set(component_names) - set(coefficients))
                extra = sorted(set(coefficients) - set(component_names))
                raise ContractError(
                    f"coefficient keys must match component keys; missing={missing}, extra={extra}"
                )
            functions = tuple(components[name] for name in component_names)
            coefficient_values = tuple(coefficients[name] for name in component_names)
        else:
            if isinstance(coefficients, Mapping):
                raise TypeError("sequence components require a coefficient sequence")
            functions = tuple(components)
            coefficient_values = tuple(coefficients)
            component_names = tuple(f"component_{index}" for index in range(len(functions)))
        if not functions:
            raise ContractError("at least one component function is required")
        if len(coefficient_values) != len(functions):
            raise ContractError(
                f"expected {len(functions)} coefficients, got {len(coefficient_values)}"
            )
        if any(not callable(function) for function in functions):
            raise TypeError("every component must be callable")
        resolved_names = _names(component_names, len(functions), prefix="component")
        coefficient_array = np.asarray(coefficient_values, dtype=float)
        if coefficient_array.shape != (len(functions),):
            raise ContractError("coefficients must be a one-dimensional scalar sequence")
        if not np.isfinite(coefficient_array).all():
            raise ContractError("coefficients must be finite")
        variable_names = None if variables is None else tuple(variables)
        resolved_variables = (
            None
            if variable_names is None
            else _names(variable_names, len(variable_names), prefix="variable")
        )
        if provenance is None:
            provenance = ScoreProvenance(kind="exact", description="linear-component event score")
        elif not isinstance(provenance, ScoreProvenance):
            raise TypeError("provenance must be a ScoreProvenance")
        object.__setattr__(self, "components", functions)
        object.__setattr__(self, "coefficients", tuple(float(value) for value in coefficient_array))
        object.__setattr__(self, "component_names", resolved_names)
        object.__setattr__(self, "variables", resolved_variables)
        object.__setattr__(self, "provenance", provenance)

    @property
    def schema(self) -> ScoreSchema:
        """Name each score column after the component it differentiates.

        A linear-intensity model emits one score column per component and the
        component names are already declared, so the schema is derived rather
        than asked for.
        """
        return ScoreSchema(self.component_names)

    @execution_scope
    def evaluate_components(
        self,
        X: ArrayLike,
        *,
        execution: ExecutionConfig | None = None,
    ) -> np.ndarray:
        """Evaluate every component function.

        Parameters
        ----------
        X
            Finite numeric physical-variable matrix with shape ``[N, K]``.

        Returns
        -------
        numpy.ndarray
            Evaluated component matrix with shape ``[N, M]``.
        """
        del execution
        observations = np.asarray(X)
        if observations.ndim != 2 or min(observations.shape) == 0:
            raise ContractError("X must have non-empty shape [N, K]")
        if not np.issubdtype(observations.dtype, np.number) or not np.isfinite(observations).all():
            raise ContractError("X must be numeric and finite")
        if self.variables is not None and observations.shape[1] != len(self.variables):
            raise ContractError(
                f"X has {observations.shape[1]} variables, expected {len(self.variables)}"
            )
        columns: list[jnp.ndarray] = []
        for name, function in zip(self.component_names, self.components, strict=True):
            values = jnp.asarray(function(observations))
            if values.shape != (observations.shape[0],):
                raise ContractError(
                    f"component {name!r} must return shape [{observations.shape[0]}], "
                    f"got {values.shape}"
                )
            if not bool(np.asarray(jnp.all(jnp.isfinite(values)))):
                raise ContractError(f"component {name!r} returned a non-finite value")
            columns.append(values)
        return canonical_array(jnp.stack(columns, axis=1))

    @execution_scope
    def score(
        self,
        observations: ArrayLike,
        *,
        execution: ExecutionConfig | None = None,
    ) -> np.ndarray:
        """Evaluate the components and their reference score ``phi_k / lambda``."""
        components = self.evaluate_components(observations, execution=execution)
        return _scores_from_components(components, self.coefficients, execution=execution)
