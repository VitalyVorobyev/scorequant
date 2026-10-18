"""Model density-ratio algebra: prior correction, ratio-to-score maps, closure.

For a parametric model ``p(x | theta)`` with reference point ``theta_0``, the
local score is the gradient of a log density *ratio*,

``s(x) = grad_theta log(p(x | theta) / p(x | theta_0)) at theta_0``,

so absolute densities are never required: any oracle for model density ratios
determines the score. The maps in this module convert ratios into the score
coordinates the optimizers consume. Importance ratios ``p_theta0(x) / g(x)``
are a different object — they reweight expectations and enter ScoreQuant as
source *weights*, never through these maps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._errors import ContractError
from ._execution import canonical_array, canonicalize_public, execution_scope, input_array
from ._execution import xp as jnp
from ._typing import ArrayLike
from ._validation import promote_low_precision, validate_weights
from .components import scores_from_components
from .config import ExecutionConfig
from .reports import RatioClosureReport

_SIMPLEX_RTOL = 1e-5
_SIMPLEX_ATOL = 1e-7


def _validate_simplex_vector(vector: jnp.ndarray, name: str, size: int) -> None:
    if vector.shape != (size,):
        raise ContractError(f"{name} must have shape [{size}], got {vector.shape}")
    if not bool(np.asarray(jnp.all(jnp.isfinite(vector)))):
        raise ContractError(f"{name} must be finite")
    if bool(np.asarray(jnp.any(vector <= 0))):
        raise ContractError(f"{name} must be strictly positive")
    if not bool(
        np.asarray(jnp.allclose(jnp.sum(vector), 1.0, rtol=_SIMPLEX_RTOL, atol=_SIMPLEX_ATOL))
    ):
        raise ContractError(f"{name} must sum to one")


def _validate_ratio_matrix(ratios: ArrayLike, name: str = "ratios") -> jnp.ndarray:
    values = input_array(ratios)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ContractError(f"{name} must have non-empty shape [N, K] with K >= 2")
    values = promote_low_precision(values)
    if not bool(np.asarray(jnp.all(jnp.isfinite(values)))):
        raise ContractError(f"{name} must be finite")
    if bool(np.asarray(jnp.any(values < 0))):
        raise ContractError(f"{name} must be nonnegative")
    return values


@execution_scope
def ratios_from_posteriors(
    posteriors: ArrayLike,
    class_priors: ArrayLike,
    *,
    execution: ExecutionConfig | None = None,
) -> np.ndarray:
    """Convert calibrated class posteriors into model density ratios.

    Parameters
    ----------
    posteriors
        Calibrated class-posterior matrix with shape ``[N, K]``. Rows must be
        finite, nonnegative, and sum to one.
    class_priors
        Strictly positive class priors under which ``posteriors`` were
        estimated, with shape ``[K]`` and unit sum.

    Returns
    -------
    numpy.ndarray
        Ratio matrix ``r_k(x) = posteriors_k(x) / class_priors_k`` with shape
        ``[N, K]``: the component density ratios ``phi_k(x) / p_train(x)``,
        where ``p_train`` is the training mixture. The common event-wise
        factor ``p_train`` cancels in every downstream score map.

    Raises
    ------
    ContractError
        If shapes, finiteness, nonnegativity, or simplex normalization
        violate the posterior contract.

    Notes
    -----
    Only a calibrated posterior carries ratio information. A ranking score or
    an arbitrary monotone transform of a likelihood ratio is not sufficient:
    score construction needs a quantitatively meaningful ratio, so
    calibration or a ratio-estimation loss is required upstream.

    The function does not calibrate, clip, or renormalize classifier output.
    Those operations change the implied density ratios and belong to the
    upstream ratio-estimation workflow.
    """
    del execution
    values = input_array(posteriors)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ContractError("posteriors must have non-empty shape [N, K] with K >= 2")
    values = promote_low_precision(values)
    if not bool(np.asarray(jnp.all(jnp.isfinite(values)))):
        raise ContractError("posteriors must be finite")
    if bool(np.asarray(jnp.any(values < 0))):
        raise ContractError("posteriors must be nonnegative")
    if not bool(
        np.asarray(
            jnp.allclose(jnp.sum(values, axis=1), 1.0, rtol=_SIMPLEX_RTOL, atol=_SIMPLEX_ATOL)
        )
    ):
        raise ContractError("posterior rows must sum to one")
    priors = jnp.asarray(class_priors, dtype=values.dtype)
    _validate_simplex_vector(priors, "class_priors", values.shape[1])
    return canonical_array(values / priors[None, :])


@execution_scope
def mixture_scores_from_ratios(
    ratios: ArrayLike,
    reference_fractions: ArrayLike,
    *,
    reference_component: int = -1,
    execution: ExecutionConfig | None = None,
) -> np.ndarray:
    """Construct normalized-mixture scores from component density ratios.

    Parameters
    ----------
    ratios
        Finite nonnegative component density ratios with shape ``[N, K]``,
        defined up to one common event-wise factor.
    reference_fractions
        Strictly positive mixture fractions at the score reference point,
        with shape ``[K]`` and unit sum.
    reference_component
        Component treated as dependent under the simplex constraint. Negative
        indices follow ordinary Python indexing. The last component is used
        by default.

    Returns
    -------
    numpy.ndarray
        Score matrix ``(r_k - r_ref) / sum_j reference_fractions[j] * r_j``
        with shape ``[N, K - 1]``. Columns follow the original component
        order with ``reference_component`` omitted.

    Raises
    ------
    TypeError
        If ``reference_component`` is not an integer.
    ContractError
        If shapes, finiteness, positivity, normalization, or the reference
        component violate the mixture-score contract.

    Notes
    -----
    This is the score of the normalized mixture
    ``p(x; theta) = sum_k theta_k phi_k(x)`` with ``sum_k theta_k = 1`` and
    the reference component dependent on the others. The map is invariant
    under a common event-wise rescaling ``r_k(x) -> c(x) r_k(x)``, so any
    gauge works: ratios relative to one component (``r_ref = 1``, giving
    ``(r_k - 1) / sum_j theta_j r_j``), relative to the training mixture, or
    relative to the reference density itself.
    """
    del execution
    values = _validate_ratio_matrix(ratios)
    n_components = values.shape[1]
    fractions = jnp.asarray(reference_fractions, dtype=values.dtype)
    _validate_simplex_vector(fractions, "reference_fractions", n_components)
    if isinstance(reference_component, bool) or not isinstance(reference_component, int):
        raise TypeError("reference_component must be an integer")
    if not -n_components <= reference_component < n_components:
        raise ContractError("reference_component is outside the component range")
    resolved_reference = reference_component % n_components
    density = values @ fractions
    if bool(np.asarray(jnp.any(density <= 0))):
        raise ContractError(
            "the reference-density denominator must be strictly positive at every row"
        )
    kept_components = [index for index in range(n_components) if index != resolved_reference]
    scores = (values[:, kept_components] - values[:, resolved_reference, None]) / density[:, None]
    if not bool(np.asarray(jnp.all(jnp.isfinite(scores)))):
        raise ContractError("mixture score construction produced non-finite values")
    return canonical_array(scores)


@dataclass(frozen=True, slots=True, init=False)
class IntensityParameterization:
    """Declare an extended linear-intensity model over component ratios.

    The model is ``lambda(x; theta) = sum_k theta_k phi_k(x)`` with free,
    unconstrained coefficients, so every one of the ``K`` score columns is
    retained, including the overall-normalization direction.

    Parameters
    ----------
    coefficients
        Finite reference coefficients ``theta_0`` with shape ``[K]``.
    """

    coefficients: jnp.ndarray

    def __init__(self, coefficients: ArrayLike) -> None:
        array = jnp.asarray(coefficients)
        if array.ndim != 1 or array.shape[0] < 2:
            raise ContractError("coefficients must have shape [K] with K >= 2")
        if not bool(np.asarray(jnp.all(jnp.isfinite(array)))):
            raise ContractError("coefficients must be finite")
        object.__setattr__(self, "coefficients", canonical_array(array))

    @property
    def n_components(self) -> int:
        """Number of ratio columns the parameterization consumes."""
        return int(self.coefficients.shape[0])

    @execution_scope
    def scores(
        self,
        ratios: ArrayLike,
        *,
        execution: ExecutionConfig | None = None,
    ) -> np.ndarray:
        """Return the ``[N, K]`` intensity scores ``r_k / sum_j theta_j r_j``.

        Because ``scores_from_components`` is invariant under a common
        event-wise rescaling of its rows, evaluating it on density ratios in
        any gauge yields exactly the component scores ``phi_k / lambda``.
        """
        values = _validate_ratio_matrix(ratios)
        if values.shape[1] != self.n_components:
            raise ContractError(f"ratios must have shape [N, {self.n_components}]")
        return scores_from_components(values, self.coefficients, execution=execution)


@dataclass(frozen=True, slots=True, init=False)
class MixtureParameterization:
    """Declare a normalized mixture with one simplex-dependent component.

    The model is ``p(x; theta) = sum_k theta_k phi_k(x)`` with
    ``sum_k theta_k = 1``; the reference component is dependent, so scores
    have ``K - 1`` columns.

    Parameters
    ----------
    reference_fractions
        Strictly positive mixture fractions ``theta_0`` with shape ``[K]``
        and unit sum.
    reference_component
        Component treated as dependent under the simplex constraint.
    """

    reference_fractions: jnp.ndarray
    reference_component: int

    def __init__(self, reference_fractions: ArrayLike, *, reference_component: int = -1) -> None:
        array = jnp.asarray(reference_fractions)
        if array.ndim != 1 or array.shape[0] < 2:
            raise ContractError("reference_fractions must have shape [K] with K >= 2")
        _validate_simplex_vector(array, "reference_fractions", int(array.shape[0]))
        if isinstance(reference_component, bool) or not isinstance(reference_component, int):
            raise TypeError("reference_component must be an integer")
        if not -array.shape[0] <= reference_component < array.shape[0]:
            raise ContractError("reference_component is outside the component range")
        object.__setattr__(self, "reference_fractions", canonical_array(array))
        object.__setattr__(self, "reference_component", reference_component)

    @property
    def n_components(self) -> int:
        """Number of ratio columns the parameterization consumes."""
        return int(self.reference_fractions.shape[0])

    @execution_scope
    def scores(
        self,
        ratios: ArrayLike,
        *,
        execution: ExecutionConfig | None = None,
    ) -> np.ndarray:
        """Return the ``[N, K - 1]`` constrained-mixture scores."""
        return mixture_scores_from_ratios(
            ratios,
            self.reference_fractions,
            reference_component=self.reference_component,
            execution=execution,
        )


type RatioParameterization = IntensityParameterization | MixtureParameterization


@execution_scope
def ratio_closure_report(
    ratios: ArrayLike,
    weights: ArrayLike,
    *,
    execution: ExecutionConfig | None = None,
) -> RatioClosureReport:
    """Check that density ratios integrate to one under the declared measure.

    Parameters
    ----------
    ratios
        Finite nonnegative model density ratios with shape ``[N, K]``,
        relative to the measure the weights carry.
    weights
        Finite nonnegative weights with shape ``[N]`` and at least one
        positive entry, carrying the reference measure of the ratio
        denominator.

    Returns
    -------
    RatioClosureReport
        Per-component weighted means ``sum_i w_i r_ik / sum_i w_i`` and their
        largest absolute deviation from one.

    Raises
    ------
    ContractError
        If shapes, finiteness, or nonnegativity violate the contract.

    Notes
    -----
    For exact ratios ``r_k = phi_k / p_ref`` every column integrates to one
    under the reference measure, so a nonzero residual bounds estimator bias
    from below: it is model error, never compression loss. The test is joint
    in the estimator, the declared training priors, and the measure, and it
    is necessary but not sufficient — closure never justifies upgrading
    estimated provenance to exact.
    """
    del execution
    values = _validate_ratio_matrix(ratios)
    weight_array = jnp.asarray(validate_weights(weights, int(values.shape[0]), dtype=values.dtype))
    normalizers = (weight_array @ values) / jnp.sum(weight_array)
    max_residual = float(jnp.max(jnp.abs(normalizers - 1.0)))
    return canonicalize_public(
        RatioClosureReport(normalizers=normalizers, max_residual=max_residual)
    )
