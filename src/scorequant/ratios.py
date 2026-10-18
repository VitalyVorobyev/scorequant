"""Density-ratio score providers, their parameterizations and diagnostics.

This is the public home of everything ratio-specific. The route is one
provider: :class:`DensityRatioScore` pairs a model density-ratio callback with
a declared parameterization (:class:`MixtureParameterization` or
:class:`IntensityParameterization`), and
:meth:`DensityRatioScore.from_classifier` builds that callback from a
calibrated classifier. A parameterization's ``scores`` method is the same map
applied to a ratio array. :class:`CentralLogRatioScore` covers classifiers
trained at ``theta_0 +- delta``; :func:`ratio_closure_report` checks estimated
ratios against their measure.

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

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from ._errors import ContractError
from ._execution import canonical_array, canonicalize_public, execution_scope, input_array
from ._execution import xp as jnp
from ._typing import ArrayLike, JsonValue
from ._validation import promote_low_precision, validate_weights
from .components import _scores_from_components
from .config import ExecutionConfig
from .reports import RatioClosureReport
from .sources import RatioProvenance, ScoreProvenance, ScoreSchema

__all__ = [
    "CentralLogRatioScore",
    "DensityRatioScore",
    "IntensityParameterization",
    "MixtureParameterization",
    "RatioClosureReport",
    "RatioProvenance",
    "ratio_closure_report",
    "ratios_from_posteriors",
]

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


def _mixture_scores(
    ratios: ArrayLike,
    reference_fractions: ArrayLike,
    *,
    reference_component: int = -1,
) -> np.ndarray:
    """Construct normalized-mixture scores from component density ratios.

    The array map behind :meth:`MixtureParameterization.scores`, run inside
    its execution scope.

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

        The map is invariant under a common event-wise rescaling of a row, so
        density ratios in any gauge, component densities included, yield
        exactly the component scores ``phi_k / lambda``.
        """
        values = _validate_ratio_matrix(ratios)
        if values.shape[1] != self.n_components:
            raise ContractError(f"ratios must have shape [N, {self.n_components}]")
        return _scores_from_components(values, self.coefficients, execution=execution)


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
        """Return the ``[N, K - 1]`` constrained-mixture scores.

        Columns follow the original component order with the reference
        component omitted; each is ``(r_k - r_ref) / sum_j theta_j r_j``.
        """
        del execution
        return _mixture_scores(
            ratios,
            self.reference_fractions,
            reference_component=self.reference_component,
        )


type RatioParameterization = IntensityParameterization | MixtureParameterization


def _parameterization_facts(parameterization: RatioParameterization) -> RatioProvenance:
    if isinstance(parameterization, IntensityParameterization):
        return RatioProvenance(
            parameterization="intensity",
            coefficients=tuple(float(value) for value in parameterization.coefficients),
        )
    return RatioProvenance(
        parameterization="mixture",
        reference_fractions=tuple(float(value) for value in parameterization.reference_fractions),
        reference_component=parameterization.reference_component % parameterization.n_components,
    )


def _merge_ratio_provenance(
    supplied: RatioProvenance | None, facts: RatioProvenance
) -> RatioProvenance:
    if supplied is None:
        return facts
    for field_name in (
        "parameterization",
        "coefficients",
        "reference_fractions",
        "reference_component",
    ):
        supplied_value = getattr(supplied, field_name)
        derived_value = getattr(facts, field_name)
        if supplied_value is not None and supplied_value != derived_value:
            raise ContractError(
                f"provenance.ratio.{field_name}={supplied_value!r} conflicts with the "
                f"declared parameterization value {derived_value!r}"
            )
    return RatioProvenance(
        estimator=supplied.estimator,
        parameterization=facts.parameterization,
        coefficients=facts.coefficients,
        reference_fractions=facts.reference_fractions,
        reference_component=facts.reference_component,
        training_priors=supplied.training_priors,
        calibration=supplied.calibration,
        deltas=supplied.deltas,
        gauges=supplied.gauges,
    )


_TRAINING_MIXTURE_GAUGE = "training_mixture"


def _require_gauge_count(ratio: RatioProvenance | None, n_components: int) -> None:
    """Require one declared gauge per ratio column the parameterization consumes.

    ``RatioProvenance`` itself refuses mixed gauges, wherever the record is
    built; only the column count needs the provider's parameterization.
    """
    if ratio is None or ratio.gauges is None:
        return
    if len(ratio.gauges) != n_components:
        raise ContractError(
            f"provenance.ratio.gauges declares {len(ratio.gauges)} columns; the "
            f"parameterization consumes {n_components}"
        )


@dataclass(frozen=True, slots=True, init=False)
class DensityRatioScore:
    """Map observations to model density ratios and evaluate declared scores.

    The ratio callback is the statistical representation: any oracle for the
    component density ratios — an analytic formula, a calibrated classifier,
    a direct ratio estimator such as KLIEP or uLSIF, or an external ratio
    model — determines the score once a parameterization declares how the
    components combine. Ratio estimation, calibration, and cross-fitting stay
    outside the library.

    Parameters
    ----------
    ratio
        Callable ``[N, D] -> [N, K]`` returning finite nonnegative model
        density ratios, defined up to one common event-wise factor.
    parameterization
        ``IntensityParameterization`` or ``MixtureParameterization`` declaring
        the ratio-to-score map and the reference point.
    provenance
        Optional score provenance. Estimated ratios are the default
        (``kind="estimated_ratio"``); an analytic ratio may declare
        ``kind="exact"`` under the same caller responsibility as
        ``ScoreFunction``. Parameterization facts are always recorded in
        ``provenance.ratio`` and a conflicting supplied record is rejected.
        Declaring ``provenance.ratio.gauges`` makes the provider refuse ratio
        columns taken against different measures.
    """

    ratio: Callable[[ArrayLike], ArrayLike]
    parameterization: RatioParameterization
    provenance: ScoreProvenance
    schema: ScoreSchema | None

    def __init__(
        self,
        ratio: Callable[[ArrayLike], ArrayLike],
        parameterization: RatioParameterization,
        *,
        provenance: ScoreProvenance | None = None,
        schema: ScoreSchema | None = None,
    ) -> None:
        if not callable(ratio):
            raise TypeError("ratio must be callable")
        if schema is not None and not isinstance(schema, ScoreSchema):
            raise TypeError("schema must be a ScoreSchema")
        if not isinstance(parameterization, (IntensityParameterization, MixtureParameterization)):
            raise TypeError(
                "parameterization must be IntensityParameterization or MixtureParameterization"
            )
        facts = _parameterization_facts(parameterization)
        if provenance is None:
            resolved = ScoreProvenance(kind="estimated_ratio", ratio=facts)
        else:
            resolved = ScoreProvenance(
                kind=provenance.kind,
                description=provenance.description,
                reference_point=provenance.reference_point,
                metadata=provenance.metadata,
                ratio=_merge_ratio_provenance(provenance.ratio, facts),
            )
        _require_gauge_count(resolved.ratio, parameterization.n_components)
        object.__setattr__(self, "ratio", ratio)
        object.__setattr__(self, "parameterization", parameterization)
        object.__setattr__(self, "provenance", resolved)
        object.__setattr__(self, "schema", schema)

    @classmethod
    def from_classifier(
        cls,
        predict: Callable[[ArrayLike], ArrayLike],
        class_priors: ArrayLike,
        parameterization: RatioParameterization,
        *,
        calibration: str | None = None,
        description: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> DensityRatioScore:
        """Build a ratio provider from a calibrated multiclass classifier.

        The classifier is one estimator of density ratios: calibrated
        posteriors ``eta_k`` under training priors ``pi_k`` give
        ``r_k = eta_k / pi_k`` up to a common event-wise factor, so the
        scores are ``(eta_k / pi_k) / sum_j theta_j eta_j / pi_j`` under the
        intensity parameterization. When ``pi`` is proportional to
        ``theta_0`` the denominator is identically one. The provider always
        records estimated provenance — a classifier-derived ratio can never
        claim exact Fisher semantics — and declares every column in the
        ``"training_mixture"`` gauge.

        Parameters
        ----------
        predict
            Callable ``[N, D] -> [N, K]`` returning calibrated posterior
            rows summing to one.
        class_priors
            Strictly positive training priors with shape ``[K]`` and unit
            sum.
        parameterization
            Ratio-to-score map; ``K`` must match its component count.
        calibration
            Optional name of the upstream calibration method, recorded in
            provenance.
        description, metadata
            Optional free-form provenance carried on the score record.
        """
        if not callable(predict):
            raise TypeError("predict must be callable")
        priors = jnp.asarray(class_priors)
        if not isinstance(parameterization, (IntensityParameterization, MixtureParameterization)):
            raise TypeError(
                "parameterization must be IntensityParameterization or MixtureParameterization"
            )
        _validate_simplex_vector(priors, "class_priors", parameterization.n_components)

        def ratio(observations: ArrayLike) -> jnp.ndarray:
            return ratios_from_posteriors(predict(observations), priors)

        provenance = ScoreProvenance(
            kind="estimated_ratio",
            description=description,
            metadata={} if metadata is None else metadata,
            ratio=RatioProvenance(
                estimator="calibrated_classifier",
                training_priors=tuple(float(value) for value in priors),
                calibration=calibration,
                gauges=(_TRAINING_MIXTURE_GAUGE,) * parameterization.n_components,
            ),
        )
        return cls(ratio, parameterization, provenance=provenance)

    @execution_scope
    def score(
        self,
        observations: ArrayLike,
        *,
        execution: ExecutionConfig | None = None,
    ) -> np.ndarray:
        """Evaluate the ratio callback and apply the declared score map."""
        values = jnp.asarray(self.ratio(observations))
        if values.ndim != 2 or values.shape[0] != jnp.asarray(observations).shape[0]:
            raise ContractError("ratio callback must return shape [N, K]")
        expected = self.parameterization.n_components
        if values.shape[1] != expected:
            raise ContractError(
                f"ratio callback must return {expected} components, got {values.shape[1]}"
            )
        return self.parameterization.scores(values, execution=execution)


@dataclass(frozen=True, slots=True, init=False)
class CentralLogRatioScore:
    """Estimate central finite-difference scores from paired density ratios.

    A calibrated classifier trained to separate samples generated at
    ``theta_0 - delta e_p`` from ``theta_0 + delta e_p`` estimates the
    directional log density ratio, and
    ``(log(p_plus / p_minus) - log(pi_plus / pi_minus)) / (2 delta)`` is a
    central finite-difference estimate of the score component ``s_p``. The
    prediction callback must return shape ``[N, P, 2]`` with class order
    ``(minus, plus)``; a two-dimensional ``[N, 2]`` input is accepted for a
    single score direction. Provenance is always estimated.

    Parameters
    ----------
    predict
        Callable returning calibrated minus/plus probability pairs.
    deltas
        Strictly positive finite-difference offsets with shape ``[P]``.
    class_priors
        Training priors per direction with shape ``[2]`` or ``[P, 2]``;
        rows are normalized to sum to one.
    description, metadata
        Optional free-form provenance carried on the score record.
    """

    predict: Callable[[ArrayLike], ArrayLike]
    deltas: jnp.ndarray
    class_priors: jnp.ndarray
    description: str | None
    metadata: Mapping[str, JsonValue]
    schema: ScoreSchema | None

    def __init__(
        self,
        predict: Callable[[ArrayLike], ArrayLike],
        deltas: ArrayLike,
        class_priors: ArrayLike,
        *,
        description: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        schema: ScoreSchema | None = None,
    ) -> None:
        if not callable(predict):
            raise TypeError("predict must be callable")
        if schema is not None and not isinstance(schema, ScoreSchema):
            raise TypeError("schema must be a ScoreSchema")
        delta_array = jnp.asarray(deltas)
        if delta_array.ndim != 1 or delta_array.shape[0] == 0:
            raise ContractError("deltas must have shape [P]")
        if not bool(np.asarray(jnp.all(jnp.isfinite(delta_array)))) or bool(
            np.asarray(jnp.any(delta_array <= 0))
        ):
            raise ContractError("deltas must be finite and positive")
        priors = jnp.asarray(class_priors, dtype=delta_array.dtype)
        if priors.shape == (2,):
            priors = jnp.broadcast_to(priors, (delta_array.shape[0], 2))
        if priors.shape != (delta_array.shape[0], 2):
            raise ContractError("class_priors must have shape [2] or [P, 2]")
        if not bool(np.asarray(jnp.all(jnp.isfinite(priors)))) or bool(
            np.asarray(jnp.any(priors <= 0))
        ):
            raise ContractError("class_priors must be finite and positive")
        priors = priors / jnp.sum(priors, axis=1, keepdims=True)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "predict", predict)
        object.__setattr__(self, "deltas", canonical_array(delta_array))
        object.__setattr__(self, "class_priors", canonical_array(priors))
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "metadata", {} if metadata is None else metadata)

    @property
    def provenance(self) -> ScoreProvenance:
        """Return estimated-ratio provenance with the central-difference facts."""
        return ScoreProvenance(
            kind="estimated_ratio",
            description=self.description,
            metadata=self.metadata,
            ratio=RatioProvenance(
                estimator="calibrated_classifier",
                parameterization="central_log_ratio",
                training_priors=tuple(
                    (float(row[0]), float(row[1])) for row in np.asarray(self.class_priors)
                ),
                deltas=tuple(float(value) for value in self.deltas),
            ),
        )

    @execution_scope
    def score(
        self,
        observations: ArrayLike,
        *,
        execution: ExecutionConfig | None = None,
    ) -> np.ndarray:
        """Apply prior correction and divide central logits by ``2 * delta``."""
        del execution
        values = jnp.asarray(self.predict(observations), dtype=self.deltas.dtype)
        if values.ndim == 2 and values.shape[1] == 2 and self.deltas.shape[0] == 1:
            values = values[:, None, :]
        if values.ndim != 3 or values.shape[1:] != self.class_priors.shape:
            raise ContractError(f"probabilities must have shape [N, {self.deltas.shape[0]}, 2]")
        if not bool(np.asarray(jnp.all(jnp.isfinite(values)))) or bool(
            np.asarray(jnp.any(values <= 0))
        ):
            raise ContractError("probabilities must be finite and strictly positive")
        if not bool(np.asarray(jnp.allclose(jnp.sum(values, axis=2), 1))):
            raise ContractError("classifier probability pairs must sum to one")
        log_ratio = jnp.log(values[:, :, 1] / values[:, :, 0])
        prior_log_ratio = jnp.log(self.class_priors[:, 1] / self.class_priors[:, 0])
        return canonical_array((log_ratio - prior_log_ratio[None, :]) / (2 * self.deltas[None, :]))


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
