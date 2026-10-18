"""Framework-neutral observation-to-score providers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ._errors import ContractError
from ._execution import canonical_array, execution_scope
from ._execution import xp as jnp
from ._typing import ArrayLike, JsonValue
from .components import LinearComponents, scores_from_components
from .config import ExecutionConfig
from .ratios import (
    IntensityParameterization,
    MixtureParameterization,
    RatioParameterization,
    _validate_simplex_vector,
    ratios_from_posteriors,
)
from .sources import RatioProvenance, ScoreProvenance, ScoreSchema


@dataclass(frozen=True, slots=True)
class ScoreFunction:
    """Wrap an exact or estimated callable ``[N, D] -> [N, P]``."""

    function: Callable[[ArrayLike], ArrayLike]
    provenance: ScoreProvenance = ScoreProvenance()
    schema: ScoreSchema | None = None

    def __post_init__(self) -> None:
        """Reject non-callable providers at construction time."""
        if not callable(self.function):
            raise TypeError("function must be callable")
        if self.schema is not None and not isinstance(self.schema, ScoreSchema):
            raise TypeError("schema must be a ScoreSchema")

    @execution_scope
    def score(
        self,
        observations: ArrayLike,
        *,
        execution: ExecutionConfig | None = None,
    ) -> np.ndarray:
        """Evaluate and validate a finite score matrix."""
        del execution
        values = jnp.asarray(self.function(observations))
        if values.ndim != 2 or values.shape[0] != jnp.asarray(observations).shape[0]:
            raise ContractError("score function must return shape [N, P]")
        if values.shape[1] == 0 or not bool(np.asarray(jnp.all(jnp.isfinite(values)))):
            raise ContractError("score function must return finite non-empty scores")
        return canonical_array(values)


@dataclass(frozen=True, slots=True)
class LinearComponentScore:
    """Evaluate a frozen linear-component model and return local scores."""

    model: LinearComponents
    provenance: ScoreProvenance = ScoreProvenance(
        kind="exact", description="linear-component event score"
    )

    def __post_init__(self) -> None:
        """Validate the model boundary."""
        if not isinstance(self.model, LinearComponents):
            raise TypeError("model must be LinearComponents")

    @property
    def schema(self) -> ScoreSchema:
        """Name each score column after the component it differentiates.

        A linear-intensity model emits one score column per component and the
        component names are already declared, so the schema is derived rather
        than asked for.
        """
        return ScoreSchema(self.model.component_names)

    @execution_scope
    def score(
        self,
        observations: ArrayLike,
        *,
        execution: ExecutionConfig | None = None,
    ) -> np.ndarray:
        """Evaluate components and their frozen reference score."""
        components = self.model.evaluate_components(observations, execution=execution)
        return scores_from_components(
            components,
            self.model.coefficients,
            execution=execution,
        )


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


@runtime_checkable
class ScoreProvider(Protocol):
    """The observation-to-score contract every fitting route goes through.

    ScoreQuant does not care how a score was obtained -- an analytic
    derivative, autodiff, a component model, a density-ratio estimator, or an
    external inference package -- only that observations can be mapped to a
    finite score matrix and that the mapping says what it is. Those are the two
    members here, so a caller can supply their own object rather than wrapping
    it:

    .. code-block:: python

        class MyExternalScore:
            provenance = ScoreProvenance(kind="estimated_ratio")

            def score(self, observations):
                return my_package.evaluate(observations)

    The built-in providers -- :class:`ScoreFunction`,
    :class:`LinearComponentScore`, :class:`DensityRatioScore` and
    :class:`CentralLogRatioScore` -- are convenience implementations of this
    protocol, not a closed list of what is allowed.

    A provider may additionally expose ``schema``, a
    :class:`~scorequant.ScoreSchema` naming its score columns; it is used when
    present and is not part of the required contract.

    Notes
    -----
    ``score`` takes observations alone. The execution backend is ambient
    context established by the public task, not an argument a provider has to
    thread through, so an external implementation needs no knowledge of
    :class:`~scorequant.ExecutionConfig`.
    """

    @property
    def provenance(self) -> ScoreProvenance:
        """Describe how this score representation was obtained."""
        ...

    def score(self, observations: ArrayLike) -> ArrayLike:
        """Map observations with shape ``[N, D]`` to scores with shape ``[N, P]``."""
        ...


def validate_provider(provider: object) -> None:
    """Reject an object that cannot act as a score provider.

    ``runtime_checkable`` only proves the attributes exist, so the provenance
    type is checked here too: a provider whose ``provenance`` is not a
    :class:`~scorequant.ScoreProvenance` would otherwise reach the result and
    silently decide whether exact-Fisher language is justified.
    """
    if not isinstance(provider, ScoreProvider):
        missing = [name for name in ("score", "provenance") if not hasattr(provider, name)]
        raise TypeError(
            "provider must implement the ScoreProvider protocol; "
            f"{type(provider).__name__} is missing {', '.join(missing)}"
        )
    if not isinstance(provider.provenance, ScoreProvenance):
        raise TypeError(
            "provider.provenance must be a ScoreProvenance, not "
            f"{type(provider.provenance).__name__}"
        )
    if not callable(provider.score):
        raise TypeError("provider.score must be callable")
