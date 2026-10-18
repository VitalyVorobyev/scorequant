"""Reference-measure sources for empirical and bounded population fitting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ._errors import ContractError
from ._execution import canonical_array
from ._json import json_ready
from ._typing import ArrayLike, JsonValue
from ._validation import canonical_sample, validate_weights

type ScoreKind = Literal[
    "unknown",
    "exact",
    "autodiff",
    "estimated_ratio",
    "custom_estimated",
]

type RatioParameterizationKind = Literal["intensity", "mixture", "central_log_ratio"]

type SourceKind = Literal["score_sample", "observation_sample", "integration_source"]

type InformationKind = Literal["exact_fisher", "supplied_score_surrogate"]


@dataclass(frozen=True, slots=True)
class ScoreSchema:
    """Name the parameter each score coordinate differentiates.

    A score table is a matrix of partial derivatives, one column per model
    parameter, and the column order is meaningful but invisible. Declaring the
    names lets a profiled criterion say ``interest=("HSPCs",)`` instead of
    ``interest=(4,)``, and lets reports and saved rules print the parameter
    rather than its position.

    This answers only *what each coordinate means*. Where the numbers came from
    and at which reference point remains the job of
    :class:`ScoreProvenance`, which already carries ``kind`` and
    ``reference_point``; the two are validated against each other rather than
    duplicating the reference point.

    Parameters
    ----------
    parameters
        Parameter names in score-column order. Names must be non-empty,
        unique, and at least one must be present.

    Examples
    --------
    >>> schema = ScoreSchema(("T cells", "B cells", "HSPCs"))
    >>> schema.index("HSPCs")
    2
    >>> schema.select("B cells", "HSPCs")
    (1, 2)
    """

    parameters: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the name vector independently of any score table."""
        if not isinstance(self.parameters, tuple):
            raise TypeError("parameters must be a tuple of names")
        if not self.parameters:
            raise ContractError("parameters must contain at least one name")
        if any(not isinstance(name, str) for name in self.parameters):
            raise TypeError("parameter names must be strings")
        if any(not name.strip() for name in self.parameters):
            raise ContractError("parameter names must be non-empty")
        if len(set(self.parameters)) != len(self.parameters):
            duplicates = sorted(
                {name for name in self.parameters if self.parameters.count(name) > 1}
            )
            raise ContractError(
                f"parameter names must be unique; repeated: {', '.join(duplicates)}"
            )

    @property
    def dimension(self) -> int:
        """Return the number of score columns this schema describes."""
        return len(self.parameters)

    def index(self, name: str) -> int:
        """Return the score column of one named parameter.

        Raises
        ------
        KeyError
            When the name is not declared, naming the available parameters so
            a typo is diagnosable without reading the constructor call.
        """
        try:
            return self.parameters.index(name)
        except ValueError:
            raise KeyError(
                f"unknown parameter {name!r}; schema declares {', '.join(self.parameters)}"
            ) from None

    def select(self, *names: str) -> tuple[int, ...]:
        """Return the score columns of several named parameters, in the order given."""
        return tuple(self.index(name) for name in names)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return JSON-ready schema state."""
        return json_ready({"parameters": list(self.parameters)})


@dataclass(frozen=True, slots=True)
class RatioProvenance:
    """Describe how a model density-ratio representation was obtained.

    Together with ``ScoreProvenance.kind`` and ``reference_point``, these
    fields reconstruct the statistical representation behind a ratio-derived
    score: which estimator produced the ratios, under which training priors
    and calibration, and through which parameterization they became scores.
    Fields that do not apply to a given construction stay ``None``.

    ``gauges`` names, for each ratio column, the measure that column is a
    density ratio against: ``"training_mixture"`` for prior-corrected
    classifier posteriors, ``"absolute"`` for a Monte Carlo or analytic
    density, or any label of the caller's choosing. The score map cancels only
    a factor common to every column of an event, so columns declared in
    different gauges are refused when the record is built; no provider, score
    sample or saved rule can carry them. ``None`` declares nothing and is not
    checked.
    """

    estimator: str | None = None
    parameterization: RatioParameterizationKind | None = None
    coefficients: tuple[float, ...] | None = None
    reference_fractions: tuple[float, ...] | None = None
    reference_component: int | None = None
    training_priors: tuple[float, ...] | tuple[tuple[float, float], ...] | None = None
    calibration: str | None = None
    deltas: tuple[float, ...] | None = None
    gauges: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Refuse ratio columns declared against different measures.

        Every ratio-to-score map is invariant only under a factor common to all
        columns of one event. Columns in different gauges, such as a
        prior-corrected classifier ratio over ``p_train`` next to an absolute
        Monte Carlo density, leave the event-dependent ``p_train(x)``
        uncancelled, and every score is silently wrong.
        """
        gauges = self.gauges
        if gauges is None:
            return
        if not isinstance(gauges, tuple) or not all(isinstance(label, str) for label in gauges):
            raise TypeError("RatioProvenance.gauges must be a tuple of strings")
        if len(set(gauges)) > 1:
            raise ContractError(
                f"ratio columns are declared in different gauges {gauges}: scores cancel only "
                "a factor common to every column of an event, so mixed gauges leave an "
                "event-dependent denominator in every score. Return every column in one "
                "gauge; for a two-class signal-versus-background classifier with priors pi, "
                "the columns [(eta_sig / pi_sig) / (eta_bkg / pi_bkg), 1] share the background "
                "gauge and need no Monte Carlo density"
            )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible ratio-provenance mapping."""
        return json_ready(
            {
                "estimator": self.estimator,
                "parameterization": self.parameterization,
                "coefficients": self.coefficients,
                "reference_fractions": self.reference_fractions,
                "reference_component": self.reference_component,
                "training_priors": self.training_priors,
                "calibration": self.calibration,
                "deltas": self.deltas,
                "gauges": self.gauges,
            }
        )


@dataclass(frozen=True, slots=True)
class ScoreProvenance:
    """Describe where supplied score coordinates came from.

    ``exact_fisher`` is derived from ``kind`` rather than accepted as an
    independent flag, so estimated scores cannot accidentally claim exact
    Fisher semantics. Scores built from model density ratios additionally
    carry a ``ratio`` record describing how the ratios were obtained.
    """

    kind: ScoreKind = "unknown"
    description: str | None = None
    reference_point: tuple[float, ...] | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    ratio: RatioProvenance | None = None

    @property
    def exact_fisher(self) -> bool:
        """Return whether exact-Fisher language is permitted."""
        return self.kind in ("exact", "autodiff")

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible provenance mapping."""
        return json_ready(
            {
                "kind": self.kind,
                "description": self.description,
                "reference_point": self.reference_point,
                "metadata": dict(self.metadata),
                "ratio": None if self.ratio is None else self.ratio.to_dict(),
                "exact_fisher": self.exact_fisher,
            }
        )


def validate_schema(
    schema: ScoreSchema | None,
    dimension: int,
    provenance: ScoreProvenance | None = None,
) -> None:
    """Check a schema against the score dimension it is meant to describe.

    The reference point deliberately lives on :class:`ScoreProvenance` rather
    than on the schema, so this is also where the two are held to the same
    parameter count instead of being allowed to disagree silently.
    """
    if schema is None:
        return
    if not isinstance(schema, ScoreSchema):
        raise TypeError("schema must be a ScoreSchema")
    if schema.dimension != dimension:
        raise ContractError(
            f"schema names {schema.dimension} parameters but the scores have {dimension} columns"
        )
    reference_point = None if provenance is None else provenance.reference_point
    if reference_point is not None and len(reference_point) != schema.dimension:
        raise ContractError(
            f"schema names {schema.dimension} parameters but the provenance "
            f"reference point has {len(reference_point)} entries"
        )


@dataclass(frozen=True, slots=True, init=False)
class ScoreSample:
    """A finite weighted score table representing an empirical score law.

    Parameters
    ----------
    scores, weights
        Score rows with shape ``[N, P]`` and their nonnegative measure.
    schema
        Optional :class:`ScoreSchema` naming what each of the ``P`` columns
        differentiates. When present it must have exactly ``P`` names, and a
        profiled criterion may then declare its parameters of interest by name.
    provenance
        Optional record of how the scores were obtained.
    """

    scores: np.ndarray
    weights: np.ndarray
    provenance: ScoreProvenance
    schema: ScoreSchema | None

    def __init__(
        self,
        scores: ArrayLike,
        weights: ArrayLike | None = None,
        *,
        schema: ScoreSchema | None = None,
        provenance: ScoreProvenance | None = None,
    ) -> None:
        sample = canonical_sample(scores, weights)
        resolved_provenance = provenance or ScoreProvenance()
        validate_schema(schema, sample.scores.shape[1], resolved_provenance)
        object.__setattr__(self, "scores", canonical_array(sample.scores))
        object.__setattr__(self, "weights", canonical_array(sample.weights))
        object.__setattr__(self, "provenance", resolved_provenance)
        object.__setattr__(self, "schema", schema)


@dataclass(frozen=True, slots=True, init=False)
class ObservationSample:
    """A finite weighted observation table requiring a score provider."""

    observations: np.ndarray
    weights: np.ndarray

    def __init__(self, observations: ArrayLike, weights: ArrayLike | None = None) -> None:
        array = np.asarray(observations)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ContractError("observations must have non-empty shape [N, D]")
        if not np.issubdtype(array.dtype, np.inexact):
            array = array.astype(np.float32)
        if not bool(np.all(np.isfinite(array))):
            raise ContractError("observations must be finite")
        weight_array = validate_weights(weights, array.shape[0], dtype=array.dtype)
        object.__setattr__(self, "observations", canonical_array(array))
        object.__setattr__(self, "weights", canonical_array(weight_array))


@dataclass(frozen=True, slots=True)
class GaussLegendreConfig:
    """Configure deterministic tensor-product Gauss-Legendre quadrature."""

    order: int = 16
    max_points: int = 1_000_000

    def __post_init__(self) -> None:
        """Validate quadrature capacity before materialization."""
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 2:
            raise ContractError("order must be an integer of at least two")
        if (
            isinstance(self.max_points, bool)
            or not isinstance(self.max_points, int)
            or self.max_points < 1
        ):
            raise ContractError("max_points must be a positive integer")


@dataclass(frozen=True, slots=True, init=False)
class IntegrationSource:
    """A finite box, density, and deterministic quadrature reference measure.

    This source is intentionally limited to low-dimensional bounded domains.
    The density or intensity is mandatory; bounds alone never imply a uniform
    statistical measure.
    """

    bounds: np.ndarray
    density: Callable[[ArrayLike], ArrayLike]
    quadrature: GaussLegendreConfig

    def __init__(
        self,
        bounds: ArrayLike,
        *,
        density: Callable[[ArrayLike], ArrayLike],
        quadrature: GaussLegendreConfig | None = None,
    ) -> None:
        array = np.asarray(bounds)
        if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
            raise ContractError("bounds must have shape [D, 2]")
        if not np.issubdtype(array.dtype, np.inexact):
            array = array.astype(np.float32)
        if not bool(np.all(np.isfinite(array))):
            raise ContractError("bounds must be finite")
        if not bool(np.all(array[:, 1] > array[:, 0])):
            raise ContractError("every upper bound must be greater than its lower bound")
        if not callable(density):
            raise TypeError("density must be callable")
        object.__setattr__(self, "bounds", canonical_array(array))
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "quadrature", quadrature or GaussLegendreConfig())

    def materialize(self) -> ObservationSample:
        """Evaluate tensor quadrature nodes and density-weighted measure weights."""
        dimension = int(self.bounds.shape[0])
        point_count = self.quadrature.order**dimension
        if point_count > self.quadrature.max_points:
            raise ContractError(
                "tensor quadrature would create "
                f"{point_count} points, exceeding max_points={self.quadrature.max_points}"
            )
        canonical_nodes, canonical_weights = np.polynomial.legendre.leggauss(self.quadrature.order)
        nodes_per_axis: list[np.ndarray] = []
        weights_per_axis: list[np.ndarray] = []
        for lower, upper in np.asarray(self.bounds):
            half_width = 0.5 * (upper - lower)
            midpoint = 0.5 * (upper + lower)
            nodes_per_axis.append(midpoint + half_width * canonical_nodes)
            weights_per_axis.append(half_width * canonical_weights)
        node_mesh = np.meshgrid(*nodes_per_axis, indexing="ij")
        weight_mesh = np.meshgrid(*weights_per_axis, indexing="ij")
        observations = np.stack([axis.reshape(-1) for axis in node_mesh], axis=1)
        quadrature_weights = np.prod(
            np.stack([axis.reshape(-1) for axis in weight_mesh], axis=1), axis=1
        )
        density_values = np.asarray(self.density(observations))
        if density_values.shape != (point_count,):
            raise ContractError(f"density must return shape [{point_count}]")
        if not bool(np.all(np.isfinite(density_values))):
            raise ContractError("density values must be finite")
        if bool(np.any(density_values < 0)):
            raise ContractError("density values must be nonnegative")
        weights = density_values * np.asarray(quadrature_weights, dtype=density_values.dtype)
        return ObservationSample(observations, weights)


type Source = ScoreSample | ObservationSample | IntegrationSource
