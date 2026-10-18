"""The deployable quantizer: a frozen score-space rule and its storage format.

A fitted result carries everything the fit produced -- labels, reports, an
optimization trace, solver diagnostics. Almost none of that is needed to assign
a bin to a future event. :class:`Quantizer` is the part that is: a transform, a
set of centers, an optional metric, and the names and provenance that say what
those numbers mean.

Separating it is what makes the rule deployable. The artifact written by
:meth:`Quantizer.save` contains no pickled objects and no JAX, so a rule fitted
offline on the accelerated backend can be loaded and applied in a process that
has neither the fitting code path nor JAX installed.
"""

from __future__ import annotations

import importlib.util
import json
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

from ._errors import ContractError
from ._execution import canonical_array, use_execution
from ._json import json_ready
from ._predict import chunked_predict_labels
from ._typing import ArrayLike, JsonValue
from .config import ExecutionConfig
from .criteria import Criterion, DOptimality, NormalizedTrace, ProfiledDOptimality
from .reports import InformationReport
from .sources import (
    InformationKind,
    RatioParameterizationKind,
    RatioProvenance,
    ScoreProvenance,
    ScoreSchema,
)
from .transforms import FisherTransform

#: Version of the on-disk artifact layout. A reader refuses anything it does not
#: know rather than guessing at a field it has never seen.
FORMAT_VERSION = 1

#: Suffix used by :meth:`Quantizer.save` when the path carries none.
ARTIFACT_SUFFIX = ".sqz"

_MANIFEST = "manifest.json"


@dataclass(frozen=True, slots=True)
class Quantizer:
    """A frozen hard rule on score space.

    This is what a fit is *for*: the object that turns a future score vector
    into a bin. It holds no training labels, no reports and no history -- those
    stay on :class:`~scorequant.QuantizerResult`, which exposes the rule as
    ``result.quantizer``.

    Attributes
    ----------
    transform
        Projection onto the informative score subspace the rule was fitted in.
    centers
        Cell centers in transformed coordinates, shape ``[K, R]``.
    metric
        Optional common Mahalanobis metric, shape ``[R, R]``. ``None`` means
        the ordinary Euclidean nearest-center rule in transformed coordinates.
    schema
        Optional names of the raw score coordinates the rule expects.
    provenance
        How the scores the rule was fitted on were obtained.
    criterion
        The objective the rule was optimized for, recorded so a loaded artifact
        can still say what it was built to preserve.
    execution
        The backend prediction runs on unless overridden. A fitted rule keeps
        the execution it was fitted under; a loaded rule takes the loading
        environment's default, because the artifact does not record one.
    """

    transform: FisherTransform
    centers: np.ndarray
    metric: np.ndarray | None
    schema: ScoreSchema | None
    provenance: ScoreProvenance
    criterion: Criterion
    execution: ExecutionConfig

    @property
    def n_bins(self) -> int:
        """Return the number of hard output labels."""
        return int(self.centers.shape[0])

    @property
    def rank(self) -> int:
        """Return the numerically informative score-space rank."""
        return self.transform.rank

    @property
    def input_dim(self) -> int:
        """Return the raw score dimension the rule accepts."""
        return self.transform.input_dim

    @property
    def information_kind(self) -> InformationKind:
        """Describe whether supplied-score matrices justify exact Fisher language."""
        return "exact_fisher" if self.provenance.exact_fisher else "supplied_score_surrogate"

    def predict_scores(
        self,
        scores: ArrayLike,
        *,
        execution: ExecutionConfig | None = None,
    ) -> np.ndarray:
        """Assign raw score rows with the frozen rule.

        Rows are assigned in memory-bounded chunks, so predicting on a large
        sample never materializes the full ``[n_rows, n_bins, rank]`` distance
        tensor at once. Each row's nearest-center argmin is independent of every
        other row, so chunking is bit-identical to the unchunked computation.
        """
        resolved = execution or self.execution
        with use_execution(resolved):
            coordinates = self.transform.apply(scores, execution=resolved)
            return canonical_array(chunked_predict_labels(coordinates, self.centers, self.metric))

    def evaluate_scores(
        self,
        scores: ArrayLike,
        weights: ArrayLike | None = None,
        *,
        rank_rtol: float | None = None,
        execution: ExecutionConfig | None = None,
    ) -> InformationReport:
        """Measure the information this rule retains on a new weighted sample.

        This is a property of the rule rather than of the fit that produced it,
        so a loaded artifact can be scored against fresh data without the
        training result being present.

        Parameters
        ----------
        rank_rtol
            Relative rank threshold for the report. Defaults to the one the
            rule's own transform was built with.
        """
        from .information import information_report

        resolved = execution or self.execution
        return information_report(
            scores,
            self.predict_scores(scores, execution=resolved),
            weights,
            n_bins=self.n_bins,
            rank_rtol=self.transform.rank_rtol if rank_rtol is None else rank_rtol,
            execution=resolved,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return JSON-ready rule state.

        Unlike the diagnostic ``to_dict`` on the result types, this one is the
        readable face of a versioned format: :meth:`save` writes exactly these
        fields, with the arrays moved out to ``.npy`` members.
        """
        return json_ready(
            {
                "format_version": FORMAT_VERSION,
                "transform": self.transform.to_dict(),
                "centers": self.centers,
                "metric": self.metric,
                "schema": None if self.schema is None else self.schema.to_dict(),
                "provenance": self.provenance.to_dict(),
                "criterion": self.criterion.to_dict(),
                "execution": self.execution.to_dict(),
                "n_bins": self.n_bins,
                "rank": self.rank,
                "input_dim": self.input_dim,
            }
        )

    def save(self, path: str | Path) -> Path:
        """Write the rule to a versioned, non-pickle artifact.

        The file is a zip archive holding one ``manifest.json`` and one ``.npy``
        member per array. Nothing is pickled, so loading it executes no code
        from the file, and reading it back needs neither JAX nor the solver that
        produced it.

        The fitting execution is not saved: a rule fitted on a GPU must still
        load on a CPU-only or JAX-less machine. The manifest carries a default
        ``execution`` object only so 0.2 readers, which require the field, can
        read the file; :meth:`load` ignores it.

        Parameters
        ----------
        path
            Destination path. ``.sqz`` is appended when the path has no suffix.

        Returns
        -------
        pathlib.Path
            The path actually written.
        """
        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(ARTIFACT_SUFFIX)
        arrays = {
            "centers": self.centers,
            "transform/matrix": self.transform.matrix,
            "transform/eigenvectors": self.transform.eigenvectors,
            "transform/eigenvalues": self.transform.eigenvalues,
            "transform/retained_eigenvalues": self.transform.retained_eigenvalues,
        }
        if self.metric is not None:
            arrays["metric"] = self.metric
        transform_facts: dict[str, JsonValue] = {
            "rank_rtol": float(self.transform.rank_rtol),
            "threshold": float(self.transform.threshold),
            "whiten": bool(self.transform.whiten),
        }
        manifest = json_ready(
            {
                "format_version": FORMAT_VERSION,
                "arrays": sorted(arrays),
                "transform": transform_facts,
                "schema": None if self.schema is None else self.schema.to_dict(),
                "provenance": self.provenance.to_dict(),
                "criterion": self.criterion.to_dict(),
                "execution": ExecutionConfig().to_dict(),
            }
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(_MANIFEST, json.dumps(manifest, indent=2, sort_keys=True))
            for name, values in arrays.items():
                buffer = BytesIO()
                np.save(buffer, np.asarray(values), allow_pickle=False)
                archive.writestr(f"{name}.npy", buffer.getvalue())
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Quantizer:
        """Read a rule written by :meth:`save`.

        Raises
        ------
        ValueError
            When the file is not a ScoreQuant artifact, or declares a
            ``format_version`` this build does not know. An unknown version is
            refused by name rather than partially interpreted.
        """
        source = Path(path)
        with zipfile.ZipFile(source) as archive:
            names = set(archive.namelist())
            if _MANIFEST not in names:
                raise ContractError(f"{source} is not a ScoreQuant quantizer artifact")
            manifest = json.loads(archive.read(_MANIFEST))
            version = manifest.get("format_version")
            if version != FORMAT_VERSION:
                raise ContractError(
                    f"{source} declares quantizer artifact format_version {version!r}, "
                    f"but this build reads version {FORMAT_VERSION}"
                )

            def array(name: str) -> np.ndarray:
                member = f"{name}.npy"
                if member not in names:
                    raise ContractError(f"{source} is missing the {member} member")
                return np.load(BytesIO(archive.read(member)), allow_pickle=False)

            transform_facts = _mapping(manifest["transform"], "transform")
            transform = FisherTransform(
                matrix=array("transform/matrix"),
                eigenvectors=array("transform/eigenvectors"),
                eigenvalues=array("transform/eigenvalues"),
                retained_eigenvalues=array("transform/retained_eigenvalues"),
                rank_rtol=_number(transform_facts["rank_rtol"], "transform.rank_rtol"),
                threshold=_number(transform_facts["threshold"], "transform.threshold"),
                whiten=_flag(transform_facts["whiten"], "transform.whiten"),
            )
            metric = array("metric") if "metric.npy" in names else None
            schema_facts = manifest["schema"]
            schema = (
                None if schema_facts is None else ScoreSchema(tuple(schema_facts["parameters"]))
            )
            return cls(
                transform=transform,
                centers=array("centers"),
                metric=metric,
                schema=schema,
                provenance=_provenance_from(_mapping(manifest["provenance"], "provenance")),
                criterion=_criterion_from(_mapping(manifest["criterion"], "criterion")),
                execution=_loading_execution(),
            )


def _mapping(value: JsonValue, field: str) -> dict[str, JsonValue]:
    """Narrow one manifest field to a JSON object."""
    if not isinstance(value, dict):
        raise ContractError(f"quantizer artifact field {field!r} must be an object")
    return value


def _number(value: JsonValue, field: str) -> float:
    """Narrow one manifest field to a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"quantizer artifact field {field!r} must be a number")
    return float(value)


def _flag(value: JsonValue, field: str) -> bool:
    """Narrow one manifest field to a boolean."""
    if not isinstance(value, bool):
        raise ContractError(f"quantizer artifact field {field!r} must be a boolean")
    return value


def _text(value: JsonValue, field: str) -> str:
    """Narrow one manifest field to a string."""
    if not isinstance(value, str):
        raise ContractError(f"quantizer artifact field {field!r} must be a string")
    return value


def _floats(value: JsonValue, field: str) -> tuple[float, ...]:
    """Narrow one manifest field to a tuple of floats."""
    if not isinstance(value, list):
        raise ContractError(f"quantizer artifact field {field!r} must be a list")
    numbers: list[float] = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise ContractError(f"quantizer artifact field {field!r} must contain numbers")
        numbers.append(float(entry))
    return tuple(numbers)


def _provenance_from(facts: dict[str, JsonValue]) -> ScoreProvenance:
    """Rebuild score provenance from manifest facts.

    ``exact_fisher`` is deliberately not read back: it is derived from ``kind``,
    so a hand-edited artifact cannot promote an estimated ratio into exact
    Fisher semantics.
    """
    reference_point = facts.get("reference_point")
    ratio = facts.get("ratio")
    metadata = facts.get("metadata")
    kind = _text(facts.get("kind", "unknown"), "provenance.kind")
    if kind not in ("unknown", "exact", "autodiff", "estimated_ratio", "custom_estimated"):
        raise ContractError(f"unknown score provenance kind {kind!r} in quantizer artifact")
    description = facts.get("description")
    return ScoreProvenance(
        kind=kind,
        description=None if description is None else _text(description, "provenance.description"),
        reference_point=(
            None
            if reference_point is None
            else _floats(reference_point, "provenance.reference_point")
        ),
        metadata={} if metadata is None else _mapping(metadata, "provenance.metadata"),
        ratio=None if ratio is None else _ratio_from(_mapping(ratio, "provenance.ratio")),
    )


def _ratio_from(facts: dict[str, JsonValue]) -> RatioProvenance:
    """Rebuild ratio provenance field by field.

    JSON has no tuples, so every sequence field is restored explicitly rather
    than splatted back through the constructor; that also means an artifact
    carrying an unexpected key is ignored instead of raising deep inside a
    dataclass.
    """
    priors = facts.get("training_priors")
    restored_priors: tuple[float, ...] | tuple[tuple[float, float], ...] | None = None
    if isinstance(priors, list) and priors:
        if all(isinstance(row, list) for row in priors):
            pairs: list[tuple[float, float]] = []
            for row in priors:
                values = _floats(row, "ratio.training_priors")
                if len(values) != 2:
                    raise ContractError("paired training priors must have two entries per row")
                pairs.append((values[0], values[1]))
            restored_priors = tuple(pairs)
        else:
            restored_priors = _floats(priors, "ratio.training_priors")
    estimator = facts.get("estimator")
    parameterization = facts.get("parameterization")
    calibration = facts.get("calibration")
    component = facts.get("reference_component")
    return RatioProvenance(
        estimator=None if estimator is None else _text(estimator, "ratio.estimator"),
        parameterization=(
            None
            if parameterization is None
            else _parameterization(_text(parameterization, "ratio.parameterization"))
        ),
        coefficients=_optional_floats(facts.get("coefficients"), "ratio.coefficients"),
        reference_fractions=_optional_floats(
            facts.get("reference_fractions"), "ratio.reference_fractions"
        ),
        reference_component=None if component is None else _index(component),
        training_priors=restored_priors,
        calibration=None if calibration is None else _text(calibration, "ratio.calibration"),
        deltas=_optional_floats(facts.get("deltas"), "ratio.deltas"),
    )


def _optional_floats(value: JsonValue | None, field: str) -> tuple[float, ...] | None:
    """Narrow an optional manifest field to a tuple of floats."""
    return None if value is None else _floats(value, field)


def _parameterization(name: str) -> RatioParameterizationKind:
    """Narrow a recorded parameterization name to the declared vocabulary."""
    if name not in ("intensity", "mixture", "central_log_ratio"):
        raise ContractError(f"unknown ratio parameterization {name!r} in quantizer artifact")
    return name


def _criterion_from(facts: dict[str, JsonValue]) -> Criterion:
    """Rebuild the recorded criterion, refusing a name this build cannot honor."""
    name = facts.get("name")
    if name == "d_optimality":
        return DOptimality()
    if name == "normalized_trace":
        return NormalizedTrace()
    if name == "profiled_d_optimality":
        interest = facts.get("interest")
        if not isinstance(interest, list):
            raise ContractError("profiled criterion in artifact must list its interest columns")
        return ProfiledDOptimality(interest=tuple(int(_index(entry)) for entry in interest))
    raise ContractError(f"unknown criterion {name!r} in quantizer artifact")


def _index(value: JsonValue) -> int:
    """Narrow one recorded interest entry to a score-column index."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("profiled interest columns in an artifact must be integers")
    return int(value)


def _loading_execution() -> ExecutionConfig:
    """Choose the loaded rule's default runtime from the loading environment.

    JAX is preferred when it is importable and NumPy is used otherwise. The
    check locates the package without importing it, so loading stays JAX-free.
    """
    try:
        available = importlib.util.find_spec("jax") is not None
    except (ImportError, ValueError):
        available = False
    return ExecutionConfig(backend="jax" if available else "numpy")
