"""Task-explicit public partition and quantizer workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._errors import ContractError, RefusalError
from ._execution import (
    canonicalize_public,
    current_execution,
    execution_scope,
)
from ._execution import (
    xp as jnp,
)
from ._typing import ArrayLike
from ._validation import (
    _ValidatedSample,
    collapse_duplicate_scores,
    validate_n_bins,
    validate_sample,
)
from .artifact import Quantizer
from .config import (
    DExchangeConfig,
    ExecutionConfig,
    KMeansConfig,
    MahalanobisLloydConfig,
    PartitionConfig,
    QuantizerConfig,
    ScalarDPConfig,
    SoftVoronoiConfig,
)
from .criteria import Criterion, DOptimality, NormalizedTrace, ProfiledDOptimality
from .information import (
    PROFILED_RANK_ADVICE,
    binned_fisher_information,
    binned_information_is_degenerate,
    information_report,
    profiled_information_report,
)
from .partition import optimize_d_partition, optimize_profiled_d_partition
from .providers import ScoreProvider, validate_provider
from .result import InformationReport, OptimizationTrace, PartitionResult, QuantizerResult
from .solvers.common import QuantizerRun, chunked_hard_assign
from .solvers.kmeans import weighted_kmeans
from .solvers.scalar import scalar_weighted_kmeans_dp
from .solvers.soft import soft_voronoi
from .sources import (
    IntegrationSource,
    ObservationSample,
    ScoreProvenance,
    ScoreSample,
    Source,
    SourceKind,
)
from .transforms import FisherTransform, fisher_transform


@dataclass(frozen=True, slots=True)
class _PreparedFit:
    config: KMeansConfig | SoftVoronoiConfig | ScalarDPConfig
    train_sample: _ValidatedSample
    validation_sample: _ValidatedSample | None
    transform: FisherTransform
    train_coordinates: jnp.ndarray
    train_weights: jnp.ndarray
    train_objective_scores: jnp.ndarray
    all_train_coordinates: jnp.ndarray
    validation_coordinates: jnp.ndarray | None


# How many recorded center snapshots ``fit_quantizer`` re-scores with a full
# information report while building ``train_hard_retention`` /
# ``validation_hard_retention``: every snapshot ("full"), only the first and
# terminal snapshots ("endpoints"), or only the terminal snapshot ("final").
# Unscored snapshots hold ``nan`` so the history stays aligned with
# ``OptimizationTrace.steps`` regardless of the mode.
DiagnosticsMode = Literal["final", "endpoints", "full"]


@dataclass(frozen=True, slots=True)
class _FitDiagnostics:
    labels: jnp.ndarray
    train_report: InformationReport
    validation_report: InformationReport | None
    train_hard_retention: list[float]
    validation_hard_retention: list[float] | None


# Every ``Criterion`` subtype the library defines, used only to tell a
# recognized-but-mismatched criterion (``ContractError``) apart from a value
# that is not a criterion at all (``TypeError``).
_CRITERION_TYPES: tuple[type[Criterion], ...] = (DOptimality, ProfiledDOptimality, NormalizedTrace)


@dataclass(frozen=True, slots=True)
class _SolverSpec:
    """One configuration type's declared criteria for each finite task.

    An empty tuple means the type is not part of that task's own declared
    configuration contract at all: ``optimize_partition`` never accepted
    ``KMeansConfig`` and never will, so supplying it there is a Python type
    violation of that task's own signature, not a semantic choice among
    otherwise-valid alternatives.
    """

    partition_criteria: tuple[type[Criterion], ...] = ()
    quantizer_criteria: tuple[type[Criterion], ...] = ()


# The single source of truth for which (config type, criterion type) pairs
# ``optimize_partition`` and ``fit_quantizer`` accept. Both entry points
# validate against this one table instead of each hand-rolling its own
# isinstance chain, and the two tasks can (and do) disagree about which
# criteria one config type supports: ``DExchangeConfig`` and
# ``MahalanobisLloydConfig`` accept both finite criteria for
# ``optimize_partition`` but only ``DOptimality`` for ``fit_quantizer``,
# because a profiled partition has no canonical inductive rule to compile
# into a reusable quantizer.
_SOLVER_TABLE: dict[type, _SolverSpec] = {
    DExchangeConfig: _SolverSpec(
        partition_criteria=(DOptimality, ProfiledDOptimality),
        quantizer_criteria=(DOptimality,),
    ),
    MahalanobisLloydConfig: _SolverSpec(
        partition_criteria=(DOptimality, ProfiledDOptimality),
        quantizer_criteria=(DOptimality,),
    ),
    KMeansConfig: _SolverSpec(quantizer_criteria=(NormalizedTrace,)),
    SoftVoronoiConfig: _SolverSpec(quantizer_criteria=(DOptimality, ProfiledDOptimality)),
    ScalarDPConfig: _SolverSpec(quantizer_criteria=(DOptimality,)),
}


def _task_criteria(task: str, spec: _SolverSpec) -> tuple[type[Criterion], ...]:
    return spec.partition_criteria if task == "optimize_partition" else spec.quantizer_criteria


def _validate_solver(config: object, criterion: Criterion, task: str) -> None:
    """Validate one (config, criterion) pair against the declarative solver table.

    Parameters
    ----------
    config, criterion
        Resolved (non-``None``) configuration and criterion.
    task
        Either ``"optimize_partition"`` or ``"fit_quantizer"``.

    Raises
    ------
    TypeError
        ``config`` is not a type this task's own signature declares, or
        ``criterion`` is not one of the library's ``Criterion`` subtypes.
    ContractError
        Both ``config`` and ``criterion`` are individually valid, but this
        task does not implement that particular pairing. The message names
        the config, the task, and the criteria it does support.
    """
    spec = _SOLVER_TABLE.get(type(config))
    allowed = _task_criteria(task, spec) if spec is not None else ()
    if not allowed:
        recognized = sorted(
            config_type.__name__
            for config_type, entry in _SOLVER_TABLE.items()
            if _task_criteria(task, entry)
        )
        raise TypeError(f"{task} requires {' or '.join(recognized)}, got {type(config).__name__}")
    if not isinstance(criterion, _CRITERION_TYPES):
        raise TypeError(
            "criterion must be DOptimality, ProfiledDOptimality, or NormalizedTrace, "
            f"got {type(criterion).__name__}"
        )
    if not isinstance(criterion, allowed):
        names = " or ".join(sorted(t.__name__ for t in allowed))
        raise ContractError(f"{type(config).__name__} implements only {names} for {task}")


@execution_scope
def optimize_partition(
    scores: ScoreSample | ArrayLike,
    *,
    weights: ArrayLike | None = None,
    n_bins: int,
    criterion: DOptimality | ProfiledDOptimality | None = None,
    config: PartitionConfig | None = None,
    provenance: ScoreProvenance | None = None,
    initial_labels: ArrayLike | None = None,
    execution: ExecutionConfig | None = None,
) -> PartitionResult:
    """Optimize labels of one fixed score table without prediction semantics.

    Both finite criteria accept either the exact positive-gain exchange or the
    guarded Mahalanobis-Lloyd solver; the guarded batch never accepts a step
    that the exactly rebuilt objective does not certify.

    Parameters
    ----------
    scores
        Either a :class:`~scorequant.ScoreSample` -- the same weighted score
        law :func:`fit_quantizer` takes, carrying its own weights, schema and
        provenance -- or a raw score array, in which case ``weights`` and
        ``provenance`` supply those separately. Passing a sample together with
        either keyword is rejected rather than silently resolved.

        An observation source is deliberately not accepted here: converting
        observations to scores stays an explicit ``provider.score(X)`` so the
        fixed-sample boundary remains visible.
    weights, n_bins, criterion, config, provenance
        Fixed-sample assignment contract described in the API guide.
    initial_labels
        Optional starting labeling with shape ``[N]`` and values in
        ``[0, n_bins)``, for example
        ``EfficientScoreBound.labels``. Zero-weight rows carry
        no measure and their labels are ignored; identical score rows are merged
        before the solver runs and must therefore already agree on their bin,
        and every requested cell must remain nonempty afterwards. Supplied
        labels replace the seeding of the first exchange restart only, so
        ``init`` and ``initializer_restarts`` still govern any further restart; the guarded
        Mahalanobis-Lloyd solver starts from them directly.
    """
    del execution
    sample = _partition_sample(scores, weights, provenance)
    resolved_criterion = DOptimality() if criterion is None else criterion
    resolved_config = DExchangeConfig() if config is None else config
    _validate_solver(resolved_config, resolved_criterion, "optimize_partition")
    if isinstance(resolved_criterion, DOptimality):
        result = optimize_d_partition(
            sample.scores,
            weights=sample.weights,
            n_bins=n_bins,
            config=resolved_config,
            provenance=sample.provenance,
            initial_labels=initial_labels,
            schema=sample.schema,
        )
    else:
        # ``_validate_solver`` accepted the pair, and the only other criterion it
        # can have accepted for this task is ProfiledDOptimality. Names become
        # score columns here, once, so no solver or report resolves them again.
        result = optimize_profiled_d_partition(
            sample.scores,
            weights=sample.weights,
            n_bins=n_bins,
            criterion=resolved_criterion.resolve(sample.schema),
            config=resolved_config,
            provenance=sample.provenance,
            initial_labels=initial_labels,
            schema=sample.schema,
        )
    return canonicalize_public(result)


@execution_scope
def fit_quantizer(
    source: Source,
    *,
    provider: ScoreProvider | None = None,
    validation: Source | None = None,
    n_bins: int,
    criterion: Criterion | None = None,
    config: QuantizerConfig | None = None,
    diagnostics: DiagnosticsMode = "endpoints",
    execution: ExecutionConfig | None = None,
) -> QuantizerResult:
    """Fit a reusable hard rule from an empirical or bounded score law.

    A score callback alone is deliberately insufficient: observations must be
    paired with an empirical or integration source that defines their measure.

    Parameters
    ----------
    provider
        The observation-to-score map for an observation or integration source.
        It is the object contract :class:`~scorequant.ScoreProvider` describes,
        not a score array or a bare callable, and it is rejected when ``source``
        is already a :class:`~scorequant.ScoreSample`.
    diagnostics
        How much of the recorded center history to re-score into
        ``trace.train_hard_retention`` and ``trace.validation_hard_retention``.
        ``"final"`` scores only the terminal centers (one full-dataset pass),
        ``"endpoints"`` (the default) scores the first and terminal centers
        (two passes), and ``"full"`` scores every recorded snapshot, matching
        the historical behavior. Snapshots that are not scored hold ``nan``,
        so the returned history always stays aligned with ``trace.steps``.
        This only affects diagnostic reporting; it never changes ``centers``,
        ``labels``, or either report.
    """
    del execution
    resolved_execution = current_execution()
    train, source_kind = _materialize_source(source, provider)
    validation_sample = None
    if validation is not None:
        validation_provider = None if isinstance(validation, ScoreSample) else provider
        validation_sample, _ = _materialize_source(validation, validation_provider)
        _validate_validation_sample(train, validation_sample)
    resolved_config = DExchangeConfig() if config is None else config
    resolved_criterion: Criterion = DOptimality() if criterion is None else criterion
    _validate_solver(resolved_config, resolved_criterion, "fit_quantizer")
    if isinstance(resolved_criterion, ProfiledDOptimality):
        # Names become score columns once, here, so no solver or report resolves
        # them again.
        resolved_criterion = resolved_criterion.resolve(train.schema)

    if isinstance(resolved_config, (DExchangeConfig, MahalanobisLloydConfig)):
        return _fit_compiled_quantizer(
            train,
            validation_sample,
            n_bins=n_bins,
            config=resolved_config,
            source_kind=source_kind,
            execution=resolved_execution,
        )
    return _fit_geometric_quantizer(
        train,
        validation_sample,
        n_bins=n_bins,
        criterion=resolved_criterion,
        config=resolved_config,
        diagnostics=diagnostics,
        source_kind=source_kind,
        execution=resolved_execution,
    )


def _fit_compiled_quantizer(
    train: ScoreSample,
    validation: ScoreSample | None,
    *,
    n_bins: int,
    config: DExchangeConfig | MahalanobisLloydConfig,
    source_kind: SourceKind,
    execution: ExecutionConfig,
) -> QuantizerResult:
    """Fit by finite D exchange and compile the exchange-stable partition (Theorem 3).

    ``_validate_solver`` has already restricted this pairing to ``DOptimality``:
    finite profiled-D exchange has no implicit inductive rule, so a profiled fit
    must go through ``optimize_partition`` or ``SoftVoronoiConfig`` instead.
    """
    partition = optimize_d_partition(
        train.scores,
        weights=train.weights,
        n_bins=n_bins,
        config=config,
        provenance=train.provenance,
        schema=train.schema,
    )
    rule = partition.compile_quantizer(execution=execution)
    validation_report = (
        None
        if validation is None
        else information_report(
            validation.scores,
            rule.predict_scores(validation.scores),
            validation.weights,
            n_bins=n_bins,
            rank_rtol=config.rank_rtol,
        )
    )
    return canonicalize_public(
        QuantizerResult(
            quantizer=rule,
            criterion=partition.criterion,
            config=config,
            execution=execution,
            trace=_compiled_trace(partition),
            labels=partition.labels,
            train_report=partition.train_report,
            validation_report=validation_report,
            provenance=train.provenance,
            hardening_gap=0.0,
            source_kind=source_kind,
        )
    )


def _require_profiled_fit_regular(
    prepared: _PreparedFit, labels: jnp.ndarray, n_bins: int, criterion: Criterion
) -> None:
    """Refuse a profiled labeling whose binned information cannot be profiled.

    Must run after hard assignment and before any report reads ``labels`` (the
    retention history scores snapshots through the same profiled report, so a
    check placed downstream never gets to run). The rank ceiling is a
    bin-budget fact, so deciding for the final labeling decides for every
    snapshot. This is reachable because the soft solver checks ``n_bins`` only
    against the Fisher rank, which the vacuous configuration of
    CE-DS-MARGINS-RANK-VACUITY-001 satisfies.
    """
    if not isinstance(criterion, ProfiledDOptimality):
        return
    information = binned_fisher_information(
        prepared.train_sample.scores, labels, prepared.train_sample.weights, n_bins=n_bins
    )
    if binned_information_is_degenerate(information):
        raise RefusalError(
            f"profiled-D fit is degenerate: {n_bins} bins cannot generate "
            f"nonsingular {prepared.train_sample.scores.shape[1]}-dimensional binned "
            f"information. {PROFILED_RANK_ADVICE}",
            "CE-DS-MARGINS-RANK-VACUITY-001",
        )


def _fit_geometric_quantizer(
    train: ScoreSample,
    validation: ScoreSample | None,
    *,
    n_bins: int,
    criterion: Criterion,
    config: KMeansConfig | SoftVoronoiConfig | ScalarDPConfig,
    diagnostics: DiagnosticsMode,
    source_kind: SourceKind,
    execution: ExecutionConfig,
) -> QuantizerResult:
    prepared = _prepare_score_fit(
        train,
        validation,
        n_bins=n_bins,
        config=config,
    )
    run = _run_geometric_quantizer(prepared, n_bins, criterion)
    labels = chunked_hard_assign(prepared.all_train_coordinates, run.centers)
    # A report describes a labeling and does not judge it, so
    # profiled_information_report would hand back a rule that answers every
    # profiled question with zero -- and on a state this degenerate it raises
    # from its nuisance-block guard instead, blaming the parameterization for
    # what is a bin-budget fact. Refuse before anything else reads this
    # labeling instead.
    _require_profiled_fit_regular(prepared, labels, n_bins, criterion)
    fit_diagnostics = _build_fit_diagnostics(
        prepared, run, labels, n_bins, criterion, diagnostics=diagnostics
    )
    train_profiled_report = None
    validation_profiled_report = None
    hard_retention = fit_diagnostics.train_report.geometric_mean_retention
    if isinstance(criterion, ProfiledDOptimality):
        train_profiled_report = profiled_information_report(
            prepared.train_sample.scores,
            fit_diagnostics.labels,
            interest=criterion.interest_indices,
            weights=prepared.train_sample.weights,
            n_bins=n_bins,
            schema=train.schema,
        )
        hard_retention = train_profiled_report.geometric_mean_retention
        if prepared.validation_sample is not None and prepared.validation_coordinates is not None:
            validation_profiled_report = profiled_information_report(
                prepared.validation_sample.scores,
                chunked_hard_assign(prepared.validation_coordinates, run.centers),
                interest=criterion.interest_indices,
                weights=prepared.validation_sample.weights,
                n_bins=n_bins,
                schema=train.schema,
            )
    hardening_gap = None
    if run.soft_retention_history:
        hardening_gap = run.soft_retention_history[-1] - hard_retention
    return canonicalize_public(
        QuantizerResult(
            quantizer=Quantizer(
                transform=prepared.transform,
                centers=run.centers,
                metric=None,
                schema=train.schema,
                provenance=train.provenance,
                criterion=criterion,
                execution=execution,
            ),
            criterion=criterion,
            config=config,
            execution=execution,
            trace=_build_optimization_trace(run, fit_diagnostics),
            labels=fit_diagnostics.labels,
            train_report=fit_diagnostics.train_report,
            validation_report=fit_diagnostics.validation_report,
            provenance=train.provenance,
            hardening_gap=hardening_gap,
            source_kind=source_kind,
            train_profiled_report=train_profiled_report,
            validation_profiled_report=validation_profiled_report,
        )
    )


def _compiled_trace(partition: PartitionResult) -> OptimizationTrace:
    """Describe a compiled partition as a quantizer trace.

    Compilation is bookkeeping, not a second optimization, so the centers and
    cell weights are constant across the trace; what varies is the finite
    solver's own objective trajectory, reported here so a compiled rule and a
    directly fitted one expose the same history shape.
    """
    if partition.transformed_centers is None:
        raise ContractError("D compilation geometry is unavailable")
    steps = partition.objective_history.shape[0]
    return OptimizationTrace(
        steps=np.arange(steps),
        centers=np.repeat(partition.transformed_centers[None, :, :], steps, axis=0),
        objective=partition.objective_history,
        bin_weights=np.repeat(partition.cell_weights[None, :], steps, axis=0),
        train_hard_retention=np.full(
            partition.objective_history.shape,
            partition.train_report.geometric_mean_retention,
            dtype=partition.cell_weights.dtype,
        ),
        objective_label="logdet_retained",
    )


def _validate_validation_sample(train: ScoreSample, validation: ScoreSample) -> None:
    """Reject a validation sample that does not describe the same parameters.

    A column count alone catches only the coarsest mismatch. When both samples
    name their coordinates, disagreeing names are a reordering the count cannot
    see, and silently scoring against it would report a meaningless retention.
    """
    if validation.scores.shape[1] != train.scores.shape[1]:
        raise ContractError("validation scores must use the training parameter order")
    if (
        train.schema is not None
        and validation.schema is not None
        and train.schema.parameters != validation.schema.parameters
    ):
        raise ContractError(
            "validation scores must use the training parameter order; training names "
            f"{', '.join(train.schema.parameters)} but validation names "
            f"{', '.join(validation.schema.parameters)}"
        )


def _partition_sample(
    scores: ScoreSample | ArrayLike,
    weights: ArrayLike | None,
    provenance: ScoreProvenance | None,
) -> ScoreSample:
    """Canonicalize the fixed-sample input into one weighted score law.

    The array shorthand stays supported for the simple case, but a supplied
    ``ScoreSample`` already carries its weights and provenance, so combining
    the two forms is a contradiction rather than an override.
    """
    if isinstance(scores, ScoreSample):
        conflicting = [
            name
            for name, value in (("weights", weights), ("provenance", provenance))
            if value is not None
        ]
        if conflicting:
            raise ContractError(
                f"{' and '.join(conflicting)} must be omitted when scores is a "
                "ScoreSample; the sample already carries them"
            )
        return scores
    return ScoreSample(scores, weights, provenance=provenance)


def _materialize_source(
    source: Source, provider: ScoreProvider | None
) -> tuple[ScoreSample, SourceKind]:
    if isinstance(source, ScoreSample):
        if provider is not None:
            raise ContractError("provider must be omitted when source is ScoreSample")
        return source, "score_sample"
    if isinstance(source, IntegrationSource):
        if provider is None:
            raise ContractError("IntegrationSource requires a score provider")
        validate_provider(provider)
        observations = source.materialize()
        return (
            ScoreSample(
                provider.score(observations.observations),
                observations.weights,
                schema=getattr(provider, "schema", None),
                provenance=provider.provenance,
            ),
            "integration_source",
        )
    if isinstance(source, ObservationSample):
        if provider is None:
            raise ContractError("ObservationSample requires a score provider")
        validate_provider(provider)
        return (
            ScoreSample(
                provider.score(source.observations),
                source.weights,
                schema=getattr(provider, "schema", None),
                provenance=provider.provenance,
            ),
            "observation_sample",
        )
    raise TypeError("source must be ScoreSample, ObservationSample, or IntegrationSource")


def _prepare_score_fit(
    source: ScoreSample,
    validation: ScoreSample | None,
    *,
    n_bins: int,
    config: KMeansConfig | SoftVoronoiConfig | ScalarDPConfig,
) -> _PreparedFit:
    train_sample = validate_sample(source.scores, source.weights)
    validate_n_bins(n_bins, train_sample.n_effective)
    effective_scores, effective_weights, _ = collapse_duplicate_scores(
        train_sample.effective_scores, train_sample.effective_weights
    )
    if n_bins > effective_scores.shape[0]:
        raise ContractError("n_bins exceeds distinct positive-weight score rows")
    full_information = jnp.einsum(
        "n,np,nq->pq", effective_weights, effective_scores, effective_scores
    )
    transform = fisher_transform(
        full_information,
        whiten=config.whiten,
        rank_rtol=config.rank_rtol,
    )
    train_coordinates = transform.apply(effective_scores)
    validation_sample = (
        None
        if validation is None
        else validate_sample(
            validation.scores,
            validation.weights,
            expected_features=train_sample.scores.shape[1],
        )
    )
    return _PreparedFit(
        config=config,
        train_sample=train_sample,
        validation_sample=validation_sample,
        transform=transform,
        train_coordinates=train_coordinates,
        train_weights=effective_weights,
        train_objective_scores=effective_scores,
        all_train_coordinates=transform.apply(train_sample.scores),
        validation_coordinates=(
            None if validation_sample is None else transform.apply(validation_sample.scores)
        ),
    )


def _run_geometric_quantizer(
    prepared: _PreparedFit, n_bins: int, criterion: Criterion
) -> QuantizerRun:
    if isinstance(prepared.config, KMeansConfig):
        return weighted_kmeans(
            prepared.train_coordinates,
            prepared.train_weights,
            n_bins,
            prepared.config,
        )
    if isinstance(prepared.config, ScalarDPConfig):
        return scalar_weighted_kmeans_dp(
            prepared.train_coordinates,
            prepared.train_weights,
            n_bins,
            prepared.config,
        )
    # SoftVoronoiConfig is the only remaining case: ``_validate_solver`` has
    # already restricted its criterion to this union before this function
    # ever runs, so the assertion only narrows the type for ``soft_voronoi``.
    assert isinstance(criterion, (DOptimality, ProfiledDOptimality))
    return soft_voronoi(
        prepared.train_coordinates,
        prepared.train_objective_scores,
        prepared.train_weights,
        n_bins,
        prepared.transform.rank,
        criterion,
        prepared.config,
    )


def _diagnostics_snapshot_indices(n_snapshots: int, diagnostics: DiagnosticsMode) -> set[int]:
    """Return which recorded center snapshots ``diagnostics`` re-scores.

    ``"final"`` selects only the terminal snapshot, ``"endpoints"`` the first
    and terminal snapshots (deduplicated when there is only one snapshot),
    and ``"full"`` every snapshot.
    """
    if n_snapshots == 0:
        return set()
    last = n_snapshots - 1
    if diagnostics == "full":
        return set(range(n_snapshots))
    if diagnostics == "endpoints":
        return {0, last}
    return {last}


def _hard_retention_history(
    scores: jnp.ndarray,
    weights: jnp.ndarray,
    transformed_scores: jnp.ndarray,
    run: QuantizerRun,
    *,
    rank_rtol: float | None,
    criterion: Criterion,
    diagnostics: DiagnosticsMode,
) -> list[float]:
    """Re-score a subset of recorded center snapshots into a retention history.

    A full-dataset information report costs an ``O(N)`` pass, so scoring every
    recorded snapshot is expensive for long soft-Voronoi schedules. Unscored
    snapshots hold ``nan`` rather than being omitted, so the returned list
    always has the same length as ``run.center_history`` and stays aligned
    with ``OptimizationTrace.steps``.
    """
    selected = _diagnostics_snapshot_indices(len(run.center_history), diagnostics)
    values: list[float] = [float("nan")] * len(run.center_history)
    for index in selected:
        centers = run.center_history[index]
        labels = chunked_hard_assign(transformed_scores, centers)
        if isinstance(criterion, ProfiledDOptimality):
            retention = profiled_information_report(
                scores,
                labels,
                interest=criterion.interest_indices,
                weights=weights,
                n_bins=centers.shape[0],
            ).geometric_mean_retention
        else:
            retention = information_report(
                scores,
                labels,
                weights,
                n_bins=centers.shape[0],
                rank_rtol=rank_rtol,
            ).geometric_mean_retention
        values[index] = retention
    return values


def _build_fit_diagnostics(
    prepared: _PreparedFit,
    run: QuantizerRun,
    labels: jnp.ndarray,
    n_bins: int,
    criterion: Criterion,
    *,
    diagnostics: DiagnosticsMode,
) -> _FitDiagnostics:
    train_hard = _hard_retention_history(
        prepared.train_sample.scores,
        prepared.train_sample.weights,
        prepared.all_train_coordinates,
        run,
        rank_rtol=prepared.config.rank_rtol,
        criterion=criterion,
        diagnostics=diagnostics,
    )
    train_report = information_report(
        prepared.train_sample.scores,
        labels,
        prepared.train_sample.weights,
        n_bins=n_bins,
        rank_rtol=prepared.config.rank_rtol,
    )
    if prepared.validation_sample is None or prepared.validation_coordinates is None:
        return _FitDiagnostics(labels, train_report, None, train_hard, None)
    validation_hard = _hard_retention_history(
        prepared.validation_sample.scores,
        prepared.validation_sample.weights,
        prepared.validation_coordinates,
        run,
        rank_rtol=prepared.config.rank_rtol,
        criterion=criterion,
        diagnostics=diagnostics,
    )
    validation_report = information_report(
        prepared.validation_sample.scores,
        chunked_hard_assign(prepared.validation_coordinates, run.centers),
        prepared.validation_sample.weights,
        n_bins=n_bins,
        rank_rtol=prepared.config.rank_rtol,
    )
    return _FitDiagnostics(
        labels,
        train_report,
        validation_report,
        train_hard,
        validation_hard,
    )


def _build_optimization_trace(run: QuantizerRun, diagnostics: _FitDiagnostics) -> OptimizationTrace:
    return OptimizationTrace(
        steps=jnp.asarray(run.steps),
        centers=jnp.stack(run.center_history),
        objective=jnp.asarray(run.objective_history),
        bin_weights=jnp.stack(run.bin_weight_history),
        train_hard_retention=jnp.asarray(diagnostics.train_hard_retention),
        objective_label=run.objective_label,
        validation_hard_retention=(
            None
            if diagnostics.validation_hard_retention is None
            else jnp.asarray(diagnostics.validation_hard_retention)
        ),
        soft_retention=(
            None if run.soft_retention_history is None else jnp.asarray(run.soft_retention_history)
        ),
        temperatures=(
            None if run.temperature_history is None else jnp.asarray(run.temperature_history)
        ),
        gradient_norms=(
            None if run.gradient_norm_history is None else jnp.asarray(run.gradient_norm_history)
        ),
    )
