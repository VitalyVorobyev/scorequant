"""Immutable public result objects.

Diagnostic and certificate report types (``InformationReport`` and friends)
now live in ``reports.py``, which does not import this module. That is what
lets ``QuantizerResult.evaluate_scores`` import ``information_report`` at
module scope below instead of inside the method: ``result -> information ->
reports`` is a chain, not a cycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._errors import ContractError, RefusalError
from ._execution import canonicalize_public, use_execution
from ._json import json_ready
from ._predict import chunked_predict_labels
from ._typing import ArrayLike, JsonValue
from .artifact import Quantizer
from .config import ExecutionConfig, MahalanobisLloydConfig, PartitionConfig, QuantizerConfig
from .criteria import Criterion, DOptimality, ProfiledDOptimality
from .reports import (
    EfficientScoreBound,
    GeometryReport,
    InformationReport,
    PartitionCertificate,
    ProfiledGeometryReport,
    ProfiledInformationReport,
    StabilityReport,
)
from .sources import InformationKind, ScoreProvenance, ScoreSchema, SourceKind
from .transforms import FisherTransform

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "EfficientScoreBound",
    "GeometryReport",
    "InformationReport",
    "OptimizationTrace",
    "PartitionCertificate",
    "PartitionResult",
    "ProfiledGeometryReport",
    "ProfiledInformationReport",
    "QuantizerResult",
    "StabilityReport",
]


@dataclass(frozen=True, slots=True)
class OptimizationTrace:
    """Store aggregate quantizer optimization history.

    Attributes
    ----------
    objective_label
        Units of ``objective``. Solvers do not share one objective convention:
        ``"whitened_sse"`` is a minimized weighted within-cell squared error in
        Fisher-whitened coordinates, ``"logdet_retained"`` is a maximized
        retained log determinant, and ``"profiled_logdet"`` is a maximized
        profiled log determinant. Never compare two traces across labels.
    """

    steps: np.ndarray
    centers: np.ndarray
    objective: np.ndarray
    bin_weights: np.ndarray
    train_hard_retention: np.ndarray
    objective_label: str
    validation_hard_retention: np.ndarray | None = None
    soft_retention: np.ndarray | None = None
    temperatures: np.ndarray | None = None
    gradient_norms: np.ndarray | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible representation."""
        return json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class QuantizerResult:
    """Represent a reusable hard rule on raw score vectors."""

    quantizer: Quantizer
    criterion: Criterion
    config: QuantizerConfig
    execution: ExecutionConfig
    trace: OptimizationTrace
    labels: np.ndarray
    train_report: InformationReport
    validation_report: InformationReport | None
    provenance: ScoreProvenance
    hardening_gap: float | None = None
    source_kind: SourceKind = "score_sample"
    train_profiled_report: ProfiledInformationReport | None = None
    validation_profiled_report: ProfiledInformationReport | None = None

    @property
    def centers(self) -> np.ndarray:
        """Return the frozen cell centers in transformed coordinates."""
        return self.quantizer.centers

    @property
    def metric(self) -> np.ndarray | None:
        """Return the frozen common Mahalanobis metric, if the rule carries one."""
        return self.quantizer.metric

    @property
    def transform(self) -> FisherTransform:
        """Return the informative-subspace projection the rule was fitted in."""
        return self.quantizer.transform

    @property
    def schema(self) -> ScoreSchema | None:
        """Return the names of the raw score coordinates, when they were declared."""
        return self.quantizer.schema

    @property
    def n_bins(self) -> int:
        """Return the number of hard output labels."""
        return self.quantizer.n_bins

    @property
    def rank(self) -> int:
        """Return the numerically informative score-space rank."""
        return self.quantizer.rank

    @property
    def information_kind(self) -> InformationKind:
        """Describe whether supplied-score matrices justify exact Fisher language.

        ``"supplied_score_surrogate"`` means the reported matrices measure
        ``Var(E[s_hat | q])`` for the supplied vectors ``s_hat``, which is
        the model's Fisher information only when ``s_hat`` equals the model
        score.
        """
        return "exact_fisher" if self.provenance.exact_fisher else "supplied_score_surrogate"

    def predict_scores(
        self,
        scores: ArrayLike,
        *,
        execution: ExecutionConfig | None = None,
    ) -> np.ndarray:
        """Assign raw score rows with the frozen score-space rule.

        Rows are assigned in memory-bounded chunks so that predicting on a
        large sample never materializes the full ``[n_rows, n_bins, rank]``
        distance tensor at once; each row's distance and nearest-center
        argmin are independent of every other row, so chunking is
        bit-identical to the unchunked computation.
        """
        return self.quantizer.predict_scores(scores, execution=execution or self.execution)

    def evaluate_scores(
        self,
        scores: ArrayLike,
        weights: ArrayLike | None = None,
        *,
        execution: ExecutionConfig | None = None,
    ) -> InformationReport:
        """Evaluate the frozen rule on a new weighted score sample."""
        return self.quantizer.evaluate_scores(
            scores,
            weights,
            rank_rtol=self.config.rank_rtol,
            execution=execution or self.execution,
        )

    def report(self) -> InformationReport:
        """Return final hard training-sample diagnostics."""
        return self.train_report

    def to_dict(self) -> dict[str, JsonValue]:
        """Return JSON-ready in-memory state, not a versioned artifact format."""
        return json_ready(
            {
                "centers": self.centers,
                "metric": self.metric,
                "transform": self.transform.to_dict(),
                "criterion": self.criterion.to_dict(),
                "config": self.config.to_dict(),
                "execution": self.execution.to_dict(),
                "trace": self.trace.to_dict(),
                "labels": self.labels,
                "train_report": self.train_report.to_dict(),
                "validation_report": (
                    None if self.validation_report is None else self.validation_report.to_dict()
                ),
                "provenance": self.provenance.to_dict(),
                "information_kind": self.information_kind,
                "schema": None if self.schema is None else self.schema.to_dict(),
                "hardening_gap": self.hardening_gap,
                "source_kind": self.source_kind,
                "train_profiled_report": (
                    None
                    if self.train_profiled_report is None
                    else self.train_profiled_report.to_dict()
                ),
                "validation_profiled_report": (
                    None
                    if self.validation_profiled_report is None
                    else self.validation_profiled_report.to_dict()
                ),
            }
        )

    def plot_summary(self, scores: ArrayLike, weights: ArrayLike | None = None) -> Figure:
        """Create the optional score-space summary figure."""
        from .visualization import plot_summary

        return plot_summary(self, scores, weights)


@dataclass(frozen=True, slots=True)
class PartitionResult:
    """Represent optimized labels of one fixed weighted score table.

    The solver counters are reported separately and never merged: ``scans`` and
    ``accepted_moves`` describe exchange work, while ``lloyd_iterations`` and
    ``accepted_lloyd_steps`` describe guarded batch relabelings. Both stay zero
    for a solver that performs neither. ``objective_history`` records every
    accepted step of every phase in order and is strictly increasing.

    Geometry diagnostics are criterion-specific and never shared: a
    ``DOptimality`` result carries ``geometry`` and no ``profiled_geometry``, a
    ``ProfiledDOptimality`` result carries ``profiled_geometry`` and no
    ``geometry``. The two measure different objects — a strict Mahalanobis
    Voronoi rule that exchange stability guarantees, and an efficient
    semimetric whose Voronoi rule a stable profiled partition may violate — so
    one name for both would claim an implication that does not hold.

    ``exchange_stable`` and ``geometry`` are verdicts at ``config.gain_tolerance``,
    which ``GeometryReport`` records, and never claims at tolerance zero. A
    finite solver stops at that threshold, so verifying its output against a
    stricter one would reject partitions it legitimately converged on.
    """

    labels: np.ndarray
    training_scores: np.ndarray
    cell_weights: np.ndarray
    cell_score_sums: np.ndarray
    cell_score_means: np.ndarray
    information_full: np.ndarray
    information_partitioned: np.ndarray
    objective: float
    transform: FisherTransform
    transformed_centers: np.ndarray | None
    metric: np.ndarray | None
    criterion: DOptimality | ProfiledDOptimality
    config: PartitionConfig
    execution: ExecutionConfig
    train_report: InformationReport
    provenance: ScoreProvenance
    accepted_moves: int
    scans: int
    exchange_stable: bool
    best_remaining_gain: float
    objective_history: np.ndarray
    positive_weight_mask: np.ndarray
    lloyd_iterations: int = 0
    accepted_lloyd_steps: int = 0
    geometry: GeometryReport | None = None
    profiled_report: ProfiledInformationReport | None = None
    profiled_geometry: ProfiledGeometryReport | None = None
    schema: ScoreSchema | None = None

    @property
    def n_bins(self) -> int:
        """Return the number of nonempty requested cells."""
        return int(self.cell_weights.shape[0])

    @property
    def rank(self) -> int:
        """Return the numerically informative score-space rank."""
        return self.transform.rank

    @property
    def information_kind(self) -> InformationKind:
        """Describe whether supplied-score matrices justify exact Fisher language.

        ``"supplied_score_surrogate"`` means the reported matrices measure
        ``Var(E[s_hat | q])`` for the supplied vectors ``s_hat``, which is
        the model's Fisher information only when ``s_hat`` equals the model
        score.
        """
        return "exact_fisher" if self.provenance.exact_fisher else "supplied_score_surrogate"

    def report(self) -> InformationReport:
        """Return supplied-score information for the fixed partition."""
        return self.train_report

    def compile_quantizer(
        self,
        *,
        execution: ExecutionConfig | None = None,
    ) -> Quantizer:
        r"""Compile an exchange-stable D partition into its canonical rule.

        Theorem 3 makes a one-point-exchange-stable, nonsingular D partition a
        self-consistent \(I^{-1}\)-Mahalanobis Voronoi partition of the observed
        rows, so the compiled rule
        \(\hat q(s)=\arg\min_b (s-\mu_b)^\top I^{-1}(s-\mu_b)\) is bookkeeping
        rather than a new fit. The theorem is exact; the partition behind it is
        not. A finite solver stops at ``config.gain_tolerance``, so the
        guarantee this method can offer is self-consistency *at that tolerance*:
        each admissible individual prediction disagreement has relocation gain at
        most ``gain_tolerance``, priced against the original partition. Successful
        solver result construction separately checks that every positive-weight
        disagreement is admissible. Positive tolerance alone does not guarantee
        compilability: a singleton tie can be refused
        (``CE-D-COMPILE-SINGLETON-TIE-001``). No bound on simultaneous relabeling,
        global suboptimality or population loss follows from the tolerance
        (``CE-D-COMPILE-BATCH-TOLERANCE-001``). These are numerical exact-formula
        diagnostics in the retained score subspace, not interval arithmetic.

        Boundary ties are never resolved here. ``predict_scores`` keeps the
        ordinary ``argmin`` rule, which is deterministic and breaks a tie toward
        the lowest cell index; the tolerance governs verification, not
        assignment.

        Returns
        -------
        Quantizer
            The deployable rule itself -- the partition's centers, metric and
            transform. It is a rule, not a new fit, so it carries no labels,
            reports or history; those already belong to this partition.

        Raises
        ------
        ContractError
            When the compilation geometry is missing.
        RefusalError
            When the criterion is not ``DOptimality``, when the partition is not
            exchange-stable, or when the rule relabels a training row by more
            than ``gain_tolerance``.
        """
        resolved_execution = execution or self.execution
        if not isinstance(self.criterion, DOptimality):
            raise RefusalError(
                "finite profiled-D labels have no canonical inductive compilation; "
                "fit an explicit quantizer instead",
                "CE-DS-GLOBAL-GEOMETRY-001",
            )
        if not self.exchange_stable:
            remedy = (
                "set guard='exchange'"
                if isinstance(self.config, MahalanobisLloydConfig)
                else "raise max_scans, or leave it unset to run until stability"
            )
            raise RefusalError(
                "only an exchange-stable D partition can be compiled; inspect "
                f"best_remaining_gain and {remedy}",
                "CE-D-VORONOI-CONVERSE-001",
            )
        if self.transformed_centers is None or self.metric is None:
            raise ContractError("D compilation geometry is unavailable")
        with use_execution(resolved_execution):
            coordinates = self.transform.apply(self.training_scores, execution=resolved_execution)
            predicted = chunked_predict_labels(coordinates, self.transformed_centers, self.metric)
        positive = np.asarray(self.positive_weight_mask)
        if not np.array_equal(np.asarray(predicted)[positive], np.asarray(self.labels)[positive]):
            # Only the solver-side certificate can price a disagreement, because
            # only it holds the row weights the exact relocation gain needs.
            if self.geometry is None or not self.geometry.voronoi_consistent:
                raise RefusalError(
                    "D compilation is degenerate: the compiled rule relabels training rows "
                    "by more than the gain tolerance the partition was certified at; "
                    "inspect geometry.maximum_violation_gain",
                    "CE-D-UNMERGED-DUPLICATES-001",
                )
        return canonicalize_public(
            Quantizer(
                transform=self.transform,
                centers=self.transformed_centers,
                metric=self.metric,
                schema=self.schema,
                provenance=self.provenance,
                criterion=self.criterion,
                execution=resolved_execution,
            )
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the fixed-sample result as JSON-ready data."""
        return json_ready(
            {
                "labels": self.labels,
                "cell_weights": self.cell_weights,
                "cell_score_sums": self.cell_score_sums,
                "cell_score_means": self.cell_score_means,
                "information_full": self.information_full,
                "information_partitioned": self.information_partitioned,
                "objective": self.objective,
                "transform": self.transform.to_dict(),
                "transformed_centers": self.transformed_centers,
                "metric": self.metric,
                "criterion": self.criterion.to_dict(),
                "config": self.config.to_dict(),
                "execution": self.execution.to_dict(),
                "train_report": self.train_report.to_dict(),
                "provenance": self.provenance.to_dict(),
                "information_kind": self.information_kind,
                "schema": None if self.schema is None else self.schema.to_dict(),
                "accepted_moves": self.accepted_moves,
                "scans": self.scans,
                "lloyd_iterations": self.lloyd_iterations,
                "accepted_lloyd_steps": self.accepted_lloyd_steps,
                "exchange_stable": self.exchange_stable,
                "best_remaining_gain": self.best_remaining_gain,
                "objective_history": self.objective_history,
                "geometry": (None if self.geometry is None else self.geometry.to_dict()),
                "profiled_report": (
                    None if self.profiled_report is None else self.profiled_report.to_dict()
                ),
                "profiled_geometry": (
                    None if self.profiled_geometry is None else self.profiled_geometry.to_dict()
                ),
            }
        )
