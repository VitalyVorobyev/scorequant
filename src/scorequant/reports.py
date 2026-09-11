"""Immutable diagnostic and certificate reports.

These types carry no dependency on ``result.py`` or ``information.py`` at
runtime, which is what lets ``information.py`` build them and ``result.py``
consume ``information_report`` without the two modules importing each other
at module scope. ``EfficientScoreBound.gap_to`` still describes its argument
as a ``PartitionResult`` for documentation, but only under ``TYPE_CHECKING``:
at runtime it duck-types the three attributes it reads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ._errors import ContractError
from ._json import json_ready
from ._typing import JsonValue
from .criteria import DOptimality, ProfiledDOptimality
from .sources import ScoreSchema

if TYPE_CHECKING:
    from .result import PartitionResult


@dataclass(frozen=True, slots=True)
class InformationReport:
    """Report supplied-score retention and per-bin diagnostics for one sample."""

    fisher_unbinned: np.ndarray
    fisher_binned: np.ndarray
    retained_matrix: np.ndarray
    retained_eigenvalues: np.ndarray
    arithmetic_mean_retention: float
    geometric_mean_retention: float
    logdet_retention: float
    bin_weights: np.ndarray
    bin_counts: np.ndarray
    bin_effective_sample_sizes: np.ndarray
    effective_rank: int
    rank_threshold: float
    psd_residual_min_eigenvalue: float

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible representation."""
        return json_ready(asdict(self))

    def __str__(self) -> str:
        """Format headline supplied-score diagnostics."""
        eigenvalues = ", ".join(f"{float(value):.4f}" for value in self.retained_eigenvalues)
        return (
            "ScoreQuant information report\n"
            f"  effective rank: {self.effective_rank}\n"
            f"  D-efficiency: {self.geometric_mean_retention:.6f}\n"
            f"  mean retention: {self.arithmetic_mean_retention:.6f}\n"
            f"  retained eigenvalues: [{eigenvalues}]\n"
            f"  minimum PSD residual eigenvalue: {self.psd_residual_min_eigenvalue:.3e}"
        )


type RetentionUncertaintyStatus = Literal[
    "ok",
    "insufficient_cells",
    "singular_full_information",
    "singular_retained_information",
    "degenerate_variance",
]


@dataclass(frozen=True, slots=True)
class RetentionUncertainty:
    r"""Sampling uncertainty of a frozen rule's geometric-mean retention on held-out true scores.

    Produced by ``retention_uncertainty``. The point estimate is the same
    uncentred plug-in \(\hat\eta_D=(\det\hat I_Z/\det\hat V)^{1/d}\) that
    ``information_report`` reports as ``geometric_mean_retention`` on a
    full-rank sample; the standard error is the influence-function estimate
    of ``RETENTION-PLUGIN-CLT-FROZEN-VECTOR`` (scalar case
    ``RETENTION-PLUGIN-CLT-FROZEN-SCALAR``), and the interval is the
    untruncated two-sided Wald interval
    \(\hat\eta_D\pm z_{1-\alpha/2}\,\widehat{\mathrm{SE}}\). It is never
    clipped, so an endpoint can fall below zero or above one; that is a
    property of the first-order interval, not a claim about the retention.

    The interval covers sampling variability only: the rule is frozen, the
    rows are independent and equally weighted draws from the reference law,
    and the scores are the model's true scores. A classifier or other
    estimated score adds a separate reporting bias that no error bar on the
    proxy sample measures.

    Attributes
    ----------
    estimate
        Plug-in geometric-mean retention, ``None`` when the rule declares too
        few cells or the full moment matrix is rank deficient, and exactly
        ``0.0`` by convention when the retained matrix is numerically rank deficient.
    standard_error
        Influence-function standard error, ``0.0`` when the influence values
        cancel to rounding noise, and ``None`` whenever no first-order theory
        applies.
    confidence_interval
        ``(lower, upper)`` Wald endpoints, or ``None`` when the interval is
        unavailable.
    confidence_level
        Nominal two-sided level the interval was built for.
    n_observations
        Number of evaluation rows the moments were formed from.
    status
        ``"ok"``: estimate, standard error and interval are all available.
        ``"insufficient_cells"``: the rule declares at most ``d`` cells, so
        the population retention is zero at the reference law
        (``FI-RANK-CEILING``) and the plug-in is the biased endpoint
        estimator; nothing is reported.
        ``"singular_full_information"``: the supplied scores lose a direction
        at the rank threshold, so the geometric-mean retention of all ``d``
        directions is undefined and nothing is reported rather than a
        silently projected surrogate. ``"singular_retained_information"``:
        the full matrix is regular but the between-cell matrix is numerically
        rank deficient on this sample, so the diagnostic reports zero by its
        numerical-rank convention and withholds the interval. The exact
        plug-in can still be positive below the threshold; the verdict does
        not identify the population rank
        (``RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE``).
        ``"degenerate_variance"``: the influence values vanish to rounding,
        so the interval is withheld; this is a numerical guard and does not
        assert that the population variance is zero.
    """

    estimate: float | None
    standard_error: float | None
    confidence_interval: tuple[float, float] | None
    confidence_level: float
    n_observations: int
    status: RetentionUncertaintyStatus

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible representation without non-finite sentinels."""
        return json_ready(asdict(self))

    def __str__(self) -> str:
        """Format the estimate, its standard error and the interval, when available."""
        estimate = "unavailable" if self.estimate is None else f"{self.estimate:.6f}"
        error = "unavailable" if self.standard_error is None else f"{self.standard_error:.6f}"
        if self.confidence_interval is None:
            interval = "unavailable"
        else:
            lower, upper = self.confidence_interval
            interval = f"[{lower:.6f}, {upper:.6f}]"
        return (
            "ScoreQuant retention uncertainty\n"
            f"  status: {self.status}\n"
            f"  estimate: {estimate}\n"
            f"  standard error: {error}\n"
            f"  {self.confidence_level:.0%} Wald interval: {interval}\n"
            f"  observations: {self.n_observations}"
        )


@dataclass(frozen=True, slots=True)
class RatioClosureReport:
    """Report how far model density ratios are from unit normalization.

    Exact component ratios relative to a reference measure integrate to one
    under that measure: ``sum_i w_i r_ik / sum_i w_i == 1`` for every
    component ``k`` when the weights carry the measure the ratio denominator
    defines. A large residual signals estimator bias, a misdeclared training
    prior, or a measure mismatch. The check is necessary but not sufficient:
    a ratio can close marginally while still being wrong pointwise, so a
    small residual never upgrades estimated provenance.

    Attributes
    ----------
    normalizers
        Weighted mean of each ratio column, shape ``[K]``.
    max_residual
        Largest absolute deviation of ``normalizers`` from one.
    """

    normalizers: np.ndarray
    max_residual: float

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible closure representation."""
        return json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class ProfiledInformationReport:
    """Report same-label profiled information for interest and nuisance blocks."""

    interest: tuple[int, ...]
    nuisance: tuple[int, ...]
    schur_unbinned: np.ndarray
    schur_binned: np.ndarray
    nuisance_unbinned: np.ndarray
    nuisance_binned: np.ndarray
    objective: float
    logdet_retention: float
    geometric_mean_retention: float
    interest_rank: int
    nuisance_rank: int
    schema: ScoreSchema | None = None

    @property
    def interest_names(self) -> tuple[str, ...] | None:
        """Return the names of the parameters of interest, when they were declared."""
        return self._names(self.interest)

    @property
    def nuisance_names(self) -> tuple[str, ...] | None:
        """Return the names of the profiled nuisance parameters, when declared."""
        return self._names(self.nuisance)

    def _names(self, columns: tuple[int, ...]) -> tuple[str, ...] | None:
        if self.schema is None:
            return None
        return tuple(self.schema.parameters[index] for index in columns)

    def describe(self) -> str:
        """Summarize the profiling split in one line, by name when one is available.

        ``interest: HSPCs`` says what was optimized; ``interest: (4,)`` requires
        the reader to remember the column order.
        """
        interest = self.interest_names or tuple(str(index) for index in self.interest)
        nuisance = self.nuisance_names or tuple(str(index) for index in self.nuisance)
        return f"interest: {', '.join(interest)}\nnuisance: {', '.join(nuisance)}"

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible profiled-information representation."""
        facts = json_ready(asdict(self))
        facts["interest_names"] = None if self.interest_names is None else list(self.interest_names)
        facts["nuisance_names"] = None if self.nuisance_names is None else list(self.nuisance_names)
        return facts


@dataclass(frozen=True, slots=True)
class GeometryReport:
    r"""Certify the self-consistent Voronoi geometry of a finite D partition.

    At a one-point-exchange-stable, positive-definite D partition every
    admissible move that violates the Mahalanobis-Voronoi rule of the terminal
    metric \(I^{-1}\) would raise the log determinant by at least
    \(\log(1+\alpha\beta q_\delta^2/4)>0\), where
    \(q_\delta=(\mu_a-\mu_b)^\top I^{-1}(\mu_a-\mu_b)\) separates the two cell
    means. Exact zero-tolerance stability under the merged-atom hypotheses
    therefore forces strict Voronoi geometry, which is
    what makes ``PartitionResult.compile_quantizer`` well posed. This report
    measures both sides of that statement on the terminal state instead of
    assuming them.

    A finite solver stops at a tolerance, so this certificate states one too.
    The theorem is exact, but the guaranteed gain of a violation shrinks with
    the cell separation, and on a large sample \(q_\delta\) shrinks with the
    sample size: once \(\log(1+\alpha\beta q_\delta^2/4)\) falls below the
    solver's ``gain_tolerance``, exchange stability *at that tolerance* no
    longer forbids a row from sitting a hair past a cell boundary. Verifying
    such a state at tolerance zero rejects a partition the solver never claimed
    to have refined further, so every field below is judged against
    ``gain_tolerance`` instead.

    At positive tolerance this report covers admissible individual moves only.
    It does not certify that every prediction disagreement is admissible; the
    solver checks that separately and refuses singleton disagreements. It also
    supplies no bound on simultaneous relabeling or population loss.

    All quadratic forms use the same metric and cell means the solver ended
    with, evaluated over the distinct positive-weight score atoms.

    Attributes
    ----------
    maximum_voronoi_violation
        Largest value over rows of the own-cell distance minus the smallest
        other-cell distance. A nonpositive value means every row already sits in
        its nearest cell under the terminal metric. It is ``-inf`` for a
        single-cell partition, which has no alternative destination. This is a
        Mahalanobis distance gap, not a criterion gain, so it is a diagnostic
        and never the verdict.
    guaranteed_violation_gain
        Largest Theorem-3 lower bound \(\log(1+\alpha\beta q_\delta^2/4)\) over
        admissible Voronoi-violating moves, and exactly ``0.0`` when no such
        move exists.
    maximum_violation_gain
        Largest *exact* log-determinant gain over the same admissible
        Voronoi-violating moves, and exactly ``0.0`` when none exists. Theorem 3
        bounds this from below, so ``guaranteed_violation_gain <=
        maximum_violation_gain`` always holds. This is the quantity the solver
        drives below its tolerance, so it is the one the verdict uses.
    maximum_separation_residual
        Largest value over unordered cell pairs of \(q_\delta-(1/W_a+1/W_b)\).
        The leverage lemma makes this nonpositive for every labeling, so a
        positive value indicates numerical trouble rather than a better
        partition. It is ``-inf`` for a single-cell partition.
    violating_moves, evaluated_moves
        Number of admissible Voronoi-violating moves and of admissible moves.
        A move is admissible when its source cell keeps positive weight and its
        destination differs from its source.
    voronoi_consistent
        Whether ``maximum_violation_gain`` is at most ``gain_tolerance``: no
        Voronoi violation is worth more than the solver's own stopping
        threshold. It is ``True`` whenever no row is misplaced at all.
    separation_certified
        Whether ``maximum_separation_residual`` respects the leverage lemma up
        to a small relative floating-point tolerance.
    gain_tolerance
        Log-determinant gain tolerance this certificate holds at, taken from the
        configuration that produced the labels. ``voronoi_consistent`` means
        self-consistent at this tolerance and claims nothing at tolerance zero.
    """

    maximum_voronoi_violation: float
    guaranteed_violation_gain: float
    maximum_violation_gain: float
    maximum_separation_residual: float
    violating_moves: int
    evaluated_moves: int
    voronoi_consistent: bool
    separation_certified: bool
    gain_tolerance: float

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible Voronoi-geometry representation."""
        return json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class StabilityReport:
    """Certify that no single-row relocation improves one supplied labeling.

    The report is produced by ``exchange_stability_report`` from exactly one
    complete exact scan, so it verifies labels of any origin: a guarded
    Mahalanobis-Lloyd run that stopped early, an external tool, or a hand edit.

    Stability is always a statement at a tolerance, never at tolerance zero, so
    the certificate carries the tolerance it was issued at.

    Attributes
    ----------
    stable
        Whether no admissible relocation improves the criterion by more than
        ``gain_tolerance``.
    best_gain
        Largest exact objective gain found in the scan. It is ``-inf`` when the
        labeling admits no relocation at all.
    best_move
        ``(row, destination)`` of that gain in original input row indexing, or
        ``None`` when the labeling is stable.
    objective
        Exact criterion value of the supplied labeling, in the convention
        ``PartitionResult.objective`` uses for the same criterion.
    n_bins
        Number of cells the labeling declares.
    criterion
        Criterion the scan certified against.
    gain_tolerance
        Strict minimum gain the scan counted as an improvement. ``stable``
        means ``best_gain <= gain_tolerance``; a labeling certified at one
        tolerance is not certified at a smaller one.
    """

    stable: bool
    best_gain: float
    best_move: tuple[int, int] | None
    objective: float
    n_bins: int
    criterion: DOptimality | ProfiledDOptimality
    gain_tolerance: float

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible stability representation."""
        return json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class ProfiledGeometryReport:
    """Diagnose the finite efficient-semimetric gap of a profiled partition."""

    metric: np.ndarray
    maximum_positive_violation: float
    maximum_theoretical_bound: float
    maximum_bound_residual: float
    violating_moves: int
    evaluated_moves: int
    bound_certified: bool

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible geometry-gap representation."""
        return json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class PartitionCertificate:
    r"""Report what a bounded global D search actually proved.

    ``certify_partition`` explores hard labelings with the singleton-completion
    upper bound: any completion of a partial assignment is coarser than the
    partial cells together with singleton cells for the unassigned atoms, so
    Loewner monotonicity of the log determinant makes
    \(\log\det(I_{\text{partial}}+R_t)\) a valid ceiling for the whole subtree.
    The search is exponential in the worst case and therefore explicitly
    bounded; the certificate always states which of the two outcomes occurred.

    Attributes
    ----------
    status
        ``"optimal"`` when the tree was exhausted, so no labeling beats
        ``objective`` by more than the configured gain tolerance.
        ``"budget_exhausted"`` when the node budget stopped the search first.
    objective
        Best log determinant found, in the Fisher-whitened convention of
        ``PartitionResult.objective`` under ``DOptimality``.
    labels
        Labels attaining ``objective``, defined for every input row.
        Zero-weight rows carry the label of their nearest cell mean in the
        terminal metric and never influenced the search.
    upper_bound
        Global ceiling at termination. It equals ``objective`` for a proved
        optimum and is otherwise the best bound still outstanding on an
        abandoned subtree.
    gap
        ``upper_bound`` minus ``objective``, nonnegative by construction and
        exactly zero for a proved optimum.
    nodes_explored
        Number of search nodes visited, including pruned children.
    incumbent_was_optimal
        Whether the search proved the starting incumbent optimal without
        improving it. It is ``False`` whenever the budget was exhausted,
        because an unfinished search proves nothing about the incumbent.
    """

    status: Literal["optimal", "budget_exhausted"]
    objective: float
    labels: np.ndarray
    upper_bound: float
    gap: float
    nodes_explored: int
    incumbent_was_optimal: bool

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible representation of the certificate."""
        return json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class EfficientScoreBound:
    r"""Certify a ceiling on profiled information from the full-data efficient score.

    Let \(\hat s=s_\psi-B^\ast s_\lambda\) be the efficient score built from the
    *full-data* information matrix, and let \(q\) be any hard rule with at most
    ``n_bins`` cells. Efficient-score domination states

    \[
        \mathrm{Schur}_\psi\!\left(I_q\right)\;\preceq\;
        \mathbb{E}\!\left[\hat s \mid q\right]\text{-between-cell information},
    \]

    so maximizing the right-hand side over all ``n_bins``-cell rules of
    \(\hat s\) upper-bounds the profiled objective of every ``n_bins``-cell rule
    of the *full* score space. For one parameter of interest the right-hand side
    is scalar, the maximizer has ordered interval cells, and the exact weighted
    interval dynamic program attains it. ``upper_bound`` is the logarithm of
    that maximum, in the same convention as ``PartitionResult.objective`` under
    ``ProfiledDOptimality``: an uncentered between-cell second moment of raw
    score columns, never a mean-centered variance.

    Attributes
    ----------
    upper_bound
        Log-scale certified ceiling on the profiled objective.
    labels
        Interval labels of the efficient score, defined for every input row.
        Zero-weight rows carry the label of their nearest cell mean and never
        influence the bound. These labels are also a strong initializer: pass
        them as ``initial_labels`` to ``optimize_partition`` under
        ``ProfiledDOptimality``.
    efficient_scores
        Full-information efficient scores with shape ``[N, 1]``.
    n_bins, interest
        Cell budget and interest columns the bound was certified for.

    Notes
    -----
    The bound is a property of one weighted score table. Comparing it to a
    partition of different scores or weights is meaningless, and ``gap_to``
    cannot detect that mismatch; it only checks the criterion convention and the
    cell budget. Refinement monotonicity makes the bound valid for any partition
    with at most ``n_bins`` cells.
    """

    upper_bound: float
    labels: np.ndarray
    efficient_scores: np.ndarray
    n_bins: int
    interest: tuple[int, ...]

    def gap_to(self, partition_result: PartitionResult) -> float:
        r"""Return the certified slack between the bound and an achieved objective.

        Parameters
        ----------
        partition_result
            Profiled-\(D_s\) result on the same weighted score table, with the
            same interest columns and at most ``n_bins`` cells.

        Returns
        -------
        float
            ``upper_bound`` minus the achieved profiled objective. The value is
            nonnegative up to floating-point error on valid inputs.
        """
        criterion = partition_result.criterion
        if not isinstance(criterion, ProfiledDOptimality):
            raise ContractError(
                "the efficient-score bound compares only against a profiled-D partition"
            )
        if criterion.interest_indices != self.interest:
            raise ContractError(
                f"partition interest {criterion.interest_indices} differs from the certified "
                f"interest {self.interest}"
            )
        if partition_result.n_bins > self.n_bins:
            raise ContractError(
                f"the bound certifies at most {self.n_bins} cells, but the partition "
                f"has {partition_result.n_bins}"
            )
        return self.upper_bound - partition_result.objective

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible representation of the certified bound."""
        return json_ready(asdict(self))
