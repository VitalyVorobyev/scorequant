"""Profiled D_s extension of the FlowCyt study: one fraction against the rest.

The main experiment treats all five target fractions symmetrically. A reported
measurement rarely does. This module fixes one population fraction as the
parameter of interest, treats the remaining four target fractions as nuisances,
and compares plain `scorequant.DOptimality` with
`scorequant.ProfiledDOptimality` on exactly the rows, weights, and splits the
main experiment uses.

The nuisance parameterization is not invented here. The study's score matrix
already carries one column per free fraction of the parameterization

    theta = (theta_T, theta_B, theta_mono, theta_mast, theta_HSPC),
    theta_other = 1 - sum(theta),

so column ``a`` *is* the derivative of the log density with respect to the
fraction of population ``a`` at the reference composition, with ``other``
absorbing the sum constraint. Profiling column ``a`` against the other four is
therefore the exact asymptotic operation the downstream measurement performs
when it reports one fraction with the rest of the marrow composition unknown.

Every number this module produces is application code built on ScoreQuant's
public surface; it adds nothing to the library.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import scorequant as sq

from .closure import (
    conditional_binned_fisher_information,
    conditional_fisher_information,
)
from .data import CLASS_NAMES, FlowCytData
from .experiment import _prepare_experiment, predict_score_bins
from .likelihood import estimate_bin_templates, fit_binned_mixture
from .scores import integration_weights

#: Score column of the headline parameter of interest. HSPCs are the CD34+
#: progenitor compartment, the number a bone-marrow report is usually written
#: around, and their reference fraction is far enough from zero that the local
#: covariance of the downstream fit stays interpretable -- unlike mast cells,
#: whose reference fraction the main study already flags as boundary dominated.
INTEREST_INDEX = 4

#: Operating bin budget, shared with the main experiment.
OPERATING_BINS = 8

#: Bin budgets swept against the certified ceiling. Six is the smallest budget
#: at which a fixed-total six-class mixture is identifiable at all, so the sweep
#: brackets the main study's identifiability threshold on both sides.
BUDGETS = (5, 6, 8, 10, 15, 30)

#: Seed shared by every solver in this study.
SEED = 2026

#: Relative tolerance for the rank of a fixed-total information matrix.
_RANK_RTOL = 1e-10


def _matrix_rank(values: np.ndarray) -> int:
    singular_values = np.linalg.svd(values, compute_uv=False)
    if len(singular_values) == 0 or singular_values[0] == 0:
        return 0
    return int(np.count_nonzero(singular_values > singular_values[0] * _RANK_RTOL))


def profiled_scalar(matrix: np.ndarray, index: int) -> float:
    """Return the scalar Schur complement of one interest column.

    Parameters
    ----------
    matrix
        Symmetric information matrix with shape ``[P, P]``.
    index
        Column of the parameter of interest. Every other column is nuisance.

    Returns
    -------
    float
        ``I_aa - I_an pinv(I_nn) I_na``, the information about parameter
        ``index`` once every other parameter floats.
    """
    values = np.asarray(matrix, dtype=np.float64)
    others = [position for position in range(values.shape[0]) if position != index]
    cross = values[index, others]
    nuisance = values[np.ix_(others, others)]
    return float(values[index, index] - cross @ np.linalg.pinv(nuisance, hermitian=True) @ cross)


@dataclass(frozen=True, slots=True)
class ProfiledInputs:
    """Score-space inputs the profiled study consumes.

    Attributes
    ----------
    theta0
        Reference six-class composition.
    partition_scores, partition_weights
        The partition subsample and its reference-measure weights, identical to
        the rows every learned rule in the main study is fitted on.
    validation_scores, validation_weights
        Diagnostic-only rows and weights.
    template_scores, template_classes, template_patients
        Independent labelled reference rows used to estimate ``P(bin | class)``.
    test_scores, test_patients
        The frozen held-out cohort.
    patient_ids, true_fractions
        Test patient identifiers and their expert compositions.
    rows
        Row counts of every role, recorded for provenance.
    preparation_seconds
        Wall-clock seconds the score model and score construction took, carried
        with the inputs so a resumed run still publishes an honest total.
    """

    theta0: np.ndarray
    partition_scores: np.ndarray
    partition_weights: np.ndarray
    validation_scores: np.ndarray
    validation_weights: np.ndarray
    template_scores: np.ndarray
    template_classes: np.ndarray
    template_patients: np.ndarray
    test_scores: np.ndarray
    test_patients: np.ndarray
    patient_ids: np.ndarray
    true_fractions: np.ndarray
    rows: dict[str, int]
    preparation_seconds: float


def profiled_inputs_from_data(
    data: FlowCytData,
    *,
    quick: bool,
    seed: int = SEED,
    score_max_per_patient_class: int | None = None,
    score_max_iter: int | None = None,
) -> ProfiledInputs:
    """Build the profiled inputs with the main study's own preparation path.

    Parameters
    ----------
    data
        The frozen fixture or the bounded all-patient sample.
    quick
        Use the short score-model settings.
    seed
        Seed of the score model and the deterministic role masks.
    score_max_per_patient_class, score_max_iter
        Optional overrides of the score-model budget. They exist so a
        fixture-scale regression test can exercise this path in seconds;
        published runs leave them unset.

    Returns
    -------
    ProfiledInputs
        The prepared score-space inputs, carrying the preparation wall clock.
    """
    started = time.perf_counter()
    context = _prepare_experiment(
        data,
        quick=quick,
        seed=seed,
        score_max_per_patient_class=score_max_per_patient_class,
        score_max_iter=score_max_iter,
    )
    seconds = time.perf_counter() - started
    inputs = ProfiledInputs(
        theta0=context.theta0,
        partition_scores=context.reference_scores[context.partition_mask],
        partition_weights=context.weights,
        validation_scores=context.reference_scores[context.validation_mask],
        validation_weights=integration_weights(
            context.reference.labels[context.validation_mask],
            context.reference.patients[context.validation_mask],
            context.theta0,
        ),
        template_scores=context.reference_scores[context.template_mask],
        template_classes=np.asarray(context.reference.labels[context.template_mask]),
        template_patients=np.asarray(context.reference.patients[context.template_mask]),
        test_scores=context.test_scores,
        test_patients=np.asarray(context.test.patients),
        patient_ids=np.asarray(context.patients),
        true_fractions=context.true_fractions,
        rows={
            "total": int(len(data.labels)),
            "reference": int(len(context.reference.labels)),
            "test": int(len(context.test.labels)),
            "partition": int(np.count_nonzero(context.partition_mask)),
            "validation": int(np.count_nonzero(context.validation_mask)),
            "templates": int(np.count_nonzero(context.template_mask)),
        },
        preparation_seconds=seconds,
    )
    return inputs


def save_profiled_inputs(inputs: ProfiledInputs, path: Path) -> None:
    """Cache prepared inputs so a long run can resume without refitting scores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rows_json=np.asarray(json.dumps(inputs.rows)),
        preparation_seconds=np.asarray(inputs.preparation_seconds),
        **{
            name: np.asarray(getattr(inputs, name))
            for name in (
                "theta0",
                "partition_scores",
                "partition_weights",
                "validation_scores",
                "validation_weights",
                "template_scores",
                "template_classes",
                "template_patients",
                "test_scores",
                "test_patients",
                "patient_ids",
                "true_fractions",
            )
        },
    )


def load_profiled_inputs(path: Path) -> ProfiledInputs:
    """Load inputs cached by `save_profiled_inputs`."""
    with np.load(path, allow_pickle=False) as payload:
        return ProfiledInputs(
            theta0=payload["theta0"],
            partition_scores=payload["partition_scores"],
            partition_weights=payload["partition_weights"],
            validation_scores=payload["validation_scores"],
            validation_weights=payload["validation_weights"],
            template_scores=payload["template_scores"],
            template_classes=payload["template_classes"],
            template_patients=payload["template_patients"],
            test_scores=payload["test_scores"],
            test_patients=payload["test_patients"],
            patient_ids=payload["patient_ids"],
            true_fractions=payload["true_fractions"],
            rows=json.loads(str(payload["rows_json"])),
            preparation_seconds=float(payload["preparation_seconds"]),
        )


@dataclass(frozen=True, slots=True)
class LabelingScore:
    """Every retention number one hard labeling earns in this study.

    Attributes
    ----------
    full_retention
        Geometric-mean retention of the whole five-parameter Fisher matrix.
    profiled_retention
        Retention of the profiled information of the interest fraction alone,
        in the library's uncentered intensity convention.
    profiled_information
        The binned profiled information itself, in the same convention.
    fixed_total_profiled_retention
        The same ratio in the fixed-total convention the FlowCyt patient
        likelihood actually uses. It vanishes exactly when the labeling cannot
        identify all five free fractions, which is why `fixed_total_rank` is
        published beside it.
    fixed_total_rank
        Rank of the retained fixed-total information. Five is required for the
        patient likelihood to identify the six-class composition at all.
    occupied_bins
        Number of bins carrying positive measure.
    """

    full_retention: float
    profiled_retention: float
    profiled_information: float
    fixed_total_profiled_retention: float
    fixed_total_rank: int
    occupied_bins: int


def score_labeling(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    interest_index: int,
    n_bins: int,
) -> LabelingScore:
    """Score one hard labeling on the full, profiled, and fixed-total criteria.

    Parameters
    ----------
    scores
        Score matrix with shape ``[N, 5]`` in the declared fraction order.
    labels
        Integer bin label per row.
    weights
        Nonnegative measure weights with shape ``[N]``.
    interest_index
        Score column of the parameter of interest.
    n_bins
        Number of declared bins, including empty ones.

    Returns
    -------
    LabelingScore
        Numbers that do not depend on which criterion produced `labels`, which
        is what makes the two criteria comparable at all.
    """
    full = sq.information_report(scores, labels, weights, n_bins=n_bins)
    profiled = sq.information.profiled_information_report(
        scores, labels, interest=(interest_index,), weights=weights, n_bins=n_bins
    )
    conditional_binned = conditional_binned_fisher_information(
        scores, labels, weights, n_bins=n_bins
    )
    retained_rank = _matrix_rank(conditional_binned)
    # A rank-deficient retained matrix makes this Schur complement exactly zero
    # by algebra: with fewer than five independent bin frequencies the interest
    # fraction is not estimable at all once the other four float. Reporting the
    # floating-point residual instead would dress a rank statement up as a small
    # number, so the deficient case is published as the zero it is, beside the
    # rank that explains it.
    if retained_rank < scores.shape[1]:
        fixed_total = 0.0
    else:
        conditional_full = conditional_fisher_information(scores, weights)
        fixed_total = float(
            profiled_scalar(conditional_binned, interest_index)
            / profiled_scalar(conditional_full, interest_index)
        )
    return LabelingScore(
        full_retention=float(full.geometric_mean_retention),
        profiled_retention=float(profiled.geometric_mean_retention),
        profiled_information=float(np.asarray(profiled.schur_binned)[0, 0]),
        fixed_total_profiled_retention=max(fixed_total, 0.0),
        fixed_total_rank=retained_rank,
        occupied_bins=int(np.count_nonzero(np.asarray(full.bin_weights) > 0)),
    )


@dataclass(frozen=True, slots=True)
class PartitionRow:
    """One finite partition of the FlowCyt partition subsample, scored.

    Attributes
    ----------
    key, label, criterion
        Stable identifier, published name, and the criterion that produced it.
    full_retention, profiled_retention, fixed_total_profiled_retention
        The three retention numbers from `score_labeling`.
    objective
        The criterion's own objective at the terminal labeling.
    scans, accepted_moves, exchange_stable
        Exchange counters and the stability flag of the run.
    seconds
        Wall-clock seconds of the single fit.
    """

    key: str
    label: str
    criterion: str
    full_retention: float
    profiled_retention: float
    fixed_total_profiled_retention: float
    objective: float
    scans: int
    accepted_moves: int
    exchange_stable: bool
    seconds: float


def _partition_row(
    key: str,
    label: str,
    criterion: str,
    result: sq.results.PartitionResult,
    score: LabelingScore,
    seconds: float,
) -> PartitionRow:
    return PartitionRow(
        key=key,
        label=label,
        criterion=criterion,
        full_retention=score.full_retention,
        profiled_retention=score.profiled_retention,
        fixed_total_profiled_retention=score.fixed_total_profiled_retention,
        objective=float(result.objective),
        scans=int(result.scans),
        accepted_moves=int(result.accepted_moves),
        exchange_stable=bool(result.exchange_stable),
        seconds=seconds,
    )


@dataclass(frozen=True, slots=True)
class DownstreamRow:
    """What one reusable rule reports for the interest fraction.

    Attributes
    ----------
    key, label
        Identity of the rule.
    interest_rmse
        Held-out root-mean-square error of the interest fraction over the test
        patients.
    macro_rmse
        Held-out macro error over all five target fractions, for comparison
        with the main study's table.
    mean_half_width
        Mean local 68% half-width the fit reports for the interest fraction
        with the other fractions floating. This is the profiled interval, and it
        is the number the measurement would actually quote.
    converged_patients, total_patients, maximum_iterations
        Convergence record of the patient likelihoods.
    """

    key: str
    label: str
    interest_rmse: float
    macro_rmse: float
    mean_half_width: float
    converged_patients: int
    total_patients: int
    maximum_iterations: int


@dataclass(frozen=True, slots=True)
class RuleRow:
    """One reusable score-space rule, scored on training and held-out rows.

    Attributes
    ----------
    key, label, criterion, solver
        Identity of the fit.
    train_full_retention, train_profiled_retention
        Retention of the rule's own labels on the partition subsample.
    test_full_retention, test_profiled_retention
        Retention of the same rule on the untouched test cohort, measured under
        the empirical test measure rather than the reference integration
        weights, exactly as the main study's held-out column is.
    test_occupied_bins
        Bins the rule actually fills on the test cohort. High retention does not
        prevent an empty transported bin.
    hardening_gap
        Soft-to-hard objective gap the fit reports, when it has one.
    seconds
        Wall-clock seconds of the fit.
    downstream
        The interest-fraction measurement this rule implies.
    """

    key: str
    label: str
    criterion: str
    solver: str
    train_full_retention: float
    train_profiled_retention: float
    test_full_retention: float
    test_profiled_retention: float
    test_occupied_bins: int
    hardening_gap: float | None
    seconds: float
    downstream: DownstreamRow


def downstream_measurement(
    inputs: ProfiledInputs,
    *,
    key: str,
    label: str,
    template_labels: np.ndarray,
    test_labels: np.ndarray,
    n_bins: int,
    interest_index: int,
) -> DownstreamRow:
    """Fit every test patient from hard counts and report the interest fraction.

    Parameters
    ----------
    inputs
        The prepared score-space inputs.
    key, label
        Identity of the rule whose labels are being measured.
    template_labels
        Bin labels of the independent labelled template rows.
    test_labels
        Bin labels of the held-out cohort.
    n_bins
        Declared bin budget.
    interest_index
        Score column of the parameter of interest.

    Returns
    -------
    DownstreamRow
        Held-out error and the mean profiled half-width for the interest
        fraction.
    """
    templates = estimate_bin_templates(
        inputs.template_classes,
        template_labels,
        inputs.template_patients,
        n_bins=n_bins,
    )
    estimates = []
    for patient in inputs.patient_ids:
        mask = inputs.test_patients == patient
        estimates.append(
            fit_binned_mixture(np.bincount(test_labels[mask], minlength=n_bins), templates)
        )
    predicted = np.asarray([estimate.fractions for estimate in estimates])
    half_widths = np.asarray([estimate.standard_errors[interest_index] for estimate in estimates])
    per_class = np.sqrt(np.mean((predicted - inputs.true_fractions) ** 2, axis=0))
    return DownstreamRow(
        key=key,
        label=label,
        interest_rmse=float(per_class[interest_index]),
        macro_rmse=float(np.mean(per_class[:5])),
        mean_half_width=float(np.mean(half_widths)),
        converged_patients=sum(estimate.converged for estimate in estimates),
        total_patients=len(estimates),
        maximum_iterations=max(estimate.iterations for estimate in estimates),
    )


@dataclass(frozen=True, slots=True)
class BudgetRow:
    """One bin budget, both criteria, and the certified ceiling.

    Attributes
    ----------
    n_bins
        Bin budget.
    d_profiled_retention, ds_seeded_retention, ds_initialized_retention
        Profiled retention of the plain-D partition, of the generically seeded
        profiled partition, and of the profiled partition started from the
        ceiling's own interval labels.
    d_full_retention, ds_initialized_full_retention
        What each of those two partitions retains about all five fractions.
    d_fixed_total_retention, ds_initialized_fixed_total_retention
        The same profiled ratio in the fixed-total convention the patient
        likelihood uses. It is exactly zero below the identifiability
        threshold, whatever the intensity-convention column says.
    ceiling_retention
        The certified ceiling no rule of any kind can exceed at this budget.
    seeded_gap, initialized_gap
        Certified gaps in nats of the profiled log determinant.
    seeded_moves, initialized_moves, seeded_scans, initialized_scans
        Exchange counters of the two profiled runs.
    fixed_total_rank_d, fixed_total_rank_ds
        Rank of the retained fixed-total information under each partition.
    bound_seconds
        Wall-clock seconds of the exact scalar dynamic program.
    """

    n_bins: int
    d_profiled_retention: float
    ds_seeded_retention: float
    ds_initialized_retention: float
    d_full_retention: float
    ds_initialized_full_retention: float
    d_fixed_total_retention: float
    ds_initialized_fixed_total_retention: float
    ceiling_retention: float
    seeded_gap: float
    initialized_gap: float
    seeded_moves: int
    initialized_moves: int
    seeded_scans: int
    initialized_scans: int
    fixed_total_rank_d: int
    fixed_total_rank_ds: int
    bound_seconds: float


@dataclass(frozen=True, slots=True)
class InterestRow:
    """The operating-point comparison with a different fraction of interest.

    Attributes
    ----------
    index, population, reference_fraction
        Which fraction is of interest and how much reference mass it carries.
    d_profiled_retention, ds_seeded_retention, ds_initialized_retention
        Profiled retention of the shared plain-D partition and of both profiled
        runs. Both profiled runs are published so the row states an outcome
        rather than a choice.
    ceiling_retention
        The certified ceiling for this interest column.
    ds_full_retention
        What the ceiling-initialized profiled partition retains about all five
        fractions.
    gain
        ``ds_initialized_retention - d_profiled_retention``.
    """

    index: int
    population: str
    reference_fraction: float
    d_profiled_retention: float
    ds_seeded_retention: float
    ds_initialized_retention: float
    ds_full_retention: float
    ceiling_retention: float
    gain: float


@dataclass(frozen=True, slots=True)
class ProfiledStudy:
    """Everything the profiled FlowCyt page publishes from one run."""

    metrics: dict[str, object]


def _scalar_dp_config(n_rows: int, *, seed: int) -> sq.ScalarDPConfig:
    """Size the exact scalar solver for this study's partition subsample."""
    return sq.ScalarDPConfig(max_rows=max(n_rows, 1))


def _optimize(
    scores: np.ndarray,
    weights: np.ndarray,
    *,
    n_bins: int,
    criterion: sq.DOptimality | sq.ProfiledDOptimality,
    config: sq.DExchangeConfig,
    initial_labels: np.ndarray | None = None,
) -> tuple[sq.results.PartitionResult, float]:
    started = time.perf_counter()
    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=n_bins,
        criterion=criterion,
        config=config,
        initial_labels=initial_labels,
    )
    return result, time.perf_counter() - started


def budget_sweep(
    inputs: ProfiledInputs,
    *,
    budgets: tuple[int, ...],
    interest_index: int,
    seed: int,
    initializer_restarts: int,
) -> list[BudgetRow]:
    """Sweep the bin budget under both criteria against the certified ceiling.

    Parameters
    ----------
    inputs
        The prepared score-space inputs.
    budgets
        Bin budgets to evaluate.
    interest_index
        Score column of the parameter of interest.
    seed, initializer_restarts
        Exchange settings shared by every fit.

    Returns
    -------
    list of BudgetRow
        One row per budget.
    """
    scores, weights = inputs.partition_scores, inputs.partition_weights
    criterion = sq.ProfiledDOptimality((interest_index,))
    config = sq.DExchangeConfig(seed=seed, initializer_restarts=initializer_restarts)
    dp_config = _scalar_dp_config(len(scores), seed=seed)
    unbinned = profiled_scalar(
        np.asarray(sq.information.fisher_information(scores, weights)), interest_index
    )
    rows: list[BudgetRow] = []
    for n_bins in budgets:
        started = time.perf_counter()
        bound = sq.efficient_score_bound(
            scores,
            interest=(interest_index,),
            weights=weights,
            n_bins=n_bins,
            config=dp_config,
        )
        bound_seconds = time.perf_counter() - started
        plain, _ = _optimize(
            scores, weights, n_bins=n_bins, criterion=sq.DOptimality(), config=config
        )
        seeded, _ = _optimize(scores, weights, n_bins=n_bins, criterion=criterion, config=config)
        initialized, _ = _optimize(
            scores,
            weights,
            n_bins=n_bins,
            criterion=criterion,
            config=config,
            initial_labels=np.asarray(bound.labels),
        )
        plain_score = score_labeling(
            scores,
            np.asarray(plain.labels),
            weights,
            interest_index=interest_index,
            n_bins=n_bins,
        )
        seeded_score = score_labeling(
            scores,
            np.asarray(seeded.labels),
            weights,
            interest_index=interest_index,
            n_bins=n_bins,
        )
        initialized_score = score_labeling(
            scores,
            np.asarray(initialized.labels),
            weights,
            interest_index=interest_index,
            n_bins=n_bins,
        )
        rows.append(
            BudgetRow(
                n_bins=n_bins,
                d_profiled_retention=plain_score.profiled_retention,
                ds_seeded_retention=seeded_score.profiled_retention,
                ds_initialized_retention=initialized_score.profiled_retention,
                d_full_retention=plain_score.full_retention,
                ds_initialized_full_retention=initialized_score.full_retention,
                d_fixed_total_retention=plain_score.fixed_total_profiled_retention,
                ds_initialized_fixed_total_retention=(
                    initialized_score.fixed_total_profiled_retention
                ),
                ceiling_retention=float(np.exp(bound.upper_bound - np.log(unbinned))),
                seeded_gap=float(bound.gap_to(seeded)),
                initialized_gap=float(bound.gap_to(initialized)),
                seeded_moves=int(seeded.accepted_moves),
                initialized_moves=int(initialized.accepted_moves),
                seeded_scans=int(seeded.scans),
                initialized_scans=int(initialized.scans),
                fixed_total_rank_d=plain_score.fixed_total_rank,
                fixed_total_rank_ds=initialized_score.fixed_total_rank,
                bound_seconds=bound_seconds,
            )
        )
    return rows


def interest_sweep(
    inputs: ProfiledInputs,
    *,
    n_bins: int,
    plain_labels: np.ndarray,
    seed: int,
    initializer_restarts: int,
) -> list[InterestRow]:
    """Repeat the operating-point comparison for every declared fraction.

    Parameters
    ----------
    inputs
        The prepared score-space inputs.
    n_bins
        Operating bin budget.
    plain_labels
        Labels of the plain-D partition, which does not depend on the interest
        column and is therefore shared by every row.
    seed, initializer_restarts
        Exchange settings shared by every fit.

    Returns
    -------
    list of InterestRow
        One row per free fraction.
    """
    scores, weights = inputs.partition_scores, inputs.partition_weights
    config = sq.DExchangeConfig(seed=seed, initializer_restarts=initializer_restarts)
    dp_config = _scalar_dp_config(len(scores), seed=seed)
    information = np.asarray(sq.information.fisher_information(scores, weights))
    rows: list[InterestRow] = []
    for index in range(scores.shape[1]):
        criterion = sq.ProfiledDOptimality((index,))
        bound = sq.efficient_score_bound(
            scores, interest=(index,), weights=weights, n_bins=n_bins, config=dp_config
        )
        seeded, _ = _optimize(scores, weights, n_bins=n_bins, criterion=criterion, config=config)
        initialized, _ = _optimize(
            scores,
            weights,
            n_bins=n_bins,
            criterion=criterion,
            config=config,
            initial_labels=np.asarray(bound.labels),
        )
        plain_retention = score_labeling(
            scores, plain_labels, weights, interest_index=index, n_bins=n_bins
        ).profiled_retention
        seeded_retention = score_labeling(
            scores, np.asarray(seeded.labels), weights, interest_index=index, n_bins=n_bins
        ).profiled_retention
        initialized_score = score_labeling(
            scores,
            np.asarray(initialized.labels),
            weights,
            interest_index=index,
            n_bins=n_bins,
        )
        rows.append(
            InterestRow(
                index=index,
                population=CLASS_NAMES[index],
                reference_fraction=float(inputs.theta0[index]),
                d_profiled_retention=plain_retention,
                ds_seeded_retention=seeded_retention,
                ds_initialized_retention=initialized_score.profiled_retention,
                ds_full_retention=initialized_score.full_retention,
                ceiling_retention=float(
                    np.exp(bound.upper_bound - np.log(profiled_scalar(information, index)))
                ),
                gain=initialized_score.profiled_retention - plain_retention,
            )
        )
    return rows


def reusable_rules(
    inputs: ProfiledInputs,
    *,
    n_bins: int,
    interest_index: int,
    seed: int,
    initializer_restarts: int,
    soft_steps: int,
) -> list[RuleRow]:
    """Fit one reusable rule per criterion and carry both through the fit.

    An exchange-stable plain-D partition compiles into a Mahalanobis rule that
    reproduces its own labels. A profiled partition has no such canonical
    extension, so a reusable profiled rule has to be fitted as one, which
    `scorequant.SoftVoronoiConfig` does.

    Parameters
    ----------
    inputs
        The prepared score-space inputs.
    n_bins
        Bin budget shared by both fits.
    interest_index
        Score column of the parameter of interest.
    seed, initializer_restarts, soft_steps
        Solver settings.

    Returns
    -------
    list of RuleRow
        One row per criterion, each carrying a held-out column and the
        downstream interest-fraction measurement.
    """
    source = sq.ScoreSample(inputs.partition_scores, inputs.partition_weights)
    validation = sq.ScoreSample(inputs.validation_scores, inputs.validation_weights)
    test_weights = np.ones(len(inputs.test_scores))
    specifications: list[
        tuple[str, str, str, str, sq.DOptimality | sq.ProfiledDOptimality, object]
    ] = [
        (
            "d_rule",
            "Plain D, compiled exchange",
            "DOptimality",
            "DExchangeConfig",
            sq.DOptimality(),
            sq.DExchangeConfig(seed=seed, initializer_restarts=initializer_restarts),
        ),
        (
            "ds_rule",
            "Profiled D_s, soft Voronoi",
            "ProfiledDOptimality",
            "SoftVoronoiConfig",
            sq.ProfiledDOptimality((interest_index,)),
            sq.SoftVoronoiConfig(
                seed=seed,
                initializer_restarts=initializer_restarts,
                max_iter=soft_steps,
                record_every=max(soft_steps // 8, 1),
            ),
        ),
    ]
    rows: list[RuleRow] = []
    for key, label, criterion_name, solver, criterion, config in specifications:
        started = time.perf_counter()
        rule = sq.fit_quantizer(
            source,
            validation=validation,
            n_bins=n_bins,
            criterion=criterion,
            config=config,  # type: ignore[arg-type]
        )
        seconds = time.perf_counter() - started
        test_labels = predict_score_bins(rule, inputs.test_scores)
        on_train = score_labeling(
            inputs.partition_scores,
            np.asarray(rule.predict_scores(inputs.partition_scores)),
            inputs.partition_weights,
            interest_index=interest_index,
            n_bins=n_bins,
        )
        on_test = score_labeling(
            inputs.test_scores,
            test_labels,
            test_weights,
            interest_index=interest_index,
            n_bins=n_bins,
        )
        rows.append(
            RuleRow(
                key=key,
                label=label,
                criterion=criterion_name,
                solver=solver,
                train_full_retention=on_train.full_retention,
                train_profiled_retention=on_train.profiled_retention,
                test_full_retention=on_test.full_retention,
                test_profiled_retention=on_test.profiled_retention,
                test_occupied_bins=on_test.occupied_bins,
                hardening_gap=(None if rule.hardening_gap is None else float(rule.hardening_gap)),
                seconds=seconds,
                downstream=downstream_measurement(
                    inputs,
                    key=key,
                    label=label,
                    template_labels=predict_score_bins(rule, inputs.template_scores),
                    test_labels=test_labels,
                    n_bins=n_bins,
                    interest_index=interest_index,
                ),
            )
        )
    return rows


def run_profiled_study(
    inputs: ProfiledInputs,
    *,
    quick: bool,
    interest_index: int = INTEREST_INDEX,
    n_bins: int = OPERATING_BINS,
    budgets: tuple[int, ...] = BUDGETS,
    seed: int = SEED,
    sweep_interest: bool = True,
    provenance: dict[str, object] | None = None,
) -> ProfiledStudy:
    """Run the profiled-D_s extension of the FlowCyt study.

    Parameters
    ----------
    inputs
        Prepared score-space inputs from `profiled_inputs_from_data`.
    quick
        Use the short optimizer settings.
    interest_index
        Score column of the headline parameter of interest.
    n_bins
        Operating bin budget of the headline comparison.
    budgets
        Bin budgets swept against the certified ceiling.
    seed
        Seed shared by every solver.
    sweep_interest
        Repeat the operating-point comparison with every fraction of interest.
    provenance
        Free-form provenance recorded verbatim in the metrics.

    Returns
    -------
    ProfiledStudy
        The metrics written to the committed JSON evidence.
    """
    if not 0 <= interest_index < len(CLASS_NAMES) - 1:
        raise ValueError("interest_index must select one of the five free fractions")
    if n_bins not in budgets:
        raise ValueError("n_bins must appear in the swept budgets")
    initializer_restarts = 3 if quick else 8
    soft_steps = 50 if quick else 160
    started = time.perf_counter()

    scores, weights = inputs.partition_scores, inputs.partition_weights
    config = sq.DExchangeConfig(seed=seed, initializer_restarts=initializer_restarts)
    criterion = sq.ProfiledDOptimality((interest_index,))

    bound_started = time.perf_counter()
    bound = sq.efficient_score_bound(
        scores,
        interest=(interest_index,),
        weights=weights,
        n_bins=n_bins,
        config=_scalar_dp_config(len(scores), seed=seed),
    )
    bound_seconds = time.perf_counter() - bound_started

    plain, plain_seconds = _optimize(
        scores, weights, n_bins=n_bins, criterion=sq.DOptimality(), config=config
    )
    seeded, seeded_seconds = _optimize(
        scores, weights, n_bins=n_bins, criterion=criterion, config=config
    )
    initialized, initialized_seconds = _optimize(
        scores,
        weights,
        n_bins=n_bins,
        criterion=criterion,
        config=config,
        initial_labels=np.asarray(bound.labels),
    )

    def scored(result: sq.results.PartitionResult) -> LabelingScore:
        return score_labeling(
            scores,
            np.asarray(result.labels),
            weights,
            interest_index=interest_index,
            n_bins=n_bins,
        )

    partitions = [
        _partition_row(
            "d_partition", "Plain D", "DOptimality", plain, scored(plain), plain_seconds
        ),
        _partition_row(
            "ds_partition_seeded",
            "Profiled D_s, generic seeding",
            "ProfiledDOptimality",
            seeded,
            scored(seeded),
            seeded_seconds,
        ),
        _partition_row(
            "ds_partition_initialized",
            "Profiled D_s, ceiling-initialized",
            "ProfiledDOptimality",
            initialized,
            scored(initialized),
            initialized_seconds,
        ),
    ]
    unbinned = profiled_scalar(
        np.asarray(sq.information.fisher_information(scores, weights)), interest_index
    )
    rules = reusable_rules(
        inputs,
        n_bins=n_bins,
        interest_index=interest_index,
        seed=seed,
        initializer_restarts=initializer_restarts,
        soft_steps=soft_steps,
    )
    budget_rows = budget_sweep(
        inputs,
        budgets=budgets,
        interest_index=interest_index,
        seed=seed,
        initializer_restarts=initializer_restarts,
    )
    interest_rows = (
        interest_sweep(
            inputs,
            n_bins=n_bins,
            plain_labels=np.asarray(plain.labels),
            seed=seed,
            initializer_restarts=initializer_restarts,
        )
        if sweep_interest
        else []
    )

    metrics: dict[str, object] = {
        "study": "flowcyt_profiled_ds",
        "interest_index": interest_index,
        "interest_population": CLASS_NAMES[interest_index],
        "nuisance_populations": [
            name for index, name in enumerate(CLASS_NAMES[:-1]) if index != interest_index
        ],
        "reference_component": CLASS_NAMES[-1],
        "n_bins": n_bins,
        "budgets": list(budgets),
        "reference_composition": [float(value) for value in inputs.theta0],
        "unbinned_profiled_information": unbinned,
        "partitions": [asdict(row) for row in partitions],
        "rules": [asdict(row) for row in rules],
        "budget_sweep": [asdict(row) for row in budget_rows],
        "interest_sweep": [asdict(row) for row in interest_rows],
        "bound": {
            "upper_bound": float(bound.upper_bound),
            "ceiling_retention": float(np.exp(bound.upper_bound - np.log(unbinned))),
            "seconds": bound_seconds,
            "seeded_gap": float(bound.gap_to(seeded)),
            "initialized_gap": float(bound.gap_to(initialized)),
        },
        "run": {
            "quick": quick,
            "seed": seed,
            "initializer_restarts": initializer_restarts,
            "soft_steps": soft_steps,
            "rows": dict(inputs.rows),
            "provenance": dict(provenance or {}),
            "seconds": {
                "score_model_and_scores": inputs.preparation_seconds,
                "efficient_score_bound": bound_seconds,
                "study": time.perf_counter() - started,
            },
        },
    }
    return ProfiledStudy(metrics=metrics)


def write_profiled_metrics(metrics: dict[str, object], path: Path) -> None:
    """Write one scale's metrics into the committed multi-scale evidence file.

    Parameters
    ----------
    metrics
        Metrics returned by `run_profiled_study`.
    path
        JSON evidence path. Scales already present are preserved, so the
        fixture-scale and sample-scale runs regenerate independently.
    """
    run = metrics["run"]
    if not isinstance(run, dict):
        raise TypeError("metrics['run'] must be a mapping")
    scale = "fixture_scale" if run["quick"] else "sample_scale"
    payload: dict[str, object] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded
    payload[scale] = metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
