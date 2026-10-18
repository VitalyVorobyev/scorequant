"""End-to-end FlowCyt population-quantification experiment."""

from __future__ import annotations

import resource
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import cast

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import scorequant as sq

from .closure import ClosureInputs, run_scientific_closure
from .data import CLASS_NAMES, REFERENCE_PATIENTS, TEST_PATIENTS, FlowCytData
from .likelihood import estimate_bin_templates, fit_binned_mixture, fit_unbinned_mixture
from .scores import (
    ScoreFit,
    fit_score_model,
    integration_weights,
    reference_composition,
)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Machine-readable metrics and arrays needed for documentation figures."""

    metrics: dict[str, object]
    true_fractions: np.ndarray
    predicted_fractions: dict[str, np.ndarray]
    bin_counts: tuple[int, ...]
    operating_n_bins: int
    operating_bin_composition: np.ndarray
    patient_ids: np.ndarray


@dataclass(frozen=True, slots=True)
class _ExperimentContext:
    data: FlowCytData
    reference: FlowCytData
    test: FlowCytData
    score_fit: ScoreFit
    theta0: np.ndarray
    reference_scores: np.ndarray
    test_scores: np.ndarray
    patients: np.ndarray
    true_fractions: np.ndarray
    unbinned_fractions: np.ndarray
    unbinned_convergence: dict[str, int]
    partition_mask: np.ndarray
    validation_mask: np.ndarray
    template_mask: np.ndarray
    weights: np.ndarray
    transformed_markers: np.ndarray
    transformed_test_markers: np.ndarray


@dataclass(frozen=True, slots=True)
class _SweepResult:
    metrics: dict[str, object]
    predicted: dict[str, np.ndarray]
    operating_bin_composition: np.ndarray
    timings: dict[str, float]


def _role_masks(data: FlowCytData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = data.source_rows + 17 * data.patients + 101 * data.labels
    roles = np.mod(keys, 4)
    return roles <= 1, roles == 2, roles == 3


def _cap_group_mask(
    data: FlowCytData,
    base_mask: np.ndarray,
    *,
    max_per_patient_class: int,
    seed: int,
) -> np.ndarray:
    """Cap computation rows without changing patient/class integration mass."""
    rng = np.random.default_rng(seed)
    selected = np.zeros(len(data.labels), dtype=bool)
    for patient in np.unique(data.patients):
        for label in range(len(CLASS_NAMES)):
            indices = np.flatnonzero(
                base_mask & (data.patients == patient) & (data.labels == label)
            )
            if len(indices) > max_per_patient_class:
                indices = np.sort(rng.choice(indices, size=max_per_patient_class, replace=False))
            selected[indices] = True
    return selected


def _calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
) -> dict[str, object]:
    """Summarize out-of-fold calibration with equal patient/class influence."""
    uniform = np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
    weights = integration_weights(labels, patients, uniform)
    confidence = np.max(probabilities, axis=1)
    predicted = np.argmax(probabilities, axis=1)
    correct = predicted == labels
    chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    one_hot = np.eye(len(CLASS_NAMES))[labels]
    edges = np.linspace(0.0, 1.0, 11)
    bins: list[dict[str, object]] = []
    calibration_error = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (confidence >= lower) & (
            (confidence <= upper) if index == len(edges) - 2 else (confidence < upper)
        )
        mass = float(np.sum(weights[mask]))
        if mass == 0:
            bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "weight": 0.0,
                    "mean_confidence": None,
                    "accuracy": None,
                }
            )
            continue
        mean_confidence = float(np.average(confidence[mask], weights=weights[mask]))
        accuracy = float(np.average(correct[mask], weights=weights[mask]))
        calibration_error += mass * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "weight": mass,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "balanced_log_loss": float(np.sum(weights * -np.log(chosen))),
        "balanced_brier_score": float(np.sum(weights[:, None] * (probabilities - one_hot) ** 2)),
        "balanced_accuracy": float(np.sum(weights * correct)),
        "expected_calibration_error": calibration_error,
        "reliability_bins": bins,
    }


def _peak_rss_megabytes() -> float:
    """Return process peak resident memory using platform-native units."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0**2 if sys.platform == "darwin" else 1024.0)


def _nearest(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def predict_score_bins(
    result: sq.results.QuantizerResult,
    scores: np.ndarray,
    *,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Predict a large score matrix without materializing all distances at once."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    labels = [
        np.asarray(result.predict_scores(scores[start : start + chunk_size]))
        for start in range(0, len(scores), chunk_size)
    ]
    return np.concatenate(labels) if labels else np.empty(0, dtype=np.int64)


def _factor_pair(n_bins: int) -> tuple[int, int]:
    for first in range(int(np.sqrt(n_bins)), 0, -1):
        if n_bins % first == 0:
            return first, n_bins // first
    raise AssertionError("every positive integer has a factor pair")


def _grid_edges(values: np.ndarray, n_parts: int) -> np.ndarray:
    if n_parts == 1:
        return np.empty(0)
    return np.quantile(values, np.linspace(0, 1, n_parts + 1)[1:-1])


def _grid_labels(
    points: np.ndarray, first_edges: np.ndarray, second_edges: np.ndarray
) -> np.ndarray:
    first = np.digitize(points[:, 0], first_edges)
    second = np.digitize(points[:, 1], second_edges)
    return first * (len(second_edges) + 1) + second


def _true_patient_fractions(test: FlowCytData) -> tuple[np.ndarray, np.ndarray]:
    patients = np.unique(test.patients)
    fractions = []
    for patient in patients:
        counts = np.bincount(test.labels[test.patients == patient], minlength=len(CLASS_NAMES))
        fractions.append(counts / np.sum(counts))
    return patients, np.asarray(fractions)


def _fit_patient_fractions(
    test: FlowCytData,
    test_labels: np.ndarray,
    templates: np.ndarray,
    patients: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    estimates = []
    for patient in patients:
        mask = test.patients == patient
        counts = np.bincount(test_labels[mask], minlength=templates.shape[0])
        estimates.append(fit_binned_mixture(counts, templates))
    return np.asarray([estimate.fractions for estimate in estimates]), {
        "converged_patients": sum(estimate.converged for estimate in estimates),
        "total_patients": len(estimates),
        "maximum_iterations": max(estimate.iterations for estimate in estimates),
    }


def _fit_unbinned_patient_fractions(
    test: FlowCytData,
    ratios: np.ndarray,
    patients: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    estimates = [fit_unbinned_mixture(ratios[test.patients == patient]) for patient in patients]
    return np.asarray([estimate.fractions for estimate in estimates]), {
        "converged_patients": sum(estimate.converged for estimate in estimates),
        "total_patients": len(estimates),
        "maximum_iterations": max(estimate.iterations for estimate in estimates),
    }


def _summary_metrics(
    true_fractions: np.ndarray,
    predicted: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    n_bins: int,
) -> dict[str, object]:
    errors = predicted - true_fractions
    per_class = np.sqrt(np.mean(errors**2, axis=0))
    report = sq.information_report(test_scores, test_labels, n_bins=n_bins)
    return {
        "target_macro_rmse": float(np.mean(per_class[:5])),
        "per_class_rmse": per_class.tolist(),
        "held_out_d_efficiency": report.geometric_mean_retention,
        "minimum_bin_count": int(np.min(np.bincount(test_labels, minlength=n_bins))),
    }


def _evaluate_partition(
    *,
    name: str,
    n_bins: int,
    reference: FlowCytData,
    template_mask: np.ndarray,
    template_labels: np.ndarray,
    test: FlowCytData,
    test_labels: np.ndarray,
    patients: np.ndarray,
    true_fractions: np.ndarray,
    test_scores: np.ndarray,
) -> tuple[str, dict[str, object], np.ndarray]:
    templates = estimate_bin_templates(
        reference.labels[template_mask],
        template_labels,
        reference.patients[template_mask],
        n_bins=n_bins,
    )
    predicted, convergence = _fit_patient_fractions(test, test_labels, templates, patients)
    metrics = _summary_metrics(true_fractions, predicted, test_scores, test_labels, n_bins)
    metrics["likelihood_convergence"] = convergence
    return name, metrics, predicted


def _prepare_experiment(
    data: FlowCytData,
    *,
    quick: bool,
    seed: int,
    score_max_per_patient_class: int | None = None,
    score_max_iter: int | None = None,
) -> _ExperimentContext:
    # The two score-model overrides exist so a regression test can exercise the
    # whole context on the frozen fixture in seconds. run_experiment never sets
    # them, so the published settings block stays the description of the run.
    reference = data.patients_in(REFERENCE_PATIENTS)
    test = data.patients_in(TEST_PATIENTS)
    if len(reference.labels) == 0 or len(test.labels) == 0:
        raise ValueError("data must contain both declared reference and test patients")
    score_fit = fit_score_model(
        reference,
        max_per_patient_class=(
            (256 if quick else 2_000)
            if score_max_per_patient_class is None
            else score_max_per_patient_class
        ),
        max_iter=(35 if quick else 120) if score_max_iter is None else score_max_iter,
        seed=seed,
    )
    theta0 = reference_composition(reference.labels, reference.patients)
    reference_scores = np.asarray(
        sq.ratios.MixtureParameterization(theta0).scores(
            sq.ratios.ratios_from_posteriors(
                score_fit.out_of_fold_probabilities, score_fit.model.class_priors
            )
        )
    )
    test_probabilities = score_fit.model.predict_proba(test.features)
    test_scores = np.asarray(
        sq.ratios.MixtureParameterization(theta0).scores(
            sq.ratios.ratios_from_posteriors(test_probabilities, score_fit.model.class_priors)
        )
    )
    patients, true_fractions = _true_patient_fractions(test)
    unbinned, convergence = _fit_unbinned_patient_fractions(
        test, test_probabilities / score_fit.model.class_priors[None, :], patients
    )
    raw_partition_mask, raw_validation_mask, template_mask = _role_masks(reference)
    partition_mask = _cap_group_mask(
        reference,
        raw_partition_mask,
        max_per_patient_class=256 if quick else 512,
        seed=seed + 101,
    )
    validation_mask = _cap_group_mask(
        reference,
        raw_validation_mask,
        max_per_patient_class=128 if quick else 256,
        seed=seed + 102,
    )
    weights = integration_weights(
        reference.labels[partition_mask], reference.patients[partition_mask], theta0
    )
    return _ExperimentContext(
        data=data,
        reference=reference,
        test=test,
        score_fit=score_fit,
        theta0=theta0,
        reference_scores=reference_scores,
        test_scores=test_scores,
        patients=patients,
        true_fractions=true_fractions,
        unbinned_fractions=unbinned,
        unbinned_convergence=convergence,
        partition_mask=partition_mask,
        validation_mask=validation_mask,
        template_mask=template_mask,
        weights=weights,
        transformed_markers=score_fit.model.transform.apply(reference.features),
        transformed_test_markers=score_fit.model.transform.apply(test.features),
    )


def _store_partition(
    context: _ExperimentContext,
    *,
    name: str,
    n_bins: int,
    template_labels: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[str, dict[str, object], np.ndarray]:
    return _evaluate_partition(
        name=name,
        n_bins=n_bins,
        reference=context.reference,
        template_mask=context.template_mask,
        template_labels=template_labels,
        test=context.test,
        test_labels=test_labels,
        patients=context.patients,
        true_fractions=context.true_fractions,
        test_scores=context.test_scores,
    )


def _evaluate_bin_count(
    context: _ExperimentContext,
    n_bins: int,
    *,
    operating_n_bins: int,
    quick: bool,
    seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray], np.ndarray]:
    metrics: dict[str, object] = {}
    predicted_by_method: dict[str, np.ndarray] = {}
    score_provenance = sq.ScoreProvenance(
        kind="estimated_ratio",
        description="patient-cross-fitted calibrated FlowCyt mixture score",
        ratio=sq.ratios.RatioProvenance(
            estimator="calibrated_classifier",
            parameterization="mixture",
            reference_fractions=tuple(float(value) for value in context.theta0),
            reference_component=len(context.theta0) - 1,
            training_priors=tuple(float(value) for value in context.score_fit.model.class_priors),
            calibration=str(context.score_fit.calibration_selection["selected_strategy"]),
        ),
    )
    common = {
        "n_bins": n_bins,
        "validation": sq.ScoreSample(
            context.reference_scores[context.validation_mask],
            integration_weights(
                context.reference.labels[context.validation_mask],
                context.reference.patients[context.validation_mask],
                context.theta0,
            ),
            provenance=score_provenance,
        ),
    }
    train_source = sq.ScoreSample(
        context.reference_scores[context.partition_mask],
        context.weights,
        provenance=score_provenance,
    )
    kmeans_result = sq.fit_quantizer(
        train_source,
        criterion=sq.NormalizedTrace(),
        config=sq.KMeansConfig(seed=seed, solver_restarts=3 if quick else 8),
        **common,
    )
    soft_result = sq.fit_quantizer(
        train_source,
        criterion=sq.DOptimality(),
        config=sq.SoftVoronoiConfig(
            seed=seed,
            initializer_restarts=3 if quick else 4,
            max_iter=50 if quick else 160,
            record_every=10,
        ),
        **common,
    )
    for method, result in (("score_kmeans", kmeans_result), ("soft_voronoi", soft_result)):
        key, values, predicted = _store_partition(
            context,
            name=f"{method}:{n_bins}",
            n_bins=n_bins,
            template_labels=np.asarray(
                result.predict_scores(context.reference_scores[context.template_mask])
            ),
            test_labels=predict_score_bins(result, context.test_scores),
        )
        if result.validation_report is None:
            raise AssertionError("validation inputs must produce a validation report")
        values["validation_d_efficiency"] = result.validation_report.geometric_mean_retention
        values["train_d_efficiency"] = result.train_report.geometric_mean_retention
        values["hardening_gap"] = result.hardening_gap
        values["information_kind"] = result.information_kind
        values["score_provenance"] = result.provenance.to_dict()
        metrics[key] = values
        predicted_by_method[key] = predicted

    # Runs at whatever the operating point is, not only at the default 8. The
    # extra literal meant the compile path was unreachable from the fast fixture
    # test, which runs a five-bin operating point - so a change that broke the
    # compiled rule passed the whole suite and only failed in the benchmarks.
    if n_bins == operating_n_bins:
        finite_d = sq.optimize_partition(
            context.reference_scores[context.partition_mask],
            weights=context.weights,
            n_bins=n_bins,
            criterion=sq.DOptimality(),
            config=sq.DExchangeConfig(
                seed=seed,
                initializer_restarts=3 if quick else 8,
            ),
            provenance=score_provenance,
        )
        if not finite_d.exchange_stable:
            raise RuntimeError("finite D FlowCyt assignment did not reach exchange stability")
        compiled_d = finite_d.compile_quantizer()
        key, values, predicted = _store_partition(
            context,
            name=f"finite_d_exchange:{n_bins}",
            n_bins=n_bins,
            template_labels=np.asarray(
                compiled_d.predict_scores(context.reference_scores[context.template_mask])
            ),
            test_labels=predict_score_bins(compiled_d, context.test_scores),
        )
        values.update(
            {
                "finite_assignment_d_efficiency": finite_d.train_report.geometric_mean_retention,
                # The compiled rule carries no report: it is a rule, not a fit.
                # Measuring it on the training scores is what this number always
                # meant, and it now says so rather than reading a report that
                # the partition happened to also own.
                "compiled_train_d_efficiency": compiled_d.evaluate_scores(
                    context.reference_scores[context.partition_mask],
                    context.weights,
                ).geometric_mean_retention,
                "accepted_moves": finite_d.accepted_moves,
                "exchange_stable": finite_d.exchange_stable,
                "best_remaining_gain": finite_d.best_remaining_gain,
                "compiled_training_labels_reproduced": True,
                "geometry_gap": 0.0,
                "information_kind": finite_d.information_kind,
                "score_provenance": finite_d.provenance.to_dict(),
            }
        )
        metrics[key] = values
        predicted_by_method[key] = predicted

    marker_model = KMeans(n_clusters=n_bins, n_init=5, random_state=seed).fit(
        context.transformed_markers[context.partition_mask], sample_weight=context.weights
    )
    key, values, predicted = _store_partition(
        context,
        name=f"marker_kmeans:{n_bins}",
        n_bins=n_bins,
        template_labels=marker_model.predict(context.transformed_markers[context.template_mask]),
        test_labels=marker_model.predict(context.transformed_test_markers),
    )
    metrics[key] = values
    predicted_by_method[key] = predicted

    partition_scores = context.reference_scores[context.partition_mask]
    fisher_matrix = (partition_scores * context.weights[:, None]).T @ partition_scores
    eigenvalues, eigenvectors = np.linalg.eigh((fisher_matrix + fisher_matrix.T) / 2.0)
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    projected_reference = context.reference_scores @ direction
    projected_test = context.test_scores @ direction
    edges = np.quantile(
        projected_reference[context.partition_mask], np.linspace(0, 1, n_bins + 1)[1:-1]
    )
    key, values, predicted = _store_partition(
        context,
        name=f"one_dimensional_score:{n_bins}",
        n_bins=n_bins,
        template_labels=np.digitize(projected_reference[context.template_mask], edges),
        test_labels=np.digitize(projected_test, edges),
    )
    metrics[key] = values
    predicted_by_method[key] = predicted

    pca = PCA(n_components=2, random_state=seed).fit(
        context.transformed_markers[context.partition_mask]
    )
    pca_reference = pca.transform(context.transformed_markers)
    pca_test = pca.transform(context.transformed_test_markers)
    first_parts, second_parts = _factor_pair(n_bins)
    first_edges = _grid_edges(pca_reference[context.partition_mask, 0], first_parts)
    second_edges = _grid_edges(pca_reference[context.partition_mask, 1], second_parts)
    key, values, predicted = _store_partition(
        context,
        name=f"two_dimensional_grid:{n_bins}",
        n_bins=n_bins,
        template_labels=_grid_labels(
            pca_reference[context.template_mask], first_edges, second_edges
        ),
        test_labels=_grid_labels(pca_test, first_edges, second_edges),
    )
    metrics[key] = values
    predicted_by_method[key] = predicted

    random_metrics: list[dict[str, object]] = []
    rng = np.random.default_rng(seed + n_bins)
    coordinates_reference = np.asarray(kmeans_result.transform.apply(context.reference_scores))
    coordinates_test = np.asarray(kmeans_result.transform.apply(context.test_scores))
    repeats = 5 if quick else 20
    for repeat in range(repeats):
        centers = coordinates_reference[
            rng.choice(np.flatnonzero(context.partition_mask), size=n_bins, replace=False)
        ]
        _, random_values, _ = _store_partition(
            context,
            name=f"random:{n_bins}:{repeat}",
            n_bins=n_bins,
            template_labels=_nearest(coordinates_reference[context.template_mask], centers),
            test_labels=_nearest(coordinates_test, centers),
        )
        random_metrics.append(random_values)
    metrics[f"random_score_voronoi:{n_bins}"] = {
        "target_macro_rmse_median": float(
            np.median(
                [float(cast(float, values["target_macro_rmse"])) for values in random_metrics]
            )
        ),
        "held_out_d_efficiency_median": float(
            np.median(
                [float(cast(float, values["held_out_d_efficiency"])) for values in random_metrics]
            )
        ),
        "repeats": repeats,
    }

    operating_composition = np.empty((0, len(CLASS_NAMES)))
    template_labels = np.asarray(
        soft_result.predict_scores(context.reference_scores[context.template_mask])
    )
    if n_bins == operating_n_bins:
        templates = estimate_bin_templates(
            context.reference.labels[context.template_mask],
            template_labels,
            context.reference.patients[context.template_mask],
            n_bins=n_bins,
        )
        operating_composition = templates * context.theta0[None, :]
        operating_composition /= np.sum(operating_composition, axis=1, keepdims=True)
    return metrics, predicted_by_method, operating_composition


def _run_partition_sweep(
    context: _ExperimentContext,
    bin_counts: tuple[int, ...],
    *,
    operating_n_bins: int,
    quick: bool,
    seed: int,
) -> _SweepResult:
    all_metrics: dict[str, object] = {}
    all_predicted: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    operating_composition = np.empty((0, len(CLASS_NAMES)))
    for n_bins in bin_counts:
        started = perf_counter()
        metrics, predicted, composition = _evaluate_bin_count(
            context,
            n_bins,
            operating_n_bins=operating_n_bins,
            quick=quick,
            seed=seed,
        )
        all_metrics.update(metrics)
        all_predicted.update(predicted)
        if len(composition):
            operating_composition = composition
        timings[f"partitions_and_baselines_{n_bins}_bins"] = perf_counter() - started
    return _SweepResult(
        all_metrics,
        all_predicted,
        operating_composition,
        timings,
    )


def _acceptance_metrics(
    metrics: dict[str, object], bin_counts: tuple[int, ...]
) -> dict[str, object]:
    rows = []
    for n_bins in bin_counts:
        learned = cast(dict[str, object], metrics[f"soft_voronoi:{n_bins}"])
        random = cast(dict[str, object], metrics[f"random_score_voronoi:{n_bins}"])
        marker = cast(dict[str, object], metrics[f"marker_kmeans:{n_bins}"])
        rows.append(
            {
                "n_bins": n_bins,
                "beats_random_d_efficiency": float(cast(float, learned["held_out_d_efficiency"]))
                > float(cast(float, random["held_out_d_efficiency_median"])),
                "no_worse_than_marker_rmse": float(cast(float, learned["target_macro_rmse"]))
                <= float(cast(float, marker["target_macro_rmse"])),
            }
        )
    return {
        "rows": rows,
        "random_d_efficiency_wins": sum(bool(row["beats_random_d_efficiency"]) for row in rows),
        "marker_rmse_noninferiority_wins": sum(
            bool(row["no_worse_than_marker_rmse"]) for row in rows
        ),
        "required_random_wins": 5,
        "required_marker_wins": 4,
    }


def _assemble_metrics(
    context: _ExperimentContext,
    sweep: _SweepResult,
    bin_counts: tuple[int, ...],
    *,
    operating_n_bins: int,
    uncertainty_n_bins: int,
    scientific_closure: dict[str, object],
    quick: bool,
    elapsed_seconds: float,
    score_elapsed_seconds: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    unbinned_errors = context.unbinned_fractions - context.true_fractions
    metrics = {
        "dataset": {
            "patient_compositions": [
                {
                    "patient": int(patient),
                    "split": "reference" if patient in REFERENCE_PATIENTS else "test",
                    "fractions": (
                        np.bincount(
                            context.data.labels[context.data.patients == patient],
                            minlength=len(CLASS_NAMES),
                        )
                        / np.count_nonzero(context.data.patients == patient)
                    ).tolist(),
                }
                for patient in np.unique(context.data.patients)
            ],
            "class_names": list(CLASS_NAMES),
        },
        "unbinned_classifier_ratio": {
            "target_macro_rmse": float(np.mean(np.sqrt(np.mean(unbinned_errors**2, axis=0))[:5])),
            "per_class_rmse": np.sqrt(np.mean(unbinned_errors**2, axis=0)).tolist(),
            "likelihood_convergence": context.unbinned_convergence,
        },
        **sweep.metrics,
    }
    predicted = {"unbinned_classifier_ratio": context.unbinned_fractions, **sweep.predicted}
    metrics["calibration"] = _calibration_metrics(
        context.score_fit.out_of_fold_probabilities,
        context.reference.labels,
        context.reference.patients,
    )
    metrics["calibration_selection"] = context.score_fit.calibration_selection
    metrics["operating_partition"] = {
        "n_bins": operating_n_bins,
        "reference_bin_composition": sweep.operating_bin_composition.tolist(),
    }
    reference_median = np.median(context.transformed_markers, axis=0)
    test_median = np.median(context.transformed_test_markers, axis=0)
    metrics["shift"] = {
        "median_absolute_standardized_marker_shift": float(
            np.median(np.abs(test_median - reference_median))
        ),
        "maximum_absolute_standardized_marker_shift": float(
            np.max(np.abs(test_median - reference_median))
        ),
        "score_mean_shift_norm": float(
            np.linalg.norm(
                np.mean(context.test_scores, axis=0) - np.mean(context.reference_scores, axis=0)
            )
        ),
    }
    final_predictions = predicted[f"soft_voronoi:{operating_n_bins}"]
    metrics["patients"] = [
        {
            "patient": int(patient),
            "true_fractions": truth.tolist(),
            "soft_voronoi_fractions": learned.tolist(),
            "unbinned_classifier_ratio_fractions": unbinned.tolist(),
        }
        for patient, truth, learned, unbinned in zip(
            context.patients,
            context.true_fractions,
            final_predictions,
            context.unbinned_fractions,
            strict=True,
        )
    ]
    metrics["acceptance"] = _acceptance_metrics(metrics, bin_counts)
    metrics["scientific_closure"] = scientific_closure
    timings = {"score_model_and_scores": score_elapsed_seconds, **sweep.timings}
    metrics["run"] = {
        "reference_patients": list(REFERENCE_PATIENTS),
        "test_patients": list(TEST_PATIENTS),
        "reference_composition": context.theta0.tolist(),
        "score_calibration_strategy": context.score_fit.model.calibration.strategy,
        "score_temperature": context.score_fit.model.temperature,
        "score_class_priors": context.score_fit.model.class_priors.tolist(),
        "classifier_test_posterior_evaluations": 1,
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_megabytes": _peak_rss_megabytes(),
        "quick": quick,
        "rows": {
            "reference": len(context.reference.labels),
            "test": len(context.test.labels),
            "partition": int(np.count_nonzero(context.partition_mask)),
            "validation": int(np.count_nonzero(context.validation_mask)),
            "templates": int(np.count_nonzero(context.template_mask)),
        },
        "settings": {
            "operating_n_bins": operating_n_bins,
            "uncertainty_n_bins": uncertainty_n_bins,
            "score_max_per_patient_class": 256 if quick else 2_000,
            "score_max_iter": 35 if quick else 120,
            "partition_max_per_patient_class": 256 if quick else 512,
            "validation_max_per_patient_class": 128 if quick else 256,
            "soft_n_init": 3 if quick else 4,
            "soft_max_steps": 50 if quick else 160,
            "random_repeats": 5 if quick else 20,
            "closure_pseudo_repeats": 1 if quick else 20,
            "closure_seed_repeats": 1 if quick else 10,
            "uncertainty_coverage_draws": 30 if quick else 1_000,
        },
        "timings_seconds": timings,
    }
    metrics["uncertainty"] = {
        "n_bins": uncertainty_n_bins,
        **cast(
            dict[str, object],
            scientific_closure["uncertainty_coverage"],
        ),
    }
    return metrics, predicted


def run_experiment(
    data: FlowCytData,
    *,
    bin_counts: tuple[int, ...] = (5, 8, 10, 15, 20, 30),
    operating_n_bins: int = 8,
    uncertainty_n_bins: int = 30,
    quick: bool = False,
    seed: int = 2026,
) -> ExperimentResult:
    """Run the complete label-blind held-out population experiment."""
    if operating_n_bins not in bin_counts:
        raise ValueError("operating_n_bins must be included in bin_counts")
    if uncertainty_n_bins not in bin_counts:
        raise ValueError("uncertainty_n_bins must be included in bin_counts")
    started = perf_counter()
    score_started = perf_counter()
    context = _prepare_experiment(data, quick=quick, seed=seed)
    score_elapsed = perf_counter() - score_started
    sweep = _run_partition_sweep(
        context,
        bin_counts,
        operating_n_bins=operating_n_bins,
        quick=quick,
        seed=seed,
    )
    closure_started = perf_counter()
    scientific_closure = run_scientific_closure(
        ClosureInputs(
            reference=context.reference,
            score_fit=context.score_fit,
            theta0=context.theta0,
            reference_scores=context.reference_scores,
            partition_mask=context.partition_mask,
            validation_mask=context.validation_mask,
            template_mask=context.template_mask,
            partition_weights=context.weights,
        ),
        uncertainty_n_bins=uncertainty_n_bins,
        quick=quick,
        seed=seed,
    )
    sweep.timings["reference_only_scientific_closure"] = perf_counter() - closure_started
    metrics, predicted = _assemble_metrics(
        context,
        sweep,
        bin_counts,
        operating_n_bins=operating_n_bins,
        uncertainty_n_bins=uncertainty_n_bins,
        scientific_closure=scientific_closure,
        quick=quick,
        elapsed_seconds=perf_counter() - started,
        score_elapsed_seconds=score_elapsed,
    )
    return ExperimentResult(
        metrics=metrics,
        true_fractions=context.true_fractions,
        predicted_fractions=predicted,
        bin_counts=bin_counts,
        operating_n_bins=operating_n_bins,
        operating_bin_composition=sweep.operating_bin_composition,
        patient_ids=context.patients,
    )
