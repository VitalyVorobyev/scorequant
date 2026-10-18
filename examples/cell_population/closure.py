"""Reference-only scientific closure diagnostics for the FlowCyt study."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

import scorequant as sq

from .data import CLASS_NAMES, FlowCytData
from .likelihood import estimate_bin_templates, fit_binned_mixture, fit_unbinned_mixture
from .scores import (
    CALIBRATION_STRATEGIES,
    ScoreFit,
    _fit_posterior_calibration,
    integration_weights,
)

_RANK_RTOL = 1e-10
_BOUNDARY_EVENT_EQUIVALENT = 0.5


@dataclass(frozen=True, slots=True)
class ClosureInputs:
    """Reference-only inputs needed by the application audit."""

    reference: FlowCytData
    score_fit: ScoreFit
    theta0: np.ndarray
    reference_scores: np.ndarray
    partition_mask: np.ndarray
    validation_mask: np.ndarray
    template_mask: np.ndarray
    partition_weights: np.ndarray


def _normalized_weights(weights: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (size,) or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("weights must be finite, nonnegative, and match the rows")
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("weights must have positive total")
    return values / total


def conditional_fisher_information(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute fixed-total information without changing the supplied score vectors."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite matrix")
    measure = _normalized_weights(weights, len(values))
    mean = np.sum(measure[:, None] * values, axis=0)
    centered = values - mean
    information = (centered * measure[:, None]).T @ centered
    return (information + information.T) / 2.0


def conditional_binned_fisher_information(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    n_bins: int,
) -> np.ndarray:
    """Compute fixed-total information retained by hard bin labels."""
    values = np.asarray(scores, dtype=np.float64)
    assignments = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or assignments.shape != (len(values),):
        raise ValueError("scores and labels must have matching rows")
    if n_bins < 1 or np.any((assignments < 0) | (assignments >= n_bins)):
        raise ValueError("labels must lie inside the declared bin range")
    measure = _normalized_weights(weights, len(values))
    global_mean = np.sum(measure[:, None] * values, axis=0)
    information = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    for bin_index in range(n_bins):
        mask = assignments == bin_index
        mass = float(np.sum(measure[mask]))
        if mass == 0:
            continue
        bin_mean = np.sum(measure[mask, None] * values[mask], axis=0) / mass
        displacement = bin_mean - global_mean
        information += mass * np.outer(displacement, displacement)
    return (information + information.T) / 2.0


def _matrix_rank(values: np.ndarray, *, rtol: float = _RANK_RTOL) -> int:
    singular_values = np.linalg.svd(values, compute_uv=False)
    if len(singular_values) == 0 or singular_values[0] == 0:
        return 0
    return int(np.count_nonzero(singular_values > singular_values[0] * rtol))


def fixed_total_partition_audit(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    n_bins: int,
) -> dict[str, object]:
    """Summarize conditional information and its exact rank limitation."""
    full = conditional_fisher_information(scores, weights)
    binned = conditional_binned_fisher_information(scores, labels, weights, n_bins=n_bins)
    eigenvalues, eigenvectors = np.linalg.eigh(full)
    maximum = float(np.max(eigenvalues, initial=0.0))
    keep = eigenvalues > maximum * _RANK_RTOL
    full_rank = int(np.count_nonzero(keep))
    if full_rank:
        basis = eigenvectors[:, keep]
        scale = np.sqrt(eigenvalues[keep])
        retained = (basis.T @ binned @ basis) / scale[:, None] / scale[None, :]
        retained = (retained + retained.T) / 2.0
        retained_eigenvalues = np.linalg.eigvalsh(retained)
        retained_eigenvalues = np.clip(retained_eigenvalues, 0.0, 1.0)
    else:
        retained_eigenvalues = np.empty(0, dtype=np.float64)
    retained_rank = int(
        np.count_nonzero(
            retained_eigenvalues > (float(np.max(retained_eigenvalues, initial=0.0)) * _RANK_RTOL)
        )
    )
    d_efficiency = (
        float(np.exp(np.mean(np.log(retained_eigenvalues))))
        if full_rank and retained_rank == full_rank and np.all(retained_eigenvalues > 0)
        else 0.0
    )
    residual = (full - binned + (full - binned).T) / 2.0
    measure = _normalized_weights(weights, len(scores))
    mean_score = np.sum(measure[:, None] * np.asarray(scores), axis=0)
    return {
        "full_rank": full_rank,
        "retained_rank": retained_rank,
        "rank_bound": min(full_rank, n_bins - 1),
        "retained_eigenvalues": retained_eigenvalues.tolist(),
        "d_efficiency": d_efficiency,
        "mean_score": mean_score.tolist(),
        "mean_score_norm": float(np.linalg.norm(mean_score)),
        "psd_residual_min_eigenvalue": float(np.min(np.linalg.eigvalsh(residual))),
    }


def template_identifiability_audit(
    templates: np.ndarray,
    reference_fractions: np.ndarray,
) -> dict[str, object]:
    """Diagnose identifiability of fixed-total mixture fractions from templates."""
    matrix = np.asarray(templates, dtype=np.float64)
    reference = np.asarray(reference_fractions, dtype=np.float64)
    if matrix.ndim != 2 or reference.shape != (matrix.shape[1],):
        raise ValueError("templates and reference fractions have incompatible shapes")
    if np.any(matrix < 0) or not np.isfinite(matrix).all():
        raise ValueError("templates must be finite and nonnegative")
    if np.any(reference <= 0) or not np.isclose(np.sum(reference), 1.0):
        raise ValueError("reference fractions must be an interior simplex point")
    contrast = matrix[:, :-1] - matrix[:, [-1]]
    _, singular_values, right_vectors = np.linalg.svd(contrast, full_matrices=True)
    rank = _matrix_rank(contrast)
    n_free = matrix.shape[1] - 1
    full_rank = rank == n_free
    condition_number: float | None = None
    if full_rank:
        condition_number = float(singular_values[0] / singular_values[-1])

    witness: dict[str, object] | None = None
    if not full_rank:
        free_direction = right_vectors[-1]
        direction = np.concatenate([free_direction, [-np.sum(free_direction)]])
        direction /= np.linalg.norm(direction)
        nonzero = np.abs(direction) > 0
        maximum_step = float(np.min(reference[nonzero] / np.abs(direction[nonzero])))
        step = 0.25 * maximum_step
        first = reference + step * direction
        second = reference - step * direction
        witness = {
            "first_fractions": first.tolist(),
            "second_fractions": second.tolist(),
            "fraction_separation_norm": float(np.linalg.norm(first - second)),
            "maximum_bin_probability_difference": float(
                np.max(np.abs(matrix @ first - matrix @ second))
            ),
        }
    return {
        "n_bins": matrix.shape[0],
        "n_classes": matrix.shape[1],
        "required_rank": n_free,
        "rank_bound": min(matrix.shape[0] - 1, n_free),
        "effective_rank": rank,
        "full_rank": full_rank,
        "singular_values": singular_values.tolist(),
        "condition_number": condition_number,
        "nonidentifiability_witness": witness,
    }


def _patient_residuals(
    ratios: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
) -> list[float]:
    residuals: list[float] = []
    uniform = np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
    for patient in np.unique(patients):
        mask = patients == patient
        if set(int(value) for value in np.unique(labels[mask])) != set(range(len(CLASS_NAMES))):
            continue
        weights = integration_weights(labels[mask], patients[mask], uniform)
        normalizers = np.sum(weights[:, None] * ratios[mask], axis=0)
        residuals.append(float(np.max(np.abs(normalizers - 1.0))))
    return residuals


def ratio_model_audit(
    score_fit: ScoreFit,
    reference: FlowCytData,
    theta0: np.ndarray,
) -> dict[str, object]:
    """Audit every fixed calibration candidate on reference OOF predictions."""
    raw = score_fit.raw_out_of_fold_probabilities
    uniform = np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
    uniform_weights = integration_weights(reference.labels, reference.patients, uniform)
    theta_weights = integration_weights(reference.labels, reference.patients, theta0)
    rows: dict[str, object] = {}
    for strategy in CALIBRATION_STRATEGIES:
        calibration = _fit_posterior_calibration(
            strategy, raw, reference.labels, reference.patients
        )
        probabilities = calibration.apply(raw)
        ratios = np.asarray(
            sq.ratios.ratios_from_posteriors(probabilities, calibration.class_priors)
        )
        closure = sq.ratios.ratio_closure_report(ratios, uniform_weights)
        scores = np.asarray(sq.ratios.MixtureParameterization(theta0).scores(ratios))
        mean_score = np.sum(theta_weights[:, None] * scores, axis=0)
        patient_residuals = _patient_residuals(ratios, reference.labels, reference.patients)
        rows[strategy] = {
            "class_priors": calibration.class_priors.tolist(),
            "temperature": calibration.temperature,
            "component_ratio_normalizers": np.asarray(closure.normalizers).tolist(),
            "maximum_normalization_residual": closure.max_residual,
            "mean_score": mean_score.tolist(),
            "mean_score_norm": float(np.linalg.norm(mean_score)),
            "patient_residual_count": len(patient_residuals),
            "patient_residual_median": (
                float(np.median(patient_residuals)) if patient_residuals else None
            ),
            "patient_residual_maximum": max(patient_residuals, default=None),
        }
    return {
        "selected_strategy": score_fit.model.calibration.strategy,
        "strategies": rows,
    }


def _composition_grid(theta0: np.ndarray) -> list[tuple[str, np.ndarray]]:
    rows = [("reference", np.asarray(theta0, dtype=np.float64))]
    for class_index, class_name in enumerate(CLASS_NAMES[:-1]):
        for factor in (0.5, 2.0):
            composition = np.asarray(theta0, dtype=np.float64).copy()
            composition[class_index] *= factor
            composition /= np.sum(composition)
            rows.append((f"{class_name}_x{factor:g}", composition))
    return rows


def _largest_remainder_counts(composition: np.ndarray, total: int) -> np.ndarray:
    expected = np.asarray(composition) * total
    counts = np.floor(expected).astype(np.int64)
    remainder = total - int(np.sum(counts))
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts


def _sample_reference_indices(
    reference: FlowCytData,
    mask: np.ndarray,
    composition: np.ndarray,
    *,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    counts = _largest_remainder_counts(composition, size)
    selected: list[np.ndarray] = []
    for class_index, count in enumerate(counts):
        available_patients = [
            patient
            for patient in np.unique(reference.patients[mask])
            if np.any(mask & (reference.patients == patient) & (reference.labels == class_index))
        ]
        if count and not available_patients:
            raise ValueError(f"class {class_index} has no validation source rows")
        chosen_patients = rng.choice(available_patients, size=count, replace=True)
        class_indices: list[np.ndarray] = []
        for patient in np.unique(chosen_patients):
            patient_count = int(np.count_nonzero(chosen_patients == patient))
            pool = np.flatnonzero(
                mask & (reference.patients == patient) & (reference.labels == class_index)
            )
            class_indices.append(rng.choice(pool, size=patient_count, replace=True))
        if class_indices:
            selected.append(np.concatenate(class_indices))
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    realized = np.bincount(reference.labels[indices], minlength=len(CLASS_NAMES)) / len(indices)
    return indices, realized


def _fit_partition(
    inputs: ClosureInputs,
    method: str,
    n_bins: int,
    *,
    seed: int,
    quick: bool,
) -> sq.QuantizerResult:
    validation_weights = integration_weights(
        inputs.reference.labels[inputs.validation_mask],
        inputs.reference.patients[inputs.validation_mask],
        inputs.theta0,
    )
    if method == "score_kmeans":
        config: sq.KMeansConfig | sq.SoftVoronoiConfig = sq.KMeansConfig(
            seed=seed, solver_restarts=3 if quick else 8
        )
    elif method == "soft_voronoi":
        config = sq.SoftVoronoiConfig(
            seed=seed,
            initializer_restarts=3 if quick else 4,
            max_iter=50 if quick else 160,
            record_every=10,
        )
    else:
        raise ValueError(f"unknown closure partition method: {method}")
    return sq.fit_quantizer(
        sq.ScoreSample(inputs.reference_scores[inputs.partition_mask], inputs.partition_weights),
        validation=sq.ScoreSample(
            inputs.reference_scores[inputs.validation_mask], validation_weights
        ),
        n_bins=n_bins,
        criterion=(
            sq.NormalizedTrace() if isinstance(config, sq.KMeansConfig) else sq.DOptimality()
        ),
        config=config,
    )


def _partition_templates(
    inputs: ClosureInputs,
    result: sq.QuantizerResult,
    n_bins: int,
) -> np.ndarray:
    labels = np.asarray(result.predict_scores(inputs.reference_scores[inputs.template_mask]))
    return estimate_bin_templates(
        inputs.reference.labels[inputs.template_mask],
        labels,
        inputs.reference.patients[inputs.template_mask],
        n_bins=n_bins,
    )


def _population_limit_metrics(
    templates: np.ndarray,
    compositions: list[tuple[str, np.ndarray]],
    *,
    event_count: int,
) -> dict[str, object]:
    errors: list[np.ndarray] = []
    converged = 0
    maximum_iterations = 0
    for _, composition in compositions:
        estimate = fit_binned_mixture(
            event_count * (templates @ composition), templates, max_iter=2_000
        )
        errors.append(estimate.fractions - composition)
        converged += int(estimate.converged)
        maximum_iterations = max(maximum_iterations, estimate.iterations)
    error_matrix = np.asarray(errors)
    return {
        "maximum_absolute_error": float(np.max(np.abs(error_matrix))),
        "maximum_absolute_mean_bias": float(np.max(np.abs(np.mean(error_matrix, axis=0)))),
        "target_macro_rmse": float(np.mean(np.sqrt(np.mean(error_matrix[:, :-1] ** 2, axis=0)))),
        "converged_compositions": converged,
        "total_compositions": len(compositions),
        "maximum_iterations": maximum_iterations,
    }


def _summarize_errors(
    errors: list[np.ndarray], converged: int, iterations: int
) -> dict[str, object]:
    values = np.asarray(errors)
    per_class = np.sqrt(np.mean(values**2, axis=0))
    return {
        "target_macro_rmse": float(np.mean(per_class[:-1])),
        "per_class_rmse": per_class.tolist(),
        "mean_bias": np.mean(values, axis=0).tolist(),
        "converged_pseudo_patients": converged,
        "total_pseudo_patients": len(values),
        "maximum_iterations": iterations,
    }


def _pseudo_patient_audit(
    inputs: ClosureInputs,
    partitions: Mapping[str, tuple[sq.QuantizerResult, np.ndarray]],
    compositions: list[tuple[str, np.ndarray]],
    *,
    repeats: int,
    event_count: int,
    seed: int,
) -> dict[str, object]:
    errors: dict[str, list[np.ndarray]] = {"unbinned_classifier_ratio": []}
    errors.update({name: [] for name in partitions})
    convergence = {name: 0 for name in errors}
    maximum_iterations = {name: 0 for name in errors}
    composition_errors = {
        composition_name: {name: [] for name in errors} for composition_name, _ in compositions
    }
    composition_convergence = {
        composition_name: {name: 0 for name in errors} for composition_name, _ in compositions
    }
    composition_iterations = {
        composition_name: {name: 0 for name in errors} for composition_name, _ in compositions
    }
    ratios = (
        inputs.score_fit.out_of_fold_probabilities / inputs.score_fit.model.class_priors[None, :]
    )
    rng = np.random.default_rng(seed)
    for composition_name, composition in compositions:
        for _ in range(repeats):
            indices, truth = _sample_reference_indices(
                inputs.reference,
                inputs.validation_mask,
                composition,
                size=event_count,
                rng=rng,
            )
            unbinned = fit_unbinned_mixture(ratios[indices], max_iter=2_000)
            unbinned_error = unbinned.fractions - truth
            errors["unbinned_classifier_ratio"].append(unbinned_error)
            composition_errors[composition_name]["unbinned_classifier_ratio"].append(unbinned_error)
            convergence["unbinned_classifier_ratio"] += int(unbinned.converged)
            composition_convergence[composition_name]["unbinned_classifier_ratio"] += int(
                unbinned.converged
            )
            maximum_iterations["unbinned_classifier_ratio"] = max(
                maximum_iterations["unbinned_classifier_ratio"], unbinned.iterations
            )
            composition_iterations[composition_name]["unbinned_classifier_ratio"] = max(
                composition_iterations[composition_name]["unbinned_classifier_ratio"],
                unbinned.iterations,
            )
            for name, (partition, templates) in partitions.items():
                labels = np.asarray(partition.predict_scores(inputs.reference_scores[indices]))
                counts = np.bincount(labels, minlength=templates.shape[0])
                estimate = fit_binned_mixture(counts, templates, max_iter=2_000)
                error = estimate.fractions - truth
                errors[name].append(error)
                composition_errors[composition_name][name].append(error)
                convergence[name] += int(estimate.converged)
                composition_convergence[composition_name][name] += int(estimate.converged)
                maximum_iterations[name] = max(maximum_iterations[name], estimate.iterations)
                composition_iterations[composition_name][name] = max(
                    composition_iterations[composition_name][name], estimate.iterations
                )
    return {
        "protocol": {
            "source": "reference_validation_rows_only",
            "compositions": [
                {"name": name, "fractions": composition.tolist()}
                for name, composition in compositions
            ],
            "repeats_per_composition": repeats,
            "events_per_pseudo_patient": event_count,
            "seed": seed,
        },
        "methods": {
            name: _summarize_errors(method_errors, convergence[name], maximum_iterations[name])
            for name, method_errors in errors.items()
        },
        "by_composition": {
            composition_name: {
                name: _summarize_errors(
                    method_errors,
                    composition_convergence[composition_name][name],
                    composition_iterations[composition_name][name],
                )
                for name, method_errors in methods.items()
            }
            for composition_name, methods in composition_errors.items()
        },
    }


def uncertainty_coverage_audit(
    templates: np.ndarray,
    theta0: np.ndarray,
    *,
    draws: int,
    event_count: int,
    seed: int,
) -> dict[str, object]:
    """Check local covariance only where the constrained MLE remains interior."""
    enriched = np.asarray(theta0, dtype=np.float64).copy()
    remaining = 1.0 - 0.005
    without_mast = np.delete(enriched, 3)
    enriched[np.arange(len(enriched)) != 3] = remaining * without_mast / np.sum(without_mast)
    enriched[3] = 0.005
    scenarios = (("reference_like", np.asarray(theta0)), ("mast_enriched", enriched))
    rng = np.random.default_rng(seed)
    boundary_tolerance = _BOUNDARY_EVENT_EQUIVALENT / event_count
    results: dict[str, object] = {}
    for scenario_name, truth in scenarios:
        estimates: list[np.ndarray] = []
        standard_errors: list[np.ndarray] = []
        converged = 0
        maximum_iterations = 0
        for _ in range(draws):
            counts = rng.multinomial(event_count, templates @ truth)
            estimate = fit_binned_mixture(counts, templates, max_iter=2_000)
            estimates.append(estimate.fractions)
            standard_errors.append(estimate.standard_errors)
            converged += int(estimate.converged)
            maximum_iterations = max(maximum_iterations, estimate.iterations)
        fitted = np.asarray(estimates)
        errors = fitted - truth
        predicted = np.asarray(standard_errors)
        class_rows: dict[str, object] = {}
        for class_index, class_name in enumerate(CLASS_NAMES):
            boundary = fitted[:, class_index] <= boundary_tolerance
            boundary_fraction = float(np.mean(boundary))
            status = "boundary_dominated" if boundary_fraction > 0.05 else "interior"
            interior = ~boundary
            coverage: float | None = None
            ratio: float | None = None
            if status == "interior" and np.any(interior):
                empirical = float(np.std(fitted[:, class_index], ddof=1))
                local = float(np.median(predicted[interior, class_index]))
                ratio = local / empirical if empirical > 0 else None
                coverage = float(
                    np.mean(
                        np.abs(errors[interior, class_index]) <= predicted[interior, class_index]
                    )
                )
            class_rows[class_name] = {
                "status": status,
                "truth": float(truth[class_index]),
                "bias": float(np.mean(errors[:, class_index])),
                "empirical_standard_deviation": float(np.std(fitted[:, class_index], ddof=1)),
                "median_local_fisher_error": float(np.median(predicted[:, class_index])),
                "boundary_hit_fraction": boundary_fraction,
                "interior_standard_error_ratio": ratio,
                "interior_68_percent_coverage": coverage,
            }
        results[scenario_name] = {
            "fractions": truth.tolist(),
            "converged_draws": converged,
            "total_draws": draws,
            "maximum_iterations": maximum_iterations,
            "classes": class_rows,
        }
    return {
        "protocol": {
            "source": "frozen_reference_templates",
            "draws_per_scenario": draws,
            "events_per_draw": event_count,
            "boundary_threshold": boundary_tolerance,
            "boundary_event_equivalent": _BOUNDARY_EVENT_EQUIVALENT,
            "boundary_dominated_threshold": 0.05,
            "seed": seed,
        },
        "scenarios": results,
    }


def _range(values: list[float]) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def _seed_stability_audit(
    inputs: ClosureInputs,
    compositions: list[tuple[str, np.ndarray]],
    representative: Mapping[str, tuple[sq.QuantizerResult, np.ndarray]],
    *,
    seeds: range,
    quick: bool,
) -> dict[str, object]:
    validation_scores = inputs.reference_scores[inputs.validation_mask]
    validation_weights = integration_weights(
        inputs.reference.labels[inputs.validation_mask],
        inputs.reference.patients[inputs.validation_mask],
        inputs.theta0,
    )
    rows: list[dict[str, object]] = []
    for method in ("score_kmeans", "soft_voronoi"):
        for n_bins in (5, 6, 8):
            for seed in seeds:
                name = f"{method}:{n_bins}"
                if seed == seeds.start and name in representative:
                    partition, templates = representative[name]
                else:
                    partition = _fit_partition(inputs, method, n_bins, seed=seed, quick=quick)
                    templates = _partition_templates(inputs, partition, n_bins)
                validation_labels = np.asarray(partition.predict_scores(validation_scores))
                information = fixed_total_partition_audit(
                    validation_scores,
                    validation_labels,
                    validation_weights,
                    n_bins=n_bins,
                )
                identifiability = template_identifiability_audit(templates, inputs.theta0)
                population_limit = _population_limit_metrics(
                    templates, compositions, event_count=20_000
                )
                rows.append(
                    {
                        "method": method,
                        "n_bins": n_bins,
                        "seed": seed,
                        "fixed_total_rank": information["retained_rank"],
                        "fixed_total_d_efficiency": information["d_efficiency"],
                        "template_rank": identifiability["effective_rank"],
                        "minimum_validation_bin_count": int(
                            np.min(np.bincount(validation_labels, minlength=n_bins))
                        ),
                        "population_limit_target_macro_rmse": population_limit["target_macro_rmse"],
                        "population_limit_maximum_absolute_mean_bias": population_limit[
                            "maximum_absolute_mean_bias"
                        ],
                        "population_limit_converged": population_limit["converged_compositions"],
                    }
                )
    summaries: dict[str, object] = {}
    for method in ("score_kmeans", "soft_voronoi"):
        for n_bins in (5, 6, 8):
            selected = [row for row in rows if row["method"] == method and row["n_bins"] == n_bins]
            summaries[f"{method}:{n_bins}"] = {
                "fixed_total_rank_range": [
                    min(int(row["fixed_total_rank"]) for row in selected),
                    max(int(row["fixed_total_rank"]) for row in selected),
                ],
                "template_rank_range": [
                    min(int(row["template_rank"]) for row in selected),
                    max(int(row["template_rank"]) for row in selected),
                ],
                "fixed_total_d_efficiency": _range(
                    [float(row["fixed_total_d_efficiency"]) for row in selected]
                ),
                "minimum_validation_bin_count": _range(
                    [float(row["minimum_validation_bin_count"]) for row in selected]
                ),
                "population_limit_target_macro_rmse": _range(
                    [float(row["population_limit_target_macro_rmse"]) for row in selected]
                ),
                "population_limit_maximum_absolute_mean_bias": _range(
                    [float(row["population_limit_maximum_absolute_mean_bias"]) for row in selected]
                ),
                "population_limit_converged_compositions": _range(
                    [float(row["population_limit_converged"]) for row in selected]
                ),
            }
    return {"seeds": list(seeds), "rows": rows, "summary": summaries}


def run_scientific_closure(
    inputs: ClosureInputs,
    *,
    uncertainty_n_bins: int,
    quick: bool,
    seed: int,
) -> dict[str, object]:
    """Run the complete reference-only FlowCyt scientific audit."""
    compositions = _composition_grid(inputs.theta0)
    representative: dict[str, tuple[sq.QuantizerResult, np.ndarray]] = {}
    fixed_total: dict[str, object] = {}
    identifiability: dict[str, object] = {}
    population_limit: dict[str, object] = {}
    validation_scores = inputs.reference_scores[inputs.validation_mask]
    validation_weights = integration_weights(
        inputs.reference.labels[inputs.validation_mask],
        inputs.reference.patients[inputs.validation_mask],
        inputs.theta0,
    )
    for method in ("score_kmeans", "soft_voronoi"):
        for n_bins in (5, 6, 8):
            name = f"{method}:{n_bins}"
            partition = _fit_partition(inputs, method, n_bins, seed=seed, quick=quick)
            templates = _partition_templates(inputs, partition, n_bins)
            representative[name] = (partition, templates)
            validation_labels = np.asarray(partition.predict_scores(validation_scores))
            fixed_total[name] = fixed_total_partition_audit(
                validation_scores,
                validation_labels,
                validation_weights,
                n_bins=n_bins,
            )
            identifiability[name] = template_identifiability_audit(templates, inputs.theta0)
            population_limit[name] = _population_limit_metrics(
                templates, compositions, event_count=20_000
            )

    uncertainty_name = f"soft_voronoi:{uncertainty_n_bins}"
    if uncertainty_name in representative:
        uncertainty_templates = representative[uncertainty_name][1]
    else:
        uncertainty_partition = _fit_partition(
            inputs, "soft_voronoi", uncertainty_n_bins, seed=seed, quick=quick
        )
        uncertainty_templates = _partition_templates(
            inputs, uncertainty_partition, uncertainty_n_bins
        )

    pseudo_patients = _pseudo_patient_audit(
        inputs,
        representative,
        compositions,
        repeats=1 if quick else 20,
        event_count=2_000 if quick else 20_000,
        seed=seed + 30_000,
    )
    seed_count = 1 if quick else 10
    stability = _seed_stability_audit(
        inputs,
        compositions,
        representative,
        seeds=range(seed, seed + seed_count),
        quick=quick,
    )
    uncertainty = uncertainty_coverage_audit(
        uncertainty_templates,
        inputs.theta0,
        draws=30 if quick else 1_000,
        event_count=2_000 if quick else 20_000,
        seed=seed + 40_000,
    )
    return {
        "reference_only": True,
        "ratio_model": ratio_model_audit(inputs.score_fit, inputs.reference, inputs.theta0),
        "fixed_total_information": fixed_total,
        "template_identifiability": identifiability,
        "population_limit": population_limit,
        "pseudo_patients": pseudo_patients,
        "seed_stability": stability,
        "uncertainty_coverage": uncertainty,
    }
