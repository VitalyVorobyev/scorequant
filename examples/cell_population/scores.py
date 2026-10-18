"""Calibrated classifier-to-score bridge for the FlowCyt example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import scorequant as sq

from .data import (
    CLASS_NAMES,
    REFERENCE_FOLDS,
    FlowCytData,
    RobustArcsinhTransform,
    deterministic_group_sample,
)

CalibrationStrategy = Literal[
    "raw_declared_prior",
    "raw_oof_prior",
    "temperature_oof_prior",
]
CALIBRATION_STRATEGIES: tuple[CalibrationStrategy, ...] = (
    "raw_declared_prior",
    "raw_oof_prior",
    "temperature_oof_prior",
)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return _softmax(np.log(clipped) / temperature)


def _balanced_row_weights(labels: np.ndarray, patients: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=np.float64)
    groups = 0
    for patient in np.unique(patients):
        for label in range(len(CLASS_NAMES)):
            mask = (patients == patient) & (labels == label)
            count = int(np.count_nonzero(mask))
            if count:
                weights[mask] = 1.0 / count
                groups += 1
    if groups == 0:
        raise ValueError("cannot balance an empty labeled sample")
    return weights * (len(weights) / np.sum(weights))


def _balanced_log_loss(
    probabilities: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    temperature: float,
) -> float:
    calibrated = _temperature_scale(probabilities, temperature)
    weights = _balanced_row_weights(labels, patients)
    losses = -np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-12, 1.0))
    return float(np.average(losses, weights=weights))


def _fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
) -> float:
    """Minimize balanced multiclass log loss with deterministic golden-section search."""
    left, right = np.log(0.25), np.log(4.0)
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1 = _balanced_log_loss(probabilities, labels, patients, float(np.exp(x1)))
    f2 = _balanced_log_loss(probabilities, labels, patients, float(np.exp(x2)))
    for _ in range(48):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - golden * (right - left)
            f1 = _balanced_log_loss(probabilities, labels, patients, float(np.exp(x1)))
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + golden * (right - left)
            f2 = _balanced_log_loss(probabilities, labels, patients, float(np.exp(x2)))
    return float(np.exp((left + right) / 2.0))


@dataclass(frozen=True, slots=True)
class PosteriorCalibration:
    """Frozen posterior transform and the priors implied by its reference fit."""

    strategy: CalibrationStrategy
    temperature: float
    class_priors: np.ndarray

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply the selected scalar posterior transform without changing class order."""
        values = np.asarray(probabilities, dtype=np.float64)
        if self.strategy == "temperature_oof_prior":
            return _temperature_scale(values, self.temperature)
        return values


@dataclass(frozen=True, slots=True)
class ScoreModel:
    """Frozen preprocessing, final classifier, and posterior calibration."""

    transform: RobustArcsinhTransform
    classifier: HistGradientBoostingClassifier
    calibration: PosteriorCalibration

    @property
    def temperature(self) -> float:
        """Return the selected scalar calibration temperature."""
        return self.calibration.temperature

    @property
    def class_priors(self) -> np.ndarray:
        """Return the priors consistent with the calibrated posteriors."""
        return self.calibration.class_priors

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict posteriors under the frozen reference calibration."""
        transformed = self.transform.apply(features)
        return self.calibration.apply(self.classifier.predict_proba(transformed))

    def likelihood_ratios(self, features: np.ndarray) -> np.ndarray:
        """Return component density ratios up to a common event-wise factor."""
        probabilities = self.predict_proba(features)
        return probabilities / self.class_priors[None, :]


@dataclass(frozen=True, slots=True)
class ScoreFit:
    """Return the score model, OOF posteriors, and calibration-selection evidence."""

    model: ScoreModel
    raw_out_of_fold_probabilities: np.ndarray
    out_of_fold_probabilities: np.ndarray
    calibration_selection: dict[str, object]


def _reference_folds(reference: FlowCytData) -> list[tuple[int, ...]]:
    patients = set(int(value) for value in np.unique(reference.patients))
    folds = [tuple(patient for patient in fold if patient in patients) for fold in REFERENCE_FOLDS]
    folds = [fold for fold in folds if fold]
    if len(folds) < 3:
        raise ValueError("nested score calibration requires at least three reference folds")
    return folds


def _fit_classifier(
    training_data: FlowCytData,
    transform: RobustArcsinhTransform,
    *,
    max_per_patient_class: int,
    max_iter: int,
    seed: int,
) -> HistGradientBoostingClassifier:
    training = deterministic_group_sample(
        training_data,
        max_per_patient_class=max_per_patient_class,
        seed=seed,
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=max_iter,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        early_stopping=False,
        random_state=seed,
    )
    classifier.fit(
        transform.apply(training.features),
        training.labels,
        sample_weight=_balanced_row_weights(training.labels, training.patients),
    )
    expected_classes = tuple(range(len(CLASS_NAMES)))
    if tuple(int(value) for value in classifier.classes_) != expected_classes:
        raise ValueError("every classifier fit must contain all declared classes")
    return classifier


def _out_of_fold_probabilities(
    reference: FlowCytData,
    folds: list[tuple[int, ...]],
    transform: RobustArcsinhTransform,
    *,
    max_per_patient_class: int,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    transformed = transform.apply(reference.features)
    probabilities = np.full((len(reference.labels), len(CLASS_NAMES)), np.nan)
    for fold_index, held_out_patients in enumerate(folds):
        train_mask = ~np.isin(reference.patients, held_out_patients)
        held_out_mask = np.isin(reference.patients, held_out_patients)
        classifier = _fit_classifier(
            reference.select(train_mask),
            transform,
            max_per_patient_class=max_per_patient_class,
            max_iter=max_iter,
            seed=seed + fold_index,
        )
        probabilities[held_out_mask] = classifier.predict_proba(transformed[held_out_mask])
    if not np.isfinite(probabilities).all():
        raise ValueError("every reference patient must receive an out-of-fold prediction")
    return probabilities


def _posterior_marginal(
    probabilities: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
) -> np.ndarray:
    uniform = np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
    weights = integration_weights(labels, patients, uniform)
    marginal = np.sum(weights[:, None] * probabilities, axis=0)
    if np.any(marginal <= 0) or not np.isfinite(marginal).all():
        raise ValueError("calibrated posterior marginals must be finite and positive")
    return marginal / np.sum(marginal)


def _fit_posterior_calibration(
    strategy: CalibrationStrategy,
    probabilities: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
) -> PosteriorCalibration:
    if strategy == "temperature_oof_prior":
        temperature = _fit_temperature(probabilities, labels, patients)
        calibrated = _temperature_scale(probabilities, temperature)
    else:
        temperature = 1.0
        calibrated = probabilities
    if strategy == "raw_declared_prior":
        priors = np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
    else:
        priors = _posterior_marginal(calibrated, labels, patients)
    return PosteriorCalibration(strategy, temperature, priors)


def _patient_fraction_errors(
    data: FlowCytData,
    probabilities: np.ndarray,
    priors: np.ndarray,
) -> np.ndarray:
    from .likelihood import fit_unbinned_mixture

    ratios = probabilities / priors[None, :]
    errors: list[np.ndarray] = []
    for patient in np.unique(data.patients):
        mask = data.patients == patient
        counts = np.bincount(data.labels[mask], minlength=len(CLASS_NAMES))
        truth = counts / np.sum(counts)
        errors.append(fit_unbinned_mixture(ratios[mask]).fractions - truth)
    return np.asarray(errors)


def _calibration_metrics_from_errors(errors: np.ndarray) -> dict[str, object]:
    per_class = np.sqrt(np.mean(errors**2, axis=0))
    return {
        "target_macro_rmse": float(np.mean(per_class[:-1])),
        "per_class_rmse": per_class.tolist(),
        "mean_bias": np.mean(errors, axis=0).tolist(),
    }


def _select_calibration_strategy(
    reference: FlowCytData,
    folds: list[tuple[int, ...]],
    *,
    max_per_patient_class: int,
    max_iter: int,
    seed: int,
) -> tuple[CalibrationStrategy, dict[str, object]]:
    errors_by_strategy: dict[CalibrationStrategy, list[np.ndarray]] = {
        strategy: [] for strategy in CALIBRATION_STRATEGIES
    }
    outer_rows: list[dict[str, object]] = []
    for outer_index, outer_patients in enumerate(folds):
        outer_mask = np.isin(reference.patients, outer_patients)
        development = reference.select(~outer_mask)
        held_out = reference.select(outer_mask)
        transform = RobustArcsinhTransform.fit(development.features)
        inner_folds = [fold for index, fold in enumerate(folds) if index != outer_index]
        inner_probabilities = _out_of_fold_probabilities(
            development,
            inner_folds,
            transform,
            max_per_patient_class=max_per_patient_class,
            max_iter=max_iter,
            seed=seed + 100 * (outer_index + 1),
        )
        outer_classifier = _fit_classifier(
            development,
            transform,
            max_per_patient_class=max_per_patient_class,
            max_iter=max_iter,
            seed=seed + 1_000 + outer_index,
        )
        outer_raw = outer_classifier.predict_proba(transform.apply(held_out.features))
        row: dict[str, object] = {"held_out_patients": list(outer_patients)}
        for strategy in CALIBRATION_STRATEGIES:
            calibration = _fit_posterior_calibration(
                strategy,
                inner_probabilities,
                development.labels,
                development.patients,
            )
            errors = _patient_fraction_errors(
                held_out,
                calibration.apply(outer_raw),
                calibration.class_priors,
            )
            errors_by_strategy[strategy].append(errors)
            row[strategy] = _calibration_metrics_from_errors(errors)
        outer_rows.append(row)

    candidate_metrics = {
        strategy: _calibration_metrics_from_errors(np.concatenate(errors_by_strategy[strategy]))
        for strategy in CALIBRATION_STRATEGIES
    }
    selected = CALIBRATION_STRATEGIES[0]
    best = float(candidate_metrics[selected]["target_macro_rmse"])
    for strategy in CALIBRATION_STRATEGIES[1:]:
        value = float(candidate_metrics[strategy]["target_macro_rmse"])
        if value < best - 1e-6:
            selected = strategy
            best = value
    return selected, {
        "candidate_order": list(CALIBRATION_STRATEGIES),
        "selected_strategy": selected,
        "tie_tolerance": 1e-6,
        "candidates": candidate_metrics,
        "outer_folds": outer_rows,
    }


def fit_score_model(
    reference: FlowCytData,
    *,
    max_per_patient_class: int = 2_000,
    max_iter: int = 120,
    seed: int = 2026,
) -> ScoreFit:
    """Fit a nested patient-calibrated density-ratio model on labeled references."""
    folds = _reference_folds(reference)
    selected_strategy, selection = _select_calibration_strategy(
        reference,
        folds,
        max_per_patient_class=max_per_patient_class,
        max_iter=max_iter,
        seed=seed,
    )
    transform = RobustArcsinhTransform.fit(reference.features)
    raw_out_of_fold = _out_of_fold_probabilities(
        reference,
        folds,
        transform,
        max_per_patient_class=max_per_patient_class,
        max_iter=max_iter,
        seed=seed + 10_000,
    )
    calibration = _fit_posterior_calibration(
        selected_strategy,
        raw_out_of_fold,
        reference.labels,
        reference.patients,
    )
    calibrated = calibration.apply(raw_out_of_fold)
    final_classifier = _fit_classifier(
        reference,
        transform,
        max_per_patient_class=max_per_patient_class,
        max_iter=max_iter,
        seed=seed + 20_000,
    )
    reference_weights = integration_weights(
        reference.labels,
        reference.patients,
        np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES)),
    )
    closure = sq.ratios.ratio_closure_report(
        sq.ratios.ratios_from_posteriors(calibrated, calibration.class_priors), reference_weights
    )
    selection["final_calibration"] = {
        "strategy": calibration.strategy,
        "temperature": calibration.temperature,
        "class_priors": calibration.class_priors.tolist(),
        "ratio_normalizers": np.asarray(closure.normalizers).tolist(),
        "maximum_normalization_residual": closure.max_residual,
    }
    return ScoreFit(
        model=ScoreModel(transform, final_classifier, calibration),
        raw_out_of_fold_probabilities=raw_out_of_fold,
        out_of_fold_probabilities=calibrated,
        calibration_selection=selection,
    )


def reference_composition(labels: np.ndarray, patients: np.ndarray) -> np.ndarray:
    """Compute the equal-patient mean six-class composition."""
    compositions: list[np.ndarray] = []
    for patient in np.unique(patients):
        counts = np.bincount(labels[patients == patient], minlength=len(CLASS_NAMES))
        compositions.append(counts / np.sum(counts))
    result = np.mean(compositions, axis=0)
    if np.any(result <= 0):
        raise ValueError("every class must occur in the reference composition")
    return result / np.sum(result)


def integration_weights(
    labels: np.ndarray,
    patients: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Weight rows so patients are equal within class and class mass is ``reference``."""
    weights = np.zeros(len(labels), dtype=np.float64)
    for label in range(len(CLASS_NAMES)):
        present = [
            patient
            for patient in np.unique(patients)
            if np.any((patients == patient) & (labels == label))
        ]
        if not present:
            raise ValueError(f"class {label} has no integration rows")
        for patient in present:
            mask = (patients == patient) & (labels == label)
            weights[mask] = reference[label] / (len(present) * np.count_nonzero(mask))
    return weights
