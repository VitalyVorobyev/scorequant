"""Classifier-to-score bridge for the FAIR Universe HiggsML showcase.

Two classifiers, two different ratio doors:

* a signal-vs-background classifier feeds `scorequant.DensityRatioScore` under
  `scorequant.IntensityParameterization`, giving the two rate columns
  (`mu_htautau`, `nu_background`) in closed form once the ratio is known;
* a `tes`-minus-vs-plus classifier feeds `scorequant.CentralLogRatioScore`,
  the library's central-difference door, giving the one nuisance-shape
  column (`tes`) that has no closed form.

Both classifiers are cross-fitted out-of-fold with **one fold id per event**,
reused by every `tes` copy of that event: a plain per-row split lets the
`tes` classifier memorize an event from its eighteen `tes`-inert columns and
invert the label. Both are trained under the Monte Carlo event weights, so
the ratios they estimate are ratios of the *physical* weighted mixture. The
signal/background classifier additionally normalizes each class to mass
one half, because the raw weights make the signal class statistically
invisible (weighted fraction ~0.001); the physical rate then enters through
`IntensityParameterization` coefficients, never through the priors. The
`tes` classifier needs no such balancing: both copies of an event carry the
same weight, so the two classes are balanced by construction.

The fold models are kept, so the same out-of-fold rule can score an event's
`tes`-shifted copies. That is what turns a reusable bin rule into yield
templates for the downstream signal-strength fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import scorequant as sq

from .data import HepData

#: The three score columns every labeling in this example is reported
#: against, in the order the composed provider emits them.
SCHEMA = sq.ScoreSchema(("mu_htautau", "nu_background", "tes"))
#: `mu_htautau` is the sole parameter of interest for every profiled-D_s call.
INTEREST = SCHEMA.select("mu_htautau")

#: A calibrated posterior within this of 0.5 counts as "not separated" for
#: the `tes` near-boundary diagnostic.
NEAR_HALF_TOLERANCE = 0.01


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return _softmax(np.log(clipped) / temperature)


def _weighted_log_loss(
    probabilities: np.ndarray, labels: np.ndarray, weights: np.ndarray, temperature: float
) -> float:
    calibrated = _temperature_scale(probabilities, temperature)
    losses = -np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-12, 1.0))
    return float(np.average(losses, weights=weights))


def _fit_temperature(probabilities: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    """Minimize weighted binary log loss with a deterministic golden-section search.

    Mirrors the calibration bridge in `examples/cell_population/scores.py`,
    simplified to a flat weight vector: this example has no patient grouping
    to balance across.
    """
    left, right = np.log(0.25), np.log(4.0)
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1 = _weighted_log_loss(probabilities, labels, weights, float(np.exp(x1)))
    f2 = _weighted_log_loss(probabilities, labels, weights, float(np.exp(x2)))
    for _ in range(48):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - golden * (right - left)
            f1 = _weighted_log_loss(probabilities, labels, weights, float(np.exp(x1)))
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + golden * (right - left)
            f2 = _weighted_log_loss(probabilities, labels, weights, float(np.exp(x2)))
    return float(np.exp((left + right) / 2.0))


def _balanced_class_weights(labels: np.ndarray, raw_weights: np.ndarray) -> np.ndarray:
    """Normalize raw weights so each of the two classes carries mass 0.5.

    Without this, the raw Monte Carlo weights make the signal class
    statistically invisible to the classifier: background events carry weight
    ~1582, signal ~3, so the weighted signal fraction is ~0.001.
    """
    weights = np.zeros_like(raw_weights)
    for label in (0, 1):
        mask = labels == label
        total = float(raw_weights[mask].sum())
        if total <= 0:
            raise ValueError(f"class {label} has no positive weight")
        weights[mask] = raw_weights[mask] * (0.5 / total)
    return weights


def event_folds(is_signal: np.ndarray, *, n_folds: int, seed: int) -> np.ndarray:
    """Assign one deterministic fold id per event, stratified by signal label.

    Both classifiers reuse this same per-event assignment: the `tes` minus
    and plus copies of one event always share its fold id, so a fold boundary
    never separates an event from its own paired variant.

    Parameters
    ----------
    is_signal
        Boolean signal label per event, shape ``[N]``.
    n_folds
        Number of stratified folds.
    seed
        Deterministic shuffle seed.

    Returns
    -------
    numpy.ndarray
        Integer fold id per event, shape ``[N]``, values in ``[0, n_folds)``.
    """
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_ids = np.full(is_signal.shape[0], -1, dtype=np.int64)
    dummy_features = np.zeros((is_signal.shape[0], 1))
    for fold_index, (_, held_out) in enumerate(splitter.split(dummy_features, is_signal)):
        fold_ids[held_out] = fold_index
    if np.any(fold_ids < 0):
        raise ValueError("every event must receive a fold id")
    return fold_ids


def _fit_signal_background_classifier(
    features: np.ndarray,
    is_signal: np.ndarray,
    raw_weights: np.ndarray,
    *,
    max_iter: int,
    seed: int,
) -> HistGradientBoostingClassifier:
    labels = is_signal.astype(np.int64)
    weights = _balanced_class_weights(labels, raw_weights)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=max_iter,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        early_stopping=False,
        random_state=seed,
    )
    classifier.fit(features, labels, sample_weight=weights)
    if tuple(int(value) for value in classifier.classes_) != (0, 1):
        raise ValueError("signal/background classifier must see both classes")
    return classifier


def _fit_tes_classifier(
    minus_features: np.ndarray,
    plus_features: np.ndarray,
    raw_weights: np.ndarray,
    *,
    max_iter: int,
    seed: int,
) -> HistGradientBoostingClassifier:
    features = np.concatenate([minus_features, plus_features], axis=0)
    labels = np.concatenate(
        [np.zeros(len(minus_features), dtype=np.int64), np.ones(len(plus_features), dtype=np.int64)]
    )
    # The Monte Carlo weight of an event applies to both of its copies, so the
    # two classes stay balanced 1:1 and no class correction is needed. The
    # weights are still essential: the `tes` score column is the derivative
    # of the *physical* mixture's log density, and the unweighted event list
    # is one-third signal where the weighted mixture is one-thousandth.
    weights = np.concatenate([raw_weights, raw_weights]) / np.mean(raw_weights)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=max_iter,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        early_stopping=False,
        random_state=seed,
    )
    classifier.fit(features, labels, sample_weight=weights)
    if tuple(int(value) for value in classifier.classes_) != (0, 1):
        raise ValueError("tes classifier must see both the minus and plus classes")
    return classifier


def _predict_out_of_fold(
    models: Sequence[HistGradientBoostingClassifier],
    fold_ids: np.ndarray,
    features: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Score every row with the fold model that never saw its event."""
    raw = np.full((features.shape[0], 2), np.nan)
    for fold, model in enumerate(models):
        held_mask = fold_ids == fold
        if np.any(held_mask):
            raw[held_mask] = model.predict_proba(features[held_mask])
    if not np.isfinite(raw).all():
        raise ValueError("every row must be scored by exactly one fold model")
    return _temperature_scale(raw, temperature)


@dataclass(frozen=True, slots=True)
class SignalBackgroundOOF:
    """Out-of-fold calibrated signal/background posteriors and diagnostics.

    Attributes
    ----------
    probabilities
        Calibrated out-of-fold posteriors at the nominal features, shape
        ``[N, 2]``, columns ``[background, signal]`` (classifier class order
        0, 1).
    temperature
        Fitted temperature-scaling scalar.
    signal_fraction
        Weighted signal fraction using the raw Monte Carlo weights -- the
        physical rate that feeds `IntensityParameterization` coefficients.
    weighted_auc
        Out-of-fold AUC of the calibrated signal posterior, weighted by the
        raw Monte Carlo event weights.
    fold_ids, models
        The per-event fold assignment and the fold models, kept so that
        `predict` can score an event's shifted copies out of fold.
    """

    probabilities: np.ndarray
    temperature: float
    signal_fraction: float
    weighted_auc: float
    fold_ids: np.ndarray = field(repr=False)
    models: tuple[HistGradientBoostingClassifier, ...] = field(repr=False)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Calibrated out-of-fold posteriors for row-aligned features."""
        return _predict_out_of_fold(self.models, self.fold_ids, features, self.temperature)


def fit_signal_background_oof(
    data: HepData, *, fold_ids: np.ndarray, max_iter: int, seed: int
) -> SignalBackgroundOOF:
    """Cross-fit the signal/background classifier out-of-fold on nominal features.

    Parameters
    ----------
    data
        The loaded fixture.
    fold_ids
        Per-event fold assignment from `event_folds`.
    max_iter
        Boosting round budget per fold's classifier.
    seed
        Base seed; fold ``k`` uses ``seed + k``.

    Returns
    -------
    SignalBackgroundOOF
        Calibrated out-of-fold posteriors and related diagnostics.
    """
    features = data.features_at(1.0)
    labels = data.is_signal.astype(np.int64)
    raw_oof = np.full((data.n_events, 2), np.nan)
    models: list[HistGradientBoostingClassifier] = []
    for fold in range(int(np.max(fold_ids)) + 1):
        train_mask = fold_ids != fold
        held_mask = fold_ids == fold
        classifier = _fit_signal_background_classifier(
            features[train_mask],
            data.is_signal[train_mask],
            data.weights[train_mask],
            max_iter=max_iter,
            seed=seed + int(fold),
        )
        raw_oof[held_mask] = classifier.predict_proba(features[held_mask])
        models.append(classifier)
    if not np.isfinite(raw_oof).all():
        raise ValueError("every event must receive an out-of-fold signal/background prediction")
    temperature = _fit_temperature(raw_oof, labels, _balanced_class_weights(labels, data.weights))
    calibrated = _temperature_scale(raw_oof, temperature)
    signal_fraction = float(np.sum(data.weights[data.is_signal]) / np.sum(data.weights))
    weighted_auc = float(roc_auc_score(labels, calibrated[:, 1], sample_weight=data.weights))
    return SignalBackgroundOOF(
        calibrated, temperature, signal_fraction, weighted_auc, fold_ids, tuple(models)
    )


@dataclass(frozen=True, slots=True)
class TesOOF:
    """Out-of-fold calibrated `tes` posteriors, evaluated at nominal features.

    Attributes
    ----------
    delta
        The finite-difference half-offset used to build the minus/plus pair.
    probabilities
        Calibrated posteriors evaluated at each event's *nominal* (`tes=1`)
        features, shape ``[N, 2]``, columns ``[minus, plus]``. There is no
        ground-truth label at the nominal point; the calibration temperature
        is fit on the genuine minus/plus classification task and then
        applied here.
    temperature
        Fitted temperature-scaling scalar.
    minus_plus_auc
        Out-of-fold AUC of the minus/plus classification task itself,
        weighted by the Monte Carlo event weights (the pairing is balanced
        1:1 by construction, so no class correction enters).
    near_half_fraction
        Fraction of events whose nominal-point calibrated posterior falls
        within `NEAR_HALF_TOLERANCE` of 0.5 -- a noise diagnostic.
    fold_ids, models
        The per-event fold assignment and the fold models, kept so that
        `predict` can score an event's shifted copies out of fold.
    """

    delta: float
    probabilities: np.ndarray
    temperature: float
    minus_plus_auc: float
    near_half_fraction: float
    fold_ids: np.ndarray = field(repr=False)
    models: tuple[HistGradientBoostingClassifier, ...] = field(repr=False)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Calibrated out-of-fold minus/plus posteriors for row-aligned features."""
        return _predict_out_of_fold(self.models, self.fold_ids, features, self.temperature)


def fit_tes_oof(
    data: HepData, *, delta: float, fold_ids: np.ndarray, max_iter: int, seed: int
) -> TesOOF:
    """Cross-fit the `tes` minus/plus classifier out-of-fold, grouped by event.

    Parameters
    ----------
    data
        The loaded fixture.
    delta
        Finite-difference half-offset; ``1 - delta`` and ``1 + delta`` must
        both be committed `tes` points.
    fold_ids
        Per-event fold assignment from `event_folds`, reused for both the
        minus and plus copy of every event.
    max_iter
        Boosting round budget per fold's classifier.
    seed
        Base seed; fold ``k`` uses ``seed + k``.

    Returns
    -------
    TesOOF
        Calibrated nominal-point posteriors and related diagnostics.
    """
    nominal = data.features_at(1.0)
    minus = data.features_at(round(1.0 - delta, 4))
    plus = data.features_at(round(1.0 + delta, 4))
    raw_oof = np.full((data.n_events, 2), np.nan)
    pooled_probabilities: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []
    pooled_weights: list[np.ndarray] = []
    models: list[HistGradientBoostingClassifier] = []
    for fold in range(int(np.max(fold_ids)) + 1):
        train_mask = fold_ids != fold
        held_mask = fold_ids == fold
        classifier = _fit_tes_classifier(
            minus[train_mask],
            plus[train_mask],
            data.weights[train_mask],
            max_iter=max_iter,
            seed=seed + int(fold),
        )
        raw_oof[held_mask] = classifier.predict_proba(nominal[held_mask])
        held_count = int(np.count_nonzero(held_mask))
        pooled_probabilities.append(classifier.predict_proba(minus[held_mask]))
        pooled_probabilities.append(classifier.predict_proba(plus[held_mask]))
        pooled_labels.append(np.zeros(held_count, dtype=np.int64))
        pooled_labels.append(np.ones(held_count, dtype=np.int64))
        pooled_weights.append(data.weights[held_mask])
        pooled_weights.append(data.weights[held_mask])
        models.append(classifier)
    if not np.isfinite(raw_oof).all():
        raise ValueError("every event must receive an out-of-fold tes prediction")
    task_probabilities = np.concatenate(pooled_probabilities, axis=0)
    task_labels = np.concatenate(pooled_labels, axis=0)
    task_weights = np.concatenate(pooled_weights, axis=0)
    temperature = _fit_temperature(task_probabilities, task_labels, task_weights)
    calibrated_nominal = _temperature_scale(raw_oof, temperature)
    minus_plus_auc = float(
        roc_auc_score(task_labels, task_probabilities[:, 1], sample_weight=task_weights)
    )
    near_half_fraction = float(
        np.mean(np.abs(calibrated_nominal[:, 1] - 0.5) < NEAR_HALF_TOLERANCE)
    )
    return TesOOF(
        delta,
        calibrated_nominal,
        temperature,
        minus_plus_auc,
        near_half_fraction,
        fold_ids,
        tuple(models),
    )


def _score_columns(
    rate_probabilities: np.ndarray,
    tes_probabilities: np.ndarray,
    *,
    signal_fraction: float,
    delta: float,
) -> np.ndarray:
    """Turn the two calibrated posterior tables into the three score columns."""
    rate_provider = sq.DensityRatioScore.from_classifier(
        lambda observations: rate_probabilities,
        [0.5, 0.5],
        sq.IntensityParameterization([1.0 - signal_fraction, signal_fraction]),
        calibration="temperature",
        description="signal-vs-background classifier, out-of-fold",
    )
    tes_provider = sq.CentralLogRatioScore(
        lambda observations: tes_probabilities,
        deltas=[delta],
        class_priors=[0.5, 0.5],
        description="tes minus/plus classifier, out-of-fold",
    )
    placeholder = np.zeros((rate_probabilities.shape[0], 1))
    rate_scores = np.asarray(rate_provider.score(placeholder))  # columns [background, signal]
    tes_scores = np.asarray(tes_provider.score(placeholder))  # one column
    return np.concatenate([rate_scores[:, [1, 0]], tes_scores], axis=1)


def out_of_fold_scores(
    data: HepData, sigbg: SignalBackgroundOOF, tes: TesOOF, *, tes_point: float = 1.0
) -> np.ndarray:
    """Score every event's copy at one `tes` point with the out-of-fold models.

    The parameterization (the physical signal fraction) and both calibration
    temperatures are the full-sample values frozen inside `sigbg` and `tes`,
    so every table this returns -- nominal or shifted, any subset of rows --
    lives in one parameterization and is comparable with every other.

    Parameters
    ----------
    data
        The loaded fixture.
    sigbg, tes
        The cross-fitted classifiers.
    tes_point
        Which committed feature copy to score; ``1.0`` is the nominal table
        `assemble_score_sample` wraps.

    Returns
    -------
    numpy.ndarray
        Score table with shape ``[N, 3]`` in `SCHEMA` order.
    """
    features = data.features_at(tes_point)
    return _score_columns(
        sigbg.predict(features),
        tes.predict(features),
        signal_fraction=sigbg.signal_fraction,
        delta=tes.delta,
    )


def assemble_score_sample(data: HepData, sigbg: SignalBackgroundOOF, tes: TesOOF) -> sq.ScoreSample:
    """Combine out-of-fold posteriors into the three-column `ScoreSample`.

    This is the leakage-free score table the study runs on: every column is
    a fold-cross-fitted prediction, never a model evaluated on the event it
    was trained on.

    Parameters
    ----------
    data
        The loaded fixture.
    sigbg
        Out-of-fold signal/background posteriors from `fit_signal_background_oof`.
    tes
        Out-of-fold `tes` posteriors from `fit_tes_oof`, at the matching `delta`.

    Returns
    -------
    scorequant.ScoreSample
        Weighted score table with `SCHEMA` and `kind="estimated_ratio"`
        provenance.
    """
    scores = _score_columns(
        sigbg.probabilities,
        tes.probabilities,
        signal_fraction=sigbg.signal_fraction,
        delta=tes.delta,
    )
    provenance = sq.ScoreProvenance(
        kind="estimated_ratio",
        description="Out-of-fold classifier scores, FAIR Universe HiggsML fixture",
        metadata={
            "delta": tes.delta,
            "signal_fraction": sigbg.signal_fraction,
            "signal_weighted_auc": sigbg.weighted_auc,
            "tes_minus_plus_auc": tes.minus_plus_auc,
        },
    )
    return sq.ScoreSample(scores, data.weights, schema=SCHEMA, provenance=provenance)


def tes_score_reliability(
    data: HepData, *, delta: float, max_iter: int, seed: int
) -> dict[str, float]:
    """Measure how much of the `tes` score column is reproducible.

    Two `tes` classifiers are trained on disjoint thirds of the events and
    both evaluated on the remaining third. Their central-difference scores
    would agree perfectly if the column were a property of the events; the
    part that does not agree is per-event estimation noise. That noise
    inflates the unbinned nuisance information and therefore biases every
    reported profiled retention *down*, so the correlation is published as
    the diagnostic that bounds how much of the reported loss is the proxy's
    rather than the binning's.

    Parameters
    ----------
    data
        The loaded fixture.
    delta
        Finite-difference half-offset.
    max_iter
        Boosting round budget for each classifier.
    seed
        Deterministic seed for the three-way split and the two fits.

    Returns
    -------
    dict of float
        ``weighted_correlation`` of the two score columns on the common
        held-out third, the two classifiers' weighted minus/plus AUCs on
        that third, and the number of held-out events.
    """
    thirds = event_folds(data.is_signal, n_folds=3, seed=seed)
    minus = data.features_at(round(1.0 - delta, 4))
    plus = data.features_at(round(1.0 + delta, 4))
    nominal = data.features_at(1.0)
    held = thirds == 2
    held_count = int(np.count_nonzero(held))
    columns: list[np.ndarray] = []
    aucs: list[float] = []
    for fold, offset in ((0, 1), (1, 2)):
        train = thirds == fold
        model = _fit_tes_classifier(
            minus[train], plus[train], data.weights[train], max_iter=max_iter, seed=seed + offset
        )
        task_probabilities = np.concatenate(
            [model.predict_proba(minus[held]), model.predict_proba(plus[held])], axis=0
        )
        task_labels = np.concatenate(
            [np.zeros(held_count, dtype=np.int64), np.ones(held_count, dtype=np.int64)]
        )
        task_weights = np.concatenate([data.weights[held], data.weights[held]])
        aucs.append(
            float(roc_auc_score(task_labels, task_probabilities[:, 1], sample_weight=task_weights))
        )
        posterior = np.clip(model.predict_proba(nominal[held]), 1e-12, 1.0)
        columns.append(np.log(posterior[:, 1] / posterior[:, 0]) / (2.0 * delta))
    weights = data.weights[held]
    centered = [column - np.average(column, weights=weights) for column in columns]
    covariance = float(np.average(centered[0] * centered[1], weights=weights))
    variances = [float(np.average(column**2, weights=weights)) for column in centered]
    correlation = covariance / float(np.sqrt(variances[0] * variances[1]))
    return {
        "weighted_correlation": correlation,
        "first_minus_plus_auc": aucs[0],
        "second_minus_plus_auc": aucs[1],
        "held_out_events": float(held_count),
    }
