"""Concise test adapter around the task-explicit public quantizer API."""

from __future__ import annotations

from typing import Literal

from scorequant import (
    DOptimality,
    KMeansConfig,
    NormalizedTrace,
    ScoreSample,
    SoftVoronoiConfig,
    fit_quantizer,
)
from scorequant._typing import ArrayLike
from scorequant.results import QuantizerResult


def fit_test_quantizer(
    scores: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    n_bins: int,
    config: KMeansConfig | SoftVoronoiConfig | None = None,
    validation_scores: ArrayLike | None = None,
    validation_weights: ArrayLike | None = None,
    diagnostics: Literal["final", "endpoints", "full"] = "endpoints",
) -> QuantizerResult:
    """Fit a reusable score quantizer for tests focused below orchestration."""
    resolved = KMeansConfig() if config is None else config
    validation = (
        None if validation_scores is None else ScoreSample(validation_scores, validation_weights)
    )
    criterion = NormalizedTrace() if isinstance(resolved, KMeansConfig) else DOptimality()
    return fit_quantizer(
        ScoreSample(scores, weights),
        validation=validation,
        n_bins=n_bins,
        criterion=criterion,
        config=resolved,
        diagnostics=diagnostics,
    )
