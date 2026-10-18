from __future__ import annotations

import numpy as np
import pytest

import scorequant
from examples.run import run_experiment
from examples.synthetic_problems import (
    SyntheticProblem,
    gaussian_location,
    spatial_sources,
    spectral_templates,
)
from tests._fit import fit_test_quantizer


@pytest.mark.parametrize(
    "problem",
    [
        gaussian_location(sizes=(1_000, 800, 4_000)),
        spectral_templates(sizes=(1_500, 1_000, 5_000)),
        spatial_sources(sizes=(2_000, 1_200, 6_000)),
    ],
    ids=lambda problem: problem.name,
)
def test_end_to_end_synthetic_evidence(problem: SyntheticProblem) -> None:
    experiment = run_experiment(problem, soft_steps=80, n_random=10)
    metrics = experiment.metrics
    assert metrics["soft_test_retention"] >= metrics["kmeans_test_retention"] - 0.01
    assert abs(metrics["soft_test_retention"] - metrics["soft_validation_retention"]) <= 0.03
    assert len(metrics["random_test_retentions"]) == 10


def test_score_space_beats_observation_space_on_nonlinear_problem() -> None:
    experiment = run_experiment(
        spectral_templates(sizes=(2_000, 1_000, 6_000)), soft_steps=80, n_random=10
    )
    assert (
        experiment.metrics["kmeans_test_retention"]
        >= experiment.metrics["observation_kmeans_test_retention"] + 0.05
    )


def test_soft_d_optimality_improves_a_targeted_nonlinear_fixture() -> None:
    """Exercise a low-bin regime where trace and determinant objectives differ."""

    amplitude = 0.6169019184314102
    phase = 5.032003641496173

    def generate(seed: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        coordinate = rng.uniform(-3, 3, size)
        scores = np.column_stack(
            [
                coordinate,
                amplitude * np.sin(3 * coordinate + phase) + rng.normal(0, 0.12, size),
            ]
        )
        return scores, np.ones(size)

    train_scores, train_weights = generate(1003, 5_000)
    test_scores, test_weights = generate(2003, 20_000)
    kmeans = fit_test_quantizer(
        train_scores,
        weights=train_weights,
        n_bins=2,
        config=scorequant.KMeansConfig(seed=0, solver_restarts=8),
    )
    soft = fit_test_quantizer(
        train_scores,
        weights=train_weights,
        n_bins=2,
        config=scorequant.SoftVoronoiConfig(
            seed=0, initializer_restarts=8, max_steps=400, record_every=100
        ),
    )
    improvement = (
        soft.evaluate_scores(test_scores, test_weights).geometric_mean_retention
        - kmeans.evaluate_scores(test_scores, test_weights).geometric_mean_retention
    )
    assert improvement >= 0.02
