from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import scorequant
from scorequant._chunking import chunk_rows_for
from scorequant._execution import use_execution
from scorequant.solvers.soft import soft_objective_and_center_gradient
from tests._fit import fit_test_quantizer


def _scores(n: int = 400) -> jnp.ndarray:
    first = jax.random.normal(jax.random.PRNGKey(1), (n // 2, 2)) * 0.25 - 1
    second = jax.random.normal(jax.random.PRNGKey(2), (n - n // 2, 2)) * 0.25 + 1
    return jnp.concatenate([first, second])


def test_kmeans_fit_predict_evaluate_and_monotonic_trace() -> None:
    scores = _scores()
    result = fit_test_quantizer(
        scores,
        n_bins=4,
        config=scorequant.KMeansConfig(seed=4, solver_restarts=3, max_iter=40),
    )
    assert result.predict_scores(scores).shape == (scores.shape[0],)
    assert result.evaluate_scores(scores).geometric_mean_retention > 0.7
    objective = np.asarray(result.trace.objective)
    assert np.all(np.diff(objective) <= 1e-5)
    assert result.train_report.psd_residual_min_eigenvalue >= -1e-4


def test_zero_weights_are_ignored_but_rows_remain_predictable() -> None:
    scores = jnp.asarray([[-2.0], [-1.0], [1.0], [2.0], [100.0]])
    weights = jnp.asarray([1.0, 1.0, 1.0, 1.0, 0.0])
    result = fit_test_quantizer(
        scores,
        weights=weights,
        n_bins=2,
        config=scorequant.KMeansConfig(solver_restarts=2),
    )
    assert result.predict_scores(scores).shape == (5,)
    assert int(np.asarray(result.train_report.bin_counts).sum()) == 4


def test_validation_is_diagnostic_only() -> None:
    scores = _scores()
    config = scorequant.KMeansConfig(seed=9, solver_restarts=2)
    without = fit_test_quantizer(scores, n_bins=3, config=config)
    with_validation = fit_test_quantizer(
        scores,
        n_bins=3,
        config=config,
        validation_scores=scores[::-1],
    )
    np.testing.assert_allclose(without.centers, with_validation.centers)
    assert with_validation.validation_report is not None
    assert with_validation.trace.validation_hard_retention is not None


def test_same_seed_reproduces_centers_and_trace() -> None:
    scores = _scores(240)
    config = scorequant.KMeansConfig(seed=17, solver_restarts=3)
    first = fit_test_quantizer(scores, n_bins=5, config=config)
    second = fit_test_quantizer(scores, n_bins=5, config=config)
    np.testing.assert_allclose(first.centers, second.centers)
    np.testing.assert_allclose(first.trace.objective, second.trace.objective)


def test_soft_voronoi_has_finite_trace_and_hard_result() -> None:
    scores = _scores(240)
    result = fit_test_quantizer(
        scores,
        n_bins=3,
        config=scorequant.SoftVoronoiConfig(
            seed=3,
            initializer_restarts=2,
            kmeans_max_iter=30,
            max_steps=30,
            record_every=5,
        ),
    )
    assert np.isfinite(np.asarray(result.trace.objective)).all()
    assert np.isfinite(np.asarray(result.trace.gradient_norms)).all()
    assert np.isfinite(np.asarray(result.trace.soft_retention)).all()
    assert result.train_report.geometric_mean_retention > 0.55


def test_soft_refuses_to_stall_on_a_singular_fisher_matrix() -> None:
    # An exactly centred sample makes the soft moments sum to zero, so two cells
    # span one direction at every temperature and the two-dimensional soft
    # Fisher matrix is singular for the whole schedule.
    half = np.asarray(_scores(60))
    scores = np.concatenate([half, -half])
    with pytest.raises(scorequant.RefusalError, match="stalled") as refusal:
        fit_test_quantizer(
            scores,
            n_bins=2,
            config=scorequant.SoftVoronoiConfig(seed=0, max_steps=5, record_every=5),
        )
    assert refusal.value.counterexample == "CE-DS-MARGINS-RANK-VACUITY-001"


@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize(
    "criterion", [scorequant.DOptimality(), scorequant.ProfiledDOptimality(interest=(0,))]
)
def test_chunked_soft_gradient_matches_one_chunk(
    backend: str, criterion: scorequant.DOptimality | scorequant.ProfiledDOptimality
) -> None:
    rng = np.random.default_rng(7)
    points = rng.normal(size=(300, 2))
    objective_scores = np.hstack([points, rng.normal(size=(300, 1))]) + 0.4
    weights = rng.uniform(0.2, 1.5, size=300)
    centers = rng.normal(size=(4, 2))
    execution = scorequant.ExecutionConfig(backend=backend, precision="float64", device="cpu")

    def evaluate(budget: int) -> tuple[float, np.ndarray]:
        with patch("scorequant._chunking.WORKING_SET_BYTES", budget), use_execution(execution):
            loss, gradient = soft_objective_and_center_gradient(
                points, objective_scores, weights, centers, 0.8, criterion
            )
        return float(loss), np.asarray(gradient)

    # About forty rows per chunk against a sample of three hundred.
    budget = 40 * 8 * (4 * 6 + 8)
    with patch("scorequant._chunking.WORKING_SET_BYTES", budget):
        assert chunk_rows_for(points, 4) == 40
    chunked_loss, chunked_gradient = evaluate(budget)
    loss, gradient = evaluate(1 << 40)
    assert np.isfinite(loss)
    assert chunked_loss == pytest.approx(loss, rel=1e-12)
    np.testing.assert_allclose(chunked_gradient, gradient, rtol=1e-10, atol=1e-12)


def test_soft_requires_enough_bins_for_rank() -> None:
    with pytest.raises(ValueError, match="n_bins >="):
        fit_test_quantizer(
            _scores(100),
            n_bins=1,
            config=scorequant.SoftVoronoiConfig(max_steps=2),
        )


def test_whitened_partition_is_parameter_reparameterization_invariant() -> None:
    scores = _scores(300)
    change = jnp.asarray([[2.0, 0.3], [-0.4, 1.3]])
    transformed_scores = scores @ jnp.linalg.inv(change)
    config = scorequant.KMeansConfig(seed=11, solver_restarts=4)
    original = np.asarray(
        fit_test_quantizer(scores, n_bins=4, config=config).predict_scores(scores)
    )
    transformed_result = fit_test_quantizer(transformed_scores, n_bins=4, config=config)
    changed = np.asarray(transformed_result.predict_scores(transformed_scores))
    np.testing.assert_array_equal(
        original[:, None] == original[None, :], changed[:, None] == changed[None, :]
    )


def test_too_many_distinct_bins_fails() -> None:
    with pytest.raises(ValueError, match="distinct"):
        fit_test_quantizer(jnp.asarray([[0.0], [0.0], [1.0]]), n_bins=3)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: scorequant.KMeansConfig(solver_restarts=0), "solver_restarts"),
        (lambda: scorequant.KMeansConfig(seed=-1), "seed"),
        (lambda: scorequant.KMeansConfig(whiten=1), "whiten"),
        (lambda: scorequant.KMeansConfig(rank_rtol=True), "rank_rtol"),
        (lambda: scorequant.KMeansConfig(rank_rtol=1.0), "rank_rtol"),
        (lambda: scorequant.SoftVoronoiConfig(learning_rate=np.nan), "learning_rate"),
        (lambda: scorequant.SoftVoronoiConfig(temperature_end_ratio=1.1), "temperature"),
    ],
)
def test_configs_fail_during_construction(
    factory: Callable[[], scorequant.KMeansConfig | scorequant.SoftVoronoiConfig],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_config_method_is_derived_and_serialized() -> None:
    config = scorequant.KMeansConfig()
    assert config.method == "kmeans"
    assert config.to_dict()["method"] == "kmeans"
    with pytest.raises(TypeError, match="method"):
        scorequant.KMeansConfig(method="kmeans")


def test_evaluate_reuses_fitted_rank_tolerance() -> None:
    scores = jnp.asarray(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -0.01],
            [0.0, 0.01],
        ]
    )
    result = fit_test_quantizer(
        scores,
        n_bins=2,
        config=scorequant.KMeansConfig(rank_rtol=1e-3, solver_restarts=2),
    )
    assert result.transform.rank == 1
    assert result.evaluate_scores(scores).effective_rank == 1
