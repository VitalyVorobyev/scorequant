from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import scorequant
from scorequant.config import dtype_tolerance
from scorequant.transforms import fisher_transform
from tests._fit import fit_test_quantizer


def test_float32_fit_and_soft_step() -> None:
    scores = jax.random.normal(jax.random.PRNGKey(5), (256, 3), dtype=jnp.float32)
    result = fit_test_quantizer(
        scores,
        n_bins=5,
        config=scorequant.SoftVoronoiConfig(
            initializer_restarts=1,
            kmeans_max_iter=20,
            max_steps=5,
            record_every=5,
        ),
    )
    assert result.centers.dtype == jnp.float32
    assert np.isfinite(result.train_report.geometric_mean_retention)
    assert result.train_report.psd_residual_min_eigenvalue >= -1e-3


def test_float32_ratio_algebra_promotes_low_precision() -> None:
    posteriors = jnp.asarray(
        [[0.5, 0.25, 0.25], [0.125, 0.375, 0.5]],
        dtype=jnp.float16,
    )
    ratios = scorequant.ratios_from_posteriors(posteriors, [0.5, 0.25, 0.25])
    scores = scorequant.mixture_scores_from_ratios(ratios, [0.25, 0.25, 0.5])
    assert ratios.dtype == jnp.float32
    assert scores.dtype == jnp.float32
    assert np.isfinite(np.asarray(scores)).all()


def test_float32_retention_uncertainty_is_regular_on_both_backends() -> None:
    rng = np.random.default_rng(23)
    scores = (rng.normal(size=(200, 2)) + 0.3).astype(np.float32)
    labels = (scores[:, 0] > 0).astype(int) + 2 * (scores[:, 1] > 0.2).astype(int)
    reports = {
        backend: scorequant.retention_uncertainty(
            scores,
            labels,
            execution=scorequant.ExecutionConfig(
                backend=backend, precision="float32", device="cpu"
            ),
        )
        for backend in ("jax", "numpy")
    }
    for report in reports.values():
        assert report.status == "ok"
        assert np.isfinite(report.standard_error) and report.standard_error > 0
        lower, upper = report.confidence_interval
        assert lower < report.estimate < upper
    assert reports["jax"].estimate == pytest.approx(reports["numpy"].estimate, rel=1e-4)
    assert reports["jax"].standard_error == pytest.approx(reports["numpy"].standard_error, rel=1e-3)


def test_float32_backend_parity_of_continuous_quantities() -> None:
    """The M9 gate's float32 leg, stated at the level where it actually holds.

    The continuous pipeline -- Fisher information and the retained-information
    report on a fixed labeling -- agrees across backends to float32 round-off.
    The discrete solvers deliberately are not compared here: in float32 a
    relocation gain can sit inside the noise floor, so the two backends can walk
    to different, individually exchange-stable optima. That is a property of
    hard assignment in low precision, not a backend disagreement, so the gate
    asserts per-backend validity for solvers and cross-backend agreement only
    for the continuous quantities.
    """
    rng = np.random.default_rng(19)
    scores = rng.normal(size=(120, 2)).astype(np.float32)
    weights = rng.uniform(0.3, 1.5, scores.shape[0]).astype(np.float32)
    labels = np.tile(np.arange(3), 40)

    def execution(backend: str) -> scorequant.ExecutionConfig:
        return scorequant.ExecutionConfig(backend=backend, precision="float32", device="cpu")

    fishers = {
        backend: np.asarray(
            scorequant.fisher_information(scores, weights, execution=execution(backend))
        )
        for backend in ("jax", "numpy")
    }
    np.testing.assert_allclose(fishers["jax"], fishers["numpy"], rtol=1e-5, atol=1e-6)

    reports = {
        backend: scorequant.information_report(
            scores, labels, weights, n_bins=3, execution=execution(backend)
        )
        for backend in ("jax", "numpy")
    }
    np.testing.assert_allclose(
        np.asarray(reports["jax"].retained_matrix),
        np.asarray(reports["numpy"].retained_matrix),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reports["jax"].geometric_mean_retention,
        reports["numpy"].geometric_mean_retention,
        rtol=1e-5,
        atol=1e-6,
    )


def test_float32_solvers_are_individually_valid_on_each_backend() -> None:
    rng = np.random.default_rng(19)
    scores = rng.normal(size=(120, 2)).astype(np.float32)
    weights = rng.uniform(0.3, 1.5, scores.shape[0]).astype(np.float32)
    for backend in ("jax", "numpy"):
        result = scorequant.optimize_partition(
            scores,
            weights=weights,
            n_bins=3,
            config=scorequant.DExchangeConfig(initializer_restarts=2),
            execution=scorequant.ExecutionConfig(
                backend=backend, precision="float32", device="cpu"
            ),
        )
        assert result.exchange_stable
        assert np.unique(result.labels).size == 3
        assert np.isfinite(result.objective)


def test_default_tolerances_follow_the_working_dtype() -> None:
    """Float64 keeps its constants bit for bit; float32 derives them from its epsilon."""
    assert dtype_tolerance(1e-10, np.float64) == 1e-10
    share = np.log(1e-10) / np.log(np.finfo(np.float64).eps)
    float32_default = float(np.finfo(np.float32).eps) ** share
    assert dtype_tolerance(1e-10, np.float32) == pytest.approx(float32_default, rel=1e-12)
    assert 100 * np.finfo(np.float32).eps < float32_default < 1e-3

    fisher = np.diag([2.0, 1.0])
    numpy64 = scorequant.ExecutionConfig(backend="numpy", precision="float64")
    assert fisher_transform(fisher, execution=numpy64).rank_rtol == 1e-10
    transform32 = fisher_transform(fisher.astype(np.float32))
    assert transform32.rank_rtol == pytest.approx(float32_default, rel=1e-12)


def test_float32_certificates_hold_at_the_float32_default() -> None:
    rng = np.random.default_rng(29)
    scores = (rng.normal(size=(300, 2)) + 0.4).astype(np.float32)
    float32_default = dtype_tolerance(1e-10, np.float32)
    result = scorequant.optimize_partition(
        scores, n_bins=4, config=scorequant.DExchangeConfig(initializer_restarts=2)
    )
    assert result.config.gain_tolerance == pytest.approx(float32_default, rel=1e-12)
    assert result.exchange_stable
    assert result.geometry is not None
    assert result.geometry.gain_tolerance == result.config.gain_tolerance
    assert result.geometry.voronoi_consistent
    assert result.geometry.separation_certified
    rule = result.compile_quantizer()
    assert rule.predict_scores(scores).shape == (300,)

    report = scorequant.exchange_stability_report(scores, result.labels)
    assert report.gain_tolerance == result.config.gain_tolerance
    assert report.stable


def test_float32_explicit_gain_tolerance_is_used_unchanged() -> None:
    rng = np.random.default_rng(29)
    scores = (rng.normal(size=(120, 2)) + 0.4).astype(np.float32)
    config = scorequant.DExchangeConfig(initializer_restarts=1, gain_tolerance=1e-10)
    result = scorequant.optimize_partition(scores, n_bins=3, config=config)
    assert result.config.gain_tolerance == 1e-10
    report = scorequant.exchange_stability_report(scores, result.labels, gain_tolerance=1e-10)
    assert report.gain_tolerance == 1e-10
