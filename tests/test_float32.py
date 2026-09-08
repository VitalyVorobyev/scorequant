from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import scorequant
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
