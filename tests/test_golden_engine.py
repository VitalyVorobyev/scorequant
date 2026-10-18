"""Golden regression fixtures for the D and profiled-D exchange solvers.

These tests freeze the exact numerical behavior of ``optimize_partition`` and
``fit_quantizer`` (D-exchange and k-means paths) against seeded problems, so
that an upcoming refactor toward a unified exchange engine can be checked for
behavior-preserving equivalence. All frozen literals were produced under
``JAX_ENABLE_X64=1`` by running the current implementation once; they are not
independently re-derived here. Under float32 (x64 disabled) the frozen values
do not apply and the tests skip.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import scorequant as sq

from ._oracles import _exhaustive_d_oracle


def _require_x64() -> None:
    if not jax.config.jax_enable_x64:
        pytest.skip("golden fixtures were frozen under JAX_ENABLE_X64=1")


def _legacy_config() -> sq.DExchangeConfig:
    """Pin the single-move, single-restart, uncapped behavior these values froze.

    Every frozen case converged in far fewer than the historical 200-sweep cap,
    so an uncapped scan budget reproduces the same terminal labels bit for bit.
    """
    return sq.DExchangeConfig(batch_moves=False, solver_restarts=1, max_iter=None)


def test_golden_d_case_a_unit_weight_small() -> None:
    """N=60, P=2, n_bins=3, unit weights."""
    _require_x64()
    rng = np.random.default_rng(101)
    scores = rng.normal(size=(60, 2))
    result = sq.optimize_partition(scores, n_bins=3, config=_legacy_config())

    expected_labels = [
        2,
        0,
        1,
        0,
        0,
        1,
        1,
        2,
        1,
        2,
        1,
        0,
        1,
        0,
        0,
        2,
        1,
        1,
        0,
        0,
        1,
        0,
        2,
        2,
        0,
        0,
        0,
        2,
        1,
        2,
        1,
        0,
        2,
        2,
        2,
        0,
        1,
        2,
        2,
        2,
        0,
        0,
        2,
        1,
        0,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        0,
        2,
        2,
        1,
        1,
        2,
        2,
        1,
    ]
    assert np.asarray(result.labels).tolist() == expected_labels
    assert result.accepted_moves == 0
    assert result.exchange_stable is True
    assert result.objective == pytest.approx(-1.075507073539631, rel=1e-9)
    assert result.train_report.logdet_retention == pytest.approx(-1.0755070735396306, rel=1e-9)


def test_golden_d_case_b_weighted_medium() -> None:
    """N=200, P=3, n_bins=5, nonuniform positive weights."""
    _require_x64()
    rng = np.random.default_rng(102)
    scores = rng.normal(size=(200, 3))
    weights = rng.uniform(0.5, 2.0, size=200)
    result = sq.optimize_partition(scores, weights=weights, n_bins=5, config=_legacy_config())

    expected_labels = [
        2,
        0,
        1,
        1,
        3,
        1,
        1,
        4,
        1,
        3,
        2,
        1,
        1,
        0,
        0,
        0,
        3,
        3,
        3,
        0,
        2,
        4,
        1,
        4,
        0,
        4,
        0,
        4,
        3,
        2,
        0,
        4,
        3,
        1,
        0,
        0,
        1,
        4,
        1,
        3,
        1,
        2,
        1,
        1,
        4,
        1,
        2,
        0,
        0,
        0,
        0,
        3,
        2,
        0,
        0,
        2,
        4,
        1,
        3,
        4,
        4,
        4,
        0,
        4,
        1,
        1,
        1,
        2,
        2,
        3,
        0,
        3,
        1,
        1,
        1,
        0,
        3,
        4,
        4,
        3,
        2,
        3,
        3,
        4,
        2,
        4,
        2,
        2,
        0,
        4,
        0,
        3,
        0,
        0,
        0,
        0,
        4,
        3,
        4,
        3,
        3,
        4,
        0,
        4,
        4,
        4,
        3,
        0,
        1,
        1,
        1,
        1,
        3,
        0,
        0,
        3,
        3,
        2,
        0,
        3,
        2,
        0,
        0,
        4,
        4,
        2,
        1,
        1,
        1,
        1,
        3,
        1,
        1,
        4,
        0,
        3,
        4,
        1,
        1,
        1,
        1,
        1,
        2,
        1,
        1,
        2,
        2,
        4,
        1,
        2,
        4,
        3,
        4,
        1,
        4,
        3,
        2,
        1,
        4,
        4,
        4,
        3,
        1,
        0,
        2,
        2,
        3,
        4,
        4,
        0,
        0,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        3,
        0,
        3,
        4,
        4,
        4,
        0,
        4,
        0,
        4,
        3,
        4,
        4,
        4,
        4,
        0,
        1,
        4,
        1,
        4,
        0,
        0,
    ]
    assert np.asarray(result.labels).tolist() == expected_labels
    assert result.accepted_moves == 27
    assert result.exchange_stable is True
    assert result.objective == pytest.approx(-1.7516715044751872, rel=1e-9)
    assert result.train_report.logdet_retention == pytest.approx(-1.7516715044751832, rel=1e-9)


def test_golden_profiled_d_case_c_unit_weight() -> None:
    """N=80, P=3, n_bins=4, ProfiledDOptimality(interest=(0,)), unit weights."""
    _require_x64()
    rng = np.random.default_rng(103)
    scores = rng.normal(size=(80, 3))
    result = sq.optimize_partition(
        scores,
        n_bins=4,
        criterion=sq.ProfiledDOptimality((0,)),
        config=_legacy_config(),
    )

    expected_labels = [
        2,
        1,
        1,
        0,
        3,
        2,
        0,
        3,
        0,
        0,
        2,
        3,
        3,
        3,
        3,
        2,
        0,
        1,
        0,
        1,
        2,
        2,
        3,
        3,
        2,
        3,
        2,
        0,
        1,
        1,
        3,
        3,
        3,
        1,
        1,
        3,
        0,
        0,
        2,
        0,
        2,
        2,
        1,
        2,
        1,
        0,
        3,
        2,
        0,
        3,
        0,
        0,
        3,
        2,
        0,
        1,
        0,
        3,
        1,
        3,
        0,
        0,
        2,
        3,
        3,
        2,
        1,
        0,
        0,
        3,
        1,
        2,
        2,
        3,
        1,
        0,
        0,
        1,
        0,
        2,
    ]
    assert np.asarray(result.labels).tolist() == expected_labels
    assert result.accepted_moves == 36
    assert result.exchange_stable is True
    assert result.objective == pytest.approx(4.20461018260305, rel=1e-9)


def test_golden_profiled_d_case_d_weighted_larger() -> None:
    """N=150, P=4, n_bins=5, ProfiledDOptimality(interest=(0, 1)), nonuniform weights."""
    _require_x64()
    rng = np.random.default_rng(104)
    scores = rng.normal(size=(150, 4))
    weights = rng.uniform(0.5, 2.0, size=150)
    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=5,
        criterion=sq.ProfiledDOptimality((0, 1)),
        config=_legacy_config(),
    )

    expected_labels = [
        3,
        3,
        3,
        1,
        0,
        4,
        4,
        1,
        3,
        0,
        3,
        2,
        1,
        4,
        3,
        3,
        3,
        0,
        3,
        3,
        0,
        2,
        3,
        1,
        4,
        3,
        3,
        3,
        2,
        0,
        4,
        2,
        1,
        0,
        3,
        4,
        3,
        2,
        3,
        1,
        3,
        3,
        2,
        3,
        2,
        2,
        0,
        2,
        4,
        3,
        2,
        4,
        4,
        0,
        2,
        1,
        3,
        3,
        4,
        0,
        0,
        1,
        3,
        4,
        4,
        1,
        1,
        0,
        4,
        2,
        3,
        0,
        2,
        0,
        3,
        0,
        1,
        4,
        2,
        3,
        4,
        3,
        3,
        0,
        3,
        0,
        0,
        2,
        2,
        0,
        1,
        1,
        2,
        2,
        2,
        0,
        1,
        2,
        3,
        2,
        1,
        1,
        0,
        3,
        0,
        2,
        4,
        3,
        2,
        2,
        4,
        1,
        4,
        3,
        3,
        0,
        4,
        3,
        4,
        3,
        3,
        1,
        3,
        0,
        1,
        2,
        1,
        0,
        2,
        2,
        3,
        4,
        0,
        3,
        4,
        3,
        0,
        2,
        0,
        3,
        4,
        3,
        2,
        2,
        3,
        4,
        4,
        1,
        2,
        3,
    ]
    assert np.asarray(result.labels).tolist() == expected_labels
    assert result.accepted_moves == 60
    assert result.exchange_stable is True
    assert result.objective == pytest.approx(9.498238390095434, rel=1e-9)


def test_golden_fit_quantizer_kmeans_normalized_trace() -> None:
    """N=300, P=3, n_bins=4, KMeansConfig + NormalizedTrace, held-out prediction."""
    _require_x64()
    rng = np.random.default_rng(105)
    train_scores = rng.normal(size=(300, 3))
    result = sq.fit_quantizer(
        sq.ScoreSample(train_scores),
        n_bins=4,
        criterion=sq.NormalizedTrace(),
        config=sq.KMeansConfig(),
    )

    held_out_rng = np.random.default_rng(205)
    held_out = held_out_rng.normal(size=(120, 3))
    predicted = np.asarray(result.predict_scores(held_out))
    counts = np.bincount(predicted, minlength=4).tolist()

    assert counts == [34, 32, 26, 28]
    assert result.train_report.geometric_mean_retention == pytest.approx(
        0.49095221105798853, rel=1e-9
    )


def test_exhaustive_oracle_also_trace_can_disagree_with_d_optimal() -> None:
    """The trace-optimal and D-optimal partitions can genuinely differ."""
    _require_x64()
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(9, 2))

    d_labels, d_objective, trace_labels, trace_objective = _exhaustive_d_oracle(
        scores, None, 3, also_trace=True
    )

    assert not np.array_equal(np.asarray(d_labels), np.asarray(trace_labels))
    assert np.isfinite(d_objective)
    assert np.isfinite(trace_objective)

    # The oracle without also_trace must still agree with the D-optimal branch.
    only_d_labels, only_d_objective = _exhaustive_d_oracle(scores, None, 3)
    assert np.array_equal(np.asarray(only_d_labels), np.asarray(d_labels))
    assert only_d_objective == pytest.approx(d_objective, rel=1e-9)
