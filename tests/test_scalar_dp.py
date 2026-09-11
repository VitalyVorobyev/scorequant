"""Exactness and equivalence gates for the scalar interval dynamic program."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import scorequant as sq
from scorequant.solvers import scalar as scalar_dp
from scorequant.solvers.scalar import scalar_interval_dp
from tests._fit import fit_test_quantizer

from ._oracles import _exhaustive_d_oracle


def _objective_tolerance() -> float:
    """Return the dtype-appropriate slack for an exact log-determinant comparison."""
    return 1e-12 if jax.config.jax_enable_x64 else 1e-5


def _reference_interval_dp(
    values: np.ndarray, weights: np.ndarray, n_bins: int
) -> tuple[np.ndarray, float]:
    """Port of the original per-stop Python interval program.

    This is deliberately the pre-vectorization implementation, kept here as an
    independent reference so the stripe-vectorized solver can be certified to
    reproduce its labels and its objective bit for bit rather than merely to
    agree within a tolerance.
    """
    n_rows = int(values.shape[0])
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    prefix_weight = np.r_[0.0, np.cumsum(ordered_weights)]
    prefix_sum = np.r_[0.0, np.cumsum(ordered_weights * ordered_values)]
    prefix_square = np.r_[0.0, np.cumsum(ordered_weights * ordered_values**2)]
    dynamic = np.full((n_bins + 1, n_rows + 1), np.inf)
    predecessor = np.full((n_bins + 1, n_rows + 1), -1, dtype=np.int32)
    dynamic[0, 0] = 0.0
    for bin_count in range(1, n_bins + 1):
        for stop in range(bin_count, n_rows + 1):
            starts = np.arange(bin_count - 1, stop)
            segment_weight = prefix_weight[stop] - prefix_weight[starts]
            segment_sum = prefix_sum[stop] - prefix_sum[starts]
            segment_square = prefix_square[stop] - prefix_square[starts]
            costs = segment_square - segment_sum**2 / segment_weight
            candidates = dynamic[bin_count - 1, starts] + costs
            selected = int(np.argmin(candidates))
            dynamic[bin_count, stop] = candidates[selected]
            predecessor[bin_count, stop] = starts[selected]
    ordered_labels = np.empty(n_rows, dtype=np.int32)
    stop = n_rows
    for label in range(n_bins - 1, -1, -1):
        start = int(predecessor[label + 1, stop])
        ordered_labels[start:stop] = label
        stop = start
    labels = np.empty(n_rows, dtype=np.int32)
    labels[order] = ordered_labels
    return labels, float(dynamic[n_bins, n_rows])


def test_vectorized_dynamic_program_reproduces_the_reference_loop() -> None:
    rng = np.random.default_rng(5)
    for _ in range(20):
        n_rows = int(rng.integers(4, 60))
        n_bins = int(rng.integers(1, min(n_rows, 6) + 1))
        values = rng.normal(size=n_rows)
        weights = rng.uniform(0.1, 2.0, size=n_rows)
        labels, objective = scalar_interval_dp(values, weights, n_bins)
        expected_labels, expected_objective = _reference_interval_dp(values, weights, n_bins)
        np.testing.assert_array_equal(labels, expected_labels)
        assert objective == expected_objective


def test_striped_evaluation_does_not_change_the_optimum(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tiny memory budget forces many stripes and must not alter the result."""
    rng = np.random.default_rng(11)
    values = rng.normal(size=57)
    weights = rng.uniform(0.3, 1.7, size=57)
    labels, objective = scalar_interval_dp(values, weights, 4)
    monkeypatch.setattr(scalar_dp, "_DYNAMIC_WORKING_SET_BYTES", 512)
    striped_labels, striped_objective = scalar_interval_dp(values, weights, 4)
    np.testing.assert_array_equal(striped_labels, labels)
    assert striped_objective == objective


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_dynamic_program_attains_the_exhaustive_optimum(seed: int) -> None:
    """Interval optimality in one dimension makes the program globally exact."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(9, 1))
    weights = rng.uniform(0.5, 1.5, size=9)
    _, oracle_objective = _exhaustive_d_oracle(scores, weights, 3)
    result = sq.fit_quantizer(
        sq.ScoreSample(scores, weights),
        n_bins=3,
        criterion=sq.DOptimality(),
        config=sq.ScalarDPConfig(),
    )
    achieved = sq.information_report(scores, result.labels, weights, n_bins=3).logdet_retention
    assert achieved == pytest.approx(oracle_objective, abs=_objective_tolerance())


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_dynamic_program_is_never_worse_than_whitened_kmeans(seed: int) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(300, 1)) * np.where(rng.random((300, 1)) < 0.5, 0.4, 2.0)
    weights = rng.uniform(0.2, 1.8, size=300)
    exact = sq.fit_quantizer(
        sq.ScoreSample(scores, weights),
        n_bins=6,
        criterion=sq.DOptimality(),
        config=sq.ScalarDPConfig(),
    )
    approximate = fit_test_quantizer(
        scores,
        weights=weights,
        n_bins=6,
        config=sq.KMeansConfig(seed=seed, solver_restarts=8),
    )
    assert (
        exact.train_report.geometric_mean_retention
        >= approximate.train_report.geometric_mean_retention - 1e-12
    )


def test_scalar_solver_rejects_a_higher_rank_score_space() -> None:
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="rank of one, got rank 2"):
        sq.fit_quantizer(
            sq.ScoreSample(rng.normal(size=(40, 2))),
            n_bins=3,
            criterion=sq.DOptimality(),
            config=sq.ScalarDPConfig(),
        )


def test_scalar_solver_enforces_its_capacity_guard() -> None:
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="max_rows"):
        sq.fit_quantizer(
            sq.ScoreSample(rng.normal(size=(40, 1))),
            n_bins=3,
            criterion=sq.DOptimality(),
            config=sq.ScalarDPConfig(max_rows=10),
        )


def test_scalar_config_validates_and_serializes() -> None:
    config = sq.ScalarDPConfig()
    assert config.method == "scalar_dp"
    assert config.whiten is True
    assert config.seed == 0
    assert config.max_rows == 20_000
    assert config.to_dict()["method"] == "scalar_dp"
    with pytest.raises(TypeError, match="whiten"):
        sq.ScalarDPConfig(whiten=1)
    with pytest.raises(ValueError, match="seed"):
        sq.ScalarDPConfig(seed=-1)
    with pytest.raises(ValueError, match="max_rows"):
        sq.ScalarDPConfig(max_rows=0)
    with pytest.raises(ValueError, match="rank_rtol"):
        sq.ScalarDPConfig(rank_rtol=1.0)


def test_whitening_does_not_move_the_interval_optimum() -> None:
    """Scalar whitening is a positive rescaling, so the labels are invariant."""
    rng = np.random.default_rng(21)
    scores = rng.normal(size=(120, 1)) * 7.0
    weights = rng.uniform(0.5, 1.5, size=120)
    whitened = sq.fit_quantizer(
        sq.ScoreSample(scores, weights),
        n_bins=4,
        criterion=sq.DOptimality(),
        config=sq.ScalarDPConfig(whiten=True),
    )
    raw = sq.fit_quantizer(
        sq.ScoreSample(scores, weights),
        n_bins=4,
        criterion=sq.DOptimality(),
        config=sq.ScalarDPConfig(whiten=False),
    )
    # The retained eigenvector carries an arbitrary sign, so the invariant is
    # the induced partition rather than the numbering of its cells.
    left = np.asarray(whitened.labels)
    right = np.asarray(raw.labels)
    np.testing.assert_array_equal(left[:, None] == left[None, :], right[:, None] == right[None, :])


def test_scalar_trace_declares_its_objective_units() -> None:
    rng = np.random.default_rng(6)
    scores = rng.normal(size=(80, 1))
    result = sq.fit_quantizer(
        sq.ScoreSample(scores),
        n_bins=3,
        criterion=sq.DOptimality(),
        config=sq.ScalarDPConfig(),
    )
    assert result.trace.objective_label == "whitened_sse"
    assert result.to_dict()["trace"]["objective_label"] == "whitened_sse"
