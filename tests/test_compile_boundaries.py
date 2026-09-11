"""Compilation contracts at tolerated, duplicate and singular boundaries."""

import numpy as np
import pytest

import scorequant as sq


def test_tolerated_singleton_tie_is_refused_before_compilation() -> None:
    scores = np.array([[0.0], [1.0], [2.0]])
    with pytest.raises(sq.RefusalError, match="geometrically degenerate"):
        sq.optimize_partition(
            scores,
            n_bins=2,
            initial_labels=[0, 1, 0],
            config=sq.DExchangeConfig(gain_tolerance=0.5),
        )


def test_individual_compile_tolerance_does_not_bound_batch_change() -> None:
    scores = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    result = sq.optimize_partition(
        scores,
        n_bins=2,
        initial_labels=[0, 1, 0, 1],
        config=sq.DExchangeConfig(gain_tolerance=1.2),
    )
    predicted = np.asarray(result.compile_quantizer().predict_scores(scores))
    assert result.exchange_stable
    assert result.geometry is not None
    assert result.geometry.maximum_violation_gain == pytest.approx(np.log(3))
    assert result.geometry.maximum_violation_gain < 1.2
    np.testing.assert_array_equal(predicted, [0, 0, 1, 1])
    before = np.asarray(result.information_partitioned)[0, 0]
    after = np.asarray(sq.binned_fisher_information(scores, predicted, n_bins=2))[0, 0]
    assert np.log(after / before) == pytest.approx(np.log(4))
    assert np.log(after / before) > result.config.gain_tolerance


def test_compile_duplicate_inheritance_zero_weights_and_projected_rank() -> None:
    scores = np.array([[-3.0, -6.0], [-1.0, -2.0], [1.0, 2.0], [3.0, 6.0]])
    expanded = np.vstack([scores, scores[0], [100.0, -50.0]])
    result = sq.optimize_partition(
        expanded,
        weights=[0.5, 1.0, 1.0, 1.0, 0.5, 0.0],
        n_bins=2,
        initial_labels=[0, 0, 1, 1, 0, 1],
        config=sq.DExchangeConfig(collapse_duplicates=True, gain_tolerance=0.0),
    )
    predicted = np.asarray(result.compile_quantizer().predict_scores(expanded))
    assert result.rank == 1
    assert predicted[0] == predicted[4]
    np.testing.assert_array_equal(predicted, result.labels)
    assert np.linalg.matrix_rank(np.asarray(result.information_partitioned)) == 1
