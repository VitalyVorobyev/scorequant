from unittest.mock import patch

import numpy as np
import pytest

import scorequant as sq
from scorequant._exchange import DObjective, ExchangeState, cell_statistics, relocation
from scorequant.solvers.exchange import scan

from ._oracles import _exact_d_move_gain


def _scores(seed: int = 11, n_rows: int = 72) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n_rows, 2))
    return scores - scores.mean(axis=0)


def _d_state(
    scores: np.ndarray, weights: np.ndarray, labels: np.ndarray, n_bins: int
) -> ExchangeState:
    return DObjective().init_state(cell_statistics(scores, weights, labels, n_bins))


def test_exact_move_gain_matches_direct_recomputation() -> None:
    scores = _scores(n_rows=12)
    weights = np.full(12, 1 / 12)
    labels = np.repeat(np.arange(3), 4)
    information = np.asarray(sq.binned_fisher_information(scores, labels, weights, n_bins=3))
    row, source, destination = 1, 0, 2
    source_mask = labels == source
    destination_mask = labels == destination
    source_weight = float(weights[source_mask].sum())
    destination_weight = float(weights[destination_mask].sum())
    source_mean = np.average(scores[source_mask], axis=0, weights=weights[source_mask])
    destination_mean = np.average(
        scores[destination_mask], axis=0, weights=weights[destination_mask]
    )
    predicted = _exact_d_move_gain(
        np.linalg.inv(information),
        scores[row] - source_mean,
        scores[row] - destination_mean,
        point_weight=float(weights[row]),
        source_weight=source_weight,
        destination_weight=destination_weight,
    )
    moved = labels.copy()
    moved[row] = destination
    updated = np.asarray(sq.binned_fisher_information(scores, moved, weights, n_bins=3))
    direct = np.linalg.slogdet(updated)[1] - np.linalg.slogdet(information)[1]
    assert predicted == pytest.approx(direct, abs=1e-10)


def test_chunked_scan_and_rank_two_state_update_match_full_recomputation() -> None:
    scores = _scores(seed=19, n_rows=48)
    weights = np.linspace(0.5, 1.5, len(scores))
    labels = np.tile(np.arange(3), 16)
    state = _d_state(scores, weights, labels, 3)
    objective = DObjective()
    config = sq.DExchangeConfig(max_iter=10)
    with patch("scorequant._chunking.WORKING_SET_BYTES", 256):
        chunked = scan(scores, weights, labels, state, objective, config, rows=True).best
    with patch("scorequant._chunking.WORKING_SET_BYTES", 1 << 40):
        unchunked = scan(scores, weights, labels, state, objective, config, rows=True).best
    assert chunked == unchunked
    assert chunked is not None
    updated = objective.apply_move(
        state, relocation(scores, weights, labels, state, chunked.row, chunked.destination)
    )
    moved_labels = labels.copy()
    moved_labels[chunked.row] = chunked.destination
    recomputed = _d_state(scores, weights, moved_labels, 3)
    np.testing.assert_allclose(updated.weights, recomputed.weights, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(updated.means, recomputed.means, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(updated.information, recomputed.information, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(updated.inverse, recomputed.inverse, rtol=1e-10, atol=1e-10)
    assert updated.objective == pytest.approx(recomputed.objective, abs=1e-12)


def test_repeated_rank_two_updates_do_not_accumulate_material_drift() -> None:
    scores = _scores(seed=191, n_rows=96)
    weights = np.linspace(0.25, 2.0, len(scores))
    labels = np.tile(np.arange(4), 24)
    state = _d_state(scores, weights, labels, 4)
    objective = DObjective()
    config = sq.DExchangeConfig(max_iter=20)
    for _ in range(12):
        move = scan(scores, weights, labels, state, objective, config, rows=False).best
        assert move is not None and move.gain > 0
        state = objective.apply_move(
            state, relocation(scores, weights, labels, state, move.row, move.destination)
        )
        labels[move.row] = move.destination
        recomputed = _d_state(scores, weights, labels, 4)
        np.testing.assert_allclose(
            state.information, recomputed.information, rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(state.inverse, recomputed.inverse, rtol=1e-9, atol=1e-9)
        assert state.objective == pytest.approx(recomputed.objective, abs=1e-11)


def test_partition_is_transductive_and_d_compilation_is_explicit() -> None:
    scores = _scores()
    partition = sq.optimize_partition(
        scores,
        n_bins=3,
        config=sq.DExchangeConfig(seed=4, initializer_restarts=3, max_iter=200),
    )
    assert partition.exchange_stable
    assert partition.rank == 2
    assert partition.best_remaining_gain <= partition.config.gain_tolerance
    assert not hasattr(partition, "predict")
    assert np.all(np.diff(np.asarray(partition.objective_history)) > 0)
    quantizer = partition.compile_quantizer()
    assert quantizer.rank == partition.rank
    assert np.array_equal(
        np.asarray(quantizer.predict_scores(scores)), np.asarray(partition.labels)
    )


def test_zero_weight_rows_are_predictable_but_do_not_change_objective() -> None:
    scores = _scores(n_rows=40)
    extra = np.array([[100.0, -100.0], [-80.0, 120.0]])
    config = sq.DExchangeConfig(seed=9, initializer_restarts=3, max_iter=200)
    reference = sq.optimize_partition(scores, n_bins=3, config=config)
    extended = sq.optimize_partition(
        np.vstack([scores, extra]),
        weights=np.r_[np.ones(len(scores)), 0.0, 0.0],
        n_bins=3,
        config=config,
    )
    assert extended.objective == pytest.approx(reference.objective, abs=1e-9)
    assert len(extended.labels) == len(scores) + 2
    extended.compile_quantizer()


def test_solver_criterion_pairs_are_explicit() -> None:
    sample = sq.ScoreSample(_scores())
    with pytest.raises(ValueError, match="NormalizedTrace"):
        sq.fit_quantizer(sample, n_bins=3, config=sq.KMeansConfig(solver_restarts=2))
    trace = sq.fit_quantizer(
        sample,
        n_bins=3,
        criterion=sq.NormalizedTrace(),
        config=sq.KMeansConfig(solver_restarts=2),
    )
    assert trace.n_bins == 3


def test_exact_partition_preserves_core_invariances_without_centering() -> None:
    scores = _scores(seed=31, n_rows=36) + np.array([2.0, -0.7])
    weights = np.linspace(0.4, 1.8, len(scores))
    config = sq.DExchangeConfig(seed=14, initializer_restarts=8, max_iter=300)
    original = sq.optimize_partition(scores, weights=weights, n_bins=3, config=config)
    np.testing.assert_allclose(original.training_scores, scores)

    permutation = np.random.default_rng(2).permutation(len(scores))
    reordered = sq.optimize_partition(
        scores[permutation], weights=weights[permutation], n_bins=3, config=config
    )
    restored = np.empty(len(scores), dtype=int)
    restored[permutation] = np.asarray(reordered.labels)
    original_pairs = np.asarray(original.labels)[:, None] == np.asarray(original.labels)[None, :]
    restored_pairs = restored[:, None] == restored[None, :]
    assert np.array_equal(original_pairs, restored_pairs)

    scaled = sq.optimize_partition(scores, weights=17 * weights, n_bins=3, config=config)
    scaled_pairs = np.asarray(scaled.labels)[:, None] == np.asarray(scaled.labels)[None, :]
    assert np.array_equal(original_pairs, scaled_pairs)
    assert scaled.train_report.geometric_mean_retention == pytest.approx(
        original.train_report.geometric_mean_retention, abs=1e-10
    )

    reparameterization = np.array([[1.3, -0.2], [0.4, 0.9]])
    transformed = sq.optimize_partition(
        scores @ reparameterization, weights=weights, n_bins=3, config=config
    )
    transformed_pairs = (
        np.asarray(transformed.labels)[:, None] == np.asarray(transformed.labels)[None, :]
    )
    assert np.array_equal(original_pairs, transformed_pairs)

    duplicated_scores = np.repeat(scores, 2, axis=0)
    duplicated_weights = np.repeat(weights / 2, 2)
    duplicated = sq.optimize_partition(
        duplicated_scores, weights=duplicated_weights, n_bins=3, config=config
    )
    duplicate_labels = np.asarray(duplicated.labels).reshape(-1, 2)
    assert np.array_equal(duplicate_labels[:, 0], duplicate_labels[:, 1])
    duplicate_pairs = duplicate_labels[:, 0, None] == duplicate_labels[:, 0][None, :]
    assert np.array_equal(original_pairs, duplicate_pairs)
