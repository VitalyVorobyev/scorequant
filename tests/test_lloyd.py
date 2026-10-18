"""Behavioral gates for the guarded Mahalanobis-Lloyd solver.

The unguarded batch iteration is not monotone, so every gate here is about the
guard: accepted steps must strictly improve the exactly rebuilt objective, the
frozen counterexample must not be able to lower it, and the ``"exchange"``
guard must leave an exchange-stable, compilable terminal state.
"""

from __future__ import annotations

import numpy as np
import pytest

import scorequant as sq

_COMPARISON_SEEDS = (1, 2, 3, 4, 5)


def _seeded_problem(
    seed: int, n_rows: int = 1_200, n_features: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n_rows, n_features))
    return scores - scores.mean(axis=0), rng.uniform(0.4, 2.0, size=n_rows)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
@pytest.mark.parametrize("guard", ["exchange", "reject"])
@pytest.mark.parametrize("criterion", [None, sq.ProfiledDOptimality((0,))], ids=["d", "profiled_d"])
def test_guarded_history_never_decreases(
    seed: int, guard: str, criterion: sq.ProfiledDOptimality | None
) -> None:
    """Every recorded step of either phase is certified by an exact rebuild."""
    scores, weights = _seeded_problem(seed)
    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=5,
        criterion=criterion,
        config=sq.MahalanobisLloydConfig(seed=seed, guard=guard),  # type: ignore[arg-type]
    )
    history = np.asarray(result.objective_history)
    assert history.shape[0] >= 1
    assert np.all(np.diff(history) > 0)
    assert result.objective >= float(history[0])
    assert result.objective == pytest.approx(float(history[-1]), abs=0)
    assert result.lloyd_iterations >= 1
    assert result.accepted_lloyd_steps <= result.lloyd_iterations
    if guard == "reject":
        assert history.shape[0] == 1 + result.accepted_lloyd_steps
        assert result.accepted_moves == 0
    else:
        assert history.shape[0] >= 1 + result.accepted_lloyd_steps
        assert result.exchange_stable is True


def test_frozen_counterexample_batch_step_is_rejected_not_recorded() -> None:
    """The committed adaptive-Lloyd fixture loses 0.136521 nat; the guard cannot."""
    scores = np.array(
        [
            [0.1116, 0.4427],
            [-0.2932, 0.6537],
            [-0.5995, -1.2685],
            [-0.6848, -1.5456],
            [0.4810, 0.9521],
            [1.6707, 0.9370],
            [0.1689, 1.7090],
            [-0.8548, -1.8805],
        ]
    )
    labels = np.array([1, 0, 0, 1, 2, 2, 2, 1])
    weights = np.full(len(scores), 1 / len(scores))
    information = np.asarray(
        sq.information.binned_fisher_information(scores, labels, weights, n_bins=3)
    )
    means = np.asarray([scores[labels == label].mean(axis=0) for label in range(3)])
    residuals = scores[:, None, :] - means[None, :, :]
    distances = np.einsum("nkp,pq,nkq->nk", residuals, np.linalg.inv(information), residuals)
    batch_labels = np.argmin(distances, axis=1)
    batch = np.asarray(
        sq.information.binned_fisher_information(scores, batch_labels, weights, n_bins=3)
    )
    raw_step = np.linalg.slogdet(batch)[1] - np.linalg.slogdet(information)[1]
    assert raw_step == pytest.approx(-0.136521, abs=2e-6)

    for guard in ("exchange", "reject"):
        result = sq.optimize_partition(
            scores,
            weights=weights,
            n_bins=3,
            config=sq.MahalanobisLloydConfig(seed=0, initializer_restarts=4, guard=guard),  # type: ignore[arg-type]
        )
        history = np.asarray(result.objective_history)
        assert np.all(np.diff(history) > 0)
        assert result.objective >= float(history[0])
        assert result.objective >= np.linalg.slogdet(information)[1] - 1e-12


def test_exchange_guard_leaves_a_stable_compilable_partition() -> None:
    """The handoff guarantees the terminal contract that compilation requires."""
    scores, weights = _seeded_problem(17, n_rows=2_000)
    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=6,
        config=sq.MahalanobisLloydConfig(seed=17),
    )
    assert result.exchange_stable is True
    assert result.best_remaining_gain <= result.config.gain_tolerance
    quantizer = result.compile_quantizer()
    assert np.array_equal(np.asarray(quantizer.predict_scores(scores)), np.asarray(result.labels))


def test_reject_guard_reports_its_terminal_stability_honestly() -> None:
    """``"reject"`` certifies with one scan and refuses to compile an unstable state."""
    scores, weights = _seeded_problem(23, n_rows=1_500)
    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=5,
        config=sq.MahalanobisLloydConfig(seed=23, guard="reject"),
    )
    assert result.scans == 1
    assert result.accepted_moves == 0
    if result.exchange_stable:
        result.compile_quantizer()
    else:
        assert result.best_remaining_gain > result.config.gain_tolerance
        with pytest.raises(sq.RefusalError, match="guard='exchange'"):
            result.compile_quantizer()


def test_guarded_batches_match_or_approach_plain_exchange() -> None:
    """The guarded solver reaches comparable exchange-stable optima."""
    matched = 0
    for seed in _COMPARISON_SEEDS:
        scores, weights = _seeded_problem(seed, n_rows=1_200)
        lloyd = sq.optimize_partition(
            scores, weights=weights, n_bins=5, config=sq.MahalanobisLloydConfig(seed=seed)
        )
        exchange = sq.optimize_partition(
            scores, weights=weights, n_bins=5, config=sq.DExchangeConfig(seed=seed)
        )
        assert lloyd.exchange_stable is True
        assert lloyd.objective >= float(np.asarray(lloyd.objective_history)[0])
        matched += lloyd.objective >= exchange.objective - 1e-9
    assert 2 * matched >= len(_COMPARISON_SEEDS)


def test_full_data_pass_counts_are_recorded_at_scale() -> None:
    """Report guarded batch passes against exchange scans without asserting speed."""
    scores, weights = _seeded_problem(3, n_rows=20_000)
    lloyd = sq.optimize_partition(
        scores, weights=weights, n_bins=6, config=sq.MahalanobisLloydConfig(seed=3)
    )
    exchange = sq.optimize_partition(
        scores, weights=weights, n_bins=6, config=sq.DExchangeConfig(seed=3)
    )
    print(
        f"lloyd passes={lloyd.lloyd_iterations + lloyd.scans} "
        f"(iterations={lloyd.lloyd_iterations}, accepted={lloyd.accepted_lloyd_steps}, "
        f"scans={lloyd.scans}, moves={lloyd.accepted_moves}) "
        f"exchange passes={exchange.scans} (moves={exchange.accepted_moves}) "
        f"objective gap={lloyd.objective - exchange.objective:.3e}"
    )
    assert lloyd.exchange_stable is True
    assert np.all(np.diff(np.asarray(lloyd.objective_history)) > 0)
    assert lloyd.lloyd_iterations >= 1


def test_quantizer_fit_is_end_to_end_and_deterministic() -> None:
    """``fit_quantizer`` compiles the guarded partition into a reusable rule."""
    rng = np.random.default_rng(55)
    scores = rng.normal(size=(20_000, 3))
    config = sq.MahalanobisLloydConfig(seed=6, initializer_restarts=2)
    first = sq.fit_quantizer(sq.ScoreSample(scores), n_bins=6, config=config)
    second = sq.fit_quantizer(sq.ScoreSample(scores), n_bins=6, config=config)
    assert first.n_bins == 6
    assert np.array_equal(np.asarray(first.labels), np.asarray(second.labels))
    assert np.array_equal(np.asarray(first.predict_scores(scores)), np.asarray(first.labels))
    assert np.isfinite(first.train_report.geometric_mean_retention)
    assert first.predict_scores(rng.normal(size=(64, 3))).shape == (64,)


def test_profiled_labels_refuse_implicit_compilation() -> None:
    """A guarded profiled partition keeps the finite/inductive boundary."""
    scores, weights = _seeded_problem(29, n_rows=900)
    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=5,
        criterion=sq.ProfiledDOptimality((0, 1)),
        config=sq.MahalanobisLloydConfig(seed=29),
    )
    assert result.profiled_report is not None
    assert result.profiled_geometry is not None
    with pytest.raises(sq.RefusalError, match="no canonical inductive compilation"):
        result.compile_quantizer()
    with pytest.raises(ValueError, match="implements only DOptimality"):
        sq.fit_quantizer(
            sq.ScoreSample(scores, weights),
            n_bins=5,
            criterion=sq.ProfiledDOptimality((0, 1)),
            config=sq.MahalanobisLloydConfig(),
        )


def test_configuration_validates_its_contract() -> None:
    """The guarded batch configuration validates at construction time."""
    with pytest.raises(ValueError, match="max_iter"):
        sq.MahalanobisLloydConfig(max_iter=0)
    with pytest.raises(ValueError, match="guard must be"):
        sq.MahalanobisLloydConfig(guard="none")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initializer_restarts"):
        sq.MahalanobisLloydConfig(initializer_restarts=0)
    with pytest.raises(ValueError, match="seed"):
        sq.MahalanobisLloydConfig(seed=-1)
    with pytest.raises(TypeError, match="max_iter"):
        sq.MahalanobisLloydConfig(max_iter=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank_rtol"):
        sq.MahalanobisLloydConfig(rank_rtol=1.0)
    with pytest.raises(ValueError, match="gain_tolerance"):
        sq.MahalanobisLloydConfig(gain_tolerance=-1e-3)
    assert sq.MahalanobisLloydConfig().to_dict()["method"] == "mahalanobis_lloyd"
    assert sq.MahalanobisLloydConfig().to_dict()["guard"] == "exchange"


def test_partition_rejects_an_unsupported_configuration_object() -> None:
    """Only the two finite solvers are accepted by fixed-sample assignment."""
    scores, _ = _seeded_problem(3, n_rows=200)
    with pytest.raises(TypeError, match="MahalanobisLloydConfig"):
        sq.optimize_partition(
            scores,
            n_bins=4,
            config=sq.KMeansConfig(),  # type: ignore[arg-type]
        )
