"""Behavioral gates for the unified exchange engine.

These tests cover the properties the single scan loop must hold for both finite
criteria: guarded batch acceptance never degrades the exact objective, every
accepted step strictly increases it, a large default fit reaches exchange
stability instead of silently stopping at a scan cap, and restarts are
deterministic.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pytest

import scorequant as sq

_SCALE_ROWS = 80_000
_SCALE_BINS = 6


def _seeded_scores(
    seed: int, n_rows: int, n_features: int, *, weighted: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n_rows, n_features))
    weights = rng.uniform(0.4, 2.0, size=n_rows) if weighted else None
    return scores, weights


def test_batch_acceptance_matches_or_beats_single_moves_on_the_golden_fixture() -> None:
    """Guarded batching must never land in a worse exchange-stable state."""
    rng = np.random.default_rng(102)
    scores = rng.normal(size=(200, 3))
    weights = rng.uniform(0.5, 2.0, size=200)
    single = sq.optimize_partition(
        scores, weights=weights, n_bins=5, config=sq.DExchangeConfig(batch_moves=False)
    )
    batched = sq.optimize_partition(
        scores, weights=weights, n_bins=5, config=sq.DExchangeConfig(batch_moves=True)
    )
    assert single.exchange_stable is True
    assert batched.exchange_stable is True
    assert batched.objective >= single.objective - 1e-12


@pytest.mark.parametrize("seed", [11, 12, 13])
@pytest.mark.parametrize(
    ("criterion", "n_features", "n_bins"),
    [(None, 3, 5), (sq.ProfiledDOptimality((0,)), 3, 5)],
    ids=["d", "profiled_d"],
)
def test_batched_objective_history_is_strictly_increasing(
    seed: int, criterion: sq.ProfiledDOptimality | None, n_features: int, n_bins: int
) -> None:
    """Every accepted batch or single move is verified against the exact objective."""
    scores, weights = _seeded_scores(seed, 400, n_features, weighted=seed % 2 == 1)
    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=n_bins,
        criterion=criterion,
        config=sq.DExchangeConfig(seed=seed, batch_moves=True),
    )
    history = np.asarray(result.objective_history)
    assert result.exchange_stable is True
    assert history.shape[0] >= 2
    assert np.all(np.diff(history) > 0)
    assert result.objective == pytest.approx(float(history[-1]), abs=0)


def test_batched_and_single_agree_with_the_exhaustive_oracle() -> None:
    """Both acceptance modes still reach the global optimum of a tiny instance."""
    from ._oracles import _exhaustive_d_oracle

    scores = np.array(
        [[-2.0, -1.0], [-1.0, 1.0], [-0.2, -0.4], [0.3, 0.8], [1.1, -0.8], [1.8, 0.7]]
    )
    scores -= scores.mean(axis=0)
    _, optimum = _exhaustive_d_oracle(scores, None, 3)
    for batch_moves in (False, True):
        result = sq.optimize_partition(
            scores,
            n_bins=3,
            config=sq.DExchangeConfig(seed=8, initializer_restarts=12, batch_moves=batch_moves),
        )
        assert result.objective == pytest.approx(optimum, abs=1e-10)


def test_default_configuration_reaches_stability_and_compiles_at_scale() -> None:
    """The default path must not stop at a scan cap on a large sample."""
    rng = np.random.default_rng(2027)
    scores = rng.normal(size=(_SCALE_ROWS, 3))
    started = perf_counter()
    partition = sq.optimize_partition(scores, n_bins=_SCALE_BINS, config=sq.DExchangeConfig())
    assert partition.exchange_stable is True
    assert partition.accepted_moves > partition.scans
    assert partition.best_remaining_gain <= partition.config.gain_tolerance

    quantizer = sq.fit_quantizer(sq.ScoreSample(scores), n_bins=_SCALE_BINS)
    elapsed = perf_counter() - started
    assert quantizer.n_bins == _SCALE_BINS
    assert np.array_equal(
        np.asarray(quantizer.predict_scores(scores)), np.asarray(quantizer.labels)
    )
    assert elapsed < 120


def test_capped_scans_report_the_remaining_gain_and_refuse_compilation() -> None:
    """An explicit scan cap must fail loudly instead of compiling an unstable state."""
    scores, weights = _seeded_scores(31, 2_000, 3, weighted=True)
    partition = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=6,
        config=sq.DExchangeConfig(batch_moves=False, max_iter=2),
    )
    assert partition.scans == 3
    assert partition.exchange_stable is False
    assert partition.best_remaining_gain > partition.config.gain_tolerance
    with pytest.raises(sq.RefusalError, match="max_iter"):
        partition.compile_quantizer()


def test_restarts_are_deterministic_and_never_worse_than_one_restart() -> None:
    """Restart seeding is derived from the configured seed alone."""
    scores, weights = _seeded_scores(44, 600, 3, weighted=True)
    config = sq.DExchangeConfig(seed=5, initializer_restarts=2, solver_restarts=3)
    first = sq.optimize_partition(scores, weights=weights, n_bins=6, config=config)
    second = sq.optimize_partition(scores, weights=weights, n_bins=6, config=config)
    assert np.array_equal(np.asarray(first.labels), np.asarray(second.labels))
    assert first.objective == pytest.approx(second.objective, abs=0)

    single = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=6,
        config=sq.DExchangeConfig(seed=5, initializer_restarts=2, solver_restarts=1),
    )
    assert first.objective >= single.objective - 1e-12


def test_random_initialization_is_deterministic_and_reaches_stability() -> None:
    """Balanced random labels are a supported, seeded starting point."""
    scores, weights = _seeded_scores(77, 500, 3, weighted=False)
    config = sq.DExchangeConfig(seed=3, init="random", solver_restarts=2)
    first = sq.optimize_partition(scores, weights=weights, n_bins=4, config=config)
    second = sq.optimize_partition(scores, weights=weights, n_bins=4, config=config)
    assert first.exchange_stable is True
    assert np.array_equal(np.asarray(first.labels), np.asarray(second.labels))


def test_first_improvement_still_accepts_one_move_per_scan() -> None:
    """``first_improvement`` keeps its single-move contract and ignores batching."""
    scores, weights = _seeded_scores(19, 300, 2, weighted=True)
    config = sq.DExchangeConfig(
        seed=6, initializer_restarts=2, first_improvement=True, batch_moves=True
    )
    result = sq.optimize_partition(scores, weights=weights, n_bins=4, config=config)
    assert result.exchange_stable is True
    assert result.scans == result.accepted_moves + 1
    assert np.all(np.diff(np.asarray(result.objective_history)) > 0)


def test_configuration_rejects_invalid_engine_settings() -> None:
    """The exchange configuration validates its new fields at construction time."""
    with pytest.raises(ValueError, match="max_iter"):
        sq.DExchangeConfig(max_iter=0)
    with pytest.raises(ValueError, match="solver_restarts"):
        sq.DExchangeConfig(solver_restarts=0)
    with pytest.raises(ValueError, match="init must be"):
        sq.DExchangeConfig(init="lloyd")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="batch_moves"):
        sq.DExchangeConfig(batch_moves=1)  # type: ignore[arg-type]
    assert sq.DExchangeConfig().to_dict()["max_iter"] is None


def test_reported_objective_is_rebuilt_from_the_final_labels() -> None:
    """Single moves update the state incrementally; the result is the exact rebuild."""
    scores, weights = _seeded_scores(41, 300, 3, weighted=True)
    config = sq.DExchangeConfig(batch_moves=False, init="random", collapse_duplicates=False)
    result = sq.optimize_partition(scores, weights=weights, n_bins=5, config=config)
    assert result.accepted_moves > 100
    report = sq.certify.exchange_stability_report(scores, result.labels, weights=weights)
    assert result.objective == report.objective
    assert result.objective_history[-1] == result.objective
