"""Property gates for the certified efficient-score upper bound."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import scorequant as sq

_PROBLEMS = [
    (301, 90, 3, (0,), True, 4),
    (302, 120, 3, (1,), False, 4),
    (303, 140, 4, (2,), True, 5),
    (304, 80, 4, (0,), False, 4),
    (305, 160, 5, (3,), True, 6),
]


def _instance(
    seed: int, n_rows: int, n_columns: int, weighted: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Build a correlated seeded score table with an optional nonuniform measure."""
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(n_columns, n_columns))
    scores = rng.normal(size=(n_rows, n_columns)) @ mixing
    weights = rng.uniform(0.3, 1.7, size=n_rows) if weighted else np.ones(n_rows)
    return scores, weights


@pytest.mark.parametrize(
    ("seed", "n_rows", "n_columns", "interest", "weighted", "n_bins"), _PROBLEMS
)
def test_bound_dominates_the_achieved_profiled_objective(
    seed: int,
    n_rows: int,
    n_columns: int,
    interest: tuple[int, ...],
    weighted: bool,
    n_bins: int,
) -> None:
    scores, weights = _instance(seed, n_rows, n_columns, weighted)
    bound = sq.efficient_score_bound(scores, interest=interest, weights=weights, n_bins=n_bins)
    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=n_bins,
        criterion=sq.ProfiledDOptimality(interest),
        config=sq.DExchangeConfig(seed=3, initializer_restarts=8, solver_restarts=2),
    )
    assert bound.gap_to(result) >= -1e-9
    assert bound.labels.shape == (n_rows,)
    assert bound.efficient_scores.shape == (n_rows, 1)
    assert bound.to_dict()["upper_bound"] == pytest.approx(bound.upper_bound)


def test_bound_matches_the_between_cell_moment_of_its_own_labels() -> None:
    """The reported ceiling is exactly the objective of the labels it returns."""
    scores, weights = _instance(311, 100, 3, True)
    bound = sq.efficient_score_bound(scores, interest=(0,), weights=weights, n_bins=4)
    efficient = np.asarray(bound.efficient_scores)[:, 0]
    labels = np.asarray(bound.labels)
    cell_weights = np.bincount(labels, weights=weights, minlength=4)
    cell_sums = np.bincount(labels, weights=weights * efficient, minlength=4)
    between = float(np.sum(cell_weights * (cell_sums / cell_weights) ** 2))
    tolerance = 1e-12 if jax.config.jax_enable_x64 else 1e-5
    assert bound.upper_bound == pytest.approx(float(np.log(between)), abs=tolerance)


def test_bound_is_tight_when_nuisance_is_orthogonal_and_interest_separates() -> None:
    """A block-diagonal score law leaves almost no profiling loss to absorb."""
    rng = np.random.default_rng(404)
    n_rows = 200
    half = n_rows // 2
    interest_column = np.concatenate(
        [rng.normal(-3.0, 0.4, half), rng.normal(3.0, 0.4, n_rows - half)]
    )
    nuisance_columns = rng.normal(size=(n_rows, 2))
    scores = np.column_stack([interest_column, nuisance_columns])
    weights = np.ones(n_rows)
    bound = sq.efficient_score_bound(scores, interest=(0,), weights=weights, n_bins=5)
    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=5,
        criterion=sq.ProfiledDOptimality((0,)),
        config=sq.DExchangeConfig(seed=1, initializer_restarts=8, solver_restarts=3),
    )
    gap = bound.gap_to(result)
    assert 0 <= gap < 0.1 * abs(bound.upper_bound)


def test_bound_labels_initialize_profiled_exchange() -> None:
    """The relaxed upper problem is a better start than generic k-means seeding."""
    wins = 0
    for seed, n_rows, n_columns, interest, weighted, n_bins in _PROBLEMS:
        scores, weights = _instance(seed, n_rows, n_columns, weighted)
        criterion = sq.ProfiledDOptimality(interest)
        config = sq.DExchangeConfig(seed=7, initializer_restarts=4, solver_restarts=1)
        bound = sq.efficient_score_bound(scores, interest=interest, weights=weights, n_bins=n_bins)
        seeded = sq.optimize_partition(
            scores, weights=weights, n_bins=n_bins, criterion=criterion, config=config
        )
        initialized = sq.optimize_partition(
            scores,
            weights=weights,
            n_bins=n_bins,
            criterion=criterion,
            config=config,
            initial_labels=bound.labels,
        )
        assert bound.gap_to(seeded) >= -1e-9
        assert bound.gap_to(initialized) >= -1e-9
        wins += int(initialized.objective >= seeded.objective - 1e-12)
    assert wins * 2 >= len(_PROBLEMS)


def test_further_restarts_still_use_ordinary_seeding() -> None:
    """Supplied labels replace only the first restart, so extra restarts still explore."""
    scores, weights = _instance(321, 100, 3, True)
    criterion = sq.ProfiledDOptimality((0,))
    bound = sq.efficient_score_bound(scores, interest=(0,), weights=weights, n_bins=4)
    single = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=4,
        criterion=criterion,
        config=sq.DExchangeConfig(seed=2, initializer_restarts=4, solver_restarts=1),
        initial_labels=bound.labels,
    )
    multiple = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=4,
        criterion=criterion,
        config=sq.DExchangeConfig(seed=2, initializer_restarts=4, solver_restarts=4),
        initial_labels=bound.labels,
    )
    assert multiple.objective >= single.objective - 1e-12
    assert bound.gap_to(multiple) >= -1e-9


def test_initial_labels_are_validated() -> None:
    scores, weights = _instance(331, 60, 3, True)
    criterion = sq.ProfiledDOptimality((0,))
    with pytest.raises(ValueError, match=r"shape \[60\]"):
        sq.optimize_partition(
            scores, weights=weights, n_bins=4, criterion=criterion, initial_labels=np.zeros(59, int)
        )
    with pytest.raises(TypeError, match="integer bin labels"):
        sq.optimize_partition(
            scores, weights=weights, n_bins=4, criterion=criterion, initial_labels=np.zeros(60)
        )
    with pytest.raises(ValueError, match=r"outside \[0, n_bins\)"):
        sq.optimize_partition(
            scores,
            weights=weights,
            n_bins=4,
            criterion=criterion,
            initial_labels=np.full(60, 4, dtype=int),
        )
    with pytest.raises(ValueError, match="nonempty"):
        sq.optimize_partition(
            scores,
            weights=weights,
            n_bins=4,
            criterion=criterion,
            initial_labels=np.arange(60) % 3,
        )


def test_zero_weight_rows_do_not_constrain_the_initializer() -> None:
    scores, weights = _instance(341, 60, 3, True)
    weights = np.asarray(weights).copy()
    weights[:5] = 0.0
    criterion = sq.ProfiledDOptimality((0,))
    bound = sq.efficient_score_bound(scores, interest=(0,), weights=weights, n_bins=4)
    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=4,
        criterion=criterion,
        config=sq.DExchangeConfig(seed=5, initializer_restarts=4),
        initial_labels=bound.labels,
    )
    assert bound.gap_to(result) >= -1e-9


def test_multivariate_interest_is_refused_rather_than_approximated() -> None:
    scores, weights = _instance(351, 80, 4, True)
    with pytest.raises(NotImplementedError, match="one interest column"):
        sq.efficient_score_bound(scores, interest=(0, 1), weights=weights, n_bins=4)


def test_bound_rejects_invalid_interest_and_configuration() -> None:
    scores, weights = _instance(361, 60, 3, True)
    with pytest.raises(ValueError, match="unique nonnegative"):
        sq.efficient_score_bound(scores, interest=(), weights=weights, n_bins=3)
    with pytest.raises(ValueError, match="unique nonnegative"):
        sq.efficient_score_bound(scores, interest=(-1,), weights=weights, n_bins=3)
    with pytest.raises(ValueError, match="smaller than score dimension"):
        sq.efficient_score_bound(scores, interest=(3,), weights=weights, n_bins=3)
    with pytest.raises(ValueError, match="nuisance"):
        sq.efficient_score_bound(scores[:, :1], interest=(0,), weights=weights, n_bins=3)
    with pytest.raises(TypeError, match="ScalarDPConfig"):
        sq.efficient_score_bound(
            scores,
            interest=(0,),
            weights=weights,
            n_bins=3,
            config=sq.KMeansConfig(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="n_bins"):
        sq.efficient_score_bound(scores, interest=(0,), weights=weights, n_bins=61)


def test_gap_to_refuses_a_mismatched_partition() -> None:
    scores, weights = _instance(371, 60, 3, True)
    bound = sq.efficient_score_bound(scores, interest=(0,), weights=weights, n_bins=4)
    ordinary = sq.optimize_partition(scores, weights=weights, n_bins=4)
    with pytest.raises(ValueError, match="profiled-D partition"):
        bound.gap_to(ordinary)
    other_interest = sq.optimize_partition(
        scores, weights=weights, n_bins=4, criterion=sq.ProfiledDOptimality((1,))
    )
    with pytest.raises(ValueError, match="differs from the certified"):
        bound.gap_to(other_interest)
    larger = sq.optimize_partition(
        scores, weights=weights, n_bins=6, criterion=sq.ProfiledDOptimality((0,))
    )
    with pytest.raises(ValueError, match="at most 4 cells"):
        bound.gap_to(larger)


def test_bound_also_certifies_a_coarser_partition() -> None:
    """Refinement monotonicity makes a five-cell ceiling bound a four-cell rule."""
    scores, weights = _instance(381, 100, 3, True)
    bound = sq.efficient_score_bound(scores, interest=(0,), weights=weights, n_bins=5)
    coarse = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=4,
        criterion=sq.ProfiledDOptimality((0,)),
        config=sq.DExchangeConfig(seed=2, initializer_restarts=8),
    )
    assert bound.gap_to(coarse) >= -1e-9


def test_scalar_dp_quantizer_fits_and_predicts_on_held_out_scores() -> None:
    """The previously crashing ScalarDP end-to-end path now fits and predicts."""
    rng = np.random.default_rng(391)
    scores = rng.normal(size=(240, 1))
    weights = rng.uniform(0.4, 1.6, size=240)
    held_out = rng.normal(size=(70, 1))
    result = sq.fit_quantizer(
        sq.ScoreSample(scores, weights),
        validation=sq.ScoreSample(held_out),
        n_bins=5,
        criterion=sq.DOptimality(),
        config=sq.ScalarDPConfig(),
    )
    assert result.n_bins == 5
    assert result.rank == 1
    assert result.trace.objective_label == "whitened_sse"
    assert result.validation_report is not None
    predicted = np.asarray(result.predict_scores(held_out))
    assert predicted.shape == (70,)
    assert set(np.unique(predicted)).issubset(set(range(5)))
    assert result.train_report.geometric_mean_retention > 0.8
