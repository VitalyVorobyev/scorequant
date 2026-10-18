from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest

import scorequant as sq
from scorequant._chunking import chunk_rows_for
from scorequant._exchange import ProfiledDObjective, cell_statistics
from scorequant.diagnostics import profiled_geometry_report

from ._oracles import _exact_profiled_d_move_gain


def _schur_objective(information: np.ndarray, interest: tuple[int, ...]) -> float:
    interest_array = np.asarray(interest)
    nuisance = np.asarray([index for index in range(len(information)) if index not in interest])
    a = information[np.ix_(interest_array, interest_array)]
    b = information[np.ix_(interest_array, nuisance)]
    c = information[np.ix_(nuisance, nuisance)]
    return float(np.linalg.slogdet(a - b @ np.linalg.solve(c, b.T))[1])


def test_exact_profiled_move_gain_matches_direct_recomputation() -> None:
    rng = np.random.default_rng(401)
    scores = rng.normal(size=(24, 3))
    weights = rng.uniform(0.2, 1.4, size=len(scores))
    labels = np.tile(np.arange(4), 6)
    state = ProfiledDObjective(interest=(0,), nuisance=(1, 2)).init_state(
        cell_statistics(scores, weights, labels, 4)
    )
    row, destination = 3, 2
    source = labels[row]
    predicted = _exact_profiled_d_move_gain(
        state.inverse,
        state.nuisance_inverse,
        scores[row] - np.asarray(state.means[source]),
        scores[row] - np.asarray(state.means[destination]),
        nuisance=(1, 2),
        point_weight=float(weights[row]),
        source_weight=float(np.asarray(state.weights[source])),
        destination_weight=float(np.asarray(state.weights[destination])),
    )
    moved = labels.copy()
    moved[row] = destination
    before = np.asarray(sq.information.binned_fisher_information(scores, labels, weights, n_bins=4))
    after = np.asarray(sq.information.binned_fisher_information(scores, moved, weights, n_bins=4))
    direct = _schur_objective(after, (0,)) - _schur_objective(before, (0,))
    assert predicted == pytest.approx(direct, abs=1e-11)


def test_profiled_exchange_reaches_exact_counterexample_and_does_not_compile() -> None:
    raw = np.asarray(
        [(4, -4), (-5, 2), (-1, 0), (-5, -1), (2, -2), (4, 3), (2, 4), (2, -4)],
        dtype=float,
    )
    scores = raw - raw.mean(axis=0)
    result = sq.optimize_partition(
        scores,
        n_bins=3,
        criterion=sq.ProfiledDOptimality((0,)),
        config=sq.DExchangeConfig(seed=1, initializer_restarts=32, max_iter=200),
    )
    assert result.exchange_stable
    assert np.exp(result.objective) == pytest.approx(8 * 20449 / 1920, abs=1e-10)
    assert np.all(np.diff(np.asarray(result.objective_history)) > 0)
    assert result.profiled_report is not None
    assert result.profiled_geometry is not None
    assert result.profiled_geometry.maximum_positive_violation > 0
    assert result.profiled_geometry.maximum_bound_residual <= 1e-12
    with pytest.raises(sq.RefusalError, match="no canonical inductive compilation"):
        result.compile_quantizer()


def test_soft_profiled_fit_is_an_explicit_reusable_rule() -> None:
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(180, 3))
    scores -= scores.mean(axis=0)
    validation = rng.normal(size=(80, 3))
    criterion = sq.ProfiledDOptimality((0, 1))
    result = sq.fit_quantizer(
        sq.ScoreSample(scores),
        validation=sq.ScoreSample(validation),
        n_bins=5,
        criterion=criterion,
        config=sq.SoftVoronoiConfig(seed=4, initializer_restarts=2, max_iter=20, record_every=10),
    )
    assert result.train_profiled_report is not None
    assert result.validation_profiled_report is not None
    assert result.hardening_gap is not None
    assert result.predict_scores(validation).shape == (len(validation),)
    assert np.isfinite(result.train_profiled_report.objective)


def test_efficient_score_upper_problem_is_explicit_and_block_invariant() -> None:
    rng = np.random.default_rng(23)
    scores = rng.normal(size=(120, 4))
    weights = rng.uniform(0.5, 1.5, size=len(scores))
    efficient = np.asarray(
        sq.information.efficient_scores(scores, interest=(0, 2), weights=weights)
    )
    nuisance = scores[:, [1, 3]]
    weighted_cross = np.einsum("n,ni,nj->ij", weights, efficient, nuisance)
    np.testing.assert_allclose(weighted_cross, 0, atol=1e-10)

    interest_map = np.asarray([[1.2, 0.3], [-0.1, 0.8]])
    nuisance_map = np.asarray([[0.9, -0.2], [0.4, 1.1]])
    transformed = np.column_stack(
        [
            (scores[:, [0, 2]] @ interest_map)[:, 0],
            (scores[:, [1, 3]] @ nuisance_map)[:, 0],
            (scores[:, [0, 2]] @ interest_map)[:, 1],
            (scores[:, [1, 3]] @ nuisance_map)[:, 1],
        ]
    )
    transformed_efficient = np.asarray(
        sq.information.efficient_scores(transformed, interest=(0, 2), weights=weights)
    )
    np.testing.assert_allclose(transformed_efficient, efficient @ interest_map, atol=1e-10)


def test_profiled_contract_rejects_ambiguous_or_singular_cases() -> None:
    with pytest.raises(ValueError, match="unique"):
        sq.ProfiledDOptimality((0, 0))
    scores = np.column_stack([np.linspace(-1, 1, 20)] * 2)
    with pytest.raises(ValueError, match="full-rank"):
        sq.optimize_partition(
            scores,
            n_bins=3,
            criterion=sq.ProfiledDOptimality((0,)),
        )
    with pytest.raises(ValueError, match="implements only DOptimality"):
        sq.fit_quantizer(
            sq.ScoreSample(np.random.default_rng(8).normal(size=(40, 2))),
            n_bins=3,
            criterion=sq.ProfiledDOptimality((0,)),
            config=sq.DExchangeConfig(),
        )


def test_chunked_profiled_geometry_matches_one_chunk() -> None:
    rng = np.random.default_rng(23)
    scores = rng.normal(size=(200, 3)) + 0.3
    weights = rng.uniform(0.2, 1.4, size=len(scores))
    labels = np.tile(np.arange(4), 50)
    state = ProfiledDObjective(interest=(0,), nuisance=(1, 2)).init_state(
        cell_statistics(scores, weights, labels, 4)
    )

    def report(budget: int) -> dict[str, object]:
        with patch("scorequant._chunking.WORKING_SET_BYTES", budget):
            if budget < 1 << 40:
                assert chunk_rows_for(scores, 4) < len(scores) // 4
            return asdict(
                profiled_geometry_report(
                    scores, weights, labels, state, nuisance=(1, 2), exchange_stable=False
                )
            )

    chunked = report(2048)
    whole = report(1 << 40)
    np.testing.assert_array_equal(chunked.pop("metric"), whole.pop("metric"))
    assert chunked == pytest.approx(whole, rel=1e-12)
    assert whole["evaluated_moves"] == len(scores) * 3
