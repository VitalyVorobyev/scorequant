from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import scorequant
from tests._fit import fit_test_quantizer


def test_hard_information_loss_identity_and_psd() -> None:
    scores = jnp.asarray([[-1.0, 0.2], [-0.8, 0.1], [0.7, 0.4], [1.2, 0.5]])
    weights = jnp.asarray([1.0, 2.0, 1.5, 0.5])
    labels = jnp.asarray([0, 0, 1, 1])
    full = scorequant.fisher_information(scores, weights)
    binned = scorequant.binned_fisher_information(scores, labels, weights, n_bins=2)
    residual = np.asarray(full - binned)
    assert np.linalg.eigvalsh(residual).min() >= -1e-6

    direct = np.zeros((2, 2))
    for bin_index in range(2):
        mask = np.asarray(labels) == bin_index
        mean = np.average(np.asarray(scores)[mask], axis=0, weights=np.asarray(weights)[mask])
        differences = np.asarray(scores)[mask] - mean
        direct += np.einsum("n,np,nq->pq", np.asarray(weights)[mask], differences, differences)
    np.testing.assert_allclose(residual, direct, rtol=1e-6, atol=1e-6)


def test_fractional_information_is_bounded() -> None:
    scores = jnp.asarray([[-1.0], [0.0], [1.0], [2.0]])
    responsibilities = jnp.asarray([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.05, 0.95]])
    full = scorequant.fisher_information(scores)
    soft = scorequant.fractional_fisher_information(scores, responsibilities)
    assert float((full - soft)[0, 0]) >= -1e-6


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_psd_loss_property_across_seeded_samples(seed: int) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(30, 3))
    weights = rng.lognormal(size=30)
    labels = rng.integers(0, 5, size=30)
    full = scorequant.fisher_information(scores, weights)
    binned = scorequant.binned_fisher_information(scores, labels, weights, n_bins=5)
    scale = max(float(np.linalg.eigvalsh(np.asarray(full)).max()), 1.0)
    assert np.linalg.eigvalsh(np.asarray(full - binned)).min() >= -1e-5 * scale


def test_weight_splitting_and_global_scaling_preserve_normalized_report() -> None:
    scores = jnp.asarray([[-1.0], [0.5], [2.0]])
    weights = jnp.asarray([2.0, 1.0, 3.0])
    labels = jnp.asarray([0, 0, 1])
    report = scorequant.information_report(scores, labels, weights, n_bins=2)

    split_scores = jnp.asarray([[-1.0], [-1.0], [0.5], [2.0]])
    split_weights = jnp.asarray([0.75, 1.25, 1.0, 3.0])
    split_labels = jnp.asarray([0, 0, 0, 1])
    split = scorequant.information_report(split_scores, split_labels, split_weights, n_bins=2)
    scaled = scorequant.information_report(scores, labels, weights * 11, n_bins=2)
    np.testing.assert_allclose(report.retained_matrix, split.retained_matrix, atol=1e-6)
    np.testing.assert_allclose(report.retained_matrix, scaled.retained_matrix, atol=1e-6)


def test_report_reuses_hard_statistics_and_ignores_zero_weight_rows() -> None:
    scores = jnp.asarray([[-1.0], [100.0], [0.5], [2.0]])
    weights = jnp.asarray([1.0, 0.0, 2.0, 3.0])
    # An out-of-range label is harmless for a row excluded from the measure.
    labels = jnp.asarray([0, 99, 1, 1])

    report = scorequant.information_report(scores, labels, weights, n_bins=2)
    direct = scorequant.binned_fisher_information(scores, labels, weights, n_bins=2)

    np.testing.assert_allclose(report.fisher_binned, direct)
    np.testing.assert_allclose(report.bin_weights, [1.0, 5.0])
    np.testing.assert_array_equal(report.bin_counts, [1, 2])
    np.testing.assert_allclose(report.bin_effective_sample_sizes, [1.0, 25.0 / 13.0])


def test_event_order_bin_relabeling_and_partition_limits() -> None:
    scores = jnp.asarray([[-2.0, 0.5], [-0.5, 1.0], [1.0, -0.2], [2.0, 0.7]])
    weights = jnp.asarray([1.0, 2.0, 0.5, 3.0])
    labels = jnp.asarray([0, 0, 1, 2])
    expected = scorequant.binned_fisher_information(scores, labels, weights, n_bins=3)
    order = jnp.asarray([2, 0, 3, 1])
    reordered = scorequant.binned_fisher_information(
        scores[order], labels[order], weights[order], n_bins=3
    )
    relabeled = scorequant.binned_fisher_information(
        scores, jnp.asarray([2, 2, 0, 1]), weights, n_bins=3
    )
    np.testing.assert_allclose(expected, reordered, atol=1e-6)
    np.testing.assert_allclose(expected, relabeled, atol=1e-6)

    one_bin = scorequant.binned_fisher_information(
        scores, jnp.zeros(4, dtype=jnp.int32), weights, n_bins=1
    )
    mean = np.average(np.asarray(scores), axis=0, weights=np.asarray(weights))
    np.testing.assert_allclose(one_bin, np.sum(np.asarray(weights)) * np.outer(mean, mean))
    unique = scorequant.binned_fisher_information(scores, jnp.arange(4), weights, n_bins=4)
    np.testing.assert_allclose(unique, scorequant.fisher_information(scores, weights), atol=1e-6)


def test_rank_projection_and_json_are_explicit() -> None:
    scores = jnp.asarray([[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]])
    result = fit_test_quantizer(scores, n_bins=2)
    assert result.transform.rank == 1
    assert result.transform.dropped_directions == 1
    json.dumps(result.to_dict(), allow_nan=False)


def test_component_adapter() -> None:
    components = jnp.asarray([[1.0, 2.0], [3.0, 1.0]])
    coefficients = jnp.asarray([2.0, 0.5])
    expected = np.asarray(components) / (np.asarray(components) @ np.asarray(coefficients))[:, None]
    np.testing.assert_allclose(
        scorequant.scores_from_components(components, coefficients), expected
    )
    with pytest.raises(ValueError, match="strictly positive"):
        scorequant.scores_from_components(jnp.zeros((2, 2)), coefficients)


def test_component_adapter_allows_signed_basis_terms() -> None:
    components = jnp.asarray([[2.0, -1.0], [1.0, 0.5], [0.5, -0.25]])
    coefficients = jnp.asarray([1.0, -0.2])
    density = np.asarray(components @ coefficients)
    scores = scorequant.scores_from_components(components, coefficients)
    np.testing.assert_allclose(scores, np.asarray(components) / density[:, None])


@pytest.mark.parametrize("bad_weights", [[1.0, -1.0], [0.0, 0.0], [1.0, np.nan]])
def test_invalid_weights_fail(bad_weights: list[float]) -> None:
    with pytest.raises(ValueError):
        scorequant.fisher_information(jnp.asarray([[0.0], [1.0]]), bad_weights)


def _nearly_collinear_cells() -> tuple[np.ndarray, np.ndarray]:
    """Three cells whose means are collinear up to a 1e-6 bend.

    Each cell holds two rows placed symmetrically about its mean, so the
    unbinned information is regular while the binned information has a
    relative eigenvalue near 1e-14: positive, so a ``slogdet`` sign and NumPy's
    default ``matrix_rank`` call it regular, yet far below the 1e-10 rank
    threshold.
    """
    t = np.array([1.0, 2.0, 4.0])
    means = np.stack([t, 3.0 * t + 1e-6 * t**2], axis=1)
    spread = np.array([0.5, -0.25])
    scores = np.concatenate([means + spread, means - spread])
    labels = np.concatenate([np.arange(3), np.arange(3)])
    return scores, labels


def test_nearly_singular_profiled_report_uses_the_shared_rank_rule() -> None:
    """A deficient binned matrix is singular even when its log-det sign is positive."""
    scores, labels = _nearly_collinear_cells()
    binned = np.asarray(scorequant.binned_fisher_information(scores, labels, n_bins=3))
    schur = binned[:1, :1] - np.outer(binned[0, 1], binned[0, 1]) / binned[1, 1]
    # The rules being replaced call this state regular.
    assert np.linalg.slogdet(binned)[0] > 0
    assert np.linalg.slogdet(schur)[0] > 0
    assert np.linalg.matrix_rank(schur) == 1

    report = scorequant.profiled_information_report(scores, labels, interest=(0,), n_bins=3)
    assert report.objective == float("-inf")
    assert report.geometric_mean_retention == 0.0
    assert report.interest_rank == 0
    assert report.nuisance_rank == 1
    # The exchange engine reaches the same verdict on the same labels.
    with pytest.raises(scorequant.RefusalError, match="singular"):
        scorequant.exchange_stability_report(
            scores, labels, criterion=scorequant.ProfiledDOptimality((0,))
        )


def test_nearly_singular_information_report_agrees_with_retention_uncertainty() -> None:
    scores, labels = _nearly_collinear_cells()
    report = scorequant.information_report(scores, labels, n_bins=3)
    assert np.all(np.asarray(report.retained_eigenvalues) > 0)
    assert report.geometric_mean_retention == 0.0
    assert report.logdet_retention == float("-inf")
    uncertainty = scorequant.retention_uncertainty(scores, labels)
    assert uncertainty.status == "singular_retained_information"
    assert uncertainty.estimate == 0.0


def test_efficient_scores_refuse_a_nearly_singular_nuisance_block() -> None:
    rng = np.random.default_rng(31)
    interest = rng.normal(size=(200, 1))
    nuisance = rng.normal(size=(200, 1))
    scores = np.hstack([interest, nuisance, nuisance + 1e-7 * rng.normal(size=(200, 1))])
    block = np.asarray(scorequant.fisher_information(scores))[1:, 1:]
    assert np.linalg.slogdet(block)[0] > 0
    with pytest.raises(scorequant.ContractError, match="nonsingular nuisance"):
        scorequant.efficient_scores(scores, interest=(0,))


def test_rank_rtol_does_not_judge_the_retained_matrix() -> None:
    """``rank_rtol`` selects full-data directions; retained singularity uses the dtype default."""
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(2000, 2)) + 0.3
    labels = np.digitize(scores[:, 0] + 0.2 * scores[:, 1], [-0.2, 0.8])
    default = scorequant.information_report(scores, labels, n_bins=3)
    eigenvalues = np.asarray(default.retained_eigenvalues)
    assert eigenvalues.min() < 0.1 * eigenvalues.max()
    loose = scorequant.information_report(scores, labels, n_bins=3, rank_rtol=0.1)
    assert loose.geometric_mean_retention == default.geometric_mean_retention > 0.1
    uncertainty = scorequant.retention_uncertainty(scores, labels, rank_rtol=0.1)
    assert uncertainty.status == "ok"
    assert uncertainty.estimate == pytest.approx(default.geometric_mean_retention, rel=1e-12)
