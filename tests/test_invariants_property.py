"""Property tests for the invariants AGENTS.md pins on the score-space contract.

Four invariants, drawn at random rather than fixed by hand: relabeling a hard
partition's bins, reordering its rows, uniformly rescaling its weights, and
splitting one weighted row into two rows with the same label and score. Each
report-based check runs on both the JAX and NumPy execution backends so a
divergence between them shows up here rather than only in the conformance
suite. The row-ordering solver check runs on one fixed problem shape so JAX
compiles it once.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import scorequant as sq

settings.register_profile("scorequant", deadline=None, max_examples=25, derandomize=True)
settings.load_profile("scorequant")

BACKENDS = ("jax", "numpy")
_RETENTION_TOLERANCE = {"rtol": 1e-10, "atol": 1e-12}


def _execution(backend: str) -> sq.ExecutionConfig:
    return sq.ExecutionConfig(backend=backend, precision="float64", device="cpu")


def _require_x64(backend: str) -> None:
    # Mirrors test_golden_engine._require_x64: the retention tolerance above
    # is a float64 tolerance, and CI sets JAX_ENABLE_X64=1 to earn it.
    if backend == "jax" and not jax.config.jax_enable_x64:
        pytest.skip("property fixtures compare at float64 tolerance; run under JAX_ENABLE_X64=1")


@st.composite
def problem(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Draw a random report problem: scores, weights, labels covering every bin, and n_bins."""
    n = draw(st.integers(8, 20))
    d = draw(st.integers(1, 3))
    k = draw(st.integers(d + 1, 4))  # one bin above rank keeps D feasible on a centered draw
    scores = draw(hnp.arrays(np.float64, (n, d), elements=st.floats(-3, 3, allow_nan=False)))
    weights = draw(hnp.arrays(np.float64, (n,), elements=st.floats(0.1, 2.0)))
    labels = draw(hnp.arrays(np.int64, (n,), elements=st.integers(0, k - 1)))
    assume(np.linalg.matrix_rank(scores) == d)
    assume(len(set(labels.tolist())) == k)
    # Reject a binned-information rank decision sitting on the float64 rounding
    # boundary: near the cutoff, a scale or summation-order change (uniform
    # weight scaling, row permutation, split duplication) can flip which
    # eigenvalues clear the relative rank threshold by rounding alone, which
    # would break the exact-arithmetic invariant below without violating it.
    probe = sq.information_report(scores, labels, weights, n_bins=k, execution=_execution("numpy"))
    assume(probe.effective_rank == d)
    if probe.retained_eigenvalues.size:
        assume(np.min(probe.retained_eigenvalues) > 1e-6 * np.max(probe.retained_eigenvalues))
    return scores, weights, labels, k


@st.composite
def relabeling_problem(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    scores, weights, labels, k = draw(problem())
    permutation = np.asarray(draw(st.permutations(range(k))), dtype=np.int64)
    return scores, weights, labels, k, permutation


@st.composite
def row_order_problem(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    scores, weights, labels, k = draw(problem())
    order = np.asarray(draw(st.permutations(range(scores.shape[0]))), dtype=np.int64)
    return scores, weights, labels, k, order


@st.composite
def weight_scaling_problem(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    scores, weights, labels, k = draw(problem())
    factor = draw(st.floats(0.5, 20.0))
    return scores, weights, labels, k, factor


@st.composite
def split_weight_problem(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    scores, weights, labels, k = draw(problem())
    row = draw(st.integers(0, scores.shape[0] - 1))
    split = draw(st.floats(0.1, 0.9))
    return scores, weights, labels, k, row, split


@st.composite
def solver_row_order_problem(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fix n=12, d=2 so the JAX solver path compiles a single shape across examples."""
    n, d, n_bins = 12, 2, 3
    scores = draw(hnp.arrays(np.float64, (n, d), elements=st.floats(-3, 3, allow_nan=False)))
    weights = draw(hnp.arrays(np.float64, (n,), elements=st.floats(0.1, 2.0)))
    assume(np.linalg.matrix_rank(scores) == d)
    # All weights are positive, so distinct rows are exactly the effective rows
    # the D-exchange solver sees after duplicate collapse; too few and n_bins
    # cannot be satisfied.
    assume(np.unique(scores, axis=0).shape[0] >= n_bins)
    order = np.asarray(draw(st.permutations(range(n))), dtype=np.int64)
    return scores, weights, order


@pytest.mark.parametrize("backend", BACKENDS)
@given(drawn=relabeling_problem())
def test_bin_relabeling_preserves_retention(
    backend: str, drawn: tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]
) -> None:
    _require_x64(backend)
    scores, weights, labels, k, permutation = drawn
    execution = _execution(backend)
    base = sq.information_report(scores, labels, weights, n_bins=k, execution=execution)
    relabeled = sq.information_report(
        scores, permutation[labels], weights, n_bins=k, execution=execution
    )
    np.testing.assert_allclose(
        relabeled.geometric_mean_retention, base.geometric_mean_retention, **_RETENTION_TOLERANCE
    )
    np.testing.assert_allclose(
        relabeled.logdet_retention, base.logdet_retention, **_RETENTION_TOLERANCE
    )
    np.testing.assert_allclose(
        relabeled.retained_eigenvalues, base.retained_eigenvalues, **_RETENTION_TOLERANCE
    )
    # relabeled.bin_weights[permutation[i]] holds what base.bin_weights[i] held, so gathering
    # relabeled.bin_weights at `permutation` reproduces base.bin_weights in original bin order.
    np.testing.assert_allclose(
        relabeled.bin_weights[permutation], base.bin_weights, **_RETENTION_TOLERANCE
    )


@pytest.mark.parametrize("backend", BACKENDS)
@given(drawn=row_order_problem())
def test_row_ordering_preserves_report_retention(
    backend: str, drawn: tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]
) -> None:
    _require_x64(backend)
    scores, weights, labels, k, order = drawn
    execution = _execution(backend)
    base = sq.information_report(scores, labels, weights, n_bins=k, execution=execution)
    permuted = sq.information_report(
        scores[order], labels[order], weights[order], n_bins=k, execution=execution
    )
    np.testing.assert_allclose(
        permuted.geometric_mean_retention, base.geometric_mean_retention, **_RETENTION_TOLERANCE
    )
    np.testing.assert_allclose(
        permuted.logdet_retention, base.logdet_retention, **_RETENTION_TOLERANCE
    )
    np.testing.assert_allclose(
        permuted.retained_eigenvalues, base.retained_eigenvalues, **_RETENTION_TOLERANCE
    )


@pytest.mark.parametrize("backend", BACKENDS)
@given(drawn=solver_row_order_problem())
def test_row_ordering_preserves_the_solver_partition(
    backend: str, drawn: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    _require_x64(backend)
    scores, weights, order = drawn
    execution = _execution(backend)
    config = sq.DExchangeConfig(solver_restarts=1, batch_moves=False, max_iter=None, seed=0)
    base = sq.optimize_partition(
        scores, weights=weights, n_bins=3, config=config, execution=execution
    )
    permuted = sq.optimize_partition(
        scores[order], weights=weights[order], n_bins=3, config=config, execution=execution
    )
    assert permuted.objective == pytest.approx(base.objective, abs=1e-9)
    base_labels = np.asarray(base.labels)
    permuted_labels = np.asarray(permuted.labels)
    # permuted_labels[i] == permuted_labels[j] iff base_labels[order[i]] == base_labels[order[j]]:
    # the row permutation is a relabeling of the same induced partition.
    reordered_base_labels = base_labels[order]
    same_in_permuted = permuted_labels[:, None] == permuted_labels[None, :]
    same_in_base = reordered_base_labels[:, None] == reordered_base_labels[None, :]
    np.testing.assert_array_equal(same_in_permuted, same_in_base)


@pytest.mark.parametrize("backend", BACKENDS)
@given(drawn=weight_scaling_problem())
def test_uniform_weight_scaling_preserves_retention(
    backend: str, drawn: tuple[np.ndarray, np.ndarray, np.ndarray, int, float]
) -> None:
    _require_x64(backend)
    scores, weights, labels, k, factor = drawn
    execution = _execution(backend)
    base = sq.information_report(scores, labels, weights, n_bins=k, execution=execution)
    scaled = sq.information_report(scores, labels, weights * factor, n_bins=k, execution=execution)
    np.testing.assert_allclose(
        scaled.geometric_mean_retention, base.geometric_mean_retention, **_RETENTION_TOLERANCE
    )
    np.testing.assert_allclose(
        scaled.logdet_retention, base.logdet_retention, **_RETENTION_TOLERANCE
    )
    np.testing.assert_allclose(
        scaled.retained_eigenvalues, base.retained_eigenvalues, **_RETENTION_TOLERANCE
    )
    np.testing.assert_allclose(
        scaled.fisher_binned, base.fisher_binned * factor, **_RETENTION_TOLERANCE
    )


@pytest.mark.parametrize("backend", BACKENDS)
@given(drawn=split_weight_problem())
def test_split_weight_duplication_preserves_retention(
    backend: str, drawn: tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]
) -> None:
    _require_x64(backend)
    scores, weights, labels, k, row, split = drawn
    execution = _execution(backend)
    base = sq.information_report(scores, labels, weights, n_bins=k, execution=execution)

    split_scores = np.concatenate([scores, scores[row : row + 1]], axis=0)
    split_labels = np.concatenate([labels, labels[row : row + 1]], axis=0)
    split_weights = np.concatenate([weights, weights[row : row + 1] * split], axis=0)
    split_weights[row] = weights[row] * (1.0 - split)

    duplicated = sq.information_report(
        split_scores, split_labels, split_weights, n_bins=k, execution=execution
    )
    np.testing.assert_allclose(
        duplicated.geometric_mean_retention,
        base.geometric_mean_retention,
        **_RETENTION_TOLERANCE,
    )
    np.testing.assert_allclose(
        duplicated.logdet_retention, base.logdet_retention, **_RETENTION_TOLERANCE
    )
    np.testing.assert_allclose(
        duplicated.retained_eigenvalues, base.retained_eigenvalues, **_RETENTION_TOLERANCE
    )
    np.testing.assert_allclose(duplicated.fisher_binned, base.fisher_binned, **_RETENTION_TOLERANCE)
