from __future__ import annotations

from dataclasses import fields, is_dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import scorequant as sq
from scorequant._execution import use_execution
from scorequant.config import QuantizerConfig
from scorequant.criteria import Criterion
from scorequant.solvers.soft import soft_objective_and_center_gradient


def _execution(backend: str) -> sq.ExecutionConfig:
    return sq.ExecutionConfig(backend=backend, precision="float64", device="cpu")


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(102)
    scores = np.concatenate(
        [
            rng.normal((-1.2, 0.2), (0.35, 0.25), size=(18, 2)),
            rng.normal((0.9, -0.1), (0.4, 0.3), size=(18, 2)),
        ]
    )
    weights = rng.uniform(0.2, 1.8, scores.shape[0])
    return scores, weights


def _assert_public_arrays_are_numpy(value: object) -> None:
    if isinstance(value, np.ndarray):
        return
    assert not type(value).__module__.startswith("jax")
    if is_dataclass(value) and not isinstance(value, type):
        for record_field in fields(value):
            _assert_public_arrays_are_numpy(getattr(value, record_field.name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_public_arrays_are_numpy(item)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_information_partition_and_prediction_conform(backend: str) -> None:
    scores, weights = _fixture()
    execution = _execution(backend)
    labels = np.repeat(np.arange(3), 12)

    fisher = sq.fisher_information(scores, weights, execution=execution)
    report = sq.information_report(
        scores,
        labels,
        weights,
        n_bins=3,
        execution=execution,
    )
    uncertainty = sq.retention_uncertainty(scores, labels, n_bins=3, execution=execution)
    partition = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=3,
        config=sq.DExchangeConfig(initializer_restarts=2),
        execution=execution,
    )
    quantizer = partition.compile_quantizer()

    assert isinstance(fisher, np.ndarray)
    assert partition.execution.backend == backend
    assert quantizer.execution == partition.execution
    assert isinstance(quantizer.predict_scores(scores), np.ndarray)
    assert uncertainty.status == "ok"
    _assert_public_arrays_are_numpy(report)
    _assert_public_arrays_are_numpy(uncertainty)
    _assert_public_arrays_are_numpy(partition)
    _assert_public_arrays_are_numpy(quantizer)


@pytest.mark.parametrize(
    ("config", "criterion"),
    [
        (sq.DExchangeConfig(initializer_restarts=2, max_scans=30), sq.DOptimality()),
        (
            sq.MahalanobisLloydConfig(initializer_restarts=2, max_iter=20),
            sq.DOptimality(),
        ),
        (sq.KMeansConfig(solver_restarts=2, max_iter=20), sq.NormalizedTrace()),
        (sq.ScalarDPConfig(max_rows=100), sq.DOptimality()),
        (
            sq.SoftVoronoiConfig(
                initializer_restarts=1,
                kmeans_max_iter=12,
                max_steps=12,
                record_every=6,
            ),
            sq.DOptimality(),
        ),
        (
            sq.SoftVoronoiConfig(
                initializer_restarts=1,
                kmeans_max_iter=12,
                max_steps=12,
                record_every=6,
            ),
            sq.ProfiledDOptimality((0,)),
        ),
    ],
)
@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_quantizer_solver_matrix_runs(
    backend: str,
    config: QuantizerConfig,
    criterion: Criterion,
) -> None:
    scores, weights = _fixture()
    if isinstance(config, sq.ScalarDPConfig):
        scores = scores[:, :1]
    result = sq.fit_quantizer(
        sq.ScoreSample(scores, weights),
        n_bins=3,
        criterion=criterion,
        config=config,
        execution=_execution(backend),
    )
    assert result.labels.shape == (scores.shape[0],)
    assert np.unique(result.labels).size == 3
    assert np.isfinite(result.train_report.geometric_mean_retention)


@pytest.mark.parametrize(
    "config",
    [
        sq.DExchangeConfig(initializer_restarts=2, max_scans=30),
        sq.MahalanobisLloydConfig(initializer_restarts=2, max_iter=20),
    ],
)
@pytest.mark.parametrize("criterion", [sq.DOptimality(), sq.ProfiledDOptimality((0,))])
@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_partition_solver_matrix_runs(
    backend: str,
    criterion: sq.DOptimality | sq.ProfiledDOptimality,
    config: sq.DExchangeConfig | sq.MahalanobisLloydConfig,
) -> None:
    scores, weights = _fixture()
    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=3,
        criterion=criterion,
        config=config,
        execution=_execution(backend),
    )
    assert result.execution.backend == backend
    assert np.unique(result.labels).size == 3
    assert result.exchange_stable


def test_hard_backend_parity() -> None:
    scores, weights = _fixture()
    results = {
        backend: sq.optimize_partition(
            scores,
            weights=weights,
            n_bins=3,
            config=sq.DExchangeConfig(initializer_restarts=2),
            execution=_execution(backend),
        )
        for backend in ("jax", "numpy")
    }
    np.testing.assert_allclose(
        results["jax"].information_partitioned,
        results["numpy"].information_partitioned,
        rtol=1e-10,
        atol=1e-12,
    )
    assert results["jax"].objective == pytest.approx(results["numpy"].objective, abs=1e-12)
    uncertainties = {
        backend: sq.retention_uncertainty(
            scores, np.repeat(np.arange(3), 12), execution=_execution(backend)
        )
        for backend in ("jax", "numpy")
    }
    assert uncertainties["jax"].status == uncertainties["numpy"].status == "ok"
    assert uncertainties["jax"].estimate == pytest.approx(
        uncertainties["numpy"].estimate, rel=1e-12
    )
    assert uncertainties["jax"].standard_error == pytest.approx(
        uncertainties["numpy"].standard_error, rel=1e-10
    )


@pytest.mark.parametrize("criterion", [sq.DOptimality(), sq.ProfiledDOptimality((0,))])
def test_analytic_soft_gradient_matches_autodiff_and_finite_difference(
    criterion: sq.DOptimality | sq.ProfiledDOptimality,
) -> None:
    rng = np.random.default_rng(11)
    points = rng.normal(size=(8, 2))
    objective_scores = rng.normal(size=(8, 3))
    weights = rng.uniform(0.3, 1.6, size=8)
    centers = rng.normal(size=(3, 2))
    temperature = 0.75

    with use_execution(_execution("jax")):
        _, analytic = soft_objective_and_center_gradient(
            jnp.asarray(points),
            jnp.asarray(objective_scores),
            jnp.asarray(weights),
            jnp.asarray(centers),
            temperature,
            criterion,
        )

    def oracle(center_values: jax.Array) -> jax.Array:
        point_values = jnp.asarray(points)
        score_values = jnp.asarray(objective_scores)
        weight_values = jnp.asarray(weights)
        logits = -jnp.sum(
            (point_values[:, None, :] - center_values[None, :, :]) ** 2,
            axis=2,
        ) / (2 * temperature**2)
        responsibilities = jax.nn.softmax(logits, axis=1)
        weighted = weight_values[:, None] * responsibilities
        occupancy = jnp.sum(weighted, axis=0)
        moments = weighted.T @ score_values
        fisher = jnp.einsum("bp,bq,b->pq", moments, moments, 1 / occupancy)
        fisher = 0.5 * (fisher + fisher.T)
        if isinstance(criterion, sq.DOptimality):
            return -jnp.linalg.slogdet(fisher)[1]
        nuisance = jnp.asarray([1, 2])
        nuisance_fisher = fisher[jnp.ix_(nuisance, nuisance)]
        return -(jnp.linalg.slogdet(fisher)[1] - jnp.linalg.slogdet(nuisance_fisher)[1])

    autodiff = jax.grad(oracle)(jnp.asarray(centers))
    np.testing.assert_allclose(analytic, autodiff, rtol=1e-10, atol=1e-12)

    epsilon = 1e-6
    finite_difference = np.empty_like(centers)
    with use_execution(_execution("numpy")):
        for index in np.ndindex(centers.shape):
            plus = centers.copy()
            minus = centers.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_loss, _ = soft_objective_and_center_gradient(
                points, objective_scores, weights, plus, temperature, criterion
            )
            minus_loss, _ = soft_objective_and_center_gradient(
                points, objective_scores, weights, minus, temperature, criterion
            )
            finite_difference[index] = (float(plus_loss) - float(minus_loss)) / (2 * epsilon)
    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-5, atol=1e-6)


def test_execution_validation_precedes_computation() -> None:
    with pytest.raises(ValueError, match="NumPy backend supports only"):
        sq.ExecutionConfig(backend="numpy", device="gpu")
    with pytest.raises(TypeError, match="ExecutionConfig"):
        sq.fisher_information([[1.0]], execution="numpy")  # type: ignore[arg-type]
    if not bool(jax.config.x64_enabled):
        with pytest.raises(RuntimeError, match="JAX_ENABLE_X64"):
            sq.fisher_information(
                [[1.0]],
                execution=sq.ExecutionConfig(backend="jax", precision="float64"),
            )


def _cells(labels: np.ndarray) -> frozenset[frozenset[int]]:
    """Return the partition itself, free of cell numbering.

    Bin relabeling is an invariant of the library, so two backends that induce
    the same grouping have agreed even when their cell indices differ.
    """
    array = np.asarray(labels)
    return frozenset(
        frozenset(np.flatnonzero(array == value).tolist()) for value in np.unique(array)
    )


# The M9 gate promises cross-backend agreement for the whole declared solver
# matrix, not merely that each solver runs. These are the tests that hold it to
# that: `test_quantizer_solver_matrix_runs` above is a smoke test and cannot
# catch a backend that silently converges somewhere else.
_QUANTIZER_MATRIX: list[tuple[QuantizerConfig, Criterion]] = [
    (sq.DExchangeConfig(initializer_restarts=2, max_scans=30), sq.DOptimality()),
    (sq.MahalanobisLloydConfig(initializer_restarts=2, max_iter=20), sq.DOptimality()),
    (sq.KMeansConfig(solver_restarts=2, max_iter=20), sq.NormalizedTrace()),
    (sq.ScalarDPConfig(max_rows=100), sq.DOptimality()),
    (
        sq.SoftVoronoiConfig(
            initializer_restarts=1, kmeans_max_iter=12, max_steps=12, record_every=6
        ),
        sq.DOptimality(),
    ),
    (
        sq.SoftVoronoiConfig(
            initializer_restarts=1, kmeans_max_iter=12, max_steps=12, record_every=6
        ),
        sq.ProfiledDOptimality((0,)),
    ),
]


@pytest.mark.parametrize(("config", "criterion"), _QUANTIZER_MATRIX)
def test_quantizer_solver_matrix_backend_parity(
    config: QuantizerConfig, criterion: Criterion
) -> None:
    scores, weights = _fixture()
    if isinstance(config, sq.ScalarDPConfig):
        scores = scores[:, :1]
    results = {
        backend: sq.fit_quantizer(
            sq.ScoreSample(scores, weights),
            n_bins=3,
            criterion=criterion,
            config=config,
            execution=_execution(backend),
        )
        for backend in ("jax", "numpy")
    }
    # The hard solvers are exact and must agree to float64 round-off. Soft
    # assignment anneals through Adam, so it is held to the schedule's own
    # tolerance rather than to round-off.
    soft = isinstance(config, sq.SoftVoronoiConfig)
    assert _cells(results["jax"].labels) == _cells(results["numpy"].labels)
    np.testing.assert_allclose(
        results["jax"].train_report.geometric_mean_retention,
        results["numpy"].train_report.geometric_mean_retention,
        rtol=1e-4 if soft else 1e-10,
        atol=1e-4 if soft else 1e-12,
    )


@pytest.mark.parametrize(
    "config",
    [
        sq.DExchangeConfig(initializer_restarts=2, max_scans=30),
        sq.MahalanobisLloydConfig(initializer_restarts=2, max_iter=20),
    ],
)
@pytest.mark.parametrize("criterion", [sq.DOptimality(), sq.ProfiledDOptimality((0,))])
def test_partition_solver_matrix_backend_parity(
    criterion: sq.DOptimality | sq.ProfiledDOptimality,
    config: sq.DExchangeConfig | sq.MahalanobisLloydConfig,
) -> None:
    scores, weights = _fixture()
    results = {
        backend: sq.optimize_partition(
            scores,
            weights=weights,
            n_bins=3,
            criterion=criterion,
            config=config,
            execution=_execution(backend),
        )
        for backend in ("jax", "numpy")
    }
    assert _cells(results["jax"].labels) == _cells(results["numpy"].labels)
    np.testing.assert_allclose(
        results["jax"].information_partitioned,
        results["numpy"].information_partitioned,
        rtol=1e-10,
        atol=1e-12,
    )
    assert results["jax"].objective == pytest.approx(results["numpy"].objective, abs=1e-12)


def test_certificate_backend_parity() -> None:
    scores, weights = _fixture()
    certificates = {}
    for backend in ("jax", "numpy"):
        partition = sq.optimize_partition(
            scores[:12],
            weights=weights[:12],
            n_bins=3,
            config=sq.DExchangeConfig(initializer_restarts=2),
            execution=_execution(backend),
        )
        certificates[backend] = sq.certify_partition(
            scores[:12],
            weights=weights[:12],
            n_bins=3,
            incumbent=partition.labels,
            execution=_execution(backend),
        )
    assert certificates["jax"].status == certificates["numpy"].status
    assert certificates["jax"].objective == pytest.approx(
        certificates["numpy"].objective, abs=1e-12
    )
