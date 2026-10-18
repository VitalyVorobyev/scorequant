from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

import scorequant as sq
from scorequant.components import _scores_from_components


def _model() -> sq.LinearComponentScore:
    def signal(values: np.ndarray) -> np.ndarray:
        energy, angle = values.T
        return 0.1 + np.exp(-0.5 * ((energy - 0.4) / 0.2) ** 2) * (1 + 0.3 * angle)

    def background(values: np.ndarray) -> np.ndarray:
        energy, angle = values.T
        return 0.2 + (1 - energy) ** 2 * (1 - 0.2 * angle)

    return sq.LinearComponentScore(
        components={"signal": signal, "background": background},
        coefficients={"background": 0.4, "signal": 1.0},
        variables=("energy", "cos_theta"),
    )


def _observations(seed: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, 1, size), rng.uniform(-1, 1, size)])


def _same_partition(left: np.ndarray, right: np.ndarray) -> None:
    np.testing.assert_array_equal(left[:, None] == left[None, :], right[:, None] == right[None, :])


def test_named_model_evaluation_preserves_explicit_metadata() -> None:
    values = _observations(1, 50)
    model = _model()
    components = model.evaluate_components(values)
    assert components.shape == (50, 2)
    assert model.component_names == ("signal", "background")
    assert model.coefficients == (1.0, 0.4)
    assert model.variables == ("energy", "cos_theta")
    coefficients = np.asarray(model.coefficients)
    scores = model.score(values)
    np.testing.assert_allclose(scores, components / (components @ coefficients)[:, None])


def test_score_component_and_observation_workflows_are_equivalent() -> None:
    values = _observations(2, 300)
    weights = np.linspace(0.2, 1.8, len(values))
    model = _model()
    scores = model.score(values)
    config = sq.KMeansConfig(seed=9, solver_restarts=3)
    common = {
        "n_bins": 5,
        "criterion": sq.NormalizedTrace(),
        "config": config,
    }
    score_result = sq.fit_quantizer(sq.ScoreSample(scores, weights), **common)
    observation_result = sq.fit_quantizer(
        sq.ObservationSample(values, weights),
        provider=model,
        **common,
    )
    _same_partition(np.asarray(score_result.labels), np.asarray(observation_result.labels))
    np.testing.assert_allclose(score_result.centers, observation_result.centers)


def test_observation_prediction_keeps_score_step_explicit() -> None:
    train = _observations(3, 250)
    data = _observations(4, 80)
    provider = _model()
    result = sq.fit_quantizer(
        sq.ObservationSample(train),
        provider=provider,
        n_bins=6,
        criterion=sq.NormalizedTrace(),
        config=sq.KMeansConfig(seed=3),
    )
    assert not hasattr(result, "predict")
    labels = result.predict_scores(provider.score(data))
    assert np.bincount(np.asarray(labels), minlength=result.n_bins).sum() == len(data)
    json.dumps(result.to_dict(), allow_nan=False)


def test_validation_is_diagnostic_through_observation_provider() -> None:
    train = _observations(5, 240)
    validation = _observations(6, 120)
    provider = _model()
    config = sq.KMeansConfig(seed=5)
    without = sq.fit_quantizer(
        sq.ObservationSample(train),
        provider=provider,
        n_bins=5,
        criterion=sq.NormalizedTrace(),
        config=config,
    )
    with_validation = sq.fit_quantizer(
        sq.ObservationSample(train),
        provider=provider,
        validation=sq.ObservationSample(validation),
        n_bins=5,
        criterion=sq.NormalizedTrace(),
        config=config,
    )
    np.testing.assert_allclose(without.centers, with_validation.centers)
    assert with_validation.validation_report is not None


def test_sequence_model_uses_stable_generated_names() -> None:
    model = sq.LinearComponentScore(
        components=[lambda values: np.ones(len(values)), lambda values: 1 + values[:, 0]],
        coefficients=[1.0, 0.5],
    )
    assert model.component_names == (
        "component_0",
        "component_1",
    )


def test_component_and_model_failures_are_explicit() -> None:
    values = _observations(8, 20)
    with pytest.raises(ValueError, match="keys must match"):
        sq.LinearComponentScore(
            components={"a": lambda rows: np.ones(len(rows))},
            coefficients={"b": 1.0},
        )
    with pytest.raises(ValueError, match="must return shape"):
        sq.LinearComponentScore(
            components=[lambda rows: np.ones((len(rows), 1))], coefficients=[1.0]
        ).evaluate_components(values)
    with pytest.raises(ValueError, match="variables"):
        sq.LinearComponentScore(
            components=[lambda rows: np.ones(len(rows))],
            coefficients=[1.0],
            variables=["only_one"],
        ).evaluate_components(values)
    with pytest.raises(ValueError, match="strictly positive"):
        sq.LinearComponentScore([lambda rows: -np.ones(len(rows))], [1.0]).score(values)
    with pytest.raises(TypeError, match="callable"):
        sq.LinearComponentScore([object()], [1.0])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="provenance"):
        sq.LinearComponentScore(
            [lambda rows: np.ones(len(rows))],
            [1.0],
            provenance="exact",  # type: ignore[arg-type]
        )


def test_component_score_map_returns_canonical_numpy() -> None:
    scores = _scores_from_components(
        jnp.asarray([[1.0, 0.2], [0.5, 1.0]]),
        jnp.asarray([1.0, 0.3]),
    )
    assert isinstance(scores, np.ndarray)
