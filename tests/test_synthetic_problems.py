from __future__ import annotations

import numpy as np
import pytest

import scorequant as sq
from examples.synthetic_problems import (
    SignalBackgroundProblem,
    SyntheticProblem,
    separable_1d_projection,
    signal_background_shape,
    two_parameter_gaussian_mixture,
)


def test_two_parameter_gaussian_mixture_shapes_and_determinism() -> None:
    sizes = (300, 200, 400)
    problem = two_parameter_gaussian_mixture(seed=1, sizes=sizes, n_bins=9)
    again = two_parameter_gaussian_mixture(seed=1, sizes=sizes, n_bins=9)
    assert isinstance(problem, SyntheticProblem)
    assert problem.name == "two_parameter_gaussian_mixture"
    for split, size in zip((problem.train, problem.validation, problem.test), sizes, strict=True):
        assert split.observations.shape == (size, 2)
        assert split.scores.shape == (size, 2)
        assert split.weights.shape == (size,)
        assert np.all(np.isfinite(split.observations))
        assert np.all(np.isfinite(split.scores))
        assert np.all(np.isfinite(split.weights))
        assert np.all(split.weights > 0)
    np.testing.assert_array_equal(problem.train.scores, again.train.scores)
    np.testing.assert_array_equal(problem.train.observations, again.train.observations)


def test_two_parameter_gaussian_mixture_overlap_changes_scores() -> None:
    tight = two_parameter_gaussian_mixture(seed=2, sizes=(200, 100, 100), separation=0.2)
    wide = two_parameter_gaussian_mixture(seed=2, sizes=(200, 100, 100), separation=3.0)
    # Same seed drives identical observations; wider separation should change
    # the induced component scores at those observations.
    assert not np.allclose(tight.train.scores, wide.train.scores)


def test_two_parameter_gaussian_mixture_is_registered_and_reusable_by_run() -> None:
    from examples.synthetic_problems import PROBLEMS

    assert "two_parameter_gaussian_mixture" in PROBLEMS
    problem = PROBLEMS["two_parameter_gaussian_mixture"]()
    assert isinstance(problem, SyntheticProblem)


def test_signal_background_shape_score_layout_and_determinism() -> None:
    sizes = (300, 200, 400)
    problem = signal_background_shape(seed=3, sizes=sizes, background_rates=(1.0, 3.0, 6.0))
    again = signal_background_shape(seed=3, sizes=sizes, background_rates=(1.0, 3.0, 6.0))
    assert isinstance(problem, SignalBackgroundProblem)
    assert problem.interest == (0,)
    assert problem.nuisance == (1, 2, 3)
    assert problem.coefficients.shape == (4,)
    assert problem.component_names == ("signal", "background_1", "background_2", "background_3")
    np.testing.assert_allclose(problem.coefficients.sum(), 1.0)

    for split, size in zip((problem.train, problem.validation, problem.test), sizes, strict=True):
        assert split.observations.shape == (size, 1)
        assert split.scores.shape == (size, 4)
        assert split.weights.shape == (size,)
        assert np.all(np.isfinite(split.scores))
        assert np.all(split.weights > 0)
    np.testing.assert_array_equal(problem.train.scores, again.train.scores)


def test_signal_background_shape_exposes_exact_component_densities() -> None:
    problem = signal_background_shape(seed=4, sizes=(50, 50, 50))
    x = np.linspace(0.0, 1.0, 25)
    components = problem.evaluate_components(x)
    assert components.shape == (25, 1 + len(problem.nuisance))
    assert np.all(components > 0)

    intensity = problem.intensity(x)
    np.testing.assert_allclose(intensity, components @ problem.coefficients)

    # The intensity map on the exact densities reproduces the stored scores.
    rebuilt = np.asarray(
        sq.ratios.IntensityParameterization(problem.coefficients).scores(components)
    )
    direct = np.asarray(
        sq.ratios.IntensityParameterization(problem.coefficients).scores(
            problem.evaluate_components(x)
        )
    )
    np.testing.assert_allclose(rebuilt, direct)


def test_signal_background_shape_works_with_profiled_d_optimality() -> None:
    problem = signal_background_shape(
        seed=5, sizes=(600, 200, 200), n_bins=4, background_rates=(1.0, 3.0)
    )
    result = sq.optimize_partition(
        problem.train.scores,
        weights=problem.train.weights,
        n_bins=4,
        criterion=sq.ProfiledDOptimality(problem.interest),
        config=sq.DExchangeConfig(seed=0, initializer_restarts=4),
    )
    assert result.profiled_report is not None


def test_signal_background_shape_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        signal_background_shape(signal_fraction=0.0)
    with pytest.raises(ValueError):
        signal_background_shape(signal_fraction=1.0)
    with pytest.raises(ValueError):
        signal_background_shape(background_rates=())


def test_separable_1d_projection_shape_and_determinism() -> None:
    rng = np.random.default_rng(6)
    scores = rng.normal(size=(200, 3)) * np.array([2.0, 0.5, 0.1])
    weights = rng.uniform(0.5, 1.5, size=200)
    first = separable_1d_projection(scores, weights)
    second = separable_1d_projection(scores, weights)
    assert first.shape == (200, 1)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))


def test_separable_1d_projection_does_not_center_the_output() -> None:
    # The projection direction can be estimated from centered data, but the
    # returned coordinates must come from the original (uncentered) scores.
    scores = np.array([[10.0], [11.0], [12.0]])
    projected = separable_1d_projection(scores)
    assert projected[0, 0] != pytest.approx(0.0)
    np.testing.assert_allclose(np.abs(projected[:, 0]), scores[:, 0], rtol=1e-8)


def test_separable_1d_projection_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        separable_1d_projection(np.zeros((0, 2)))
    with pytest.raises(ValueError):
        separable_1d_projection(np.zeros((5, 2)), weights=np.zeros(5))
