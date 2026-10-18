import numpy as np
import pytest

import scorequant as sq
from examples.synthetic_problems import signal_background_shape


def test_score_callback_requires_a_reference_measure() -> None:
    provider = sq.ScoreFunction(lambda values: values)
    with pytest.raises(TypeError, match="source"):
        sq.fit_quantizer(provider, n_bins=2)  # type: ignore[arg-type]


def test_observation_and_score_sources_are_equivalent() -> None:
    observations = np.linspace(-1, 1, 60)[:, None]
    weights = np.linspace(1, 2, 60)
    provenance = sq.ScoreProvenance(kind="exact", reference_point=(0.0,))
    provider = sq.ScoreFunction(lambda values: np.asarray(values), provenance=provenance)
    config = sq.KMeansConfig(seed=3, solver_restarts=3)
    direct = sq.fit_quantizer(
        sq.ScoreSample(observations, weights, provenance=provenance),
        n_bins=3,
        criterion=sq.NormalizedTrace(),
        config=config,
    )
    composed = sq.fit_quantizer(
        sq.ObservationSample(observations, weights),
        provider=provider,
        n_bins=3,
        criterion=sq.NormalizedTrace(),
        config=config,
    )
    assert np.array_equal(
        np.asarray(direct.predict_scores(observations)),
        np.asarray(composed.predict_scores(observations)),
    )
    assert composed.provenance.exact_fisher
    assert composed.information_kind == "exact_fisher"


def test_precomputed_score_validation_accepts_observation_training() -> None:
    observations = np.linspace(-1, 1, 60)[:, None]
    provider = sq.ScoreFunction(lambda values: np.asarray(values))
    result = sq.fit_quantizer(
        sq.ObservationSample(observations),
        provider=provider,
        validation=sq.ScoreSample(observations[::3]),
        n_bins=3,
        criterion=sq.NormalizedTrace(),
        config=sq.KMeansConfig(seed=3, solver_restarts=3),
    )
    assert result.validation_report is not None


def test_bounded_quadrature_matches_symmetric_reference_law() -> None:
    source = sq.IntegrationSource(
        [[-1.0, 1.0]],
        density=lambda values: np.full(len(values), 0.5),
        quadrature=sq.GaussLegendreConfig(order=20),
    )
    provider = sq.ScoreFunction(
        lambda values: np.asarray(values),
        provenance=sq.ScoreProvenance(kind="exact"),
    )
    materialized = source.materialize()
    total_weight = float(np.sum(np.asarray(materialized.weights)))
    second_moment = float(
        np.sum(
            np.asarray(materialized.weights)
            * np.square(np.asarray(materialized.observations[:, 0]))
        )
        / total_weight
    )
    assert total_weight == pytest.approx(1.0, abs=1e-12)
    assert second_moment == pytest.approx(1 / 3, abs=1e-12)
    result = sq.fit_quantizer(
        source,
        provider=provider,
        n_bins=2,
        config=sq.DExchangeConfig(seed=2, initializer_restarts=3, max_iter=200),
    )
    labels = np.asarray(result.predict_scores([[-0.5], [0.5]]))
    assert labels[0] != labels[1]
    assert result.source_kind == "integration_source"


def test_quadrature_rejects_implicit_measure_and_capacity_explosion() -> None:
    with pytest.raises(TypeError):
        sq.IntegrationSource([[-1.0, 1.0]])  # type: ignore[call-arg]
    source = sq.IntegrationSource(
        np.tile([[-1.0, 1.0]], (5, 1)),
        density=lambda values: np.ones(len(values)),
        quadrature=sq.GaussLegendreConfig(order=20, max_points=1000),
    )
    with pytest.raises(ValueError, match="exceeding max_points"):
        source.materialize()


def test_central_log_ratio_score_corrects_training_priors() -> None:
    deltas = [0.2]
    priors = [0.25, 0.75]
    target_score = np.array([[-1.5], [0.4]])
    prior_odds = priors[1] / priors[0]
    odds = prior_odds * np.exp(2 * deltas[0] * target_score[:, 0])
    probabilities = np.column_stack([1 / (1 + odds), odds / (1 + odds)])
    provider = sq.ratios.CentralLogRatioScore(lambda values: probabilities, deltas, priors)
    assert np.allclose(np.asarray(provider.score(target_score)), target_score)


def test_central_log_ratio_score_is_always_estimated() -> None:
    def predict(values: object) -> np.ndarray:
        observations = np.asarray(values)[:, 0]
        plus = 1 / (1 + np.exp(-0.2 * observations))
        return np.column_stack([1 - plus, plus])

    provider = sq.ratios.CentralLogRatioScore(
        predict,
        [0.1],
        [0.5, 0.5],
        description="ready calibrated model",
    )
    assert provider.provenance.kind == "estimated_ratio"
    assert not provider.provenance.exact_fisher
    assert provider.provenance.ratio is not None
    assert provider.provenance.ratio.parameterization == "central_log_ratio"
    assert provider.provenance.ratio.estimator == "calibrated_classifier"
    assert provider.provenance.ratio.deltas == (0.1,)
    assert provider.provenance.ratio.training_priors == ((0.5, 0.5),)
    result = sq.fit_quantizer(
        sq.ObservationSample(np.arange(12, dtype=float)[:, None]),
        provider=provider,
        n_bins=2,
        criterion=sq.NormalizedTrace(),
        config=sq.KMeansConfig(solver_restarts=2),
    )
    assert result.information_kind == "supplied_score_surrogate"
    assert result.provenance.to_dict()["ratio"]["deltas"] == [0.1]


def _mixture_predict(values: object) -> np.ndarray:
    observations = np.asarray(values)[:, 0]
    logits = np.column_stack([0.3 * observations, -0.1 * observations, 0.0 * observations])
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def test_classifier_derived_ratio_provider_records_full_provenance() -> None:
    provider = sq.ratios.DensityRatioScore.from_classifier(
        _mixture_predict,
        [0.4, 0.35, 0.25],
        sq.ratios.MixtureParameterization([0.2, 0.3, 0.5]),
        calibration="temperature_oof",
        description="ready calibrated model",
    )
    assert provider.provenance.kind == "estimated_ratio"
    assert not provider.provenance.exact_fisher
    ratio_record = provider.provenance.ratio
    assert ratio_record is not None
    assert ratio_record.estimator == "calibrated_classifier"
    assert ratio_record.parameterization == "mixture"
    assert ratio_record.training_priors == (0.4, 0.35, 0.25)
    assert ratio_record.reference_fractions == (0.2, 0.3, 0.5)
    assert ratio_record.reference_component == 2
    assert ratio_record.calibration == "temperature_oof"
    result = sq.fit_quantizer(
        sq.ObservationSample(np.linspace(-2, 2, 40)[:, None]),
        provider=provider,
        n_bins=3,
        criterion=sq.NormalizedTrace(),
        config=sq.KMeansConfig(seed=5, solver_restarts=2),
    )
    assert result.information_kind == "supplied_score_surrogate"
    assert result.provenance.to_dict()["ratio"]["calibration"] == "temperature_oof"


def test_density_ratio_provider_matches_the_explicit_composition() -> None:
    priors = np.asarray([0.4, 0.35, 0.25])
    fractions = np.asarray([0.2, 0.3, 0.5])
    observations = np.linspace(-2, 2, 40)[:, None]
    provider = sq.ratios.DensityRatioScore.from_classifier(
        _mixture_predict, priors, sq.ratios.MixtureParameterization(fractions)
    )
    expected = sq.ratios.MixtureParameterization(fractions).scores(
        sq.ratios.ratios_from_posteriors(_mixture_predict(observations), priors)
    )
    np.testing.assert_allclose(np.asarray(provider.score(observations)), np.asarray(expected))


def test_analytic_component_ratios_reproduce_the_component_score() -> None:
    coefficients = np.asarray([0.7, 1.8])
    observations = np.linspace(0.1, 3.0, 50)[:, None]

    def signal(values: np.ndarray) -> np.ndarray:
        return np.exp(-np.asarray(values)[:, 0])

    def background(values: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(values).shape[0], 0.5)

    def component_ratios(values: object) -> np.ndarray:
        array = np.asarray(values)
        return np.column_stack([signal(array) / background(array), np.ones(array.shape[0])])

    exact = sq.LinearComponentScore([signal, background], list(coefficients)).score(observations)
    via_ratios = sq.ratios.DensityRatioScore(
        component_ratios,
        sq.ratios.IntensityParameterization(coefficients),
        provenance=sq.ScoreProvenance(kind="exact"),
    )
    np.testing.assert_allclose(
        np.asarray(via_ratios.score(observations)), np.asarray(exact), rtol=1e-6
    )
    assert via_ratios.provenance.exact_fisher
    assert via_ratios.provenance.ratio is not None
    assert via_ratios.provenance.ratio.parameterization == "intensity"


def test_density_ratio_provider_rejects_invalid_constructions() -> None:
    parameterization = sq.ratios.MixtureParameterization([0.5, 0.5])
    with pytest.raises(TypeError, match="callable"):
        sq.ratios.DensityRatioScore("not callable", parameterization)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="parameterization"):
        sq.ratios.DensityRatioScore(lambda values: values, "mixture")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="conflicts"):
        sq.ratios.DensityRatioScore(
            lambda values: values,
            parameterization,
            provenance=sq.ScoreProvenance(
                ratio=sq.ratios.RatioProvenance(reference_fractions=(0.4, 0.6))
            ),
        )
    with pytest.raises(ValueError, match="class_priors"):
        sq.ratios.DensityRatioScore.from_classifier(
            lambda values: values, [0.5, 0.4, 0.1], parameterization
        )

    provider = sq.ratios.DensityRatioScore(
        lambda values: -np.ones((np.asarray(values).shape[0], 2)), parameterization
    )
    with pytest.raises(ValueError, match="nonnegative"):
        provider.score(np.zeros((4, 1)))
    wrong_width = sq.ratios.DensityRatioScore(
        lambda values: np.ones((np.asarray(values).shape[0], 3)), parameterization
    )
    with pytest.raises(ValueError, match="2 components"):
        wrong_width.score(np.zeros((4, 1)))


def test_central_log_ratio_score_rejects_invalid_constructions() -> None:
    with pytest.raises(TypeError, match="callable"):
        sq.ratios.CentralLogRatioScore("not callable", [0.1], [0.5, 0.5])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="deltas"):
        sq.ratios.CentralLogRatioScore(lambda values: values, [-0.1], [0.5, 0.5])
    with pytest.raises(ValueError, match="class_priors"):
        sq.ratios.CentralLogRatioScore(lambda values: values, [0.1], [0.5, 0.5, 0.5])
    provider = sq.ratios.CentralLogRatioScore(
        lambda values: np.column_stack(
            [np.full(np.asarray(values).shape[0], 0.7), np.full(np.asarray(values).shape[0], 0.7)]
        ),
        [0.1],
        [0.5, 0.5],
    )
    with pytest.raises(ValueError, match="sum to one"):
        provider.score(np.zeros((3, 1)))


def test_integration_source_end_to_end_with_two_score_columns() -> None:
    """`fit_quantizer(IntegrationSource(...), provider=...)` with a multi-parameter score.

    The existing quadrature test above covers only a single score column from an
    identity provider. Two score directions (a signal fraction and one
    background-shape nuisance) exercise the bounded-quadrature path the way a real
    linear-component model would use it, closing that coverage gap.
    """
    problem = signal_background_shape(background_rates=(2.5,), n_bins=4)

    def signal_component(x: np.ndarray) -> np.ndarray:
        return problem.signal_density(np.asarray(x)[:, 0])

    def background_component(x: np.ndarray) -> np.ndarray:
        return problem.background_densities[0](np.asarray(x)[:, 0])

    provider = sq.LinearComponentScore(
        components={"signal": signal_component, "background": background_component},
        coefficients={
            "signal": float(problem.coefficients[0]),
            "background": float(problem.coefficients[1]),
        },
        variables=["x"],
    )

    source = sq.IntegrationSource(
        problem.bounds, density=problem.intensity, quadrature=sq.GaussLegendreConfig(order=48)
    )
    result = sq.fit_quantizer(
        source,
        provider=provider,
        n_bins=problem.n_bins,
        criterion=sq.DOptimality(),
        config=sq.DExchangeConfig(seed=50, initializer_restarts=4),
    )
    assert result.source_kind == "integration_source"
    assert result.transform.rank == 2

    test = problem.test
    report = result.evaluate_scores(test.scores, test.weights)
    assert report.geometric_mean_retention > 0.9

    profiled = sq.information.profiled_information_report(
        test.scores,
        np.asarray(result.predict_scores(test.scores)),
        interest=problem.interest,
        weights=test.weights,
        n_bins=problem.n_bins,
    )
    assert profiled.geometric_mean_retention > 0.9
