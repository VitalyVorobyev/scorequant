from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import scorequant as sq
from tests._fit import fit_test_quantizer


def _posteriors_from_ratios(ratios: np.ndarray, priors: np.ndarray) -> np.ndarray:
    values = ratios * priors[None, :]
    return values / np.sum(values, axis=1, keepdims=True)


def test_mixture_scores_match_simplex_derivative() -> None:
    posteriors = jnp.asarray([[0.6, 0.3, 0.1], [0.15, 0.35, 0.5]])
    priors = jnp.asarray([0.5, 0.3, 0.2])
    reference = jnp.asarray([0.2, 0.3, 0.5])
    ratios = np.asarray(posteriors) / np.asarray(priors)
    density = ratios @ np.asarray(reference)
    expected = (ratios[:, :2] - ratios[:, [2]]) / density[:, None]

    scores = sq.ratios.MixtureParameterization(reference).scores(
        sq.ratios.ratios_from_posteriors(posteriors, priors)
    )

    np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-7)
    assert scores.shape == (2, 2)


def test_ratios_from_posteriors_apply_the_declared_prior_correction() -> None:
    posteriors = np.asarray([[0.6, 0.3, 0.1], [0.15, 0.35, 0.5]])
    priors = np.asarray([0.5, 0.3, 0.2])

    ratios = sq.ratios.ratios_from_posteriors(posteriors, priors)

    np.testing.assert_allclose(ratios, posteriors / priors[None, :], rtol=1e-6)


def test_prior_correction_recovers_the_same_density_ratio_scores() -> None:
    rng = np.random.default_rng(20)
    ratios = np.exp(rng.normal(size=(50, 4)))
    reference = np.asarray([0.16, 0.14, 0.20, 0.50])
    uniform_prior = np.full(4, 0.25)
    skewed_prior = np.asarray([0.45, 0.15, 0.10, 0.30])

    uniform = sq.ratios.ratios_from_posteriors(
        _posteriors_from_ratios(ratios, uniform_prior), uniform_prior
    )
    skewed = sq.ratios.ratios_from_posteriors(
        _posteriors_from_ratios(ratios, skewed_prior), skewed_prior
    )

    np.testing.assert_allclose(
        sq.ratios.MixtureParameterization(reference).scores(uniform),
        sq.ratios.MixtureParameterization(reference).scores(skewed),
        rtol=1e-6,
        atol=1e-6,
    )


def test_mixture_scores_are_gauge_invariant() -> None:
    rng = np.random.default_rng(21)
    ratios = np.exp(rng.normal(size=(60, 3)))
    gauge = np.exp(rng.normal(size=(60, 1)))
    reference = np.asarray([0.2, 0.3, 0.5])

    plain = sq.ratios.MixtureParameterization(reference).scores(ratios)
    rescaled = sq.ratios.MixtureParameterization(reference).scores(ratios * gauge)

    np.testing.assert_allclose(plain, rescaled, rtol=1e-6, atol=1e-6)


def test_intensity_scores_are_gauge_invariant() -> None:
    rng = np.random.default_rng(23)
    ratios = np.exp(rng.normal(size=(60, 3)))
    gauge = np.exp(rng.normal(size=(60, 1)))
    coefficients = np.asarray([0.6, 1.4, 0.9])

    plain = sq.ratios.IntensityParameterization(coefficients).scores(ratios)
    rescaled = sq.ratios.IntensityParameterization(coefficients).scores(ratios * gauge)

    np.testing.assert_allclose(plain, rescaled, rtol=1e-6, atol=1e-6)


def test_mixture_scores_are_pairwise_differences_of_intensity_scores() -> None:
    rng = np.random.default_rng(24)
    ratios = np.exp(rng.normal(size=(40, 4)))
    reference = np.asarray([0.16, 0.14, 0.20, 0.50])

    intensity = np.asarray(sq.ratios.IntensityParameterization(reference).scores(ratios))
    mixture = np.asarray(sq.ratios.MixtureParameterization(reference).scores(ratios))

    np.testing.assert_allclose(mixture, intensity[:, :3] - intensity[:, [3]], rtol=1e-6, atol=1e-6)


def test_priors_proportional_to_reference_reduce_to_raw_ratios() -> None:
    rng = np.random.default_rng(25)
    ratios = np.exp(rng.normal(size=(30, 3)))
    reference = np.asarray([0.2, 0.3, 0.5])
    posteriors = _posteriors_from_ratios(ratios, reference)

    corrected = sq.ratios.ratios_from_posteriors(posteriors, reference)
    intensity = sq.ratios.IntensityParameterization(reference).scores(corrected)

    np.testing.assert_allclose(intensity, corrected, rtol=1e-6, atol=1e-6)


def test_reference_component_changes_only_the_score_parameterization() -> None:
    rng = np.random.default_rng(22)
    posteriors = rng.dirichlet(np.ones(4), size=300)
    priors = np.full(4, 0.25)
    reference = np.asarray([0.2, 0.25, 0.15, 0.4])
    ratios = sq.ratios.ratios_from_posteriors(posteriors, priors)
    last_scores = sq.ratios.MixtureParameterization(reference).scores(ratios)
    first_scores = sq.ratios.MixtureParameterization(reference, reference_component=0).scores(
        ratios
    )
    config = sq.KMeansConfig(seed=7, solver_restarts=4)

    last_labels = np.asarray(fit_test_quantizer(last_scores, n_bins=6, config=config).labels)
    first_labels = np.asarray(fit_test_quantizer(first_scores, n_bins=6, config=config).labels)

    np.testing.assert_array_equal(
        last_labels[:, None] == last_labels[None, :],
        first_labels[:, None] == first_labels[None, :],
    )


def test_class_permutation_preserves_scores_up_to_column_order() -> None:
    posteriors = np.asarray([[0.6, 0.3, 0.1], [0.15, 0.35, 0.5]])
    priors = np.asarray([0.5, 0.3, 0.2])
    reference = np.asarray([0.2, 0.3, 0.5])
    permutation = np.asarray([2, 0, 1])

    expected = sq.ratios.MixtureParameterization(reference).scores(
        sq.ratios.ratios_from_posteriors(posteriors, priors)
    )
    permuted = sq.ratios.MixtureParameterization(
        reference[permutation], reference_component=0
    ).scores(sq.ratios.ratios_from_posteriors(posteriors[:, permutation], priors[permutation]))

    np.testing.assert_allclose(permuted, expected)


def test_zero_posteriors_are_allowed_without_clipping() -> None:
    ratios = sq.ratios.ratios_from_posteriors(
        jnp.asarray([[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]], dtype=jnp.float32),
        [0.4, 0.3, 0.3],
    )
    scores = sq.ratios.MixtureParameterization([0.2, 0.3, 0.5]).scores(ratios)
    assert scores.dtype == jnp.float32
    assert np.isfinite(np.asarray(scores)).all()


def test_ratio_closure_report_flags_a_misdeclared_prior() -> None:
    rng = np.random.default_rng(26)
    ratios = np.exp(rng.normal(size=(2_000, 3)))
    ratios /= np.mean(ratios, axis=0, keepdims=True)
    weights = np.ones(ratios.shape[0])

    closed = sq.ratios.ratio_closure_report(ratios, weights)
    skew = np.asarray([1.4, 0.8, 0.9])
    biased = sq.ratios.ratio_closure_report(ratios * skew[None, :], weights)

    assert closed.max_residual < 1e-6
    assert biased.max_residual > 0.1
    assert closed.normalizers.shape == (3,)
    assert "max_residual" in closed.to_dict()


def test_ratio_closure_report_is_invariant_to_weight_scaling() -> None:
    rng = np.random.default_rng(27)
    ratios = np.exp(rng.normal(size=(100, 3)))
    weights = rng.uniform(0.1, 2.0, size=100)

    base = sq.ratios.ratio_closure_report(ratios, weights)
    scaled = sq.ratios.ratio_closure_report(ratios, 7.5 * weights)

    np.testing.assert_allclose(base.normalizers, scaled.normalizers, rtol=1e-6)


@pytest.mark.parametrize(
    ("posteriors", "priors", "message"),
    [
        ([[0.4, 0.4]], [0.5, 0.5], "sum to one"),
        ([[1.1, -0.1]], [0.5, 0.5], "nonnegative"),
        ([[0.5, 0.5]], [1.0, 0.0], "strictly positive"),
    ],
)
def test_invalid_posterior_inputs_fail(
    posteriors: list[list[float]],
    priors: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sq.ratios.ratios_from_posteriors(posteriors, priors)


@pytest.mark.parametrize(
    ("ratios", "reference", "reference_component", "message"),
    [
        ([[1.0, -0.5]], [0.5, 0.5], -1, "nonnegative"),
        ([[1.0, 0.5]], [0.2, 0.2], -1, "sum to one"),
        ([[1.0, 0.5]], [0.5, 0.5], 2, "outside"),
        ([[0.0, 0.0]], [0.5, 0.5], -1, "strictly positive at every row"),
    ],
)
def test_invalid_mixture_score_inputs_fail(
    ratios: list[list[float]],
    reference: list[float],
    reference_component: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sq.ratios.MixtureParameterization(
            reference, reference_component=reference_component
        ).scores(ratios)


def test_reference_component_rejects_non_integer() -> None:
    with pytest.raises(TypeError, match="integer"):
        sq.ratios.MixtureParameterization([0.5, 0.5], reference_component=True).scores([[1.0, 0.5]])


# Signal N(1, 1) over background N(-1, 1), a classifier trained at priors 0.3/0.7,
# and an intensity model with reference yields (background, signal) = (2, 0.5).
_GAUGE_PRIORS = np.array([0.3, 0.7])
_GAUGE_YIELDS = sq.ratios.IntensityParameterization([2.0, 0.5])


def _gaussian(x: np.ndarray, mean: float) -> np.ndarray:
    return np.exp(-0.5 * (x - mean) ** 2) / np.sqrt(2 * np.pi)


def _signal_background_posteriors(observations: np.ndarray) -> np.ndarray:
    x = np.asarray(observations)[:, 0]
    joint = np.column_stack(
        [_GAUGE_PRIORS[0] * _gaussian(x, 1.0), _GAUGE_PRIORS[1] * _gaussian(x, -1.0)]
    )
    return joint / joint.sum(axis=1, keepdims=True)


def _exact_intensity_scores(observations: np.ndarray) -> np.ndarray:
    x = np.asarray(observations)[:, 0]
    densities = np.column_stack([_gaussian(x, -1.0), _gaussian(x, 1.0)])
    return densities / (densities @ np.asarray(_GAUGE_YIELDS.coefficients))[:, None]


def _background_gauge_ratio(observations: np.ndarray) -> np.ndarray:
    """The recipe: the classifier's signal ratio over its background ratio, then 1."""
    eta = _signal_background_posteriors(observations)
    signal = (eta[:, 0] / _GAUGE_PRIORS[0]) / (eta[:, 1] / _GAUGE_PRIORS[1])
    return np.column_stack([np.ones_like(signal), signal])


def _ratio_provenance(gauges: tuple[str, ...]) -> sq.ScoreProvenance:
    return sq.ScoreProvenance(
        kind="estimated_ratio",
        ratio=sq.ratios.RatioProvenance(estimator="calibrated_classifier", gauges=gauges),
    )


def test_classifier_provider_declares_the_training_mixture_gauge() -> None:
    provider = sq.ratios.DensityRatioScore.from_classifier(
        _signal_background_posteriors,
        _GAUGE_PRIORS,
        sq.ratios.IntensityParameterization([0.5, 2.0]),
    )
    assert provider.provenance.ratio is not None
    assert provider.provenance.ratio.gauges == ("training_mixture", "training_mixture")


def test_mixed_gauge_ratio_columns_are_refused_with_the_fix() -> None:
    def mixed(observations: np.ndarray) -> np.ndarray:
        # Absolute Monte Carlo background density next to a classifier ratio over p_train.
        x = np.asarray(observations)[:, 0]
        eta = _signal_background_posteriors(observations)
        return np.column_stack([_gaussian(x, -1.0), eta[:, 0] / _GAUGE_PRIORS[0]])

    observations = np.linspace(-3.0, 3.0, 50)[:, None]
    unchecked = sq.ratios.DensityRatioScore(mixed, _GAUGE_YIELDS)
    # Undeclared, the mixed construction runs and is silently wrong.
    assert not np.allclose(unchecked.score(observations), _exact_intensity_scores(observations))
    with pytest.raises(sq.ContractError, match="one gauge"):
        sq.ratios.DensityRatioScore(
            mixed, _GAUGE_YIELDS, provenance=_ratio_provenance(("absolute", "training_mixture"))
        )
    with pytest.raises(sq.ContractError, match="declares 3 columns"):
        sq.ratios.DensityRatioScore(
            mixed, _GAUGE_YIELDS, provenance=_ratio_provenance(("absolute",) * 3)
        )


def test_no_record_can_carry_mixed_gauges() -> None:
    """External providers and ready score samples meet the same refusal."""
    with pytest.raises(sq.ContractError, match="different gauges"):
        sq.ratios.RatioProvenance(gauges=("training_mixture", "absolute"))
    with pytest.raises(TypeError, match="tuple of strings"):
        sq.ratios.RatioProvenance(gauges=["absolute", "absolute"])  # type: ignore[arg-type]
    with pytest.raises(sq.ContractError, match="different gauges"):
        sq.ScoreSample(
            _exact_intensity_scores(np.linspace(-3.0, 3.0, 40)[:, None]),
            provenance=_ratio_provenance(("training_mixture", "absolute")),
        )


def test_one_gauge_recipe_recovers_the_exact_scores() -> None:
    """Every column in the background gauge needs no Monte Carlo background density."""
    observations = np.linspace(-3.0, 3.0, 61)[:, None]
    recipe = sq.ratios.DensityRatioScore(
        _background_gauge_ratio,
        _GAUGE_YIELDS,
        provenance=_ratio_provenance(("background", "background")),
    )
    exact = _exact_intensity_scores(observations)
    np.testing.assert_allclose(recipe.score(observations), exact, rtol=1e-12, atol=1e-12)
    classifier = sq.ratios.DensityRatioScore.from_classifier(
        lambda rows: _signal_background_posteriors(rows)[:, ::-1],
        _GAUGE_PRIORS[::-1],
        _GAUGE_YIELDS,
    )
    np.testing.assert_allclose(classifier.score(observations), exact, rtol=1e-12, atol=1e-12)


def test_declared_gauges_survive_a_saved_rule(tmp_path: Path) -> None:
    provider = sq.ratios.DensityRatioScore(
        _background_gauge_ratio,
        _GAUGE_YIELDS,
        provenance=_ratio_provenance(("background", "background")),
    )
    sample = sq.ObservationSample(np.random.default_rng(3).normal(size=(200, 1)))
    fit = sq.fit_quantizer(sample, provider=provider, n_bins=3)
    restored = sq.Quantizer.load(fit.quantizer.save(tmp_path / "rule"))
    assert restored.provenance.ratio is not None
    assert restored.provenance.ratio.gauges == ("background", "background")
