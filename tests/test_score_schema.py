"""Named score coordinates, the provider protocol, and the canonical sample input.

The score matrix is a table of partial derivatives whose column order is
meaningful but invisible; these tests pin the three places that invisibility
used to leak into the API -- a profiled criterion addressed by index, a
fixed-sample task that could not take the weighted score law the fitting task
takes, and a closed provider union no external estimator could join.
"""

from __future__ import annotations

import numpy as np
import pytest

import scorequant as sq

SEED = 11


def _scores(rows: int = 400, dims: int = 3) -> np.ndarray:
    return np.asarray(np.random.default_rng(SEED).normal(size=(rows, dims)))


def _schema() -> sq.ScoreSchema:
    return sq.ScoreSchema(("mu_x", "mu_y", "mu_z"))


def test_schema_resolves_names_to_columns() -> None:
    schema = _schema()
    assert schema.dimension == 3
    assert schema.index("mu_z") == 2
    assert schema.select("mu_z", "mu_x") == (2, 0)
    assert schema.to_dict() == {"parameters": ["mu_x", "mu_y", "mu_z"]}


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ((), ValueError, "at least one name"),
        (("a", "a"), ValueError, "unique"),
        (("a", "  "), ValueError, "non-empty"),
        (("a", 2), TypeError, "strings"),
    ],
)
def test_schema_rejects_a_meaningless_name_vector(
    parameters: tuple[object, ...], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        sq.ScoreSchema(parameters)  # type: ignore[arg-type]


def test_unknown_name_reports_the_declared_parameters() -> None:
    with pytest.raises(KeyError, match="mu_x, mu_y, mu_z"):
        _schema().index("HSPCs")


def test_schema_must_agree_with_the_score_and_the_reference_point() -> None:
    with pytest.raises(ValueError, match="names 3 parameters but the scores have 2"):
        sq.ScoreSample(_scores(dims=2), schema=_schema())
    with pytest.raises(ValueError, match="reference point has 2 entries"):
        sq.ScoreSample(
            _scores(),
            schema=_schema(),
            provenance=sq.ScoreProvenance(kind="exact", reference_point=(0.0, 0.0)),
        )


def test_named_interest_and_index_interest_agree_exactly() -> None:
    """Names are a spelling of the same columns, not a different problem."""
    sample = sq.ScoreSample(_scores(), schema=_schema())
    config = sq.DExchangeConfig(seed=SEED)
    named = sq.optimize_partition(
        sample, n_bins=4, criterion=sq.ProfiledDOptimality(interest=("mu_z",)), config=config
    )
    indexed = sq.optimize_partition(
        sample, n_bins=4, criterion=sq.ProfiledDOptimality(interest=(2,)), config=config
    )
    assert np.array_equal(named.labels, indexed.labels)
    assert named.objective == indexed.objective
    assert named.criterion.interest == indexed.criterion.interest == (2,)


def test_named_interest_without_a_schema_names_the_remedy() -> None:
    with pytest.raises(ValueError, match="no ScoreSchema"):
        sq.optimize_partition(
            _scores(), n_bins=4, criterion=sq.ProfiledDOptimality(interest=("mu_z",))
        )


def test_unresolved_criterion_refuses_to_index_a_score_matrix() -> None:
    """The boundary is the only place names are translated; bypassing it fails loudly."""
    with pytest.raises(ValueError, match="did not come through"):
        _ = sq.ProfiledDOptimality(interest=("mu_z",)).interest_indices


def test_interest_may_not_mix_names_and_indices() -> None:
    with pytest.raises(TypeError, match="all integer indices or all parameter names"):
        sq.ProfiledDOptimality(interest=("mu_z", 1))  # type: ignore[arg-type]


def test_score_sample_and_array_shorthand_partition_identically() -> None:
    scores = _scores()
    config = sq.DExchangeConfig(seed=SEED)
    from_sample = sq.optimize_partition(sq.ScoreSample(scores), n_bins=4, config=config)
    from_array = sq.optimize_partition(scores, n_bins=4, config=config)
    assert np.array_equal(from_sample.labels, from_array.labels)


@pytest.mark.parametrize("keyword", ["weights", "provenance"])
def test_a_sample_and_its_loose_parts_may_not_be_combined(keyword: str) -> None:
    sample = sq.ScoreSample(_scores())
    extra = {
        "weights": np.ones(sample.scores.shape[0]),
        "provenance": sq.ScoreProvenance(kind="exact"),
    }[keyword]
    with pytest.raises(ValueError, match=f"{keyword} must be omitted"):
        sq.optimize_partition(sample, n_bins=4, **{keyword: extra})


def test_schema_reaches_both_results() -> None:
    sample = sq.ScoreSample(_scores(), schema=_schema())
    partition = sq.optimize_partition(sample, n_bins=4, config=sq.DExchangeConfig(seed=SEED))
    quantizer = sq.fit_quantizer(sample, n_bins=4, config=sq.DExchangeConfig(seed=SEED))
    assert partition.schema is not None
    assert quantizer.schema is not None
    assert partition.schema.parameters == quantizer.schema.parameters == _schema().parameters


def test_linear_component_provider_derives_its_schema() -> None:
    """Component names already exist, so the schema is derived rather than asked for."""
    provider = sq.LinearComponentScore(
        components={
            "peak": lambda X: np.exp(-0.5 * ((X[:, 0] - 1.0) / 0.4) ** 2),
            "flat": lambda X: np.ones(X.shape[0]),
        },
        coefficients={"peak": 1.0, "flat": 0.5},
        variables=["mass"],
    )
    assert provider.schema.parameters == ("peak", "flat")
    source = sq.IntegrationSource(
        [[-2.0, 3.0]],
        density=lambda X: np.exp(-0.5 * ((X[:, 0] - 1.0) / 0.4) ** 2) + 0.5,
        quadrature=sq.GaussLegendreConfig(order=48),
    )
    result = sq.fit_quantizer(
        source, provider=provider, n_bins=4, config=sq.DExchangeConfig(seed=SEED)
    )
    assert result.schema is not None
    assert result.schema.parameters == ("peak", "flat")


class _ExternalScore:
    """A provider written outside the library, wrapping nothing."""

    provenance = sq.ScoreProvenance(kind="estimated_ratio", description="external estimator")

    def score(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations)
        return np.column_stack([values[:, 0], values[:, 0] ** 2 - 1.0])


def test_an_external_class_satisfies_the_provider_protocol() -> None:
    observations = np.asarray(np.random.default_rng(SEED).normal(size=(600, 1)))
    result = sq.fit_quantizer(
        sq.ObservationSample(observations),
        provider=_ExternalScore(),
        n_bins=4,
        config=sq.DExchangeConfig(seed=SEED),
    )
    assert isinstance(_ExternalScore(), sq.ScoreProvider)
    assert result.n_bins == 4
    assert result.information_kind == "supplied_score_surrogate"


def test_built_in_providers_implement_the_protocol_they_document() -> None:
    provider = sq.ScoreFunction(
        lambda X: np.asarray(X), provenance=sq.ScoreProvenance(kind="exact")
    )
    assert isinstance(provider, sq.ScoreProvider)


def test_a_provider_that_cannot_declare_its_provenance_is_refused() -> None:
    class _Broken:
        provenance = "estimated"

        def score(self, observations: np.ndarray) -> np.ndarray:
            return np.asarray(observations)

    with pytest.raises(TypeError, match="provenance must be a ScoreProvenance"):
        sq.fit_quantizer(sq.ObservationSample(_scores(dims=1)), provider=_Broken(), n_bins=3)

    with pytest.raises(TypeError, match="missing score, provenance"):
        sq.fit_quantizer(sq.ObservationSample(_scores(dims=1)), provider=object(), n_bins=3)


def test_validation_scores_must_name_the_same_parameters() -> None:
    """A reordering is invisible to a column count but not to the names."""
    scores = _scores()
    train = sq.ScoreSample(scores, schema=_schema())
    reordered = sq.ScoreSample(scores, schema=sq.ScoreSchema(("mu_z", "mu_y", "mu_x")))
    with pytest.raises(ValueError, match="training parameter order"):
        sq.fit_quantizer(
            train, validation=reordered, n_bins=4, config=sq.DExchangeConfig(seed=SEED)
        )
