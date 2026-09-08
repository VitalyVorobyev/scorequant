"""Deterministic tests for the held-out retention uncertainty diagnostic.

The estimator is the O7 plug-in of ``RETENTION-PLUGIN-CLT-FROZEN-VECTOR``
with its influence-function standard error and an untruncated Wald interval.
Every check here is either exact (rational arithmetic, closed forms, pinned
counterexample fixtures) or a seeded Monte Carlo experiment judged at a
tolerance derived from its own replicate count.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import scorequant as sq
from scorequant._validation import validate_sample
from scorequant.information import _retention_influence

from ._oracles import _o7_exact_plugin

RESEARCH_WORKSPACE = Path(__file__).parents[1] / "agenticresearch"
BACKENDS = ("jax", "numpy")
_TOLERANCE = {"rtol": 1e-10, "atol": 1e-12}


def _execution(backend: str) -> sq.ExecutionConfig:
    return sq.ExecutionConfig(backend=backend, precision="float64", device="cpu")


def _fixture(name: str) -> dict:
    path = RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / f"{name}.json"
    assert path.is_file(), f"counterexample fixture missing at {path}"
    return json.loads(path.read_text())


def _regular_sample(
    seed: int, n_rows: int = 240, dimension: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n_rows, dimension)) + 0.3
    labels = (scores[:, 0] > 0).astype(int) + 2 * (scores[:, 1] > 0.2).astype(int)
    return scores, labels


def _influence(scores: np.ndarray, labels: np.ndarray, n_bins: int | None = None) -> np.ndarray:
    sample = validate_sample(scores)
    resolved = int(labels.max()) + 1 if n_bins is None else n_bins
    outcome = _retention_influence(sample, np.asarray(labels), resolved, None)
    assert outcome.influence is not None
    return np.asarray(outcome.influence)


def test_scalar_case_reduces_to_the_o6_closed_form() -> None:
    """O6: psi = ((1 - eta) S^2 - (S - c_Z)^2) / v and eta = 1 - RSS / sum S^2."""
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(150, 1)) + 0.4
    labels = np.digitize(scores[:, 0], [-0.5, 0.3, 1.1])
    report = sq.retention_uncertainty(scores, labels)

    s = scores[:, 0]
    means = np.array([s[labels == b].mean() for b in range(4)])
    v = np.mean(s * s)
    eta = 1.0 - np.sum((s - means[labels]) ** 2) / np.sum(s * s)
    psi = ((1.0 - eta) * s * s - (s - means[labels]) ** 2) / v
    assert report.status == "ok"
    assert report.estimate == pytest.approx(eta, rel=1e-12)
    assert report.standard_error == pytest.approx(math.sqrt(np.mean(psi * psi) / len(s)), rel=1e-10)
    np.testing.assert_allclose(_influence(scores, labels), psi, **_TOLERANCE)


def test_vector_case_matches_the_audited_exact_o7_influence_values() -> None:
    """The library's psi equals the exact rational O7 influence on a sample with ties,
    a duplicate atom, a singleton cell and a declared empty cell."""
    scores = [[3, 1], [3, 1], [-1, 2], [0, -2], [2, -1], [-2, -1], [1, 1], [-3, 2]]
    labels = [0, 0, 1, 1, 0, 2, 2, 3]
    n_bins = 5
    exact_scores = [[Fraction(x) for x in row] for row in scores]
    uniform = [Fraction(1, len(scores))] * len(scores)
    ratio, psi_r, _, _ = _o7_exact_plugin(exact_scores, labels, uniform, n_bins)

    array = np.asarray(scores, dtype=float)
    report = sq.retention_uncertainty(array, np.asarray(labels), n_bins=n_bins)
    assert report.status == "ok"
    assert report.estimate**2 == pytest.approx(float(ratio), rel=1e-12)
    eta = math.sqrt(float(ratio))
    expected = np.array([float(value) for value in psi_r]) / (2.0 * eta)
    influence = _influence(array, np.asarray(labels), n_bins)
    np.testing.assert_allclose(influence, expected, rtol=1e-10, atol=1e-12)
    assert abs(influence.sum()) < 1e-12
    assert report.standard_error == pytest.approx(
        math.sqrt(np.mean(expected**2) / len(scores)), rel=1e-10
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dimension", [2, 3])
def test_full_rank_estimate_agrees_with_information_report(backend: str, dimension: int) -> None:
    scores, labels = _regular_sample(21, dimension=dimension)
    labels = labels + 4 * (scores[:, -1] > -0.1).astype(int)
    execution = _execution(backend)
    report = sq.retention_uncertainty(scores, labels, execution=execution)
    reference = sq.information_report(scores, labels, execution=execution)
    assert reference.effective_rank == dimension
    assert report.status == "ok"
    assert report.estimate == pytest.approx(reference.geometric_mean_retention, rel=1e-12)
    assert report.n_observations == scores.shape[0]
    assert report.standard_error is not None and report.standard_error > 0
    lower, upper = report.confidence_interval
    assert lower < report.estimate < upper


def test_jax_and_numpy_backends_agree() -> None:
    scores, labels = _regular_sample(5)
    reports = {
        backend: sq.retention_uncertainty(scores, labels, execution=_execution(backend))
        for backend in BACKENDS
    }
    assert reports["jax"].status == reports["numpy"].status == "ok"
    assert reports["jax"].estimate == pytest.approx(reports["numpy"].estimate, rel=1e-12)
    assert reports["jax"].standard_error == pytest.approx(
        reports["numpy"].standard_error, rel=1e-10
    )


def test_reparameterization_event_order_and_bin_relabeling_invariance() -> None:
    scores, labels = _regular_sample(9)
    base = sq.retention_uncertainty(scores, labels)

    mixing = np.array([[1.3, -0.4], [0.7, 2.1]])
    reparameterized = sq.retention_uncertainty(scores @ mixing.T, labels)
    order = np.random.default_rng(1).permutation(scores.shape[0])
    reordered = sq.retention_uncertainty(scores[order], labels[order])
    relabeled = sq.retention_uncertainty(scores, np.array([3, 0, 2, 1])[labels])

    for other in (reparameterized, reordered, relabeled):
        assert other.status == base.status == "ok"
        assert other.estimate == pytest.approx(base.estimate, rel=1e-10, abs=1e-12)
        assert other.standard_error == pytest.approx(base.standard_error, rel=1e-10, abs=1e-12)
        np.testing.assert_allclose(
            other.confidence_interval, base.confidence_interval, **_TOLERANCE
        )

    # Duplicating rows is deliberately not an invariance: it asserts a larger
    # iid sample, so the estimate is unchanged but the standard error shrinks.
    doubled = sq.retention_uncertainty(
        np.concatenate([scores, scores]), np.concatenate([labels, labels])
    )
    assert doubled.estimate == pytest.approx(base.estimate, rel=1e-10, abs=1e-12)
    assert doubled.standard_error == pytest.approx(base.standard_error / math.sqrt(2), rel=1e-10)


def test_empty_bins_contribute_nothing() -> None:
    scores, labels = _regular_sample(13)
    inferred = sq.retention_uncertainty(scores, labels)
    padded = sq.retention_uncertainty(scores, labels, n_bins=9)
    spread = sq.retention_uncertainty(scores, 2 * labels, n_bins=8)
    for other in (padded, spread):
        assert other.to_dict() == inferred.to_dict()


def test_confidence_level_sets_the_normal_quantile() -> None:
    scores, labels = _regular_sample(17)
    narrow = sq.retention_uncertainty(scores, labels, confidence_level=0.5)
    wide = sq.retention_uncertainty(scores, labels, confidence_level=0.99)
    assert narrow.confidence_level == 0.5
    half_narrow = 0.5 * (narrow.confidence_interval[1] - narrow.confidence_interval[0])
    half_wide = 0.5 * (wide.confidence_interval[1] - wide.confidence_interval[0])
    assert half_narrow == pytest.approx(0.6744897501960817 * narrow.standard_error, rel=1e-10)
    assert half_wide == pytest.approx(2.5758293035489004 * wide.standard_error, rel=1e-10)


def test_confidence_level_within_an_ulp_of_one_is_accepted() -> None:
    """Regression for the review of PR #66: the quantile is evaluated from
    the tail, so a level the contract accepts never rounds its argument to one."""
    scores, labels = _regular_sample(17)
    level = float(np.nextafter(1.0, 0.0))
    report = sq.retention_uncertainty(scores, labels, confidence_level=level)
    assert report.status == "ok"
    assert report.confidence_level == level
    lower, upper = report.confidence_interval
    assert math.isfinite(lower) and math.isfinite(upper)
    assert upper - lower > 2 * 8 * report.standard_error


def test_interval_is_not_clipped_to_the_unit_interval() -> None:
    # Six rows spread over three cells at unit retention minus a small
    # perturbation: the standard error dwarfs the distance to one.
    scores = np.array([[1.0, 0.0], [1.02, 0.0], [0.0, 1.0], [0.0, 0.98], [1.0, 1.0], [1.03, 0.97]])
    labels = np.array([0, 0, 1, 1, 2, 2])
    report = sq.retention_uncertainty(scores, labels)
    assert report.status == "ok"
    assert report.estimate < 1
    assert report.confidence_interval[1] > 1


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, 1.5, -0.2, float("nan"), float("inf")])
def test_confidence_level_outside_the_open_unit_interval_is_a_contract_error(
    confidence_level: float,
) -> None:
    scores, labels = _regular_sample(2)
    with pytest.raises(sq.ContractError):
        sq.retention_uncertainty(scores, labels, confidence_level=confidence_level)


@pytest.mark.parametrize("confidence_level", [True, "0.9", None])
def test_confidence_level_of_the_wrong_type_is_a_type_error(confidence_level: object) -> None:
    scores, labels = _regular_sample(2)
    with pytest.raises(TypeError):
        sq.retention_uncertainty(scores, labels, confidence_level=confidence_level)  # type: ignore[arg-type]


def test_malformed_inputs_are_refused_before_computation() -> None:
    scores, labels = _regular_sample(3)
    with pytest.raises(sq.ContractError):
        sq.retention_uncertainty(scores[:1], labels[:1])
    with pytest.raises(sq.ContractError):
        sq.retention_uncertainty(np.array([[1.0, np.nan], [0.5, 1.0]]), np.array([0, 1]))
    with pytest.raises(sq.ContractError):
        sq.retention_uncertainty(scores, labels[:-1])
    with pytest.raises(TypeError):
        sq.retention_uncertainty(scores, labels.astype(float))
    with pytest.raises(sq.ContractError):
        sq.retention_uncertainty(scores, labels, n_bins=2)
    with pytest.raises(TypeError):
        sq.retention_uncertainty(scores, labels, n_bins=4.0)  # type: ignore[arg-type]
    with pytest.raises(sq.ContractError):
        sq.retention_uncertainty(scores, labels, rank_rtol=1.0)
    with pytest.raises(TypeError):
        sq.retention_uncertainty(scores, labels, execution="numpy")  # type: ignore[arg-type]


def test_singular_full_information_is_refused_not_projected() -> None:
    scores, labels = _regular_sample(4)
    collinear = np.column_stack([scores[:, 0], 2.0 * scores[:, 0]])
    report = sq.retention_uncertainty(collinear, labels)
    assert report.status == "singular_full_information"
    assert report.estimate is None
    assert report.standard_error is None
    assert report.confidence_interval is None
    assert report.n_observations == scores.shape[0]

    # CE-O7-UNIT-RETENTION-SINGULAR-SAMPLE-001: both draws in one cell of a
    # lossless law. The library's projected report says 1; the plug-in's own
    # convention says 0; the diagnostic reports neither. Its two-cell rule
    # is refused structurally; declared under a third cell the same rows
    # reach the full-moment guard.
    fixture = _fixture("CE-O7-UNIT-RETENTION-SINGULAR-SAMPLE-001")
    sample = np.asarray(fixture["scores"], dtype=float)
    sample_labels = np.asarray(fixture["labels_before"])
    projected = sq.information_report(sample, sample_labels, n_bins=fixture["K"])
    assert projected.effective_rank == 1
    assert projected.geometric_mean_retention == pytest.approx(1.0)
    withheld = sq.retention_uncertainty(sample, sample_labels, n_bins=fixture["K"])
    assert withheld.status == "insufficient_cells"
    assert withheld.estimate is None
    singular = sq.retention_uncertainty(sample, sample_labels, n_bins=fixture["K"] + 1)
    assert singular.status == "singular_full_information"
    assert singular.estimate is None


def test_all_zero_scores_are_singular_full_information() -> None:
    report = sq.retention_uncertainty(np.zeros((6, 2)), np.array([0, 1, 2, 0, 1, 2]))
    assert report.status == "singular_full_information"


# Six rows in three cells whose means are c, -c and 0: the full moment is
# positive definite while the between-cell moment has exact rank one.
_COLLINEAR_CELLS = np.array([[1, 0], [2, 3], [-1, 0], [-2, -3], [0, 1], [0, -1]], dtype=float)
_COLLINEAR_LABELS = np.array([0, 0, 1, 1, 2, 2])


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_rule_with_at_most_d_cells_is_refused_before_any_moment(backend: str) -> None:
    """FI-RANK-CEILING: with K <= d declared cells the population retention is
    zero at the reference law whatever the empirical rank, and the audit
    requires the interval to be refused rather than built on the biased
    plug-in (AUDIT-RETENTION-PLUGIN-VECTOR-001, sections 10 and 11)."""
    execution = _execution(backend)
    rng = np.random.default_rng(31)
    half = rng.normal(size=(60, 2))
    centred = sq.retention_uncertainty(
        np.concatenate([half, -half]), np.repeat([0, 1], 60), execution=execution
    )
    fewer_cells = sq.retention_uncertainty(
        rng.normal(size=(50, 3)), np.arange(50) % 2, execution=execution
    )
    # A constant scalar rule (K = d = 1) whose uncentred sample plug-in is
    # positive: information_report shows that number, the diagnostic does not.
    constant = sq.retention_uncertainty(
        np.array([[-2.0], [-1.0], [1.0], [3.0]]),
        np.zeros(4, dtype=int),
        n_bins=1,
        execution=execution,
    )
    for report in (centred, fewer_cells, constant):
        assert report.status == "insufficient_cells"
        assert report.estimate is None
        assert report.standard_error is None
        assert report.confidence_interval is None
    # The declared count decides, empty cells included: the same rows under a
    # three-cell rule pass this guard and fall to the retained-rank guard.
    declared = sq.retention_uncertainty(
        np.concatenate([half, -half]), np.repeat([0, 1], 60), n_bins=3, execution=execution
    )
    assert declared.status == "singular_retained_information"


@pytest.mark.parametrize("backend", BACKENDS)
def test_singular_retained_information_is_caught_before_any_root(backend: str) -> None:
    # In floating point the lost eigenvalue is rounding noise whose square
    # root the plug-in would otherwise report as a regular endpoint.
    report = sq.retention_uncertainty(
        _COLLINEAR_CELLS, _COLLINEAR_LABELS, execution=_execution(backend)
    )
    assert report.status == "singular_retained_information"
    assert report.estimate == 0.0
    assert report.standard_error is None
    assert report.confidence_interval is None


@pytest.mark.parametrize(
    ("backend", "precision", "delta"),
    [
        ("numpy", "float64", 2.0**-9),
        ("jax", "float64", 2.0**-9),
        ("numpy", "float32", 2.0**-2),
        ("jax", "float32", 2.0**-3),
    ],
)
def test_retained_rank_guard_survives_an_ill_conditioned_reparameterization(
    backend: str, precision: str, delta: float
) -> None:
    """Regression for the review of PR #66: a nonsingular coordinate change
    whose full moment stays above the rank threshold must not turn the exact
    rank-one between-cell moment into a regular-looking interval. The retained
    matrix is therefore aggregated from whitened rows, not whitened after
    being formed in the supplied coordinates."""
    execution = sq.ExecutionConfig(backend=backend, precision=precision, device="cpu")
    mixing = np.array([[1.0, 1.0], [1.0, 1.0 + delta]])
    before = sq.retention_uncertainty(_COLLINEAR_CELLS, _COLLINEAR_LABELS, execution=execution)
    after = sq.retention_uncertainty(
        _COLLINEAR_CELLS @ mixing, _COLLINEAR_LABELS, execution=execution
    )
    assert before.status == after.status == "singular_retained_information"
    assert after.estimate == 0.0
    assert after.confidence_interval is None


def test_rank_rtol_override_moves_the_retained_guard() -> None:
    rng = np.random.default_rng(41)
    scores = rng.normal(size=(200, 2))
    scores[:, 1] = 1e-4 * scores[:, 1]
    labels = (scores[:, 0] > 0).astype(int) + 2 * (scores[:, 1] > 0).astype(int)
    default = sq.retention_uncertainty(scores, labels)
    assert default.status == "ok"
    strict = sq.retention_uncertainty(scores, labels, rank_rtol=0.5)
    assert strict.status in {"singular_full_information", "singular_retained_information"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_o7_ellipsoid_zero_variance_fixture_withholds_the_interval(backend: str) -> None:
    """CE-O7-ELLIPSOID-ZERO-VARIANCE-001 as an equally weighted sample of its eight atoms."""
    fixture = _fixture("CE-O7-ELLIPSOID-ZERO-VARIANCE-001")
    assert all(Fraction(value) == Fraction(1, 8) for value in fixture["weights"])
    scores = np.asarray(fixture["scores"], dtype=float)
    labels = np.asarray(fixture["labels_before"])
    report = sq.retention_uncertainty(
        scores, labels, n_bins=fixture["K"], execution=_execution(backend)
    )
    assert report.status == "degenerate_variance"
    assert report.estimate == pytest.approx(9 / 25, rel=1e-12)
    assert report.standard_error == 0.0
    assert report.confidence_interval is None


def test_lossless_rule_has_unit_estimate_and_degenerate_variance() -> None:
    atoms = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    labels = np.array([0, 0, 1, 2, 2, 1, 0])
    report = sq.retention_uncertainty(atoms[labels], labels)
    assert report.status == "degenerate_variance"
    assert report.estimate == pytest.approx(1.0, rel=1e-12)
    assert report.standard_error == 0.0
    assert report.confidence_interval is None


def test_every_status_serializes_to_strict_json() -> None:
    scores, labels = _regular_sample(8)
    fixture = _fixture("CE-O7-ELLIPSOID-ZERO-VARIANCE-001")
    reports = [
        sq.retention_uncertainty(scores, labels),
        sq.retention_uncertainty(scores, labels % 2),
        sq.retention_uncertainty(np.column_stack([scores[:, 0], scores[:, 0]]), labels),
        sq.retention_uncertainty(_COLLINEAR_CELLS, _COLLINEAR_LABELS),
        sq.retention_uncertainty(
            np.asarray(fixture["scores"], dtype=float), np.asarray(fixture["labels_before"])
        ),
    ]
    assert [report.status for report in reports] == [
        "ok",
        "insufficient_cells",
        "singular_full_information",
        "singular_retained_information",
        "degenerate_variance",
    ]
    for report in reports:
        facts = json.loads(json.dumps(report.to_dict(), allow_nan=False))
        assert set(facts) == {
            "estimate",
            "standard_error",
            "confidence_interval",
            "confidence_level",
            "n_observations",
            "status",
        }
        assert facts["status"] == report.status
        assert str(report).startswith("ScoreQuant retention uncertainty")


# --- seeded coverage experiments -------------------------------------------------

_ATOMS = np.array(
    [
        [2.0, 0.5],
        [1.5, -0.5],
        [3.0, 1.0],
        [-1.0, 2.0],
        [-0.5, 1.0],
        [-2.0, 2.5],
        [-1.0, -1.5],
        [0.5, -2.0],
        [-2.0, -3.0],
    ]
)
_CELLS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
_MASSES = np.array([0.15, 0.10, 0.08, 0.12, 0.14, 0.06, 0.13, 0.12, 0.10])
_MASSES = _MASSES / _MASSES.sum()


def _population_retention(atoms: np.ndarray, cells: np.ndarray, masses: np.ndarray) -> float:
    full = (atoms * masses[:, None]).T @ atoms
    retained = np.zeros_like(full)
    for cell in np.unique(cells):
        mass = masses[cells == cell].sum()
        moment = (atoms[cells == cell] * masses[cells == cell, None]).sum(axis=0)
        retained += np.outer(moment, moment) / mass
    return float((np.linalg.det(retained) / np.linalg.det(full)) ** (1 / atoms.shape[1]))


def _coverage(
    draw: Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]],
    target: float,
    n_rows: int,
    replicates: int,
    seed: int,
) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    execution = _execution("numpy")
    hits = 0
    regular = 0
    for _ in range(replicates):
        scores, labels = draw(rng, n_rows)
        report = sq.retention_uncertainty(scores, labels, execution=execution)
        if report.status != "ok":
            continue
        regular += 1
        lower, upper = report.confidence_interval
        hits += lower <= target <= upper
    return hits / regular, regular


def test_wald_coverage_on_a_bounded_regular_law_matches_the_nominal_level() -> None:
    """A bounded atom law with three atoms per cell satisfies every O7 condition.
    The theorem is asymptotic, so this pins what was measured for this law,
    N = 200 and seed 11: the finite-sample coverage sits within three binomial
    Monte Carlo standard errors of 95%. The atoms have nonzero mean, so the
    test exercises the uncentred plug-in algebra, not the reference-law
    reading of the number as a Fisher retention."""
    target = _population_retention(_ATOMS, _CELLS, _MASSES)

    def draw(rng: np.random.Generator, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
        index = rng.choice(len(_ATOMS), n_rows, p=_MASSES)
        return _ATOMS[index], _CELLS[index]

    replicates = 400
    coverage, regular = _coverage(draw, target, 200, replicates, 11)
    assert regular == replicates
    monte_carlo_error = math.sqrt(0.95 * 0.05 / replicates)
    assert abs(coverage - 0.95) <= 3 * monte_carlo_error


def test_heavy_tails_produce_material_undercoverage() -> None:
    """A Student-t law with three degrees of freedom has infinite fourth moment,
    outside O7's conditions. Under the sign rule its retention is exactly
    (E|S|)^2 / E S^2 = 4 / pi^2, and the seeded Wald coverage falls far below
    the nominal level: a limitation reproduced at one seed, not a universal
    coverage claim."""
    target = 4 / math.pi**2

    def draw(rng: np.random.Generator, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
        scores = rng.standard_t(3.0, size=(n_rows, 1))
        return scores, (scores[:, 0] > 0).astype(int)

    replicates = 400
    coverage, regular = _coverage(draw, target, 200, replicates, 11)
    assert regular == replicates
    monte_carlo_error = math.sqrt(0.95 * 0.05 / replicates)
    assert coverage < 0.95 - 3 * monte_carlo_error
    assert coverage < 0.75
