"""Standalone stability certificates, D geometry reports, and global certification.

The gates here separate three claims that are easy to conflate: one scan
certifies that no relocation improves a labeling, the geometry report measures
the strict Voronoi structure that Theorem 3 attaches to such a labeling, and
branch-and-bound certification decides global optimality against the exhaustive
oracle.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

import scorequant as sq

from ._oracles import _exhaustive_d_oracle

_ORACLE_CASES = [(1, 8), (2, 9), (3, 9), (4, 10), (5, 10), (6, 8)]

# Tolerance of the boundary-row fixture below, chosen just above the single
# improving move that fixture leaves on the table (0.0022702 nats).
_BOUNDARY_GAIN_TOLERANCE = 2.5e-3


def _seeded_scores(seed: int, n_rows: int, n_features: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n_rows, n_features))
    return scores - scores.mean(axis=0)


def _raw_geometry(
    scores: np.ndarray, weights: np.ndarray, labels: np.ndarray, n_bins: int
) -> tuple[float, float, float, int, int]:
    """Recompute the D geometry diagnostics directly from raw scores.

    The quadratic forms of the report are invariant under the Fisher whitening
    the solver works in, so this deliberately naive loop over untransformed
    scores is an independent check rather than a restatement.
    """
    masses = np.bincount(labels, weights=weights, minlength=n_bins)
    means = np.stack(
        [
            np.average(scores[labels == cell], axis=0, weights=weights[labels == cell])
            for cell in range(n_bins)
        ]
    )
    metric = np.linalg.inv(
        np.asarray(sq.binned_fisher_information(scores, labels, weights, n_bins=n_bins))
    )
    separations = np.asarray(
        [
            [(means[a] - means[b]) @ metric @ (means[a] - means[b]) for b in range(n_bins)]
            for a in range(n_bins)
        ]
    )
    violation = -np.inf
    guaranteed = 0.0
    violating = 0
    evaluated = 0
    for row, source in enumerate(labels):
        distances = np.asarray(
            [
                (scores[row] - means[cell]) @ metric @ (scores[row] - means[cell])
                for cell in range(n_bins)
            ]
        )
        others = [cell for cell in range(n_bins) if cell != source]
        violation = max(violation, distances[source] - min(distances[cell] for cell in others))
        if masses[source] <= weights[row]:
            continue
        alpha = weights[row] * masses[source] / (masses[source] - weights[row])
        for cell in others:
            evaluated += 1
            if distances[source] < distances[cell]:
                continue
            violating += 1
            beta = weights[row] * masses[cell] / (masses[cell] + weights[row])
            guaranteed = max(
                guaranteed, float(np.log1p(alpha * beta * separations[source, cell] ** 2 / 4))
            )
    residual = max(
        float(separations[a, b] - 1 / masses[a] - 1 / masses[b])
        for a in range(n_bins)
        for b in range(a)
    )
    return float(violation), guaranteed, residual, violating, evaluated


def test_stability_report_certifies_an_exchange_result() -> None:
    """A terminal exchange state must scan clean and report the same objective."""
    scores = _seeded_scores(3, 60)
    result = sq.optimize_partition(scores, n_bins=3)
    report = sq.exchange_stability_report(scores, result.labels)
    assert report.stable is True
    assert report.best_move is None
    assert report.n_bins == 3
    assert report.criterion == sq.DOptimality()
    assert report.objective == pytest.approx(result.objective, abs=1e-12)
    assert report.best_gain == pytest.approx(result.best_remaining_gain, abs=1e-12)
    assert report.gain_tolerance == result.config.gain_tolerance == 1e-10
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_stability_report_reproduces_the_engine_verdict_after_a_rejected_lloyd(seed: int) -> None:
    """A guarded batch stopped by ``guard='reject'`` is certified from outside."""
    rng = np.random.default_rng(seed)
    scores = _seeded_scores(seed, 200, 3)
    weights = rng.uniform(0.4, 2.0, size=len(scores))
    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=5,
        config=sq.MahalanobisLloydConfig(seed=seed, guard="reject"),
    )
    report = sq.exchange_stability_report(scores, result.labels, weights=weights)
    assert report.stable is result.exchange_stable
    assert report.best_gain == pytest.approx(result.best_remaining_gain, abs=1e-12)
    assert (report.best_move is None) is result.exchange_stable


def test_stability_report_names_a_move_whose_exact_gain_it_predicts() -> None:
    """The reported move must reproduce the reported gain when it is applied."""
    scores = _seeded_scores(3, 60)
    labels = np.asarray(sq.optimize_partition(scores, n_bins=3).labels).copy()
    labels[0] = (labels[0] + 1) % 3
    report = sq.exchange_stability_report(scores, labels)
    assert report.stable is False
    assert report.best_gain > 0
    assert report.best_move is not None

    row, destination = report.best_move
    moved = labels.copy()
    moved[row] = destination
    after = sq.exchange_stability_report(scores, moved)
    assert after.objective - report.objective == pytest.approx(report.best_gain, abs=1e-10)

    before_information = np.asarray(sq.binned_fisher_information(scores, labels, n_bins=3))
    after_information = np.asarray(sq.binned_fisher_information(scores, moved, n_bins=3))
    direct = np.linalg.slogdet(after_information)[1] - np.linalg.slogdet(before_information)[1]
    assert direct == pytest.approx(report.best_gain, abs=1e-10)


def test_stability_report_certifies_the_profiled_criterion() -> None:
    """Profiled labels are certified against the profiled objective, not the D one."""
    rng = np.random.default_rng(17)
    scores = _seeded_scores(17, 120, 3)
    weights = rng.uniform(0.4, 2.0, size=len(scores))
    criterion = sq.ProfiledDOptimality((0,))
    result = sq.optimize_partition(scores, weights=weights, n_bins=4, criterion=criterion)
    report = sq.exchange_stability_report(
        scores, result.labels, weights=weights, criterion=criterion
    )
    assert report.stable is True
    assert report.criterion == criterion
    assert report.objective == pytest.approx(result.objective, abs=1e-12)

    mismatched = sq.exchange_stability_report(scores, result.labels, weights=weights)
    assert mismatched.objective != pytest.approx(result.objective, abs=1e-6)


def test_stability_report_rejects_labelings_it_cannot_certify() -> None:
    """Every invalid labeling fails by name instead of scanning a broken state."""
    scores = _seeded_scores(9, 30)
    labels = np.arange(len(scores)) % 3
    with pytest.raises(ValueError, match=r"labels must have shape \[30\]"):
        sq.exchange_stability_report(scores, labels[:-1])
    with pytest.raises(TypeError, match="integer bin labels"):
        sq.exchange_stability_report(scores, labels.astype(float))
    with pytest.raises(ValueError, match="nonnegative bin indices"):
        sq.exchange_stability_report(scores, labels - 1)
    empty = labels.copy()
    empty[empty == 1] = 0
    with pytest.raises(ValueError, match="holds no positive-weight row"):
        sq.exchange_stability_report(scores, empty)
    with pytest.raises(TypeError, match="DOptimality or ProfiledDOptimality"):
        sq.exchange_stability_report(scores, labels, criterion=sq.NormalizedTrace())  # type: ignore[arg-type]


def test_geometry_report_matches_a_direct_raw_score_computation() -> None:
    """The chunked whitened diagnostics equal an untransformed brute-force loop."""
    rng = np.random.default_rng(64)
    scores = _seeded_scores(64, 40)
    weights = rng.uniform(0.3, 2.0, size=len(scores))
    result = sq.optimize_partition(
        scores, weights=weights, n_bins=3, config=sq.DExchangeConfig(max_iter=1, init="random")
    )
    geometry = result.geometry
    assert geometry is not None
    violation, guaranteed, residual, violating, evaluated = _raw_geometry(
        scores, weights, np.asarray(result.labels), 3
    )
    assert geometry.maximum_voronoi_violation == pytest.approx(violation, abs=1e-9)
    assert geometry.guaranteed_violation_gain == pytest.approx(guaranteed, abs=1e-12)
    assert geometry.maximum_separation_residual == pytest.approx(residual, abs=1e-9)
    assert geometry.violating_moves == violating
    assert geometry.evaluated_moves == evaluated


def test_geometry_report_certifies_a_stable_partition_and_its_separation() -> None:
    """Theorem 3 and the leverage lemma must both hold at a terminal D state."""
    rng = np.random.default_rng(91)
    scores = _seeded_scores(91, 48)
    weights = rng.uniform(0.3, 2.0, size=len(scores))
    result = sq.optimize_partition(scores, weights=weights, n_bins=3)
    geometry = result.geometry
    assert geometry is not None
    assert result.exchange_stable is True
    assert geometry.voronoi_consistent is True
    assert geometry.maximum_voronoi_violation < 0
    assert geometry.violating_moves == 0
    assert geometry.guaranteed_violation_gain == 0.0
    assert geometry.separation_certified is True
    assert geometry.maximum_separation_residual <= 1e-8
    assert geometry.evaluated_moves == 2 * len(scores)
    json.dumps(geometry.to_dict(), allow_nan=False)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_geometry_report_bounds_the_remaining_gain_of_a_violating_partition(seed: int) -> None:
    """A Voronoi-violating state must leave at least the guaranteed gain on the table."""
    rng = np.random.default_rng(seed)
    scores = _seeded_scores(seed, 300, 3)
    weights = rng.uniform(0.4, 2.0, size=len(scores))
    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=5,
        config=sq.DExchangeConfig(seed=seed, init="random", max_iter=1, batch_moves=False),
    )
    geometry = result.geometry
    assert geometry is not None
    assert result.exchange_stable is False
    assert geometry.voronoi_consistent is False
    assert geometry.violating_moves > 0
    assert geometry.guaranteed_violation_gain > 0
    assert result.best_remaining_gain >= geometry.guaranteed_violation_gain
    assert geometry.separation_certified is True


def _boundary_fixture(nudge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    r"""Build a labeling whose only improving move crosses a cell boundary.

    The table is exactly symmetric under \(\sigma(x,y)=(-x,y)\): cell 0 is a
    blob at \((+3,0)\), cell 1 is its mirror image, and cell 2 is a mirrored
    pair of blobs at \((0,4)\). Mirror symmetry forces \(\mu_0=\sigma\mu_1\),
    \(\sigma\mu_2=\mu_2\), and \(\sigma I\sigma=I\), so the \(0|1\)
    Mahalanobis bisector is the axis \(x=0\) exactly, whatever the blobs do.

    Two extra rows sit at \((\pm\eta, 0)\) and are labeled across that axis, so
    each is a genuine Voronoi violator by a definite margin far above rounding
    yet ``nudge``-small in distance. The pair keeps the mirror symmetry, so the
    bisector does not move when they are added.
    """
    rng = np.random.default_rng(19)
    n_per = 20
    right = np.column_stack([3.0 + rng.normal(0, 0.3, n_per), rng.normal(0, 0.3, n_per)])
    left = np.column_stack([-right[:, 0], right[:, 1]])
    top_half = np.column_stack([rng.uniform(0.2, 1.0, n_per), 4.0 + rng.normal(0, 0.3, n_per)])
    top = np.vstack([top_half, np.column_stack([-top_half[:, 0], top_half[:, 1]])])
    scores = np.vstack([right, left, top, np.array([[nudge, 0.0], [-nudge, 0.0]])])
    labels = np.concatenate(
        [
            np.zeros(n_per, dtype=np.int64),
            np.ones(n_per, dtype=np.int64),
            np.full(2 * n_per, 2, dtype=np.int64),
            np.array([1, 0], dtype=np.int64),
        ]
    )
    return scores, labels


def _boundary_configs() -> list[sq.PartitionConfig]:
    return [
        sq.DExchangeConfig(gain_tolerance=_BOUNDARY_GAIN_TOLERANCE, solver_restarts=1),
        sq.MahalanobisLloydConfig(gain_tolerance=_BOUNDARY_GAIN_TOLERANCE),
    ]


def test_a_boundary_row_is_not_certifiable_at_tolerance_zero() -> None:
    """The fixture's only improving move is the boundary crossing itself.

    This is the whole defect in one assertion pair: the labeling is stable at
    the tolerance the solver runs at and unstable at tolerance zero, so a
    verification that compares at zero contradicts the certificate the same
    engine issues.
    """
    scores, labels = _boundary_fixture()
    exact = sq.exchange_stability_report(scores, labels, gain_tolerance=0.0)
    assert exact.stable is False
    assert exact.gain_tolerance == 0.0
    assert exact.best_move == (80, 0)
    assert exact.best_gain == pytest.approx(0.002270181625500942, rel=1e-9)

    tolerated = sq.exchange_stability_report(
        scores, labels, gain_tolerance=_BOUNDARY_GAIN_TOLERANCE
    )
    assert tolerated.stable is True
    assert tolerated.best_move is None
    assert tolerated.gain_tolerance == _BOUNDARY_GAIN_TOLERANCE
    assert tolerated.best_gain == pytest.approx(exact.best_gain, abs=0)


@pytest.mark.parametrize("config", _boundary_configs(), ids=["exchange", "lloyd"])
def test_a_boundary_row_within_the_tolerance_keeps_the_partition_compilable(
    config: sq.PartitionConfig,
) -> None:
    """A row a hair past a boundary must not reject an otherwise stable partition.

    Both finite solvers share the terminal geometry check, and both rejected
    this state before the verification was made tolerance-consistent. The same
    thing happens on real data once the sample is large enough for the
    Theorem-3 guaranteed gain to fall under the tolerance: at N = 10^6 it is
    2e-11 against the default 1e-10, and 13 rows in a million land inside it.
    Raising the tolerance puts this 82-row fixture in the same regime.
    """
    scores, labels = _boundary_fixture()
    result = sq.optimize_partition(scores, n_bins=3, initial_labels=labels, config=config)
    assert result.exchange_stable is True
    assert np.array_equal(np.asarray(result.labels), labels)

    geometry = result.geometry
    assert geometry is not None
    assert geometry.gain_tolerance == _BOUNDARY_GAIN_TOLERANCE
    # Both predicates the pre-fix verification used are violated here: the
    # distance gap is positive, and the terminal rule relabels two rows.
    assert geometry.maximum_voronoi_violation > 0.0
    assert geometry.maximum_voronoi_violation < 1e-7
    assert geometry.violating_moves == 2
    # Theorem 3 bounds the exact gain from below, and the exact gain is what the
    # solver stopped on, so only the latter can decide the verdict.
    assert geometry.guaranteed_violation_gain <= geometry.maximum_violation_gain
    assert geometry.maximum_violation_gain == pytest.approx(0.002270181625500942, rel=1e-9)
    assert geometry.maximum_violation_gain <= geometry.gain_tolerance
    assert geometry.maximum_violation_gain > 1e-10  # the default would not certify it
    assert geometry.voronoi_consistent is True
    json.dumps(geometry.to_dict(), allow_nan=False)

    compiled = result.compile_quantizer()
    predicted = np.asarray(compiled.predict_scores(scores))
    # Assignment is untouched: argmin still decides, and the two boundary rows
    # are the only rows the rule and the labels disagree on.
    assert np.flatnonzero(predicted != labels).tolist() == [80, 81]
    assert np.array_equal(predicted, np.asarray(compiled.predict_scores(scores)))


def test_compilation_still_refuses_a_disagreement_its_certificate_does_not_cover() -> None:
    """The tolerance excuses the disagreement; nothing else does."""
    scores, labels = _boundary_fixture()
    result = sq.optimize_partition(
        scores, n_bins=3, initial_labels=labels, config=_boundary_configs()[0]
    )
    assert result.geometry is not None
    uncertified = dataclasses.replace(
        result, geometry=dataclasses.replace(result.geometry, voronoi_consistent=False)
    )
    with pytest.raises(sq.RefusalError, match="more than the gain tolerance"):
        uncertified.compile_quantizer()
    with pytest.raises(sq.RefusalError, match="more than the gain tolerance"):
        dataclasses.replace(result, geometry=None).compile_quantizer()


def test_geometry_reports_are_criterion_specific() -> None:
    """A D result carries only D geometry and a profiled result only profiled geometry."""
    scores = _seeded_scores(5, 80, 3)
    d_result = sq.optimize_partition(scores, n_bins=4)
    profiled = sq.optimize_partition(scores, n_bins=4, criterion=sq.ProfiledDOptimality((0,)))
    assert d_result.geometry is not None
    assert d_result.profiled_geometry is None
    assert profiled.geometry is None
    assert profiled.profiled_geometry is not None
    json.dumps(d_result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(("seed", "n_rows"), _ORACLE_CASES)
def test_certificate_agrees_with_the_exhaustive_oracle(seed: int, n_rows: int) -> None:
    """Branch and bound must prove exactly the exhaustively enumerated optimum."""
    rng = np.random.default_rng(seed)
    scores = _seeded_scores(seed, n_rows)
    weights = rng.uniform(0.3, 2.0, size=n_rows) if seed % 2 else None
    _, optimum = _exhaustive_d_oracle(scores, weights, 3)
    certificate = sq.certify_partition(scores, weights=weights, n_bins=3)
    assert certificate.status == "optimal"
    assert certificate.gap == 0.0
    assert certificate.upper_bound == certificate.objective
    assert certificate.objective == pytest.approx(optimum, abs=1e-10)
    assert certificate.labels.shape == (n_rows,)

    replayed = sq.exchange_stability_report(scores, certificate.labels, weights=weights)
    assert replayed.stable is True
    assert replayed.objective == pytest.approx(certificate.objective, abs=1e-10)


def test_certificate_confirms_or_improves_an_exchange_incumbent() -> None:
    """The two committed fixtures show both certificate outcomes on real incumbents."""
    confirmed_scores = _seeded_scores(1, 8)
    confirmed = sq.optimize_partition(confirmed_scores, n_bins=3)
    confirmed_certificate = sq.certify_partition(
        confirmed_scores, n_bins=3, incumbent=confirmed.labels
    )
    assert confirmed_certificate.status == "optimal"
    assert confirmed_certificate.incumbent_was_optimal is True
    assert confirmed_certificate.objective == pytest.approx(confirmed.objective, abs=1e-10)

    improved_scores = _seeded_scores(15, 10)
    improved_weights = np.random.default_rng(115).uniform(0.3, 2.0, size=10)
    improved = sq.optimize_partition(improved_scores, weights=improved_weights, n_bins=3)
    improved_certificate = sq.certify_partition(
        improved_scores, weights=improved_weights, n_bins=3, incumbent=improved.labels
    )
    assert improved_certificate.status == "optimal"
    assert improved_certificate.incumbent_was_optimal is False
    assert improved_certificate.objective > improved.objective
    assert not np.array_equal(np.asarray(improved_certificate.labels), np.asarray(improved.labels))


def test_certificate_reports_an_outstanding_bound_when_the_budget_runs_out() -> None:
    """An exhausted budget downgrades the status and keeps a genuine ceiling."""
    rng = np.random.default_rng(11)
    scores = _seeded_scores(11, 40, 3)
    weights = rng.uniform(0.4, 2.0, size=len(scores))
    incumbent = sq.optimize_partition(scores, weights=weights, n_bins=4)
    certificate = sq.certify_partition(
        scores,
        weights=weights,
        n_bins=4,
        incumbent=incumbent.labels,
        config=sq.CertificationConfig(max_nodes=200),
    )
    assert certificate.status == "budget_exhausted"
    assert certificate.incumbent_was_optimal is False
    assert certificate.nodes_explored <= 201
    assert certificate.upper_bound >= certificate.objective
    assert certificate.gap == pytest.approx(certificate.upper_bound - certificate.objective)
    assert certificate.gap > 0
    assert certificate.objective >= incumbent.objective - 1e-12
    json.dumps(certificate.to_dict(), allow_nan=False)


def test_certification_refuses_what_it_cannot_certify() -> None:
    """Capacity, criterion, and configuration boundaries all fail by name."""
    scores = _seeded_scores(7, 30)
    with pytest.raises(ValueError, match="exceeding max_rows=10"):
        sq.certify_partition(scores, n_bins=3, config=sq.CertificationConfig(max_rows=10))
    with pytest.raises(ValueError, match="DOptimality only"):
        sq.certify_partition(scores, n_bins=3, criterion=sq.ProfiledDOptimality((0,)))
    with pytest.raises(TypeError, match="CertificationConfig"):
        sq.certify_partition(scores, n_bins=3, config=sq.DExchangeConfig())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_nodes"):
        sq.CertificationConfig(max_nodes=0)
    with pytest.raises(ValueError, match="max_rows must be at most 512"):
        sq.CertificationConfig(max_rows=1024)
    with pytest.raises(TypeError, match="gain_tolerance"):
        sq.CertificationConfig(gain_tolerance="tight")  # type: ignore[arg-type]
    assert sq.CertificationConfig().to_dict()["method"] == "branch_and_bound"


def test_certification_is_deterministic_and_handles_zero_weight_rows() -> None:
    """Zero-weight rows never enter the search but still receive canonical labels."""
    rng = np.random.default_rng(23)
    scores = _seeded_scores(23, 12)
    weights = rng.uniform(0.5, 1.5, size=len(scores))
    weights[3] = 0.0
    first = sq.certify_partition(scores, weights=weights, n_bins=3)
    second = sq.certify_partition(scores, weights=weights, n_bins=3)
    assert first.status == "optimal"
    assert np.array_equal(np.asarray(first.labels), np.asarray(second.labels))
    assert first.objective == pytest.approx(second.objective, abs=0)

    _, optimum = _exhaustive_d_oracle(scores, weights, 3)
    assert first.objective == pytest.approx(optimum, abs=1e-10)
