"""Exact counterexamples and oracles that pin the library's guarantees and refusals.

Each fixture under ``tests/fixtures/counterexamples`` is an exact rational
configuration. These tests check that the library refuses where a fixture shows
a guarantee fails, and agrees with exhaustive or exact oracles where it holds.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import pytest

import scorequant as sq
from scorequant import api
from scorequant.information import binned_information_is_degenerate

from ._oracles import _exhaustive_d_oracle, _o7_exact_plugin

FIXTURES = Path(__file__).parent / "fixtures"


def _canonical_partitions(n_rows: int, n_bins: int) -> list[tuple[int, ...]]:
    return [
        labels
        for labels in product(range(n_bins), repeat=n_rows)
        if labels[0] == 0
        and set(labels) == set(range(n_bins))
        and all(labels[index] <= max(labels[:index]) + 1 for index in range(1, n_rows))
    ]


def test_small_d_exchange_matches_exhaustive_global_oracle() -> None:
    scores = np.array(
        [[-2.0, -1.0], [-1.0, 1.0], [-0.2, -0.4], [0.3, 0.8], [1.1, -0.8], [1.8, 0.7]]
    )
    scores -= scores.mean(axis=0)
    _, optimum = _exhaustive_d_oracle(scores, None, 3)
    result = sq.optimize_partition(
        scores,
        n_bins=3,
        config=sq.DExchangeConfig(seed=8, initializer_restarts=12, max_scans=200),
    )
    assert result.objective == pytest.approx(optimum, abs=1e-10)


def test_d_cell_separation_bound_and_training_geometry() -> None:
    rng = np.random.default_rng(91)
    scores = rng.normal(size=(48, 2))
    scores -= scores.mean(axis=0)
    result = sq.optimize_partition(scores, n_bins=3)
    means = np.asarray(result.transformed_centers)
    metric = np.asarray(result.metric)
    masses = np.asarray(result.cell_weights)
    for first in range(result.n_bins):
        for second in range(first):
            difference = means[first] - means[second]
            separation = difference @ metric @ difference
            assert separation <= 1 / masses[first] + 1 / masses[second] + 1e-8
    assert np.array_equal(
        np.asarray(result.compile_quantizer().predict_scores(scores)),
        np.asarray(result.labels),
    )


def test_d_voronoi_violation_has_positive_gain_lower_bound() -> None:
    rng = np.random.default_rng(2026)
    scores = rng.normal(size=(24, 2))
    weights = rng.uniform(0.2, 2.0, size=len(scores))
    weights /= weights.sum()
    labels = np.tile(np.arange(3), 8)
    masses = np.bincount(labels, weights=weights, minlength=3)
    means = np.asarray(
        [
            np.average(scores[labels == label], axis=0, weights=weights[labels == label])
            for label in range(3)
        ]
    )
    information = np.asarray(sq.binned_fisher_information(scores, labels, weights, n_bins=3))
    metric = np.linalg.inv(information)
    checked = 0
    for row, source in enumerate(labels):
        if masses[source] <= weights[row]:
            continue
        source_residual = scores[row] - means[source]
        q_source = source_residual @ metric @ source_residual
        alpha = weights[row] * masses[source] / (masses[source] - weights[row])
        for destination in range(3):
            if destination == source:
                continue
            destination_residual = scores[row] - means[destination]
            q_destination = destination_residual @ metric @ destination_residual
            if q_source < q_destination:
                continue
            beta = weights[row] * masses[destination] / (masses[destination] + weights[row])
            q_cross = source_residual @ metric @ destination_residual
            determinant_increment = (
                alpha * q_source
                - beta * q_destination
                - alpha * beta * (q_source * q_destination - q_cross**2)
            )
            center_difference = means[destination] - means[source]
            separation = center_difference @ metric @ center_difference
            lower_bound = alpha * beta * separation**2 / 4
            assert determinant_increment >= lower_bound - 1e-10
            checked += 1
    assert checked > 0


def test_unmerged_duplicate_atoms_are_an_exact_boundary_failure() -> None:
    """Split duplicates can be vacuously stable without strict score geometry."""
    fixture_path = FIXTURES / "counterexamples" / "CE-D-UNMERGED-DUPLICATES-001.json"
    assert fixture_path.is_file(), (
        f"counterexample fixture missing at {fixture_path}; the research "
        "workspace may have moved — update FIXTURES"
    )
    fixture = json.loads(fixture_path.read_text())
    scores = tuple(Fraction(row[0]) for row in fixture["scores"])
    weights = tuple(Fraction(value) for value in fixture["weights"])
    labels = tuple(fixture["labels_before"])

    assert sum(weight * score for score, weight in zip(scores, weights, strict=True)) == 0
    assert sum(weight * score**2 for score, weight in zip(scores, weights, strict=True)) == 1
    assert all(labels.count(label) == 1 for label in range(fixture["K"]))
    assert scores[0] == scores[1]
    centroids = {
        label: sum(
            weight * score
            for score, weight, row_label in zip(scores, weights, labels, strict=True)
            if row_label == label
        )
        / sum(
            weight for weight, row_label in zip(weights, labels, strict=True) if row_label == label
        )
        for label in range(fixture["K"])
    }
    # the two coincident atoms sit exactly on both of their singleton centroids,
    # so no deterministic score-only rule can separate their labels
    assert (scores[0] - centroids[labels[0]]) ** 2 == 0
    assert (scores[0] - centroids[labels[1]]) ** 2 == 0


def test_adaptive_mahalanobis_lloyd_step_can_decrease_d_objective() -> None:
    scores = np.array(
        [
            [0.1116, 0.4427],
            [-0.2932, 0.6537],
            [-0.5995, -1.2685],
            [-0.6848, -1.5456],
            [0.4810, 0.9521],
            [1.6707, 0.9370],
            [0.1689, 1.7090],
            [-0.8548, -1.8805],
        ]
    )
    labels = np.array([1, 0, 0, 1, 2, 2, 2, 1])
    weights = np.full(len(scores), 1 / len(scores))
    information = np.asarray(sq.binned_fisher_information(scores, labels, weights, n_bins=3))
    means = np.asarray([scores[labels == label].mean(axis=0) for label in range(3)])
    residuals = scores[:, None, :] - means[None, :, :]
    distances = np.einsum("nkp,pq,nkq->nk", residuals, np.linalg.inv(information), residuals)
    updated_labels = np.argmin(distances, axis=1)
    updated = np.asarray(sq.binned_fisher_information(scores, updated_labels, weights, n_bins=3))
    decrease = np.linalg.slogdet(information)[1] - np.linalg.slogdet(updated)[1]
    assert np.array_equal(updated_labels, [0, 0, 1, 1, 2, 1, 0, 1])
    assert decrease == pytest.approx(0.136521, abs=2e-6)


def test_global_profiled_ds_partition_can_violate_its_own_metric_rule() -> None:
    raw = [(4, -4), (-5, 2), (-1, 0), (-5, -1), (2, -2), (4, 3), (2, 4), (2, -4)]
    n_rows, n_bins = 8, 3
    weight = Fraction(1, n_rows)
    mean = [Fraction(sum(row[column] for row in raw), n_rows) for column in range(2)]
    scores = [[Fraction(row[column]) - mean[column] for column in range(2)] for row in raw]

    def statistics(labels: tuple[int, ...]) -> tuple[list[Fraction], list[list[Fraction]]]:
        masses = [Fraction(0)] * n_bins
        moments = [[Fraction(0), Fraction(0)] for _ in range(n_bins)]
        for row, label in enumerate(labels):
            masses[label] += weight
            for column in range(2):
                moments[label][column] += weight * scores[row][column]
        return masses, moments

    def information(labels: tuple[int, ...]) -> list[list[Fraction]]:
        masses, moments = statistics(labels)
        return [
            [
                sum(
                    moments[label][row] * moments[label][column] / masses[label]
                    for label in range(n_bins)
                )
                for column in range(2)
            ]
            for row in range(2)
        ]

    def objective(labels: tuple[int, ...]) -> Fraction:
        matrix = information(labels)
        return matrix[0][0] - matrix[0][1] * matrix[1][0] / matrix[1][1]

    ranked = sorted(
        ((objective(labels), labels) for labels in _canonical_partitions(n_rows, n_bins)),
        reverse=True,
    )
    best, labels = ranked[0]
    matrix = information(labels)
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    inverse = [
        [matrix[1][1] / determinant, -matrix[0][1] / determinant],
        [-matrix[1][0] / determinant, matrix[0][0] / determinant],
    ]
    metric = [
        [inverse[0][0], inverse[0][1]],
        [inverse[1][0], inverse[1][1] - 1 / matrix[1][1]],
    ]
    masses, moments = statistics(labels)
    means = [
        [moments[label][column] / masses[label] for column in range(2)] for label in range(n_bins)
    ]

    def distance(row: int, label: int) -> Fraction:
        residual = [scores[row][column] - means[label][column] for column in range(2)]
        return sum(
            residual[first] * metric[first][second] * residual[second]
            for first in range(2)
            for second in range(2)
        )

    row = 6
    distances = [distance(row, label) for label in range(n_bins)]
    nearest = min(range(n_bins), key=distances.__getitem__)
    assert labels == (0, 1, 2, 1, 2, 0, 0, 2)
    assert best == Fraction(20449, 1920)
    assert best - ranked[1][0] == Fraction(2929, 21120)
    assert nearest == 2
    assert distances[labels[row]] - distances[nearest] == Fraction(8, 195)


def test_ds15_rank_vacuity_diagnosis_does_not_depend_on_the_labeling() -> None:
    """Every onto labeling is degenerate by a wide margin, so the message is stable.

    Two guards can refuse this configuration: the whole-matrix rank test, which
    names the bin budget, and the nuisance-block test, which names the
    parameterization. Only the first is the true cause. The library orders the
    rank test first, but that ordering is only worth something if the rank test
    fires for *every* labeling a solver might land on - otherwise the surviving
    message would be decided by whichever state the platform's linear algebra
    happened to produce, which is exactly how this was first observed to differ
    between macOS and Linux CI.

    The theorem supplies that guarantee: at K = d_lambda + 1 on a centered
    sample the binned information is rank deficient for every feasible labeling.
    This checks the numerical version of it, with the margin, so a future change
    to the rank tolerance cannot silently reopen the platform split.
    """
    fixture = json.loads(
        (FIXTURES / "counterexamples" / "CE-DS-MARGINS-RANK-VACUITY-001.json").read_text()
    )
    scores = np.array([[float(Fraction(v)) for v in row] for row in fixture["scores"]])
    weights = np.array([float(Fraction(w)) for w in fixture["weights"]])
    n_bins = int(fixture["K"])

    worst_ratio = 0.0
    labelings = 0
    for labels in product(range(n_bins), repeat=scores.shape[0]):
        if len(set(labels)) != n_bins:
            continue
        labelings += 1
        binned = sq.information_report(
            scores, np.array(labels), weights=weights, n_bins=n_bins
        ).fisher_binned
        assert binned_information_is_degenerate(binned)
        eigenvalues = np.linalg.eigvalsh(np.asarray(binned))
        worst_ratio = max(worst_ratio, float(eigenvalues.min() / eigenvalues.max()))

    assert labelings > 0
    # Six orders of magnitude below the float64 threshold of 1e-10. The smallest
    # eigenvalue is rounding noise, which is why the sign of a log determinant -
    # the test this replaced - was not reproducible across platforms.
    assert worst_ratio < 1e-14


def _refuse_to_report(*args: object, **kwargs: object) -> object:
    """Stand in for the profiled report so that reaching it is a test failure."""
    raise AssertionError("a degenerate profiled budget must be refused before anything is reported")


def test_ds15_rank_vacuity_is_refused_by_both_public_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The library refuses the vacuous configuration instead of scoring it.

    The exact-arithmetic test above proves the profiled value is identically
    zero at K = d_lambda + 1 on a centered sample. This drives the same fixture
    through the public API, where the same fact has to surface as a refusal
    rather than as a number: ``optimize_partition`` hits a singular initial
    state, and ``fit_quantizer`` would otherwise return a rule whose profiled
    retention is zero, because the soft solver checks n_bins only against the
    Fisher rank, which this configuration satisfies.

    The K = d_lambda + 2 control on the same atoms must still succeed and
    reproduce the fixture's recorded exact value 9/5.
    """
    fixture = json.loads(
        (FIXTURES / "counterexamples" / "CE-DS-MARGINS-RANK-VACUITY-001.json").read_text()
    )
    scores = np.array([[float(Fraction(v)) for v in row] for row in fixture["scores"]])
    weights = np.array([float(Fraction(w)) for w in fixture["weights"]])
    interest = tuple(fixture["poi_indices"])
    vacuous_bins = int(fixture["K"])
    dimension = scores.shape[1]
    assert vacuous_bins == len(fixture["nuisance_indices"]) + 1 == dimension
    np.testing.assert_allclose(weights @ scores, 0.0, atol=0.0)

    criterion = sq.ProfiledDOptimality(interest=interest)
    provenance = sq.ScoreProvenance(kind="exact", reference_point=(0.0,) * dimension)
    for config in (sq.DExchangeConfig(seed=0), sq.MahalanobisLloydConfig(seed=0)):
        with pytest.raises(sq.RefusalError, match="rank at most n_bins"):
            sq.optimize_partition(
                scores,
                weights=weights,
                n_bins=vacuous_bins,
                criterion=criterion,
                config=config,
                provenance=provenance,
            )

    sample = sq.ScoreSample(scores, weights, provenance=provenance)
    with pytest.raises(sq.RefusalError, match="profiled-D fit is degenerate"):
        sq.fit_quantizer(
            sample,
            n_bins=vacuous_bins,
            criterion=criterion,
            config=sq.SoftVoronoiConfig(seed=0),
        )

    # The refusal above is a claim about ordering, not only about wording. The
    # fit builds a retention history by scoring recorded snapshots through
    # profiled_information_report, whose own nuisance guard decides singularity
    # from a factorization whose last bits are platform dependent. Whenever that
    # guard fires first it blames the nuisance parameterization for what is a
    # bin-budget fact, and the refusal below never runs - which is exactly how
    # this regressed. So the invariant is stronger than the message: on a
    # degenerate budget nothing profiled may be reported at all.
    with monkeypatch.context() as patched:
        patched.setattr(
            api,
            "profiled_information_report",
            _refuse_to_report,
        )
        with pytest.raises(sq.RefusalError, match="profiled-D fit is degenerate"):
            sq.fit_quantizer(
                sample,
                n_bins=vacuous_bins,
                criterion=criterion,
                config=sq.SoftVoronoiConfig(seed=0),
            )

    # One more bin lifts the rank ceiling and the value becomes positive. All
    # four atoms are singletons there, which is the state the fixture priced.
    restored = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=vacuous_bins + 1,
        criterion=criterion,
        config=sq.DExchangeConfig(seed=0),
        provenance=provenance,
    )
    expected = fixture["exact_quantities"]["k_equals_d_lambda_plus_2_all_singletons_value"]
    assert restored.objective == pytest.approx(float(np.log(float(Fraction(expected)))), abs=1e-12)


def test_profiled_bins_equal_to_dimension_stay_legal_off_the_centered_class() -> None:
    """The bin guard is centering-agnostic on purpose, so this must not regress.

    The rank ceiling that makes n_bins == dimension vacuous is
    ``sum_b w_b m_b = 0``, which holds only for an exactly centered sample.
    Scores away from the true reference point have a nonzero weighted mean, the
    ceiling rises to n_bins, and the configuration is feasible. Tightening the
    guard to n_bins > dimension for every sample - the obvious reading of
    CE-DS-MARGINS-RANK-VACUITY-001 - would refuse this legitimate case.
    """
    rng = np.random.default_rng(11)
    scores = rng.normal(size=(60, 2)) + 0.9
    weights = np.full(60, 1.0 / 60)
    assert not np.allclose(weights @ scores, 0.0, atol=1e-3)

    result = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=scores.shape[1],
        criterion=sq.ProfiledDOptimality(interest=(0,)),
        config=sq.DExchangeConfig(seed=0),
        provenance=sq.ScoreProvenance(kind="exact", reference_point=(0.0, 0.0)),
    )
    assert np.isfinite(result.objective)


def test_o6_plugin_retention_influence_function_matches_finite_differences() -> None:
    """O6 (RETENTION-PLUGIN-CLT-FROZEN-SCALAR): finite identities behind the CLT.

    Pins, on a deterministic sample with one empty cell: the residual-sum-of-
    squares form of the plug-in ratio, agreement with the library's scalar
    retention, the exact zero mean of the plug-in influence function, and the
    closed form ``psi = ((1 - eta) S^2 - (S - c_Z)^2) / v`` against Gateaux
    finite differences of the same functional on the empirical measure.
    """
    rng = np.random.default_rng(20260905)
    n, n_bins = 40, 4
    scores = rng.standard_t(df=5, size=n) + 0.3 * rng.integers(0, 2, size=n)
    labels = rng.integers(0, n_bins, size=n)
    labels[labels == 3] = 1  # cell 3 stays empty on purpose

    def functional(weights: np.ndarray) -> float:
        w = weights / weights.sum()
        p = np.bincount(labels, weights=w, minlength=n_bins)
        m = np.bincount(labels, weights=w * scores, minlength=n_bins)
        v = float(np.sum(w * scores * scores))
        between = float(np.sum(np.divide(m * m, p, out=np.zeros(n_bins), where=p > 0)))
        return between / v

    counts = np.bincount(labels, minlength=n_bins).astype(float)
    sums = np.bincount(labels, weights=scores, minlength=n_bins)
    means = np.divide(sums, counts, out=np.zeros(n_bins), where=counts > 0)
    v_hat = float(np.mean(scores * scores))
    eta_hat = functional(np.ones(n))
    psi_hat = ((1.0 - eta_hat) * scores * scores - (scores - means[labels]) ** 2) / v_hat

    rss = float(np.sum((scores - means[labels]) ** 2))
    assert eta_hat == pytest.approx(1.0 - rss / float(np.sum(scores * scores)), abs=1e-14)
    report = sq.information_report(scores[:, None], labels, n_bins=n_bins)
    library = float(report.geometric_mean_retention)
    assert library == pytest.approx(eta_hat, abs=1e-12)
    assert float(np.sum(psi_hat)) == pytest.approx(0.0, abs=1e-12)

    eps = 1e-6
    for j in range(n):
        plus = np.full(n, (1 - eps) / n)
        plus[j] += eps
        minus = np.full(n, (1 + eps) / n)
        minus[j] -= eps
        gateaux = (functional(plus) - functional(minus)) / (2 * eps)
        assert gateaux == pytest.approx(psi_hat[j], abs=1e-7)


def test_o6_audit_eta_zero_law_has_zero_influence_variance_with_many_atoms() -> None:
    """CE-O6-ETA-ZERO-MULTIATOM-VARIANCE-001 (AUDIT-SCORE-ORACLE-ROBUSTNESS).

    At eta = 0 the root equation (s - c_b)^2 = (1 - eta) s^2 is the identity,
    so every law whose cell means vanish has psi = 0 identically: sigma^2 = 0
    is not restricted to two atoms per cell, and an atomless cell does not
    imply (A4) unless eta > 0. The fixture has four atoms in cell 0.
    """
    fixture_path = FIXTURES / "counterexamples" / "CE-O6-ETA-ZERO-MULTIATOM-VARIANCE-001.json"
    assert fixture_path.is_file(), f"counterexample fixture missing at {fixture_path}"
    fixture = json.loads(fixture_path.read_text())
    atoms = [
        (Fraction(row[0]), label, Fraction(weight))
        for row, label, weight in zip(
            fixture["scores"], fixture["labels_before"], fixture["weights"], strict=True
        )
    ]
    n_bins = fixture["K"]
    assert sum(w for _, _, w in atoms) == 1
    p = [sum(w for _, z, w in atoms if z == b) for b in range(n_bins)]
    m = [sum(w * s for s, z, w in atoms if z == b) for b in range(n_bins)]
    v = sum(w * s * s for s, _, w in atoms)
    assert all(x == 0 for x in m) and v > 0 and all(x > 0 for x in p)
    eta = sum(m[b] ** 2 / p[b] for b in range(n_bins)) / v
    assert eta == 0
    psi = [((1 - eta) * s * s - (s - m[z] / p[z]) ** 2) / v for s, z, _ in atoms]
    assert psi == [0] * len(atoms)
    assert sum(1 for _, z, _ in atoms if z == 0) == 4
    # The same numbers through the library on the weighted atom sample.
    report = sq.information_report(
        np.array([[float(s)] for s, _, _ in atoms]),
        np.array([z for _, z, _ in atoms]),
        np.array([float(w) for _, _, w in atoms]),
        n_bins=n_bins,
    )
    assert float(report.geometric_mean_retention) == 0.0


def test_o7_vector_plugin_influence_function_matches_exact_gateaux_differences() -> None:
    """O7 (RETENTION-PLUGIN-CLT-FROZEN-VECTOR): finite identities behind the vector CLT.

    Pins, in exact rational arithmetic on a d = 2 sample with a duplicate atom
    and a declared-empty cell: the within-cell-scatter form of V_hat - I_hat_Z,
    the exact zero mean of the influence function, and the closed form
    ``psi_r = r [2 S^T I_Z^{-1} c_Z - c_Z^T I_Z^{-1} c_Z - S^T V^{-1} S]`` of the
    determinant ratio r against exact Gateaux difference quotients whose O(eps)
    remainder halves when eps halves; then agreement of r^(1/2) with the
    library's ``geometric_mean_retention``.
    """
    rows = [
        (2, 1),
        (-1, 3),
        (0, -2),
        (3, 3),
        (-2, -1),
        (1, 0),
        (1, 0),
        (-3, 2),
        (2, -3),
        (0, 1),
        (4, -1),
        (-1, -1),
    ]
    labels = [0, 1, 2, 0, 1, 2, 2, 1, 3, 3, 0, 1]
    n_bins = 5  # cell 4 is declared and stays empty
    n = len(rows)
    scores = [[Fraction(a), Fraction(b)] for a, b in rows]
    uniform = [Fraction(1, n)] * n
    ratio, psi_r, v, i_z = _o7_exact_plugin(scores, labels, uniform, n_bins)
    assert 0 < ratio < 1
    assert sum(psi_r) == 0
    counts = [labels.count(b) for b in range(n_bins)]
    means = [
        [
            sum(scores[k][i] for k in range(n) if labels[k] == b) / counts[b]
            if counts[b]
            else Fraction(0)
            for i in range(2)
        ]
        for b in range(n_bins)
    ]
    within = [
        [
            sum(
                (s[i] - means[z][i]) * (s[j] - means[z][j])
                for s, z in zip(scores, labels, strict=True)
            )
            / n
            for j in range(2)
        ]
        for i in range(2)
    ]
    assert [[v[i][j] - i_z[i][j] for j in range(2)] for i in range(2)] == within
    for j in range(n):
        gaps = []
        for k in (10, 11, 12):
            eps = Fraction(1, 2**k)
            w = [(1 - eps) / n] * n
            w[j] += eps
            ratio_eps, _, _, _ = _o7_exact_plugin(scores, labels, w, n_bins)
            gaps.append((ratio_eps - ratio) / eps - psi_r[j])
        assert abs(gaps[0]) < Fraction(1, 100)
        for a, b in ((gaps[0], gaps[1]), (gaps[1], gaps[2])):
            assert b != 0 and abs(a / b - 2) < Fraction(1, 5)
    report = sq.information_report(np.array(rows, dtype=float), np.array(labels), n_bins=n_bins)
    assert float(report.geometric_mean_retention) == pytest.approx(
        math.sqrt(float(ratio)), abs=1e-12
    )


def test_o7_ellipsoid_law_has_zero_influence_variance_on_the_whole_circle() -> None:
    """CE-O7-ELLIPSOID-ZERO-VARIANCE-001 (RETENTION-PLUGIN-VECTOR, 6 Sep 2026).

    Four cells related by quarter turns with two atoms each give E[S] = 0,
    V = (25/2) I, I_Z = (9/2) I, eta_D = 9/25 and psi = 0 at every atom; as a
    polynomial, psi_r is a multiple of the circle |s - (25/3, 0)|^2 = (20/3)^2,
    the cell-0 ellipsoid of O7.4(a), so every law on that circle with the same
    cell mean - atomless ones included - has sigma^2 = 0. O6's "atomless cell
    implies sigma^2 > 0 when eta > 0" does not lift to d >= 2.
    """
    fixture_path = FIXTURES / "counterexamples" / "CE-O7-ELLIPSOID-ZERO-VARIANCE-001.json"
    assert fixture_path.is_file(), f"counterexample fixture missing at {fixture_path}"
    fixture = json.loads(fixture_path.read_text())
    scores = [[Fraction(x) for x in row] for row in fixture["scores"]]
    labels = list(fixture["labels_before"])
    weights = [Fraction(w) for w in fixture["weights"]]
    n_bins = fixture["K"]
    assert sum(weights) == 1 and n_bins == 4 and len(scores) == 8
    assert [sum(w * s[i] for s, w in zip(scores, weights, strict=True)) for i in range(2)] == [0, 0]
    ratio, psi_r, v, i_z = _o7_exact_plugin(scores, labels, weights, n_bins)
    assert v == [[Fraction(25, 2), 0], [0, Fraction(25, 2)]]
    assert i_z == [[Fraction(9, 2), 0], [0, Fraction(9, 2)]]
    assert ratio == Fraction(81, 625)
    assert psi_r == [0] * 8
    # psi_r as a polynomial in s for cell 0, c_0 = (3, 0):
    # r [2 s^T I_Z^{-1} c - c^T I_Z^{-1} c - s^T V^{-1} s]
    c0 = [Fraction(3), Fraction(0)]

    def psi_r_at(s: list[Fraction]) -> Fraction:
        return ratio * (
            2 * Fraction(2, 9) * (s[0] * c0[0] + s[1] * c0[1])
            - Fraction(2, 9) * (c0[0] ** 2 + c0[1] ** 2)
            - Fraction(2, 25) * (s[0] ** 2 + s[1] ** 2)
        )

    def circle(s: list[Fraction]) -> Fraction:
        return s[0] ** 2 + s[1] ** 2 - Fraction(50, 3) * s[0] + 25

    for num in range(-7, 8):
        t = Fraction(num, 3)
        on_circle = [
            Fraction(25, 3) + Fraction(20, 3) * (1 - t * t) / (1 + t * t),
            Fraction(20, 3) * 2 * t / (1 + t * t),
        ]
        assert circle(on_circle) == 0 and psi_r_at(on_circle) == 0
        generic = [t + Fraction(1, 7), t * t - 2]
        assert psi_r_at(generic) + Fraction(2, 25) * ratio * circle(generic) == 0
    report = sq.information_report(
        np.array([[float(x) for x in s] for s in scores]),
        np.array(labels),
        np.array([float(w) for w in weights]),
        n_bins=n_bins,
    )
    assert float(report.geometric_mean_retention) == pytest.approx(0.36, abs=1e-14)


def test_o7_audit_unit_retention_singular_sample_fixture() -> None:
    """CE-O7-UNIT-RETENTION-SINGULAR-SAMPLE-001 (AUDIT-RETENTION-PLUGIN-VECTOR).

    At eta_D = 1 (S = c_Z a.s., cell means (1, 0) and (0, 1)) the plug-in is 1
    only on samples whose occupied cells span R^d: a sample confined to one
    cell has V_hat singular, the estimator's own functional returns 0 by its
    det V_hat = 0 convention, and the library projects the null direction out
    and reports 1 on the retained subspace. Every composition with both cells
    occupied gives the ratio exactly 1.
    """
    fixture_path = FIXTURES / "counterexamples" / "CE-O7-UNIT-RETENTION-SINGULAR-SAMPLE-001.json"
    assert fixture_path.is_file(), f"counterexample fixture missing at {fixture_path}"
    fixture = json.loads(fixture_path.read_text())
    law = fixture["population_law"]
    atoms = [[Fraction(x) for x in s] for s in law["atoms"]]
    weights = [Fraction(w) for w in law["weights"]]
    ratio, psi_r, v, i_z = _o7_exact_plugin(atoms, list(law["labels"]), weights, fixture["K"])
    assert ratio == 1 and v == i_z == [[Fraction(1, 2), 0], [0, Fraction(1, 2)]]
    assert psi_r == [0, 0]
    # the failing sample: both draws in cell 0
    scores = [[Fraction(x) for x in s] for s in fixture["scores"]]
    labels = list(fixture["labels_before"])
    assert labels == [0, 0]
    v_hat = [[sum(s[i] * s[j] for s in scores) / len(scores) for j in range(2)] for i in range(2)]
    assert v_hat[0][0] * v_hat[1][1] - v_hat[0][1] * v_hat[1][0] == 0
    report = sq.information_report(
        np.array([[float(x) for x in s] for s in scores]), np.array(labels), n_bins=fixture["K"]
    )
    assert int(report.effective_rank) == 1
    assert float(report.geometric_mean_retention) == pytest.approx(1.0, abs=1e-12)
    # every composition with both cells occupied has plug-in exactly 1
    for n in range(2, 7):
        for n0 in range(1, n):
            sample = [atoms[0]] * n0 + [atoms[1]] * (n - n0)
            sample_labels = [0] * n0 + [1] * (n - n0)
            r_s, _, _, _ = _o7_exact_plugin(
                sample, sample_labels, [Fraction(1, n)] * n, fixture["K"]
            )
            assert r_s == 1


def _o8_moments(
    atoms: list[list[Fraction]], weights: list[Fraction], labels: list[int], n_bins: int
) -> tuple[list[Fraction], list[list[Fraction]], list[list[Fraction]], list[list[Fraction]]]:
    """p_b, c_b, V = E[s s^T], I_Z = sum_b p_b c_b c_b^T in exact arithmetic."""
    d = len(atoms[0])
    p = [Fraction(0)] * n_bins
    m = [[Fraction(0)] * d for _ in range(n_bins)]
    v = [[Fraction(0)] * d for _ in range(d)]
    for s, w, z in zip(atoms, weights, labels, strict=True):
        p[z] += w
        for i in range(d):
            m[z][i] += w * s[i]
            for j in range(d):
                v[i][j] += w * s[i] * s[j]
    c = [[x / p[b] for x in m[b]] if p[b] > 0 else [Fraction(0)] * d for b in range(n_bins)]
    i_z = [[Fraction(0)] * d for _ in range(d)]
    for b in range(n_bins):
        if p[b] > 0:
            for i in range(d):
                for j in range(d):
                    i_z[i][j] += m[b][i] * m[b][j] / p[b]
    return p, c, v, i_z


def _o8_det(m: list[list[Fraction]]) -> Fraction:
    if len(m) == 1:
        return m[0][0]
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def _o8_fixture(name: str) -> dict:
    path = FIXTURES / "counterexamples" / f"{name}.json"
    assert path.is_file(), f"counterexample fixture missing at {path}"
    return json.loads(path.read_text())


def test_o8_rho_min_fixture_fixed_score_error_unbounded_reporting_ratio() -> None:
    """CE-SCORE-ERROR-RHO-MIN-NECESSARY-001 (SCORE-ERROR-RETENTION-BUDGET).

    d = 2, three cells, E[s] = 0 exactly; the proxy adds (0, kappa) on the
    singleton cell only. Along delta = 1/2, 1/10, 1/100 the whitened score error
    eps^2 stays in [0.026, 0.028] while the reported-over-true retention ratio
    grows 1.75 -> 5.1 -> 42.8, because the least retained eigenvalue rho_min
    tends to 0. The B1 core inequality holds on every member: the gap is the
    spurious-information term eps_R^2 = sum_b p_b e_b^T I_Z^{-1} e_b, not eps.
    """
    fixture = _o8_fixture("CE-SCORE-ERROR-RHO-MIN-NECESSARY-001")
    kappa = Fraction(fixture["exact_quantities"]["kappa"])
    ratios = []
    for member in fixture["exact_quantities"]["family_by_delta"]:
        atoms = [[Fraction(x) for x in s] for s in member["atoms"]]
        weights = [Fraction(w) for w in member["weights"]]
        labels = list(fixture["labels_before"])
        assert sum(weights) == 1
        for i in range(2):
            assert sum(w * s[i] for s, w in zip(atoms, weights, strict=True)) == 0
        proxy = [list(s) for s in atoms]
        proxy[4] = [atoms[4][0], atoms[4][1] + kappa]
        _, _, v, i_z = _o8_moments(atoms, weights, labels, fixture["K"])
        _, _, vt, it = _o8_moments(proxy, weights, labels, fixture["K"])
        true_ratio = _o8_det(i_z) / _o8_det(v)
        proxy_ratio = _o8_det(it) / _o8_det(vt)
        assert true_ratio == Fraction(member["det_ratio_true"])
        assert proxy_ratio == Fraction(member["det_ratio_proxy"])
        # whitened score error: eps^2 = E[e^T V^{-1} e] = w_5 kappa^2 (V^{-1})_22
        v_inv_22 = v[0][0] / _o8_det(v)
        eps2 = weights[4] * kappa * kappa * v_inv_22
        assert eps2 == Fraction(member["eps2"])
        assert Fraction(26, 1000) <= eps2 <= Fraction(28, 1000)
        # B1 core in the second coordinate: (a^T(I~ - I)a - a^T E_Z a)^2 <= 4 (a^T I a)(a^T E_Z a)
        e_z_22 = weights[4] * kappa * kappa  # only the singleton cell carries e_b
        lhs = (it[1][1] - i_z[1][1]) - e_z_22
        assert lhs * lhs <= 4 * i_z[1][1] * e_z_22
        ratios.append(proxy_ratio / true_ratio)
        # library agreement: geometric_mean_retention^2 is the determinant ratio
        report = sq.information_report(
            np.array([[float(x) for x in s] for s in proxy]),
            np.array(labels),
            n_bins=fixture["K"],
            weights=np.array([float(w) for w in weights]),
        )
        retention = float(report.geometric_mean_retention)
        assert retention**2 == pytest.approx(float(proxy_ratio), rel=1e-10)
    assert ratios[0] < ratios[1] < ratios[2]
    assert ratios[2] > 1000  # squared ratio; eta ratio > 30


def test_o8_auc_invariant_fixture_same_ranking_different_reported_retention() -> None:
    """CE-AUC-INVARIANT-PROXY-RETENTION-001 (SCORE-ERROR-RETENTION-BUDGET, O8.5).

    Two strictly increasing distortions of a four-atom scalar law keep the labels
    of corresponding rank-cut thresholds and the true retention 9/10, and report 100/101 and
    5105/10006 through the library's own information_report on the proxy.
    """
    fixture = _o8_fixture("CE-AUC-INVARIANT-PROXY-RETENTION-001")
    atoms = [[Fraction(x) for x in s] for s in fixture["scores"]]
    weights = [Fraction(w) for w in fixture["weights"]]
    labels = list(fixture["labels_before"])
    _, _, v, i_z = _o8_moments(atoms, weights, labels, 2)
    assert i_z[0][0] / v[0][0] == Fraction(9, 10)
    expected = {
        "compress_within_cells": Fraction(100, 101),
        "stretch_one_atom": Fraction(5105, 10006),
    }
    for row in fixture["exact_quantities"]["distortions"]:
        proxy = [[Fraction(x) for x in s] for s in row["proxy_scores"]]
        assert all(a[0] < b[0] for a, b in zip(proxy[:-1], proxy[1:], strict=True))
        # every threshold between consecutive atoms gives the same labels on truth and proxy
        for k in range(1, 4):
            true_labels = [int(i >= k) for i in range(4)]
            assert [int(s[0] > (proxy[k - 1][0] + proxy[k][0]) / 2) for s in proxy] == true_labels
            assert [int(s[0] > (atoms[k - 1][0] + atoms[k][0]) / 2) for s in atoms] == true_labels
        _, _, vt, it = _o8_moments(proxy, weights, labels, 2)
        assert it[0][0] / vt[0][0] == expected[row["distortion"]]
        report = sq.information_report(
            np.array([[float(x) for x in s] for s in proxy]), np.array(labels), n_bins=2
        )
        assert float(report.geometric_mean_retention) == pytest.approx(
            float(expected[row["distortion"]]), abs=1e-12
        )


@pytest.mark.parametrize(
    ("fixture_id", "maximum_ratio", "batch_ratio"),
    [
        ("CE-D-COMPILE-SINGLETON-TIE-001", Fraction(3, 2), Fraction(1)),
        ("CE-D-COMPILE-BATCH-TOLERANCE-001", Fraction(3), Fraction(4)),
    ],
)
def test_compile_tolerance_boundary_exact_ratios(
    fixture_id: str, maximum_ratio: Fraction, batch_ratio: Fraction
) -> None:
    fixture = json.loads((FIXTURES / "counterexamples" / f"{fixture_id}.json").read_text())
    scores = [Fraction(row[0]) for row in fixture["scores"]]
    labels = fixture["labels_before"]
    n_bins = fixture["K"]

    def information(assignment: list[int]) -> Fraction:
        total = Fraction(0)
        for cell in range(n_bins):
            members = [
                score for score, label in zip(scores, assignment, strict=True) if label == cell
            ]
            if members:
                total += sum(members, Fraction(0)) ** 2 / len(members)
        return total

    before = information(labels)
    ratios = []
    for row, source in enumerate(labels):
        if labels.count(source) == 1:
            continue
        for destination in range(n_bins):
            if destination != source:
                moved = labels.copy()
                moved[row] = destination
                ratios.append(information(moved) / before)
    assert max(ratios) == maximum_ratio
    predicted = fixture["labels_after_or_optimum"]
    centers = [
        sum(
            (score for score, label in zip(scores, labels, strict=True) if label == cell),
            Fraction(0),
        )
        / labels.count(cell)
        for cell in range(n_bins)
    ]
    assert predicted == [
        min(range(n_bins), key=lambda cell: (score - centers[cell]) ** 2) for score in scores
    ]
    assert information(predicted) / before == batch_ratio
    tolerance = float(Fraction(fixture["exact_quantities"]["gain_tolerance"]))
    assert math.log(float(maximum_ratio)) < tolerance
    if "SINGLETON" in fixture_id:
        assert labels.count(labels[1]) == 1 and predicted[1] != labels[1]
    else:
        assert math.log(float(batch_ratio)) > tolerance
