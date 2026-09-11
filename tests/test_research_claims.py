from __future__ import annotations

import json
import math
from fractions import Fraction
from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np
import pytest

import scorequant as sq
from scorequant import api
from scorequant.information import binned_information_is_degenerate

from ._oracles import _exhaustive_d_oracle, _o7_exact_plugin

# The research workspace holding the counterexample bank; if it moves, update
# this constant and tests/test_research_registry.py together.
RESEARCH_WORKSPACE = Path(__file__).parents[1] / "agenticresearch"


def _canonical_partitions(n_rows: int, n_bins: int) -> list[tuple[int, ...]]:
    return [
        labels
        for labels in product(range(n_bins), repeat=n_rows)
        if labels[0] == 0
        and set(labels) == set(range(n_bins))
        and all(labels[index] <= max(labels[:index]) + 1 for index in range(1, n_rows))
    ]


def _information(scores: np.ndarray, labels: tuple[int, ...], n_bins: int) -> np.ndarray:
    return np.asarray(sq.binned_fisher_information(scores, labels, n_bins=n_bins))


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
    fixture_path = RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-D-UNMERGED-DUPLICATES-001.json"
    assert fixture_path.is_file(), (
        f"counterexample fixture missing at {fixture_path}; the research "
        "workspace may have moved — update RESEARCH_WORKSPACE"
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


def test_global_e_partition_can_violate_simple_eigenvector_rule() -> None:
    scores = np.array(
        [
            [-0.226534, 0.428773],
            [-0.629944, -1.223406],
            [1.253439, -0.109445],
            [1.807897, 0.734952],
            [-1.520937, -0.061786],
            [-0.488606, -0.002247],
            [0.710355, 1.154412],
            [-0.905669, -0.921253],
        ]
    )
    scores -= scores.mean(axis=0)
    ranked = sorted(
        (
            (np.linalg.eigvalsh(_information(scores, labels, 3))[0], labels)
            for labels in _canonical_partitions(8, 3)
        ),
        reverse=True,
    )
    _, labels = ranked[0]
    matrix = _information(scores, labels, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    direction = eigenvectors[:, 0]
    label_array = np.asarray(labels)
    means = np.asarray([scores[label_array == label].mean(axis=0) for label in range(3)])
    row = 7
    distances = np.square((scores[row] - means) @ direction)
    assert labels == (0, 1, 1, 2, 0, 0, 0, 1)
    assert eigenvalues[1] - eigenvalues[0] > 1e-4
    assert int(np.argmin(distances)) == 2
    assert distances[labels[row]] - distances.min() > 0.06


def _exact_ds_cells(
    scores: list[list[Fraction]],
    labels: tuple[int, ...],
    n_bins: int,
    weights: list[Fraction] | None = None,
) -> tuple[list[Fraction], list[list[Fraction]]]:
    row_weights = weights or [Fraction(1, len(scores))] * len(scores)
    masses = [Fraction(0)] * n_bins
    moments = [[Fraction(0), Fraction(0)] for _ in range(n_bins)]
    for row, (label, weight) in enumerate(zip(labels, row_weights, strict=True)):
        masses[label] += weight
        for column in range(2):
            moments[label][column] += weight * scores[row][column]
    return masses, moments


def _exact_binned_information(
    scores: list[list[Fraction]],
    labels: tuple[int, ...],
    n_bins: int,
    weights: list[Fraction] | None = None,
) -> list[list[Fraction]]:
    masses, moments = _exact_ds_cells(scores, labels, n_bins, weights)
    return [
        [
            sum(
                moments[label][row] * moments[label][column] / masses[label]
                for label in range(n_bins)
                if masses[label] > 0
            )
            for column in range(2)
        ]
        for row in range(2)
    ]


def test_global_profiled_ds_optimum_can_be_a_degenerate_tie_class() -> None:
    path = RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-DEGENERATE-GLOBAL-TIE-001.json"
    fixture = json.loads(path.read_text())
    scores = [[Fraction(value) for value in row] for row in fixture["scores"]]
    n_rows, n_bins = 8, 3

    def profiled(labels: tuple[int, ...]) -> Fraction | None:
        info = _exact_binned_information(scores, labels, n_bins)
        if info[1][1] == 0:
            return None
        return info[0][0] - info[0][1] * info[1][0] / info[1][1]

    ranked: list[tuple[Fraction, tuple[int, ...]]] = []
    infeasible: list[tuple[int, ...]] = []
    for labels in _canonical_partitions(n_rows, n_bins):
        value = profiled(labels)
        if value is None:
            infeasible.append(labels)
        else:
            ranked.append((value, labels))
    ranked.sort(reverse=True)
    best = ranked[0][0]
    ties = [labels for value, labels in ranked if value == best]
    next_distinct = next(value for value, _ in ranked if value < best)

    assert len(ranked) == 964
    assert best == Fraction(1083, 4096)
    assert len(ties) == 31
    assert best - next_distinct == Fraction(237, 16640)

    # Every tied optimum refines the reduced bipartition {0,1,2,4,6,7} | {3,5}
    # and has two cells with exactly coincident projected centroids.
    group = {0, 1, 2, 4, 6, 7}
    for labels in ties:
        cells = [{row for row in range(n_rows) if labels[row] == label} for label in range(n_bins)]
        assert all(cell <= group or cell <= (set(range(n_rows)) - group) for cell in cells)
        info = _exact_binned_information(scores, labels, n_bins)
        slope = info[0][1] / info[1][1]
        masses, moments = _exact_ds_cells(scores, labels, n_bins)
        projected = sorted(
            moments[label][0] / masses[label] - slope * moments[label][1] / masses[label]
            for label in range(n_bins)
        )
        assert projected in (
            [Fraction(-19, 64), Fraction(-19, 64), Fraction(57, 64)],
            [Fraction(-19, 64), Fraction(57, 64), Fraction(57, 64)],
        )

    # The unique infeasible refinement has an exactly singular nuisance block and
    # a generalized pseudo-inverse value strictly above the feasible optimum.
    assert len(infeasible) == 2
    singular = (0, 0, 1, 2, 1, 2, 0, 1)
    assert singular in infeasible
    info = _exact_binned_information(scores, singular, n_bins)
    assert info[1][1] == 0
    assert info[0][1] == 0
    assert info[0][0] == Fraction(1191, 4096)
    assert info[0][0] > best


def test_symmetric_wasted_cells_defeat_the_efficient_semimetric_rule() -> None:
    fixture = json.loads(
        (RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-POP-WASTED-CELLS-001.json").read_text()
    )
    scores = [[Fraction(value) for value in row] for row in fixture["scores"]]
    labels = tuple(fixture["labels_after_or_optimum"])
    coarse = tuple(0 if scores[row][0] < 0 else 1 for row in range(8))

    fine_info = _exact_binned_information(scores, labels, 4)
    coarse_info = _exact_binned_information(scores, coarse, 2)
    assert coarse_info[1][1] == 0
    assert coarse_info[0][0] == Fraction(4)
    assert fine_info[1][1] == Fraction(9, 4)
    assert fine_info[0][1] == 0
    profiled = fine_info[0][0] - fine_info[0][1] * fine_info[1][0] / fine_info[1][1]
    assert profiled == Fraction(4)

    masses, moments = _exact_ds_cells(scores, labels, 4)
    determinant = fine_info[0][0] * fine_info[1][1] - fine_info[0][1] * fine_info[1][0]
    inverse = [
        [fine_info[1][1] / determinant, -fine_info[0][1] / determinant],
        [-fine_info[1][0] / determinant, fine_info[0][0] / determinant],
    ]
    metric = [row[:] for row in inverse]
    metric[1][1] -= 1 / fine_info[1][1]
    means = [[moments[label][column] / masses[label] for column in range(2)] for label in range(4)]
    slope = fine_info[0][1] / fine_info[1][1]
    projected = [means[label][0] - slope * means[label][1] for label in range(4)]
    assert sorted(projected) == [Fraction(-2), Fraction(-2), Fraction(2), Fraction(2)]

    def distance(row: int, label: int) -> Fraction:
        residual = [scores[row][column] - means[label][column] for column in range(2)]
        return sum(
            residual[first] * metric[first][second] * residual[second]
            for first in range(2)
            for second in range(2)
        )

    for row, label in enumerate(labels):
        distances = [distance(row, other) for other in range(4)]
        assert distances[label] == min(distances)


def test_ds13_leverage_bound_at_every_stable_state_with_vector_nuisance() -> None:
    """DS13 audit pin: the leverage bound holds at every exchange-stable state.

    Exhaustive exact verification in a configuration class the original
    packet evidence never exercised: d=3 with a two-dimensional nuisance
    block. All canonical surjective labelings are enumerated, one-point
    exchange stability is decided by exact determinant-ratio comparison, and
    the DS13 bound s_aa - s_bb <= beta * q_aa * q_bb is asserted for every
    admissible move at every stable state, including moves whose destination
    nuisance block is exactly singular.
    """
    scores = [
        [Fraction(2), Fraction(1), Fraction(0)],
        [Fraction(-1), Fraction(1), Fraction(1)],
        [Fraction(0), Fraction(-2), Fraction(1)],
        [Fraction(1), Fraction(0), Fraction(-1)],
        [Fraction(-2), Fraction(-1), Fraction(0)],
        [Fraction(1), Fraction(2), Fraction(2)],
        [Fraction(-1), Fraction(-1), Fraction(-2)],
    ]
    weight = Fraction(1, 7)
    n_bins = 3
    nuisance = (1, 2)

    def determinant(matrix: list[list[Fraction]]) -> Fraction:
        size = len(matrix)
        if size == 1:
            return matrix[0][0]
        if size == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        total = Fraction(0)
        for column in range(size):
            minor = [row[:column] + row[column + 1 :] for row in matrix[1:]]
            total += (-1 if column % 2 else 1) * matrix[0][column] * determinant(minor)
        return total

    def inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
        size = len(matrix)
        full = determinant(matrix)
        cofactors = [
            [
                (-1 if (i + j) % 2 else 1)
                * determinant([row[:j] + row[j + 1 :] for k, row in enumerate(matrix) if k != i])
                for j in range(size)
            ]
            for i in range(size)
        ]
        return [[cofactors[j][i] / full for j in range(size)] for i in range(size)]

    def binned(labels: tuple[int, ...]) -> tuple[list[Fraction], list[list[Fraction]]]:
        masses = [Fraction(0)] * n_bins
        moments = [[Fraction(0)] * 3 for _ in range(n_bins)]
        for row, label in enumerate(labels):
            masses[label] += weight
            for column in range(3):
                moments[label][column] += weight * scores[row][column]
        info = [[Fraction(0)] * 3 for _ in range(3)]
        for mass, moment in zip(masses, moments, strict=True):
            if mass == 0:
                continue
            for i in range(3):
                for j in range(3):
                    info[i][j] += moment[i] * moment[j] / mass
        return masses, info

    def dets(labels: tuple[int, ...]) -> tuple[Fraction, Fraction]:
        _, info = binned(labels)
        lam = [[info[i][j] for j in nuisance] for i in nuisance]
        return determinant(info), determinant(lam)

    def quadratic(u: list[Fraction], g: list[list[Fraction]]) -> Fraction:
        return sum(u[i] * g[i][j] * u[j] for i in range(3) for j in range(3))

    feasible = stable = checked = degenerate_destinations = 0
    max_ratio = Fraction(0)
    for labels in _canonical_partitions(7, n_bins):
        masses, info = binned(labels)
        lam = [[info[i][j] for j in nuisance] for i in nuisance]
        det_info, det_lam = determinant(info), determinant(lam)
        if det_info <= 0 or det_lam <= 0:
            continue
        feasible += 1
        counts = [labels.count(label) for label in range(n_bins)]
        improvable = False
        for row, source in enumerate(labels):
            if counts[source] <= 1:
                continue
            for destination in range(n_bins):
                if destination == source:
                    continue
                moved = labels[:row] + (destination,) + labels[row + 1 :]
                moved_info, moved_lam = dets(moved)
                if moved_info > 0 and moved_lam > 0 and moved_info * det_lam > det_info * moved_lam:
                    improvable = True
                    break
            if improvable:
                break
        if improvable:
            continue
        stable += 1
        full_inverse = inverse(info)
        lam_inverse = inverse(lam)
        metric = [row[:] for row in full_inverse]
        for a, i in enumerate(nuisance):
            for b, j in enumerate(nuisance):
                metric[i][j] -= lam_inverse[a][b]
        cell_moments = [[Fraction(0)] * 3 for _ in range(n_bins)]
        for row, label in enumerate(labels):
            for column in range(3):
                cell_moments[label][column] += weight * scores[row][column]
        means = [
            [cell_moments[label][column] / masses[label] for column in range(3)]
            for label in range(n_bins)
        ]
        for row, source in enumerate(labels):
            if counts[source] <= 1:
                continue
            residual_source = [scores[row][c] - means[source][c] for c in range(3)]
            violation_own = quadratic(residual_source, metric)
            leverage_own = quadratic(residual_source, full_inverse)
            for destination in range(n_bins):
                if destination == source:
                    continue
                residual_dest = [scores[row][c] - means[destination][c] for c in range(3)]
                gap = violation_own - quadratic(residual_dest, metric)
                beta = weight * masses[destination] / (masses[destination] + weight)
                bound = beta * leverage_own * quadratic(residual_dest, full_inverse)
                assert gap <= bound
                checked += 1
                if gap > 0 and bound > 0:
                    max_ratio = max(max_ratio, gap / bound)
                moved = labels[:row] + (destination,) + labels[row + 1 :]
                if dets(moved)[1] == 0:
                    degenerate_destinations += 1
    assert (feasible, stable, checked) == (258, 17, 212)
    assert degenerate_destinations == 14
    assert max_ratio == Fraction(4, 15)


def test_ds15_profiled_value_is_bounded_by_the_efficient_score_interval_optimum() -> None:
    """DS15's finite half: exact projection-tax identity and sandwich.

    For every feasible labeling z of an exactly centered sample, with the
    efficient scores shat built from the full-sample empirical regression,

        profiled(z) = between(shat; z) - cross(z)^2 / I_ll(z)
                    <= between(shat; z) <= scalar interval optimum of shat,

    all in exact rationals, and the scalar optimum over arbitrary groupings is
    attained by interval groupings (1-D contiguity). The global profiled
    optimum sits strictly below the interval optimum on this sample.
    """
    fixture = json.loads(
        (RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-GLOBAL-GEOMETRY-001.json").read_text()
    )
    scores = [[Fraction(value) for value in row] for row in fixture["scores"]]
    n_rows, n_bins = len(scores), 3
    weight = Fraction(1, n_rows)
    full = [[sum(weight * row[a] * row[b] for row in scores) for b in range(2)] for a in range(2)]
    slope = full[0][1] / full[1][1]
    shat = [row[0] - slope * row[1] for row in scores]
    assert sum(weight * shat[row] * scores[row][1] for row in range(n_rows)) == 0

    def between(labels: tuple[int, ...]) -> Fraction:
        masses = [Fraction(0)] * n_bins
        sums = [Fraction(0)] * n_bins
        for row, label in enumerate(labels):
            masses[label] += weight
            sums[label] += weight * shat[row]
        return sum(sums[b] ** 2 / masses[b] for b in range(n_bins) if masses[b] > 0)

    order = sorted(range(n_rows), key=shat.__getitem__)
    interval_best = Fraction(0)
    for first_cut in range(1, n_rows - 1):
        for second_cut in range(first_cut + 1, n_rows):
            labels = [0] * n_rows
            for position, row in enumerate(order):
                labels[row] = 0 if position < first_cut else (1 if position < second_cut else 2)
            interval_best = max(interval_best, between(tuple(labels)))

    n_feasible = 0
    best_profiled = Fraction(0)
    grouping_best = Fraction(0)
    for labels in _canonical_partitions(n_rows, n_bins):
        masses, moments = _exact_ds_cells(scores, labels, n_bins)
        info = _exact_binned_information(scores, labels, n_bins)
        cell_between = between(labels)
        grouping_best = max(grouping_best, cell_between)
        assert cell_between <= interval_best
        if info[1][1] == 0:
            continue
        n_feasible += 1
        profiled = info[0][0] - info[0][1] * info[1][0] / info[1][1]
        cross = sum(
            (moments[b][0] - slope * moments[b][1]) * moments[b][1] / masses[b]
            for b in range(n_bins)
        )
        assert profiled == cell_between - cross**2 / info[1][1]
        assert profiled <= cell_between
        best_profiled = max(best_profiled, profiled)

    assert n_feasible == 966
    assert grouping_best == interval_best
    assert best_profiled == Fraction(6241, 984)
    assert interval_best == Fraction(135987641, 19993584)
    assert best_profiled < interval_best


def test_ds15_rank_deficiency_zeroes_every_feasible_profiled_value() -> None:
    """DS15's K = d_lambda + 1 rank boundary (CE-DS-MARGINS-RANK-VACUITY-001).

    Exact centering forces rank(I_z) <= K - 1, so with a scalar POI and a
    d_lambda = K - 1 nuisance every labeling with a nonsingular binned
    nuisance block has profiled Schur value exactly 0 - on every sample -
    while the efficient-score interval optimum stays strictly positive and
    K = d_lambda + 2 restores a positive value on the same atoms. Conclusion
    (i) of DS15 therefore needs K >= d_lambda + 2, not K >= 3.
    """
    fixture = json.loads(
        (RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-MARGINS-RANK-VACUITY-001.json").read_text()
    )
    scores = [[Fraction(value) for value in row] for row in fixture["scores"]]
    n_rows = len(scores)
    weight = Fraction(1, n_rows)
    assert all(sum(row[column] for row in scores) == 0 for column in range(3))

    def blocks(labels: tuple[int, ...], n_bins: int) -> list[list[Fraction]]:
        masses = [Fraction(0)] * n_bins
        moments = [[Fraction(0)] * 3 for _ in range(n_bins)]
        for row, label in enumerate(labels):
            masses[label] += weight
            for column in range(3):
                moments[label][column] += weight * scores[row][column]
        return [
            [
                sum(
                    moments[b][row] * moments[b][column] / masses[b]
                    for b in range(n_bins)
                    if masses[b] > 0
                )
                for column in range(3)
            ]
            for row in range(3)
        ]

    def schur(info: list[list[Fraction]]) -> Fraction | None:
        det_l = info[1][1] * info[2][2] - info[1][2] * info[2][1]
        if det_l == 0:
            return None
        adj = info[0][1] * (info[2][2] * info[0][1] - info[1][2] * info[0][2]) + info[0][2] * (
            info[1][1] * info[0][2] - info[1][2] * info[0][1]
        )
        return info[0][0] - adj / det_l

    n_feasible = 0
    for labels in _canonical_partitions(n_rows, 3):
        value = schur(blocks(labels, 3))
        if value is not None:
            n_feasible += 1
            assert value == 0
    assert n_feasible == 6

    full = blocks((0, 1, 2, 3), 4)
    det_l = full[1][1] * full[2][2] - full[1][2] * full[2][1]
    slope_1 = (full[0][1] * full[2][2] - full[0][2] * full[1][2]) / det_l
    slope_2 = (full[0][2] * full[1][1] - full[0][1] * full[2][1]) / det_l
    shat = [row[0] - slope_1 * row[1] - slope_2 * row[2] for row in scores]
    order = sorted(range(n_rows), key=shat.__getitem__)
    interval_best = Fraction(0)
    for first_cut in range(1, n_rows - 1):
        for second_cut in range(first_cut + 1, n_rows):
            masses = [Fraction(0)] * 3
            sums = [Fraction(0)] * 3
            for position, row in enumerate(order):
                cell = 0 if position < first_cut else (1 if position < second_cut else 2)
                masses[cell] += weight
                sums[cell] += weight * shat[row]
            interval_best = max(
                interval_best, sum(sums[b] ** 2 / masses[b] for b in range(3) if masses[b] > 0)
            )
    assert interval_best == Fraction(81, 50)
    assert schur(full) == Fraction(9, 5)


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
        (RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-MARGINS-RANK-VACUITY-001.json").read_text()
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
        (RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-MARGINS-RANK-VACUITY-001.json").read_text()
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


def test_ds15_projection_tax_identity_survives_ties_duplicates_and_unequal_weights() -> None:
    """Proposition 4 is pure finite algebra: it holds off DS15's assumptions.

    Duplicate atoms with unequal positive weights, and a nuisance-symmetric
    sample whose efficient scores carry exact duplicate values, both satisfy
    the exact identity profiled = between - cross^2 / I_ll on every feasible
    labeling; on singular-nuisance labelings the pseudo-inverse value
    collapses to the between value exactly.
    """
    datasets = [
        (
            [(2, 1), (2, 1), (-1, 2), (0, -3), (-1, -1), (-2, 0)],
            [Fraction(1, 4)] + [Fraction(1, 8)] * 3 + [Fraction(1, 4), Fraction(1, 8)],
            88,
            2,
        ),
        (
            [(1, 1), (1, -1), (-1, 2), (-1, -2), (0, 3), (0, -3)],
            [Fraction(1, 6)] * 6,
            89,
            1,
        ),
    ]
    for raw, weights, expected_feasible, expected_singular in datasets:
        total = sum(weights)
        mean = [
            sum(w * Fraction(r[c]) for w, r in zip(weights, raw, strict=True)) / total
            for c in range(2)
        ]
        scores = [[Fraction(r[c]) - mean[c] for c in range(2)] for r in raw]
        n_rows = len(scores)
        full = [
            [
                sum(w * row[a] * row[b] for w, row in zip(weights, scores, strict=True)) / total
                for b in range(2)
            ]
            for a in range(2)
        ]
        slope = full[0][1] / full[1][1]
        shat = [row[0] - slope * row[1] for row in scores]
        assert sum(w * s * row[1] for w, s, row in zip(weights, shat, scores, strict=True)) == 0

        n_feasible = n_singular = 0
        for labels in _canonical_partitions(n_rows, 3):
            masses = [Fraction(0)] * 3
            shat_sums = [Fraction(0)] * 3
            lam_sums = [Fraction(0)] * 3
            psi_sums = [Fraction(0)] * 3
            for row, label in enumerate(labels):
                masses[label] += weights[row]
                shat_sums[label] += weights[row] * shat[row]
                lam_sums[label] += weights[row] * scores[row][1]
                psi_sums[label] += weights[row] * scores[row][0]
            between = sum(s**2 / m for s, m in zip(shat_sums, masses, strict=True))
            cross = sum(s * ell / m for s, ell, m in zip(shat_sums, lam_sums, masses, strict=True))
            i_ll = sum(ell**2 / m for ell, m in zip(lam_sums, masses, strict=True))
            i_pp = sum(p**2 / m for p, m in zip(psi_sums, masses, strict=True))
            i_pl = sum(p * ell / m for p, ell, m in zip(psi_sums, lam_sums, masses, strict=True))
            if i_ll == 0:
                n_singular += 1
                assert cross == 0 and i_pl == 0
                assert i_pp == between
                continue
            n_feasible += 1
            assert i_pp - i_pl**2 / i_ll == between - cross**2 / i_ll
        assert (n_feasible, n_singular) == (expected_feasible, expected_singular)


def test_ds16_exchange_stable_state_can_retain_macroscopic_margins() -> None:
    """One-point exchange stability does not force the DS15 degeneracy.

    A non-global one-point exchange-stable profiled-Ds labeling - all 16
    admissible one-point moves have exact profiled gain <= 0 - carries a
    macroscopic nuisance block, conditioning bound, minimum cell mass, and
    projected-centroid separation, at a price below the exact efficient-score
    interval optimum v_K. The global optimum's nuisance block is four times
    smaller than the witness's: the value ranking is anti-aligned with the
    conditioning margin. Exchange stability does not preclude the DS14
    margins; it prices them (CE-DS-STABLE-MARGIN-RETAINING-001).
    """
    fixture = json.loads(
        (
            RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-STABLE-MARGIN-RETAINING-001.json"
        ).read_text()
    )
    scores = [[Fraction(value) for value in row] for row in fixture["scores"]]
    n_rows, n_bins = len(scores), 3
    labels = tuple(fixture["labels_before"])
    exact = fixture["exact_quantities"]

    def profiled(candidate: tuple[int, ...]) -> Fraction | None:
        info = _exact_binned_information(scores, candidate, n_bins)
        if info[1][1] == 0:
            return None
        return info[0][0] - info[0][1] * info[1][0] / info[1][1]

    witness_value = profiled(labels)
    assert witness_value == Fraction(exact["witness_profiled_value"])

    info = _exact_binned_information(scores, labels, n_bins)
    masses, moments = _exact_ds_cells(scores, labels, n_bins)
    assert info[1][1] == Fraction(exact["witness_nuisance_block_I11"])
    assert min(masses) == Fraction(exact["witness_min_cell_mass"])

    slope = info[0][1] / info[1][1]
    projected = [
        moments[b][0] / masses[b] - slope * moments[b][1] / masses[b] for b in range(n_bins)
    ]
    separation = min(
        abs(projected[b] - projected[c]) for b in range(n_bins) for c in range(b + 1, n_bins)
    )
    assert separation == Fraction(exact["witness_projected_centroid_separation"])

    determinant = info[0][0] * info[1][1] - info[0][1] * info[1][0]
    trace = info[0][0] + info[1][1]
    assert determinant / trace == Fraction(exact["witness_lambda_min_lower_bound_det_over_trace"])

    # Every admissible one-point move (source cell has >= 2 rows) has exact
    # nonpositive profiled gain: this is one-point exchange stability.
    gains: list[Fraction] = []
    for row in range(n_rows):
        source = labels[row]
        if sum(1 for label in labels if label == source) < 2:
            continue
        for target in range(n_bins):
            if target == source:
                continue
            moved = list(labels)
            moved[row] = target
            moved_value = profiled(tuple(moved))
            assert moved_value is not None
            gains.append(moved_value - witness_value)
    assert len(gains) == 16
    assert all(gain <= 0 for gain in gains)
    assert max(gains) == Fraction(exact["witness_max_residual_move_gain"])

    # The witness is exchange-stable but not the exact global optimum, and the
    # global optimum's nuisance block is markedly smaller than the witness's.
    best_value: Fraction | None = None
    best_labels: tuple[int, ...] | None = None
    for candidate in _canonical_partitions(n_rows, n_bins):
        value = profiled(candidate)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value, best_labels = value, candidate
    assert best_labels is not None
    assert best_value == Fraction(exact["global_optimum_value"])
    assert best_value > witness_value
    assert "".join(str(label) for label in best_labels) == exact["global_optimum_labels"]

    global_info = _exact_binned_information(scores, best_labels, n_bins)
    assert global_info[1][1] == Fraction(exact["global_optimum_nuisance_block_I11"])
    assert global_info[1][1] < info[1][1]

    # The efficient-score interval optimum v_K strictly exceeds the witness,
    # by the recorded exact gap.
    weight = Fraction(1, n_rows)
    full = [[sum(weight * row[a] * row[b] for row in scores) for b in range(2)] for a in range(2)]
    bhat = full[0][1] / full[1][1]
    assert bhat == Fraction(exact["full_sample_regression_slope_Bhat"])
    shat = [row[0] - bhat * row[1] for row in scores]

    order = sorted(range(n_rows), key=shat.__getitem__)
    interval_optimum = Fraction(0)
    for first_cut in range(1, n_rows - 1):
        for second_cut in range(first_cut + 1, n_rows):
            cell_masses = [Fraction(0)] * n_bins
            cell_sums = [Fraction(0)] * n_bins
            for position, row in enumerate(order):
                cell = 0 if position < first_cut else (1 if position < second_cut else 2)
                cell_masses[cell] += weight
                cell_sums[cell] += weight * shat[row]
            interval_optimum = max(
                interval_optimum,
                sum(
                    cell_sums[b] ** 2 / cell_masses[b] for b in range(n_bins) if cell_masses[b] > 0
                ),
            )
    assert interval_optimum == Fraction(exact["efficient_score_interval_optimum_v_K"])
    assert witness_value < interval_optimum
    assert interval_optimum - witness_value == Fraction(exact["value_gap_v_K_minus_witness"])


def test_ds16_efficient_score_interval_seed_is_not_exchange_stable() -> None:
    """The efficient-score interval labeling is not one-point exchange-stable.

    The exact interval optimum of the full-sample efficient score shat - the
    documented profiled initializer, and the finite analogue of DS15's
    degenerate attainer J* - admits an improving one-point move: relocating
    row 7 raises the profiled value by an exact positive gain while growing
    the binned nuisance block 27-fold, buying back nuisance information, the
    finite mechanism behind DS15 Proposition 6's steering
    (CE-DS-INTERVAL-SEED-UNSTABLE-001).
    """
    fixture = json.loads(
        (
            RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-INTERVAL-SEED-UNSTABLE-001.json"
        ).read_text()
    )
    scores = [[Fraction(value) for value in row] for row in fixture["scores"]]
    n_rows, n_bins = len(scores), 3
    exact = fixture["exact_quantities"]
    weight = Fraction(1, n_rows)

    full = [[sum(weight * row[a] * row[b] for row in scores) for b in range(2)] for a in range(2)]
    bhat = full[0][1] / full[1][1]
    assert bhat == Fraction(exact["full_sample_regression_slope_Bhat"])
    shat = [row[0] - bhat * row[1] for row in scores]

    # Enumerate the C(7, 2) contiguous cut placements over the sorted shat
    # order and keep the best-scoring interval grouping.
    order = sorted(range(n_rows), key=shat.__getitem__)
    interval_optimum = Fraction(0)
    interval_labels: tuple[int, ...] | None = None
    for first_cut in range(1, n_rows - 1):
        for second_cut in range(first_cut + 1, n_rows):
            labeling = [0] * n_rows
            for position, row in enumerate(order):
                labeling[row] = 0 if position < first_cut else (1 if position < second_cut else 2)
            cell_masses = [Fraction(0)] * n_bins
            cell_sums = [Fraction(0)] * n_bins
            for row in range(n_rows):
                cell_masses[labeling[row]] += weight
                cell_sums[labeling[row]] += weight * shat[row]
            value = sum(
                cell_sums[b] ** 2 / cell_masses[b] for b in range(n_bins) if cell_masses[b] > 0
            )
            if value > interval_optimum:
                interval_optimum = value
                interval_labels = tuple(labeling)
    assert interval_labels is not None
    assert interval_optimum == Fraction(exact["interval_value_v_K"])

    def canonicalize(labeling: tuple[int, ...]) -> tuple[int, ...]:
        seen: dict[int, int] = {}
        canonical = []
        for label in labeling:
            if label not in seen:
                seen[label] = len(seen)
            canonical.append(seen[label])
        return tuple(canonical)

    labels_before = canonicalize(interval_labels)
    assert labels_before == tuple(fixture["labels_before"])

    def profiled(candidate: tuple[int, ...]) -> Fraction | None:
        info = _exact_binned_information(scores, candidate, n_bins)
        if info[1][1] == 0:
            return None
        return info[0][0] - info[0][1] * info[1][0] / info[1][1]

    value_before = profiled(labels_before)
    assert value_before == Fraction(exact["interval_labeling_profiled_value"])
    info_before = _exact_binned_information(scores, labels_before, n_bins)
    assert info_before[1][1] == Fraction(exact["interval_labeling_nuisance_block_I11"])

    move = exact["improving_move"]
    assert labels_before[move["row"]] == move["from_cell"]
    labels_after = list(labels_before)
    labels_after[move["row"]] = move["to_cell"]
    labels_after = tuple(labels_after)
    assert labels_after == tuple(fixture["labels_after_or_optimum"])

    value_after = profiled(labels_after)
    gain = value_after - value_before
    assert gain == Fraction(exact["exact_gain"])
    assert gain > 0

    info_after = _exact_binned_information(scores, labels_after, n_bins)
    assert info_after[1][1] == Fraction(exact["post_move_nuisance_block_I11"])
    assert info_after[1][1] > info_before[1][1]


def test_ds17_signsplit_stationary_state_retains_margins_without_separation() -> None:
    """A bounded-packet stationary K=3 state can retain (M2)+(M3) margins
    with zero projected-centroid separation, i.e. without (M5).

    The sign-split sibling of CE-DS-POP-WASTED-CELLS-001 on the same 8-atom
    nuisance-sign-symmetric law splits the left half {s_psi < 0} by
    sign(s_lambda) into two cells and leaves the right half as one cell.
    The binned information is full rank with a macroscopic minimum
    eigenvalue and minimum cell mass, yet the two left cells' projected
    centroids exactly coincide - zero separation - and every atom still
    satisfies the nearest-projected-centroid stationarity rule, with ties.
    Merging the coincident pair collapses to the K=2 s_psi-threshold rule,
    whose nuisance block is exactly singular: the compilable reduction
    carries no margin (CE-DS-LCM-SIGNSPLIT-MARGIN-001).
    """
    fixture = json.loads(
        (RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-LCM-SIGNSPLIT-MARGIN-001.json").read_text()
    )
    scores = [[Fraction(value) for value in row] for row in fixture["scores"]]
    weights = [Fraction(value) for value in fixture["weights"]]
    assert weights == [Fraction(1, 8)] * 8
    labels = tuple(fixture["labels_before"])
    assert labels == (0, 1, 0, 1, 2, 2, 2, 2)
    n_rows, n_bins = len(scores), 3
    exact = fixture["exact_quantities"]

    # (1) cell masses.
    masses, moments = _exact_ds_cells(scores, labels, n_bins)
    assert masses == [Fraction(value) for value in exact["cell_masses"]]
    assert masses == [Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)]

    # (2) binned information.
    info = _exact_binned_information(scores, labels, n_bins)
    assert info == [[Fraction(value) for value in row] for row in exact["binned_information"]]
    assert info == [[Fraction(4), Fraction(0)], [Fraction(0), Fraction(9, 8)]]

    # (3) profiled value and lambda_min (the off-diagonal is exactly zero,
    # so lambda_min of the 2x2 block is the smaller diagonal entry).
    profiled_value = info[0][0] - info[0][1] * info[1][0] / info[1][1]
    assert profiled_value == Fraction(4)
    assert profiled_value == Fraction(exact["profiled_value"])
    assert info[0][1] == 0
    lambda_min = min(info[0][0], info[1][1])
    assert lambda_min == Fraction(9, 8)
    assert lambda_min == Fraction(exact["lambda_min"])

    # (4) slope, projected centroids, and the coincident pair's separation.
    slope = info[0][1] / info[1][1]
    assert slope == 0
    projected = [
        moments[b][0] / masses[b] - slope * moments[b][1] / masses[b] for b in range(n_bins)
    ]
    assert projected == [Fraction(-2), Fraction(-2), Fraction(2)]
    assert projected == [Fraction(value) for value in exact["projected_centroids"]]
    separation = min(
        abs(projected[b] - projected[c]) for b in range(n_bins) for c in range(b + 1, n_bins)
    )
    assert separation == Fraction(0)

    # (5) stationarity with ties: every atom is at least as close (in the
    # efficient score e(s) = s_psi, since B* = 0) to its own cell's
    # projected centroid as to any other cell's.
    violations = 0
    for row in range(n_rows):
        e_row = scores[row][0] - slope * scores[row][1]
        own_distance = abs(e_row - projected[labels[row]])
        for cell in range(n_bins):
            other_distance = abs(e_row - projected[cell])
            assert own_distance <= other_distance
            if other_distance < own_distance:
                violations += 1
    assert violations == 0
    assert violations == exact["first_order_violations"]

    # (6) the reduced rule: merging cells 0 and 1 gives the K=2
    # s_psi-threshold rule, whose nuisance block is exactly singular.
    reduced_labels = tuple(0 if label in (0, 1) else 1 for label in labels)
    assert reduced_labels == (0, 0, 0, 0, 1, 1, 1, 1)
    reduced_info = _exact_binned_information(scores, reduced_labels, 2)
    assert reduced_info[1][1] == Fraction(0)

    # (7) value identity: the K=3 profiled value equals the K'=2
    # between-value of the projected law.
    w_left = masses[0] + masses[1]
    w_right = masses[2]
    assert w_left == Fraction(1, 2)
    assert w_right == Fraction(1, 2)
    group_between = w_left * Fraction(-2) ** 2 + w_right * Fraction(2) ** 2
    assert group_between == Fraction(4)
    assert profiled_value == group_between


def test_ds17_minimal_atomic_signsplit_is_only_a_boundary_witness() -> None:
    """The N=3 sign-split algebra is minimal but outside DS17's hypotheses."""
    fixture = json.loads(
        (
            RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-LCM-SIGNSPLIT-MINIMAL-001.json"
        ).read_text()
    )
    scores = [[Fraction(value) for value in row] for row in fixture["scores"]]
    weights = [Fraction(value) for value in fixture["weights"]]
    labels = tuple(fixture["labels_before"])
    n_rows, n_bins = len(scores), fixture["K"]
    exact = fixture["exact_quantities"]

    assert n_rows == n_bins == 3
    assert weights == [Fraction(1, 3)] * 3
    assert labels == (0, 1, 2)
    assert [
        sum(weight * score[column] for weight, score in zip(weights, scores, strict=True))
        for column in range(2)
    ] == [Fraction(value) for value in exact["weighted_score_mean"]]

    masses, moments = _exact_ds_cells(scores, labels, n_bins)
    assert masses == [Fraction(value) for value in exact["cell_masses"]]
    info = _exact_binned_information(scores, labels, n_bins)
    assert info == [[Fraction(value) for value in row] for row in exact["binned_information"]]
    assert info == [[Fraction(2), Fraction(0)], [Fraction(0), Fraction(2, 3)]]

    slope = info[0][1] / info[1][1]
    projected = [
        moments[cell][0] / masses[cell] - slope * moments[cell][1] / masses[cell]
        for cell in range(n_bins)
    ]
    assert slope == 0
    assert projected == [Fraction(value) for value in exact["projected_centroids"]]
    assert projected == [Fraction(-1), Fraction(-1), Fraction(2)]
    assert (
        min(
            abs(projected[first] - projected[second])
            for first in range(n_bins)
            for second in range(first + 1, n_bins)
        )
        == 0
    )

    profiled = info[0][0] - info[0][1] * info[1][0] / info[1][1]
    assert profiled == Fraction(exact["profiled_value"])
    assert min(info[0][0], info[1][1]) == Fraction(exact["lambda_min"])

    admissible_moves = 0
    for source in labels:
        if labels.count(source) <= 1:
            continue
        for destination in range(n_bins):
            if destination != source:
                admissible_moves += 1
    assert admissible_moves == exact["admissible_nonempty_preserving_one_point_moves"] == 0

    reduced_labels = (0, 0, 1)
    reduced_info = _exact_binned_information(scores, reduced_labels, 2)
    assert reduced_info[1][1] == Fraction(exact["reduced_rule_nuisance_block"])

    # An atom sits on its own zero-width slab with mass 1/3 for every positive
    # width, so no uniform slab modulus can tend to zero: (M4) fails. The
    # coincident projected centroids also make (M5) fail. These checks keep
    # the fixture classified as a boundary witness, never a DS17 refutation.
    slab_center = scores[0][0]
    for width in (Fraction(1, 10), Fraction(1, 100), Fraction(1, 1000)):
        slab_mass = sum(
            weight
            for weight, score in zip(weights, scores, strict=True)
            if abs(score[0] - slab_center) <= width
        )
        assert slab_mass == Fraction(2, 3)
    assert fixture["claim_falsified"].startswith("None:")


def test_ds18_population_cut_labels_need_not_be_exchange_stable() -> None:
    """Boundary noise can defeat the raw population labels at finite N."""
    fixture = json.loads(
        (
            RESEARCH_WORKSPACE
            / "COUNTEREXAMPLES"
            / "CE-DS-NONCENTERED-POPULATION-CUT-UNSTABLE-001.json"
        ).read_text()
    )
    scores = [[Fraction(value) for value in row] for row in fixture["scores"]]
    weights = [Fraction(value) for value in fixture["weights"]]
    before = tuple(fixture["labels_before"])
    after = tuple(fixture["labels_after_or_optimum"])

    assert len(scores) == 4
    assert fixture["K"] == 3
    assert [
        sum(weight * score[column] for weight, score in zip(weights, scores, strict=True))
        for column in range(2)
    ] == [Fraction(0), Fraction(0)]

    before_information = _exact_binned_information(scores, before, 3)
    after_information = _exact_binned_information(scores, after, 3)
    assert before_information == [
        [Fraction(value) for value in row]
        for row in fixture["exact_quantities"]["information_before"]
    ]
    assert after_information == [
        [Fraction(value) for value in row]
        for row in fixture["exact_quantities"]["information_after"]
    ]

    before_value = (
        before_information[0][0] - before_information[0][1] ** 2 / before_information[1][1]
    )
    after_value = after_information[0][0] - after_information[0][1] ** 2 / after_information[1][1]
    assert before_value == Fraction(fixture["objective_before"])
    assert after_value == Fraction(fixture["objective_after"])
    assert after_value - before_value == Fraction(fixture["exact_quantities"]["exact_gain"])
    assert after_value > before_value

    partitions = _canonical_partitions(4, 3)
    regular_values = []
    for labels in partitions:
        information = _exact_binned_information(scores, labels, 3)
        if information[1][1] == 0:
            continue
        value = information[0][0] - information[0][1] ** 2 / information[1][1]
        regular_values.append((value, labels))
    assert max(regular_values) == (after_value, after)


def test_ds18_named_off_class_root_and_margins_are_exact() -> None:
    """The bounded off-(L) law has the registered regular DS17 root exactly."""
    masses = [Fraction(1, 3)] * 3
    means_psi = [Fraction(-2, 3), Fraction(0), Fraction(2, 3)]
    means_lambda = [Fraction(4, 9), Fraction(-8, 9), Fraction(4, 9)]
    information = [
        [
            sum(
                mass
                * (means_psi if first == 0 else means_lambda)[cell]
                * (means_psi if second == 0 else means_lambda)[cell]
                for cell, mass in enumerate(masses)
            )
            for second in range(2)
        ]
        for first in range(2)
    ]
    full_information = [
        [Fraction(1, 3), Fraction(0)],
        [Fraction(0), Fraction(17, 15)],
    ]

    assert information == [
        [Fraction(8, 27), Fraction(0)],
        [Fraction(0), Fraction(32, 81)],
    ]
    assert min(information[0][0], information[1][1]) == Fraction(8, 27)
    assert information[0][1] == 0  # DS17 root residual and beta numerator
    assert means_psi == [Fraction(-2, 3), Fraction(0), Fraction(2, 3)]
    assert [(means_psi[index] + means_psi[index + 1]) / 2 for index in range(2)] == [
        Fraction(-1, 3),
        Fraction(1, 3),
    ]
    assert min(means_psi[index + 1] - means_psi[index] for index in range(2)) == Fraction(2, 3)
    assert information[0][0] / full_information[0][0] == Fraction(8, 9)
    assert Fraction(1, 4) < min(masses)
    assert Fraction(1, 4) < min(information[0][0], information[1][1])
    assert Fraction(1, 2) < Fraction(2, 3)

    artifact = json.loads(
        (
            RESEARCH_WORKSPACE
            / "WORK"
            / "artifacts"
            / "OPEN-DS-MARGINS-NONCENTERED"
            / "exact-falsification.json"
        ).read_text()
    )
    exact = artifact["population_exact"]
    assert exact["binned_information"] == [["8/27", "0"], ["0", "32/81"]]
    assert exact["full_information"] == [["1/3", "0"], ["0", "17/15"]]
    assert exact["root_residual"] == "0"
    assert exact["ds_retention"] == "8/9"


def _ds18_law_scores(xs: tuple[Fraction, ...], zs: tuple[Fraction, ...]) -> list[list[Fraction]]:
    """Build DS18 score rows from the law S = (X, 3X^2 - 1 + Z), never from a fixture."""
    return [[x, 3 * x * x - 1 + z] for x, z in zip(xs, zs, strict=True)]


def _ds18_integrate(
    poly: dict[tuple[int, int], Fraction], x_lo: Fraction, x_hi: Fraction
) -> Fraction:
    """Exact E[poly(X, Z) 1{x_lo <= X <= x_hi}] for X, Z iid Uniform[-1, 1].

    The joint density is 1/4 on the square and the shear to score space has
    unit Jacobian, so this integrates the law itself rather than any stored
    constant.
    """
    total = Fraction(0)
    for (i, j), coefficient in poly.items():
        x_part = (x_hi ** (i + 1) - x_lo ** (i + 1)) / (i + 1)
        z_part = (Fraction(1) ** (j + 1) - Fraction(-1) ** (j + 1)) / (j + 1)
        total += coefficient * x_part * z_part
    return total / 4


def _ds18_multiply(
    left: dict[tuple[int, int], Fraction], right: dict[tuple[int, int], Fraction]
) -> dict[tuple[int, int], Fraction]:
    out: dict[tuple[int, int], Fraction] = {}
    for (i0, j0), a in left.items():
        for (i1, j1), b in right.items():
            key = (i0 + i1, j0 + j1)
            out[key] = out.get(key, Fraction(0)) + a * b
    return out


def _ds18_profiled_value(information: list[list[Fraction]]) -> Fraction:
    """In-bin profiled value, with the DS11 pseudo-inverse extension at a singular block."""
    if information[1][1] == 0:
        return information[0][0]
    return information[0][0] - information[0][1] ** 2 / information[1][1]


def _ds18_between(scores: list[list[Fraction]], labels: tuple[int, ...], n_bins: int) -> Fraction:
    """Uncentered between-value of the POI column: sum_b (sum_{i in b} w s_psi)^2 / W_b."""
    masses, moments = _exact_ds_cells(scores, labels, n_bins)
    return sum(
        (
            moments[label][0] * moments[label][0] / masses[label]
            for label in range(n_bins)
            if masses[label] > 0
        ),
        Fraction(0),
    )


def _ds18_scalar_distortion(codebook: list[Fraction]) -> Fraction:
    """Exact E[min_b (X - c_b)^2] for X ~ Uniform[-1, 1]."""
    points = sorted(codebook)
    edges = [Fraction(-1)]
    for left, right in zip(points, points[1:], strict=False):
        edges.append(min(Fraction(1), max(Fraction(-1), (left + right) / 2)))
    edges.append(Fraction(1))
    total = Fraction(0)
    for index, centre in enumerate(points):
        lo, hi = edges[index], edges[index + 1]
        if hi > lo:
            total += ((hi - centre) ** 3 - (lo - centre) ** 3) / 6
    return total


def test_ds18_population_law_integrates_to_the_registered_optimum() -> None:
    """DS18's population constants follow from the law, not from stored numbers."""
    psi = {(1, 0): Fraction(1)}
    lam = {(2, 0): Fraction(3), (0, 0): Fraction(-1), (0, 1): Fraction(1)}
    one = {(0, 0): Fraction(1)}
    cut = Fraction(1, 3)

    full = [
        [_ds18_integrate(_ds18_multiply(psi, psi), Fraction(-1), Fraction(1))],
        [_ds18_integrate(_ds18_multiply(lam, lam), Fraction(-1), Fraction(1))],
    ]
    cross = _ds18_integrate(_ds18_multiply(psi, lam), Fraction(-1), Fraction(1))
    assert full[0][0] == Fraction(1, 3)
    assert full[1][0] == Fraction(17, 15)
    assert cross == 0  # B* = 0, so the efficient score is X itself

    # The law is outside class (L): E[S_lambda | X] = 3X^2 - 1, verified by orthogonality.
    residual = {(2, 0): Fraction(3), (0, 0): Fraction(-1), (0, 1): Fraction(1)}
    for power in range(6):
        against = {(power, 0): Fraction(1)}
        conditional = _ds18_integrate(
            _ds18_multiply({(2, 0): Fraction(3), (0, 0): Fraction(-1)}, against),
            Fraction(-1),
            Fraction(1),
        )
        assert _ds18_integrate(_ds18_multiply(residual, against), Fraction(-1), Fraction(1)) == (
            conditional
        )

    edges = ((Fraction(-1), -cut), (-cut, cut), (cut, Fraction(1)))
    masses = [_ds18_integrate(one, lo, hi) for lo, hi in edges]
    mean_psi = [
        _ds18_integrate(psi, lo, hi) / mass for (lo, hi), mass in zip(edges, masses, strict=True)
    ]
    mean_lam = [
        _ds18_integrate(lam, lo, hi) / mass for (lo, hi), mass in zip(edges, masses, strict=True)
    ]
    assert masses == [Fraction(1, 3)] * 3
    assert mean_psi == [Fraction(-2, 3), Fraction(0), Fraction(2, 3)]
    assert mean_lam == [Fraction(4, 9), Fraction(-8, 9), Fraction(4, 9)]

    information = [[Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]]
    for mass, first, second in zip(masses, mean_psi, mean_lam, strict=True):
        information[0][0] += mass * first * first
        information[0][1] += mass * first * second
        information[1][0] += mass * first * second
        information[1][1] += mass * second * second
    assert information == [[Fraction(8, 27), Fraction(0)], [Fraction(0), Fraction(32, 81)]]
    assert _ds18_profiled_value(information) == Fraction(8, 27)
    assert Fraction(8, 27) / full[0][0] == Fraction(8, 9)

    # The scalar upper problem: v_3 = 8/27 uniquely, and v_2 = 1/4 is strictly worse.
    assert _ds18_scalar_distortion(mean_psi) == Fraction(1, 27)
    assert full[0][0] - _ds18_scalar_distortion(mean_psi) == Fraction(8, 27)
    two_cell = full[0][0] - _ds18_scalar_distortion([Fraction(-1, 2), Fraction(1, 2)])
    assert two_cell == Fraction(1, 4) < Fraction(8, 27)

    # (M3) is lambda_min, not det/tr: the det/tr reading would fail kappa = 1/4.
    assert min(information[0][0], information[1][1]) > Fraction(1, 4)
    determinant = information[0][0] * information[1][1]
    assert determinant / (information[0][0] + information[1][1]) == Fraction(32, 189)
    assert Fraction(32, 189) < Fraction(1, 4)


def test_ds18_profiled_scalar_sandwich_holds_on_every_small_partition() -> None:
    """Profiled <= uncentered between <= best three-group value, on arbitrary cells."""
    n_bins = 3
    cases = [
        (
            (Fraction(-3, 4), Fraction(-1, 4), Fraction(1, 4), Fraction(3, 4), Fraction(-1, 2)),
            (Fraction(-1), Fraction(-3, 4), Fraction(1), Fraction(1), Fraction(1, 2)),
        ),
        (
            (
                Fraction(-1),
                Fraction(-1, 3),
                Fraction(1, 3),
                Fraction(1),
                Fraction(0),
                Fraction(1, 2),
            ),
            (Fraction(1), Fraction(-1), Fraction(1, 4), Fraction(-1), Fraction(0), Fraction(3, 4)),
        ),
        (
            (
                Fraction(-1),
                Fraction(0),
                Fraction(1, 2),
                Fraction(1, 2),
                Fraction(1),
                Fraction(-1, 2),
            ),
            (Fraction(-1), Fraction(1), Fraction(-3, 4), Fraction(1, 4), Fraction(-1), Fraction(0)),
        ),
    ]
    for xs, zs in cases:
        scores = _ds18_law_scores(xs, zs)
        n_rows = len(scores)
        partitions = _canonical_partitions(n_rows, n_bins)
        upper = max(_ds18_between(scores, labels, n_bins) for labels in partitions)
        for labels in partitions:
            information = _exact_binned_information(scores, labels, n_bins)
            value = _ds18_profiled_value(information)
            between = _ds18_between(scores, labels, n_bins)
            assert value <= between <= upper
            if information[1][1] > 0:
                # The Schur value equals the minimum of the DS11 variational form.
                masses, moments = _exact_ds_cells(scores, labels, n_bins)
                slope = information[0][1] / information[1][1]
                variational = sum(
                    (
                        (moments[label][0] - slope * moments[label][1]) ** 2 / masses[label]
                        for label in range(n_bins)
                        if masses[label] > 0
                    ),
                    Fraction(0),
                )
                assert variational == value


def test_ds18_singular_destination_beats_the_regular_optimum() -> None:
    """The in-bin feasibility convention is load-bearing at the minimal support."""
    fixture = json.loads(
        (
            RESEARCH_WORKSPACE
            / "COUNTEREXAMPLES"
            / "CE-DS-NONCENTERED-SINGULAR-DESTINATION-001.json"
        ).read_text()
    )
    xs = tuple(Fraction(value) for value in fixture["construction_x"])
    zs = tuple(Fraction(value) for value in fixture["construction_z"])
    scores = _ds18_law_scores(xs, zs)
    n_bins = fixture["K"]

    assert [[str(value) for value in row] for row in scores] == fixture["scores"]
    assert all(abs(x) <= 1 and abs(z) <= 1 for x, z in zip(xs, zs, strict=True))
    weight = Fraction(1, len(scores))
    assert [sum(weight * row[column] for row in scores) for column in range(2)] == [
        Fraction(0),
        Fraction(0),
    ]

    partitions = _canonical_partitions(len(scores), n_bins)
    assert len(partitions) == fixture["exact_quantities"]["canonical_partitions"]
    regular: list[tuple[Fraction, tuple[int, ...]]] = []
    singular: list[tuple[Fraction, tuple[int, ...]]] = []
    for labels in partitions:
        information = _exact_binned_information(scores, labels, n_bins)
        value = _ds18_profiled_value(information)
        (regular if information[1][1] > 0 else singular).append((value, labels))
    best = max(value for value, _ in regular)
    optima = sorted(labels for value, labels in regular if value == best)
    assert best == Fraction(fixture["objective_before"])
    assert optima == sorted(
        tuple(labels) for labels in fixture["exact_quantities"]["global_regular_optima_labels"]
    )
    assert [labels for _, labels in singular] == [
        tuple(labels) for labels in fixture["exact_quantities"]["singular_labelings"]
    ]
    assert singular[0][0] == Fraction(fixture["objective_after"])
    assert singular[0][0] - best == Fraction(fixture["exact_quantities"]["exact_gain"])

    # Every global regular optimum escapes into the singular labeling by one move
    # whose source stays nonempty, and no regular destination improves.
    for optimum in optima:
        counts = [optimum.count(label) for label in range(n_bins)]
        escapes = 0
        for row, source in enumerate(optimum):
            if counts[source] <= 1:
                continue
            for destination in range(n_bins):
                if destination == source:
                    continue
                moved = list(optimum)
                moved[row] = destination
                information = _exact_binned_information(scores, tuple(moved), n_bins)
                gain = _ds18_profiled_value(information) - best
                if information[1][1] > 0:
                    assert gain <= 0
                elif gain > 0:
                    escapes += 1
        assert escapes == 1

    # A zero binned nuisance block forces every cell lambda-sum, hence the total, to
    # vanish -- which is why this table can host one and why the law cannot, a.s.
    assert sum(row[1] for row in scores) == 0


def test_ds19_tilt_dual_has_a_support_minimal_exact_positive_gap() -> None:
    """Two tilt quadratics certify a strict gap without locating the algebraic minimizer."""
    fixture = json.loads(
        (RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-TILT-DUAL-GAP-001.json").read_text()
    )
    scores = [[Fraction(value) for value in row] for row in fixture["scores"]]
    weights = [Fraction(value) for value in fixture["weights"]]
    assert weights == [Fraction(1, 4)] * 4
    n_bins = fixture["K"]
    first = tuple(fixture["labels_before"])
    optimum = tuple(fixture["labels_after_or_optimum"])
    exact = fixture["exact_quantities"]

    partitions = _canonical_partitions(len(scores), n_bins)
    assert len(partitions) == exact["canonical_partitions"] == 6
    values: list[tuple[Fraction, tuple[int, ...]]] = []
    quadratics: dict[tuple[int, ...], tuple[Fraction, Fraction, Fraction]] = {}
    for labels in partitions:
        information = _exact_binned_information(scores, labels, n_bins)
        assert information[1][1] > 0
        value = information[0][0] - information[0][1] ** 2 / information[1][1]
        values.append((value, labels))
        quadratics[labels] = (
            information[1][1],
            -2 * information[0][1],
            information[0][0],
        )

    global_value, global_labels = max(values)
    assert global_labels == optimum
    assert global_value == Fraction(exact["global_profiled_value"])
    assert global_value == Fraction(116805, 11816)
    assert quadratics[first] == tuple(
        Fraction(value) for value in exact["first_tilt_quadratic_ABC"]
    )
    assert quadratics[optimum] == tuple(
        Fraction(value) for value in exact["optimum_tilt_quadratic_ABC"]
    )

    alpha = Fraction(exact["convex_mixture_alpha_on_first"])
    mixture = tuple(
        alpha * left + (1 - alpha) * right
        for left, right in zip(quadratics[first], quadratics[optimum], strict=True)
    )
    assert mixture == tuple(Fraction(value) for value in exact["mixture_quadratic_ABC"])
    a_value, b_value, c_value = mixture
    vertex = -b_value / (2 * a_value)
    lower = a_value * vertex * vertex + b_value * vertex + c_value
    gap = lower - global_value
    assert vertex == Fraction(exact["mixture_vertex_beta"])
    assert lower == Fraction(exact["mixture_global_minimum"])
    assert gap == Fraction(exact["certified_duality_gap_lower_bound"])
    assert gap == Fraction(105329256, 154014175) > 0

    # Weak duality itself survives: every partition's profiled value is no
    # larger than its own tilted between-value, hence no larger than the dual.
    for beta in (Fraction(-1), vertex, Fraction(0), Fraction(1)):
        tilted = {
            labels: quadratic[0] * beta * beta + quadratic[1] * beta + quadratic[2]
            for labels, quadratic in quadratics.items()
        }
        dual_value = max(tilted.values())
        for value, labels in values:
            assert value <= tilted[labels] <= dual_value
        mixed_value = mixture[0] * beta * beta + mixture[1] * beta + mixture[2]
        assert dual_value >= mixed_value >= lower

    # K=3, N=3 has only the all-singleton partition, so the bracket closes and
    # N=4 is the first support size where a strict gap is possible.
    assert _canonical_partitions(3, 3) == [(0, 1, 2)]


def _exact_scalar_between_for_ds19(
    values: list[Fraction],
    weights: list[Fraction],
    labels: tuple[int, ...],
    n_bins: int,
) -> Fraction:
    masses = [Fraction(0)] * n_bins
    moments = [Fraction(0)] * n_bins
    for value, weight, label in zip(values, weights, labels, strict=True):
        masses[label] += weight
        moments[label] += weight * value
    return sum(
        moment * moment / mass for moment, mass in zip(moments, masses, strict=True) if mass > 0
    )


def _canonicalize_ds19(labels: tuple[int, ...]) -> tuple[int, ...]:
    mapping: dict[int, int] = {}
    out = []
    for label in labels:
        if label not in mapping:
            mapping[label] = len(mapping)
        out.append(mapping[label])
    return tuple(out)


def _exact_tie_interval_optimum_ds19(
    values: list[Fraction], weights: list[Fraction], n_bins: int
) -> Fraction:
    """Exhaust total orders inside exact tie blocks, then every interval cut."""
    best: Fraction | None = None
    for order in permutations(range(len(values))):
        if any(values[order[index]] > values[order[index + 1]] for index in range(len(order) - 1)):
            continue
        for cuts in combinations(range(1, len(values)), n_bins - 1):
            bounds = (0, *cuts, len(values))
            labels = [0] * len(values)
            for cell, (start, stop) in enumerate(zip(bounds, bounds[1:], strict=False)):
                for position in range(start, stop):
                    labels[order[position]] = cell
            candidate = _canonicalize_ds19(tuple(labels))
            value = _exact_scalar_between_for_ds19(values, weights, candidate, n_bins)
            best = value if best is None else max(best, value)
    assert best is not None
    return best


def test_ds19_weak_duality_and_scalar_contiguity_survive_adversarial_ties() -> None:
    """Duplicates, unequal weights, singular nuisance, and tilt ties do not break the ceiling."""
    cases = [
        (
            [[-1, 0], [-1, 0], [0, 1], [1, 0], [1, 0]],
            [Fraction(1, 5)] * 5,
        ),
        (
            [[-1, -1], [0, 0], [1, 1], [2, 0]],
            [Fraction(1, 4)] * 4,
        ),
        (
            [
                [Fraction(-3, 4), Fraction(-5, 16)],
                [Fraction(-1, 4), Fraction(-25, 16)],
                [Fraction(1, 4), Fraction(3, 16)],
                [Fraction(3, 4), Fraction(27, 16)],
            ],
            [Fraction(1, 10), Fraction(1, 5), Fraction(3, 10), Fraction(2, 5)],
        ),
        (
            [[-1, 0], [0, 0], [1, 0]],
            [Fraction(1, 3)] * 3,
        ),
        (
            [
                [-1, 0],
                [Fraction(-1, 2), Fraction(1, 1000)],
                [Fraction(1, 2), Fraction(-1, 1000)],
                [1, 0],
            ],
            [Fraction(1, 4)] * 4,
        ),
    ]
    for raw_scores, weights in cases:
        scores = [[Fraction(value) for value in row] for row in raw_scores]
        partitions = _canonical_partitions(len(scores), 3)
        for beta in (Fraction(-1), Fraction(0), Fraction(1)):
            tilted_values = [row[0] - beta * row[1] for row in scores]
            brute = max(
                _exact_scalar_between_for_ds19(tilted_values, weights, labels, 3)
                for labels in partitions
            )
            interval = _exact_tie_interval_optimum_ds19(tilted_values, weights, 3)
            assert interval == brute
            for labels in partitions:
                information = _exact_binned_information(scores, labels, 3, weights)
                profiled = _ds18_profiled_value(information)
                same_labels = _exact_scalar_between_for_ds19(tilted_values, weights, labels, 3)
                assert profiled <= same_labels <= brute

    # Split-weight duplication is exactly invariant after pooling identical
    # full-score atoms, which is the ScoreQuant default on these small tables.
    fixture = json.loads(
        (RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-TILT-DUAL-GAP-001.json").read_text()
    )
    pooled: dict[tuple[Fraction, Fraction], Fraction] = {}
    for row, raw_weight in zip(fixture["scores"], fixture["weights"], strict=True):
        score = tuple(Fraction(value) for value in row)
        weight = Fraction(raw_weight)
        for _ in range(2):
            pooled[score] = pooled.get(score, Fraction(0)) + weight / 2
    assert sorted(pooled) == sorted(
        tuple(Fraction(value) for value in row) for row in fixture["scores"]
    )
    assert sorted(pooled.values()) == sorted(Fraction(value) for value in fixture["weights"])


def test_ds19_saddle_closure_pseudoinverse_scope_and_centered_cardinality() -> None:
    """The closure equation is exact, while DS11 and centered cardinality keep their scopes."""
    weights = [Fraction(1, 3)] * 3
    scores = [[Fraction(-1), Fraction(-1)], [Fraction(0), Fraction(1)], [Fraction(2), Fraction(0)]]
    singleton = (0, 1, 2)
    information = _exact_binned_information(scores, singleton, 3)
    beta = information[0][1] / information[1][1]
    profiled = _ds18_profiled_value(information)
    tilted = [row[0] - beta * row[1] for row in scores]
    dual = _exact_scalar_between_for_ds19(tilted, weights, singleton, 3)
    assert beta == Fraction(1, 2)
    assert beta * information[1][1] == information[0][1]
    assert profiled == dual == Fraction(3, 2)
    assert _canonical_partitions(3, 3) == [singleton]

    # The singular table remains in DS11's pseudo-inverse comparison class,
    # while DS9's regular class is empty at K=N=3.
    singular_scores = [
        [Fraction(-1), Fraction(0)],
        [Fraction(0), Fraction(0)],
        [Fraction(1), Fraction(0)],
    ]
    singular_information = _exact_binned_information(singular_scores, singleton, 3)
    assert singular_information[1][1] == singular_information[0][1] == 0
    assert _ds18_profiled_value(singular_information) == Fraction(2, 3)

    # With centered two-dimensional rows and only K=2 cells, the binned
    # information is a sum of two opposite rank-one cell moments and cannot
    # be positive definite. This pins the K >= d+1 cardinality boundary.
    centered = [
        [Fraction(-1), Fraction(0)],
        [Fraction(0), Fraction(-1)],
        [Fraction(1), Fraction(1)],
    ]
    for labels in _canonical_partitions(3, 2):
        matrix = _exact_binned_information(centered, labels, 2)
        assert matrix[0][0] * matrix[1][1] - matrix[0][1] ** 2 == 0


def test_ds19_ds18_empirical_projection_tax_identity_is_exact_on_practical_cases() -> None:
    """The DS18 strip-primal inputs satisfy the exact empirical tax identity partitionwise."""
    cases: list[tuple[list[list[Fraction]], list[Fraction]]] = []
    for size in range(4, 9):
        xs = [Fraction(2 * index + 1 - size, size) for index in range(size)]
        scores = _ds18_law_scores(tuple(xs), tuple(Fraction(0) for _ in xs))
        cases.append((scores, [Fraction(1, size)] * size))
    boundary_x = (Fraction(-3, 4), Fraction(-1, 4), Fraction(1, 4), Fraction(3, 4))
    boundary_z = (Fraction(-1), Fraction(-3, 4), Fraction(1), Fraction(1))
    cases.append((_ds18_law_scores(boundary_x, boundary_z), [Fraction(1, 4)] * 4))

    checked = 0
    for scores, weights in cases:
        full_cross = sum(
            weight * row[0] * row[1] for weight, row in zip(weights, scores, strict=True)
        )
        full_nuisance = sum(
            weight * row[1] * row[1] for weight, row in zip(weights, scores, strict=True)
        )
        assert full_nuisance > 0
        beta_hat = full_cross / full_nuisance
        efficient = [row[0] - beta_hat * row[1] for row in scores]
        assert (
            sum(
                weight * value * row[1]
                for weight, value, row in zip(weights, efficient, scores, strict=True)
            )
            == 0
        )

        for labels in _canonical_partitions(len(scores), 3):
            information = _exact_binned_information(scores, labels, 3)
            profiled = _ds18_profiled_value(information)
            between = _exact_scalar_between_for_ds19(efficient, weights, labels, 3)
            cross = information[0][1] - beta_hat * information[1][1]
            if information[1][1] == 0:
                assert cross == 0
                assert profiled == between
            else:
                assert profiled == between - cross * cross / information[1][1]
            assert profiled <= between
            checked += 1

    assert checked == 1394
    artifact = json.loads(
        (
            RESEARCH_WORKSPACE
            / "WORK"
            / "artifacts"
            / "DS-PRACTICAL-CERTIFIED-SOLVER"
            / "ds18-search.json"
        ).read_text()
    )
    assert artifact["summary"]["tax_identity_violations"] == 0
    assert artifact["summary"]["contiguity_disagreements"] == 0
    assert artifact["summary"]["positive_probe_gaps"] == 1


def test_ds19_matrix_tilt_outer_map_is_not_quasiconvex() -> None:
    """Tier B exact midpoint violation (CE-DS-MATRIX-TILT-NONQUASICONVEX-001)."""
    fixture = json.loads(
        (
            RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-MATRIX-TILT-NONQUASICONVEX-001.json"
        ).read_text()
    )
    scores = [[Fraction(value) for value in row] for row in fixture["scores"]]
    weights = [Fraction(value) for value in fixture["weights"]]
    assert all(
        sum(weight * row[column] for row, weight in zip(scores, weights, strict=True)) == 0
        for column in range(4)
    )
    information = [
        [
            sum(
                weight * score[row] * score[column]
                for score, weight in zip(scores, weights, strict=True)
            )
            for column in range(4)
        ]
        for row in range(4)
    ]
    assert information == [
        [Fraction(int(row == column)) for column in range(4)] for row in range(4)
    ]

    def determinant_at(matrix: list[list[Fraction]]) -> Fraction:
        # V(B)=I_2+B B^T for this unique singleton partition.
        first = 1 + matrix[0][0] ** 2 + matrix[0][1] ** 2
        second = 1 + matrix[1][0] ** 2 + matrix[1][1] ** 2
        cross = matrix[0][0] * matrix[1][0] + matrix[0][1] * matrix[1][1]
        return first * second - cross * cross

    first = [[Fraction(4), Fraction(0)], [Fraction(0), Fraction(0)]]
    second = [[Fraction(0), Fraction(0)], [Fraction(0), Fraction(4)]]
    midpoint = [
        [(left + right) / 2 for left, right in zip(left_row, right_row, strict=True)]
        for left_row, right_row in zip(first, second, strict=True)
    ]
    assert determinant_at(first) == determinant_at(second) == Fraction(17)
    assert determinant_at(midpoint) == Fraction(25) > Fraction(17)


# --------------------------------------------------------------------------- #
# AUDIT-DS-PRACTICAL-CERTIFIED-SOLVER: independent DS19 regressions (2 Sep 2026)
# --------------------------------------------------------------------------- #
def _audit19_info(
    scores: list[list[Fraction]],
    weights: list[Fraction],
    labels: tuple[int, ...],
    n_bins: int,
) -> tuple[Fraction, Fraction, Fraction]:
    masses = [Fraction(0)] * n_bins
    poi = [Fraction(0)] * n_bins
    lam = [Fraction(0)] * n_bins
    for row, weight, label in zip(scores, weights, labels, strict=True):
        masses[label] += weight
        poi[label] += weight * row[0]
        lam[label] += weight * row[1]
    assert all(mass > 0 for mass in masses)
    return (
        sum(p * p / m for p, m in zip(poi, masses, strict=True)),
        sum(p * q / m for p, q, m in zip(poi, lam, masses, strict=True)),
        sum(q * q / m for q, m in zip(lam, masses, strict=True)),
    )


def _audit19_phi(info: tuple[Fraction, Fraction, Fraction]) -> tuple[Fraction, bool]:
    poi, cross, lam = info
    if lam == 0:
        assert cross == 0
        return poi, False
    return poi - cross * cross / lam, True


def _audit19_quadratic(
    info: tuple[Fraction, Fraction, Fraction],
) -> tuple[Fraction, Fraction, Fraction]:
    poi, cross, lam = info
    return lam, -2 * cross, poi


def _audit19_eval(quad: tuple[Fraction, Fraction, Fraction], beta: Fraction) -> Fraction:
    a, b, c = quad
    return a * beta * beta + b * beta + c


def _audit19_interval_dp(
    values: list[Fraction],
    weights: list[Fraction],
    n_bins: int,
    tie_key: list[Fraction] | None = None,
) -> tuple[Fraction, tuple[int, ...]]:
    """Exact O(K N^2) contiguous DP in the sorted order with a supplied tie key."""
    order = sorted(
        range(len(values)),
        key=lambda i: (values[i], Fraction(0) if tie_key is None else tie_key[i], i),
    )
    n = len(order)
    pw = [Fraction(0)]
    pm = [Fraction(0)]
    for idx in order:
        pw.append(pw[-1] + weights[idx])
        pm.append(pm[-1] + weights[idx] * values[idx])
    best: list[list[Fraction | None]] = [[None] * (n + 1) for _ in range(n_bins + 1)]
    arg = [[-1] * (n + 1) for _ in range(n_bins + 1)]
    best[0][0] = Fraction(0)
    for cell in range(1, n_bins + 1):
        for j in range(cell, n + 1):
            for i in range(cell - 1, j):
                prev = best[cell - 1][i]
                if prev is None:
                    continue
                moment = pm[j] - pm[i]
                cand = prev + moment * moment / (pw[j] - pw[i])
                current = best[cell][j]
                if current is None or cand > current:
                    best[cell][j] = cand
                    arg[cell][j] = i
    value = best[n_bins][n]
    assert value is not None
    labels = [0] * n
    j = n
    for cell in range(n_bins, 0, -1):
        i = arg[cell][j]
        for pos in range(i, j):
            labels[order[pos]] = cell - 1
        j = i
    return value, _canonicalize_ds19(tuple(labels))


def _audit19_exact_dual_min(
    scores: list[list[Fraction]], weights: list[Fraction], n_bins: int
) -> tuple[Fraction, Fraction, list[tuple[int, ...]]]:
    """Exact rational dual minimizer for tables whose envelope minimum is rational.

    Enumerates every labeling quadratic, every vertex and every rational
    pairwise crossing, and certifies a candidate by the subgradient condition
    ``min active derivative <= 0 <= max active derivative``.
    """
    partitions = _canonical_partitions(len(scores), n_bins)
    quads = {z: _audit19_quadratic(_audit19_info(scores, weights, z, n_bins)) for z in partitions}
    candidates: set[Fraction] = set()
    for a, b, _c in quads.values():
        if a > 0:
            candidates.add(-b / (2 * a))
    items = list(quads.values())
    for first, second in combinations(items, 2):
        da, db, dc = (x - y for x, y in zip(first, second, strict=True))
        if da == 0:
            if db != 0:
                candidates.add(-dc / db)
            continue
        disc = db * db - 4 * da * dc
        if disc < 0:
            continue
        num, den = disc.numerator, disc.denominator
        root_num = math.isqrt(num)
        root_den = math.isqrt(den)
        if root_num * root_num == num and root_den * root_den == den:
            sqrt_disc = Fraction(root_num, root_den)
            candidates.add((-db - sqrt_disc) / (2 * da))
            candidates.add((-db + sqrt_disc) / (2 * da))
    for beta in sorted(candidates):
        values = {z: _audit19_eval(q, beta) for z, q in quads.items()}
        top = max(values.values())
        active = [z for z, v in values.items() if v == top]
        derivs = [2 * quads[z][0] * beta + quads[z][1] for z in active]
        if min(derivs) <= 0 <= max(derivs):
            return beta, top, active
    raise AssertionError("no rational certified minimizer among the candidates")


def test_ds19_audit_exact_dual_minimum_of_gap_witness() -> None:
    """The exact envelope minimum of the N=4 gap witness is rational and above the certificate."""
    fixture = json.loads(
        (RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-TILT-DUAL-GAP-001.json").read_text()
    )
    scores = [[Fraction(v) for v in row] for row in fixture["scores"]]
    weights = [Fraction(v) for v in fixture["weights"]]
    n_bins = fixture["K"]
    partitions = _canonical_partitions(len(scores), n_bins)
    g_plus = max(_audit19_phi(_audit19_info(scores, weights, z, n_bins))[0] for z in partitions)
    beta_star, d_exact, active = _audit19_exact_dual_min(scores, weights, n_bins)
    assert beta_star == Fraction(-8, 23)
    assert d_exact == Fraction(44729, 4232)
    assert sorted(active) == [(0, 0, 1, 2), (0, 1, 2, 2)]
    assert g_plus == Fraction(116805, 11816)
    certificate = Fraction(fixture["exact_quantities"]["certified_duality_gap_lower_bound"])
    assert d_exact - g_plus == Fraction(534361, 781333) >= certificate
    # the DP at beta* reproduces the envelope value in both perturbation orders
    tilted = [row[0] - beta_star * row[1] for row in scores]
    lam = [row[1] for row in scores]
    for key in ([-x for x in lam], lam):
        value, _labels = _audit19_interval_dp(tilted, weights, n_bins, key)
        assert value == d_exact
    # the bracket is open: no labeling is DP-optimal at its own normal-equation tilt
    for z in partitions:
        info = _audit19_info(scores, weights, z, n_bins)
        beta_z = info[1] / info[2]
        tilted_z = [row[0] - beta_z * row[1] for row in scores]
        brute = max(
            _exact_scalar_between_for_ds19(tilted_z, weights, labels, n_bins)
            for labels in partitions
        )
        assert _exact_scalar_between_for_ds19(tilted_z, weights, z, n_bins) < brute


def test_ds19_audit_support_minimal_gap_fixture_002() -> None:
    """CE-DS-TILT-DUAL-GAP-002: an N=3, K=2 gap with a fully rational proof of d=1/2."""
    fixture = json.loads(
        (RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-TILT-DUAL-GAP-002.json").read_text()
    )
    scores = [[Fraction(v) for v in row] for row in fixture["scores"]]
    weights = [Fraction(v) for v in fixture["weights"]]
    assert scores == [[-1, 0], [0, -1], [1, 0]]
    assert weights == [Fraction(1, 3)] * 3
    n_bins = fixture["K"]
    assert n_bins == 2
    partitions = _canonical_partitions(3, 2)
    assert len(partitions) == 3
    phis = {z: _audit19_phi(_audit19_info(scores, weights, z, 2)) for z in partitions}
    assert all(regular for _, regular in phis.values())
    g_plus = max(v for v, _ in phis.values())
    assert (
        g_plus == Fraction(1, 3) == Fraction(fixture["exact_quantities"]["global_profiled_value"])
    )
    beta_star, d_exact, active = _audit19_exact_dual_min(scores, weights, 2)
    assert beta_star == 0 and d_exact == Fraction(1, 2)
    assert sorted(active) == [(0, 0, 1), (0, 1, 1)]
    assert (
        d_exact - g_plus
        == Fraction(1, 6)
        == Fraction(fixture["exact_quantities"]["exact_duality_gap"])
    )
    # rational mixture proof: alpha = 1/2 of the two active quadratics is beta^2/6 + 1/2
    quads = {z: _audit19_quadratic(_audit19_info(scores, weights, z, 2)) for z in partitions}
    mixture = tuple(
        (left + right) / 2 for left, right in zip(quads[(0, 0, 1)], quads[(0, 1, 1)], strict=True)
    )
    assert mixture == (Fraction(1, 6), Fraction(0), Fraction(1, 2))
    # weak duality on every labeling at every probe tilt and at beta*
    for beta in (Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)):
        tilted = [row[0] - beta * row[1] for row in scores]
        brute = max(_exact_scalar_between_for_ds19(tilted, weights, z, 2) for z in partitions)
        assert brute >= d_exact
        for z in partitions:
            assert phis[z][0] <= _exact_scalar_between_for_ds19(tilted, weights, z, 2) <= brute
    # support minimality: N=2 (K=2) and N=3 (K=3) admit a single labeling
    assert _canonical_partitions(2, 2) == [(0, 1)]
    assert _canonical_partitions(3, 3) == [(0, 1, 2)]


def test_ds19_audit_tie_masked_closure_fixture() -> None:
    """CE-DS-TILT-DUAL-TIE-MASK-001: the bracket closes but one DP tie order hides it."""
    fixture = json.loads(
        (RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-TILT-DUAL-TIE-MASK-001.json").read_text()
    )
    scores = [[Fraction(v) for v in row] for row in fixture["scores"]]
    weights = [Fraction(v) for v in fixture["weights"]]
    assert scores == [[-1, -1], [0, -2], [0, 0]]
    n_bins = fixture["K"]
    partitions = _canonical_partitions(3, n_bins)
    infos = {z: _audit19_info(scores, weights, z, n_bins) for z in partitions}
    phis = {z: _audit19_phi(infos[z]) for z in partitions}
    assert all(regular for _, regular in phis.values())
    g_plus = max(v for v, _ in phis.values())
    beta_star, d_exact, active = _audit19_exact_dual_min(scores, weights, n_bins)
    assert beta_star == Fraction(1, 3) and d_exact == Fraction(2, 9) == g_plus
    assert sorted(active) == [(0, 1, 0), (0, 1, 1)]
    tilted = [row[0] - beta_star * row[1] for row in scores]
    assert len(set(tilted)) == 3  # a DP value tie, not a tilted-value tie
    quads = {z: _audit19_quadratic(infos[z]) for z in partitions}
    derivative = {z: 2 * quads[z][0] * beta_star + quads[z][1] for z in active}
    assert derivative[(0, 1, 1)] == 0 and derivative[(0, 1, 0)] == Fraction(2, 3)
    assert phis[(0, 1, 1)][0] == g_plus and phis[(0, 1, 0)][0] == Fraction(4, 27) < g_plus
    # normal equation holds only for the saddle labeling
    assert beta_star * infos[(0, 1, 1)][2] == infos[(0, 1, 1)][1]
    assert beta_star * infos[(0, 1, 0)][2] != infos[(0, 1, 0)][1]
    # both members are contiguous in the (tie-free) sorted order and both attain the
    # DP value, so a deterministic tie policy decides which one is reported
    order = sorted(range(3), key=lambda i: tilted[i])
    for z in active:
        cells = [z[i] for i in order]
        assert cells == sorted(cells) or cells == sorted(cells, reverse=True)
        assert _exact_scalar_between_for_ds19(tilted, weights, z, n_bins) == d_exact
    dp_value, dp_labels = _audit19_interval_dp(tilted, weights, n_bins)
    assert dp_value == d_exact and dp_labels in active
    # a policy preferring the larger one-sided derivative (the member active on the
    # right of beta*) returns the non-closing labeling and reports an open interval
    preferred = max(active, key=lambda z: derivative[z])
    assert preferred == (0, 1, 0)
    reported = [phis[preferred][0], d_exact]
    assert reported[0] < reported[1]  # an open reported interval on a closed bracket
    shifted = [row[0] - (beta_star + Fraction(1, 10**6)) * row[1] for row in scores]
    right_value = max(
        _exact_scalar_between_for_ds19(shifted, weights, z, n_bins) for z in partitions
    )
    assert _exact_scalar_between_for_ds19(shifted, weights, preferred, n_bins) == right_value


def test_ds19_audit_weak_ceiling_and_domain_split_on_adversarial_tables() -> None:
    """Weak duality, the DS11 identity, and the DS9/DS11 split on the audit's attack tables."""
    tables: list[tuple[list[list[Fraction]], list[Fraction], int]] = [
        (
            [
                [Fraction(-2), Fraction(1)],
                [Fraction(-1), Fraction(-1)],
                [Fraction(1), Fraction(-1)],
                [Fraction(2), Fraction(1)],
            ],
            [Fraction(1, 4)] * 4,
            2,
        ),
        (
            [
                [Fraction(-1), Fraction(1)],
                [Fraction(0), Fraction(0)],
                [Fraction(1), Fraction(-1)],
                [Fraction(2), Fraction(0)],
                [Fraction(1, 2), Fraction(2)],
            ],
            [Fraction(2, 7), Fraction(1, 7), Fraction(2, 7), Fraction(1, 7), Fraction(1, 7)],
            3,
        ),
        (
            [
                [Fraction(0), Fraction(0)],
                [Fraction(1), Fraction(1)],
                [Fraction(2), Fraction(2)],
                [Fraction(3), Fraction(3)],
                [Fraction(1), Fraction(-1)],
            ],
            [Fraction(1, 15), Fraction(2, 15), Fraction(3, 15), Fraction(4, 15), Fraction(5, 15)],
            3,
        ),
    ]
    split_seen = False
    singular_dp_state_seen = False
    for scores, weights, n_bins in tables:
        partitions = _canonical_partitions(len(scores), n_bins)
        infos = {z: _audit19_info(scores, weights, z, n_bins) for z in partitions}
        phis = {z: _audit19_phi(infos[z]) for z in partitions}
        g_plus = max(v for v, _ in phis.values())
        regular_values = [v for v, regular in phis.values() if regular]
        g_reg = max(regular_values) if regular_values else None
        if g_reg is not None and g_plus > g_reg:
            split_seen = True
        for beta in (Fraction(-2), Fraction(-1), Fraction(0), Fraction(1), Fraction(2)):
            tilted = [row[0] - beta * row[1] for row in scores]
            brute = max(
                _exact_scalar_between_for_ds19(tilted, weights, z, n_bins) for z in partitions
            )
            dp_value, dp_labels = _audit19_interval_dp(tilted, weights, n_bins)
            assert dp_value == brute  # contiguity, including exact ties at beta = 1
            if not phis[dp_labels][1]:
                singular_dp_state_seen = True
            for z in partitions:
                info = infos[z]
                tilted_form = _audit19_eval(_audit19_quadratic(info), beta)
                assert phis[z][0] <= tilted_form <= brute
                # DS11 completion of squares: V_z(beta) - Phi+(z) = (beta - beta_z)^2 I_ll
                if info[2] > 0:
                    beta_z = info[1] / info[2]
                    assert tilted_form - phis[z][0] == (beta - beta_z) ** 2 * info[2]
                else:
                    assert tilted_form == phis[z][0]
            assert g_plus <= brute
    assert split_seen
    assert singular_dp_state_seen


def test_ds19_audit_saddle_closure_iff_on_small_tables() -> None:
    """Closure <=> a DP-optimal labeling solving its normal equation, on exact small tables."""
    tables: list[tuple[list[list[Fraction]], list[Fraction], int]] = [
        ([[-1, 0], [0, -1], [1, 0]], [Fraction(1, 3)] * 3, 2),  # open
        ([[-1, -1], [0, -2], [0, 0]], [Fraction(1, 3)] * 3, 2),  # closed, regular saddle
        ([[-2, 1], [-1, -1], [1, -1], [2, 1]], [Fraction(1, 4)] * 4, 2),  # closed, singular saddle
        ([[-1, -1], [0, 1], [2, 0]], [Fraction(1, 3)] * 3, 3),  # K = N closes
        (
            [
                [Fraction(-11, 2), Fraction(39, 8)],
                [Fraction(3, 2), Fraction(-65, 8)],
                [Fraction(7, 2), Fraction(31, 8)],
                [Fraction(9, 2), Fraction(-49, 8)],
            ],
            [Fraction(1, 4)] * 4,
            3,
        ),  # open
    ]
    outcomes = []
    for raw_scores, weights, n_bins in tables:
        scores = [[Fraction(v) for v in row] for row in raw_scores]
        partitions = _canonical_partitions(len(scores), n_bins)
        infos = {z: _audit19_info(scores, weights, z, n_bins) for z in partitions}
        phis = {z: _audit19_phi(infos[z]) for z in partitions}
        g_plus = max(v for v, _ in phis.values())
        beta_star, d_exact, active = _audit19_exact_dual_min(scores, weights, n_bins)
        closed = g_plus == d_exact
        assert g_plus <= d_exact
        saddles = []
        for z in partitions:
            info = infos[z]
            if info[2] > 0:
                beta_z = info[1] / info[2]
                tilted = [row[0] - beta_z * row[1] for row in scores]
                brute = max(
                    _exact_scalar_between_for_ds19(tilted, weights, y, n_bins) for y in partitions
                )
                if _exact_scalar_between_for_ds19(tilted, weights, z, n_bins) == brute:
                    saddles.append(z)
            elif phis[z][0] == d_exact:
                saddles.append(z)
        assert bool(saddles) == closed
        if closed:
            # every Phi+-maximiser is active at beta* with zero derivative
            for z in partitions:
                if phis[z][0] == g_plus:
                    assert z in active
                    assert beta_star * infos[z][2] == infos[z][1]
        outcomes.append((closed, any(phis[z][1] for z in saddles)))
    assert outcomes == [(False, False), (True, True), (True, False), (True, True), (False, False)]


def test_ds19_audit_tie_order_independence_and_one_sided_derivatives() -> None:
    """Tie lemma: every tie order gives v_K; perturbation orders give one-sided derivatives."""
    scores = [
        [Fraction(0), Fraction(0)],
        [Fraction(1), Fraction(1)],
        [Fraction(2), Fraction(2)],
        [Fraction(3), Fraction(3)],
        [Fraction(1), Fraction(-1)],
    ]
    weights = [Fraction(1, 15), Fraction(2, 15), Fraction(3, 15), Fraction(4, 15), Fraction(5, 15)]
    beta = Fraction(1)
    tilted = [row[0] - beta * row[1] for row in scores]
    assert tilted[:4] == [0, 0, 0, 0]  # a four-way exact tie with unequal weights
    for n_bins in (2, 3, 4):
        partitions = _canonical_partitions(5, n_bins)
        brute = max(_exact_scalar_between_for_ds19(tilted, weights, z, n_bins) for z in partitions)
        values = set()
        for order in permutations(range(4)):
            key = [Fraction(order.index(i)) if i < 4 else Fraction(0) for i in range(5)]
            value, _ = _audit19_interval_dp(tilted, weights, n_bins, key)
            values.add(value)
        assert values == {brute}
        # one-sided derivatives: the labeling with the extreme derivative (and, among
        # equal derivatives, the largest curvature) is contiguous in the perturbation
        # order and stays optimal on its side of beta
        quads = {
            z: _audit19_quadratic(_audit19_info(scores, weights, z, n_bins)) for z in partitions
        }
        active = [z for z in partitions if _audit19_eval(quads[z], beta) == brute]
        lam = [row[1] for row in scores]
        h = Fraction(1, 10**6)
        for sign, key in ((1, [-x for x in lam]), (-1, lam)):
            extreme = max(
                active, key=lambda z: (sign * (2 * quads[z][0] * beta + quads[z][1]), quads[z][0])
            )
            order = sorted(range(5), key=lambda i: (tilted[i], key[i], i))
            cells = [extreme[i] for i in order]
            assert len(set(cells)) == n_bins
            # contiguity in the perturbation order: each cell occupies one run
            runs = 1 + sum(cells[i] != cells[i + 1] for i in range(4))
            assert runs == n_bins
            shifted = [row[0] - (beta + sign * h) * row[1] for row in scores]
            side_value = max(
                _exact_scalar_between_for_ds19(shifted, weights, z, n_bins) for z in partitions
            )
            assert _exact_scalar_between_for_ds19(shifted, weights, extreme, n_bins) == side_value
    # the convexity step behind the lemma: g -> (M + t g)^2 / (W + g) is convex on g >= 0
    M, t, W = Fraction(3, 2), Fraction(-2), Fraction(5, 4)
    f = [(M + t * g) ** 2 / (W + g) for g in (Fraction(0), Fraction(1, 2), Fraction(2))]
    assert f[1] <= f[0] + Fraction(1, 4) * (f[2] - f[0])


def test_ds19_audit_ds18_strip_dp_delta_chain_on_seeded_sample() -> None:
    """On an exact dyadic DS18 sample the beta-zero DP labeling obeys the Delta chain exactly."""
    size = 96
    state = 20260902 + 1000 * size
    denominator = 1 << 16
    xs: list[Fraction] = []
    zs: list[Fraction] = []
    for _ in range(size):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        xs.append(Fraction(2 * ((state >> 33) % denominator) + 1, denominator) - 1)
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        zs.append(Fraction(2 * ((state >> 33) % denominator) + 1, denominator) - 1)
    scores = _ds18_law_scores(tuple(xs), tuple(zs))
    weights = [Fraction(1, size)] * size
    v_hat, labels = _audit19_interval_dp(xs, weights, 3)
    info = _audit19_info(scores, weights, labels, 3)
    phi, regular = _audit19_phi(info)
    assert regular
    delta = v_hat - phi
    assert delta == info[1] ** 2 / info[2] >= 0
    # uncentered between value = centered between value + xbar^2 (labeling independent)
    xbar = sum(xs) / size
    centered = _exact_scalar_between_for_ds19([x - xbar for x in xs], weights, labels, 3)
    assert v_hat == centered + xbar * xbar
    # DS18's finite-N disagreement bound with the labeling's own Delta
    third = Fraction(1, 3)
    population = tuple(0 if x < -third else (1 if x < third else 2) for x in xs)
    disagreement = Fraction(sum(a != b for a, b in zip(labels, population, strict=True)), size)
    for eta in (Fraction(1, 10), Fraction(1, 20)):
        band = Fraction(sum(abs(x - third) <= eta or abs(x + third) <= eta for x in xs), size)
        assert disagreement <= 3 * delta / eta + band
    # the DP cells are intervals whose cuts sit near +/-1/3 and the nuisance block near 32/81
    order = sorted(range(size), key=lambda i: xs[i])
    cells = [labels[i] for i in order]
    assert cells == sorted(cells)
    assert abs(info[2] - Fraction(32, 81)) < Fraction(1, 5)
    assert abs(info[1]) < Fraction(1, 5)


def test_ds19_audit_matrix_tilt_midpoint_violation_by_direct_evaluation() -> None:
    """Tier B: V(B) evaluated directly from the tilted form, not from the closed form."""
    fixture = json.loads(
        (
            RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-DS-MATRIX-TILT-NONQUASICONVEX-001.json"
        ).read_text()
    )
    scores = [[Fraction(v) for v in row] for row in fixture["scores"]]
    weights = [Fraction(v) for v in fixture["weights"]]
    assert fixture["K"] == len(scores) == 8
    assert _canonical_partitions(8, 8) == [tuple(range(8))]
    info = [
        [sum(w * s[r] * s[c] for s, w in zip(scores, weights, strict=True)) for c in range(4)]
        for r in range(4)
    ]

    def tilted_form(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
        # V(B) = I_pp - B I_lp - I_pl B^T + B I_ll B^T with psi = {0,1}, lambda = {2,3}
        out = [[Fraction(0)] * 2 for _ in range(2)]
        for r in range(2):
            for c in range(2):
                value = info[r][c]
                for k in range(2):
                    value -= matrix[r][k] * info[2 + k][c] + info[r][2 + k] * matrix[c][k]
                for k in range(2):
                    for m in range(2):
                        value += matrix[r][k] * info[2 + k][2 + m] * matrix[c][m]
                out[r][c] = value
        return out

    def det(matrix: list[list[Fraction]]) -> Fraction:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    b_first = [[Fraction(4), Fraction(0)], [Fraction(0), Fraction(0)]]
    b_second = [[Fraction(0), Fraction(0)], [Fraction(0), Fraction(4)]]
    b_mid = [[Fraction(2), Fraction(0)], [Fraction(0), Fraction(2)]]
    dets = [det(tilted_form(b)) for b in (b_first, b_second, b_mid)]
    assert dets == [Fraction(17), Fraction(17), Fraction(25)]
    assert dets[2] > max(dets[:2])
    # weak matrix-tilt duality: det S+ = det I_2 = 1 <= every det V(B)
    assert all(d >= 1 for d in dets)
    # and the generic closed form V(B) = I_2 + B B^T at a non-diagonal probe
    probe = [[Fraction(1), Fraction(-2)], [Fraction(3), Fraction(1, 2)]]
    expected = [
        [(1 if r == c else 0) + sum(probe[r][k] * probe[c][k] for k in range(2)) for c in range(2)]
        for r in range(2)
    ]
    assert tilted_form(probe) == expected


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


def _o6_audit_plugin(
    scores: list[Fraction], labels: list[int], n_bins: int
) -> tuple[Fraction, Fraction]:
    """Exact plug-in retention (0/0 := 0) and its influence variance for one sample."""
    n = len(scores)
    counts = [labels.count(b) for b in range(n_bins)]
    sums = [sum(s for s, z in zip(scores, labels, strict=True) if z == b) for b in range(n_bins)]
    means = [sums[b] / counts[b] if counts[b] else Fraction(0) for b in range(n_bins)]
    v = sum(s * s for s in scores) / n
    if v == 0:
        return Fraction(0), Fraction(0)
    eta = sum(sums[b] ** 2 / counts[b] for b in range(n_bins) if counts[b]) / (n * v)
    psi = [
        ((1 - eta) * s * s - (s - means[z]) ** 2) / v for s, z in zip(scores, labels, strict=True)
    ]
    assert sum(psi) == 0
    return eta, sum(x * x for x in psi) / n


def test_o6_audit_eta_zero_law_has_zero_influence_variance_with_many_atoms() -> None:
    """CE-O6-ETA-ZERO-MULTIATOM-VARIANCE-001 (AUDIT-SCORE-ORACLE-ROBUSTNESS).

    At eta = 0 the root equation (s - c_b)^2 = (1 - eta) s^2 is the identity,
    so every law whose cell means vanish has psi = 0 identically: sigma^2 = 0
    is not restricted to two atoms per cell, and an atomless cell does not
    imply (A4) unless eta > 0. The fixture has four atoms in cell 0.
    """
    fixture_path = (
        RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-O6-ETA-ZERO-MULTIATOM-VARIANCE-001.json"
    )
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


def test_o6_audit_two_atom_law_has_zero_variance_and_upward_biased_plugin() -> None:
    """AUDIT-SCORE-ORACLE-ROBUSTNESS: an explicit sigma^2 = 0 law with 0 < eta < 1.

    Two cells of probability 1/2; on each, S sits on the roots c/(1 -+ sqrt(1 - eta))
    with weights (3/4, 1/4), giving eta = 3/4, E[S] = 0 and psi = 0 at every atom.
    Exhaustive enumeration of every sample composition with n <= 6 shows the
    plug-in ratio never falls below eta and equals eta only when sigma_hat = 0.
    """
    atoms = [
        (Fraction(2, 3), 0, Fraction(3, 8)),
        (Fraction(2), 0, Fraction(1, 8)),
        (Fraction(-2, 3), 1, Fraction(3, 8)),
        (Fraction(-2), 1, Fraction(1, 8)),
    ]
    p = [Fraction(1, 2), Fraction(1, 2)]
    c = [Fraction(1), Fraction(-1)]
    v = sum(w * s * s for s, _, w in atoms)
    eta = sum(p[b] * c[b] ** 2 for b in range(2)) / v
    assert v == Fraction(4, 3) and eta == Fraction(3, 4)
    assert sum(w * s for s, _, w in atoms) == 0
    root = 1 - eta
    assert all((s - c[z]) ** 2 == root * s * s for s, z, _ in atoms)
    assert all(((1 - eta) * s * s - (s - c[z]) ** 2) == 0 for s, z, _ in atoms)
    for n in range(1, 7):
        for counts in product(range(n + 1), repeat=4):
            if sum(counts) != n:
                continue
            scores: list[Fraction] = []
            labels: list[int] = []
            for (s, z, _), count in zip(atoms, counts, strict=True):
                scores.extend([s] * count)
                labels.extend([z] * count)
            eta_hat, sigma2_hat = _o6_audit_plugin(scores, labels, 2)
            assert eta_hat >= eta
            if eta_hat == eta:
                assert sigma2_hat == 0


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
    fixture_path = RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-O7-ELLIPSOID-ZERO-VARIANCE-001.json"
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
    fixture_path = (
        RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / "CE-O7-UNIT-RETENTION-SINGULAR-SAMPLE-001.json"
    )
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


def test_o7_audit_ellipsoid_identity_holds_on_an_anisotropic_rational_law() -> None:
    """AUDIT-RETENTION-PLUGIN-VECTOR: the O7.4(a) zero set is the ellipsoid for general V.

    psi_r(s) = -r [(s - V I_Z^{-1} c_b)^T V^{-1} (s - V I_Z^{-1} c_b)
    - c_b^T I_Z^{-1} (V - I_Z) I_Z^{-1} c_b] as a polynomial identity in s, on a
    d = 2 law whose V is not a multiple of the identity, checked at generic
    rational points off the support for every cell.
    """
    rows = [(1, 2), (3, 0), (2, -1), (-2, 1), (0, 3), (-1, -3), (1, -1), (-3, 0), (2, 2)]
    labels = [0, 0, 0, 1, 1, 2, 2, 2, 1]
    n_bins = 3
    scores = [[Fraction(a), Fraction(b)] for a, b in rows]
    weights = [Fraction(w, 12) for w in (1, 2, 1, 2, 1, 1, 1, 2, 1)]
    assert sum(weights) == 1
    ratio, _, v, i_z = _o7_exact_plugin(scores, labels, weights, n_bins)
    assert v[0][1] != 0 or v[0][0] != v[1][1]

    def det(a: list[list[Fraction]]) -> Fraction:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]

    def inv(a: list[list[Fraction]]) -> list[list[Fraction]]:
        dd = det(a)
        return [[a[1][1] / dd, -a[0][1] / dd], [-a[1][0] / dd, a[0][0] / dd]]

    def mv(a: list[list[Fraction]], x: list[Fraction]) -> list[Fraction]:
        return [a[0][0] * x[0] + a[0][1] * x[1], a[1][0] * x[0] + a[1][1] * x[1]]

    def quad(a: list[list[Fraction]], x: list[Fraction], y: list[Fraction]) -> Fraction:
        return x[0] * (a[0][0] * y[0] + a[0][1] * y[1]) + x[1] * (a[1][0] * y[0] + a[1][1] * y[1])

    iz_inv, v_inv = inv(i_z), inv(v)
    p = [sum(w for w, z in zip(weights, labels, strict=True) if z == b) for b in range(n_bins)]
    m = [
        [
            sum(w * s[i] for s, w, z in zip(scores, weights, labels, strict=True) if z == b)
            for i in range(2)
        ]
        for b in range(n_bins)
    ]
    v_minus_i = [[v[i][j] - i_z[i][j] for j in range(2)] for i in range(2)]
    for b in range(n_bins):
        c = [x / p[b] for x in m[b]]
        a = mv(iz_inv, c)
        centre = mv(v, a)
        rhs = quad(v_minus_i, a, a)
        assert rhs >= 0
        for k in range(-4, 5):
            s = [Fraction(k, 3), Fraction(2 - k, 5)]
            psi_r = ratio * (2 * quad(iz_inv, s, c) - quad(iz_inv, c, c) - quad(v_inv, s, s))
            diff = [s[0] - centre[0], s[1] - centre[1]]
            assert psi_r + ratio * (quad(v_inv, diff, diff) - rhs) == 0


# ---------------------------------------------------------------------------
# O8 — score-error budget for a frozen rule (SCORE-ERROR-BUDGET, closure step 3)
# ---------------------------------------------------------------------------


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
    path = RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / f"{name}.json"
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


def test_o8_boundary_atom_fixture_mislabel_mass_does_not_vanish() -> None:
    """CE-SCORE-ERROR-BOUNDARY-ATOM-001 (SCORE-ERROR-RULE-TRANSFER).

    Scalar centred law with an atom of mass 1/2 on the boundary of the threshold
    rule Z = 1{s > 0}. Any proxy lifting that atom relabels it: pi = 1/2 for every
    kappa > 0, the true retention of the transferred labels jumps 4/5 -> 1/2, and
    the Loewner sandwich I_Zhat >= I_Z - Gamma holds with Gamma = 4.
    """
    fixture = _o8_fixture("CE-SCORE-ERROR-BOUNDARY-ATOM-001")
    atoms = [[Fraction(x) for x in s] for s in fixture["scores"]]
    weights = [Fraction(w) for w in fixture["weights"]]
    labels = list(fixture["labels_before"])
    assert sum(w * s[0] for s, w in zip(atoms, weights, strict=True)) == 0
    for kappa in (Fraction(1, 10**6), Fraction(1, 10), Fraction(3)):
        proxy = [[s[0] + (kappa if s[0] == 0 else 0)] for s in atoms]
        labels_hat = [int(s[0] > 0) for s in proxy]
        assert labels_hat == fixture["labels_after_or_optimum"]
        pi = sum(w for w, a, b in zip(weights, labels, labels_hat, strict=True) if a != b)
        assert pi == Fraction(1, 2)
        _, c, v, i_z = _o8_moments(atoms, weights, labels, 2)
        _, _, _, i_hat = _o8_moments(atoms, weights, labels_hat, 2)
        assert i_z[0][0] / v[0][0] == Fraction(4, 5)
        assert i_hat[0][0] / v[0][0] == Fraction(1, 2)
        gamma = 2 * sum(
            w * (s[0] ** 2 + c[b][0] ** 2)
            for s, w, a, b in zip(atoms, weights, labels, labels_hat, strict=True)
            if a != b
        )
        assert gamma == 4
        assert i_hat[0][0] >= i_z[0][0] - gamma


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


def test_o8_reporting_budget_core_inequalities_on_random_rational_laws() -> None:
    """SCORE-ERROR-RETENTION-BUDGET, Lemma 2 and the sharpness attainer, exactly.

    On deterministic random rational atomic laws in d = 2 with a random rational
    proxy error: (tr(I_Z^{-1}(I~_Z - I_Z)) - eps_R^2)^2 <= 4 d eps_R^2 and the
    same for (V, V~, eps); with e_b = kappa c_b the determinant ratio is exactly
    (1 + kappa)^{2d} and eps_R^2 = d kappa^2.
    """
    rng = np.random.default_rng(20260908)
    d, n_bins, checked = 2, 3, 0
    for _ in range(40):
        n = int(rng.integers(4, 9))
        atoms = [[Fraction(int(x), 4) for x in rng.integers(-8, 9, size=d)] for _ in range(n)]
        raw = [Fraction(int(x)) for x in rng.integers(1, 7, size=n)]
        weights = [w / sum(raw) for w in raw]
        labels = [int(x) for x in rng.integers(0, n_bins, size=n)]
        labels[:n_bins] = list(range(n_bins))
        errors = [[Fraction(int(x), 8) for x in rng.integers(-4, 5, size=d)] for _ in range(n)]
        p, c, v, i_z = _o8_moments(atoms, weights, labels, n_bins)
        if _o8_det(v) == 0 or _o8_det(i_z) == 0:
            continue
        for err_scale in (None, Fraction(1, 3)):
            if err_scale is None:
                errs = errors
            else:
                errs = [[err_scale * x for x in c[z]] for z in labels]
            proxy = [
                [a + b for a, b in zip(s, e, strict=True)] for s, e in zip(atoms, errs, strict=True)
            ]
            _, _, vt, it = _o8_moments(proxy, weights, labels, n_bins)
            _, e_c, e_full, e_z = _o8_moments(errs, weights, labels, n_bins)
            for a_mat, at_mat, e_mat in ((i_z, it, e_z), (v, vt, e_full)):
                det_a = _o8_det(a_mat)
                inv_a = [
                    [a_mat[1][1] / det_a, -a_mat[0][1] / det_a],
                    [-a_mat[1][0] / det_a, a_mat[0][0] / det_a],
                ]
                eps2 = sum(inv_a[i][j] * e_mat[j][i] for i in range(d) for j in range(d))
                tr = sum(
                    inv_a[i][j] * (at_mat[j][i] - a_mat[j][i]) for i in range(d) for j in range(d)
                )
                assert (tr - eps2) ** 2 <= 4 * d * eps2
                checked += 1
            if err_scale is not None:
                assert _o8_det(it) == (1 + err_scale) ** (2 * d) * _o8_det(i_z)
                det_i = _o8_det(i_z)
                inv_i = [
                    [i_z[1][1] / det_i, -i_z[0][1] / det_i],
                    [-i_z[1][0] / det_i, i_z[0][0] / det_i],
                ]
                eps_r2 = sum(inv_i[i][j] * e_z[j][i] for i in range(d) for j in range(d))
                assert eps_r2 == d * err_scale * err_scale
    assert checked >= 100


def test_o8_audit_log_lower_positive_spectral_floor() -> None:
    """A rational upper Riemann sum refutes the old scalar lower bound."""
    fixture = _o8_fixture("CE-SCORE-ERROR-LOG-LOWER-001")
    q = fixture["exact_quantities"]
    atoms = [[Fraction(x) for x in row] for row in fixture["scores"]]
    proxy = [[Fraction(x) for x in row] for row in q["proxy_scores"]]
    weights = [Fraction(x) for x in fixture["weights"]]
    _, _, v, _ = _o8_moments(atoms, weights, fixture["labels_before"], 2)
    _, _, vt, _ = _o8_moments(proxy, weights, fixture["labels_before"], 2)
    lam = vt[0][0] / v[0][0] - 1
    assert lam == Fraction(q["lambda"])
    old_lower = lam - lam**2 / (2 * (1 + lam))
    assert old_lower == Fraction(q["claimed_log_lower"])
    # Integral of decreasing 1/t on [1, 1+lambda] is below its left sum.
    upper_sum = sum((lam / 32) / (1 + j * lam / 32) for j in range(32))
    assert upper_sum < old_lower
    # Corrected scalar lower bound uses min(0, lambda_min); a right sum suffices.
    lower_sum = sum((lam / 32) / (1 + j * lam / 32) for j in range(1, 33))
    assert lam - lam**2 / 2 < lower_sum


def test_o8_audit_alignment_bounds_are_not_ordered() -> None:
    """Both alignment bounds hold, but the displayed chained ordering is false."""
    fixture = _o8_fixture("CE-SCORE-ERROR-ALIGNMENT-ORDER-001")
    q = fixture["exact_quantities"]
    atoms = [[Fraction(x) for x in row] for row in fixture["scores"]]
    errors = [[Fraction(x) for x in row] for row in q["errors"]]
    weights = [Fraction(x) for x in fixture["weights"]]
    labels = fixture["labels_before"]
    p, c, v, retained = _o8_moments(atoms, weights, labels, fixture["K"])
    _, ec, ef, ez = _o8_moments(errors, weights, labels, fixture["K"])
    assert v[0][1] == retained[0][1] == 0
    rho = [retained[j][j] / v[j][j] for j in range(2)]
    eps2 = sum(ef[j][j] / v[j][j] for j in range(2))
    eps_z2 = sum(ez[j][j] / v[j][j] for j in range(2))
    eps_r2 = sum(ez[j][j] / retained[j][j] for j in range(2))
    assert eps2 == eps_z2 == eps_r2 == Fraction(q["eps2"])
    between = sum(p[b] * ec[b][j] * c[b][j] / retained[j][j] for b in range(3) for j in range(2))
    total = sum(
        w * e[j] * s[j] / v[j][j]
        for s, e, w in zip(atoms, errors, weights, strict=True)
        for j in range(2)
    )
    assert between - total == Fraction(q["T1"]) == 0
    trace_squared = (2 - sum(rho)) * eps_z2 * (1 - min(rho)) / min(rho)
    crude_squared = 8 * eps2  # d=2 and eps_R=eps
    assert trace_squared == Fraction(q["trace_bound_squared"])
    assert crude_squared == Fraction(q["crude_bound_squared"])
    assert trace_squared > crude_squared


def test_o8_audit_translations_change_uncentred_retention() -> None:
    """An affine translation changes a positive, nontrivial retention ratio."""
    fixture = _o8_fixture("CE-SCORE-ERROR-TRANSLATION-001")
    q = fixture["exact_quantities"]
    weights = [Fraction(x) for x in fixture["weights"]]
    for key, rows in (("eta_before", fixture["scores"]), ("eta_after", q["proxy_scores"])):
        atoms = [[Fraction(x) for x in row] for row in rows]
        _, _, v, retained = _o8_moments(atoms, weights, fixture["labels_before"], 2)
        assert retained[0][0] / v[0][0] == Fraction(q[key])
    assert q["eta_before"] != q["eta_after"]


def test_o8_audit_least_squares_map_can_be_singular() -> None:
    """Positive true and proxy second moments do not ensure an invertible A*."""
    fixture = _o8_fixture("CE-SCORE-ERROR-SINGULAR-LS-001")
    q = fixture["exact_quantities"]
    truth = [Fraction(row[0]) for row in fixture["scores"]]
    proxy = [Fraction(row[0]) for row in q["proxy_scores"]]
    weights = [Fraction(x) for x in fixture["weights"]]
    v = sum(w * s * s for w, s in zip(weights, truth, strict=True))
    vt = sum(w * h * h for w, h in zip(weights, proxy, strict=True))
    cross = sum(w * s * h for w, s, h in zip(weights, truth, proxy, strict=True))
    a = cross / vt
    assert v == vt == 1
    assert a == Fraction(q["A_star"]) == 0
    assert a * a * vt == Fraction(q["transformed_V"]) == 0
    assert 1 - cross * cross / (v * vt) == Fraction(q["eps_linear2"])


@pytest.mark.parametrize("suffix", ["CHART", "RETENTION"])
def test_o8_audit_calibration_does_not_certify_retention_distortion(suffix: str) -> None:
    """Recompute posterior, score, reliability and retention under valid mixture laws."""
    fixture = _o8_fixture(f"CE-CLASSIFIER-CALIBRATION-{suffix}-001")
    q = fixture["exact_quantities"]
    weights = [Fraction(x) for x in fixture["weights"]]
    eta = [[Fraction(x) for x in row] for row in q["posterior"]]
    hat = [[Fraction(x) for x in row] for row in q["proxy_posterior"]]
    theta = [Fraction(x) for x in q["theta0"]]
    prior = [Fraction(x) for x in q["prior"]]
    chart = [Fraction(x) for x in q["chart"][0]]
    assert theta == prior
    assert sum(chart) == 0  # valid tangent of normalized mixture fractions
    for j, pi in enumerate(prior):
        assert sum(w * row[j] for w, row in zip(weights, eta, strict=True)) == pi
    for row in eta + hat:
        assert sum(row) == 1 and min(row) >= 0
    assert len({tuple(row) for row in hat}) == len(hat)
    # Each proxy value identifies X, so E[eta | hat eta]=eta: no resolution gap.
    reliability = sum(
        w * sum((a - b) ** 2 for a, b in zip(e, h, strict=True))
        for w, e, h in zip(weights, eta, hat, strict=True)
    )
    assert reliability == Fraction(q["reliability"]) > 0

    def scores(posteriors: list[list[Fraction]]) -> list[list[Fraction]]:
        output = []
        for row in posteriors:
            ratios = [x / pi for x, pi in zip(row, prior, strict=True)]
            denom = sum(t * r for t, r in zip(theta, ratios, strict=True))
            output.append([sum(t * r / denom for t, r in zip(chart, ratios, strict=True))])
        return output

    truth, proxy = scores(eta), scores(hat)
    assert truth == [[Fraction(row[0])] for row in fixture["scores"]]
    _, _, v, retained = _o8_moments(truth, weights, fixture["labels_before"], 2)
    _, _, vt, it = _o8_moments(proxy, weights, fixture["labels_before"], 2)
    eps2 = (
        sum(w * (a[0] - b[0]) ** 2 for w, a, b in zip(weights, truth, proxy, strict=True)) / v[0][0]
    )
    assert eps2 == Fraction(q["eps2"])
    assert retained[0][0] / v[0][0] == Fraction(q["eta_true"])
    assert it[0][0] / vt[0][0] == Fraction(q["eta_reported"]) == 1


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
    fixture = json.loads(
        (RESEARCH_WORKSPACE / "COUNTEREXAMPLES" / f"{fixture_id}.json").read_text()
    )
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
