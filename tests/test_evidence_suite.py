from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scorequant as sq
from examples.baselines import rectangular_observation_bins
from examples.door3_classifier import run_study as run_door3_study
from examples.ds_geometry_counterexample import (
    canonical_labelings,
    efficient_semimetric,
    exact_table,
    library_run,
    profiled_value,
    violation_margins,
)
from examples.global_certification import restart_hit_rates
from examples.hep_classifier.experiment import run_study as run_hep_study
from examples.lloyd_nonmonotone import (
    COUNTEREXAMPLE_BINS,
    COUNTEREXAMPLE_LABELS,
    COUNTEREXAMPLE_SCORES,
    counterexample_study,
    unguarded_trajectory,
)
from examples.michelson_phase import (
    U_MAX,
    absorb_narrow_runs,
    build_provider,
    build_train_sample,
    closed_form_information,
    periodic_runs,
    run_study,
)
from examples.michelson_phase import (
    unbinned_profiled_information as michelson_unbinned_profiled_information,
)
from examples.nuisance_profiled_ds import (
    build_problem,
    finite_partitions,
    interval_study,
    partition_agreement,
    unbinned_profiled_information,
)
from examples.soft_purification import (
    center_separation,
    fractional_retention,
    hard_retention,
    softmax_responsibilities,
)
from examples.solver_shootout import (
    compare_methods,
    retention,
    scale_sensitivity,
    whitening_probe,
)
from examples.synthetic_problems import two_parameter_gaussian_mixture
from tests._fit import fit_test_quantizer

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "docs" / "examples" / "assets"
SHOOTOUT_METRICS = ASSETS / "solver_shootout.json"
NUISANCE_METRICS = ASSETS / "nuisance-profiled-ds.json"
SOFT_METRICS = ASSETS / "soft-purification.json"
LLOYD_METRICS = ASSETS / "lloyd-nonmonotone.json"
DS_GEOMETRY_METRICS = ASSETS / "ds-geometry-counterexample.json"
CERTIFICATION_METRICS = ASSETS / "global-certification.json"
MICHELSON_METRICS = ASSETS / "michelson-phase.json"
HEP_METRICS = ASSETS / "hep-classifier.json"
DOOR3_METRICS = ASSETS / "door3-classifier.json"

# Reduced sizes for the bound checks below: the committed JSON carries the
# full study, and these re-runs only have to reproduce its qualitative claims.
FAST_SIZES = (800, 400, 2_000)


def test_rank_deficient_fixture_projects_duplicate_direction() -> None:
    coordinate = np.linspace(-2, 2, 600)
    scores = np.column_stack([coordinate, 2 * coordinate])
    result = fit_test_quantizer(
        scores, n_bins=4, config=sq.KMeansConfig(seed=31, solver_restarts=3)
    )
    assert result.transform.rank == 1
    assert result.evaluate_scores(scores).geometric_mean_retention >= 0.90


def test_rare_population_fixture_retains_nonempty_hard_bins() -> None:
    rng = np.random.default_rng(32)
    common = rng.normal(0, 0.35, size=(1_900, 2))
    rare = rng.normal([3.0, -2.0], 0.12, size=(100, 2))
    scores = np.vstack([common, rare])
    result = fit_test_quantizer(
        scores, n_bins=6, config=sq.KMeansConfig(seed=32, solver_restarts=4)
    )
    assert np.all(np.asarray(result.train_report.bin_counts) > 0)
    assert result.train_report.geometric_mean_retention >= 0.70


def test_skewed_and_zero_weight_fixture_remains_finite() -> None:
    rng = np.random.default_rng(33)
    scores = rng.normal(size=(1_000, 3))
    weights = rng.lognormal(mean=0, sigma=2, size=len(scores))
    weights[::11] = 0
    result = fit_test_quantizer(
        scores,
        weights=weights,
        n_bins=8,
        config=sq.KMeansConfig(seed=33, solver_restarts=4),
    )
    assert np.isfinite(np.asarray(result.centers)).all()
    assert np.isfinite(result.train_report.geometric_mean_retention)


@pytest.mark.parametrize("shift", [0.0, 0.4, 0.8])
def test_controlled_train_test_shift_is_reported_not_optimized(shift: float) -> None:
    rng = np.random.default_rng(34)
    train = rng.normal(size=(1_200, 2))
    test = rng.normal(loc=[shift, -shift / 2], size=(2_000, 2))
    config = sq.KMeansConfig(seed=34, solver_restarts=3)
    without_validation = fit_test_quantizer(train, n_bins=6, config=config)
    with_validation = fit_test_quantizer(
        train,
        n_bins=6,
        config=config,
        validation_scores=test,
    )
    np.testing.assert_allclose(without_validation.centers, with_validation.centers)
    assert with_validation.validation_report is not None
    assert np.isfinite(with_validation.validation_report.geometric_mean_retention)


def _shootout_metrics() -> dict[str, object]:
    with SHOOTOUT_METRICS.open(encoding="utf-8") as stream:
        loaded = json.load(stream)
    assert isinstance(loaded, dict)
    return loaded


def _shootout_methods() -> dict[str, dict[str, float | str | None]]:
    methods = _shootout_metrics()["methods"]
    assert isinstance(methods, list)
    return {str(row["key"]): row for row in methods}


# Every retention number the solver-shootout page prints in its method table,
# transcribed from the committed JSON at the precision the page shows.
PAGE_RETENTION: dict[str, tuple[float, float | None]] = {
    "partition_d_exchange": (0.99916, None),
    "partition_mahalanobis_lloyd": (0.99916, None),
    "quantizer_d_exchange": (0.99916, 0.99910),
    "quantizer_mahalanobis_lloyd": (0.99916, 0.99910),
    "quantizer_whitened_kmeans": (0.99916, 0.99910),
    "quantizer_soft_voronoi": (0.99916, 0.99910),
    "quantizer_scalar_dp": (0.99917, 0.99911),
    "baseline_rectangular_observation_bins": (0.94226, 0.93762),
    "baseline_euclidean_kmeans_scores": (0.99906, 0.99897),
    "baseline_equal_frequency_1d": (0.99793, 0.99788),
}

# The relative-cost column of the same page: the published value and the
# number of decimals the page shows it to.
PAGE_COST_RATIO: dict[str, tuple[float, int]] = {
    "partition_d_exchange": (1.00, 2),
    "partition_mahalanobis_lloyd": (1.01, 2),
    "quantizer_d_exchange": (1.01, 2),
    "quantizer_mahalanobis_lloyd": (1.01, 2),
    "quantizer_whitened_kmeans": (1.03, 2),
    "quantizer_soft_voronoi": (2.21, 2),
    "quantizer_scalar_dp": (2.73, 2),
    "baseline_rectangular_observation_bins": (0.001, 3),
    "baseline_euclidean_kmeans_scores": (0.29, 2),
    "baseline_equal_frequency_1d": (0.001, 3),
}

# The bin-budget sweep table: budget, score space, rectangular grid, gap.
PAGE_BUDGET_SWEEP = [
    (4, 0.98360, 0.89767, 0.08594),
    (9, 0.99708, 0.69843, 0.29865),
    (16, 0.99910, 0.93762, 0.06148),
    (25, 0.99961, 0.90734, 0.09227),
]

# The score-rescaling table: multiplier, whitened fit, Euclidean k-means.
PAGE_SCALE_PROBE = [
    (1.0, 0.98581, 0.98029),
    (5.0, 0.98581, 0.94995),
    (25.0, 0.98581, 0.40206),
    (100.0, 0.98581, 0.39556),
]


def test_shootout_json_matches_the_full_study_scale() -> None:
    # The scale the "Analysis" section states in prose: 16 bins, 4000 training
    # and 15000 held-out events.
    metrics = _shootout_metrics()
    assert metrics["n_bins"] == 16
    assert metrics["n_train"] == 4_000
    assert metrics["n_test"] == 15_000


def test_shootout_json_matches_the_published_method_table() -> None:
    methods = _shootout_methods()
    assert set(methods) == set(PAGE_RETENTION)
    for key, (train, test) in PAGE_RETENTION.items():
        row = methods[key]
        assert round(float(row["train_retention"]), 5) == pytest.approx(train)  # type: ignore[arg-type]
        if test is None:
            assert row["test_retention"] is None
        else:
            assert round(float(row["test_retention"]), 5) == pytest.approx(test)  # type: ignore[arg-type]


def test_shootout_json_matches_the_published_cost_table() -> None:
    metrics = _shootout_metrics()
    methods = _shootout_methods()
    # The methodology the page states must be the one the study ran.
    assert metrics["timing_repeats"] == 5
    assert "warm-up" in str(metrics["timing_note"])
    assert isinstance(metrics["machine"], dict)
    for key, (ratio, decimals) in PAGE_COST_RATIO.items():
        measured = float(methods[key]["seconds_ratio"])  # type: ignore[arg-type]
        assert round(measured, decimals) == pytest.approx(ratio)
        assert float(methods[key]["seconds"]) > 0.0  # type: ignore[arg-type]
    assert float(metrics["fastest_information_aware_seconds"]) > 0.0  # type: ignore[arg-type]


def test_shootout_json_matches_the_published_sweep_and_scale_tables() -> None:
    metrics = _shootout_metrics()
    sweep = metrics["budget_sweep"]
    assert isinstance(sweep, list)
    assert len(sweep) == len(PAGE_BUDGET_SWEEP)
    for row, (n_bins, score_space, grid, gap) in zip(sweep, PAGE_BUDGET_SWEEP, strict=True):
        assert int(row["n_bins"]) == n_bins
        assert round(float(row["score_space"]), 5) == pytest.approx(score_space)
        assert round(float(row["observation_space"]), 5) == pytest.approx(grid)
        assert round(float(row["gap"]), 5) == pytest.approx(gap)

    probe = metrics["scale_probe"]
    assert isinstance(probe, dict)
    entries = probe["entries"]
    assert isinstance(entries, list)
    for row, (scale, whitened, euclidean) in zip(entries, PAGE_SCALE_PROBE, strict=True):
        assert float(row["scale"]) == pytest.approx(scale)
        assert round(float(row["whitened"]), 5) == pytest.approx(whitened)
        assert round(float(row["euclidean"]), 5) == pytest.approx(euclidean)


def test_shootout_headline_gaps_hold_in_the_committed_study() -> None:
    metrics = _shootout_metrics()
    methods = _shootout_methods()
    held_out = [
        float(row["test_retention"])  # type: ignore[arg-type]
        for row in methods.values()
        if row["family"] == "information_aware" and row["test_retention"] is not None
    ]
    grid = float(methods["baseline_rectangular_observation_bins"]["test_retention"])  # type: ignore[arg-type]

    # Claim 1: score space beats the observation-space grid, by 6.1 points at
    # the headline budget and by at least 6 points at every swept budget.
    assert max(held_out) - grid == pytest.approx(0.0615, abs=5e-4)
    assert (1.0 - grid) / (1.0 - max(held_out)) == pytest.approx(70.0, abs=1.0)
    sweep = metrics["budget_sweep"]
    assert isinstance(sweep, list)
    assert min(float(row["gap"]) for row in sweep) > 0.06
    assert max(float(row["gap"]) for row in sweep) > 0.29

    # Claim 3: the information-aware solvers agree far more closely than they
    # differ in cost. "within 0.000011 ... within 0.0000051 on training".
    assert max(held_out) - min(held_out) == pytest.approx(0.000011, abs=2e-6)
    train = [
        float(row["train_retention"])  # type: ignore[arg-type]
        for row in methods.values()
        if row["family"] == "information_aware"
    ]
    assert max(train) - min(train) == pytest.approx(0.0000051, abs=1e-6)

    # Claim 2's "loses only 0.00012" and "lost 0.0012" gaps.
    whitened_test = float(methods["quantizer_whitened_kmeans"]["test_retention"])  # type: ignore[arg-type]
    euclidean_test = float(methods["baseline_euclidean_kmeans_scores"]["test_retention"])  # type: ignore[arg-type]
    assert whitened_test - euclidean_test == pytest.approx(0.00012, abs=2e-5)
    equal_frequency_test = float(methods["baseline_equal_frequency_1d"]["test_retention"])  # type: ignore[arg-type]
    assert max(held_out) - equal_frequency_test == pytest.approx(0.0012, abs=2e-4)

    ratios = [
        float(row["seconds_ratio"])  # type: ignore[arg-type]
        for row in methods.values()
        if row["family"] == "information_aware"
    ]
    assert max(ratios) > 2.0


def test_shootout_whitening_is_inert_on_a_line_and_decisive_off_it() -> None:
    metrics = _shootout_metrics()
    probe = metrics["whitening_probe"]
    assert isinstance(probe, dict)
    # Claim 2, first half: on a two-parameter component score the whole cloud
    # lies on a line, so whitening cannot change the k-means partition. The
    # page states the full-study gap as "0.0000082".
    assert abs(float(probe["whitened"]) - float(probe["unwhitened"])) < 1e-4
    assert abs(float(probe["whitened"]) - float(probe["unwhitened"])) == pytest.approx(
        0.0000082, abs=1e-6
    )

    scale_probe = metrics["scale_probe"]
    assert isinstance(scale_probe, dict)
    entries = scale_probe["entries"]
    assert isinstance(entries, list)
    whitened = [float(row["whitened"]) for row in entries]
    euclidean = [float(row["euclidean"]) for row in entries]
    # Claim 2, second half: off the line, D-efficiency is invariant under a
    # score reparameterization and only the whitened fit reflects that. The
    # page states "invariant to twelve decimal places" and "lost 58
    # D-efficiency points" at the widest rescaling.
    assert max(whitened) - min(whitened) < 1e-9
    assert max(whitened) - min(whitened) < 5e-12
    assert euclidean[0] > 0.97
    assert min(euclidean) < 0.5
    assert euclidean[0] - min(euclidean) > 0.5
    assert euclidean[0] - min(euclidean) == pytest.approx(0.5847, abs=2e-3)


def test_fast_rerun_reproduces_the_score_space_gap_and_solver_agreement() -> None:
    problem = two_parameter_gaussian_mixture(n_bins=16, sizes=FAST_SIZES)
    methods = compare_methods(problem, soft_steps=60, timing_repeats=1)
    by_key = {entry.key: entry for entry in methods}

    held_out = [
        entry.test_retention
        for entry in methods
        if entry.family == "information_aware" and entry.test_retention is not None
    ]
    grid = by_key["baseline_rectangular_observation_bins"].test_retention
    assert grid is not None
    assert max(held_out) - grid > 0.03
    assert max(held_out) - min(held_out) < 2e-4
    assert min(held_out) > 0.99

    equal_frequency = by_key["baseline_equal_frequency_1d"].test_retention
    assert equal_frequency is not None
    assert max(held_out) - equal_frequency > 5e-4

    whitening = whitening_probe(problem)
    assert abs(whitening["whitened"] - whitening["unwhitened"]) < 1e-3


def test_fast_rerun_reproduces_the_score_rescaling_collapse() -> None:
    rows = scale_sensitivity(scales=(1.0, 25.0), n_bins=8, sizes=FAST_SIZES)
    assert rows[0]["whitened"] == pytest.approx(rows[1]["whitened"], abs=1e-9)
    assert rows[0]["euclidean"] > 0.95
    assert rows[1]["euclidean"] < 0.6


def test_rectangular_grid_is_not_monotone_in_the_bin_budget() -> None:
    """The page claims adding grid cells can lose information; measure it."""
    values = []
    for n_bins in (4, 9):
        problem = two_parameter_gaussian_mixture(n_bins=n_bins, sizes=FAST_SIZES)
        labels = rectangular_observation_bins(
            problem.test.observations, total_budget=problem.n_bins
        )
        values.append(
            retention(problem.test.scores, labels, problem.test.weights, int(labels.max()) + 1)
        )
    assert values[1] < values[0]


def _load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        loaded = json.load(stream)
    assert isinstance(loaded, dict)
    return loaded


def _listing(metrics: dict[str, object], key: str) -> list[dict[str, object]]:
    value = metrics[key]
    assert isinstance(value, list)
    return [row for row in value if isinstance(row, dict)]


def _mapping(metrics: dict[str, object], key: str) -> dict[str, object]:
    value = metrics[key]
    assert isinstance(value, dict)
    return value


# --- docs/examples/nuisance-profiled-ds.md ---------------------------------

# The retention table: full-matrix and profiled retention of each labeling,
# at the precision the page prints.
PAGE_NUISANCE_PARTITIONS: dict[str, tuple[float, float]] = {
    "d_partition": (0.94952, 0.94855),
    "ds_partition_seeded": (0.90115, 0.97903),
    "ds_partition_initialized": (0.90115, 0.97903),
}

# The held-out rows of the same table, plus the training number the prose
# quotes for the soft profiled rule.
PAGE_NUISANCE_RULES: dict[str, tuple[float, float, float]] = {
    "d_rule": (0.95158, 0.95005, 0.94855),
    "ds_rule": (0.93238, 0.97358, 0.97255),
}

# The ceiling sweep: bins, plain D, D_s from generic seeding, D_s from the
# ceiling's labels, and the certified ceiling itself.
PAGE_NUISANCE_SWEEP = [
    (3, 0.92454, 0.95294, 0.95294, 0.95364),
    (4, 0.94855, 0.97903, 0.97903, 0.97972),
    (5, 0.97517, 0.98504, 0.98620, 0.98706),
    (6, 0.98047, 0.98591, 0.99146, 0.99153),
    (8, 0.98940, 0.99322, 0.99509, 0.99518),
]

# The initializer table: bins, both certified gaps, both relocation counts.
PAGE_NUISANCE_INITIALIZER = [
    (3, 0.000735, 0.000735, 315, 5),
    (4, 0.000710, 0.000710, 1715, 17),
    (5, 0.002041, 0.000867, 672, 23),
    (6, 0.005682, 0.000071, 631, 24),
    (8, 0.001971, 0.000096, 870, 51),
]

# The downstream interval table: half-widths in signal-fraction units.
PAGE_NUISANCE_INTERVALS = {
    "unbinned": 0.011258,
    "d_partition": 0.011558,
    "ds_partition_initialized": 0.011376,
}


def test_nuisance_json_matches_the_published_retention_table() -> None:
    metrics = _load(NUISANCE_METRICS)
    assert metrics["n_bins"] == 4
    assert metrics["interest"] == [0]
    assert metrics["nuisance"] == [1, 2]
    # "at four bins on 4000 training and 15000 held-out events".
    assert metrics["n_train"] == 4_000
    assert metrics["n_test"] == 15_000

    partitions = {str(row["key"]): row for row in _listing(metrics, "partitions")}
    assert set(partitions) == set(PAGE_NUISANCE_PARTITIONS)
    for key, (full, profiled) in PAGE_NUISANCE_PARTITIONS.items():
        assert round(float(partitions[key]["full_retention"]), 5) == pytest.approx(full)  # type: ignore[arg-type]
        assert round(float(partitions[key]["profiled_retention"]), 5) == pytest.approx(profiled)  # type: ignore[arg-type]

    rules = {str(row["key"]): row for row in _listing(metrics, "rules")}
    assert set(rules) == set(PAGE_NUISANCE_RULES)
    for key, (full, profiled, train_profiled) in PAGE_NUISANCE_RULES.items():
        row = rules[key]
        assert round(float(row["test_full_retention"]), 5) == pytest.approx(full)  # type: ignore[arg-type]
        assert round(float(row["test_profiled_retention"]), 5) == pytest.approx(profiled)  # type: ignore[arg-type]
        assert round(float(row["train_profiled_retention"]), 5) == pytest.approx(train_profiled)  # type: ignore[arg-type]


def test_nuisance_json_matches_the_published_criterion_trade() -> None:
    """Each criterion wins on its own objective, by the amounts the page states."""
    partitions = {str(row["key"]): row for row in _listing(_load(NUISANCE_METRICS), "partitions")}
    plain = partitions["d_partition"]
    profiled = partitions["ds_partition_initialized"]
    given_up = float(plain["full_retention"]) - float(profiled["full_retention"])  # type: ignore[arg-type]
    gained = float(profiled["profiled_retention"]) - float(plain["profiled_retention"])  # type: ignore[arg-type]
    assert given_up == pytest.approx(0.0484, abs=5e-4)
    assert gained == pytest.approx(0.0305, abs=5e-4)

    agreement = _mapping(_load(NUISANCE_METRICS), "agreement")
    assert round(float(agreement["adjusted_rand_index"]), 3) == pytest.approx(0.629)  # type: ignore[arg-type]
    assert int(agreement["d_interval_runs"]) == 5  # type: ignore[call-overload]
    assert int(agreement["ds_interval_runs"]) == 6  # type: ignore[call-overload]


def test_nuisance_json_matches_the_published_ceiling_sweep() -> None:
    sweep = _listing(_load(NUISANCE_METRICS), "ceiling_sweep")
    assert len(sweep) == len(PAGE_NUISANCE_SWEEP)
    for row, published in zip(sweep, PAGE_NUISANCE_SWEEP, strict=True):
        n_bins, plain, seeded, initialized, ceiling = published
        assert int(float(row["n_bins"])) == n_bins  # type: ignore[arg-type]
        assert round(float(row["d_profiled_retention"]), 5) == pytest.approx(plain)  # type: ignore[arg-type]
        assert round(float(row["ds_seeded_retention"]), 5) == pytest.approx(seeded)  # type: ignore[arg-type]
        assert round(float(row["ds_initialized_retention"]), 5) == pytest.approx(initialized)  # type: ignore[arg-type]
        assert round(float(row["ceiling_retention"]), 5) == pytest.approx(ceiling)  # type: ignore[arg-type]
        # The ceiling is a ceiling: no labeling at this budget may exceed it.
        assert float(row["ds_initialized_retention"]) <= float(row["ceiling_retention"]) + 1e-9  # type: ignore[arg-type]
        assert float(row["d_profiled_retention"]) <= float(row["ceiling_retention"]) + 1e-9  # type: ignore[arg-type]


def test_nuisance_json_matches_the_published_initializer_table() -> None:
    sweep = _listing(_load(NUISANCE_METRICS), "ceiling_sweep")
    for row, published in zip(sweep, PAGE_NUISANCE_INITIALIZER, strict=True):
        n_bins, seeded_gap, initialized_gap, seeded_moves, initialized_moves = published
        assert int(float(row["n_bins"])) == n_bins  # type: ignore[arg-type]
        assert round(float(row["seeded_gap"]), 6) == pytest.approx(seeded_gap)  # type: ignore[arg-type]
        assert round(float(row["initialized_gap"]), 6) == pytest.approx(initialized_gap)  # type: ignore[arg-type]
        assert int(float(row["seeded_moves"])) == seeded_moves  # type: ignore[arg-type]
        assert int(float(row["initialized_moves"])) == initialized_moves  # type: ignore[arg-type]
        # The initializer is never worse on either axis.
        assert float(row["initialized_gap"]) <= float(row["seeded_gap"]) + 1e-12  # type: ignore[arg-type]
        assert float(row["initialized_moves"]) < float(row["seeded_moves"])  # type: ignore[arg-type]

    headline = _mapping(_load(NUISANCE_METRICS), "bound")
    assert int(headline["seeded_scans"]) == 26  # type: ignore[call-overload]
    assert int(headline["initialized_scans"]) == 5  # type: ignore[call-overload]
    assert int(headline["seeded_moves"]) == 1715  # type: ignore[call-overload]
    assert int(headline["initialized_moves"]) == 17  # type: ignore[call-overload]
    # "within 0.0007 nat", which is 0.07 percent in ratio terms.
    gap = float(headline["initialized_gap"])  # type: ignore[arg-type]
    assert gap == pytest.approx(0.00071, abs=5e-6)
    assert 100.0 * (1.0 - np.exp(-gap)) == pytest.approx(0.071, abs=5e-3)
    # "at six bins it closes the certified gap by a factor of 80".
    six = next(row for row in sweep if int(float(row["n_bins"])) == 6)  # type: ignore[arg-type]
    assert float(six["seeded_gap"]) / float(six["initialized_gap"]) == pytest.approx(80.0, abs=2.0)  # type: ignore[arg-type]


def test_nuisance_json_matches_the_published_interval_table() -> None:
    intervals = _mapping(_load(NUISANCE_METRICS), "intervals")
    unbinned = float(intervals["unbinned_half_width"])  # type: ignore[arg-type]
    assert round(unbinned, 6) == pytest.approx(PAGE_NUISANCE_INTERVALS["unbinned"])

    rows = {str(row["key"]): row for row in _listing(intervals, "rows")}
    excess = {}
    for key in ("d_partition", "ds_partition_initialized"):
        half_width = float(rows[key]["half_width"])  # type: ignore[arg-type]
        assert round(half_width, 6) == pytest.approx(PAGE_NUISANCE_INTERVALS[key])
        # The scan is a check on the library, not an illustration beside it.
        fisher = float(rows[key]["fisher_half_width"])  # type: ignore[arg-type]
        assert half_width == pytest.approx(fisher, rel=1e-3)
        excess[key] = 100.0 * (half_width / unbinned - 1.0)

    assert excess["d_partition"] == pytest.approx(2.67, abs=5e-3)
    assert excess["ds_partition_initialized"] == pytest.approx(1.04, abs=5e-3)
    narrowing = 1.0 - (
        float(rows["ds_partition_initialized"]["half_width"])  # type: ignore[arg-type]
        / float(rows["d_partition"]["half_width"])  # type: ignore[arg-type]
    )
    assert 100.0 * narrowing == pytest.approx(1.6, abs=0.05)
    # "about three fifths of the price of binning".
    assert 1.0 - excess["ds_partition_initialized"] / excess["d_partition"] == pytest.approx(
        0.61, abs=0.02
    )

    # "at eight bins the same two profiled retentions imply a narrowing of 0.29%".
    sweep = _listing(_load(NUISANCE_METRICS), "ceiling_sweep")
    eight = next(row for row in sweep if int(float(row["n_bins"])) == 8)  # type: ignore[arg-type]
    narrowing_at_eight = 1.0 - np.sqrt(
        float(eight["d_profiled_retention"]) / float(eight["ds_initialized_retention"])  # type: ignore[arg-type]
    )
    assert 100.0 * narrowing_at_eight == pytest.approx(0.29, abs=0.01)


def test_fast_rerun_reproduces_the_criterion_crossover_and_the_certified_ceiling() -> None:
    problem = build_problem(n_bins=4, sizes=FAST_SIZES)
    train = problem.train
    bound = sq.efficient_score_bound(
        train.scores, interest=problem.interest, weights=train.weights, n_bins=4
    )
    partitions = finite_partitions(problem, n_bins=4, bound=bound)
    rows = {row.key: row for row in partitions.rows}

    # Each criterion wins on its own objective and loses on the other one.
    plain = rows["d_partition"]
    profiled = rows["ds_partition_initialized"]
    assert plain.full_retention > profiled.full_retention
    assert profiled.profiled_retention > plain.profiled_retention

    # The certified ceiling dominates every labeling of this budget, whatever
    # criterion produced it.
    reference = unbinned_profiled_information(
        train.scores, train.weights, interest=problem.interest
    )
    ceiling = float(np.exp(bound.upper_bound - np.log(reference)))
    assert plain.profiled_retention <= ceiling + 1e-9
    assert profiled.profiled_retention <= ceiling + 1e-9

    # The initializer is never worse and always cheaper.
    seeded = rows["ds_partition_seeded"]
    assert profiled.objective >= seeded.objective - 1e-12
    assert profiled.accepted_moves < seeded.accepted_moves

    # The two partitions are genuinely different objects.
    assert 0.2 < partition_agreement(partitions.d_labels, partitions.warm_labels, 4) < 0.95


def test_fast_rerun_reproduces_the_narrower_profiled_interval() -> None:
    problem = build_problem(n_bins=4, sizes=FAST_SIZES)
    train = problem.train
    bound = sq.efficient_score_bound(
        train.scores, interest=problem.interest, weights=train.weights, n_bins=4
    )
    partitions = finite_partitions(problem, n_bins=4, bound=bound)
    study = interval_study(
        problem,
        {
            "d_partition": ("Plain D", partitions.d_labels),
            "ds_partition_initialized": ("Profiled D_s", partitions.warm_labels),
        },
        n_bins=4,
    )
    rows = {row.key: row for row in study.rows}
    for row in study.rows:
        assert row.half_width == pytest.approx(row.fisher_half_width, rel=1e-3)
    assert rows["ds_partition_initialized"].half_width < rows["d_partition"].half_width


def test_profiled_partition_refuses_to_compile() -> None:
    """The page's honest caveat, as an executable statement."""
    problem = build_problem(n_bins=4, sizes=FAST_SIZES)
    train = problem.train
    profiled = sq.optimize_partition(
        train.scores,
        weights=train.weights,
        n_bins=4,
        criterion=sq.ProfiledDOptimality(problem.interest),
        config=sq.DExchangeConfig(seed=11),
    )
    with pytest.raises(sq.RefusalError, match="no canonical inductive compilation"):
        profiled.compile_quantizer()


# --- docs/examples/soft-purification.md ------------------------------------

# The annealing table: the randomized rule and the hard rule it implies, at the
# first and last recorded snapshot of the traced fit.
PAGE_SOFT_SCHEDULE = {
    "first_soft_retention": 0.70076,
    "final_soft_retention": 0.97552,
    "first_hard_retention": 0.97557,
    "final_hard_retention": 0.97552,
}

# The hardening-gap ladder, at the one significant digit the page prints.
PAGE_SOFT_HARDENING: dict[str, tuple[float, ...]] = {
    "gaussian_location": (-9.2e-02, -3.5e-02, -4.3e-03, -2.9e-05, -1.4e-08),
    "spectral_templates": (-1.0e-02, -1.8e-03, -1.6e-04, -6.2e-07, -4.2e-09),
    "two_parameter_gaussian_mixture": (-1.3e-02, -2.2e-03, -1.4e-04, -4.9e-07, -1.2e-12),
    "signal_background_shape": (-2.1e-02, -2.7e-03, -2.3e-04, -3.4e-06, 8.1e-15),
}
PAGE_SOFT_RATIOS = (0.8, 0.4, 0.2, 0.05, 0.01)

# The purification table, for the traced problem.
PAGE_SOFT_PURIFICATION = [
    (1.0, 0.33641, 0.97552, 0.639),
    (0.5, 0.86238, 0.97552, 0.113),
    (0.25, 0.96805, 0.97552, 0.00747),
    (0.1, 0.97535, 0.97552, 0.000174),
    (0.05, 0.97550, 0.97552, 0.0000204),
]

# The solver table: soft, exact exchange, exchange started from soft labels.
PAGE_SOFT_SOLVERS = {
    "gaussian_location": (0.8851049, 0.8851055, 0.8851055),
    "spectral_templates": (0.9968710, 0.9969338, 0.9969318),
    "two_parameter_gaussian_mixture": (0.9964000, 0.9964524, 0.9964008),
    "signal_background_shape": (0.9755215, 0.9755675, 0.9755675),
}


def test_soft_json_matches_the_published_annealing_table() -> None:
    metrics = _load(SOFT_METRICS)
    schedule = _mapping(metrics, "schedule")
    # "the signal-plus-backgrounds problem at six cells, 300 Adam steps,
    # cooling to one fiftieth of the starting temperature", on 4000 events.
    assert schedule["problem"] == "signal_background_shape"
    assert schedule["n_bins"] == 6
    assert metrics["n_train"] == 4_000
    assert metrics["schedule_steps"] == 300
    assert float(schedule["temperature_ratio"]) == pytest.approx(0.02)  # type: ignore[arg-type]
    for key, published in PAGE_SOFT_SCHEDULE.items():
        assert round(float(schedule[key]), 5) == pytest.approx(published)  # type: ignore[arg-type]

    # "the soft objective climbs 27 D-efficiency points".
    climb = float(schedule["final_soft_retention"]) - float(schedule["first_soft_retention"])  # type: ignore[arg-type]
    assert 100.0 * climb == pytest.approx(27.5, abs=0.5)
    # "the rule that will be deployed falls by 0.000046".
    change = float(schedule["final_hard_retention"]) - float(schedule["first_hard_retention"])  # type: ignore[arg-type]
    assert change == pytest.approx(-4.6e-5, abs=5e-7)


def test_soft_json_matches_the_published_hardening_ladder() -> None:
    rows = _listing(_load(SOFT_METRICS), "hardening")
    assert len(rows) == len(PAGE_SOFT_HARDENING) * len(PAGE_SOFT_RATIOS)
    for problem, published in PAGE_SOFT_HARDENING.items():
        for ratio, value in zip(PAGE_SOFT_RATIOS, published, strict=True):
            row = next(
                entry
                for entry in rows
                if entry["problem"] == problem
                and float(entry["temperature_ratio"]) == pytest.approx(ratio)  # type: ignore[arg-type]
            )
            gap = float(row["hardening_gap"])  # type: ignore[arg-type]
            assert float(f"{gap:.1e}") == pytest.approx(value)
    # "the gap closes by six or more orders of magnitude" over the ladder, and
    # is negative wherever it is above floating-point noise.
    for problem, published in PAGE_SOFT_HARDENING.items():
        assert abs(published[0]) / abs(published[-1]) > 1e6, problem
        assert all(value < 0.0 for value in published[:-1]), problem


def test_soft_json_matches_the_published_purification_table() -> None:
    rows = [
        row
        for row in _listing(_load(SOFT_METRICS), "purification")
        if row["problem"] == "signal_background_shape"
    ]
    assert len(rows) == len(PAGE_SOFT_PURIFICATION)
    for row, published in zip(rows, PAGE_SOFT_PURIFICATION, strict=True):
        ratio, randomized, purified, gain = published
        assert float(row["temperature_ratio"]) == pytest.approx(ratio)  # type: ignore[arg-type]
        assert round(float(row["randomized_retention"]), 5) == pytest.approx(randomized)  # type: ignore[arg-type]
        assert round(float(row["purified_retention"]), 5) == pytest.approx(purified)  # type: ignore[arg-type]
        assert float(f"{float(row['purification_gain']):.3g}") == pytest.approx(gain, rel=1e-2)  # type: ignore[arg-type]

    # "positive at every temperature and every problem, ranging from 0.64 down
    # to 1.2e-6", and "retains a third of the information".
    every = [
        float(row["purification_gain"]) for row in _listing(_load(SOFT_METRICS), "purification")
    ]  # type: ignore[arg-type]
    assert min(every) > 0.0
    assert max(every) == pytest.approx(0.639, abs=5e-3)
    assert min(every) == pytest.approx(1.2e-6, abs=1e-7)
    warmest = rows[0]
    ratio = float(warmest["randomized_retention"]) / float(warmest["purified_retention"])  # type: ignore[arg-type]
    assert ratio == pytest.approx(1 / 3, abs=0.03)


def test_soft_json_matches_the_published_solver_table() -> None:
    rows = {str(row["problem"]): row for row in _listing(_load(SOFT_METRICS), "solvers")}
    assert set(rows) == set(PAGE_SOFT_SOLVERS)
    deficits = []
    for problem, (soft, exchange, from_soft) in PAGE_SOFT_SOLVERS.items():
        row = rows[problem]
        assert round(float(row["soft_retention"]), 7) == pytest.approx(soft)  # type: ignore[arg-type]
        assert round(float(row["exchange_retention"]), 7) == pytest.approx(exchange)  # type: ignore[arg-type]
        assert round(float(row["exchange_from_soft_retention"]), 7) == pytest.approx(from_soft)  # type: ignore[arg-type]
        # "exact exchange wins on all four".
        assert float(row["exchange_retention"]) >= float(row["soft_retention"])  # type: ignore[arg-type]
        deficits.append(1e6 * (float(row["exchange_retention"]) - float(row["soft_retention"])))  # type: ignore[arg-type]
    # "by between 0.6 and 63 parts per million".
    assert min(deficits) == pytest.approx(0.6, abs=0.1)
    assert max(deficits) == pytest.approx(63.0, abs=1.0)


def test_fractional_retention_agrees_with_the_public_hard_report() -> None:
    """The soft page's grounding check, on a table the page never sees."""
    rng = np.random.default_rng(77)
    scores = rng.normal(size=(500, 3)) @ np.array(
        [[1.0, 0.4, -0.2], [0.0, 1.2, 0.3], [0.0, 0.0, 0.8]]
    )
    weights = rng.uniform(0.3, 1.7, size=500)
    labels = rng.integers(0, 5, size=500)
    one_hot = np.eye(5)[labels]
    assert fractional_retention(scores, one_hot, weights) == pytest.approx(
        hard_retention(scores, labels, weights, 5), abs=1e-12
    )


def test_fast_rerun_reproduces_the_hardening_and_purification_signs() -> None:
    problem = build_problem(n_bins=6, sizes=FAST_SIZES)
    train = problem.train
    source = sq.ScoreSample(train.scores, train.weights)

    gaps = []
    for ratio in (0.5, 0.2, 0.05):
        rule = sq.fit_quantizer(
            source,
            n_bins=6,
            criterion=sq.DOptimality(),
            config=sq.SoftVoronoiConfig(
                seed=3,
                initializer_restarts=4,
                max_steps=80,
                record_every=80,
                temperature_end_ratio=ratio,
            ),
        )
        gaps.append(float(rule.hardening_gap or 0.0))
    assert all(gap < 0.0 for gap in gaps)
    assert all(abs(later) < abs(earlier) for earlier, later in zip(gaps, gaps[1:], strict=False))

    fitted = sq.fit_quantizer(
        source,
        n_bins=6,
        criterion=sq.DOptimality(),
        config=sq.SoftVoronoiConfig(seed=3, initializer_restarts=4, max_steps=80, record_every=80),
    )
    coordinates = np.asarray(fitted.transform.apply(train.scores))
    centers = np.asarray(fitted.centers)
    separation = center_separation(centers)
    gains = []
    for ratio in (1.0, 0.25):
        responsibilities = softmax_responsibilities(coordinates, centers, ratio * separation)
        gains.append(
            hard_retention(train.scores, np.argmax(responsibilities, axis=1), train.weights, 6)
            - fractional_retention(train.scores, responsibilities, train.weights)
        )
    assert all(gain > 0.0 for gain in gains)
    assert gains[0] > gains[1]

    # The soft fit is a competitor, not a winner: exchange is at least as good.
    exchange = sq.optimize_partition(
        train.scores, weights=train.weights, n_bins=6, config=sq.DExchangeConfig(seed=3)
    )
    assert (
        exchange.train_report.geometric_mean_retention
        >= float(fitted.train_report.geometric_mean_retention) - 1e-9
    )


# --- docs/examples/lloyd-nonmonotone.md ------------------------------------

# The failure ledger, exactly as the page tabulates it: how many of the 24
# unguarded runs of each configuration vacated a cell.
PAGE_LLOYD_LEDGER = {
    ("signal_background_shape", 4): (17, 11, 16),
    ("signal_background_shape", 6): (23, 22, 23),
    ("spectral_templates", 4): (3, 0, 0),
    ("spectral_templates", 6): (5, 5, 0),
    ("spatial_sources", 4): (2, 3, 0),
    ("spatial_sources", 6): (4, 4, 1),
}
PAGE_LLOYD_SIZES = (60, 250, 1_000)

# The scale table: full-data passes, accepted steps, and terminal objective of
# the guarded batch and of plain exchange from the same random start.
PAGE_LLOYD_CLIMB = {
    "n_rows": 4_000,
    "n_bins": 6,
    "lloyd_iterations": 26,
    "accepted_lloyd_steps": 25,
    "scans": 44,
    "accepted_moves": 43,
    "exchange_scans": 48,
    "exchange_moves": 3_781,
    "final_objective": -0.074225,
    "exchange_objective": -0.074208,
}


def test_lloyd_json_matches_the_published_failure_ledger() -> None:
    metrics = _load(LLOYD_METRICS)
    rows = {(str(row["problem"]), int(row["n_bins"])): row for row in _listing(metrics, "ledger")}
    assert set(rows) == set(PAGE_LLOYD_LEDGER)
    for (problem, n_bins), published in PAGE_LLOYD_LEDGER.items():
        for n_rows, emptied in zip(PAGE_LLOYD_SIZES, published, strict=True):
            row = next(
                entry
                for entry in _listing(metrics, "ledger")
                if entry["problem"] == problem
                and int(entry["n_bins"]) == n_bins  # type: ignore[call-overload]
                and int(entry["n_rows"]) == n_rows  # type: ignore[call-overload]
            )
            assert int(row["emptied_runs"]) == emptied  # type: ignore[call-overload]
            assert int(row["runs"]) == 24  # type: ignore[call-overload]

    totals = _mapping(metrics, "ledger_totals")
    # "139 of 432 did", and "not one ever stepped downhill".
    assert int(totals["runs"]) == 432  # type: ignore[call-overload]
    assert int(totals["emptied_runs"]) == 139  # type: ignore[call-overload]
    assert int(totals["downhill_runs"]) == 0  # type: ignore[call-overload]
    assert float(totals["worst_step"]) == 0.0  # type: ignore[arg-type]


def test_lloyd_json_matches_the_published_counterexample_and_scale_table() -> None:
    metrics = _load(LLOYD_METRICS)
    case = _mapping(metrics, "counterexample")
    # Every number the page's counterexample section prints.
    assert round(float(case["before"]), 6) == pytest.approx(-3.810643)  # type: ignore[arg-type]
    assert round(float(case["after"]), 6) == pytest.approx(-3.947164)  # type: ignore[arg-type]
    assert round(float(case["step"]), 6) == pytest.approx(-0.136521)  # type: ignore[arg-type]
    assert round(float(case["distortion_before"]), 4) == pytest.approx(13.5450)  # type: ignore[arg-type]
    assert round(float(case["distortion_after"]), 4) == pytest.approx(9.7464)  # type: ignore[arg-type]
    assert round(float(case["tangent_change"]), 4) == pytest.approx(8.2274)  # type: ignore[arg-type]
    assert round(float(case["whitening_offset"]), 6) == pytest.approx(-0.783062)  # type: ignore[arg-type]
    assert int(case["moved"]) == 4  # type: ignore[call-overload]
    assert int(case["rejected_iterations"]) == 1  # type: ignore[call-overload]
    assert int(case["rejected_accepted"]) == 0  # type: ignore[call-overload]
    assert case["rejected_stable"] is False
    assert int(case["rescued_moves"]) == 4  # type: ignore[call-overload]

    # The unguarded trajectory table: it dips once, then climbs to a fixed point
    # that is exactly where the guarded solver ends up.
    trajectory = [round(float(value), 6) for value in case["unguarded"]]  # type: ignore[union-attr]
    assert trajectory == [-3.810643, -3.947164, -1.366245, -1.035251, -1.035251]
    assert case["unguarded_outcome"] == "fixed"
    assert round(float(case["rescued_objective"]), 6) == pytest.approx(-1.035251)  # type: ignore[arg-type]

    climb = _mapping(metrics, "climb")
    for key in ("n_rows", "n_bins", "lloyd_iterations", "accepted_lloyd_steps", "scans"):
        assert int(climb[key]) == PAGE_LLOYD_CLIMB[key]  # type: ignore[call-overload]
    assert int(climb["accepted_moves"]) == PAGE_LLOYD_CLIMB["accepted_moves"]  # type: ignore[call-overload]
    assert int(climb["exchange_scans"]) == PAGE_LLOYD_CLIMB["exchange_scans"]  # type: ignore[call-overload]
    assert int(climb["exchange_moves"]) == PAGE_LLOYD_CLIMB["exchange_moves"]  # type: ignore[call-overload]
    assert round(float(climb["final_objective"]), 6) == pytest.approx(  # type: ignore[arg-type]
        PAGE_LLOYD_CLIMB["final_objective"]
    )
    assert round(float(climb["exchange_objective"]), 6) == pytest.approx(  # type: ignore[arg-type]
        PAGE_LLOYD_CLIMB["exchange_objective"]
    )
    assert climb["monotone"] is True
    # "relocating nearly ninety times as many rows".
    ratio = float(climb["exchange_moves"]) / float(climb["accepted_moves"])  # type: ignore[arg-type]
    assert ratio == pytest.approx(88.0, abs=2.0)


def test_fast_rerun_reproduces_the_counterexample_and_the_guard() -> None:
    """The page's core claim, recomputed rather than read from the JSON."""
    case = counterexample_study()
    assert case.step == pytest.approx(-0.136521, abs=2e-6)
    assert case.distortion_after < case.distortion_before
    # Concavity: the tangent is an upper bound, so it can rise while the
    # criterion falls, and it must never fall below the criterion's change.
    assert case.tangent_change > 0.0
    assert case.step <= case.tangent_change

    weights = np.full(COUNTEREXAMPLE_SCORES.shape[0], 1.0 / COUNTEREXAMPLE_SCORES.shape[0])
    for guard in ("exchange", "reject"):
        result = sq.optimize_partition(
            COUNTEREXAMPLE_SCORES,
            weights=weights,
            n_bins=COUNTEREXAMPLE_BINS,
            config=sq.MahalanobisLloydConfig(seed=0, guard=guard),  # type: ignore[arg-type]
            initial_labels=COUNTEREXAMPLE_LABELS,
        )
        history = np.asarray(result.objective_history)
        assert np.all(np.diff(history) > 0)
        assert result.objective >= float(history[0])
        assert result.accepted_lloyd_steps == 0

    run = unguarded_trajectory(
        COUNTEREXAMPLE_SCORES, weights, COUNTEREXAMPLE_LABELS, n_bins=COUNTEREXAMPLE_BINS
    )
    assert run.went_downhill is True
    assert run.outcome == "fixed"


# --- docs/examples/ds-geometry-counterexample.md ---------------------------


def test_ds_geometry_json_matches_the_published_survey() -> None:
    metrics = _load(DS_GEOMETRY_METRICS)
    assert int(metrics["n_labelings"]) == 966  # type: ignore[call-overload]
    assert metrics["efficient_regression"] == "-1/60"

    profiled = _mapping(metrics, "profiled")
    assert profiled["optimum"] == [0, 1, 2, 1, 2, 0, 0, 2]
    assert profiled["optimum_value"] == "20449/1920"
    assert profiled["runner_up_margin"] == "2929/21120"
    assert profiled["optimum_is_consistent"] is False
    assert profiled["optimum_margins"][6] == "8/195"  # type: ignore[index]
    assert int(profiled["singular_labelings"]) == 2  # type: ignore[call-overload]
    # "exactly one labeling of 966 satisfies it, and that labeling is
    # fifth-best, retaining 91.83% of the profiled information".
    assert profiled["consistent_ranks"] == [5]
    assert round(float(profiled["best_consistent_ratio"]), 6) == pytest.approx(0.918327)  # type: ignore[arg-type]
    assert -np.log(float(profiled["best_consistent_ratio"])) == pytest.approx(  # type: ignore[arg-type]
        0.085201, abs=5e-6
    )

    determinant = _mapping(metrics, "determinant")
    assert determinant["optimum_value"] == "71289/1024"
    assert determinant["optimum_is_consistent"] is True
    assert determinant["consistent_ranks"] == [0, 55, 60, 63, 75]
    assert float(determinant["best_consistent_ratio"]) == pytest.approx(1.0)  # type: ignore[arg-type]


def test_ds_geometry_json_matches_what_the_library_reports() -> None:
    published = _mapping(_load(DS_GEOMETRY_METRICS), "library")
    measured = library_run()
    assert measured.profiled_labels == published["profiled_labels"]
    assert measured.violating_moves == 1
    assert measured.maximum_positive_violation == pytest.approx(8 / 195, abs=1e-12)
    assert measured.maximum_bound_residual <= 0.0
    assert measured.bound_certified is True
    assert measured.compile_refusal == (
        "finite profiled-D labels have no canonical inductive compilation; "
        "fit an explicit quantizer instead [CE-DS-GLOBAL-GEOMETRY-001]"
    )
    # The determinant contrast on the very same rows.
    assert measured.d_voronoi_consistent is True
    assert measured.d_violating_moves == 0
    assert measured.d_compiles is True
    assert measured.d_certificate_status == "optimal"
    assert measured.d_incumbent_was_optimal is True
    assert measured.d_nodes_explored == int(published["d_nodes_explored"])  # type: ignore[call-overload]


def test_exact_enumeration_reproduces_the_geometry_violation() -> None:
    """The exact claim, recomputed in rational arithmetic from scratch."""
    table = exact_table()
    labelings = canonical_labelings()
    assert len(labelings) == 966

    ranked = sorted(((profiled_value(labels, table), labels) for labels in labelings), reverse=True)
    best, optimum = ranked[0]
    metric = efficient_semimetric(optimum, table)
    assert metric is not None
    margins = violation_margins(optimum, table, metric)
    assert optimum == (0, 1, 2, 1, 2, 0, 0, 2)
    assert float(best) == pytest.approx(20449 / 1920)
    assert float(margins[6]) == pytest.approx(8 / 195)
    assert sum(margin > 0 for margin in margins) == 1


# --- docs/examples/global-certification.md ---------------------------------

# The hit-rate table: restarts, then the fraction of 64 trials reaching the
# certified optimum under each seeding, at the precision the page prints.
PAGE_CERT_HIT_RATES = [
    (1, 0.359, 0.141),
    (2, 0.609, 0.203),
    (3, 0.703, 0.297),
    (4, 0.828, 0.391),
    (6, 0.953, 0.516),
    (8, 0.969, 0.672),
    (12, 0.984, 0.812),
    (16, 0.984, 0.906),
]

# The "Seconds per fit, k-means++" column of the same table, at the precision
# the page prints.
PAGE_CERT_SECONDS_PER_FIT = [
    (1, 0.014),
    (2, 0.014),
    (3, 0.019),
    (4, 0.024),
    (6, 0.035),
    (8, 0.044),
    (12, 0.065),
    (16, 0.083),
]

# The certification cost table: atoms, then nodes explored at three, four, and
# five cells.
PAGE_CERT_SCALING = [
    (12, 151, 332, 414),
    (16, 330, 847, 1_152),
    (20, 948, 7_204, 8_361),
    (24, 3_051, 27_129, 36_813),
    (28, 5_117, 26_281, 51_292),
    (32, 37_471, 263_634, 925_202),
]


def test_certification_json_matches_the_published_incumbent_cases() -> None:
    cases = {
        str(row["key"]): row for row in _listing(_load(CERTIFICATION_METRICS), "incumbent_cases")
    }
    confirmed = cases["confirmed"]
    assert confirmed["status"] == "optimal"
    assert confirmed["incumbent_was_optimal"] is True
    assert float(confirmed["gain"]) == 0.0  # type: ignore[arg-type]
    assert int(confirmed["nodes_explored"]) == 8  # type: ignore[call-overload]

    improved = cases["improved"]
    assert improved["status"] == "optimal"
    assert improved["incumbent_was_optimal"] is False
    assert round(float(improved["gain"]), 6) == pytest.approx(0.046845)  # type: ignore[arg-type]
    assert int(improved["nodes_explored"]) == 67  # type: ignore[call-overload]
    assert float(improved["certified_objective"]) > float(improved["incumbent_objective"])  # type: ignore[arg-type]


def test_certification_json_matches_the_published_hit_rate_table() -> None:
    rates = _mapping(_load(CERTIFICATION_METRICS), "hit_rates")
    # "28 events, 5 cells, proved optimal in 51292 nodes".
    assert int(rates["n_rows"]) == 28  # type: ignore[call-overload]
    assert int(rates["n_bins"]) == 5  # type: ignore[call-overload]
    assert int(rates["certified_nodes"]) == 51_292  # type: ignore[call-overload]

    rows = {
        (str(row["init"]), int(row["solver_restarts"])): row  # type: ignore[call-overload]
        for row in _listing(rates, "rows")
    }
    for solver_restarts, seeded, random_init in PAGE_CERT_HIT_RATES:
        assert int(rows[("kmeans++", solver_restarts)]["trials"]) == 64  # type: ignore[call-overload]
        assert round(float(rows[("kmeans++", solver_restarts)]["hit_rate"]), 3) == pytest.approx(
            seeded
        )  # type: ignore[arg-type]
        assert round(float(rows[("random", solver_restarts)]["hit_rate"]), 3) == pytest.approx(  # type: ignore[arg-type]
            random_init
        )
        # Seeded restarts dominate random ones at every budget.
        assert float(rows[("kmeans++", solver_restarts)]["hit_rate"]) >= float(  # type: ignore[arg-type]
            rows[("random", solver_restarts)]["hit_rate"]  # type: ignore[arg-type]
        )

    # "Six restarts reach 95%, at 0.035 seconds per fit against 3.4 seconds to
    # prove the optimum -- about a hundredfold."
    six = float(rows[("kmeans++", 6)]["seconds_per_trial"])  # type: ignore[arg-type]
    assert round(six, 3) == pytest.approx(0.035, abs=5e-4)
    certified_seconds = float(rates["certified_seconds"])  # type: ignore[arg-type]
    assert round(certified_seconds, 1) == pytest.approx(3.4, abs=0.05)
    assert certified_seconds / six == pytest.approx(100.0, abs=5.0)

    # The full "Seconds per fit, k-means++" column.
    for solver_restarts, seconds in PAGE_CERT_SECONDS_PER_FIT:
        measured = float(rows[("kmeans++", solver_restarts)]["seconds_per_trial"])  # type: ignore[arg-type]
        assert round(measured, 3) == pytest.approx(seconds, abs=5e-4)

    # "0.55 seconds for sixteen random restarts against 0.083 for sixteen
    # seeded ones".
    assert round(float(rows[("random", 16)]["seconds_per_trial"]), 2) == pytest.approx(0.55)  # type: ignore[arg-type]
    assert round(float(rows[("kmeans++", 16)]["seconds_per_trial"]), 3) == pytest.approx(0.083)  # type: ignore[arg-type]

    # "the worst of them is 2.4% of the retained information, and the most
    # common miss is 0.08%".
    shortfalls = np.asarray(rates["single_restart_shortfalls"]["kmeans++"])  # type: ignore[index]
    misses = shortfalls[shortfalls > 1e-9]
    assert 100.0 * (1.0 - np.exp(-float(misses.max()))) == pytest.approx(2.4, abs=0.1)
    assert 100.0 * (1.0 - np.exp(-float(np.median(misses)))) == pytest.approx(0.08, abs=0.01)
    # "take only four distinct values, one per local optimum reached: two
    # within 0.0008 nat of the optimum, one at 0.013, and one at 0.024".
    levels = sorted({round(float(value), 5) for value in misses})
    assert levels == [0.00065, 0.00077, 0.0133, 0.02405]


def test_certification_json_matches_the_published_cost_table() -> None:
    metrics = _load(CERTIFICATION_METRICS)
    rows = {
        (int(row["n_bins"]), int(row["n_rows"])): row  # type: ignore[call-overload]
        for row in _listing(metrics, "scaling")
    }
    for n_rows, three, four, five in PAGE_CERT_SCALING:
        for n_bins, published in ((3, three), (4, four), (5, five)):
            row = rows[(n_bins, n_rows)]
            assert row["status"] == "optimal"
            assert float(row["gap"]) == 0.0  # type: ignore[arg-type]
            assert int(row["nodes_explored"]) == published  # type: ignore[call-overload]

    # "the tree grows by a factor of 250 at three cells, 790 at four, and 2200
    # at five -- roughly 1.3 to 1.5 times per event".
    for n_bins, factor in ((3, 250.0), (4, 790.0), (5, 2_200.0)):
        growth = float(rows[(n_bins, 32)]["nodes_explored"]) / float(  # type: ignore[arg-type]
            rows[(n_bins, 12)]["nodes_explored"]  # type: ignore[arg-type]
        )
        assert growth == pytest.approx(factor, rel=0.05)
        assert 1.3 <= growth ** (1 / 20) <= 1.5

    # "26281 nodes at 28 atoms and four cells against 27129 at 24".
    assert int(rows[(4, 28)]["nodes_explored"]) < int(rows[(4, 24)]["nodes_explored"])  # type: ignore[call-overload]

    overrun = _mapping(metrics, "overrun")
    assert int(overrun["n_rows"]) == 36  # type: ignore[call-overload]
    assert int(overrun["n_bins"]) == 4  # type: ignore[call-overload]
    assert overrun["status"] == "budget_exhausted"
    assert int(overrun["nodes_explored"]) == 200_001  # type: ignore[call-overload]
    assert round(float(overrun["gap"]), 3) == pytest.approx(0.030)  # type: ignore[arg-type]


def test_fast_rerun_reproduces_the_restart_hit_rate_trend() -> None:
    """Restarts help, seeded restarts help more, and the certificate bounds both."""
    study = restart_hit_rates(restarts=(1, 8), trials=12)
    rows = {(row.init, row.solver_restarts): row for row in study.rows}
    assert study.certified_objective < 0.0
    # Hit counts over 12 trials shift a little across platforms (BLAS/float
    # differences move k-means++ seeds between basins), so assert the robust
    # ordering and a clear-majority floor; the exact 64-trial numbers are
    # asserted from the committed JSON above.
    assert rows[("kmeans++", 1)].hit_rate <= rows[("kmeans++", 8)].hit_rate
    assert rows[("random", 1)].hit_rate <= rows[("kmeans++", 8)].hit_rate
    assert rows[("kmeans++", 8)].hit_rate >= 0.75
    for row in study.rows:
        assert 0.0 <= row.hit_rate <= 1.0
        assert row.median_shortfall >= 0.0


# --- docs/examples/michelson-phase.md ---------------------------------------

# The headline table: profiled phase information retained against the
# unbinned profiled ceiling, for the three labelings, plus the certified
# efficient-score bound gap of the profiled-D_s partition, at the precision
# the page prints.
PAGE_MICHELSON_SWEEP = [
    (4, 0.0000, 0.7227, 0.8629, 5.0e-03),
    (6, 0.2054, 0.7995, 0.9483, 2.5e-05),
    (8, 0.7247, 0.8806, 0.9714, 6.5e-05),
    (10, 0.5653, 0.9267, 0.9817, 1.2e-04),
]


def test_michelson_json_matches_the_published_headline_table() -> None:
    metrics = _load(MICHELSON_METRICS)
    assert metrics["headline_bins"] == 6
    assert metrics["n_nodes"] == 8_000
    closed_form = _mapping(metrics, "closed_form")
    assert round(float(closed_form["i_phiphi"]), 12) == pytest.approx(0.2, abs=1e-12)  # type: ignore[arg-type]
    assert round(float(closed_form["correlation"]), 3) == pytest.approx(0.872)  # type: ignore[arg-type]
    assert round(float(metrics["profiled_ceiling"]), 6) == pytest.approx(0.047938, abs=1e-6)  # type: ignore[arg-type]
    assert round(100.0 * float(metrics["cost_of_profiling"]), 1) == pytest.approx(76.0)  # type: ignore[arg-type]

    sweep = {int(row["n_bins"]): row for row in _listing(metrics, "sweep")}  # type: ignore[call-overload]
    assert set(sweep) == {row[0] for row in PAGE_MICHELSON_SWEEP}
    for n_bins, equal_width, d_optimal, profiled, gap in PAGE_MICHELSON_SWEEP:
        row = sweep[n_bins]
        assert round(float(row["equal_width_retention"]), 4) == pytest.approx(equal_width, abs=5e-4)  # type: ignore[arg-type]
        assert round(float(row["d_optimal_retention"]), 4) == pytest.approx(d_optimal, abs=5e-4)  # type: ignore[arg-type]
        assert round(float(row["profiled_retention"]), 4) == pytest.approx(profiled, abs=5e-4)  # type: ignore[arg-type]
        assert float(row["bound_gap"]) == pytest.approx(gap, rel=0.1)
        # The ceiling dominates every labeling of the same budget.
        assert float(row["profiled_retention"]) <= float(row["ceiling_retention"]) + 1e-9


def test_michelson_json_matches_the_published_compile_bridge_and_rules() -> None:
    metrics = _load(MICHELSON_METRICS)
    bridge = _mapping(metrics, "compile_bridge")
    assert bridge["exchange_stable"] is True
    assert bridge["voronoi_consistent"] is True
    assert bridge["reproduces_training_labels"] is True
    assert str(bridge["refusal_message"]).endswith("[CE-DS-GLOBAL-GEOMETRY-001]")
    predictions = bridge["predictions"]
    assert isinstance(predictions, list) and len(predictions) == 3
    assert [row["u"] for row in predictions] == [1.0, 7.0, 15.0]

    rules = {str(row["key"]): row for row in _listing(metrics, "rules")}
    assert set(rules) == {"d_rule", "ds_rule"}
    assert round(float(rules["d_rule"]["criterion_efficiency"]), 4) == pytest.approx(  # type: ignore[arg-type]
        0.8345, abs=5e-4
    )
    assert float(rules["d_rule"]["hardening_gap"]) == pytest.approx(0.0, abs=1e-9)  # type: ignore[arg-type]
    # The soft profiled fit is the only route to a reusable profiled rule. It
    # retains a substantial majority of the profiled phase information, with a
    # small hardening gap between the soft objective and the deployed hard rule.
    assert 0.7 < float(rules["ds_rule"]["criterion_efficiency"]) < 0.99  # type: ignore[arg-type]
    assert abs(float(rules["ds_rule"]["hardening_gap"])) < 1e-2  # type: ignore[arg-type]
    # `criterion_efficiency` is each rule's score on its own criterion, on
    # different denominators. `profiled_retention` is the comparable column:
    # both rules' own labels through the sweep's profiled ceiling. The profiled
    # rule must win there, or the criterion is not doing its job.
    assert float(rules["ds_rule"]["profiled_retention"]) > float(  # type: ignore[arg-type]
        rules["d_rule"]["profiled_retention"]
    )
    # A nearest-centre rule in whitened score space is a restricted family: the
    # finite profiled partition, which is not one, retains more.
    sweep = {int(row["n_bins"]): row for row in _listing(metrics, "sweep")}  # type: ignore[call-overload]
    headline = sweep[int(metrics["headline_bins"])]  # type: ignore[call-overload]
    assert float(rules["ds_rule"]["profiled_retention"]) < float(headline["profiled_retention"])  # type: ignore[arg-type]
    for key in ("d_rule", "ds_rule"):
        assert int(rules[key]["n_runs"]) > int(metrics["headline_bins"])  # type: ignore[call-overload]
        assert len(rules[key]["predictions"]) == 3  # type: ignore[arg-type]


def test_michelson_json_diagnoses_the_two_partitions() -> None:
    """The D pull-back is a comb; the D_s fragments are the initializer's and carry information."""
    metrics = _load(MICHELSON_METRICS)
    n_bins = int(metrics["headline_bins"])  # type: ignore[call-overload]

    d_geometry = _mapping(metrics, "d_geometry")
    assert int(d_geometry["n_runs"]) == 24  # type: ignore[call-overload]
    # Two cells sit at the origin, where every loop begins and ends, and are
    # crossed twice per loop; the four outer cells each own the tips of two loops.
    assert sorted(d_geometry["runs_per_bin"]) == [2, 2, 2, 2, 8, 8]  # type: ignore[arg-type]
    assert sum(d_geometry["runs_per_bin"]) == int(d_geometry["n_runs"])  # type: ignore[arg-type,call-overload]
    assert float(d_geometry["min_run_width"]) > 0.0  # type: ignore[arg-type]

    diagnostics = _mapping(metrics, "profiled_diagnostics")
    initial = _mapping(diagnostics, "initial")
    final = _mapping(diagnostics, "final")
    assert diagnostics["exchange_stable"] is True
    assert int(final["n_runs"]) > n_bins  # type: ignore[call-overload]
    assert sum(final["runs_per_bin"]) == int(final["n_runs"])  # type: ignore[arg-type,call-overload]
    # The fragments are already in the certified interval initializer: the
    # exchange does not create them, and the objective gain it adds is tiny.
    assert int(initial["narrow_runs"]) >= int(final["narrow_runs"])  # type: ignore[call-overload]
    assert final["runs_per_bin"] == initial["runs_per_bin"]
    assert int(final["n_runs"]) == int(d_geometry["n_runs"])  # type: ignore[call-overload]
    assert int(final["narrow_runs"]) > int(d_geometry["narrow_runs"])  # type: ignore[call-overload]
    assert 0.0 <= float(diagnostics["retention_gain"]) < 1e-2  # type: ignore[arg-type]
    assert 0.0 <= float(diagnostics["objective_gain"]) < 1e-2  # type: ignore[arg-type]
    assert 0.0 < float(final["min_run_mass"]) < 1e-2  # type: ignore[arg-type]

    smoothing = _listing(diagnostics, "smoothing")
    assert [float(row["min_width"]) for row in smoothing] == [0.15, 0.25, 0.5, 1.0]  # type: ignore[arg-type]
    # Absorbing wider runs can only remove runs, and every rung of the ladder
    # costs retention: the fragments carry information, so the page never calls
    # them negligible.
    runs = [int(row["n_runs"]) for row in smoothing]  # type: ignore[call-overload]
    assert runs == sorted(runs, reverse=True)
    retentions = [float(row["retention"]) for row in smoothing]  # type: ignore[arg-type]
    assert retentions[0] < float(diagnostics["final_retention"])  # type: ignore[arg-type]
    assert retentions == sorted(retentions, reverse=True)
    assert all(int(row["bins_used"]) == n_bins for row in smoothing)  # type: ignore[call-overload]


def test_michelson_headline_row_runs_show_the_comb_against_the_aperture() -> None:
    """`label_runs` output already committed shows the comb the study is about.

    Read from `michelson-phase.json` only -- no refit -- since `label_runs`'s
    run-length encoding of the headline row's three labelings is exactly what
    `sweep_bin_budget` wrote there. The equal-width aperture cannot express
    the fitted rule's cell boundaries contiguously, so the fitted rule's runs
    outnumber its own bin budget while the aperture's own runs equal it
    exactly.
    """
    metrics = _load(MICHELSON_METRICS)
    headline_bins = int(metrics["headline_bins"])  # type: ignore[call-overload]
    sweep = {int(row["n_bins"]): row for row in _listing(metrics, "sweep")}  # type: ignore[call-overload]
    headline = sweep[headline_bins]
    equal_width_runs = headline["equal_width_runs"]
    profiled_runs = headline["profiled_runs"]
    assert isinstance(equal_width_runs, list)
    assert isinstance(profiled_runs, list)
    assert len(equal_width_runs) == headline_bins
    assert len(profiled_runs) > headline_bins


def test_michelson_closed_forms_hold_to_machine_precision() -> None:
    """The page's reason to exist: a check on the library, not only an illustration."""
    provider = build_provider()
    sample = build_train_sample(provider, n_nodes=8_000)
    information = np.asarray(sq.fisher_information(sample.scores, sample.weights))
    closed_form = closed_form_information()
    assert abs(float(information[0, 0]) - closed_form["i_phiphi"]) < 1e-12
    assert abs(float(information[0, 1]) - closed_form["i_phieps"]) < 1e-12


def test_michelson_absorb_narrow_runs_merges_across_the_detector_seam() -> None:
    """A fragment at `u = 0` whose wider neighbour is the last run joins it, and the loop ends."""
    n_nodes = 8_000
    observations = ((np.arange(n_nodes) + 0.5) * U_MAX / n_nodes)[:, None]
    labels = np.full(n_nodes, 1)
    labels[:40] = 0  # a fragment at the seam
    labels[40:100] = 4  # a right neighbour narrower than the wrapped left one
    labels[7_000:] = 3  # the wrapped left neighbour
    smoothed = absorb_narrow_runs(observations, labels, min_width=0.15, u_max=U_MAX)
    runs = periodic_runs(observations, smoothed, u_max=U_MAX)
    assert all(stop - start >= 0.15 for start, stop, _ in runs)
    assert runs[0][2] == 3 and runs[0][0] < 0.0
    assert 0 not in {label for _, _, label in runs}


def test_fast_rerun_reproduces_the_michelson_aliasing_and_criterion_gap() -> None:
    """A small fast-mode rerun reproduces the qualitative claims the page makes."""
    study = run_study(n_nodes=2_000, soft_steps=20, budgets=(4, 6))
    by_bins = {row.n_bins: row for row in study.sweep}

    # Four equal-width segments over four fringes retain exactly nothing.
    assert by_bins[4].equal_width_retention < 1e-6
    # Every criterion beats the naive equal-width rule at six bins.
    assert by_bins[6].equal_width_retention < by_bins[6].d_optimal_retention
    assert by_bins[6].d_optimal_retention < by_bins[6].profiled_retention_value
    # The profiled solver stays close to its own certified ceiling.
    assert by_bins[6].profiled_retention_value <= by_bins[6].ceiling_retention + 1e-9
    assert by_bins[6].bound_gap < 1e-2

    sample = build_train_sample(build_provider(), n_nodes=2_000)
    reference = michelson_unbinned_profiled_information(sample.scores, sample.weights)
    assert reference > 0.0
    assert by_bins[6].ceiling_retention <= 1.0 + 1e-9


# --- docs/usecases/hep/index.md --------------------------------------------

# The in-sample partition table on all events: n_bins, full retention,
# profiled retention, at the precision the doc page prints. The two-bin cut
# is not identified under three floating parameters, so its profiled entry
# is None rather than a number.
PAGE_HEP_PARTITIONS: dict[str, tuple[int, float | None, float | None]] = {
    "ds_partition": (6, 0.47837, 0.95503),
    "d_partition": (6, 0.87456, 0.82381),
    "ds_rule": (6, 0.56195, 0.91455),
    "d_rule": (6, 0.87456, 0.82381),
    "classifier_significance_intervals": (6, 0.23073, 0.33154),
    "classifier_logit_equal_width": (6, 0.24364, 0.33328),
    "classifier_quantile": (6, 0.23144, 0.26824),
    "threshold_cut": (2, 0.0, None),
}

# The bin-budget sweep, in sample: bins, profiled D_s, classifier-quantile
# baseline, certified ceiling, and the certified gap in nats.
PAGE_HEP_SWEEP = [
    (3, 0.81311, 0.06197, 0.84055, 0.033194),
    (4, 0.89516, 0.17765, 0.89818, 0.003367),
    (6, 0.95503, 0.26824, 0.95528, 0.000257),
    (8, 0.97526, 0.38642, 0.97530, 0.000038),
]

# Held out: every reusable rule built on one half and scored on the other,
# mean of both directions of the split.
PAGE_HEP_HELD_OUT: dict[str, float | None] = {
    "ds_rule": 0.8576,
    "d_rule": 0.8025,
    "classifier_significance_intervals": 0.3244,
    "classifier_logit_equal_width": 0.2712,
    "classifier_quantile": 0.3682,
    "threshold_cut": None,
}

# Downstream, all events: relative sigma(mu) with tes fixed and with tes
# floating, per rule.
PAGE_HEP_SIGMA: dict[str, tuple[float, float | None]] = {
    "ds_rule": (0.700, 0.720),
    "d_rule": (0.735, 0.765),
    "classifier_significance_intervals": (0.596, 1.909),
    "classifier_logit_equal_width": (0.636, 1.460),
    "classifier_quantile": (0.669, 1.190),
    "threshold_cut": (0.672, None),
}


def _hep_metrics() -> dict[str, object]:
    return _load(HEP_METRICS)


def _hep_in_sample() -> dict[str, object]:
    return _mapping(_hep_metrics(), "in_sample")


def test_hep_json_matches_the_fixture_facts() -> None:
    metrics = _hep_metrics()
    fixture = _mapping(metrics, "fixture")
    assert fixture["n_events"] == 1_000
    assert fixture["signal_events"] == 336
    assert fixture["background_events"] == 664
    assert round(float(fixture["effective_events"])) == 665  # type: ignore[arg-type]
    assert metrics["schema"] == ["mu_htautau", "nu_background", "tes"]
    assert metrics["interest"] == [0]
    assert metrics["n_bins"] == 6
    assert metrics["n_parameters"] == 3
    assert metrics["delta"] == 0.05


def test_hep_json_matches_the_published_classifier_diagnostics() -> None:
    classifiers = _mapping(_hep_metrics(), "classifiers")
    assert round(float(classifiers["signal_weighted_auc"]), 2) == pytest.approx(0.83, abs=0.02)  # type: ignore[arg-type]
    assert round(float(classifiers["signal_fraction"]), 4) == pytest.approx(0.0010, abs=2e-4)  # type: ignore[arg-type]
    # Weighted out-of-fold AUC of the tes task, comfortably above chance and
    # far from the 0.34 a plain per-row split produces.
    assert 0.55 < float(classifiers["tes_minus_plus_auc"]) < 0.65  # type: ignore[arg-type]
    # Two tes classifiers trained on disjoint events reproduce the score
    # column only in part: the page says so, and pins the order of magnitude.
    reliability = _mapping(classifiers, "tes_reliability")
    assert 0.3 < float(reliability["weighted_correlation"]) < 0.7  # type: ignore[arg-type]


def test_hep_json_matches_the_published_partition_table() -> None:
    by_key = _mapping(_hep_in_sample(), "by_key")
    assert set(by_key) == set(PAGE_HEP_PARTITIONS)
    for key, (n_bins, full, profiled) in PAGE_HEP_PARTITIONS.items():
        row = _mapping(by_key, key)
        assert int(row["n_bins"]) == n_bins  # type: ignore[call-overload]
        assert round(float(row["full_retention"]), 5) == pytest.approx(full, abs=5e-4)  # type: ignore[arg-type]
        if profiled is None:
            assert row["profiled_retention"] is None
        else:
            assert round(float(row["profiled_retention"]), 5) == pytest.approx(profiled, abs=5e-4)  # type: ignore[arg-type]


def test_hep_json_supports_the_central_prediction() -> None:
    """The prediction, measured in sample and held out.

    Profiled D_s beats every classifier-output binning on profiled
    information, in sample and on events the rule never saw, and each
    criterion wins on its own objective.
    """
    by_key = _mapping(_hep_in_sample(), "by_key")
    ds_profiled = float(_mapping(by_key, "ds_partition")["profiled_retention"])  # type: ignore[arg-type]
    d_profiled = float(_mapping(by_key, "d_partition")["profiled_retention"])  # type: ignore[arg-type]
    classifier_keys = (
        "classifier_significance_intervals",
        "classifier_logit_equal_width",
        "classifier_quantile",
    )
    best_baseline = max(
        float(_mapping(by_key, key)["profiled_retention"])  # type: ignore[arg-type]
        for key in classifier_keys
    )
    assert ds_profiled > d_profiled > best_baseline
    gap = _mapping(_hep_in_sample(), "scorequant_vs_classifier_binning")
    assert float(gap["profiled_retention_gap"]) == pytest.approx(ds_profiled - best_baseline)  # type: ignore[arg-type]

    ds_full = float(_mapping(by_key, "ds_partition")["full_retention"])  # type: ignore[arg-type]
    d_full = float(_mapping(by_key, "d_partition")["full_retention"])  # type: ignore[arg-type]
    assert d_full > ds_full

    mean = _mapping(_mapping(_hep_metrics(), "cross_evaluation"), "mean")
    assert set(mean) == set(PAGE_HEP_HELD_OUT)
    for key, expected in PAGE_HEP_HELD_OUT.items():
        value = _mapping(mean, key)["evaluation_profiled_retention"]
        if expected is None:
            assert value is None
        else:
            assert round(float(value), 4) == pytest.approx(expected, abs=5e-4)  # type: ignore[arg-type]
    held_ds = float(_mapping(mean, "ds_rule")["evaluation_profiled_retention"])  # type: ignore[arg-type]
    held_d = float(_mapping(mean, "d_rule")["evaluation_profiled_retention"])  # type: ignore[arg-type]
    for key in classifier_keys:
        assert held_ds > float(_mapping(mean, key)["evaluation_profiled_retention"])  # type: ignore[arg-type]
    assert held_ds > held_d


def test_hep_json_held_out_numbers_are_certified_and_disclosed() -> None:
    """No held-out number sits above its half's ceiling, and both directions are recorded."""
    directions = _listing(_mapping(_hep_metrics(), "cross_evaluation"), "directions")
    assert [str(direction["name"]) for direction in directions] == ["a_to_b", "b_to_a"]
    for direction in directions:
        ceiling = float(direction["evaluation_ceiling_retention"])  # type: ignore[arg-type]
        assert ceiling <= 1.0 + 1e-9
        rules = _mapping(direction, "rules")
        for key in rules:
            row = _mapping(rules, key)
            value = row["evaluation_profiled_retention"]
            if value is not None:
                assert float(value) <= ceiling + 1e-9  # type: ignore[arg-type]
                bootstrap = _mapping(row, "evaluation_bootstrap")
                assert float(bootstrap["p05"]) <= float(value) + 0.2  # type: ignore[arg-type]
                assert bootstrap["dropped"] == 0
        # The finite partition on the reference half is in sample and above
        # the rule it seeds.
        ds_rule = _mapping(rules, "ds_rule")
        assert float(direction["ds_partition_reference_profiled_retention"]) >= float(  # type: ignore[arg-type]
            ds_rule["reference_profiled_retention"]  # type: ignore[arg-type]
        )
        # The library's own validation report agrees with the re-derived number.
        assert float(direction["ds_rule_validation_report_retention"]) == pytest.approx(  # type: ignore[arg-type]
            float(ds_rule["evaluation_profiled_retention"]),  # type: ignore[arg-type]
            abs=1e-6,
        )


def test_hep_json_matches_the_published_ceiling_and_sweep() -> None:
    in_sample = _hep_in_sample()
    ceiling = _mapping(in_sample, "ceiling")
    assert round(float(ceiling["ceiling_retention"]), 5) == pytest.approx(0.95528, abs=5e-4)  # type: ignore[arg-type]
    assert float(ceiling["gap_to_ds_partition_retention"]) == pytest.approx(  # type: ignore[arg-type]
        float(ceiling["ceiling_retention"])  # type: ignore[arg-type]
        - float(_mapping(_mapping(in_sample, "by_key"), "ds_partition")["profiled_retention"])  # type: ignore[arg-type]
    )

    sweep = _listing(in_sample, "ceiling_sweep")
    assert len(sweep) == len(PAGE_HEP_SWEEP)
    for row, (n_bins, ds_profiled, quantile, published_ceiling, gap) in zip(
        sweep, PAGE_HEP_SWEEP, strict=True
    ):
        assert int(float(row["n_bins"])) == n_bins  # type: ignore[arg-type]
        assert round(float(row["ds_profiled_retention"]), 5) == pytest.approx(ds_profiled, abs=5e-4)  # type: ignore[arg-type]
        assert round(
            float(row["classifier_quantile_profiled_retention"]),
            5,  # type: ignore[arg-type]
        ) == pytest.approx(quantile, abs=5e-4)
        assert round(float(row["ceiling_retention"]), 5) == pytest.approx(  # type: ignore[arg-type]
            published_ceiling, abs=5e-4
        )
        assert round(float(row["gap"]), 4) == pytest.approx(gap, abs=5e-3)  # type: ignore[arg-type]
        assert float(row["ds_profiled_retention"]) <= float(row["ceiling_retention"]) + 1e-9  # type: ignore[arg-type]

    held_out = _listing(_mapping(_hep_metrics(), "cross_evaluation"), "held_out_budget_sweep")
    assert [int(float(row["n_bins"])) for row in held_out] == [3, 4, 6, 8]  # type: ignore[arg-type]
    values = [float(row["ds_rule_evaluation_profiled_retention"]) for row in held_out]  # type: ignore[arg-type]
    assert values == sorted(values)
    for row, in_sample_row in zip(held_out, sweep, strict=True):
        assert float(row["ds_rule_evaluation_profiled_retention"]) <= float(  # type: ignore[arg-type]
            in_sample_row["ceiling_retention"]  # type: ignore[arg-type]
        )


def test_hep_json_matches_the_published_delta_convergence() -> None:
    convergence = _mapping(_hep_in_sample(), "delta_convergence")
    rows = _listing(convergence, "rows")
    assert [float(row["delta"]) for row in rows] == [0.025, 0.05, 0.10]  # type: ignore[arg-type]
    for row in rows:
        assert 0.5 < float(row["minus_plus_auc"]) < 0.7  # type: ignore[arg-type]
        assert float(row["ds_profiled_retention"]) <= float(row["ceiling_retention"]) + 1e-9  # type: ignore[arg-type]
    agreement = _mapping(convergence, "agreement")
    assert float(agreement["headline_delta"]) == pytest.approx(0.05)  # type: ignore[arg-type]
    assert float(agreement["half_delta"]) == pytest.approx(0.025)  # type: ignore[arg-type]
    assert float(agreement["retention_gap"]) < 0.01  # type: ignore[arg-type]


def test_hep_json_matches_the_published_downstream_uncertainties() -> None:
    """The count table's own likelihood: tes fixed against tes floating, per rule."""
    downstream = _mapping(_hep_metrics(), "downstream")
    rules = _mapping(_mapping(downstream, "all_events"), "rules")
    assert set(rules) == set(PAGE_HEP_SIGMA)
    for key, (fixed, floating) in PAGE_HEP_SIGMA.items():
        row = _mapping(rules, key)
        assert round(float(row["sigma_mu_tes_fixed"]), 3) == pytest.approx(fixed, abs=2e-3)  # type: ignore[arg-type]
        if floating is None:
            assert row["sigma_mu"] is None
        else:
            assert round(float(row["sigma_mu"]), 3) == pytest.approx(floating, abs=2e-3)  # type: ignore[arg-type]
            # Floating a nuisance never helps, and the Monte Carlo statistical
            # inflation only ever widens the interval.
            assert float(row["sigma_mu"]) >= float(row["sigma_mu_tes_fixed"])  # type: ignore[arg-type]
            assert float(row["sigma_mu_mc_inflated"]) > float(row["sigma_mu"])  # type: ignore[arg-type]
    ds = _mapping(rules, "ds_rule")
    # The physical-width constraint recorded next to the unconstrained number
    # changes it by less than one percent here.
    assert float(ds["sigma_mu_tes_constrained"]) == pytest.approx(float(ds["sigma_mu"]), rel=1e-2)  # type: ignore[arg-type]
    # The scientific lesson in the reported quantity: the profiled rule loses
    # a few percent when tes floats, the classifier-output rules lose a
    # factor of two or more.
    assert float(ds["sigma_mu"]) / float(ds["sigma_mu_tes_fixed"]) < 1.1  # type: ignore[arg-type]
    for key in (
        "classifier_logit_equal_width",
        "classifier_quantile",
        "classifier_significance_intervals",
    ):
        row = _mapping(rules, key)
        assert float(row["sigma_mu"]) / float(row["sigma_mu_tes_fixed"]) > 1.7  # type: ignore[arg-type]
        assert float(ds["sigma_mu"]) < float(row["sigma_mu"])  # type: ignore[arg-type]
    held_out = _listing(downstream, "held_out")
    assert len(held_out) == 2
    for direction in held_out:
        for key in ("classifier_logit_equal_width", "classifier_quantile"):
            baseline = _mapping(_mapping(direction, "rules"), key)
            assert float(_mapping(_mapping(direction, "rules"), "ds_rule")["sigma_mu"]) < float(  # type: ignore[arg-type]
                baseline["sigma_mu"]  # type: ignore[arg-type]
            )


def test_fast_rerun_reproduces_the_hep_classifier_gap() -> None:
    """A small fast-mode rerun reproduces the qualitative claim, not the pinned numbers."""
    study = run_hep_study(
        n_folds=3, max_iter=60, soft_steps=80, budgets=(3, 6), bootstrap_replicates=10
    )
    in_sample = _mapping(study.metrics, "in_sample")
    by_key = _mapping(in_sample, "by_key")

    ds_profiled = float(_mapping(by_key, "ds_partition")["profiled_retention"])  # type: ignore[arg-type]
    quantile_profiled = float(_mapping(by_key, "classifier_quantile")["profiled_retention"])  # type: ignore[arg-type]
    logit_profiled = float(_mapping(by_key, "classifier_logit_equal_width")["profiled_retention"])  # type: ignore[arg-type]
    assert ds_profiled > max(quantile_profiled, logit_profiled)

    ceiling = _mapping(in_sample, "ceiling")
    assert ds_profiled <= float(ceiling["ceiling_retention"]) + 1e-9  # type: ignore[arg-type]

    classifiers = _mapping(study.metrics, "classifiers")
    assert float(classifiers["tes_minus_plus_auc"]) > 0.5
    assert 0.0 < float(classifiers["signal_fraction"]) < 0.01  # type: ignore[arg-type]

    # Held out, both directions exist and stay under their halves' ceilings.
    directions = _listing(_mapping(study.metrics, "cross_evaluation"), "directions")
    assert len(directions) == 2
    for direction in directions:
        ds_rule = _mapping(_mapping(direction, "rules"), "ds_rule")
        value = ds_rule["evaluation_profiled_retention"]
        assert value is not None
        assert float(value) <= float(direction["evaluation_ceiling_retention"]) + 1e-9  # type: ignore[arg-type]


# --- docs/examples/door3-classifier.md -------------------------------------

# The ladder table: labeled training size, surrogate retention, true
# retention, and the ratio-closure residual (evaluated on the dedicated
# training-measure sample, not on `train_observations`), at the precision
# the page prints. Transcribed from the committed JSON.
PAGE_DOOR3_LADDER: dict[int, tuple[float, float, float]] = {
    15: (0.9658, 0.8847, 0.0420),
    60: (0.9691, 0.9521, 0.0185),
    300: (0.9700, 0.9665, 0.0032),
}


def _door3_metrics() -> dict[str, object]:
    return _load(DOOR3_METRICS)


def test_door3_json_matches_the_published_ladder() -> None:
    metrics = _door3_metrics()
    assert metrics["n_train"] == 400
    assert metrics["n_test"] == 600
    assert metrics["n_closure"] == 50_000
    assert metrics["n_bins"] == 4
    mixture = _mapping(metrics, "mixture")
    assert mixture["signal_mu"] == pytest.approx(1.0)
    assert mixture["signal_sigma"] == pytest.approx(0.5)
    assert mixture["background_mu"] == pytest.approx(0.0)
    assert mixture["background_sigma"] == pytest.approx(1.5)
    assert list(mixture["reference_fractions"]) == pytest.approx([0.3, 0.7])  # type: ignore[arg-type]

    ladder = {int(row["n_per_class"]): row for row in _listing(metrics, "ladder")}  # type: ignore[call-overload]
    assert set(ladder) == set(PAGE_DOOR3_LADDER)
    for n_per_class, (surrogate, true, closure) in PAGE_DOOR3_LADDER.items():
        row = ladder[n_per_class]
        assert round(float(row["surrogate_retention"]), 4) == pytest.approx(surrogate, abs=5e-5)  # type: ignore[arg-type]
        assert round(float(row["true_retention"]), 4) == pytest.approx(true, abs=5e-5)  # type: ignore[arg-type]
        assert round(float(row["closure_residual"]), 4) == pytest.approx(closure, abs=5e-5)  # type: ignore[arg-type]

    # Evaluated on the measure the ratios are actually defined against, the
    # closure residual is a genuine estimator-bias signal: it falls
    # monotonically as the classifier improves.
    ordered = [float(ladder[n]["closure_residual"]) for n in (15, 60, 300)]
    assert ordered[0] > ordered[1] > ordered[2]


def test_door3_json_matches_the_published_gap_and_ceiling() -> None:
    metrics = _door3_metrics()
    gap = _mapping(metrics, "surrogate_gap")
    gap_rows = gap["rows"]
    assert isinstance(gap_rows, list)
    gaps = {int(row["n_per_class"]): float(row["gap"]) for row in gap_rows}  # type: ignore[union-attr]
    # The gap the surrogate number hides shrinks monotonically as the
    # classifier improves, and is largest at the smallest training size.
    assert gaps[15] > gaps[60] > gaps[300]
    assert gap["largest_n_per_class"] == 15
    assert round(float(gap["largest_gap"]), 4) == pytest.approx(0.0811, abs=5e-5)  # type: ignore[arg-type]

    ceiling = _mapping(metrics, "exact_ceiling")
    # The ceiling a quantizer fit directly from the exact score reaches on
    # this sample -- what the ladder's largest rung (0.9700 surrogate, 0.9665
    # true) is converging toward.
    assert round(float(ceiling["retention"]), 3) == pytest.approx(0.972, abs=5e-4)  # type: ignore[arg-type]

    # Evaluated on the correct (training) measure, the exact provider's own
    # closure residual sits at Monte Carlo noise -- three orders of
    # magnitude below the worst-trained classifier rung, and below the
    # best-trained rung too, though by a modest margin there (roughly 1.9x):
    # a classifier trained on 300 labeled events per class has essentially
    # converged, so its own bias is already close to this sample's noise
    # floor. This is the ordering the earlier (wrong-measure) version of
    # this test had inverted.
    ladder = {int(row["n_per_class"]): row for row in _listing(metrics, "ladder")}  # type: ignore[call-overload]
    assert round(float(ceiling["closure_residual"]), 4) == pytest.approx(0.0017, abs=5e-5)  # type: ignore[arg-type]
    for row in ladder.values():
        assert float(ceiling["closure_residual"]) < float(row["closure_residual"])


def test_door3_json_matches_the_published_closure_measure_mismatch() -> None:
    """The deliberate, labelled contrast: what closure looks like on the wrong measure."""
    metrics = _door3_metrics()
    mismatch = _mapping(metrics, "closure_measure_mismatch")
    assert "wrong measure" in str(mismatch["description"]).lower()
    assert "not estimator error" in str(mismatch["description"]).lower()

    assert round(float(mismatch["wrong_measure_residual"]), 3) == pytest.approx(  # type: ignore[arg-type]
        0.180, abs=5e-4
    )
    means = mismatch["analytic_reference_measure_ratio_means"]
    assert isinstance(means, list)
    assert [round(float(v), 4) for v in means] == pytest.approx(  # type: ignore[arg-type]
        [0.8411, 1.1589], abs=5e-5
    )

    # The wrong-measure residual is not smaller than the exact provider's
    # correct-measure one -- it is over a hundred times larger, which is the
    # whole point of keeping both numbers side by side.
    ceiling = _mapping(metrics, "exact_ceiling")
    assert float(mismatch["wrong_measure_residual"]) > 50 * float(ceiling["closure_residual"])  # type: ignore[arg-type]


# The naive-binning comparison: labeled training size, equal-frequency and
# equal-width retention of the same estimated score, and the fitted
# partition's own true retention alongside, at the precision the page prints.
PAGE_DOOR3_NAIVE_BINNING: dict[int, tuple[float, float, float]] = {
    15: (0.8842, 0.8648, 0.8847),
    60: (0.9304, 0.9366, 0.9521),
    300: (0.9378, 0.9678, 0.9665),
}


def test_door3_json_matches_the_published_naive_binning() -> None:
    """The one-dimensional naive-binning baseline the walkthrough's comparison beat needs.

    Measured, not assumed: in one dimension quantile cells are close to the
    D-optimal partition, and at the best-trained rung the naive equal-width
    rule actually *beats* the fitted partition. That is the qualitative
    ordering this test pins -- not the "ScoreQuant always wins" shape the
    other examples happen to show.
    """
    metrics = _door3_metrics()
    naive = _mapping(metrics, "naive_binning")
    assert "measured, not assumed" in str(naive["description"]).lower()

    rows = {int(row["n_per_class"]): row for row in _listing(naive, "rows")}  # type: ignore[call-overload]
    assert set(rows) == set(PAGE_DOOR3_NAIVE_BINNING)
    for n_per_class, (eq_freq, eq_width, true) in PAGE_DOOR3_NAIVE_BINNING.items():
        row = rows[n_per_class]
        assert round(float(row["equal_frequency_retention"]), 4) == pytest.approx(  # type: ignore[arg-type]
            eq_freq, abs=5e-5
        )
        assert round(float(row["equal_width_retention"]), 4) == pytest.approx(  # type: ignore[arg-type]
            eq_width, abs=5e-5
        )
        assert round(float(row["true_retention"]), 4) == pytest.approx(true, abs=5e-5)  # type: ignore[arg-type]

    # The qualitative finding, pinned as observed: at 15/class the fitted
    # partition and the quantile rule are within half a point of each other;
    # at 300/class the naive equal-width rule actually beats the fit.
    small = rows[15]
    assert abs(float(small["true_retention"]) - float(small["equal_frequency_retention"])) < 0.001
    large = rows[300]
    assert float(large["equal_width_retention"]) > float(large["true_retention"])

    ceiling_row = _mapping(naive, "exact_ceiling")
    assert ceiling_row["n_per_class"] is None
    assert round(float(ceiling_row["equal_width_retention"]), 4) == pytest.approx(  # type: ignore[arg-type]
        0.9679, abs=5e-5
    )
    assert round(float(ceiling_row["true_retention"]), 3) == pytest.approx(0.972, abs=5e-4)  # type: ignore[arg-type]
    # Even the exact-score ceiling only opens a modest gap over naive
    # equal-width cells of the same score -- the one-dimensional gap this
    # study measures is small everywhere, including where there is no
    # classifier to blame.
    assert float(ceiling_row["true_retention"]) - float(ceiling_row["equal_width_retention"]) < 0.01

    # The single quoted figure: the largest gap the fit opens over the
    # better naive rule, at any rung -- small, as expected for a
    # one-dimensional score, and not at the smallest (worst-classifier) rung.
    assert round(float(naive["largest_gap"]), 4) == pytest.approx(0.0155, abs=5e-5)  # type: ignore[arg-type]
    assert naive["largest_gap_n_per_class"] == 60
    assert float(naive["largest_gap"]) < 0.02


def test_door3_json_matches_the_published_provenance() -> None:
    metrics = _door3_metrics()
    provenance = _mapping(metrics, "provenance")
    classifier = provenance["classifier"]
    assert isinstance(classifier, dict)
    assert classifier["provenance_kind"] == "estimated_ratio"
    assert classifier["exact_fisher"] is False
    assert classifier["information_kind"] == "supplied_score_surrogate"

    exact = provenance["exact"]
    assert isinstance(exact, dict)
    assert exact["provenance_kind"] == "exact"
    assert exact["exact_fisher"] is True
    assert exact["information_kind"] == "exact_fisher"


def test_fast_rerun_reproduces_the_door3_surrogate_gap() -> None:
    """A small fast-mode rerun reproduces the qualitative claims, not the pinned numbers.

    `n_closure` stays at the full-scale `N_CLOSURE`: the closure check never
    enters a fit, so it costs nothing extra, and the ladder rungs and the
    exact ceiling need a shared, high-power closure sample for their
    residuals to be a meaningful comparison at all -- the fast rerun's speed
    comes entirely from the smaller `n_train`/`n_test` fits below.
    """
    study = run_door3_study(
        n_per_class_values=(15, 60, 300), n_train=200, n_test=300, n_closure=50_000, n_bins=4
    )
    metrics = study.metrics
    ladder = {int(row["n_per_class"]): row for row in _listing(metrics, "ladder")}  # type: ignore[call-overload]

    # The self-reported surrogate retention overstates the truth most sharply
    # at the smallest training size ...
    small = ladder[15]
    assert float(small["surrogate_retention"]) > float(small["true_retention"])

    # ... and the gap the surrogate hides shrinks as the classifier improves.
    gaps = [
        float(ladder[n]["surrogate_retention"]) - float(ladder[n]["true_retention"])
        for n in (15, 60, 300)
    ]
    assert gaps[0] > gaps[1] > gaps[2]

    # The provenance facts hold regardless of scale: a classifier-derived
    # result never claims exact Fisher semantics, and the exact provider
    # always does.
    provenance = _mapping(metrics, "provenance")
    assert provenance["classifier"]["information_kind"] == "supplied_score_surrogate"  # type: ignore[index]
    assert provenance["classifier"]["exact_fisher"] is False  # type: ignore[index]
    assert provenance["exact"]["information_kind"] == "exact_fisher"  # type: ignore[index]
    assert provenance["exact"]["exact_fisher"] is True  # type: ignore[index]

    # Evaluated on the correct measure, the closure residual falls
    # monotonically with classifier quality, and the exact provider's own
    # residual sits below every rung's -- the property that would catch a
    # regression back to evaluating on the wrong measure.
    closure = [float(ladder[n]["closure_residual"]) for n in (15, 60, 300)]
    assert closure[0] > closure[1] > closure[2]
    ceiling = _mapping(metrics, "exact_ceiling")
    assert all(float(ceiling["closure_residual"]) < value for value in closure)

    # The wrong-measure contrast is still visibly larger than the corrected
    # residual at this scale too.
    mismatch = _mapping(metrics, "closure_measure_mismatch")
    assert float(mismatch["wrong_measure_residual"]) > 10 * float(ceiling["closure_residual"])  # type: ignore[arg-type]
