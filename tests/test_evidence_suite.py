"""The walkthrough studies: committed evidence and fast reruns of their claims.

Each study in ``examples/`` writes a JSON file under ``examples/outputs`` that the
portal walkthroughs quote. These tests pin that evidence and rerun each study at
reduced scale to check that its central claim still holds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scorequant as sq
from examples.door3_classifier import run_study as run_door3_study
from examples.hep_classifier.experiment import run_study as run_hep_study
from examples.ideas_toys import run_study as run_ideas_study
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
from tests._fit import fit_test_quantizer

OUTPUTS = Path(__file__).resolve().parents[1] / "examples" / "outputs"
MICHELSON_METRICS = OUTPUTS / "michelson-phase.json"
HEP_METRICS = OUTPUTS / "hep-classifier.json"
DOOR3_METRICS = OUTPUTS / "door3-classifier.json"
IDEAS_METRICS = OUTPUTS / "ideas-toys.json"


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
    information = np.asarray(sq.information.fisher_information(sample.scores, sample.weights))
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

PAGE_HEP_SWEEP = [
    (3, 0.81311, 0.06197, 0.84055, 0.033194),
    (4, 0.89516, 0.17765, 0.89818, 0.003367),
    (6, 0.95503, 0.26824, 0.95528, 0.000257),
    (8, 0.97526, 0.38642, 0.97530, 0.000038),
]

PAGE_HEP_HELD_OUT: dict[str, float | None] = {
    "ds_rule": 0.8576,
    "d_rule": 0.8025,
    "classifier_significance_intervals": 0.3244,
    "classifier_logit_equal_width": 0.2712,
    "classifier_quantile": 0.3682,
    "threshold_cut": None,
}

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


def test_ideas_rerun_reproduces_the_committed_toys_and_their_claims() -> None:
    """The two Ideas toys rerun in seconds, match the evidence, and say what the posts say."""
    metrics, _ = run_ideas_study()
    committed = json.loads(IDEAS_METRICS.read_text(encoding="utf-8"))
    fringe, clusters = metrics["fringe"], metrics["clusters"]
    assert isinstance(fringe, dict) and isinstance(clusters, dict)
    np.testing.assert_allclose(
        fringe["score_intervals"], committed["fringe"]["score_intervals"], rtol=1e-9
    )
    np.testing.assert_allclose(
        fringe["equal_segments"], committed["fringe"]["equal_segments"], rtol=1e-9, atol=1e-12
    )
    for key in ("kmeans", "d_optimal"):
        for field in ("arithmetic_mean_retention", "geometric_mean_retention"):
            assert clusters[key][field] == pytest.approx(
                committed["clusters"][key][field], rel=1e-9
            )

    # One equal segment per fringe keeps nothing; score intervals keep the most at every budget.
    assert fringe["headline_equal"] == 0.0
    assert all(
        s > e for s, e in zip(fringe["score_intervals"], fringe["equal_segments"], strict=True)
    )
    # k-means wins the sum of the retention spectrum, D-optimality the product.
    kmeans, d_optimal = clusters["kmeans"], clusters["d_optimal"]
    assert kmeans["arithmetic_mean_retention"] > d_optimal["arithmetic_mean_retention"]
    assert d_optimal["geometric_mean_retention"] > kmeans["geometric_mean_retention"]
