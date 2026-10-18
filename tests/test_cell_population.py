from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import scorequant as sq
from examples.cell_population.closure import (
    ClosureInputs,
    conditional_binned_fisher_information,
    conditional_fisher_information,
    fixed_total_partition_audit,
    ratio_model_audit,
    template_identifiability_audit,
    uncertainty_coverage_audit,
)
from examples.cell_population.data import (
    CLASS_NAMES,
    FEATURE_NAMES,
    REFERENCE_PATIENTS,
    TEST_PATIENTS,
    FlowCytData,
    RobustArcsinhTransform,
    load_fixture,
)
from examples.cell_population.experiment import predict_score_bins, run_experiment
from examples.cell_population.figures import make_profiled_figure
from examples.cell_population.fixture import (
    CLASS_CODES,
    _proportional_counts,
    _stratified_patient_ranges,
    _write_remote_patient_csv,
)
from examples.cell_population.likelihood import (
    estimate_bin_templates,
    fit_binned_mixture,
    fit_unbinned_mixture,
)
from examples.cell_population.profiled import (
    BUDGETS,
    INTEREST_INDEX,
    profiled_inputs_from_data,
    profiled_scalar,
    run_profiled_study,
    score_labeling,
)
from examples.cell_population.scores import (
    fit_score_model,
    integration_weights,
    reference_composition,
)
from tests._fit import fit_test_quantizer

FIXTURE = Path("examples/data/flowcyt_fixture.npz")


def test_full_csv_download_is_chunked_labelled_and_atomic(tmp_path: Path) -> None:
    def metadata(url: str) -> tuple[dict[str, str], int, int]:
        label = CLASS_CODES.index(url.removesuffix(".fcs").rsplit("_", maxsplit=1)[1])
        return {"label": str(label)}, 100, label + 1

    def read_range(
        url: str,
        metadata_values: dict[str, str],
        data_start: int,
        start: int,
        count: int,
    ) -> np.ndarray:
        del url, data_start
        label = int(metadata_values["label"])
        return np.full((count, len(FEATURE_NAMES)), label + start / 10, dtype=np.float64)

    with (
        patch("examples.cell_population.fixture._fcs_metadata", side_effect=metadata),
        patch("examples.cell_population.fixture._read_fcs_range", side_effect=read_range),
    ):
        evidence = _write_remote_patient_csv(1, tmp_path, chunk_rows=2)

    table = pd.read_csv(tmp_path / "Case_1.csv")
    expected_rows = sum(range(1, len(CLASS_CODES) + 1))
    assert len(table) == expected_rows == evidence["rows"]
    assert np.array_equal(table.groupby("label", sort=True).size().to_numpy(), np.arange(1, 7))
    assert len(str(evidence["sha256"])) == 64
    assert not (tmp_path / "Case_1.csv.part").exists()


FULL_EVIDENCE = Path("examples/outputs/cell_population.json")
PROFILED_EVIDENCE = Path("examples/outputs/flowcyt_profiled_ds.json")
SOLVERS_EVIDENCE = Path("examples/outputs/flowcyt_solvers.json")


def _synthetic_flowcyt(seed: int = 10, per_class: int = 10) -> FlowCytData:
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    patients: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    centers = rng.normal(0, 2, size=(len(CLASS_NAMES), 12))
    for patient in (*REFERENCE_PATIENTS, *TEST_PATIENTS):
        shift = rng.normal(0, 0.08, size=12)
        for label in range(len(CLASS_NAMES)):
            block = centers[label] + shift + rng.normal(0, 0.35, size=(per_class, 12))
            features.append(150 * np.sinh(block))
            labels.append(np.full(per_class, label))
            patients.append(np.full(per_class, patient))
            rows.append(np.arange(per_class))
    return FlowCytData(
        np.concatenate(features),
        np.concatenate(labels),
        np.concatenate(patients),
        np.concatenate(rows),
    )


def test_robust_transform_is_frozen_and_finite() -> None:
    data = _synthetic_flowcyt(per_class=3)
    reference = data.patients_in(REFERENCE_PATIENTS)
    transform = RobustArcsinhTransform.fit(reference.features)
    transformed = transform.apply(data.features)
    assert transformed.shape == data.features.shape
    assert np.isfinite(transformed).all()


def test_posterior_prior_correction_recovers_same_simplex_scores() -> None:
    rng = np.random.default_rng(20)
    ratios = np.exp(rng.normal(size=(50, 6)))
    theta0 = np.asarray([0.16, 0.14, 0.20, 0.08, 0.07, 0.35])
    uniform_prior = np.full(6, 1 / 6)
    skewed_prior = np.asarray([0.35, 0.15, 0.10, 0.05, 0.05, 0.30])
    uniform_posterior = ratios * uniform_prior
    uniform_posterior /= np.sum(uniform_posterior, axis=1, keepdims=True)
    skewed_posterior = ratios * skewed_prior
    skewed_posterior /= np.sum(skewed_posterior, axis=1, keepdims=True)
    np.testing.assert_allclose(
        sq.ratios.MixtureParameterization(theta0).scores(
            sq.ratios.ratios_from_posteriors(uniform_posterior, uniform_prior)
        ),
        sq.ratios.MixtureParameterization(theta0).scores(
            sq.ratios.ratios_from_posteriors(skewed_posterior, skewed_prior)
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_binned_and_unbinned_mixture_recover_known_fractions() -> None:
    templates = np.asarray(
        [
            [0.70, 0.05, 0.05, 0.05, 0.05, 0.10],
            [0.05, 0.70, 0.05, 0.05, 0.05, 0.10],
            [0.05, 0.05, 0.70, 0.05, 0.05, 0.10],
            [0.05, 0.05, 0.05, 0.70, 0.05, 0.10],
            [0.05, 0.05, 0.05, 0.05, 0.70, 0.10],
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.50],
        ]
    )
    theta = np.asarray([0.10, 0.12, 0.15, 0.08, 0.05, 0.50])
    counts = 2_000_000 * (templates @ theta)
    binned = fit_binned_mixture(counts, templates)
    np.testing.assert_allclose(binned.fractions, theta, atol=2e-6)
    np.testing.assert_allclose(binned.covariance, binned.covariance.T)
    assert np.linalg.eigvalsh(binned.covariance).min() >= -1e-12

    ratios = np.repeat(templates.T, 20_000, axis=0)
    unbinned = fit_unbinned_mixture(ratios)
    assert np.isfinite(unbinned.fractions).all()
    np.testing.assert_allclose(np.sum(unbinned.fractions), 1.0)

    component_values = templates
    uniform_priors = np.full(6, 1 / 6)
    posterior_values = component_values * uniform_priors[None, :]
    posterior_values /= np.sum(posterior_values, axis=1, keepdims=True)
    exact_scores = sq.ratios.MixtureParameterization(theta).scores(
        sq.ratios.ratios_from_posteriors(posterior_values, uniform_priors)
    )
    mixture_weights = component_values @ theta
    coarse_bins = np.asarray([0, 0, 1, 1, 2, 2])
    full_fisher = sq.fisher_information(exact_scores, mixture_weights)
    binned_fisher = sq.binned_fisher_information(
        exact_scores, coarse_bins, mixture_weights, n_bins=3
    )
    residual = np.asarray(full_fisher - binned_fisher)
    assert np.linalg.eigvalsh((residual + residual.T) / 2).min() >= -1e-12


def test_fixed_total_information_obeys_psd_ordering_and_bin_rank_bound() -> None:
    rng = np.random.default_rng(101)
    scores = rng.normal(size=(2_000, 5))
    weights = rng.uniform(0.2, 2.0, size=len(scores))
    labels = np.argmax(scores[:, :4], axis=1)
    full = conditional_fisher_information(scores, weights)
    binned = conditional_binned_fisher_information(scores, labels, weights, n_bins=4)
    residual = (full - binned + (full - binned).T) / 2
    assert np.linalg.eigvalsh(residual).min() >= -1e-12
    assert np.linalg.matrix_rank(binned, tol=1e-10) <= 3
    audit = fixed_total_partition_audit(scores, labels, weights, n_bins=4)
    assert audit["retained_rank"] <= audit["rank_bound"] == 3
    assert audit["d_efficiency"] == 0.0


def test_scientific_closure_input_cannot_receive_test_data() -> None:
    names = {field.name for field in fields(ClosureInputs)}
    assert "test" not in names
    assert "test_labels" not in names
    assert "true_fractions" not in names


def test_five_bins_cannot_identify_six_fixed_total_fractions() -> None:
    rng = np.random.default_rng(102)
    templates = rng.dirichlet(np.ones(5), size=6).T
    theta0 = np.asarray([0.10, 0.12, 0.08, 0.05, 0.15, 0.50])
    audit = template_identifiability_audit(templates, theta0)
    assert audit["effective_rank"] <= 4
    assert audit["rank_bound"] == 4
    assert audit["full_rank"] is False
    witness = audit["nonidentifiability_witness"]
    assert isinstance(witness, dict)
    assert witness["fraction_separation_norm"] > 0
    assert witness["maximum_bin_probability_difference"] < 1e-12


def test_six_bins_can_close_a_full_rank_six_class_template() -> None:
    templates = np.full((6, 6), 0.02)
    np.fill_diagonal(templates, 0.9)
    templates /= np.sum(templates, axis=0, keepdims=True)
    theta = np.asarray([0.10, 0.12, 0.08, 0.05, 0.15, 0.50])
    audit = template_identifiability_audit(templates, theta)
    assert audit["full_rank"] is True
    estimate = fit_binned_mixture(200_000 * (templates @ theta), templates)
    np.testing.assert_allclose(estimate.fractions, theta, atol=2e-6)


def test_uncertainty_audit_marks_boundary_dominated_components() -> None:
    templates = np.full((8, 6), 0.01)
    templates[:6] += np.eye(6) * 0.9
    templates /= np.sum(templates, axis=0, keepdims=True)
    theta = np.asarray([0.15, 0.12, 0.10, 1e-12, 0.08, 0.55 - 1e-12])
    audit = uncertainty_coverage_audit(templates, theta, draws=80, event_count=2_000, seed=103)
    reference_mast = audit["scenarios"]["reference_like"]["classes"]["mast cells"]
    assert reference_mast["status"] == "boundary_dominated"
    assert reference_mast["interior_standard_error_ratio"] is None
    assert reference_mast["interior_68_percent_coverage"] is None
    enriched_t = audit["scenarios"]["mast_enriched"]["classes"]["T cells"]
    assert np.isfinite(enriched_t["empirical_standard_deviation"])


def test_template_estimation_is_patient_balanced_and_smoothed() -> None:
    labels = np.tile(np.repeat(np.arange(6), 8), 2)
    patients = np.repeat([1, 2], 48)
    bins = (labels + np.arange(len(labels)) % 2) % 4
    templates = estimate_bin_templates(labels, bins, patients, n_bins=4)
    assert templates.shape == (4, 6)
    assert np.all(templates > 0)
    np.testing.assert_allclose(np.sum(templates, axis=0), 1.0)


def test_cross_fitted_score_model_covers_every_reference_patient() -> None:
    data = _synthetic_flowcyt(per_class=8)
    reference = data.patients_in(REFERENCE_PATIENTS)
    fitted = fit_score_model(reference, max_per_patient_class=8, max_iter=8)
    assert fitted.out_of_fold_probabilities.shape == (len(reference.labels), 6)
    np.testing.assert_allclose(np.sum(fitted.out_of_fold_probabilities, axis=1), 1.0)
    theta0 = reference_composition(reference.labels, reference.patients)
    scores = sq.ratios.MixtureParameterization(theta0).scores(
        sq.ratios.ratios_from_posteriors(
            fitted.out_of_fold_probabilities, fitted.model.class_priors
        )
    )
    assert scores.shape == (len(reference.labels), 5)
    assert fitted.calibration_selection["selected_strategy"] in {
        "raw_declared_prior",
        "raw_oof_prior",
        "temperature_oof_prior",
    }
    outer_folds = fitted.calibration_selection["outer_folds"]
    assert len(outer_folds) == 5
    held_out_patients = [patient for fold in outer_folds for patient in fold["held_out_patients"]]
    assert sorted(held_out_patients) == sorted(REFERENCE_PATIENTS)
    assert len(held_out_patients) == len(set(held_out_patients))
    assert list(fitted.calibration_selection["candidates"]) == [
        "raw_declared_prior",
        "raw_oof_prior",
        "temperature_oof_prior",
    ]
    final_calibration = fitted.calibration_selection["final_calibration"]
    assert float(final_calibration["maximum_normalization_residual"]) < 0.25
    closure = ratio_model_audit(fitted, reference, theta0)
    strategies = closure["strategies"]
    assert strategies["raw_oof_prior"]["maximum_normalization_residual"] < 1e-12
    assert strategies["temperature_oof_prior"]["maximum_normalization_residual"] < 1e-12
    assert np.isfinite(strategies[closure["selected_strategy"]]["mean_score_norm"])
    weights = integration_weights(reference.labels, reference.patients, theta0)
    np.testing.assert_allclose(np.sum(weights), 1.0)


def test_committed_fixture_has_disjoint_patients_and_declared_schema() -> None:
    data = load_fixture(FIXTURE)
    reference = data.patients_in(REFERENCE_PATIENTS)
    test = data.patients_in(TEST_PATIENTS)
    assert len(reference.labels) > 0
    assert len(test.labels) > 0
    assert set(reference.patients).isdisjoint(set(test.patients))
    assert set(reference.labels) == set(range(6))
    assert data.features.shape[1] == 12


def test_chunked_score_prediction_matches_one_shot() -> None:
    rng = np.random.default_rng(44)
    scores = rng.normal(size=(127, 3))
    result = fit_test_quantizer(scores, n_bins=5)
    np.testing.assert_array_equal(
        predict_score_bins(result, scores, chunk_size=13), np.asarray(result.predict_scores(scores))
    )


def test_remote_sample_ranges_are_deterministic_and_bounded() -> None:
    totals = (17, 83, 211, 9, 31, 149)
    first = _stratified_patient_ranges(totals, 137, 12, np.random.default_rng(2026))
    second = _stratified_patient_ranges(totals, 137, 12, np.random.default_rng(2026))
    assert first == second
    assert sum(count for ranges in first for _, count in ranges) == 137
    for total, ranges in zip(totals, first, strict=True):
        assert all(start >= 0 and count > 0 and start + count <= total for start, count in ranges)
        occupied = [set(range(start, start + count)) for start, count in ranges]
        assert sum(len(values) for values in occupied) == len(set().union(*occupied))


def test_remote_sample_allocation_preserves_component_proportions() -> None:
    totals = (66_439, 11_773, 12_768, 66, 3_618, 449_803)
    counts = _proportional_counts(totals, 20_000)
    assert sum(counts) == 20_000
    assert all(0 <= count <= total for count, total in zip(counts, totals, strict=True))
    np.testing.assert_allclose(
        np.asarray(counts) / sum(counts),
        np.asarray(totals) / sum(totals),
        atol=1 / 20_000,
    )


def test_flowcyt_fixture_is_the_standard_end_to_end_use_case() -> None:
    result = run_experiment(
        load_fixture(FIXTURE),
        bin_counts=(5,),
        operating_n_bins=5,
        uncertainty_n_bins=5,
        quick=True,
    )
    soft = result.metrics["soft_voronoi:5"]
    assert np.isfinite(float(soft["target_macro_rmse"]))
    assert 0 <= float(soft["held_out_d_efficiency"]) <= 1
    assert result.predicted_fractions["soft_voronoi:5"].shape == (len(TEST_PATIENTS), 6)
    assert result.operating_bin_composition.shape == (5, 6)
    np.testing.assert_allclose(np.sum(result.operating_bin_composition, axis=1), 1.0)
    assert "unbinned_classifier_ratio" in result.metrics
    assert "unbinned" not in result.metrics
    # The compiled-rule path must be reachable from this test. It used to be
    # gated on a literal 8 bins while this fixture runs 5, so nothing executed
    # compile_quantizer here and a compiled rule that had lost an attribute
    # still passed the whole suite.
    compiled = result.metrics["finite_d_exchange:5"]
    assert compiled["exchange_stable"] is True
    assert compiled["compiled_training_labels_reproduced"] is True
    assert 0 <= float(compiled["compiled_train_d_efficiency"]) <= 1
    closure = result.metrics["scientific_closure"]
    assert closure["reference_only"] is True
    assert closure["template_identifiability"]["soft_voronoi:5"]["full_rank"] is False
    assert closure["fixed_total_information"]["soft_voronoi:5"]["retained_rank"] <= 4
    assert result.metrics["uncertainty"]["protocol"]["source"] == "frozen_reference_templates"


def test_committed_full_patient_evidence_passes_the_frozen_gate() -> None:
    metrics = json.loads(FULL_EVIDENCE.read_text(encoding="utf-8"))
    assert metrics["source"]["sample_rows"] == 600_000
    assert (
        metrics["source"]["sample_sha256"]
        == "a08e9bf183fe32b913e155d413eeacfdb65c7f99017a42e69c4b91bdde20d987"
    )
    assert metrics["run"]["quick"] is False
    assert metrics["acceptance"]["random_d_efficiency_wins"] == 6
    assert metrics["acceptance"]["marker_rmse_noninferiority_wins"] == 6
    assert metrics["soft_voronoi:8"]["held_out_d_efficiency"] >= 0.94
    assert metrics["soft_voronoi:8"]["target_macro_rmse"] <= 0.0023
    assert metrics["soft_voronoi:8"]["likelihood_convergence"]["converged_patients"] == 10
    assert metrics["soft_voronoi:8"]["information_kind"] == "supplied_score_surrogate"
    assert metrics["soft_voronoi:8"]["score_provenance"]["kind"] == "estimated_ratio"
    provenance_ratio = metrics["soft_voronoi:8"]["score_provenance"]["ratio"]
    assert provenance_ratio["estimator"] == "calibrated_classifier"
    assert provenance_ratio["calibration"] == "raw_declared_prior"
    assert np.isfinite(float(metrics["soft_voronoi:8"]["hardening_gap"]))
    finite_d = metrics["finite_d_exchange:8"]
    assert finite_d["exchange_stable"] is True
    assert finite_d["compiled_training_labels_reproduced"] is True
    assert finite_d["geometry_gap"] == 0.0
    assert finite_d["best_remaining_gain"] <= 0
    assert finite_d["information_kind"] == "supplied_score_surrogate"
    assert metrics["calibration_selection"]["selected_strategy"] == "raw_declared_prior"
    assert metrics["run"]["classifier_test_posterior_evaluations"] == 1
    assert (
        metrics["unbinned_classifier_ratio"]["target_macro_rmse"]
        < metrics["soft_voronoi:8"]["target_macro_rmse"]
    )
    composition = np.asarray(metrics["operating_partition"]["reference_bin_composition"])
    assert composition.shape == (8, 6)
    np.testing.assert_allclose(np.sum(composition, axis=1), 1.0)

    closure = metrics["scientific_closure"]
    assert closure["reference_only"] is True
    assert set(closure) == {
        "reference_only",
        "ratio_model",
        "fixed_total_information",
        "template_identifiability",
        "population_limit",
        "pseudo_patients",
        "seed_stability",
        "uncertainty_coverage",
    }
    ratio_strategies = closure["ratio_model"]["strategies"]
    assert ratio_strategies["raw_declared_prior"]["maximum_normalization_residual"] > 0.20
    assert ratio_strategies["raw_oof_prior"]["maximum_normalization_residual"] < 1e-12
    for method in ("score_kmeans", "soft_voronoi"):
        five = closure["template_identifiability"][f"{method}:5"]
        assert five["effective_rank"] == 4
        assert five["full_rank"] is False
        assert five["nonidentifiability_witness"]["maximum_bin_probability_difference"] < 1e-12
        assert closure["fixed_total_information"][f"{method}:5"]["d_efficiency"] == 0.0
        for n_bins in (6, 8):
            assert closure["template_identifiability"][f"{method}:{n_bins}"]["full_rank"]
            assert closure["fixed_total_information"][f"{method}:{n_bins}"]["retained_rank"] == 5

    pseudo = closure["pseudo_patients"]
    assert pseudo["protocol"]["source"] == "reference_validation_rows_only"
    assert pseudo["protocol"]["repeats_per_composition"] == 20
    assert len(pseudo["by_composition"]) == 11
    assert pseudo["methods"]["soft_voronoi:5"]["converged_pseudo_patients"] == 0
    assert pseudo["methods"]["soft_voronoi:6"]["converged_pseudo_patients"] == 220
    stability = closure["seed_stability"]
    assert stability["seeds"] == list(range(2026, 2036))
    assert stability["summary"]["soft_voronoi:5"]["template_rank_range"] == [4, 4]
    assert stability["summary"]["soft_voronoi:8"]["template_rank_range"] == [5, 5]

    uncertainty = metrics["uncertainty"]
    assert uncertainty["protocol"]["source"] == "frozen_reference_templates"
    reference_mast = uncertainty["scenarios"]["reference_like"]["classes"]["mast cells"]
    assert reference_mast["status"] == "boundary_dominated"
    assert reference_mast["interior_standard_error_ratio"] is None
    assert reference_mast["interior_68_percent_coverage"] is None
    enriched_mast = uncertainty["scenarios"]["mast_enriched"]["classes"]["mast cells"]
    assert enriched_mast["status"] == "interior"
    assert 0.9 <= enriched_mast["interior_standard_error_ratio"] <= 1.1


def test_profiled_scalar_matches_the_explicit_schur_complement() -> None:
    rng = np.random.default_rng(210)
    root = rng.normal(size=(9, 5))
    information = root.T @ root
    inverse = np.linalg.inv(information)
    for index in range(5):
        np.testing.assert_allclose(
            profiled_scalar(information, index), 1.0 / inverse[index, index], rtol=1e-9
        )


def test_profiled_score_labeling_separates_the_two_information_conventions() -> None:
    """Five cells can look informative uncentered and identify nothing fixed-total."""
    rng = np.random.default_rng(211)
    scores = rng.normal(size=(1_500, 5)) + 0.4
    weights = rng.uniform(0.2, 1.5, size=len(scores))
    labels = np.argmax(scores, axis=1)
    scored = score_labeling(scores, labels, weights, interest_index=2, n_bins=5)
    assert scored.occupied_bins == 5
    # Five cells give five uncentered moments but only four independent bin
    # frequencies, so the interest fraction is not estimable once the other four
    # float -- exactly the wall the five-bin FlowCyt partitions hit.
    assert scored.fixed_total_rank == 4
    assert scored.fixed_total_profiled_retention == 0.0
    assert 0.0 < scored.profiled_retention <= 1.0


def test_profiled_fixture_study_reproduces_its_qualitative_claims() -> None:
    """Run the whole profiled path on the frozen fixture, never on the 600k sample."""
    inputs = profiled_inputs_from_data(
        load_fixture(FIXTURE),
        quick=True,
        score_max_per_patient_class=48,
        score_max_iter=6,
    )
    assert inputs.partition_scores.shape[1] == 5
    assert inputs.true_fractions.shape == (len(TEST_PATIENTS), 6)
    assert inputs.preparation_seconds > 0

    study = run_profiled_study(
        inputs,
        quick=True,
        n_bins=8,
        budgets=(8,),
        sweep_interest=False,
    )
    metrics = study.metrics
    assert metrics["interest_index"] == INTEREST_INDEX
    assert metrics["interest_population"] == "HSPCs"
    assert metrics["nuisance_populations"] == ["T cells", "B cells", "monocytes", "mast cells"]
    assert metrics["reference_component"] == "other"

    partitions = {row["key"]: row for row in metrics["partitions"]}
    plain = partitions["d_partition"]
    profiled = [partitions["ds_partition_seeded"], partitions["ds_partition_initialized"]]
    ceiling = float(metrics["bound"]["ceiling_retention"])

    # Each criterion wins on its own objective: plain D on the whole matrix,
    # profiled D_s on the interest fraction, and neither may pass the ceiling.
    assert max(row["profiled_retention"] for row in profiled) > plain["profiled_retention"]
    assert all(row["full_retention"] < plain["full_retention"] for row in profiled)
    assert plain["profiled_retention"] <= ceiling + 1e-9
    for row in profiled:
        assert row["profiled_retention"] <= ceiling + 1e-9
        assert row["exchange_stable"] is True
    assert metrics["bound"]["seeded_gap"] >= -1e-9
    assert metrics["bound"]["initialized_gap"] >= -1e-9
    # The ceiling's labels are a cheaper starting point, not a better theorem.
    assert (
        partitions["ds_partition_initialized"]["accepted_moves"]
        < partitions["ds_partition_seeded"]["accepted_moves"]
    )

    rules = {row["key"]: row for row in metrics["rules"]}
    assert set(rules) == {"d_rule", "ds_rule"}
    assert rules["d_rule"]["solver"] == "DExchangeConfig"
    assert rules["ds_rule"]["solver"] == "SoftVoronoiConfig"
    for row in rules.values():
        downstream = row["downstream"]
        assert downstream["converged_patients"] == downstream["total_patients"] == 10
        assert downstream["mean_half_width"] > 0
        assert 0.0 < row["test_profiled_retention"] <= 1.0
        assert row["test_occupied_bins"] <= 8

    sweep = metrics["budget_sweep"]
    assert [row["n_bins"] for row in sweep] == [8]
    assert sweep[0]["fixed_total_rank_d"] == 5
    assert metrics["interest_sweep"] == []
    assert metrics["run"]["rows"]["total"] == 34_554


def test_committed_profiled_evidence_carries_both_declared_scales() -> None:
    evidence = json.loads(PROFILED_EVIDENCE.read_text(encoding="utf-8"))
    assert set(evidence) == {"fixture_scale", "sample_scale"}

    fixture = evidence["fixture_scale"]
    sample = evidence["sample_scale"]
    assert fixture["run"]["quick"] is True
    assert fixture["run"]["provenance"]["scale"] == "frozen CI fixture"
    assert fixture["run"]["rows"]["total"] == 34_554
    assert sample["run"]["quick"] is False
    assert sample["run"]["provenance"]["scale"] == "600,000-cell bounded sample"
    assert sample["run"]["rows"]["total"] == 600_000
    assert (
        sample["run"]["provenance"]["sample_sha256"]
        == "a08e9bf183fe32b913e155d413eeacfdb65c7f99017a42e69c4b91bdde20d987"
    )
    # The profiled study partitions exactly the rows the main study does.
    main_rows = json.loads(FULL_EVIDENCE.read_text(encoding="utf-8"))["run"]["rows"]
    assert {key: sample["run"]["rows"][key] for key in main_rows} == main_rows

    for scale in (fixture, sample):
        assert scale["study"] == "flowcyt_profiled_ds"
        assert scale["interest_index"] == INTEREST_INDEX
        assert scale["interest_population"] == "HSPCs"
        assert scale["reference_component"] == "other"
        assert scale["n_bins"] == 8
        assert scale["budgets"] == list(BUDGETS)
        assert scale["unbinned_profiled_information"] > 0

        partitions = {row["key"]: row for row in scale["partitions"]}
        assert set(partitions) == {
            "d_partition",
            "ds_partition_seeded",
            "ds_partition_initialized",
        }
        ceiling = float(scale["bound"]["ceiling_retention"])
        plain = partitions["d_partition"]
        for key in ("ds_partition_seeded", "ds_partition_initialized"):
            row = partitions[key]
            assert row["criterion"] == "ProfiledDOptimality"
            assert row["profiled_retention"] > plain["profiled_retention"]
            assert row["profiled_retention"] <= ceiling + 1e-9
            # Profiling buys interest information by discarding the rest.
            assert row["full_retention"] < 0.15 * plain["full_retention"]
        assert scale["bound"]["seeded_gap"] >= -1e-9
        assert scale["bound"]["initialized_gap"] >= -1e-9

        budgets = {row["n_bins"]: row for row in scale["budget_sweep"]}
        assert sorted(budgets) == sorted(BUDGETS)
        for row in budgets.values():
            assert row["d_profiled_retention"] <= row["ceiling_retention"] + 1e-9
            assert row["ds_seeded_retention"] <= row["ceiling_retention"] + 1e-9
            assert row["ds_initialized_retention"] <= row["ceiling_retention"] + 1e-9
            assert row["seeded_gap"] >= -1e-9 and row["initialized_gap"] >= -1e-9
        # Five bins cannot identify five free fractions in either convention,
        # however healthy the intensity-convention number looks.
        assert budgets[5]["fixed_total_rank_d"] == 4
        assert budgets[5]["fixed_total_rank_ds"] == 4
        assert budgets[5]["d_fixed_total_retention"] == 0.0
        assert budgets[5]["ds_initialized_fixed_total_retention"] == 0.0
        assert budgets[5]["ds_initialized_retention"] > 0.9
        for n_bins in (6, 8, 10, 15, 30):
            assert budgets[n_bins]["fixed_total_rank_d"] == 5
            assert budgets[n_bins]["fixed_total_rank_ds"] == 5

        rules = {row["key"]: row for row in scale["rules"]}
        assert set(rules) == {"d_rule", "ds_rule"}
        assert rules["d_rule"]["hardening_gap"] == 0.0
        assert abs(rules["ds_rule"]["hardening_gap"]) < 1e-4
        # The deployable profiled rule does not narrow the reported interval.
        assert (
            rules["ds_rule"]["downstream"]["mean_half_width"]
            > rules["d_rule"]["downstream"]["mean_half_width"]
        )
        for row in rules.values():
            assert row["downstream"]["converged_patients"] == 10
            assert row["downstream"]["total_patients"] == 10

        sweep = {row["population"]: row for row in scale["interest_sweep"]}
        assert list(sweep) == list(CLASS_NAMES[:-1])
        for row in sweep.values():
            assert row["ds_initialized_retention"] <= row["ceiling_retention"] + 1e-9
            assert row["ds_full_retention"] < 0.05


def test_profiled_figure_renders_from_fixture_scale_metrics() -> None:
    """The profiled-D_s figure exercises real metrics shape, never a mock."""
    import matplotlib.pyplot as plt

    inputs = profiled_inputs_from_data(
        load_fixture(FIXTURE),
        quick=True,
        score_max_per_patient_class=48,
        score_max_iter=6,
    )
    study = run_profiled_study(inputs, quick=True, n_bins=8, budgets=(5, 8), sweep_interest=True)
    figure = make_profiled_figure(study.metrics)
    try:
        assert len(figure.axes) == 2
    finally:
        plt.close(figure)
