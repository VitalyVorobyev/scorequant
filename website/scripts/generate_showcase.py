"""Generate the FlowCyt showcase data the portal renders.

Two tiers, because they cost very different amounts. The narrative data --
marker histograms, patient compositions, the published method comparison, the
composition of the learned bins, the per-patient estimates -- is derived from
committed artifacts in seconds and is regenerated on every build. The
fixture-scale *walkthrough table* -- reference partition rows, template rows and
held-out rows with their patient ids, expert labels and ratios, so the
walkthrough page can run the whole reference-to-held-out chain in seconds --
requires fitting the cross-fitted classifier, so it is written once and reused
unless ``--force``.

Everything here reads committed inputs only. The 3.7 GB upstream FlowCyt data is
never required, and never ships.

The FlowCyt data is licensed CC BY-NC-SA 4.0, separately from ScoreQuant's MIT
license. Attribution and share-alike travel with every array this script emits;
see ``examples/data/README.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.cell_population.data import (  # noqa: E402
    CLASS_NAMES,
    FEATURE_NAMES,
    REFERENCE_PATIENTS,
    TEST_PATIENTS,
    RobustArcsinhTransform,
    load_fixture,
)

FIXTURE = REPO_ROOT / "examples" / "data" / "flowcyt_fixture.npz"
FIXTURE_FACTS = REPO_ROOT / "examples" / "data" / "flowcyt_fixture.json"
STUDY_METRICS = REPO_ROOT / "docs" / "usecases" / "assets" / "cell_population.json"
NARRATIVE_OUT = REPO_ROOT / "website" / "src" / "generated" / "showcase-data.json"
WALKTHROUGH_OUT = REPO_ROOT / "examples" / "data" / "flowcyt_walkthrough.npz"
WALKTHROUGH_FACTS_OUT = REPO_ROOT / "examples" / "data" / "flowcyt_walkthrough.json"

#: Marker histogram resolution. Thirty-two bins over the robust range keeps the
#: whole exploration payload under ~40 KB while still showing the bimodality
#: that separates the populations.
HISTOGRAM_BINS = 32

#: Bin budget the study reports as its operating point.
OPERATING_BINS = 8

#: Held-out cells kept per patient in the walkthrough table. The fixture holds
#: 2,048 per held-out patient; half of them keep the file small while leaving
#: every population, including the rarest, with enough cells for a count.
WALKTHROUGH_HELD_OUT_PER_PATIENT = 1_024

#: Score column names. The mixture score absorbs one component as the
#: simplex-dependent reference, so there are five columns for six populations.
SCORE_PARAMETERS = CLASS_NAMES[:-1]


def _round(values: object, digits: int = 6) -> object:
    """Round nested numeric data so the committed JSON has no float noise."""
    if isinstance(values, np.ndarray):
        return _round(values.tolist(), digits)
    if isinstance(values, list):
        return [_round(item, digits) for item in values]
    if isinstance(values, (float, np.floating)):
        return round(float(values), digits)
    if isinstance(values, (int, np.integer)):
        return int(values)
    return values


def _marker_histograms(features: np.ndarray, labels: np.ndarray) -> list[dict[str, object]]:
    """Bin every marker per population over one shared robust range.

    Cytometry intensities are heavily skewed -- on a linear axis every panel is
    one spike against the origin and shows nothing. These are binned on the same
    robust arcsinh scale the study's classifier is trained on, which is what
    makes the population structure visible at all. The range is the 0.5-99.5
    percentile so a handful of saturated events do not flatten it back.
    """
    transform = RobustArcsinhTransform.fit(features)
    scaled = transform.apply(features)
    panels: list[dict[str, object]] = []
    for column, name in enumerate(FEATURE_NAMES):
        values = scaled[:, column]
        low, high = np.percentile(values, [0.5, 99.5])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(values.min()), float(values.max()) + 1e-9
        edges = np.linspace(low, high, HISTOGRAM_BINS + 1)
        series = []
        for index, population in enumerate(CLASS_NAMES):
            counts, _ = np.histogram(values[labels == index], bins=edges)
            total = int(counts.sum())
            series.append(
                {
                    "population": population,
                    "density": _round(counts / total if total else counts.astype(float), 5),
                }
            )
        panels.append(
            {
                "marker": name,
                "edges": _round(edges, 4),
                "series": series,
            }
        )
    return panels


def _patient_compositions(metrics: dict[str, object]) -> list[dict[str, object]]:
    """Per-patient population fractions, which is what the study actually estimates.

    Taken from the study's record rather than counted in the committed fixture.
    The fixture deliberately samples a fixed number of cells per class for the
    reference patients, so counting it would plot the sampling design and label
    it biology.
    """
    dataset = metrics["dataset"]
    assert isinstance(dataset, dict)
    entries = dataset["patient_compositions"]
    assert isinstance(entries, list)
    rows: list[dict[str, object]] = []
    for entry in entries:
        assert isinstance(entry, dict)
        rows.append(
            {
                "patient": int(entry["patient"]),  # type: ignore[arg-type]
                "role": "reference" if entry["split"] == "reference" else "held-out",
                "fractions": _round(entry["fractions"], 5),
            }
        )
    return sorted(rows, key=lambda row: int(row["patient"]))  # type: ignore[arg-type]


def _method_comparison(metrics: dict[str, object]) -> dict[str, object]:
    """Reshape the published study metrics into a method x budget grid.

    These are the study's own numbers, not a re-run: the showcase reports what
    was measured on the full 600,000-cell sample rather than what the committed
    34,554-cell fixture would reproduce.
    """
    labels = {
        "finite_d_exchange": "D exchange (compiled)",
        "score_kmeans": "Normalized trace (k-means)",
        "soft_voronoi": "Soft Voronoi",
        "random_score_voronoi": "Random score Voronoi",
        "one_dimensional_score": "One score direction",
        "two_dimensional_grid": "Two-dimensional grid",
        "marker_kmeans": "k-means on raw markers",
    }
    grid: dict[str, list[dict[str, object]]] = {}
    budgets: set[int] = set()
    for key, payload in metrics.items():
        if ":" not in key or not isinstance(payload, dict):
            continue
        method, _, budget_text = key.partition(":")
        if method not in labels or "target_macro_rmse" not in payload:
            continue
        budget = int(budget_text)
        budgets.add(budget)
        grid.setdefault(method, []).append(
            {
                "bins": budget,
                "macroRmse": _round(payload["target_macro_rmse"]),
                "heldOutEfficiency": _round(payload.get("held_out_d_efficiency")),
                "minimumBinCount": payload.get("minimum_bin_count"),
            }
        )
    baseline = metrics["unbinned_classifier_ratio"]
    assert isinstance(baseline, dict)
    return {
        "budgets": sorted(budgets),
        "methods": [
            {
                "key": method,
                "label": labels[method],
                "isScoreQuant": method in ("finite_d_exchange", "score_kmeans", "soft_voronoi"),
                "points": sorted(points, key=lambda point: point["bins"]),
            }
            for method, points in sorted(grid.items())
        ],
        "unbinnedBaseline": {
            "label": "Unbinned classifier ratio",
            "macroRmse": _round(baseline["target_macro_rmse"]),
            "perClassRmse": _round(baseline["per_class_rmse"]),
        },
    }


def _bin_composition(metrics: dict[str, object]) -> dict[str, object]:
    """Return each learned bin's reference population composition, ``P(k | bin, theta0)``.

    This is the projection-free description of what the eight categories
    are: the study computes it on reference template rows from the soft
    Voronoi rule at the operating budget, and it is what lets a reader see
    the biological role of a category without projecting a five-dimensional
    partition onto a plane.
    """
    operating = metrics["operating_partition"]
    assert isinstance(operating, dict)
    matrix = np.asarray(operating["reference_bin_composition"], dtype=float)
    # Order bins by their dominant population so the table reads as a gate list.
    dominant = np.argmax(matrix, axis=1)
    order = np.lexsort((-matrix.max(axis=1), dominant))
    return {
        "bins": int(operating["n_bins"]),
        "populations": list(CLASS_NAMES),
        "rows": [
            {
                "bin": int(index),
                "dominant": CLASS_NAMES[int(dominant[index])],
                "composition": _round(matrix[index], 4),
            }
            for index in order
        ],
    }


def _patient_estimates(metrics: dict[str, object]) -> list[dict[str, object]]:
    """Held-out patients: expert composition against the binned and unbinned estimates."""
    entries = metrics["patients"]
    assert isinstance(entries, list)
    rows: list[dict[str, object]] = []
    for entry in entries:
        assert isinstance(entry, dict)
        rows.append(
            {
                "patient": int(entry["patient"]),  # type: ignore[arg-type]
                "expert": _round(entry["true_fractions"], 6),
                "binned": _round(entry["soft_voronoi_fractions"], 6),
                "unbinned": _round(entry["unbinned_classifier_ratio_fractions"], 6),
            }
        )
    return sorted(rows, key=lambda row: int(row["patient"]))  # type: ignore[arg-type]


def _headline(metrics: dict[str, object]) -> dict[str, object]:
    """Read the three numbers the showcase leads with from the study's own record."""
    operating = metrics[f"soft_voronoi:{OPERATING_BINS}"]
    baseline = metrics["unbinned_classifier_ratio"]
    assert isinstance(operating, dict) and isinstance(baseline, dict)
    return {
        "bins": OPERATING_BINS,
        "heldOutEfficiency": _round(operating["held_out_d_efficiency"]),
        "macroRmse": _round(operating["target_macro_rmse"]),
        "unbinnedMacroRmse": _round(baseline["target_macro_rmse"]),
    }


def build_narrative() -> dict[str, object]:
    """Assemble everything the showcase renders without running a solver."""
    data = load_fixture(FIXTURE)
    facts = json.loads(FIXTURE_FACTS.read_text())
    metrics = json.loads(STUDY_METRICS.read_text())
    upstream = facts.get("upstream_file_totals", {})
    return {
        "dataset": {
            "name": "FlowCyt",
            "citation": "Bini, Nassajian Mojarrad, Liarou, Matthes, Marchand-Maillet (CHIL 2024)",
            "repository": facts["source_repository"],
            "license": facts["source_license"],
            "licenseUrl": facts["license_url"],
            "patients": len(set(REFERENCE_PATIENTS) | set(TEST_PATIENTS)),
            "referencePatients": list(REFERENCE_PATIENTS),
            "heldOutPatients": list(TEST_PATIENTS),
            "markers": list(FEATURE_NAMES),
            "populations": list(CLASS_NAMES),
            "fixtureCells": int(data.features.shape[0]),
            "upstreamEvents": int(sum(int(value) for value in upstream.values())),
            "studyCells": 600_000,
        },
        "scoreSchema": {"parameters": list(SCORE_PARAMETERS)},
        "headline": _headline(metrics),
        "exploration": {
            "markerScale": "robust arcsinh (cofactor 150), the study's own transform",
            "markers": _marker_histograms(data.features, data.labels),
            "patients": _patient_compositions(metrics),
        },
        "comparison": _method_comparison(metrics),
        "binComposition": _bin_composition(metrics),
        "patientEstimates": _patient_estimates(metrics),
    }


def build_walkthrough_table() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Fit the cross-fitted score model and emit the fixture-scale walkthrough table.

    This is the expensive half. It reuses the study's own preparation rather
    than reimplementing the science, so the walkthrough page runs on the same
    score construction the published numbers came from, at fixture scale:

    * ``partition`` rows -- reference cells the quantizer is fitted on, with
      the study's patient-balanced integration weights;
    * ``template`` rows -- reference cells, disjoint from the partition rows,
      that estimate ``P(bin | population)`` once the rule is frozen;
    * ``held_out`` rows -- cells of the ten held-out patients, with the
      per-cell posterior/prior ratios the unbinned comparison needs.

    The sidecar also records what the walkthrough's own chain produces on
    this table -- the held-out macro RMSE of the binned and unbinned
    estimates and the held-out D-efficiency -- so the page can print the
    fixture-scale outcome next to the full study's without typing a number
    the reader cannot regenerate.

    Returns
    -------
    tuple
        The arrays to write as ``.npz`` and the sidecar facts as JSON.
    """
    from examples.cell_population.experiment import _prepare_experiment  # noqa: PLC0415

    data = load_fixture(FIXTURE)
    context = _prepare_experiment(data, quick=True, seed=2026)
    reference = context.reference
    test = context.test
    rng = np.random.default_rng(2026)
    keep = np.zeros(len(test.labels), dtype=bool)
    for patient in np.unique(test.patients):
        index = np.flatnonzero(test.patients == patient)
        take = min(WALKTHROUGH_HELD_OUT_PER_PATIENT, index.size)
        keep[np.sort(rng.choice(index, size=take, replace=False))] = True
    test_probabilities = context.score_fit.model.predict_proba(test.features)
    ratios = test_probabilities / context.score_fit.model.class_priors[None, :]

    partition = context.partition_mask
    template = context.template_mask
    chain = _walkthrough_chain(
        context.reference_scores[partition],
        context.weights,
        context.reference_scores[template],
        reference.labels[template],
        reference.patients[template],
        context.theta0,
        context.test_scores[keep],
        ratios[keep],
        test.labels[keep],
        test.patients[keep],
    )
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(list(SCORE_PARAMETERS)),
        "class_names": np.asarray(list(CLASS_NAMES)),
        "theta0": np.asarray(context.theta0, dtype=np.float64),
        "partition_scores": np.asarray(context.reference_scores[partition], dtype=np.float32),
        "partition_weights": np.asarray(context.weights, dtype=np.float64),
        "partition_labels": np.asarray(reference.labels[partition], dtype=np.int16),
        "partition_patients": np.asarray(reference.patients[partition], dtype=np.int16),
        "template_scores": np.asarray(context.reference_scores[template], dtype=np.float32),
        "template_labels": np.asarray(reference.labels[template], dtype=np.int16),
        "template_patients": np.asarray(reference.patients[template], dtype=np.int16),
        "held_out_scores": np.asarray(context.test_scores[keep], dtype=np.float32),
        "held_out_ratios": np.asarray(ratios[keep], dtype=np.float32),
        "held_out_labels": np.asarray(test.labels[keep], dtype=np.int16),
        "held_out_patients": np.asarray(test.patients[keep], dtype=np.int16),
    }
    facts: dict[str, object] = {
        "title": "FlowCyt walkthrough table, fixture scale",
        "source": "examples/data/flowcyt_fixture.npz through examples.cell_population",
        "license": "CC-BY-NC-SA-4.0",
        "seed": 2026,
        "quick": True,
        "reference_patients": [int(value) for value in np.unique(reference.patients)],
        "held_out_patients": [int(value) for value in np.unique(test.patients)],
        "held_out_per_patient": WALKTHROUGH_HELD_OUT_PER_PATIENT,
        "rows": {
            "partition": int(partition.sum()),
            "template": int(template.sum()),
            "held_out": int(keep.sum()),
        },
        "score_parameters": list(SCORE_PARAMETERS),
        "class_names": list(CLASS_NAMES),
        "theta0": _round(context.theta0, 6),
        "structure": _score_structure(
            np.asarray(context.reference_scores[partition]), np.asarray(reference.labels[partition])
        ),
        "chain": chain,
        "notes": (
            "Expert labels of held-out rows are evaluation-only: the walkthrough forms per-patient "
            "bin counts from predicted labels and compares the fitted fractions with the label "
            "composition. Held-out ratios are posterior/prior ratios of the frozen classifier."
        ),
    }
    return arrays, facts


def _walkthrough_chain(
    partition_scores: np.ndarray,
    partition_weights: np.ndarray,
    template_scores: np.ndarray,
    template_labels: np.ndarray,
    template_patients: np.ndarray,
    theta0: np.ndarray,
    held_out_scores: np.ndarray,
    held_out_ratios: np.ndarray,
    held_out_labels: np.ndarray,
    held_out_patients: np.ndarray,
) -> dict[str, object]:
    """Run the chain the walkthrough page shows, on the table it ships.

    Fit on the partition rows, freeze, estimate ``P(bin | population)`` on
    the template rows, label the held-out cells, count, fit the fractions,
    and compare with the unbinned ratio fit. The numbers are recorded so the
    page can say what its own code produces at fixture scale.
    """
    import scorequant as sq  # noqa: PLC0415
    from examples.cell_population import (  # noqa: PLC0415
        fit_binned_mixture,
        fit_unbinned_mixture,
    )
    from examples.cell_population.likelihood import estimate_bin_templates  # noqa: PLC0415

    n_bins = OPERATING_BINS
    reference = sq.ScoreSample(np.asarray(partition_scores, dtype=float), partition_weights)
    results = {
        "soft_voronoi": sq.fit_quantizer(
            reference,
            n_bins=n_bins,
            criterion=sq.DOptimality(),
            config=sq.SoftVoronoiConfig(seed=2026, initializer_restarts=4, max_steps=160),
        ),
        "d_exchange": sq.optimize_partition(
            reference,
            n_bins=n_bins,
            criterion=sq.DOptimality(),
            config=sq.DExchangeConfig(seed=2026, initializer_restarts=8),
        ).compile_quantizer(),
    }
    patients = np.unique(held_out_patients)
    expert = np.array(
        [
            np.bincount(held_out_labels[held_out_patients == patient], minlength=len(CLASS_NAMES))
            / np.count_nonzero(held_out_patients == patient)
            for patient in patients
        ]
    )

    def macro_rmse(estimate: np.ndarray) -> float:
        per_population = np.sqrt(np.mean((estimate - expert) ** 2, axis=0))
        return float(np.mean(per_population[: len(CLASS_NAMES) - 1]))

    unbinned = np.array(
        [
            fit_unbinned_mixture(
                np.asarray(held_out_ratios[held_out_patients == patient], dtype=float)
            ).fractions
            for patient in patients
        ]
    )
    chain: dict[str, object] = {
        "n_bins": n_bins,
        "held_out_patients": [int(patient) for patient in patients],
        "unbinned_macro_rmse": macro_rmse(unbinned),
    }
    for key, rule in results.items():
        template_bins = np.asarray(rule.predict_scores(np.asarray(template_scores, dtype=float)))
        templates = estimate_bin_templates(
            template_labels, template_bins, template_patients, n_bins=n_bins
        )
        held_out_bins = np.asarray(rule.predict_scores(np.asarray(held_out_scores, dtype=float)))
        counts = np.array(
            [
                np.bincount(held_out_bins[held_out_patients == patient], minlength=n_bins)
                for patient in patients
            ]
        )
        binned = np.array([fit_binned_mixture(row, templates).fractions for row in counts])
        report = sq.information_report(
            np.asarray(held_out_scores, dtype=float), held_out_bins, n_bins=n_bins
        )
        composition = templates * theta0[None, :]
        composition /= np.sum(composition, axis=1, keepdims=True)
        chain[key] = {
            "macro_rmse": macro_rmse(binned),
            "held_out_d_efficiency": float(report.geometric_mean_retention),
            "minimum_bin_count": int(np.min(np.sum(counts, axis=0))),
            "counts": counts.tolist(),
            "bin_composition": _round(composition, 4),
        }
    return chain


def _score_structure(scores: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    """Measure how concentrated the score cloud is.

    A confidently classified cell has a nearly fixed score, so the cloud is much
    closer to a handful of atoms than to a continuum. That is the reason a small
    bin budget can retain almost all the information, and it is worth reporting
    as a measurement rather than leaving a reader to wonder why the score-space
    plot looks sparse.
    """
    plane = np.round(scores[:, :2], 1)
    distinct = int(np.unique(plane, axis=0).shape[0])
    per_population = []
    for index, name in enumerate(CLASS_NAMES):
        selected = scores[labels == index]
        if selected.shape[0] == 0:
            continue
        per_population.append(
            {
                "population": name,
                "cells": int(selected.shape[0]),
                "meanFirstScore": _round(float(selected[:, 0].mean()), 3),
                "spread": _round(float(selected[:, 0].std()), 3),
            }
        )
    return {
        "distinctPlanePositions": distinct,
        "rows": int(scores.shape[0]),
        "perPopulation": per_population,
    }


def main() -> None:
    """Write both tiers, skipping the expensive one when it is already current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="rebuild the walkthrough table even if it exists"
    )
    arguments = parser.parse_args()

    NARRATIVE_OUT.parent.mkdir(parents=True, exist_ok=True)
    NARRATIVE_OUT.write_text(json.dumps(build_narrative(), indent=2) + "\n")
    print(f"wrote {NARRATIVE_OUT.relative_to(REPO_ROOT)}")

    if WALKTHROUGH_OUT.exists() and not arguments.force:
        print(f"kept {WALKTHROUGH_OUT.relative_to(REPO_ROOT)} (pass --force to rebuild)")
        return
    arrays, facts = build_walkthrough_table()
    np.savez_compressed(WALKTHROUGH_OUT, **arrays)
    WALKTHROUGH_FACTS_OUT.write_text(json.dumps(facts, indent=2) + "\n")
    print(f"wrote {WALKTHROUGH_OUT.relative_to(REPO_ROOT)} and its sidecar")


if __name__ == "__main__":
    main()
