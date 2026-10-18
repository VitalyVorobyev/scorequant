"""Documentation rendering for the FlowCyt experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .data import CLASS_NAMES
from .experiment import ExperimentResult


def make_workflow_figure() -> Figure:
    """Illustrate the complete application-to-library-to-inference flow."""
    figure, axis = plt.subplots(figsize=(14, 3.2), constrained_layout=True)
    axis.set_xlim(-0.6, 5.6)
    axis.set_ylim(-0.8, 0.8)
    axis.axis("off")
    boxes = (
        ("Marker events", "12 measured\nchannels", "#e8f1f8"),
        ("Score model", "cross-fitted and\ncalibrated", "#e8f1f8"),
        ("Mixture scores", "five simplex\ndirections", "#f8eddc"),
        ("ScoreQuant", "learn a frozen\nscore quantizer", "#f8eddc"),
        ("Bin counts", "discard labels and\ncontinuous markers", "#e4f1e7"),
        ("Fractions", "fit the six-class\ncount likelihood", "#e4f1e7"),
    )
    for index, (title, detail, color) in enumerate(boxes):
        axis.text(
            index,
            0,
            f"{title}\n{detail}",
            ha="center",
            va="center",
            fontsize=11,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.65", "facecolor": color, "edgecolor": "#444444"},
        )
        if index:
            axis.annotate(
                "",
                xy=(index - 0.47, 0),
                xytext=(index - 0.73, 0),
                arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.5},
            )
    axis.text(
        2.5,
        -0.7,
        "application model                    generic ScoreQuant API                    "
        "downstream model",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )
    figure.suptitle("From cytometry events to a label-blind population estimate", fontsize=16)
    return figure


def make_diagnostics_figure(result: ExperimentResult) -> Figure:
    """Render real patient composition and out-of-fold reliability diagnostics."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    entries = result.metrics["dataset"]["patient_compositions"]
    patient_ids = np.asarray([int(entry["patient"]) for entry in entries])
    fractions = np.asarray([entry["fractions"] for entry in entries], dtype=np.float64)
    colors = ("#3b6fb6", "#e69f00", "#009e73", "#d55e00", "#8c6bb1")
    bottom = np.zeros(len(patient_ids))
    for class_index, (name, color) in enumerate(zip(CLASS_NAMES[:5], colors, strict=True)):
        axes[0].bar(
            patient_ids,
            fractions[:, class_index],
            bottom=bottom,
            width=0.82,
            label=name,
            color=color,
        )
        bottom += fractions[:, class_index]
    for entry in entries:
        if entry["split"] == "test":
            patient = int(entry["patient"])
            axes[0].axvspan(patient - 0.45, patient + 0.45, color="#cc79a7", alpha=0.10)
    axes[0].set(
        title="Real target-population composition",
        xlabel="patient (shaded bars are held out)",
        ylabel="fraction of all cells; ‘other’ omitted",
        xlim=(0.3, 30.7),
    )
    axes[0].legend(fontsize=8, ncols=2)

    calibration = result.metrics["calibration"]
    reliability = [
        entry for entry in calibration["reliability_bins"] if entry["accuracy"] is not None
    ]
    confidence = np.asarray([entry["mean_confidence"] for entry in reliability], dtype=float)
    accuracy = np.asarray([entry["accuracy"] for entry in reliability], dtype=float)
    mass = np.asarray([entry["weight"] for entry in reliability], dtype=float)
    axes[1].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axes[1].plot(confidence, accuracy, color="#3b6fb6", linewidth=1.5)
    axes[1].scatter(
        confidence,
        accuracy,
        s=30 + 500 * mass,
        color="#3b6fb6",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.7,
    )
    axes[1].set(
        title="Patient-cross-fitted classifier reliability",
        xlabel="mean calibrated confidence",
        ylabel="balanced empirical accuracy",
        xlim=(0.3, 1.01),
        ylim=(0.0, 1.01),
    )
    axes[1].text(
        0.34,
        0.91,
        f"balanced accuracy = {float(calibration['balanced_accuracy']):.3f}\n"
        f"ECE = {float(calibration['expected_calibration_error']):.3f}",
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )
    figure.suptitle("FlowCyt data and score-model diagnostics")
    return figure


def make_figure(result: ExperimentResult) -> Figure:
    """Render compression, inference, and learned-gate diagnostics."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {
        "score_kmeans": "#3b6fb6",
        "soft_voronoi": "#d55e00",
        "marker_kmeans": "#009e73",
        "one_dimensional_score": "#8c6bb1",
        "two_dimensional_grid": "#777777",
    }
    for method, color in colors.items():
        rmse = [
            float(result.metrics[f"{method}:{n_bins}"]["target_macro_rmse"])
            for n_bins in result.bin_counts
        ]
        retention = [
            float(result.metrics[f"{method}:{n_bins}"]["held_out_d_efficiency"])
            for n_bins in result.bin_counts
        ]
        label = method.replace("_", " ")
        axes[0, 0].plot(result.bin_counts, rmse, marker="o", label=label, color=color)
        axes[0, 1].plot(result.bin_counts, retention, marker="o", label=label, color=color)
    random_rmse = [
        float(result.metrics[f"random_score_voronoi:{n_bins}"]["target_macro_rmse_median"])
        for n_bins in result.bin_counts
    ]
    random_retention = [
        float(result.metrics[f"random_score_voronoi:{n_bins}"]["held_out_d_efficiency_median"])
        for n_bins in result.bin_counts
    ]
    axes[0, 0].plot(
        result.bin_counts,
        random_rmse,
        color="#cc79a7",
        linestyle=":",
        marker="o",
        label="random score Voronoi",
    )
    axes[0, 1].plot(
        result.bin_counts,
        random_retention,
        color="#cc79a7",
        linestyle=":",
        marker="o",
        label="random score Voronoi",
    )
    finite_key = f"finite_d_exchange:{result.operating_n_bins}"
    if finite_key in result.metrics:
        finite = cast(dict[str, object], result.metrics[finite_key])
        axes[0, 0].scatter(
            [result.operating_n_bins],
            [float(cast(float, finite["target_macro_rmse"]))],
            marker="*",
            s=120,
            color="#e69f00",
            label="finite D exchange + compiled rule",
            zorder=5,
        )
        axes[0, 1].scatter(
            [result.operating_n_bins],
            [float(cast(float, finite["held_out_d_efficiency"]))],
            marker="*",
            s=120,
            color="#e69f00",
            label="finite D exchange + compiled rule",
            zorder=5,
        )
    unbinned_rmse = float(result.metrics["unbinned_classifier_ratio"]["target_macro_rmse"])
    axes[0, 0].axhline(
        unbinned_rmse,
        color="black",
        linestyle="--",
        label="unbinned classifier ratio",
    )
    axes[0, 0].set(
        title="Held-out population error",
        xlabel="number of hard bins",
        ylabel="mean RMSE, five target fractions (log scale)",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(
        title="Held-out supplied-score information",
        xlabel="number of hard bins",
        ylabel="D-efficiency",
        ylim=(0, 1.02),
    )
    axes[0, 1].legend(fontsize=8)

    final_key = f"soft_voronoi:{result.operating_n_bins}"
    predicted = result.predicted_fractions[final_key]
    for class_index, name in enumerate(CLASS_NAMES[:5]):
        axes[1, 0].scatter(
            result.true_fractions[:, class_index],
            predicted[:, class_index],
            label=name,
            s=35,
            alpha=0.8,
        )
    limit = max(float(np.max(result.true_fractions[:, :5])), float(np.max(predicted[:, :5])))
    axes[1, 0].plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(
        title=f"Fractions from {result.operating_n_bins} learned gates",
        xlabel="expert-label fraction",
        ylabel="estimated fraction",
    )
    axes[1, 0].legend(fontsize=8)

    composition = result.operating_bin_composition.T
    image = axes[1, 1].imshow(
        composition,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    axes[1, 1].set(
        title="Reference population composition of each gate",
        xlabel="hard bin",
        ylabel="population",
        xticks=np.arange(result.operating_n_bins),
        yticks=np.arange(len(CLASS_NAMES)),
        yticklabels=CLASS_NAMES,
    )
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046, label="P(population | bin)")
    figure.suptitle("FlowCyt population quantification")
    return figure


def make_uncertainty_figure(result: ExperimentResult) -> Figure:
    """Show reference-only interior agreement and boundary-hit rates."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    uncertainty = cast(dict[str, object], result.metrics["uncertainty"])
    scenarios = cast(dict[str, object], uncertainty["scenarios"])
    markers = {"reference_like": "o", "mast_enriched": "x"}
    colors = plt.get_cmap("tab10")
    plotted_values: list[float] = []
    plotted_classes: set[str] = set()
    for scenario_name, marker in markers.items():
        scenario = cast(dict[str, object], scenarios[scenario_name])
        classes = cast(dict[str, object], scenario["classes"])
        for class_index, name in enumerate(CLASS_NAMES[:5]):
            values = cast(dict[str, object], classes[name])
            if values["status"] != "interior":
                continue
            empirical = 1_000 * float(cast(float, values["empirical_standard_deviation"]))
            predicted = 1_000 * float(cast(float, values["median_local_fisher_error"]))
            plotted_values.extend([empirical, predicted])
            axes[0].scatter(
                empirical,
                predicted,
                color=colors(class_index),
                marker=marker,
                s=45,
                label=name if name not in plotted_classes else None,
            )
            plotted_classes.add(name)
    limit = max(plotted_values, default=1.0)
    axes[0].plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    axes[0].set(
        title="Interior pseudoexperiments",
        xlabel="empirical standard deviation (×10⁻³)",
        ylabel="median local Fisher error (×10⁻³)",
    )
    axes[0].legend(fontsize=8)

    positions = np.arange(5)
    width = 0.36
    for scenario_index, scenario_name in enumerate(markers):
        scenario = cast(dict[str, object], scenarios[scenario_name])
        classes = cast(dict[str, object], scenario["classes"])
        boundary = [
            float(cast(float, cast(dict[str, object], classes[name])["boundary_hit_fraction"]))
            for name in CLASS_NAMES[:5]
        ]
        axes[1].bar(
            positions + (scenario_index - 0.5) * width,
            boundary,
            width,
            label=scenario_name.replace("_", " "),
        )
    axes[1].set(
        title="Constrained estimates at the boundary",
        ylabel="boundary-hit fraction",
        xticks=positions,
        xticklabels=CLASS_NAMES[:5],
        ylim=(0, 1),
    )
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(fontsize=8)
    uncertainty_n_bins = int(cast(int, uncertainty["n_bins"]))
    figure.suptitle(f"Reference-only uncertainty coverage from {uncertainty_n_bins} learned gates")
    return figure


def make_profiled_figure(metrics: dict[str, object]) -> Figure:
    """Render the profiled-\\(D_s\\) bin-budget sweep and the per-fraction gain.

    Parameters
    ----------
    metrics
        One scale's metrics from `examples.cell_population.profiled.run_profiled_study`.

    Returns
    -------
    matplotlib.figure.Figure
        Left: plain D, the better of the two profiled seedings, and the
        certified ceiling against the bin budget. Right: the best-of-two
        profiled gain over plain D for every declared fraction of interest.
    """
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.0), constrained_layout=True)
    budgets = cast(list[dict[str, object]], metrics["budget_sweep"])
    bins = [int(cast(int, row["n_bins"])) for row in budgets]
    plain = [float(cast(float, row["d_profiled_retention"])) for row in budgets]
    best_ds = [
        max(
            float(cast(float, row["ds_seeded_retention"])),
            float(cast(float, row["ds_initialized_retention"])),
        )
        for row in budgets
    ]
    ceiling = [float(cast(float, row["ceiling_retention"])) for row in budgets]
    axes[0].plot(bins, plain, marker="o", color="#3b6fb6", label="Plain D")
    axes[0].plot(bins, best_ds, marker="s", color="#d55e00", label="Profiled $D_s$ (best seeding)")
    axes[0].plot(
        bins, ceiling, marker="^", color="#777777", linestyle="--", label="Certified ceiling"
    )
    axes[0].set(
        xlabel="bin budget",
        ylabel="profiled retention of the interest fraction",
        xticks=bins,
        title="Plain D vs profiled $D_s$ against the certified ceiling",
    )
    axes[0].legend(fontsize=8)

    sweep = cast(list[dict[str, object]], metrics["interest_sweep"])
    names = [str(row["population"]) for row in sweep]
    gains = [100 * float(cast(float, row["gain"])) for row in sweep]
    palette = ("#3b6fb6", "#e69f00", "#009e73", "#d55e00", "#8c6bb1")
    axes[1].bar(names, gains, color=palette[: len(names)])
    axes[1].set(
        ylabel="profiled $D_s$ gain over plain D (points)",
        title="Best-of-two profiled gain, every interest fraction",
    )
    axes[1].tick_params(axis="x", rotation=25)
    figure.suptitle("FlowCyt profiled $D_s$: real-data headroom is already nearly saturated")
    return figure


def write_profiled_figure(metrics: dict[str, object], path: Path) -> None:
    """Write the committed profiled-\\(D_s\\) figure for one scale's metrics."""
    figure = make_profiled_figure(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_outputs(result: ExperimentResult, output_dir: Path) -> None:
    """Write the committed figure and JSON evidence summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = make_figure(result)
    figure.savefig(output_dir / "cell_population.png", dpi=160)
    plt.close(figure)
    uncertainty = make_uncertainty_figure(result)
    uncertainty.savefig(output_dir / "cell_population_uncertainty.png", dpi=160)
    plt.close(uncertainty)
    workflow = make_workflow_figure()
    workflow.savefig(output_dir / "cell_population_workflow.png", dpi=160)
    plt.close(workflow)
    diagnostics = make_diagnostics_figure(result)
    diagnostics.savefig(output_dir / "cell_population_diagnostics.png", dpi=160)
    plt.close(diagnostics)
    with (output_dir / "cell_population.json").open("w", encoding="utf-8") as stream:
        json.dump(result.metrics, stream, indent=2)
        stream.write("\n")
