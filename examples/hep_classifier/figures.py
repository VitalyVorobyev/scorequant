"""Committed figures for the HEP classifier showcase."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .experiment import MetricRow, Study

CELL_COLORS = (
    "#38618c",
    "#c0563c",
    "#4f9d69",
    "#8e6bb3",
    "#d59a2b",
    "#5aa9c9",
    "#7a7a7a",
    "#b5566f",
)


def _mapping(metrics: dict[str, object], key: str) -> dict[str, object]:
    value = metrics[key]
    if not isinstance(value, dict):
        raise TypeError(f"metrics[{key!r}] must be a mapping")
    return value


def _rows(metrics: dict[str, object], key: str) -> list[dict[str, object]]:
    value = metrics[key]
    if not isinstance(value, list):
        raise TypeError(f"metrics[{key!r}] must be a list")
    return [row for row in value if isinstance(row, dict)]


def _logit(posterior: np.ndarray) -> np.ndarray:
    clipped = np.clip(posterior, 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def _draw_cells(
    axis: Axes,
    study: Study,
    labels: np.ndarray,
    occupancy: list[MetricRow],
    *,
    title: str,
) -> None:
    logit = _logit(study.table.posterior)
    tes_score = study.table.scores[:, 2]
    order = np.argsort(labels, kind="stable")
    n_cells = int(np.max(labels)) + 1
    for cell in range(n_cells):
        mask = labels[order] == cell
        axis.scatter(
            logit[order][mask],
            tes_score[order][mask],
            s=9,
            alpha=0.75,
            linewidths=0,
            color=CELL_COLORS[cell % len(CELL_COLORS)],
            label=f"cell {cell}",
        )
    for row in occupancy:
        mean = row["mean_tes_score"]
        share = row["signal_fraction"]
        if mean is None or share is None:
            continue
        cell = int(row["cell"])  # type: ignore[call-overload]
        axis.axhline(
            float(mean),  # type: ignore[arg-type]
            color=CELL_COLORS[cell % len(CELL_COLORS)],
            linewidth=0.8,
            linestyle=":",
            alpha=0.9,
        )
    axis.set(xlabel="signal posterior, log-odds", ylabel="tes score", title=title)
    axis.axhline(0.0, color="#333333", linewidth=0.6)
    text = "\n".join(
        f"cell {int(row['cell'])}: {int(row['events'])} events, "  # type: ignore[call-overload]
        f"signal share {float(row['signal_fraction']):.3f}, "  # type: ignore[arg-type]
        f"mean tes score {float(row['mean_tes_score']):+.2f}"  # type: ignore[arg-type]
        for row in occupancy
        if row["signal_fraction"] is not None
    )
    axis.text(
        0.01,
        0.99,
        text,
        transform=axis.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )


def make_cells_figure(study: Study) -> Figure:
    """Render the two six-cell rules in the plane the physics lives in.

    Left: equal-width cells of the classifier's log-odds -- vertical stripes,
    because the classifier output is one number and the `tes` direction is
    invisible to it. Right: the reusable profiled-D_s rule fitted on the
    same events, whose cells also cut along the `tes` score, which is what
    lets a change in the energy scale be told apart from a change in the
    signal strength. Dotted lines mark each cell's weighted mean `tes` score.

    Parameters
    ----------
    study
        The object returned by `examples.hep_classifier.experiment.run_study`.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.
    """
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), constrained_layout=True, sharey=True)
    _draw_cells(
        axes[0],
        study,
        study.logit_labels,
        study.logit_occupancy,
        title="Classifier log-odds, equal-width cells",
    )
    _draw_cells(
        axes[1],
        study,
        study.ds_rule_labels,
        study.ds_rule_occupancy,
        title="Profiled $D_s$ rule, soft Voronoi in score space",
    )
    figure.suptitle(
        "Six cells, all 1,000 events, coloured by cell. "
        "Only the profiled rule resolves the tes direction."
    )
    return figure


def make_budget_figure(study: Study) -> Figure:
    """Render the bin-budget sweep: ceiling, in-sample partition, held-out rule, baseline.

    Parameters
    ----------
    study
        The object returned by `examples.hep_classifier.experiment.run_study`.

    Returns
    -------
    matplotlib.figure.Figure
        One panel of profiled retention against the bin budget.
    """
    metrics = study.metrics
    in_sample = _mapping(metrics, "in_sample")
    sweep = _rows(in_sample, "ceiling_sweep")
    held_out = _rows(_mapping(metrics, "cross_evaluation"), "held_out_budget_sweep")
    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    budgets = [float(row["n_bins"]) for row in sweep]
    axis.plot(
        budgets,
        [float(row["ceiling_retention"]) for row in sweep],
        marker="^",
        linestyle="--",
        color="#666666",
        label="certified ceiling, in sample",
    )
    axis.plot(
        budgets,
        [float(row["ds_profiled_retention"]) for row in sweep],
        marker="o",
        color="#38618c",
        label="profiled $D_s$ finite partition, in sample",
    )
    held_budgets = [float(row["n_bins"]) for row in held_out]
    means = [row["ds_rule_evaluation_profiled_retention"] for row in held_out]
    axis.plot(
        held_budgets,
        [np.nan if value is None else float(value) for value in means],  # type: ignore[arg-type]
        marker="D",
        color="#c0563c",
        label="profiled $D_s$ reusable rule, held out (mean of two directions)",
    )
    for row in held_out:
        directions = row["directions"]
        if isinstance(directions, list):
            for value in directions:
                if value is not None:
                    axis.plot(
                        [float(row["n_bins"])],
                        [float(value)],
                        marker="_",
                        color="#c0563c",
                        markersize=12,
                        linestyle="none",
                    )
    quantile = [row["classifier_quantile_profiled_retention"] for row in sweep]
    axis.plot(
        budgets,
        [np.nan if value is None else float(value) for value in quantile],  # type: ignore[arg-type]
        marker="s",
        color="#4f9d69",
        label="classifier quantile bins, in sample",
    )
    axis.set(
        xlabel="bin budget",
        ylabel="profiled retention of $\\mu_{h\\tau\\tau}$",
        xticks=budgets,
        ylim=(0.0, 1.02),
        title="Retention against the bin budget",
    )
    axis.legend(loc="lower right", fontsize=8.5)
    return figure
