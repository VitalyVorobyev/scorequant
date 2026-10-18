"""Optional Matplotlib views over ScoreQuant results. Not part of the stable API.

These figures are conveniences for looking at a fit. Their layout, styling and
signatures may change in any release, so code that has to keep working should
build its own figures from the documented result fields instead. Matplotlib is
imported only when a function is called; install the ``viz`` extra to use them.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np

from ._errors import ContractError
from ._typing import ArrayLike
from .results import InformationReport, OptimizationTrace, QuantizerResult

__all__ = ["plot_information", "plot_optimization", "plot_partition", "plot_summary"]

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _require_matplotlib() -> None:
    try:
        import_module("matplotlib.pyplot")
    except ImportError as error:  # pragma: no cover - exercised without the optional extra
        raise ImportError("plotting requires the `viz` optional dependency") from error


def plot_optimization(trace: OptimizationTrace) -> Figure:
    """Plot objective, hard retention, occupancies, and center motion.

    Parameters
    ----------
    trace
        Aggregate optimization history from a fitted result.

    Returns
    -------
    matplotlib.figure.Figure
        Four-panel optimization summary.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    steps = np.asarray(trace.steps)
    centers = np.asarray(trace.centers)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # The objective units differ by solver, so the legend names the convention.
    axes[0, 0].plot(steps, np.asarray(trace.objective), label=trace.objective_label)
    if trace.soft_retention is not None:
        axes[0, 0].plot(steps, np.asarray(trace.soft_retention), label="soft D-efficiency")
    axes[0, 0].set(title="Optimization", xlabel="step")
    axes[0, 0].legend()

    axes[0, 1].plot(steps, np.asarray(trace.train_hard_retention), label="train hard")
    if trace.validation_hard_retention is not None:
        axes[0, 1].plot(steps, np.asarray(trace.validation_hard_retention), label="validation hard")
    axes[0, 1].set(title="Final-partition metric during fitting", xlabel="step", ylim=(0, 1.03))
    axes[0, 1].legend()

    occupancies = np.asarray(trace.bin_weights)
    fractions = occupancies / np.maximum(occupancies.sum(axis=1, keepdims=True), 1e-300)
    axes[1, 0].plot(steps, fractions)
    axes[1, 0].set(title="Weighted bin occupancy", xlabel="step", ylabel="fraction")

    effective_rank = centers.shape[2]
    if effective_rank == 1:
        for bin_index in range(centers.shape[1]):
            axes[1, 1].plot(steps, centers[:, bin_index, 0], marker=".")
        axes[1, 1].set(xlabel="step", ylabel="optimization coordinate 1")
        axes[1, 1].set_title("Center trajectories")
    elif effective_rank == 2:
        for bin_index in range(centers.shape[1]):
            axes[1, 1].plot(
                centers[:, bin_index, 0], centers[:, bin_index, 1], marker=".", alpha=0.8
            )
        axes[1, 1].set(
            xlabel="optimization coordinate 1",
            ylabel="optimization coordinate 2",
        )
        axes[1, 1].set_title("Center trajectories")
    else:
        if len(steps) > 1:
            displacements = np.linalg.norm(np.diff(centers, axis=0), axis=2)
            axes[1, 1].plot(steps[1:], np.median(displacements, axis=1), marker=".", label="median")
            axes[1, 1].plot(steps[1:], np.max(displacements, axis=1), marker=".", label="maximum")
        axes[1, 1].set(
            title="Center displacement across all dimensions",
            xlabel="step",
            ylabel="Euclidean displacement",
        )
        axes[1, 1].legend()
    return figure


def plot_partition(
    result: QuantizerResult, scores: ArrayLike, weights: ArrayLike | None = None
) -> Figure:
    """Plot observations in the fitted informative coordinate system.

    Parameters
    ----------
    result
        Fitted score-space partition.
    scores
        Raw score matrix compatible with ``result``.
    weights
        Optional weights used only to scale marker sizes.

    Returns
    -------
    matplotlib.figure.Figure
        One- or two-dimensional partition view.

    Raises
    ------
    ContractError
        If the fitted informative space has rank above two. Use
        :func:`plot_summary` for a projection-free diagnostic instead.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    coordinates = np.asarray(result.transform.apply(scores))
    if coordinates.shape[1] > 2:
        raise ContractError(
            "plot_partition supports effective rank 1 or 2 only; "
            "use plot_summary for a projection-free diagnostic"
        )
    labels = np.asarray(result.predict_scores(scores))
    point_sizes = None
    if weights is not None:
        weight_array = np.asarray(weights)
        point_sizes = 8 + 24 * weight_array / max(float(np.max(weight_array)), 1e-300)
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if coordinates.shape[1] == 1:
        axis.scatter(coordinates[:, 0], labels, c=labels, s=point_sizes, cmap="tab20", alpha=0.6)
        axis.scatter(
            np.asarray(result.centers)[:, 0], np.arange(result.n_bins), c="black", marker="x"
        )
        axis.set(xlabel="informative coordinate 1", ylabel="hard bin")
    else:
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=labels,
            s=point_sizes,
            cmap="tab20",
            alpha=0.55,
            linewidths=0,
        )
        axis.scatter(
            np.asarray(result.centers)[:, 0],
            np.asarray(result.centers)[:, 1],
            c="black",
            marker="x",
            s=60,
        )
        axis.set(
            xlabel="informative coordinate 1",
            ylabel="informative coordinate 2",
        )
    axis.set_title("Final hard partition")
    return figure


def plot_information(report: InformationReport) -> Figure:
    """Plot retained information, its spectrum, and bin occupancy.

    Parameters
    ----------
    report
        Information report for one fixed partition and sample.

    Returns
    -------
    matplotlib.figure.Figure
        Matrix, eigenvalue, and weighted-occupancy panels. The matrix uses a
        signed scale so negative off-diagonal values remain visible.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    image = axes[0].imshow(np.asarray(report.retained_matrix), vmin=-1, vmax=1, cmap="coolwarm")
    axes[0].set_title("Normalized retained matrix")
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    eigenvalues = np.asarray(report.retained_eigenvalues)
    axes[1].bar(np.arange(1, len(eigenvalues) + 1), eigenvalues)
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Retained eigenvalues", xlabel="direction", ylim=(0, 1.05))
    weights = np.asarray(report.bin_weights)
    axes[2].bar(np.arange(len(weights)), weights / max(weights.sum(), 1e-300))
    axes[2].set(title="Weighted occupancy", xlabel="bin", ylabel="fraction")
    return figure


def plot_summary(
    result: QuantizerResult, scores: ArrayLike, weights: ArrayLike | None = None
) -> Figure:
    """Create a compact final-partition and optimization summary.

    Parameters
    ----------
    result
        Fitted score-space partition.
    scores
        Raw score matrix compatible with ``result``.
    weights
        Optional evaluation weights.

    Returns
    -------
    matplotlib.figure.Figure
        Partition, retained matrix, trace, and occupancy panels.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    coordinates = np.asarray(result.transform.apply(scores))
    labels = np.asarray(result.predict_scores(scores))
    report = result.evaluate_scores(scores, weights)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    if coordinates.shape[1] == 1:
        axes[0, 0].scatter(coordinates[:, 0], labels, c=labels, cmap="tab20", s=8, alpha=0.5)
        axes[0, 0].set(xlabel="informative coordinate 1", ylabel="hard bin")
        axes[0, 0].set_title("Final hard partition")
    elif coordinates.shape[1] == 2:
        axes[0, 0].scatter(
            coordinates[:, 0], coordinates[:, 1], c=labels, cmap="tab20", s=8, alpha=0.5
        )
        axes[0, 0].set(xlabel="informative coordinate 1", ylabel="informative coordinate 2")
        axes[0, 0].set_title("Final hard partition")
    else:
        eigenvalues = np.asarray(report.retained_eigenvalues)
        axes[0, 0].bar(np.arange(1, len(eigenvalues) + 1), eigenvalues)
        axes[0, 0].axhline(1, color="black", linestyle="--", linewidth=1)
        axes[0, 0].set(
            title="Retained-information spectrum",
            xlabel="informative direction",
            ylabel="retained eigenvalue",
            ylim=(0, 1.05),
        )
    image = axes[0, 1].imshow(np.asarray(report.retained_matrix), vmin=-1, vmax=1, cmap="coolwarm")
    axes[0, 1].set_title("Normalized retained information")
    figure.colorbar(image, ax=axes[0, 1], fraction=0.046)
    axes[1, 0].plot(
        np.asarray(result.trace.steps), np.asarray(result.trace.train_hard_retention), label="train"
    )
    if result.trace.validation_hard_retention is not None:
        axes[1, 0].plot(
            np.asarray(result.trace.steps),
            np.asarray(result.trace.validation_hard_retention),
            label="validation",
        )
    axes[1, 0].set(title="Hard D-efficiency", xlabel="step", ylim=(0, 1.03))
    axes[1, 0].legend()
    bin_weights = np.asarray(report.bin_weights)
    axes[1, 1].bar(np.arange(result.n_bins), bin_weights / max(bin_weights.sum(), 1e-300))
    axes[1, 1].set(title="Weighted occupancy", xlabel="bin", ylabel="fraction")
    return figure
