"""Shared experiment and rendering code for all synthetic demonstrations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.cluster import KMeans

import scorequant
from examples._env import is_fast_mode
from examples.synthetic_problems import PROBLEMS, SyntheticDataset, SyntheticProblem


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    problem: SyntheticProblem
    kmeans: scorequant.results.QuantizerResult
    soft: scorequant.results.QuantizerResult
    metrics: dict[str, float | int | list[float]]


def _nearest(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def _equal_grid_labels(
    train_observations: np.ndarray, test_observations: np.ndarray, n_bins: int
) -> np.ndarray:
    dimension = train_observations.shape[1]
    if dimension == 1:
        edges = np.quantile(train_observations[:, 0], np.linspace(0, 1, n_bins + 1)[1:-1])
        return np.digitize(test_observations[:, 0], edges)
    side = round(n_bins ** (1 / dimension))
    if side**dimension != n_bins:
        raise ValueError("equal-grid baseline requires n_bins to be a perfect dimension power")
    labels = np.zeros(test_observations.shape[0], dtype=int)
    multiplier = 1
    for axis in range(dimension):
        edges = np.quantile(train_observations[:, axis], np.linspace(0, 1, side + 1)[1:-1])
        labels += multiplier * np.digitize(test_observations[:, axis], edges)
        multiplier *= side
    return labels


def _report_retention(dataset: SyntheticDataset, labels: np.ndarray, n_bins: int) -> float:
    return scorequant.information_report(
        dataset.scores, labels, dataset.weights, n_bins=n_bins
    ).geometric_mean_retention


def run_experiment(
    problem: SyntheticProblem,
    *,
    soft_steps: int = 300,
    n_random: int = 50,
) -> ExperimentResult:
    """Fit both algorithms and evaluate all declared baselines on untouched test data."""

    common = dict(
        n_bins=problem.n_bins,
        validation=scorequant.ScoreSample(problem.validation.scores, problem.validation.weights),
    )
    train_source = scorequant.ScoreSample(problem.train.scores, problem.train.weights)
    kmeans = scorequant.fit_quantizer(
        train_source,
        criterion=scorequant.NormalizedTrace(),
        config=scorequant.KMeansConfig(seed=42, solver_restarts=4),
        **common,
    )
    soft = scorequant.fit_quantizer(
        train_source,
        criterion=scorequant.DOptimality(),
        config=scorequant.SoftVoronoiConfig(
            seed=42,
            initializer_restarts=4,
            max_iter=soft_steps,
            record_every=max(soft_steps // 30, 1),
        ),
        # The optimization plot below draws the full train/validation hard
        # retention curve, so every recorded snapshot must be scored.
        diagnostics="full",
        **common,
    )
    kmeans_test = kmeans.evaluate_scores(problem.test.scores, problem.test.weights)
    soft_test = soft.evaluate_scores(problem.test.scores, problem.test.weights)

    observation_model = KMeans(n_clusters=problem.n_bins, n_init=4, random_state=42).fit(
        problem.train.observations,
        sample_weight=problem.train.weights,
    )
    observation_labels = observation_model.predict(problem.test.observations)
    observation_train_labels = observation_model.predict(problem.train.observations)
    equal_labels = _equal_grid_labels(
        problem.train.observations, problem.test.observations, problem.n_bins
    )
    equal_train_labels = _equal_grid_labels(
        problem.train.observations, problem.train.observations, problem.n_bins
    )
    random_retentions: list[float] = []
    random_train_retentions: list[float] = []
    rng = np.random.default_rng(2026)
    for _ in range(n_random):
        indices = rng.choice(problem.train.scores.shape[0], size=problem.n_bins, replace=False)
        centers = problem.train.scores[indices]
        random_retentions.append(
            _report_retention(problem.test, _nearest(problem.test.scores, centers), problem.n_bins)
        )
        random_train_retentions.append(
            _report_retention(
                problem.train, _nearest(problem.train.scores, centers), problem.n_bins
            )
        )

    metrics: dict[str, float | int | list[float]] = {
        "n_bins": problem.n_bins,
        "n_train": int(problem.train.scores.shape[0]),
        "n_test": int(problem.test.scores.shape[0]),
        "kmeans_train_retention": kmeans.train_report.geometric_mean_retention,
        "kmeans_test_retention": kmeans_test.geometric_mean_retention,
        "soft_train_retention": soft.train_report.geometric_mean_retention,
        "soft_test_retention": soft_test.geometric_mean_retention,
        "observation_kmeans_train_retention": _report_retention(
            problem.train, observation_train_labels, problem.n_bins
        ),
        "observation_kmeans_test_retention": _report_retention(
            problem.test, observation_labels, problem.n_bins
        ),
        "equal_grid_train_retention": _report_retention(
            problem.train, equal_train_labels, problem.n_bins
        ),
        "equal_grid_test_retention": _report_retention(problem.test, equal_labels, problem.n_bins),
        "random_test_retentions": random_retentions,
        "random_median_train_retention": float(np.median(random_train_retentions)),
        "random_median_test_retention": float(np.median(random_retentions)),
        "soft_validation_retention": soft.validation_report.geometric_mean_retention,
    }
    return ExperimentResult(problem, kmeans, soft, metrics)


def make_example_figure(experiment: ExperimentResult) -> Figure:
    """Render the original-domain result, optimization, and final diagnostics."""

    problem = experiment.problem
    labels = np.asarray(experiment.soft.predict_scores(problem.test.scores))
    observations = problem.test.observations
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    if observations.shape[1] == 1:
        order = np.argsort(observations[:, 0])
        axes[0, 0].scatter(
            observations[order, 0],
            problem.test.scores[order, 0],
            c=labels[order],
            cmap="tab20",
            s=5,
            alpha=0.6,
        )
        axes[0, 0].set(xlabel="observation", ylabel="score coordinate 1")
    else:
        axes[0, 0].scatter(
            observations[:, 0],
            observations[:, 1],
            c=labels,
            cmap="tab20",
            s=5,
            alpha=0.55,
            linewidths=0,
        )
        axes[0, 0].set(xlabel="observation 1", ylabel="observation 2")
    axes[0, 0].set_title("Final bins in the original domain")

    trace = experiment.soft.trace
    axes[0, 1].plot(trace.steps, trace.soft_retention, label="soft objective")
    axes[0, 1].plot(trace.steps, trace.train_hard_retention, label="train hard")
    axes[0, 1].plot(trace.steps, trace.validation_hard_retention, label="validation hard")
    trace_values = np.concatenate(
        [
            np.asarray(trace.soft_retention),
            np.asarray(trace.train_hard_retention),
            np.asarray(trace.validation_hard_retention),
        ]
    )
    retention_floor = max(0.0, float(np.min(trace_values)) - 0.05)
    axes[0, 1].set(
        title="Optimization process",
        xlabel="step",
        ylabel="D-efficiency",
        ylim=(retention_floor, 1.01),
    )
    axes[0, 1].legend()

    names = ["equal grid", "observation k-means", "random median", "score k-means", "soft Voronoi"]
    values = [
        experiment.metrics["equal_grid_test_retention"],
        experiment.metrics["observation_kmeans_test_retention"],
        experiment.metrics["random_median_test_retention"],
        experiment.metrics["kmeans_test_retention"],
        experiment.metrics["soft_test_retention"],
    ]
    bars = axes[1, 0].barh(names, values)
    baseline_floor = max(0.0, float(min(values)) - 0.05)
    axes[1, 0].set(
        title="Untouched test-set retention",
        xlim=(baseline_floor, 1.01),
        xlabel="D-efficiency",
    )
    axes[1, 0].bar_label(bars, fmt="%.3f", padding=3)

    report = experiment.soft.evaluate_scores(problem.test.scores, problem.test.weights)
    matrix = np.asarray(report.retained_matrix)
    image = axes[1, 1].imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if abs(matrix[row, column]) > 0.55 else "black"
            axes[1, 1].text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color=color,
            )
    axes[1, 1].set_title("Normalized retained Fisher matrix")
    axes[1, 1].set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xlabel="informative direction",
        ylabel="informative direction",
    )
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046)
    figure.suptitle(problem.title)
    return figure


def run_and_save(
    problem_name: str, output_dir: Path, *, quick: bool | None = None
) -> ExperimentResult:
    """Run one named experiment and save its figure and summary JSON.

    Parameters
    ----------
    problem_name
        Key into `PROBLEMS`.
    output_dir
        Directory that receives the rendered figure and summary JSON.
    quick
        Use short optimizer settings. Defaults to `is_fast_mode()`, honoring
        the `SCOREQUANT_EXAMPLE_FAST` environment variable, so callers only
        need to pass an explicit value when overriding that default.
    """
    quick = is_fast_mode() if quick is None else quick
    problem = PROBLEMS[problem_name]()
    experiment = run_experiment(
        problem, soft_steps=80 if quick else 300, n_random=10 if quick else 50
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = make_example_figure(experiment)
    figure.savefig(output_dir / f"{problem.name}.png", dpi=160)
    plt.close(figure)
    with (output_dir / f"{problem.name}.json").open("w", encoding="utf-8") as stream:
        json.dump({"problem": problem.name, **experiment.metrics}, stream, indent=2)
    return experiment
