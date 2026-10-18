"""Two toy problems behind the Ideas posts on the portal.

* A fringe pattern with the phase as its only parameter: equal detector segments
  against intervals of the score, at several bin budgets.
* A two-dimensional score law on which weighted k-means and D-optimality choose
  different cells for the same budget.

Both are small and deterministic. Run with::

    JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run python -m examples.ideas_toys

which writes `examples/outputs/ideas-toys.json` and two figures beside it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

import scorequant as sq
from examples.michelson_phase import COUNTER_COLORS, CURVE_COLOR, NEUTRAL_COLOR

OUTPUTS = Path("examples/outputs")
METRICS_PATH = OUTPUTS / "ideas-toys.json"
FRINGE_FIGURE = OUTPUTS / "ideas-score-bins.png"
CLUSTER_FIGURE = OUTPUTS / "ideas-kmeans-d.png"

EXECUTION = sq.ExecutionConfig(backend="numpy", precision="float64", device="cpu")

#: Fringe visibility and the number of fringes across the detector.
VISIBILITY = 0.6
N_FRINGES = 4
#: Detector grid nodes; each carries the probability of its interval.
N_NODES = 8_000
#: Bin budgets of the sweep, and the one the post quotes.
FRINGE_BUDGETS = (2, 3, 4, 5, 6, 7, 8)
FRINGE_HEADLINE = 4

#: Rows of the two-dimensional score law, its seed, and the bin budget.
CLUSTER_ROWS = 3_000
CLUSTER_SEED = 3
CLUSTER_BINS = 3


def fringe_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return detector nodes, their probabilities, and the phase score at zero phase."""
    u_max = 2.0 * np.pi * N_FRINGES
    step = u_max / N_NODES
    u = (np.arange(N_NODES) + 0.5) * step
    weights = (1.0 + VISIBILITY * np.cos(u)) / u_max * step
    scores = -VISIBILITY * np.sin(u) / (1.0 + VISIBILITY * np.cos(u))
    return u, weights, scores[:, None]


def equal_segments(u: np.ndarray, n_bins: int) -> np.ndarray:
    """Label each node by which of ``n_bins`` equal detector segments holds it."""
    return np.minimum((u / (2.0 * np.pi * N_FRINGES) * n_bins).astype(int), n_bins - 1)


def score_intervals(scores: np.ndarray, weights: np.ndarray, n_bins: int) -> np.ndarray:
    """Label each node by the globally optimal interval of the score."""
    rule = sq.fit_quantizer(
        sq.ScoreSample(scores, weights),
        n_bins=n_bins,
        criterion=sq.DOptimality(),
        config=sq.ScalarDPConfig(),
        execution=EXECUTION,
    )
    return np.asarray(rule.predict_scores(scores, execution=EXECUTION))


def retention(
    scores: np.ndarray, weights: np.ndarray | None, labels: np.ndarray, n_bins: int
) -> float:
    """Return the geometric-mean retention of one labeling."""
    report = sq.information_report(scores, labels, weights, n_bins=n_bins, execution=EXECUTION)
    # Rounded so that a retention of exactly zero, which equal segments of whole
    # fringes give, is written as zero rather than as platform-dependent noise.
    return round(float(report.geometric_mean_retention), 12)


def cluster_scores() -> np.ndarray:
    """Return scores with a two-cluster first coordinate and a normal second one."""
    rng = np.random.default_rng(CLUSTER_SEED)
    first = rng.choice([-1.0, 1.0], size=CLUSTER_ROWS) + 0.05 * rng.normal(size=CLUSTER_ROWS)
    return np.column_stack([first, rng.normal(size=CLUSTER_ROWS)])


def spectrum(scores: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    """Return the retention spectrum of one labeling and its two means."""
    report = sq.information_report(scores, labels, n_bins=CLUSTER_BINS, execution=EXECUTION)
    eigenvalues = sorted(float(value) for value in np.asarray(report.retained_eigenvalues))
    return {
        "retained_eigenvalues": eigenvalues,
        "arithmetic_mean_retention": float(report.arithmetic_mean_retention),
        "geometric_mean_retention": float(report.geometric_mean_retention),
    }


def run_study() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run both toys; return the metrics and the arrays the figures draw."""
    u, weights, scores = fringe_table()
    equal = [retention(scores, weights, equal_segments(u, k), k) for k in FRINGE_BUDGETS]
    score = [
        retention(scores, weights, score_intervals(scores, weights, k), k) for k in FRINGE_BUDGETS
    ]
    headline = FRINGE_BUDGETS.index(FRINGE_HEADLINE)

    cluster = cluster_scores()
    kmeans = sq.fit_quantizer(
        sq.ScoreSample(cluster),
        n_bins=CLUSTER_BINS,
        criterion=sq.NormalizedTrace(),
        config=sq.KMeansConfig(seed=1, solver_restarts=8),
        execution=EXECUTION,
    )
    kmeans_labels = np.asarray(kmeans.predict_scores(cluster, execution=EXECUTION))
    d_partition = sq.optimize_partition(
        cluster,
        n_bins=CLUSTER_BINS,
        criterion=sq.DOptimality(),
        config=sq.DExchangeConfig(seed=1, solver_restarts=8),
        execution=EXECUTION,
    )
    d_labels = np.asarray(d_partition.labels)

    metrics: dict[str, object] = {
        "fringe": {
            "visibility": VISIBILITY,
            "fringes": N_FRINGES,
            "budgets": list(FRINGE_BUDGETS),
            "equal_segments": equal,
            "score_intervals": score,
            "headline_bins": FRINGE_HEADLINE,
            "headline_equal": equal[headline],
            "headline_score": score[headline],
        },
        "clusters": {
            "rows": CLUSTER_ROWS,
            "bins": CLUSTER_BINS,
            "kmeans": spectrum(cluster, kmeans_labels),
            "d_optimal": spectrum(cluster, d_labels),
        },
    }
    arrays = {
        "u": u,
        "scores": scores[:, 0],
        "headline_labels": score_intervals(scores, weights, FRINGE_HEADLINE),
        "cluster": cluster,
        "kmeans_labels": kmeans_labels,
        "d_labels": d_labels,
    }
    return metrics, arrays


def fringe_figure(metrics: dict[str, object], arrays: dict[str, np.ndarray]) -> Figure:
    """Draw the score along the detector, cut into score cells, and the retention sweep."""
    fringe = metrics["fringe"]
    assert isinstance(fringe, dict)
    figure, (curve, sweep) = plt.subplots(
        1, 2, figsize=(11.0, 3.8), width_ratios=(1.7, 1.0), constrained_layout=True
    )
    u, scores, labels = arrays["u"], arrays["scores"], arrays["headline_labels"]
    for cell in range(FRINGE_HEADLINE):
        inside = np.where(labels == cell, scores, np.nan)
        curve.plot(u / np.pi, inside, color=COUNTER_COLORS[cell], linewidth=2.0)
    for fringe_edge in range(1, N_FRINGES):
        curve.axvline(2.0 * fringe_edge, color=NEUTRAL_COLOR, linewidth=0.8, linestyle="--")
    curve.axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.6)
    curve.set(
        xlabel="detector position u / π  (dashed: equal segments)",
        ylabel="phase score",
        title=f"{FRINGE_HEADLINE} score cells along the detector",
        xlim=(0.0, 2.0 * N_FRINGES),
    )
    budgets = fringe["budgets"]
    sweep.plot(
        budgets, fringe["score_intervals"], marker="o", color=CURVE_COLOR, label="score intervals"
    )
    sweep.plot(
        budgets,
        fringe["equal_segments"],
        marker="s",
        color=COUNTER_COLORS[2],
        label="equal segments",
    )
    sweep.set(xlabel="bins", ylabel="information retained", ylim=(-0.03, 1.03), title="Retention")
    sweep.legend(loc="lower right")
    return figure


def cluster_figure(metrics: dict[str, object], arrays: dict[str, np.ndarray]) -> Figure:
    """Draw the same scores cut by weighted k-means and by D-optimality."""
    clusters = metrics["clusters"]
    assert isinstance(clusters, dict)
    figure, panels = plt.subplots(1, 2, figsize=(9.0, 4.4), sharey=True, constrained_layout=True)
    cluster = arrays["cluster"]
    for axes, key, labels, title in (
        (panels[0], "kmeans", arrays["kmeans_labels"], "k-means (trace)"),
        (panels[1], "d_optimal", arrays["d_labels"], "D-optimal (log det)"),
    ):
        colors = [COUNTER_COLORS[int(label)] for label in labels]
        axes.scatter(cluster[:, 0], cluster[:, 1], c=colors, s=4, alpha=0.5, linewidths=0)
        for cell in range(CLUSTER_BINS):
            mean = cluster[labels == cell].mean(axis=0)
            axes.scatter(
                *mean, s=90, color=COUNTER_COLORS[cell], edgecolors="black", linewidths=1.2
            )
        low, high = clusters[key]["retained_eigenvalues"]
        axes.set(
            title=f"{title}: retention {low:.2f} and {high:.2f}", xlabel="score 1", xlim=(-1.8, 1.8)
        )
    panels[0].set_ylabel("score 2")
    return figure


def main() -> None:
    """Run both toys and write the committed JSON and figures."""
    metrics, arrays = run_study()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    for path, figure in (
        (FRINGE_FIGURE, fringe_figure(metrics, arrays)),
        (CLUSTER_FIGURE, cluster_figure(metrics, arrays)),
    ):
        figure.savefig(path, dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    main()
