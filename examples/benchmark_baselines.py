"""Baselines the portal's benchmark page needs that the studies do not record.

The benchmark page sets ScoreQuant against equal-width, quantile and k-means
bins on each walkthrough problem, at that walkthrough's bin budget and in its
own metric. Most of those baselines are already in the studies' evidence under
``examples/outputs``. This script computes only the missing ones, on each
study's own score table, and writes them to
``examples/outputs/benchmark-baselines.json``:

- Michelson: quantile bins of the detector coordinate, each segment holding an
  equal share of the photons, and k-means on the two score coordinates; both
  scored by the profiled phase retention of the study's quadrature table.
- Density ratios: k-means on the estimated score of the best-trained
  classifier, scored by the retention its labels achieve against the exact
  score, exactly as the study scores its quantile and equal-width bins.
- HEP: k-means on the three score columns, fitted on one half of the events
  and scored on the other, in both directions of the study's split.

FlowCyt needs nothing new: its study already records k-means, a quantile cut
of the leading score direction and a marker grid at every budget, and its
classifier needs the full acquisition, which is not committed.

Every k-means run is plain Euclidean k-means with the table's own weights and
a fixed seed. Before writing, the script recomputes one baseline per problem
that the study did record and refuses to write if it disagrees, so the new
numbers are known to come from the same tables as the old ones.

Run with ``JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run python -m examples.benchmark_baselines``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

import scorequant as sq
from examples import door3_classifier as ratios
from examples import michelson_phase as michelson
from examples.hep_classifier import cross_fitted_table, load_fixture
from examples.hep_classifier.experiment import (
    HEADLINE_BINS,
    SOLVER_SEED,
    EventTable,
    event_split,
    quantile_rule,
    try_score_labeling,
)

OUTPUT_PATH = Path("examples/outputs/benchmark-baselines.json")
OUTPUTS = Path("examples/outputs")

#: Restarts of every k-means fit; the lowest weighted inertia is kept.
KMEANS_RESTARTS = 10


def weighted_kmeans(scores: np.ndarray, weights: np.ndarray, n_bins: int, *, seed: int) -> KMeans:
    """Fit Euclidean k-means on raw scores, each row counted with its weight.

    The weights are the table's own measure (quadrature mass or Monte Carlo
    weight), so the clusters see the same score law the fitted rules see; what
    k-means ignores is the Fisher metric, which is the point of the baseline.
    """
    model = KMeans(n_clusters=n_bins, n_init=KMEANS_RESTARTS, random_state=seed)
    return model.fit(scores, sample_weight=weights)


def _check(name: str, recomputed: float, committed: float) -> None:
    """Refuse to continue when a recomputed study baseline disagrees with its evidence.

    The tolerance admits float32 rounding, since a study may have been run
    without ``JAX_ENABLE_X64``; a different score table moves these numbers in
    the second or third digit.
    """
    if not np.isclose(recomputed, committed, rtol=1e-6, atol=1e-9):
        raise RuntimeError(
            f"{name}: recomputed {recomputed!r} but the committed evidence says {committed!r}; "
            "the score table no longer matches the study"
        )


def michelson_baselines() -> dict[str, object]:
    """Quantile and k-means bins on the Michelson study's quadrature table."""
    evidence = json.loads((OUTPUTS / "michelson-phase.json").read_text(encoding="utf-8"))
    n_bins = int(evidence["headline_bins"])
    row = next(item for item in evidence["sweep"] if item["n_bins"] == n_bins)
    sample = michelson.build_train_sample(
        michelson.build_provider(), n_nodes=int(evidence["n_nodes"])
    )
    _check(
        "michelson equal width",
        michelson.profiled_retention(
            sample, michelson.equal_width_labels(sample.observations, n_bins), n_bins
        ),
        row["equal_width_retention"],
    )

    # Equal photon probability per segment: cut the cumulative quadrature mass
    # at multiples of 1/K, assigning each node by the mass at its midpoint.
    mass = np.cumsum(sample.weights) - 0.5 * sample.weights
    mass /= sample.weights.sum()
    quantile_labels = np.minimum((mass * n_bins).astype(np.int64), n_bins - 1)

    kmeans_labels = weighted_kmeans(
        sample.scores, sample.weights, n_bins, seed=michelson.SEED
    ).labels_
    return {
        "n_bins": n_bins,
        "metric": "profiled phase retention on the quadrature table",
        "quantile_retention": michelson.profiled_retention(sample, quantile_labels, n_bins),
        "kmeans_retention": michelson.profiled_retention(sample, kmeans_labels, n_bins),
    }


def ratio_baselines() -> dict[str, object]:
    """K-means on the best-trained classifier's estimated score, scored against the exact one."""
    evidence = json.loads((OUTPUTS / "door3-classifier.json").read_text(encoding="utf-8"))
    n_bins = int(evidence["n_bins"])
    rung = len(evidence["ladder"]) - 1
    committed = evidence["naive_binning"]["rows"][rung]
    n_per_class = int(committed["n_per_class"])

    model = ratios.train_classifier(ratios.CLASSIFIER_SEED_BASE + rung, n_per_class)
    provider = ratios.classifier_provider(model, description="benchmark baseline")
    test_observations = ratios.draw_reference_mixture(ratios.TEST_SEED, int(evidence["n_test"]))
    estimated = np.asarray(provider.score(test_observations))
    exact = np.asarray(ratios.exact_provider().score(test_observations))

    naive = ratios.naive_binning_row(
        scores=estimated,
        oracle_test_scores=exact,
        n_bins=n_bins,
        true_retention=float(committed["true_retention"]),
        n_per_class=n_per_class,
    )
    _check("ratios equal width", naive.equal_width_retention, committed["equal_width_retention"])

    # Fitted on the held-out estimated scores themselves, as the study's
    # quantile and equal-width cells are, and every event counts once.
    labels = weighted_kmeans(
        estimated, np.ones(len(estimated)), n_bins, seed=ratios.SOLVER_SEED
    ).labels_
    achieved = sq.information_report(exact, labels, n_bins=n_bins).geometric_mean_retention
    return {
        "n_bins": n_bins,
        "n_per_class": n_per_class,
        "metric": "achieved retention against the exact score, held-out events",
        "kmeans_retention": float(achieved),
    }


def _held_out_profiled(evaluation: EventTable, labels: np.ndarray, n_bins: int) -> float:
    """Profiled signal-strength retention of `labels` on events no rule saw."""
    score = try_score_labeling(evaluation.scores, labels, evaluation.weights, n_bins=n_bins)
    if score is None:
        raise RuntimeError("a held-out HEP baseline left the profiled model unidentified")
    return score.profiled_retention


def hep_baselines() -> dict[str, object]:
    """K-means on the HEP score table, fitted on one half and scored on the other."""
    evidence = json.loads((OUTPUTS / "hep-classifier.json").read_text(encoding="utf-8"))
    n_bins = HEADLINE_BINS
    data = load_fixture()
    table = cross_fitted_table(data)
    halves = event_split(data.is_signal)
    split = [table.subset(halves == 0), table.subset(halves == 1)]

    quantile: list[float] = []
    kmeans: list[float] = []
    for reference, evaluation in (split, split[::-1]):
        rule = quantile_rule(reference, n_bins)
        quantile.append(_held_out_profiled(evaluation, rule.assign(evaluation), n_bins))
        model = weighted_kmeans(reference.scores, reference.weights, n_bins, seed=SOLVER_SEED)
        labels = model.predict(evaluation.scores)
        kmeans.append(_held_out_profiled(evaluation, labels, n_bins))
    _check(
        "hep quantile",
        float(np.mean(quantile)),
        evidence["cross_evaluation"]["mean"]["classifier_quantile"][
            "evaluation_profiled_retention"
        ],
    )
    return {
        "n_bins": n_bins,
        "metric": "held-out profiled signal-strength retention, mean of both directions",
        "kmeans_retention": float(np.mean(kmeans)),
        "kmeans_directions": [float(value) for value in kmeans],
    }


def run_study() -> dict[str, object]:
    """Compute every missing baseline and return the evidence record."""
    return {
        "problem": "benchmark_baselines",
        "kmeans_restarts": KMEANS_RESTARTS,
        "michelson": michelson_baselines(),
        "ratios": ratio_baselines(),
        "hep": hep_baselines(),
    }


def main() -> None:
    """Write the benchmark baselines to committed evidence."""
    record = run_study()
    OUTPUT_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
