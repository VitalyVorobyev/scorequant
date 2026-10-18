"""Naive baselines used consistently across every example and study.

Every example that compares ScoreQuant's information-aware solvers to a
naive alternative uses one of these three functions, so results are
comparable across walkthroughs:

1. `rectangular_observation_bins` -- an equal-width grid in observation
   space, ignoring the score law entirely.
2. `euclidean_kmeans_scores` -- unweighted, unwhitened k-means on raw
   scores, ignoring the Fisher metric and event weights.
3. `equal_frequency_1d` -- equal-frequency (quantile) bins on one score
   coordinate or a stated one-dimensional projection.

These functions operate on plain NumPy arrays and never import
`scorequant`; retention of a baseline's labels is computed downstream with
`scorequant.information_report`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import KMeans


def rectangular_observation_bins(
    observations: np.ndarray,
    n_bins_per_axis: int | Sequence[int] | None = None,
    *,
    total_budget: int | None = None,
) -> np.ndarray:
    """Assign labels from an equal-width grid over each observation axis.

    This is the naive "bin the raw variables" baseline: it never looks at
    scores or weights, so it has no notion of which directions matter for
    inference.

    Parameters
    ----------
    observations
        Finite observation matrix with shape ``[N, D]``.
    n_bins_per_axis
        Either a single bin count applied to every axis, or one count per
        axis with length ``D``. Mutually exclusive with `total_budget`.
    total_budget
        Total requested cell budget. The per-axis bin count is
        ``round(total_budget ** (1 / D))``, applied uniformly across axes,
        so the realized cell count need not equal `total_budget` exactly.
        Mutually exclusive with `n_bins_per_axis`.

    Returns
    -------
    numpy.ndarray
        Integer labels with shape ``[N]``, dtype ``int64``. Grid cells with
        no observations are dropped and the surviving cells are renumbered
        densely from zero, so the maximum label can be smaller than the
        requested cell budget minus one.

    Raises
    ------
    ValueError
        If `observations` is empty or malformed, if neither or both of
        `n_bins_per_axis` and `total_budget` are given, or if a resolved
        per-axis bin count is not positive.
    """
    values = np.asarray(observations, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("observations must have non-empty shape [N, D]")
    if (n_bins_per_axis is None) == (total_budget is None):
        raise ValueError("give exactly one of n_bins_per_axis or total_budget")
    n_rows, n_dims = values.shape

    if total_budget is not None:
        if total_budget < 1:
            raise ValueError("total_budget must be positive")
        side = max(1, round(total_budget ** (1.0 / n_dims)))
        axis_bins = [side] * n_dims
    elif isinstance(n_bins_per_axis, int):
        axis_bins = [n_bins_per_axis] * n_dims
    else:
        axis_bins = list(n_bins_per_axis) if n_bins_per_axis is not None else []
        if len(axis_bins) != n_dims:
            raise ValueError(f"n_bins_per_axis must have length {n_dims}")
    if any(count < 1 for count in axis_bins):
        raise ValueError("every axis bin count must be positive")

    cell_index = np.zeros(n_rows, dtype=np.int64)
    multiplier = 1
    for axis, count in enumerate(axis_bins):
        column = values[:, axis]
        lower, upper = float(column.min()), float(column.max())
        if upper > lower:
            edges = np.linspace(lower, upper, count + 1)[1:-1]
            axis_labels = np.digitize(column, edges)
        else:
            axis_labels = np.zeros(n_rows, dtype=np.int64)
        cell_index += multiplier * axis_labels
        multiplier *= count

    _, labels = np.unique(cell_index, return_inverse=True)
    return labels.astype(np.int64)


def euclidean_kmeans_scores(
    scores: np.ndarray, n_bins: int, *, seed: int = 0, n_init: int = 10
) -> np.ndarray:
    """Cluster raw scores with unweighted, unwhitened Euclidean k-means.

    This is the naive "cluster the scores as if they were generic
    coordinates" baseline: it ignores event weights and the Fisher metric
    that makes some score directions more informative than others, in
    contrast to `scorequant.fit_quantizer`.

    Parameters
    ----------
    scores
        Finite score matrix with shape ``[N, P]``.
    n_bins
        Number of requested clusters.
    seed
        Seed for scikit-learn's `KMeans`, making the result deterministic.
    n_init
        Number of k-means restarts; the best inertia is kept.

    Returns
    -------
    numpy.ndarray
        Integer labels with shape ``[N]``, dtype ``int64``, in ``[0, n_bins)``.

    Raises
    ------
    ValueError
        If `scores` is empty or malformed, or `n_bins` is not positive.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("scores must have non-empty shape [N, P]")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    model = KMeans(n_clusters=n_bins, n_init=n_init, random_state=seed)
    labels = model.fit_predict(values)
    return labels.astype(np.int64)


def equal_frequency_1d(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign equal-frequency (quantile) labels along one coordinate.

    This is the naive "one informative axis, bin it by rank" baseline. Rows
    are ranked by `values` and split into `n_bins` contiguous rank blocks of
    as-equal-as-possible size, so no bin's mass differs from another's by
    more than one row.

    Parameters
    ----------
    values
        Finite one-dimensional array with shape ``[N]``.
    n_bins
        Requested number of equal-frequency bins.

    Returns
    -------
    numpy.ndarray
        Integer labels with shape ``[N]``, dtype ``int64``. Labels are
        renumbered densely from zero even if boundary duplicates collapse a
        requested bin to zero rows.

    Raises
    ------
    ValueError
        If `values` is empty, not one-dimensional, non-finite, or `n_bins`
        is not positive.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.shape[0] == 0:
        raise ValueError("values must have non-empty shape [N]")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")

    n = array.shape[0]
    order = np.argsort(array, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    raw_labels = (rank * n_bins) // n
    _, labels = np.unique(raw_labels, return_inverse=True)
    return labels.astype(np.int64)
