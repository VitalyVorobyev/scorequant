"""Exact reference implementations independent of the library's optimized paths.

These oracles deliberately trade performance for mathematical transparency:
brute-force enumeration and closed-form single-move recomputation. They exist
only to certify the production D-exchange and profiled-D-exchange solvers in
``scorequant.partition`` against ground truth, and are therefore kept out of
``src/`` rather than shipped as part of the library.
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from scorequant._typing import ArrayLike
from scorequant._validation import collapse_duplicate_scores, validate_n_bins, validate_sample
from scorequant.partition import _cell_statistics, _DObjective
from scorequant.transforms import fisher_transform


def _exact_d_move_gain(
    information_inverse: ArrayLike,
    source_residual: ArrayLike,
    destination_residual: ArrayLike,
    *,
    point_weight: float,
    source_weight: float,
    destination_weight: float,
) -> float:
    """Return the exact log-determinant gain of one admissible relocation."""
    if point_weight <= 0:
        raise ValueError("point_weight must be positive")
    if source_weight <= point_weight:
        raise ValueError("a relocation cannot empty its source cell")
    if destination_weight <= 0:
        raise ValueError("destination_weight must be positive")
    inverse = jnp.asarray(information_inverse)
    source = jnp.asarray(source_residual, dtype=inverse.dtype)
    destination = jnp.asarray(destination_residual, dtype=inverse.dtype)
    alpha = point_weight * source_weight / (source_weight - point_weight)
    beta = point_weight * destination_weight / (destination_weight + point_weight)
    q_source = float(np.asarray(source @ inverse @ source))
    q_destination = float(np.asarray(destination @ inverse @ destination))
    q_cross = float(np.asarray(source @ inverse @ destination))
    determinant_ratio = (1 + alpha * q_source) * (
        1 - beta * q_destination
    ) + alpha * beta * q_cross**2
    return float(np.log(determinant_ratio)) if determinant_ratio > 0 else float("-inf")


def _exact_profiled_d_move_gain(
    information_inverse: ArrayLike,
    nuisance_inverse: ArrayLike,
    source_residual: ArrayLike,
    destination_residual: ArrayLike,
    *,
    nuisance: tuple[int, ...],
    point_weight: float,
    source_weight: float,
    destination_weight: float,
) -> float:
    """Return the exact same-label profiled-D gain of one relocation."""
    full_gain = _exact_d_move_gain(
        information_inverse,
        source_residual,
        destination_residual,
        point_weight=point_weight,
        source_weight=source_weight,
        destination_weight=destination_weight,
    )
    source = jnp.asarray(source_residual)
    destination = jnp.asarray(destination_residual)
    indices = jnp.asarray(nuisance)
    nuisance_gain = _exact_d_move_gain(
        nuisance_inverse,
        source[indices],
        destination[indices],
        point_weight=point_weight,
        source_weight=source_weight,
        destination_weight=destination_weight,
    )
    return full_gain - nuisance_gain


def _bin_information(
    coordinates: jnp.ndarray, weights: jnp.ndarray, labels: jnp.ndarray, n_bins: int
) -> jnp.ndarray:
    """Return the symmetrized weighted binned information matrix of one labeling."""
    cell_weights = jnp.zeros(n_bins, dtype=weights.dtype).at[labels].add(weights)
    if bool(np.asarray(jnp.any(cell_weights <= 0))):
        raise ValueError("oracle requires exactly n_bins nonempty cells")
    sums = jnp.zeros((n_bins, coordinates.shape[1]), dtype=coordinates.dtype)
    sums = sums.at[labels].add(weights[:, None] * coordinates)
    means = sums / cell_weights[:, None]
    information = jnp.einsum("b,bp,bq->pq", cell_weights, means, means)
    return 0.5 * (information + information.T)


def _exhaustive_d_oracle(
    scores: ArrayLike,
    weights: ArrayLike | None,
    n_bins: int,
    *,
    rank_rtol: float | None = None,
    also_trace: bool = False,
) -> tuple[jnp.ndarray, float] | tuple[jnp.ndarray, float, jnp.ndarray, float]:
    """Return the globally D-optimal labels for a tiny regression instance.

    This deliberately private oracle is exponentially expensive and exists for
    mathematical regression tests, not as a production solver.

    Parameters
    ----------
    scores, weights, n_bins, rank_rtol
        Same contract as ``optimize_partition``.
    also_trace
        When ``True``, additionally enumerate the trace-optimal (normalized
        Fisher-whitened trace criterion) partition and return it alongside
        the D-optimal one, mirroring ``research/dopt_experiments.py``'s
        ``exhaustive_best``. The trace objective is ``trace(information)``
        evaluated in the same Fisher-whitened coordinates used for the D
        search, which is coordinate-free and equals ``trace(I_binned @
        I_full^{-1})`` in the raw score basis.

    Returns
    -------
    tuple
        ``(labels, objective)`` when ``also_trace`` is ``False``, otherwise
        ``(d_labels, d_objective, trace_labels, trace_objective)``.
    """
    sample = validate_sample(scores, weights)
    validate_n_bins(n_bins, sample.n_effective)
    effective_scores, effective_weights, _ = collapse_duplicate_scores(
        sample.effective_scores, sample.effective_weights
    )
    if n_bins > effective_scores.shape[0]:
        raise ValueError("n_bins exceeds distinct positive-weight score rows")
    if effective_scores.shape[0] > 12:
        raise ValueError("the exhaustive D oracle is limited to 12 positive-weight rows")
    full_information = jnp.einsum(
        "n,np,nq->pq", effective_weights, effective_scores, effective_scores
    )
    transform = fisher_transform(full_information, whiten=True, rank_rtol=rank_rtol)
    coordinates = transform.apply(effective_scores)
    best_labels: jnp.ndarray | None = None
    best_objective = float("-inf")
    best_trace_labels: jnp.ndarray | None = None
    best_trace_objective = float("-inf")
    for candidate in _restricted_growth_partitions(effective_scores.shape[0], n_bins):
        labels = jnp.asarray(candidate)
        try:
            objective = (
                _DObjective()
                .init_state(_cell_statistics(coordinates, effective_weights, labels, n_bins))
                .objective
            )
        except ValueError:
            objective = None
        if objective is not None and objective > best_objective:
            best_labels = labels
            best_objective = objective
        if also_trace:
            try:
                information = _bin_information(coordinates, effective_weights, labels, n_bins)
            except ValueError:
                continue
            trace_objective = float(np.asarray(jnp.trace(information)))
            if trace_objective > best_trace_objective:
                best_trace_labels = labels
                best_trace_objective = trace_objective
    if best_labels is None:
        raise ValueError("no nonsingular D partition exists for this instance")
    if not also_trace:
        return best_labels, best_objective
    if best_trace_labels is None:
        raise ValueError("no nonempty-cell partition exists for this instance")
    return best_labels, best_objective, best_trace_labels, best_trace_objective


def _restricted_growth_partitions(n_rows: int, n_bins: int) -> Iterator[tuple[int, ...]]:
    labels = [0] * n_rows

    def visit(position: int, maximum: int) -> Iterator[tuple[int, ...]]:
        if position == n_rows:
            if maximum == n_bins - 1:
                yield tuple(labels)
            return
        for label in range(min(maximum + 1, n_bins - 1) + 1):
            labels[position] = label
            yield from visit(position + 1, max(maximum, label))

    yield from visit(1, 0)


def _o7_exact_plugin(
    scores: list[list[Fraction]], labels: list[int], weights: list[Fraction], n_bins: int
) -> tuple[Fraction, list[Fraction], list[list[Fraction]], list[list[Fraction]]]:
    """Exact O7 plug-in on a weighted atom sample in d = 2.

    Returns the determinant ratio r = eta_D^2, the influence values of r
    (psi_r = 2 eta_D psi), V and I_Z. The 2x2 inverse is explicit so the test
    carries no linear-algebra helper of its own.
    """
    d = 2
    p = [sum(w for w, z in zip(weights, labels, strict=True) if z == b) for b in range(n_bins)]
    m = [
        [
            sum(w * s[i] for s, w, z in zip(scores, weights, labels, strict=True) if z == b)
            for i in range(d)
        ]
        for b in range(n_bins)
    ]
    v = [
        [sum(w * s[i] * s[j] for s, w in zip(scores, weights, strict=True)) for j in range(d)]
        for i in range(d)
    ]
    i_z = [
        [sum(m[b][i] * m[b][j] / p[b] for b in range(n_bins) if p[b]) for j in range(d)]
        for i in range(d)
    ]

    def det(a: list[list[Fraction]]) -> Fraction:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]

    def inv(a: list[list[Fraction]]) -> list[list[Fraction]]:
        dd = det(a)
        return [[a[1][1] / dd, -a[0][1] / dd], [-a[1][0] / dd, a[0][0] / dd]]

    def quad(a: list[list[Fraction]], x: list[Fraction], y: list[Fraction]) -> Fraction:
        return sum(x[i] * a[i][j] * y[j] for i in range(d) for j in range(d))

    ratio = det(i_z) / det(v)
    iz_inv, v_inv = inv(i_z), inv(v)
    psi_r = []
    for s, z in zip(scores, labels, strict=True):
        c = [x / p[z] for x in m[z]]
        psi_r.append(ratio * (2 * quad(iz_inv, s, c) - quad(iz_inv, c, c) - quad(v_inv, s, s)))
    return ratio, psi_r, v, i_z
