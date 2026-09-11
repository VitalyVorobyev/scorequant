"""Shared weighted per-bin scatter-add statistics.

Every hard-label solver and diagnostic in ScoreQuant repeats the same three
steps on a fresh set of integer bin labels: scatter-add the row weights into
per-bin occupancy, scatter-add the weighted rows into per-bin sums, and divide
to get per-bin means. :func:`scatter_bin_statistics` is that one
implementation.

Empty-bin policy
-----------------
A bin with zero total weight divides its (also zero) sum by one instead of by
its own zero weight, so its reported mean is exactly ``0.0`` rather than
``NaN``: ``jnp.where(bin_weights > 0, bin_weights, 1)``. This is the
``where``-guard used everywhere in ScoreQuant that a hard-label mean can
legitimately be evaluated on a labeling with an empty declared cell (for
example, an externally supplied labeling passed to
:func:`scorequant.information.information_report`).

Callers for whom an empty cell must instead be a caller error (the exact
exchange engine, which must never propose a state with an unoccupied
requested cell) raise using the returned ``weights`` array themselves; the
guard still keeps the arithmetic that produces ``means`` well-defined before
that check runs, so the two concerns stay independent.

This is deliberately not used for the differentiable soft-responsibility path
in ``solvers.soft.soft_voronoi`` (``_soft_fisher``), which floors its occupancy
with ``jnp.maximum(occupancy, tiny)`` instead. That path's occupancy is a sum
of continuous softmax responsibilities that is never exactly zero but can be
arbitrarily small, and it is differentiated through by ``jax.grad``: flooring
keeps the gradient of the mean finite and well-scaled as the occupancy
shrinks, while the ``where``-guard's discontinuous branch does not have a
useful gradient at the switch point. The two guards solve different problems
and are not interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._execution import scatter_add
from ._execution import xp as jnp


@dataclass(frozen=True, slots=True)
class BinStatistics:
    """Per-bin weighted occupancy, value sums, and empty-bin-safe means."""

    weights: jnp.ndarray
    sums: jnp.ndarray
    means: jnp.ndarray


def scatter_bin_statistics(
    labels: jnp.ndarray,
    weights: jnp.ndarray,
    values: jnp.ndarray,
    n_bins: int,
) -> BinStatistics:
    """Accumulate weighted occupancy, value sums, and safe per-bin means.

    Parameters
    ----------
    labels
        Integer bin label for every row, with shape ``[N]`` and values in
        ``[0, n_bins)``.
    weights
        Nonnegative row weights with shape ``[N]``.
    values
        Row values to accumulate, with shape ``[N, D]``.
    n_bins
        Total number of bins, including bins no label selects.

    Returns
    -------
    BinStatistics
        ``weights`` and ``sums`` with shape ``[n_bins]`` and ``[n_bins, D]``,
        and ``means = sums / weights`` with the empty-bin policy documented on
        this module.
    """
    bin_weights = scatter_add(jnp.zeros(n_bins, dtype=weights.dtype), labels, weights)
    bin_sums = scatter_add(
        jnp.zeros((n_bins, values.shape[1]), dtype=values.dtype),
        labels,
        weights[:, None] * values,
    )
    safe_weights = jnp.where(bin_weights > 0, bin_weights, 1)
    means = bin_sums / safe_weights[:, None]
    return BinStatistics(weights=bin_weights, sums=bin_sums, means=means)
