"""Shared geometric-quantizer state and distance kernels."""

from __future__ import annotations

from dataclasses import dataclass

from scorequant._execution import scatter_add
from scorequant._execution import xp as jnp


@dataclass(slots=True)
class QuantizerRun:
    """Aggregate state returned by a private quantizer implementation.

    ``objective_label`` names the units of ``objective_history`` so that a
    reported trace is never read in the wrong convention: a within-segment
    squared error and a log determinant are both scalars but not comparable.
    """

    centers: jnp.ndarray
    steps: list[int]
    center_history: list[jnp.ndarray]
    objective_history: list[float]
    bin_weight_history: list[jnp.ndarray]
    objective_label: str = "whitened_sse"
    soft_retention_history: list[float] | None = None
    temperature_history: list[float] | None = None
    gradient_norm_history: list[float] | None = None


def squared_distances(points: jnp.ndarray, centers: jnp.ndarray) -> jnp.ndarray:
    """Return the dense ``[N, B]`` squared Euclidean distance matrix."""
    return jnp.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def _bin_weights(labels: jnp.ndarray, weights: jnp.ndarray, n_bins: int) -> jnp.ndarray:
    return scatter_add(jnp.zeros(n_bins, dtype=weights.dtype), labels, weights)
