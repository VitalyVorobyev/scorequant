"""Private nearest-centre assignment kernel shared by fitting, results and artifacts.

Leaf module: it depends only on ``_chunking`` and ``_execution``, so the
deployable :mod:`scorequant.artifact` layer, the fitting layers and the solvers
can all import it without pulling one another in. Every nearest-centre
assignment in the package, Euclidean or Mahalanobis, goes through
:func:`assign_nearest`.
"""

from __future__ import annotations

from ._chunking import chunk_rows_for
from ._execution import backend_array
from ._execution import xp as jnp


def _nearest(
    coordinates: jnp.ndarray, centers: jnp.ndarray, metric: jnp.ndarray | None
) -> jnp.ndarray:
    """Assign one chunk of rows to its nearest centre."""
    differences = coordinates[:, None, :] - centers[None, :, :]
    if metric is None:
        distances = jnp.sum(differences**2, axis=2)
    else:
        distances = jnp.einsum("nbr,rs,nbs->nb", differences, metric, differences)
    return jnp.argmin(distances, axis=1)


def assign_nearest(
    coordinates: jnp.ndarray, centers: jnp.ndarray, metric: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Assign every row to its nearest centre in memory-bounded chunks.

    ``metric=None`` is the Euclidean rule; a matrix gives the Mahalanobis one.
    Bit-identical to the unchunked assignment: each row's distance and argmin
    are independent of every other row, so partitioning rows into chunks never
    materializes the full ``[n_rows, n_bins, rank]`` tensor and changes nothing
    about the arithmetic. Eager only: inside traced code the chunk loop would
    unroll into an equivalent, larger graph.

    Public results store canonical NumPy arrays, so the centres and metric are
    placed on the active backend once here. Without it the broadcast difference
    and its square would be evaluated by NumPy on every chunk even under the JAX
    backend, materializing host temporaries the selected runtime should own.
    """
    coordinates = backend_array(coordinates)
    centers = backend_array(centers)
    if metric is not None:
        metric = backend_array(metric)
    n_rows = int(coordinates.shape[0])
    chunk_rows = chunk_rows_for(coordinates, int(centers.shape[0]))
    if chunk_rows >= n_rows:
        return _nearest(coordinates, centers, metric)
    return jnp.concatenate(
        [
            _nearest(coordinates[start : start + chunk_rows], centers, metric)
            for start in range(0, n_rows, chunk_rows)
        ]
    )
