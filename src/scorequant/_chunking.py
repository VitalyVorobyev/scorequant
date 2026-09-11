"""Shared memory-bounded row-chunking budget for dense distance kernels.

Several private kernels evaluate one ``[chunk_rows, n_bins]`` (or the
Mahalanobis ``[chunk_rows, n_bins, rank]`` residual behind it) distance table
per row chunk instead of materializing the full ``[n_rows, n_bins, rank]``
tensor at once. ``assignment_chunk_rows`` is the one place that budget is
computed, so ``partition.py``, ``solvers/``, and ``result.py`` size their
chunks identically instead of drifting apart under independent edits.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# One chunk's dense temporaries (the base distance table plus its
# per-dimension residual and einsum working set) are held inside this many
# bytes, independent of total row count.
WORKING_SET_BYTES = 64 * 1024 * 1024


def assignment_chunk_rows(dtype: npt.DTypeLike, n_rows: int, n_bins: int, rank: int) -> int:
    """Return how many rows one memory-bounded assignment chunk holds.

    ``n_bins * (rank + 4) + 4 * rank`` accounts for the ``[chunk, n_bins]``
    distance table, the ``[chunk, n_bins, rank]`` residual and einsum
    temporaries a Mahalanobis assignment needs, and a fixed allowance for the
    Euclidean case's smaller working set; it is a deliberately generous
    single formula shared by every chunked assignment kernel rather than a
    per-kernel estimate.
    """
    item_size = np.dtype(dtype).itemsize
    values_per_row = n_bins * (rank + 4) + 4 * rank
    return max(1, min(n_rows, WORKING_SET_BYTES // max(item_size * values_per_row, 1)))
