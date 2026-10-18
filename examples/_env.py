"""Shared fast-mode switch for example scripts.

Every example script honors one environment variable so a
single switch shrinks every dataset and optimizer budget for CI and quick
local checks, while local, unset runs keep the full research-scale sizes.
"""

from __future__ import annotations

import os

FAST_MODE_ENV_VAR = "SCOREQUANT_EXAMPLE_FAST"


def is_fast_mode() -> bool:
    """Return whether ``SCOREQUANT_EXAMPLE_FAST`` requests reduced example sizes.

    Returns
    -------
    bool
        ``True`` if the environment variable is set to any non-empty value.
    """
    return bool(os.environ.get(FAST_MODE_ENV_VAR, ""))


def example_scale[T](full: T, fast: T) -> T:
    """Choose between a full-scale and a fast-mode value.

    Parameters
    ----------
    full
        Value used when fast mode is not requested.
    fast
        Value used when ``SCOREQUANT_EXAMPLE_FAST`` is set.

    Returns
    -------
    object
        `fast` under fast mode, otherwise `full`. The two arguments may be
        sizes, tuples of sizes, optimizer step counts, or any other value an
        example needs to shrink.
    """
    return fast if is_fast_mode() else full
