"""Shared pytest configuration: worker thread pinning and role-based tiering.

Two concerns, both about how the suite is *run* rather than what it asserts.

Thread pinning. XLA sizes its CPU thread pool from the machine's core count, so
under `pytest-xdist` every worker independently tries to claim the whole host.
Even serially this suite spent 721s of system time against 824s of user time --
threads descheduling one another rather than computing -- and multiplying that
by the worker count made `-n 4` slower than no parallelism at all. Pinned, the
same tier runs in 106s with 12s of system time. Each worker therefore gets one
compute thread and the parallelism comes from xdist instead. This has to happen
before anything imports JAX, which is why it sits at conftest import time.

Tiering. The suite is split by what a test is *for*, not by how long it takes.
A plain ``uv run pytest`` still collects and runs everything, so the handoff
command in `AGENTS.md` keeps its meaning; CI splits the tiers into parallel
jobs with ``-m``.
"""

from __future__ import annotations

import os

# Set before the first `import jax` anywhere in the process. pytest imports
# conftest ahead of the test modules that pull in the library, and each xdist
# worker is its own process, so this is the earliest correct hook.
if os.environ.get("PYTEST_XDIST_WORKER"):
    os.environ.setdefault(
        "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest  # noqa: E402  -- must follow the pre-import environment setup above

#: Modules that execute published prose -- documentation snippets, README
#: fences and portal snippets -- rather than calling the library directly. They
#: re-run the same `examples/` generators that `test_evidence_suite.py` already
#: asserts against, so what they detect is presentation drift, not a numerical
#: regression. Splitting them out keeps them off CI's critical path and makes a
#: red build say which of the two broke.
_DOC_EXECUTION_MODULES = frozenset(
    {
        "test_docs_snippets.py",
        "test_portal_snippets.py",
        "test_readme.py",
    }
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tag whole modules by role so no individual test needs a decorator."""
    for item in items:
        if item.path.name in _DOC_EXECUTION_MODULES:
            item.add_marker("docs_execution")
