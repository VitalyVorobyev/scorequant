"""Generate the captured-output snippets the ``/get-started`` page renders.

``get_started_program.py`` is the single source: one runnable Python program
split into cells by ``# %% cell: <id>`` markers. This script executes that
program's preamble once and then every cell in order, in one shared
namespace -- exactly what a reader gets running the file top to bottom -- and
captures each cell's stdout. The result is written to
``website/src/generated/snippet-outputs.json``, which
``website/src/lib/snippets.ts`` is the only typed reader of and
``website/src/components/Snippet.tsx`` renders one cell of.

The ``partition`` cell's score table is not otherwise reachable from outside
the program's own namespace, and the portal's `LiveFit` demo beside that
snippet (`website/src/components/GetStartedFirstFitLiveFit.tsx`) needs it to
refit the same points, at the same bin budget and seed, in the reader's
browser. That component, and the ``firstFit`` key it reads, keep the name the
cell had when the finite fit opened the page; the cell it reads is now
``partition``, which the page reaches last. This script also writes that table
-- read straight out of the same executed namespace, never retyped -- to
``website/static/walkthrough-scores/get-started.json``, the same convention
`generate_walkthroughs.py`'s ``write_walkthrough_score_tables`` uses for the
other three walkthroughs' on-demand score tables.

Run through ``pnpm generate:snippets``, or directly::

    uv run python website/scripts/generate_snippets.py
    uv run python website/scripts/generate_snippets.py --check
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import TypedDict

# Must run before the executed cells import scorequant: every cell prints a
# bit-reproducible number, which requires float64 and a headless matplotlib
# backend (imported only transitively, but never assumed absent).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / "website/scripts/get_started_program.py"
OUTPUT = ROOT / "website/src/generated/snippet-outputs.json"
SCORE_TABLE_OUTPUT = ROOT / "website/static/walkthrough-scores/get-started.json"

#: One ``# %% cell: <id>`` marker line. The id is the cell's lookup key in
#: the generated file and in ``website/src/lib/snippets.ts``.
_CELL_MARKER = re.compile(r"^# %% cell: (?P<id>[a-z0-9-]+)[ \t]*$", re.MULTILINE)

SCHEMA_VERSION = 1

#: Provenance recorded alongside every cell: the runtime ``get_started_program.py``
#: pins in its ``setup`` cell. A drift here would mean the program and this
#: script disagree about what they ran, which is worse than either being wrong
#: alone.
EXECUTION: dict[str, object] = {"backend": "numpy", "precision": "float64", "seed": 21}


class CellRecord(TypedDict):
    """One executed cell's verbatim source, captured stdout, and file order."""

    code: str
    stdout: str
    order: int


def split_cells(source: str) -> list[tuple[str, str]]:
    """Split ``source`` into ``(cell_id, code)`` pairs in file order.

    Parameters
    ----------
    source
        Full text of ``get_started_program.py``.

    Returns
    -------
    list of (str, str)
        Each cell's id and its source, stripped of the marker line and of
        the blank lines surrounding it. The module docstring before the
        first marker is preamble, never a cell.

    Raises
    ------
    ValueError
        No marker is found, or the same cell id is declared twice.
    """
    matches = list(_CELL_MARKER.finditer(source))
    if not matches:
        raise ValueError(f"{PROGRAM} has no '# %% cell: <id>' markers")
    seen: set[str] = set()
    cells: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        cell_id = match.group("id")
        if cell_id in seen:
            raise ValueError(f"duplicate cell id {cell_id!r} in {PROGRAM}")
        seen.add(cell_id)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        cells.append((cell_id, source[start:end].strip("\n")))
    return cells


def _preamble(source: str) -> str:
    """Return the source before the first cell marker (the module docstring)."""
    match = _CELL_MARKER.search(source)
    return source if match is None else source[: match.start()]


def run_program(source: str) -> tuple[dict[str, CellRecord], dict[str, object]]:
    """Execute the preamble, then every cell in order, in one shared namespace.

    Parameters
    ----------
    source
        Full text of ``get_started_program.py``.

    Returns
    -------
    tuple of (dict, dict)
        The cell records (mapping cell id to its code, captured stdout, and
        file order), and the shared namespace every cell executed into -- the
        same namespace a reader running the file top to bottom would end up
        with. ``build_first_fit_score_table`` reads ``scores``, ``weights``
        and ``partition`` straight out of it, so the ``partition`` cell's score
        table can never drift from what the cell actually ran.
    """
    namespace: dict[str, object] = {"__name__": "__snippet_program__"}
    exec(compile(_preamble(source), str(PROGRAM), "exec"), namespace)
    records: dict[str, CellRecord] = {}
    for order, (cell_id, code) in enumerate(split_cells(source)):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, f"{PROGRAM}:{cell_id}", "exec"), namespace)
        records[cell_id] = {"code": code, "stdout": buffer.getvalue(), "order": order}
    return records, namespace


def build_payload() -> dict[str, object]:
    """Run the program and assemble the JSON-ready generated snippet payload."""
    source = PROGRAM.read_text(encoding="utf-8")
    cells, namespace = run_program(source)
    partition = namespace["partition"]
    return {
        "schemaVersion": SCHEMA_VERSION,
        "execution": EXECUTION,
        "cells": cells,
        # The retention `LiveFit` shows as this page's committed result, before
        # the reader ever clicks anything -- read from the same `partition`
        # the `partition` cell already printed `geometric_mean_retention`
        # from, not recomputed or reparsed from that cell's captured stdout.
        "firstFit": {"retention": partition.train_report.geometric_mean_retention},
    }


def build_first_fit_score_table() -> dict[str, object]:
    """Run the program and build the `/get-started` first-fit score table.

    Reads the exact objects the ``partition`` cell bound in the shared
    namespace -- ``scores``, ``weights`` and the fitted ``partition`` -- so
    the bin count, seed and solver a browser rerun is asked to use can never
    drift from what the cell actually ran, even if ``get_started_program.py``
    changes them later.

    Returns
    -------
    dict
        ``{"detail", "label", "schema", "scores", "weights", "nBins", "seed",
        "solver"}``, in exactly the shape
        ``website/src/components/GetStartedFirstFitLiveFit.tsx`` fetches.
    """
    source = PROGRAM.read_text(encoding="utf-8")
    _, namespace = run_program(source)
    return _first_fit_score_table(namespace)


def _first_fit_score_table(namespace: dict[str, object]) -> dict[str, object]:
    """Build the first-fit score table payload from an already-executed namespace."""
    scores = namespace["scores"]
    weights = namespace["weights"]
    partition = namespace["partition"]
    rows, columns = scores.shape
    return {
        "detail": (
            f"{rows:,} two-dimensional scores from `np.random.default_rng(21)` -- "
            "the /get-started first fit"
        ),
        "label": "Get-started first-fit scores",
        "schema": ["s1", "s2"],
        "scores": scores.tolist(),
        "weights": weights.tolist(),
        "nBins": int(partition.cell_score_means.shape[0]),
        "seed": int(partition.config.seed),
        "solver": str(partition.config.method),
    }


def render(payload: dict[str, object]) -> str:
    """Serialize the snippet payload exactly as it is committed: indented, newline-terminated."""
    return json.dumps(payload, indent=2) + "\n"


def render_score_table(payload: dict[str, object]) -> str:
    """Serialize a score-table payload matching the ``walkthrough-scores/`` convention."""
    return json.dumps(payload, sort_keys=True) + "\n"


def _check(path: Path, rendered: str) -> bool:
    """Compare ``rendered`` against the committed file at ``path``, printing a diff if stale.

    Returns whether the file matches, so a caller can check every generated
    output before deciding whether to exit non-zero, instead of stopping at
    the first stale file and hiding the rest.
    """
    if not path.exists():
        print(f"{path.relative_to(ROOT)} does not exist; run without --check first")
        return False
    committed = path.read_text(encoding="utf-8")
    if committed == rendered:
        print(f"{path.relative_to(ROOT)} matches a fresh run")
        return True
    diff = "".join(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=f"committed {path.relative_to(ROOT)}",
            tofile="freshly generated",
        )
    )
    print(f"{path.relative_to(ROOT)} is stale:\n{diff}")
    return False


def main() -> None:
    """Write the generated snippet files, or verify they are already current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed files match a fresh run instead of writing them",
    )
    arguments = parser.parse_args()

    # Run the program exactly once and build both outputs from that one run,
    # rather than calling `build_payload`/`build_first_fit_score_table` (each
    # of which re-runs the program on its own) twice over.
    source = PROGRAM.read_text(encoding="utf-8")
    cells, namespace = run_program(source)
    partition = namespace["partition"]
    snippet_payload = {
        "schemaVersion": SCHEMA_VERSION,
        "execution": EXECUTION,
        "cells": cells,
        "firstFit": {"retention": partition.train_report.geometric_mean_retention},
    }
    rendered_snippets = render(snippet_payload)
    rendered_score_table = render_score_table(_first_fit_score_table(namespace))

    if arguments.check:
        snippets_current = _check(OUTPUT, rendered_snippets)
        score_table_current = _check(SCORE_TABLE_OUTPUT, rendered_score_table)
        if not (snippets_current and score_table_current):
            sys.exit(1)
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered_snippets, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")

    SCORE_TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SCORE_TABLE_OUTPUT.write_text(rendered_score_table, encoding="utf-8")
    print(f"wrote {SCORE_TABLE_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
