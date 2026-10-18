"""Execute every Python snippet the portal publishes.

Two surfaces, one contract -- a published snippet runs, and it produces a
fitted result rather than merely parsing.

``website/get-started/index.mdx`` shows no code of its own. Its cells live in
``website/scripts/get_started_program.py``, one runnable file split by
``# %% cell: <id>`` markers, which ``website/scripts/generate_snippets.py``
executes to capture the output the page displays. This module executes the
same cells through the same splitter, so the page, the generator and this test
cannot drift apart: there is one source, and all three read it. The cells share
a single namespace and run in order, because that is how the page presents
them.

``website/walkthroughs/*.mdx`` holds the four walkthrough pages. Their fences
are ordinary Markdown code fences, so they reuse ``_extract_blocks`` from
``tests/test_docs_snippets.py`` rather than a second extractor; the markers
there are written ``{/* snippet: skip */}`` because MDX 3 rejects HTML
comments. A walkthrough executes in its own **fresh** namespace with no
pre-seeded fixture, cumulatively across that page's fences: a walkthrough must
run as shown, because a reader is meant to be able to reproduce it from what is
on the page. The result assertion is per page rather than per fence, since the
arc reaches a fit once.

These fences execute published prose, so this module carries the
``docs_execution`` marker that ``tests/conftest.py`` assigns to that tier.
Both surfaces are held to that same tier for the same reason: published prose
that only parses, without producing a result, has not actually been checked.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import scorequant as sq
from tests.test_docs_snippets import Snippet, _extract_blocks

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).parents[1]
WALKTHROUGHS = ROOT / "website/walkthroughs"
GET_STARTED_PAGE = ROOT / "website/get-started/index.mdx"
GET_STARTED_PROGRAM = ROOT / "website/scripts/get_started_program.py"
CAPTURED_OUTPUTS = ROOT / "website/src/generated/snippet-outputs.json"

#: The four walkthroughs S8 published. Named rather than globbed so that a page
#: silently failing to be discovered is a test failure instead of a quietly
#: smaller run.
EXPECTED_WALKTHROUGHS = ("flowcyt", "hep", "michelson", "ratios")


def _load_generator() -> ModuleType:
    """Import ``generate_snippets.py`` by path.

    It is a script rather than a package module, but its ``split_cells`` is the
    definition of what a cell *is*. Re-implementing the split here would create
    exactly the second parser the single-source design exists to avoid.
    """
    spec = importlib.util.spec_from_file_location(
        "scorequant_generate_snippets", ROOT / "website/scripts/generate_snippets.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GENERATOR = _load_generator()
_PROGRAM_SOURCE = GET_STARTED_PROGRAM.read_text(encoding="utf-8")
_CELLS: list[tuple[str, str]] = _GENERATOR.split_cells(_PROGRAM_SOURCE)
assert _CELLS, "the get-started program yielded no cells"


def test_get_started_cells_execute_and_produce_a_result() -> None:
    """Run every cell in order, in one namespace, exactly as the page shows them."""
    namespace: dict[str, object] = {}
    exec(  # noqa: S102
        compile(_GENERATOR._preamble(_PROGRAM_SOURCE), str(GET_STARTED_PROGRAM), "exec"),
        namespace,
    )
    for cell_id, code in _CELLS:
        exec(compile(code, f"{GET_STARTED_PROGRAM}:{cell_id}", "exec"), namespace)  # noqa: S102

    produced = [
        value
        for value in namespace.values()
        if isinstance(value, (sq.PartitionResult, sq.QuantizerResult))
    ]
    assert produced, (
        "the get-started program never bound a PartitionResult or QuantizerResult; "
        "the page must carry a reader all the way to a fit"
    )


@pytest.mark.parametrize("cell_id, code", _CELLS, ids=[cell_id for cell_id, _ in _CELLS])
def test_get_started_cell_uses_public_symbols(cell_id: str, code: str) -> None:
    """Every ``sq.<name>`` the page shows is in the public API."""
    public = set(sq.__all__)
    tree = ast.parse(code, filename=f"{GET_STARTED_PROGRAM}:{cell_id}")
    referenced = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sq"
    }
    assert referenced <= public, (
        f"get-started cell {cell_id!r} uses private or missing names: {sorted(referenced - public)}"
    )


def test_get_started_page_references_only_real_cells() -> None:
    """Every ``<Snippet id=...>`` on the page names a cell the program defines."""
    page = GET_STARTED_PAGE.read_text(encoding="utf-8")
    referenced = set(re.findall(r'<Snippet\s+id="([^"]+)"', page))
    defined = {cell_id for cell_id, _ in _CELLS}
    assert referenced, "the get-started page renders no snippets"
    assert referenced <= defined, f"page references undefined cells: {sorted(referenced - defined)}"
    assert defined <= referenced, (
        f"program defines cells the page never renders: {sorted(defined - referenced)}"
    )


def test_get_started_page_contains_no_output_literal() -> None:
    """The page renders captured output; it never contains one.

    Two independent checks, because either alone is evadable. The page may hold
    no verbatim output fence at all -- every output block arrives through
    ``<Snippet/>`` -- and no line of any captured stdout may appear in the page
    source, which catches an output pasted beside the component rather than
    inside a fence.
    """
    page = GET_STARTED_PAGE.read_text(encoding="utf-8")
    verbatim_fences = re.findall(r"^```(?:text|console|output)\b", page, flags=re.MULTILINE)
    assert not verbatim_fences, (
        "the get-started page carries a verbatim output fence; captured output "
        "must arrive through <Snippet/> so it cannot drift from the run"
    )

    captured = json.loads(CAPTURED_OUTPUTS.read_text(encoding="utf-8"))
    for cell_id, cell in captured["cells"].items():
        for line in cell["stdout"].splitlines():
            stripped = line.strip()
            if len(stripped) < 12:
                continue
            assert stripped not in page, (
                f"the get-started page contains a literal line of {cell_id!r}'s captured "
                f"output: {stripped!r}. The page must render it through <Snippet/>."
            )


def test_captured_outputs_are_current() -> None:
    """Re-execute the program; the committed artifact must equal what it produces.

    The analogue of ``tests/test_walkthrough_facts.py::test_generated_file_is_current``.
    Exact equality rather than a tolerance: the cells pin the NumPy backend at
    float64 with a fixed seed and format every number in the snippet itself, so
    the output is reproducible, and a tolerance-comparing test that never fails
    is not a contract.
    """
    expected = _GENERATOR.render(_GENERATOR.build_payload())
    actual = CAPTURED_OUTPUTS.read_text(encoding="utf-8")
    assert actual == expected, (
        "website/src/generated/snippet-outputs.json is stale; "
        "run `uv run python website/scripts/generate_snippets.py`"
    )


def _walkthrough_pages() -> list[tuple[str, list[Snippet]]]:
    """Return ``(slug, blocks)`` for every walkthrough page carrying a fence."""
    pages = []
    for path in sorted(WALKTHROUGHS.glob("*.mdx")):
        if path.stem == "index":
            continue
        blocks = _extract_blocks(path.read_text(encoding="utf-8"))
        if blocks:
            pages.append((path.stem, blocks))
    return pages


_WALKTHROUGHS = _walkthrough_pages()


def test_every_walkthrough_was_discovered() -> None:
    """All four walkthroughs exist and carry executable fences."""
    assert [slug for slug, _ in _WALKTHROUGHS] == sorted(EXPECTED_WALKTHROUGHS)


@pytest.mark.parametrize("slug, blocks", _WALKTHROUGHS, ids=[slug for slug, _ in _WALKTHROUGHS])
def test_walkthrough_fences_execute_and_produce_a_result(slug: str, blocks: list[Snippet]) -> None:
    """Run one walkthrough's fences in order, in a namespace it builds itself."""
    namespace: dict[str, object] = {}
    for block in blocks:
        if block.fresh:
            namespace = {}
        if block.skip:
            continue
        filename = f"website/walkthroughs/{slug}.mdx:block{block.index}"
        exec(compile(block.code, filename, "exec"), namespace)  # noqa: S102

    produced = [
        value
        for value in namespace.values()
        if isinstance(value, (sq.PartitionResult, sq.QuantizerResult))
    ]
    assert produced, (
        f"walkthrough {slug!r} never bound a PartitionResult or QuantizerResult; "
        "a walkthrough must carry its narrative all the way to a fit"
    )


@pytest.mark.parametrize("slug, blocks", _WALKTHROUGHS, ids=[slug for slug, _ in _WALKTHROUGHS])
def test_walkthrough_fences_use_public_symbols(slug: str, blocks: list[Snippet]) -> None:
    """Every ``sq.<name>`` a walkthrough shows is in the public API."""
    public = set(sq.__all__)
    for block in blocks:
        tree = ast.parse(block.code, filename=f"website/walkthroughs/{slug}.mdx:{block.index}")
        referenced = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sq"
        }
        assert referenced <= public, (
            f"walkthrough {slug!r} block {block.index} uses private or missing names: "
            f"{sorted(referenced - public)}"
        )
