"""Execute the fenced Python snippets in README.md and guard published prose.

The README is the library's front door, so every ```python fence in it is a
claim about the current API. Blocks are extracted with the same helpers the
documentation harness uses and executed in order in one shared namespace,
honoring the same ``<!-- snippet: skip -->`` / ``<!-- snippet: fresh -->``
markers.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.test_docs_snippets import REPO_ROOT, Snippet, _extract_blocks

README = REPO_ROOT / "README.md"

# Internal planning vocabulary that must not reach a reader. "adr" is matched
# on word boundaries so ordinary words containing those three letters
# (quadrature, quadratic) do not trip the guard.
_FORBIDDEN_PROSE = (
    r"\broadmap\b",
    r"\badr\b",
    r"\bpre-release\b",
    r"\bmigration history\b",
    r"\bdevelopment phase\b",
)

# Front-door pages rewritten together with the README. They additionally carry
# no work-in-progress markers.
_FRONT_DOOR = (
    "index.md",
    "motivation.md",
    "method.md",
    "three-doors.md",
    "user-workflow.md",
    "related-work.md",
)

# Pages that mkdocs.yml keeps out of the published site.
_UNPUBLISHED = {
    Path("docs/development.md"),
    Path("docs/roadmap.md"),
    Path("docs/system-design.md"),
    Path("docs/playbook.md"),
    Path("docs/decisions.md"),
}

# Whole directories mkdocs.yml keeps out of the published site. These hold
# working material -- programme packets -- which is allowed to use internal
# planning vocabulary the front door must not. (The per-decision record
# directory was retired in favour of docs/decisions.md, which is published.)
_UNPUBLISHED_DIRS = ("programme",)


def _readme_blocks() -> list[Snippet]:
    return _extract_blocks(README.read_text(encoding="utf-8"))


def test_readme_has_snippets() -> None:
    blocks = _readme_blocks()
    assert blocks, "no Python snippet was discovered in README.md"
    assert any(not block.skip for block in blocks)


def test_readme_snippets_execute() -> None:
    namespace: dict[str, object] = {}
    for block in _readme_blocks():
        if block.fresh:
            namespace = {}
        if block.skip:
            continue
        exec(compile(block.code, f"README.md:block{block.index}", "exec"), namespace)


def test_readme_contains_only_current_user_facing_language() -> None:
    text = README.read_text(encoding="utf-8")
    for pattern in (*_FORBIDDEN_PROSE, r"\bTODO\b", r"\bFIXME\b"):
        assert not re.search(pattern, text, flags=re.IGNORECASE), pattern


def test_published_markdown_has_no_internal_or_malformed_content() -> None:
    published = [
        path
        for path in (REPO_ROOT / "docs").rglob("*.md")
        if path.relative_to(REPO_ROOT) not in _UNPUBLISHED
        and not any(part in _UNPUBLISHED_DIRS for part in path.parts)
    ]
    assert published
    for path in published:
        content = path.read_text(encoding="utf-8")
        assert "\t" not in content, f"tab character in {path}"
        for pattern in _FORBIDDEN_PROSE:
            assert not re.search(pattern, content, flags=re.IGNORECASE), (path, pattern)

    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    excluded_paths = (
        "development.md",
        "roadmap.md",
        "system-design.md",
        "playbook.md",
        "decisions.md",
        *(f"{directory}/**" for directory in _UNPUBLISHED_DIRS),
    )
    for excluded in excluded_paths:
        assert excluded in mkdocs


def test_front_door_pages_have_no_work_in_progress_markers() -> None:
    for name in _FRONT_DOOR:
        content = (REPO_ROOT / "docs" / name).read_text(encoding="utf-8")
        for pattern in (r"\bTODO\b", r"\bFIXME\b", r"snippet:\s*skip"):
            assert not re.search(pattern, content, flags=re.IGNORECASE), (name, pattern)


def test_retired_pages_are_gone() -> None:
    retired_pages = (
        "docs/migration.md",
        "docs/score-sources.md",
    )
    for retired in retired_pages:
        assert not (REPO_ROOT / retired).exists()
    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for retired in retired_pages:
        assert Path(retired).name not in mkdocs
