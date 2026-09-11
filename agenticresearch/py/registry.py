"""Tooling for the agenticresearch scientific memory.

The workspace stores one claim per file under ``claims/`` with the shared
vocabularies in ``registry.json``. This module is the single seam that turns
that directory back into one in-memory registry, the validator that keeps every
cross-file pointer honest, and the lookup that lets a research agent pull a
claim plus its transitive dependencies plus its proof prose without reading the
whole memory.

Usage::

    python py/registry.py validate
    python py/registry.py reindex [--check]
    python py/registry.py show <CLAIM-ID> [--deps] [--proof]
    python py/registry.py export [--out FILE]

Pure standard library on purpose: bookkeeping sessions run on cheap models and
must never need the library's numerical environment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]

#: Heading prefixes reserved for result labels in the KNOWN_RESULTS chapters.
#: Inline assumption labels must not reuse them -- see `_check_label_collisions`.
RESULT_LABEL_PREFIXES = ("U", "T", "G", "DS", "D", "E", "A", "S", "C", "O", "I")

_RESULT_HEADING = re.compile(r"^##\s+([A-Z]+\d+)\.\s", re.MULTILINE)
_CLAIMS_LINE = re.compile(r"^\*\*Claims:\*\*\s*(.+?)\s*$", re.MULTILINE)
#: An actual inlined payload -- not prose naming one, which the docs must be free to do.
_DATA_URI = re.compile(r"data:image/[a-z+]+;base64,[A-Za-z0-9+/=]{32,}")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load(workspace: Path = WORKSPACE) -> dict:
    """Return the registry as one dict with a ``claims`` list plus vocabularies.

    Reads the sharded layout (``registry.json`` + ``claims/*.json``) and falls
    back to the pre-v4 single ``CLAIMS.json`` so the migration can be validated
    from both sides.
    """
    header_path = workspace / "registry.json"
    if header_path.is_file():
        registry = json.loads(header_path.read_text())
        claims = []
        seen: set[str] = set()
        for path in sorted((workspace / "claims").glob("*.json")):
            node = json.loads(path.read_text())
            if node.get("id") != path.stem:
                raise ValueError(f"{path.name}: id {node.get('id')!r} does not match filename")
            if node["id"] in seen:
                raise ValueError(f"duplicate claim id {node['id']}")
            seen.add(node["id"])
            claims.append(node)
        registry["claims"] = claims
        return registry
    legacy = workspace / "CLAIMS.json"
    if legacy.is_file():
        return json.loads(legacy.read_text())
    raise FileNotFoundError(f"no registry found under {workspace}")


def claims_by_id(registry: dict) -> dict[str, dict]:
    """Index the claim list by id."""
    return {claim["id"]: claim for claim in registry["claims"]}


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _known_results_files(workspace: Path) -> list[Path]:
    directory = workspace / "KNOWN_RESULTS"
    if directory.is_dir():
        return sorted(directory.glob("*.md"))
    legacy = workspace / "KNOWN_RESULTS.md"
    return [legacy] if legacy.is_file() else []


def _sections(text: str) -> dict[str, str]:
    """Map each ``## `` heading line to the body that follows it."""
    out: dict[str, str] = {}
    heading: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if heading is not None:
                out[heading] = "\n".join(body)
            heading = line[3:].strip()
            body = []
        elif line.startswith("# "):
            if heading is not None:
                out[heading] = "\n".join(body)
            heading = None
            body = []
        elif heading is not None:
            body.append(line)
    if heading is not None:
        out[heading] = "\n".join(body)
    return out


def _check_vocabularies(registry: dict, out: list[str]) -> None:
    statuses = set(registry["status_definitions"])
    publication = set(registry["publication_status_definitions"])
    search = set(registry["literature_search_status_definitions"])
    levels = set(registry["levels"])
    criteria = set(registry["criteria"])
    for claim in registry["claims"]:
        cid = claim["id"]
        if claim["status"] not in statuses:
            out.append(f"{cid}: unknown status {claim['status']!r}")
        if claim["publication_status"] not in publication:
            out.append(f"{cid}: unknown publication_status {claim['publication_status']!r}")
        if claim["level"] not in levels:
            out.append(f"{cid}: unknown level {claim['level']!r}")
        for criterion in claim["criterion"]:
            if criterion not in criteria:
                out.append(f"{cid}: unknown criterion {criterion!r}")
        search_status = claim.get("literature_search_status")
        if search_status is not None and search_status not in search:
            out.append(f"{cid}: unknown literature_search_status {search_status!r}")


def _check_graph_edges(index: dict[str, dict], out: list[str]) -> None:
    for claim in index.values():
        for field in ("dependencies", "implies", "converse_failures", "verified_by", "references"):
            targets = claim.get(field, [])
            if len(targets) != len(set(targets)):
                out.append(f"{claim['id']}: duplicate {field} edge")
            for other in claim.get(field, []):
                if other not in index:
                    out.append(f"{claim['id']}: {field} -> unknown claim {other}")
                    continue
                target = index[other]
                is_audit = target.get("proof_location", {}).get("file", "").startswith("AUDITS/")
                if field == "verified_by" and not is_audit:
                    out.append(f"{claim['id']}: verified_by target {other} is not an audit")
                if field == "implies" and is_audit:
                    out.append(f"{claim['id']}: audit target {other} belongs in verified_by, not implies")
                if field == "dependencies":
                    source_audit = claim.get("proof_location", {}).get("file", "").startswith("AUDITS/")
                    if is_audit or source_audit or target.get("status") == "measured":
                        out.append(f"{claim['id']}: evidence belongs in references or verified_by, not dependencies -> {other}")
                    if other == claim["id"]:
                        out.append(f"{claim['id']}: self-dependency")

    # Only proof prerequisites must form a DAG. Audit references may be reciprocal.
    visited: set[str] = set()
    active: list[str] = []

    def visit(cid: str) -> None:
        if cid in active:
            cycle = active[active.index(cid):] + [cid]
            out.append("dependency cycle: " + " -> ".join(cycle))
            return
        if cid in visited:
            return
        active.append(cid)
        for target in index[cid].get("dependencies", []):
            if target in index:
                visit(target)
        active.pop()
        visited.add(cid)

    for cid in sorted(index):
        visit(cid)


def _check_proof_locations(registry: dict, workspace: Path, out: list[str]) -> None:
    cache: dict[str, str] = {}
    for claim in registry["claims"]:
        location = claim["proof_location"]
        target = workspace / location["file"]
        if not target.is_file():
            out.append(f"{claim['id']}: proof_location file {location['file']} does not exist")
            continue
        if location["file"] not in cache:
            cache[location["file"]] = target.read_text()
        if location["section"] not in cache[location["file"]]:
            out.append(
                f"{claim['id']}: section {location['section']!r} not found in {location['file']}"
            )


def _check_counterexamples(registry: dict, workspace: Path, out: list[str]) -> None:
    cited: set[str] = set()
    for claim in registry["claims"]:
        ids = claim.get("counterexamples", []) + claim.get("boundary_counterexamples", [])
        for fixture_id in ids:
            cited.add(fixture_id)
            if not (workspace / "COUNTEREXAMPLES" / f"{fixture_id}.json").is_file():
                out.append(f"{claim['id']}: no fixture for {fixture_id}")
    fixtures = sorted((workspace / "COUNTEREXAMPLES").glob("CE-*.json"))
    if not fixtures:
        out.append("counterexample bank is empty")
    for path in fixtures:
        fixture = json.loads(path.read_text())
        if fixture["id"] != path.stem:
            out.append(f"{path.name}: id {fixture['id']!r} does not match filename")
        if fixture["id"] not in cited:
            out.append(f"{path.name} is cited by no claim")
        if len(fixture["labels_before"]) != len(fixture["scores"]):
            out.append(f"{path.name}: labels_before/scores length mismatch")
        if len(fixture["weights"]) != len(fixture["scores"]):
            out.append(f"{path.name}: weights/scores length mismatch")
        if not set(fixture["labels_before"]) <= set(range(fixture["K"])):
            out.append(f"{path.name}: labels_before outside range(K)")


def _check_artifacts(registry: dict, workspace: Path, out: list[str]) -> None:
    for claim in registry["claims"]:
        for field in ("artifact", "audit"):
            value = claim.get(field)
            if value is not None and not (workspace / value).is_file():
                out.append(f"{claim['id']}: {field} path {value} does not exist")


_FORMAL_PROOF_FIELDS = {"system", "spec", "file", "declaration", "statement_audit"}

# A machine-checked proof asserts the claim's own ``statement``. Statuses that do
# not assert a proved statement therefore cannot carry one.
_FORMAL_PROOF_FORBIDDEN_STATUSES = {"open", "conjecture", "measured", "counterexample"}


def _check_formal_proofs(registry: dict, workspace: Path, out: list[str]) -> None:
    """Optional machine-checked evidence must resolve and be uniquely owned."""
    seen: dict[str, str] = {}
    for claim in registry["claims"]:
        formal = claim.get("formal_proof")
        if formal is None:
            continue
        if not isinstance(formal, dict):
            out.append(f"{claim['id']}: formal_proof must be an object")
            continue
        missing = sorted(_FORMAL_PROOF_FIELDS - set(formal))
        extra = sorted(set(formal) - _FORMAL_PROOF_FIELDS)
        if missing:
            out.append(f"{claim['id']}: formal_proof is missing {', '.join(missing)}")
        if extra:
            out.append(f"{claim['id']}: formal_proof has unknown field(s) {', '.join(extra)}")
        if claim["status"] in _FORMAL_PROOF_FORBIDDEN_STATUSES:
            out.append(f"{claim['id']}: status {claim['status']} cannot carry formal_proof")
        for field in ("spec", "file", "statement_audit"):
            value = formal.get(field)
            if not isinstance(value, str) or not value:
                out.append(f"{claim['id']}: formal_proof.{field} must be a non-empty path")
                continue
            if ".." in Path(value).parts:
                out.append(f"{claim['id']}: formal_proof.{field} escapes the workspace")
                continue
            if not (workspace / value).is_file():
                out.append(f"{claim['id']}: formal_proof.{field} path {value} does not exist")
        declaration = formal.get("declaration")
        if not isinstance(declaration, str) or not declaration:
            out.append(f"{claim['id']}: formal_proof.declaration must be a non-empty name")
            continue
        if declaration in seen:
            out.append(
                f"{claim['id']}: formal_proof.declaration {declaration} is already "
                f"claimed by {seen[declaration]}"
            )
        seen[declaration] = claim["id"]
        source = formal.get("file")
        if isinstance(source, str) and (workspace / source).is_file():
            local = declaration.rsplit(".", 1)[-1]
            pattern = r"\btheorem\s+" + re.escape(local) + r"\b"
            if not re.search(pattern, (workspace / source).read_text()):
                out.append(f"{claim['id']}: {source} declares no theorem named {local}")


def bibliography_anchors(workspace: Path) -> dict[str, tuple[str, str]]:
    """Bibliography key -> (file, heading) from the ``**Key:**`` lines."""
    anchors: dict[str, tuple[str, str]] = {}
    root = workspace / "LITERATURE"
    if not root.is_dir():
        return anchors
    for path in sorted(root.rglob("*.md")):
        heading = ""
        for line in path.read_text().splitlines():
            if line.startswith("#"):
                heading = line.lstrip("#").strip()
            elif line.startswith("**Key:**"):
                for key in line[len("**Key:**") :].split(","):
                    key = key.strip()
                    if key:
                        anchors[key] = (str(path.relative_to(workspace)), heading)
    return anchors


def _check_bibliography(registry: dict, workspace: Path, out: list[str]) -> None:
    bibliography = set(registry["bibliography"])
    for claim in registry["claims"]:
        for key in claim.get("literature", []):
            if key not in bibliography:
                out.append(f"{claim['id']}: unknown bibliography key {key}")
    anchors = bibliography_anchors(workspace)
    if not anchors:
        return
    for key in sorted(bibliography - set(anchors)):
        out.append(f"bibliography key {key} is annotated nowhere in LITERATURE/")
    for key in sorted(set(anchors) - bibliography):
        out.append(f"LITERATURE/ anchors {key}, which is not in the bibliography")


def _check_work_pointers(workspace: Path, out: list[str]) -> None:
    """Every ``WORK/active/...`` path named in prose must exist.

    Completed packets move out of ``WORK/active/``; without this check the
    pointers left behind silently send the next session at a missing file.
    """
    pattern = re.compile(r"WORK/active/[A-Za-z0-9-]+\.md")
    for path in sorted(workspace.rglob("*.md")):
        if any(part in {"archive", "AUDITS", "manuscripts"} for part in path.parts):
            continue
        for match in set(pattern.findall(path.read_text())):
            if not (workspace / match).is_file():
                rel = path.relative_to(workspace)
                out.append(f"{rel}: dead pointer to {match}")


def _check_evidence_ledger(index: dict[str, dict], workspace: Path, out: list[str]) -> None:
    ledger = workspace / "NUMERICAL_EVIDENCE.md"
    if not ledger.is_file():
        out.append("NUMERICAL_EVIDENCE.md is missing")
        return
    rows = 0
    for line in ledger.read_text().splitlines():
        if not line.startswith("| N-"):
            continue
        rows += 1
        columns = [cell.strip() for cell in line.split("|")]
        if len(columns) < 7:
            out.append(f"NUMERICAL_EVIDENCE.md: malformed row {columns[1]!r}")
            continue
        cited = [token.strip() for token in columns[5].split(",") if token.strip()]
        if not cited:
            out.append(f"NUMERICAL_EVIDENCE.md: row {columns[1]} cites no claim")
        for claim_id in cited:
            if claim_id not in index:
                out.append(f"NUMERICAL_EVIDENCE.md: row {columns[1]} cites unknown {claim_id}")
    if rows == 0:
        out.append("NUMERICAL_EVIDENCE.md has no rows")


def _check_known_results_backlinks(index: dict[str, dict], workspace: Path, out: list[str]) -> None:
    """``**Claims:**`` under a result heading must agree with ``proof_location``.

    This is the link that used to live only in an operator's head: headings carry
    local labels (``D5``) and claims carry ids (``D-EXCHANGE-IMPLIES-VORONOI``).
    """
    files = _known_results_files(workspace)
    if not files:
        return
    declared: dict[str, set[str]] = {}
    for path in files:
        for heading, body in _sections(path.read_text()).items():
            match = _CLAIMS_LINE.search(body)
            if match is None:
                continue
            ids = {token.strip() for token in match.group(1).split(",") if token.strip()}
            declared[heading] = ids
            for claim_id in ids:
                if claim_id not in index:
                    rel = path.relative_to(workspace)
                    out.append(f"{rel}: heading {heading!r} declares unknown claim {claim_id}")
    if not declared:
        return
    for claim in index.values():
        location = claim["proof_location"]
        if not location["file"].startswith("KNOWN_RESULTS"):
            continue
        matches = [h for h in declared if h.startswith(location["section"])]
        if not matches:
            out.append(f"{claim['id']}: no **Claims:** line under {location['section']!r}")
            continue
        if not any(claim["id"] in declared[h] for h in matches):
            out.append(
                f"{claim['id']}: section {location['section']!r} does not list it in **Claims:**"
            )


def _check_programmes(registry: dict, workspace: Path, out: list[str]) -> None:
    programmes = registry.get("programmes")
    if not programmes:
        return
    for claim in registry["claims"]:
        if claim["status"] != "open":
            continue
        programme = claim.get("programme")
        if programme is None:
            out.append(f"{claim['id']}: open claim has no programme")
        elif programme not in programmes:
            out.append(f"{claim['id']}: unknown programme {programme!r}")
    used = {c.get("programme") for c in registry["claims"] if c["status"] == "open"}
    open_problems = workspace / "OPEN_PROBLEMS.md"
    text = open_problems.read_text() if open_problems.is_file() else ""
    for name, meta in programmes.items():
        if meta.get("readiness") not in {"ready", "blocked"}:
            out.append(f"programme {name}: readiness must be 'ready' or 'blocked'")
        # Infrastructure programmes (literature coverage) carry no theorem claims.
        if meta.get("kind") != "infrastructure" and name not in used:
            out.append(f"programme {name} has no open claim")
        if text and name not in text:
            out.append(f"programme {name} is not described in OPEN_PROBLEMS.md")


def _check_label_collisions(workspace: Path, out: list[str]) -> None:
    """Check that result labels are unique and assumption labels do not reuse them."""
    files = _known_results_files(workspace)
    owner: dict[str, str] = {}
    for path in files:
        text = path.read_text()
        rel = str(path.relative_to(workspace))
        for label in _RESULT_HEADING.findall(text):
            if label in owner:
                out.append(f"result label {label} defined in both {owner[label]} and {rel}")
            else:
                owner[label] = rel
        reserved = "|".join(RESULT_LABEL_PREFIXES)
        # Only flag *definitions* -- a list item introducing a local label. Prose
        # citations like "(DS9)" or "(by D5/D6)" legitimately name real results.
        definition = re.compile(rf"^\s*[-*]\s+\((({reserved})\d+)\)\s", re.MULTILINE)
        for token in sorted({m[0] for m in definition.findall(text)}):
            out.append(
                f"{rel}: local label ({token}) is defined with a reserved result-label "
                "prefix; use a free prefix such as (M1) for assumptions"
            )


def _check_inlined_assets(workspace: Path, out: list[str]) -> None:
    """No tracked text file carries a base64 ``data:`` payload.

    Inlined figures made the v8 manuscript a 407 KB single line-blob: unreadable
    by an agent, undiffable by git, and unsearchable by grep. Figures live in
    ``manuscripts/figures/`` and are referenced by relative path.
    """
    for suffix in ("*.md", "*.html", "*.json"):
        for path in sorted(workspace.rglob(suffix)):
            if "archive" in path.parts:
                continue
            count = len(_DATA_URI.findall(path.read_text(errors="replace")))
            if count:
                rel = path.relative_to(workspace)
                out.append(
                    f"{rel}: {count} inlined base64 asset(s); extract them to a "
                    "sibling figures/ directory and reference by relative path"
                )


def _check_index_current(registry: dict, workspace: Path, out: list[str]) -> None:
    path = workspace / "claims" / "INDEX.md"
    if not path.is_dir() and (workspace / "claims").is_dir():
        expected = render_index(registry)
        if not path.is_file():
            out.append("claims/INDEX.md is missing; run `registry.py reindex`")
        elif path.read_text() != expected:
            out.append("claims/INDEX.md is stale; run `registry.py reindex`")


def validate(workspace: Path = WORKSPACE) -> list[str]:
    """Return every registry integrity violation, newest checks last."""
    registry = load(workspace)
    index = claims_by_id(registry)
    out: list[str] = []
    _check_vocabularies(registry, out)
    _check_graph_edges(index, out)
    _check_proof_locations(registry, workspace, out)
    _check_counterexamples(registry, workspace, out)
    _check_artifacts(registry, workspace, out)
    _check_formal_proofs(registry, workspace, out)
    _check_bibliography(registry, workspace, out)
    _check_work_pointers(workspace, out)
    _check_evidence_ledger(index, workspace, out)
    _check_known_results_backlinks(index, workspace, out)
    _check_programmes(registry, workspace, out)
    _check_label_collisions(workspace, out)
    _check_inlined_assets(workspace, out)
    _check_index_current(registry, workspace, out)
    return out


# --------------------------------------------------------------------------- #
# index generation
# --------------------------------------------------------------------------- #
def _formal_marker(claim: dict) -> str:
    """The generated-index suffix announcing machine-checked evidence."""
    formal = claim.get("formal_proof")
    if not isinstance(formal, dict):
        return ""
    declaration = formal.get("declaration")
    return f" · machine-checked: `{declaration}`" if declaration else ""


def render_index(registry: dict) -> str:
    """Render the browsable claim digest, grouped by programme then status."""
    programmes = registry.get("programmes", {})
    claims = registry["claims"]
    lines = [
        "# Claim index",
        "",
        "Generated by `py/registry.py reindex` — do not edit by hand.",
        "",
        f"{len(claims)} claims. Open work is grouped by programme in queue order; "
        "everything else is grouped by status.",
        "",
    ]
    order = sorted(programmes, key=lambda name: programmes[name]["rank"])
    for name in order:
        meta = programmes[name]
        members = sorted((c for c in claims if c.get("programme") == name), key=lambda c: c["id"])
        if not members:
            continue
        lines.append(f"## {name} — {meta['title']} (rank {meta['rank']}, {meta['readiness']})")
        lines.append("")
        for claim in members:
            lines.append(
                f"- `{claim['id']}` — {claim['title']} · status: `{claim['status']}`"
                f"{_formal_marker(claim)}"
            )
        lines.append("")
    lines.append("## Settled claims by status")
    lines.append("")
    for status in sorted({c["status"] for c in claims if c.get("programme") is None}):
        members = sorted(
            (c for c in claims if c.get("programme") is None and c["status"] == status),
            key=lambda c: c["id"],
        )
        lines.append(f"### {status} ({len(members)})")
        lines.append("")
        for claim in members:
            location = claim["proof_location"]
            lines.append(
                f"- `{claim['id']}` — {claim['title']} · {location['file']}"
                f"{_formal_marker(claim)}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_counterexample_catalogue(workspace: Path) -> str:
    """Render the fixture catalogue from the fixtures themselves."""
    lines = [
        "# Counterexample catalogue",
        "",
        "Generated by `py/registry.py reindex` — do not edit by hand.",
        "The fixtures are the source of truth; `README.md` holds the admissibility",
        "criteria, the required JSON format, and the falsification checklist.",
        "",
        "| Fixture | Criterion | Level | Claim falsified | N | d | K |",
        "|---|---|---|---|---|---|---|",
    ]
    for path in sorted((workspace / "COUNTEREXAMPLES").glob("CE-*.json")):
        fixture = json.loads(path.read_text())
        scores = fixture["scores"]
        criterion = fixture["criterion"]
        criterion = ", ".join(criterion) if isinstance(criterion, list) else criterion
        dimension = len(scores[0]) if scores and isinstance(scores[0], list) else 1
        lines.append(
            f"| `{fixture['id']}` | {criterion} | {fixture['level']} "
            f"| {fixture['claim_falsified']} | {len(scores)} | {dimension} | {fixture['K']} |"
        )
    return "\n".join(lines) + "\n"


def render_bibliography(registry: dict, workspace: Path) -> str:
    """Render the key -> annotation map that makes `claims[].literature[]` resolvable."""
    anchors = bibliography_anchors(workspace)
    citing: dict[str, list[str]] = {}
    for claim in registry["claims"]:
        for key in claim.get("literature", []):
            citing.setdefault(key, []).append(claim["id"])
    lines = [
        "# Bibliography index",
        "",
        "Generated by `py/registry.py reindex` — do not edit by hand.",
        "",
        "Every key in `registry.json` `bibliography`, the heading that annotates it,",
        "and the claims that cite it.",
        "",
        "| Key | Title | Annotated in | Cited by |",
        "|---|---|---|---|",
    ]
    for key, entry in sorted(registry["bibliography"].items()):
        where, heading = anchors.get(key, ("—", "—"))
        location = f"`{where}` — {heading}" if where != "—" else "—"
        cited = ", ".join(f"`{c}`" for c in sorted(citing.get(key, []))) or "—"
        lines.append(f"| `{key}` | {entry['title']} | {location} | {cited} |")
    return "\n".join(lines) + "\n"


def reindex(workspace: Path = WORKSPACE, check: bool = False) -> list[str]:
    """Regenerate the derived indexes; with ``check`` only report drift."""
    registry = load(workspace)
    targets = [
        (workspace / "claims" / "INDEX.md", render_index(registry)),
        (workspace / "COUNTEREXAMPLES" / "INDEX.md", render_counterexample_catalogue(workspace)),
        (workspace / "LITERATURE" / "BIBLIOGRAPHY.md", render_bibliography(registry, workspace)),
    ]
    stale: list[str] = []
    for path, content in targets:
        if not path.parent.is_dir():
            continue
        current = path.read_text() if path.is_file() else None
        if current == content:
            continue
        rel = path.relative_to(workspace)
        if check:
            stale.append(f"{rel} is stale")
        else:
            path.write_text(content)
            stale.append(f"wrote {rel}")
    return stale


# --------------------------------------------------------------------------- #
# lookup
# --------------------------------------------------------------------------- #
def transitive_dependencies(index: dict[str, dict], claim_id: str) -> list[str]:
    """Ids reachable from ``claim_id`` through ``dependencies``, nearest first."""
    seen: set[str] = set()
    order: list[str] = []
    frontier = list(index[claim_id].get("dependencies", []))
    while frontier:
        current = frontier.pop(0)
        if current in seen or current not in index:
            continue
        seen.add(current)
        order.append(current)
        frontier.extend(index[current].get("dependencies", []))
    return order


def proof_text(claim: dict, workspace: Path = WORKSPACE) -> str | None:
    """Return the prose section a claim's ``proof_location`` points at."""
    location = claim["proof_location"]
    path = workspace / location["file"]
    if not path.is_file():
        return None
    for heading, body in _sections(path.read_text()).items():
        if heading.startswith(location["section"]):
            return f"## {heading}\n{body}".rstrip()
    return None


def show(claim_id: str, deps: bool, proof: bool, workspace: Path = WORKSPACE) -> str:
    """Render one claim, optionally with its dependency closure and proof prose."""
    registry = load(workspace)
    index = claims_by_id(registry)
    if claim_id not in index:
        raise KeyError(f"unknown claim {claim_id}")
    blocks = [json.dumps(index[claim_id], indent=2)]
    if proof:
        text = proof_text(index[claim_id], workspace)
        blocks.append(text or f"(no prose found at {index[claim_id]['proof_location']})")
    if deps:
        for dependency in transitive_dependencies(index, claim_id):
            blocks.append(f"--- dependency: {dependency} ---")
            blocks.append(json.dumps(index[dependency], indent=2))
            if proof:
                text = proof_text(index[dependency], workspace)
                if text:
                    blocks.append(text)
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
#: ``## D5. Title — [TAG]`` in KNOWN_RESULTS, ``## OP29. Title`` in OPEN_PROBLEMS.
_SECTION_HEADING = re.compile(
    r"^(?P<label>[A-Z]+\d+)\.\s+(?P<title>.+?)(?:\s+[—-]+\s+\[(?P<tag>[^\]]+)\])?\s*$"
)
_CHAPTER_TITLE = re.compile(r"^#\s+(?P<number>\d+[a-z]?)\.\s+(?P<title>.+?)\s*$", re.MULTILINE)
_FIELD_LINE = re.compile(r"^\*\*(?P<label>[^*:]+):\*\*\s*(?P<value>.*?)\s*$")
_LINK_LINE = re.compile(r"^-\s+(?P<label>[A-Za-z][A-Za-z0-9 ]*):\s*(?P<url>https?://\S+)\s*$")
_AUDIT_FILENAME = re.compile(r"^(?P<claim>.+?)-(?P<date>\d{1,2}-[A-Z][a-z]+-\d{4})$")
_AUDIT_SOURCE = re.compile(r"^-\s+\*\*(?P<name>[^*]+?):?\*\*:?\s*(?P<text>.+?)\s*$")
_VERDICT = re.compile(r"Verdict:\*?\*?\s*(?P<verdict>[^.*]+)")


def _parse_section_heading(heading: str) -> dict:
    match = _SECTION_HEADING.match(heading)
    if match is None:
        return {"heading": heading, "label": None, "title": heading, "tag": None}
    return {
        "heading": heading,
        "label": match.group("label"),
        "title": match.group("title"),
        "tag": match.group("tag"),
    }


def _strip_claims_line(body: str) -> str:
    return _CLAIMS_LINE.sub("", body).strip("\n")


def _export_chapters(workspace: Path) -> list[dict]:
    """Every KNOWN_RESULTS chapter with its result sections and their claims."""
    chapters: list[dict] = []
    for path in _known_results_files(workspace):
        if path.name == "index.md":
            continue
        text = path.read_text()
        title = _CHAPTER_TITLE.search(text)
        sections = []
        for heading, body in _sections(text).items():
            parsed = _parse_section_heading(heading)
            match = _CLAIMS_LINE.search(body)
            claims = (
                [token.strip() for token in match.group(1).split(",") if token.strip()]
                if match
                else []
            )
            sections.append({**parsed, "claims": claims})
        chapters.append(
            {
                "file": str(path.relative_to(workspace)),
                "number": title.group("number") if title else path.stem,
                "title": title.group("title") if title else path.stem,
                "sections": sections,
            }
        )
    return chapters


def _export_proof(claim: dict, workspace: Path) -> dict | None:
    """The prose section behind ``proof_location``, split into heading and body."""
    location = claim["proof_location"]
    path = workspace / location["file"]
    if not path.is_file():
        return None
    for heading, body in _sections(path.read_text()).items():
        if heading.startswith(location["section"]):
            return {
                **_parse_section_heading(heading),
                "file": location["file"],
                "body": _strip_claims_line(body),
            }
    return None


def _export_topics(workspace: Path) -> list[dict]:
    """The curated literature chapters: one record per annotated heading."""
    topics: list[dict] = []
    root = workspace / "LITERATURE" / "topics"
    if not root.is_dir():
        return topics
    for path in sorted(root.glob("*.md")):
        text = path.read_text()
        title = _CHAPTER_TITLE.search(text)
        entries = []
        for heading, body in _sections(text).items():
            keys: list[str] = []
            fields: dict[str, str] = {}
            links: list[dict] = []
            prose: list[str] = []
            current: str | None = None  # the field a wrapped line continues
            for line in body.splitlines():
                stripped = line.strip()
                field = _FIELD_LINE.match(stripped)
                link = _LINK_LINE.match(stripped)
                if not stripped:
                    current = None
                elif field and field.group("label") == "Key":
                    keys.extend(k.strip() for k in field.group("value").split(",") if k.strip())
                    current = None
                elif field:
                    current = field.group("label")
                    fields[current] = field.group("value")
                elif link:
                    links.append({"label": link.group("label"), "url": link.group("url")})
                    current = None
                elif current is not None:
                    fields[current] = f"{fields[current]} {stripped}".strip()
                else:
                    prose.append(stripped)
            entries.append(
                {
                    "heading": heading,
                    "keys": keys,
                    "fields": fields,
                    "links": links,
                    "prose": "\n".join(prose),
                }
            )
        topics.append(
            {
                "file": str(path.relative_to(workspace)),
                "title": title.group("title") if title else path.stem,
                "entries": entries,
            }
        )
    return topics


def _export_literature_audits(workspace: Path) -> list[dict]:
    """Per-theorem prior-art audits: claim, date, nearest sources, full body."""
    audits: list[dict] = []
    root = workspace / "LITERATURE" / "audits"
    if not root.is_dir():
        return audits
    for path in sorted(root.glob("*.md")):
        match = _AUDIT_FILENAME.match(path.stem)
        text = path.read_text()
        sources: list[dict] = []
        for line in text.splitlines():
            match_source = _AUDIT_SOURCE.match(line)
            if match_source:
                sources.append(
                    {
                        "name": match_source.group("name").strip().replace("--", "–"),
                        "text": match_source.group("text"),
                    }
                )
            elif sources and line.startswith("  ") and line.strip():
                sources[-1]["text"] = f"{sources[-1]['text']} {line.strip()}"
            elif not line.strip():
                # A blank line closes the bullet list; later bullets are not sources.
                if sources:
                    break
        audits.append(
            {
                "file": str(path.relative_to(workspace)),
                "claim": match.group("claim") if match else None,
                "date": match.group("date").replace("-", " ") if match else None,
                "sources": sources,
                "body": text,
            }
        )
    return audits


def _export_reports(workspace: Path) -> list[dict]:
    """AUDITS/ reports: publication-grade audits and Lean statement audits."""
    reports: list[dict] = []
    root = workspace / "AUDITS"
    if not root.is_dir():
        return reports
    for path in sorted(root.glob("*.md")):
        text = path.read_text()
        verdict = _VERDICT.search(text)
        reports.append(
            {
                "file": str(path.relative_to(workspace)),
                "kind": "formalization" if path.name.startswith("FORMALIZATION-") else "audit",
                "title": text.splitlines()[0].lstrip("# ").strip() if text else path.stem,
                "verdict": verdict.group("verdict").strip() if verdict else None,
            }
        )
    return reports


def _export_literature_graph(workspace: Path) -> list[dict]:
    path = workspace / "LITERATURE" / "graph.json"
    if not path.is_file():
        return []
    graph = json.loads(path.read_text())
    keep = (
        "id",
        "title",
        "year",
        "authors",
        "bibliography_key",
        "research_area",
        "relevant_claims",
        "status",
        "note",
    )
    return [{k: paper[k] for k in keep if k in paper} for paper in graph.get("papers", [])]


def export_graph(workspace: Path = WORKSPACE) -> dict:
    """Return the whole research graph as one JSON-serialisable document.

    This is the read seam for anything outside the workspace (the public
    research atlas): claims with their proof prose, the counterexample fixtures,
    the bibliography with its curated annotations, the prior-art audits, the
    proof chapters, the audit reports and the programme titles. It applies no
    publication policy — every claim is exported — and derives nothing; the
    consumer owns both.
    """
    registry = load(workspace)
    vocab_keys = (
        "status_definitions",
        "publication_status_definitions",
        "literature_search_status_definitions",
        "formal_proof_contract",
        "levels",
        "criteria",
        "primary_objectives",
        "primary_problem_levels",
        "score_oracle_regimes",
    )
    claims = [
        {**claim, "proof": _export_proof(claim, workspace)} for claim in registry["claims"]
    ]
    fixtures = [
        json.loads(path.read_text())
        for path in sorted((workspace / "COUNTEREXAMPLES").glob("CE-*.json"))
    ]
    return {
        "schemaVersion": 1,
        "vocabularies": {key: registry[key] for key in vocab_keys if key in registry},
        "programmes": registry.get("programmes", {}),
        "claims": claims,
        "counterexamples": fixtures,
        "bibliography": registry["bibliography"],
        "literature": {
            "topics": _export_topics(workspace),
            "audits": _export_literature_audits(workspace),
            "graph": _export_literature_graph(workspace),
        },
        "chapters": _export_chapters(workspace),
        "reports": _export_reports(workspace),
    }


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """Run the registry command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="report every registry integrity violation")
    reindex_parser = sub.add_parser("reindex", help="regenerate the derived indexes")
    reindex_parser.add_argument("--check", action="store_true", help="report drift, write nothing")
    show_parser = sub.add_parser("show", help="print a claim, its deps, and its proof")
    show_parser.add_argument("claim_id")
    show_parser.add_argument("--deps", action="store_true")
    show_parser.add_argument("--proof", action="store_true")
    export_parser = sub.add_parser("export", help="write the whole research graph as JSON")
    export_parser.add_argument("--out", type=Path, default=None, help="file to write; default stdout")
    args = parser.parse_args(argv)

    if args.command == "export":
        document = json.dumps(export_graph(), indent=2, ensure_ascii=False) + "\n"
        if args.out is None:
            sys.stdout.write(document)
        else:
            args.out.write_text(document)
            print(f"wrote {args.out}")
        return 0

    if args.command == "validate":
        violations = validate()
        for violation in violations:
            print(violation)
        if violations:
            print(f"\n{len(violations)} violation(s)", file=sys.stderr)
            return 1
        print("registry clean")
        return 0
    if args.command == "reindex":
        changes = reindex(check=args.check)
        for change in changes:
            print(change)
        if args.check and changes:
            return 1
        if not changes:
            print("indexes current")
        return 0
    print(show(args.claim_id, deps=args.deps, proof=args.proof))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
