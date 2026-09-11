"""Project the research registry into the public Research Atlas data.

The registry (``agenticresearch/``) is the source of truth and is read only
through ``py/registry.py export``. This script owns everything the public
atlas adds on top of that export and nothing else:

* the publication policy (``website/content/research-public.json``);
* the provenance classes and typed edges derived from registry fields;
* the editorial notes keyed by entity id (``website/content/atlas/notes/``);
* the library map (``website/content/atlas/implementation.json``);
* the public vocabulary (``website/content/atlas/config.json``): level and
  criterion labels, frontier themes, chapter and tradition names;
* the deterministic layered layout of the argument map;
* the rewriting of internal work-tracking vocabulary into public terms.

It derives no mathematics. Output: ``website/src/generated/atlas.json``, typed
by ``website/src/data/atlas.ts`` and checked against a fresh run by
``tests/test_atlas_data.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import statistics
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
WEBSITE = ROOT / "website"
RESEARCH = ROOT / "agenticresearch"
CONTENT = WEBSITE / "content"
ATLAS_CONTENT = CONTENT / "atlas"
NOTES = ATLAS_CONTENT / "notes"
OUTPUT = WEBSITE / "src" / "generated" / "atlas.json"
GITHUB = "https://github.com/VitalyVorobyev/scorequant/blob/main/agenticresearch/"

SCHEMA_VERSION = 1

#: Public provenance classes, in the order the legend lists them.
PROVENANCE = [
    ("established", "Established", "A published result, used as it stands."),
    (
        "derived",
        "Derived from the literature",
        "A direct translation or synthesis of published results into this setting.",
    ),
    (
        "proved",
        "Proved here",
        "Derived in this project and treated as proved internally; not yet publication-audited "
        "against the literature.",
    ),
    (
        "proved_new",
        "Proved here, no precedent found",
        "Proved in this project, and a targeted prior-art search found no direct equivalent. "
        "A search gap is recorded, not a claim of priority; the nearest prior work is named "
        "beside it.",
    ),
    (
        "measured",
        "Measured",
        "Supported by numerical evidence only. Never theorem authority.",
    ),
    (
        "counterexample",
        "Counterexample",
        "An explicit example falsifies the stated or generalized claim.",
    ),
    ("open", "Open", "An unresolved question."),
    (
        "verification",
        "Verification record",
        "An independent audit of another result; it carries no result of its own.",
    ),
]

EDGE_TYPES = [
    ("rests_on", "rests on", "Proof prerequisite: the source uses the target in its proof."),
    ("enables", "enables", "The source is used by, or directly yields, the target."),
    ("raises", "raises", "The source motivates the target, which is open."),
    ("verified_by", "verified by", "An independent audit re-derived the source."),
    (
        "references",
        "references",
        "Reviewed inputs or supporting evidence, not proof prerequisites.",
    ),
    ("converse_fails", "converse fails", "The target shows the converse of the source is false."),
    ("refuted_by", "refuted by", "An exact fixture falsifies the source statement."),
    ("bounded_by", "bounded by", "An exact fixture sits on the edge of the source's hypotheses."),
    ("cites", "cites", "The source rests on this published work."),
]

_INTERNAL = re.compile(r"\bP[1-8]\b|\bOP\d+\b|\bpackets?\b|WORK/|\bprogrammes?\b", re.I)
_OP_TOKEN = re.compile(r"\bOP(\d+)\b")
_P_TOKEN = re.compile(r"\bP([1-8])\b")
_DISPLAY_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_YEAR = re.compile(r"-(\d{4})$")
_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def _registry_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_scorequant_registry", RESEARCH / "py" / "registry.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _public_names() -> set[str]:
    tree = ast.parse((ROOT / "src" / "scorequant" / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            return {
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
    raise RuntimeError("scorequant.__all__ was not found")


def _read_notes() -> dict[str, dict]:
    notes: dict[str, dict] = {}
    if not NOTES.is_dir():
        return notes
    for path in sorted(NOTES.glob("*.md")):
        match = _FRONTMATTER.match(path.read_text(encoding="utf-8"))
        if match is None:
            raise RuntimeError(f"{path}: note has no front matter")
        meta = dict(line.split(":", 1) for line in match.group(1).splitlines() if ":" in line)
        meta = {k.strip(): v.strip() for k, v in meta.items()}
        entity = meta.get("entity")
        if entity != path.stem:
            raise RuntimeError(f"{path}: entity {entity!r} does not match the file name")
        notes[path.stem] = {"kind": meta.get("kind", "claim"), "markdown": match.group(2).strip()}
    return notes


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
def _normalize_math(text: str) -> str:
    text = _DISPLAY_MATH.sub(lambda m: f"$${m.group(1)}$$", text)
    return _INLINE_MATH.sub(lambda m: f"${m.group(1)}$", text)


class Sanitizer:
    """Rewrite internal work-tracking vocabulary into public terms.

    Open-problem numbers become links to the open claim they name; programme
    numbers become the public theme name; the words for work packets become
    ordinary English. The result is checked by ``tests/test_atlas_data.py`` to
    contain none of the internal tokens.
    """

    def __init__(self, op_to_claim: dict[str, str], themes: dict[str, dict]) -> None:
        self.op_to_claim = op_to_claim
        self.themes = themes

    def _op(self, match: re.Match[str]) -> str:
        claim = self.op_to_claim.get(f"OP{match.group(1)}")
        return f"`{claim}`" if claim else "an earlier, since-closed question"

    def _programme(self, match: re.Match[str]) -> str:
        theme = self.themes.get(f"P{match.group(1)}")
        if theme is None:
            return "an earlier line of work"
        return f"“{theme['label']}”"

    def __call__(self, text: str | None) -> str | None:
        """Return ``text`` with internal vocabulary rewritten and math normalised."""
        if text is None:
            return None
        text = _OP_TOKEN.sub(self._op, text)
        text = _P_TOKEN.sub(self._programme, text)
        text = re.sub(r"\bWORK/artifacts/", "artifacts/", text)
        text = re.sub(r"\bWORK/(?:active|completed)/[A-Za-z0-9-]+\.md", "a work record", text)
        text = re.sub(r"\bWORK/", "", text)
        text = re.sub(r"\bpackets\b", "studies", text)
        text = re.sub(r"\bPackets\b", "Studies", text)
        text = re.sub(r"\bpacket\b", "study", text)
        text = re.sub(r"\bPacket\b", "Study", text)
        text = re.sub(r"\bprogrammes\b", "themes", text)
        text = re.sub(r"\bprogramme\b", "theme", text)
        text = re.sub(r"\bProgramme\b", "Theme", text)
        return _normalize_math(text)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _surnames(key: str) -> list[str]:
    parts = [
        p for p in _YEAR.sub("", key).split("-") if p and p.lower() not in {"et", "al", "etal"}
    ]
    return parts


# --------------------------------------------------------------------------- #
# derivations
# --------------------------------------------------------------------------- #
def _provenance(claim: dict) -> str:
    status = claim["status"]
    if claim["id"].startswith("AUDIT-"):
        return "verification"
    if status == "literature":
        return "established"
    if status == "bridge":
        return "derived"
    if status == "project_proved":
        return "proved_new" if claim.get("literature_search_status") == "search_gap" else "proved"
    if status in {"open", "conjecture"}:
        return "open"
    return status  # measured, counterexample


def _kind(claim: dict) -> str:
    if claim["id"].startswith("AUDIT-"):
        return "audit"
    return {
        "open": "question",
        "conjecture": "question",
        "counterexample": "counterexample",
        "measured": "evidence",
    }.get(claim["status"], "result")


def _lane(criteria: list[str], lanes: dict[str, str]) -> str:
    if "D" in criteria and "Ds" in criteria:
        return "D"
    for criterion in criteria:
        if criterion in ("D", "Ds"):
            return lanes[criterion]
    for criterion in criteria:
        if criterion != "general":
            return lanes[criterion]
    return "general"


def _layout(
    nodes: dict[str, dict], neighbours: dict[str, set[str]], levels: list[str], lanes: list[str]
) -> dict:
    """Deterministic layered layout: columns are levels, bands are lanes.

    Rows inside a lane are ordered by the barycentre of each node's neighbours
    in the other columns, swept a fixed number of times, so a proof chain that
    runs across levels reads as a near-horizontal line.
    """
    col_of = {nid: levels.index(n["level"]) for nid, n in nodes.items()}
    lane_of = {nid: lanes.index(n["lane"]) for nid, n in nodes.items()}
    order: dict[tuple[int, int], list[str]] = {}
    for nid in sorted(nodes):
        order.setdefault((col_of[nid], lane_of[nid]), []).append(nid)
    row: dict[str, float] = {}
    for members in order.values():
        for i, nid in enumerate(members):
            row[nid] = float(i)
    columns = range(len(levels))
    for sweep in range(8):
        sequence = columns if sweep % 2 == 0 else reversed(list(columns))
        for col in sequence:
            for lane in range(len(lanes)):
                members = order.get((col, lane))
                if not members:
                    continue

                def bary(nid: str, col: int = col) -> float:
                    others = [row[o] for o in neighbours.get(nid, ()) if col_of[o] != col]
                    return statistics.fmean(others) if others else row[nid]

                members.sort(key=lambda nid: (bary(nid), nid))
                for i, nid in enumerate(members):
                    row[nid] = float(i)
    lane_height = {
        lane: max((len(m) for (_, band), m in order.items() if band == lane), default=0)
        for lane in range(len(lanes))
    }
    bands = []
    y0 = 0
    for lane in range(len(lanes)):
        height = lane_height[lane]
        bands.append({"id": lanes[lane], "y0": y0, "y1": y0 + max(height, 1)})
        y0 += max(height, 1) + 1
    positions = {}
    for (col, lane), members in order.items():
        for nid in members:
            positions[nid] = {"x": col, "y": bands[lane]["y0"] + int(row[nid])}
    return {
        "columns": [{"level": level, "x": i} for i, level in enumerate(levels)],
        "bands": bands,
        "height": y0 - 1,
        "positions": positions,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build() -> dict:
    """Assemble the atlas document from the export and the website-side content."""
    registry = _registry_module()
    export = registry.export_graph(RESEARCH)
    policy = _read_json(CONTENT / "research-public.json")
    config = _read_json(ATLAS_CONTENT / "config.json")
    implementation = _read_json(ATLAS_CONTENT / "implementation.json")
    notes = _read_notes()
    public_names = _public_names()

    withheld = set(policy.get("withheld", []))
    claims = {c["id"]: c for c in export["claims"] if c["id"] not in withheld}
    fixtures = {f["id"]: f for f in export["counterexamples"]}
    level_ids = [level["id"] for level in config["levels"]]
    lane_ids = [lane["id"] for lane in config["lanes"]]
    criterion_lane = {c["id"]: c["lane"] for c in config["criteria"]}
    chapters_cfg = config["chapters"]
    strips_cfg = config["strips"]
    themes_cfg = config["themes"]

    # Open-problem labels resolve to the open claim that owns the section.
    op_to_claim = {
        c["proof"]["label"]: c["id"]
        for c in claims.values()
        if c.get("proof") and c["proof"].get("label", "") and c["proof"]["label"].startswith("OP")
    }
    clean = Sanitizer(op_to_claim, themes_cfg)

    # Reports (AUDITS/) by file.
    reports = {r["file"]: r for r in export["reports"]}

    def report_ref(path: str | None) -> dict | None:
        if not path:
            return None
        report = reports.get(path, {"kind": "audit", "verdict": None, "title": path})
        return {
            "file": path,
            "url": GITHUB + path,
            "kind": report["kind"],
            "verdict": clean(re.sub(r"\s+", " ", report["verdict"]).strip())
            if report.get("verdict")
            else None,
        }

    # Chapter membership: claim id -> chapter file, from proof_location.
    chapter_of = {
        cid: c["proof_location"]["file"]
        for cid, c in claims.items()
        if c["proof_location"]["file"] in chapters_cfg
    }
    strip_of_chapter = {ch: strip["id"] for strip in strips_cfg for ch in strip["chapters"]}
    strip_of_criterion = {crit: strip["id"] for strip in strips_cfg for crit in strip["criteria"]}

    # ---- typed edges ------------------------------------------------------ #
    edges: list[dict] = []
    errors: list[str] = []

    def target_ok(source: str, target: str) -> bool:
        if target in withheld:
            errors.append(f"{source}: edge to withheld claim {target}")
            return False
        return target in claims

    for cid, c in claims.items():
        for target in c.get("dependencies", []):
            if target_ok(cid, target):
                edges.append({"source": cid, "target": target, "type": "rests_on"})
        for relation in ("verified_by", "references"):
            for target in c.get(relation, []):
                if target_ok(cid, target):
                    edges.append({"source": cid, "target": target, "type": relation})
        for target in c.get("implies", []):
            if not target_ok(cid, target):
                continue
            kind = _kind(claims[target])
            etype = {"question": "raises", "audit": "verified_by"}.get(kind, "enables")
            edges.append({"source": cid, "target": target, "type": etype})
        for target in c.get("converse_failures", []):
            if target_ok(cid, target):
                edges.append({"source": cid, "target": target, "type": "converse_fails"})
        for fixture in c.get("counterexamples", []):
            if fixture in fixtures:
                edges.append({"source": cid, "target": fixture, "type": "refuted_by"})
        for fixture in c.get("boundary_counterexamples", []):
            if fixture in fixtures:
                edges.append({"source": cid, "target": fixture, "type": "bounded_by"})
        for key in c.get("literature", []):
            edges.append({"source": cid, "target": key, "type": "cites"})
    if errors:
        raise RuntimeError("publication policy violated:\n  " + "\n  ".join(errors))

    incoming: dict[str, list[dict]] = {}
    outgoing: dict[str, list[dict]] = {}
    for edge in edges:
        outgoing.setdefault(edge["source"], []).append(edge)
        incoming.setdefault(edge["target"], []).append(edge)

    # Strip (home overview) membership: chapter, else neighbours' chapter, else criterion.
    def strip_for(cid: str) -> str | None:
        c = claims[cid]
        if _kind(c) == "audit":
            return None
        chapter = chapter_of.get(cid)
        if chapter:
            return strip_of_chapter.get(chapter)
        votes: dict[str, int] = {}
        neighbour_ids = [
            e["source"] for e in incoming.get(cid, []) if e["type"] in ("raises", "enables")
        ]
        neighbour_ids += [e["target"] for e in outgoing.get(cid, []) if e["type"] == "rests_on"]
        for nid in neighbour_ids:
            ch = chapter_of.get(nid)
            if ch and ch in strip_of_chapter:
                votes[strip_of_chapter[ch]] = votes.get(strip_of_chapter[ch], 0) + 1
        if votes:
            return max(sorted(votes), key=lambda s: votes[s])
        for crit in c["criterion"]:
            if crit in strip_of_criterion:
                return strip_of_criterion[crit]
        return strip_of_criterion.get("general")

    # ---- claims ----------------------------------------------------------- #
    library_objects = implementation["objects"]
    library_refusals = implementation["refusals"]
    for obj in library_objects:
        if not obj.get("symbols"):
            raise RuntimeError(f"implementation.json: {obj['name']} names no public symbol")
        for symbol in obj["symbols"]:
            if symbol not in public_names:
                raise RuntimeError(
                    f"implementation.json names {symbol}, which scorequant does not export"
                )
        for cid in obj["claims"]:
            if cid not in claims:
                raise RuntimeError(
                    f"implementation.json: {obj['name']} maps to unknown claim {cid}"
                )
    for refusal in library_refusals:
        if refusal["code"] not in fixtures:
            raise RuntimeError(f"implementation.json: refusal {refusal['code']} is not a fixture")
        for cid in refusal["claims"]:
            if cid not in claims:
                raise RuntimeError(
                    f"implementation.json: {refusal['code']} maps to unknown claim {cid}"
                )
    implemented_by: dict[str, list[str]] = {}
    for obj in library_objects:
        for cid in obj["claims"]:
            implemented_by.setdefault(cid, []).append(obj["name"])
    enforced_by: dict[str, list[str]] = {}
    for refusal in library_refusals:
        for cid in refusal["claims"]:
            enforced_by.setdefault(cid, []).append(refusal["code"])

    for entity in notes:
        if entity not in claims and entity not in fixtures:
            raise RuntimeError(f"note {entity} refers to no published claim or fixture")

    public_claims: dict[str, dict] = {}
    for cid, c in sorted(claims.items()):
        proof = c.get("proof")
        formal = c.get("formal_proof")
        chapter_file = chapter_of.get(cid)
        theme = themes_cfg.get(c.get("programme", ""))
        scope = clean(c.get("warning"))
        public_claims[cid] = {
            "id": cid,
            "slug": _slug(cid),
            "kind": _kind(c),
            "title": clean(c["title"]),
            "statement": clean(c["statement"]),
            "assumptions": [clean(a) for a in c.get("assumptions", [])],
            "criterion": c["criterion"],
            "level": c["level"],
            "status": c["status"],
            "publicationStatus": c["publication_status"],
            "searchStatus": c.get("literature_search_status"),
            "provenance": _provenance(c),
            "role": clean(c.get("role")),
            "scope": scope,
            "parked": bool(
                re.search(
                    r"\bparked\b", (c.get("warning") or "") + (proof["body"] if proof else ""), re.I
                )
            ),
            "theme": theme["slug"] if theme else None,
            "strip": strip_for(cid),
            "chapter": (
                {
                    "file": chapter_file,
                    "slug": chapters_cfg[chapter_file]["slug"],
                    "label": chapters_cfg[chapter_file]["label"],
                    "section": proof["label"] if proof else None,
                }
                if chapter_file
                else None
            ),
            "proof": (
                {
                    "label": None if (proof["label"] or "").startswith("OP") else proof["label"],
                    "title": clean(proof["title"]),
                    "file": proof["file"],
                    "url": GITHUB + proof["file"],
                    "markdown": clean(proof["body"]),
                }
                if proof
                else None
            ),
            "audit": report_ref(c.get("audit")),
            "artifact": (
                {"path": c["artifact"], "url": GITHUB + c["artifact"]}
                if c.get("artifact")
                else None
            ),
            "machineChecked": (
                {
                    "system": formal["system"],
                    "declaration": formal["declaration"],
                    "spec": {"path": formal["spec"], "url": GITHUB + formal["spec"]},
                    "proof": {"path": formal["file"], "url": GITHUB + formal["file"]},
                    "statementAudit": report_ref(formal["statement_audit"]),
                }
                if formal
                else None
            ),
            "note": notes[cid]["markdown"] if cid in notes else None,
            "literature": c.get("literature", []),
            "implementedBy": sorted(implemented_by.get(cid, [])),
            "enforcedBy": sorted(enforced_by.get(cid, [])),
        }

    # ---- fixtures --------------------------------------------------------- #
    cited_by: dict[str, list[dict]] = {}
    for edge in edges:
        if edge["type"] in ("refuted_by", "bounded_by"):
            cited_by.setdefault(edge["target"], []).append(
                {"claim": edge["source"], "type": edge["type"]}
            )
    refusal_of = {r["code"]: r for r in library_refusals}
    public_fixtures: dict[str, dict] = {}
    for fid, f in sorted(fixtures.items()):
        scores = f["scores"]
        dimension = len(scores[0]) if scores and isinstance(scores[0], list) else 1
        criterion = f["criterion"] if isinstance(f["criterion"], list) else [f["criterion"]]
        public_fixtures[fid] = {
            "id": fid,
            "slug": _slug(fid),
            "criterion": criterion,
            "level": f["level"],
            "falsifies": clean(f["claim_falsified"]),
            "scores": scores,
            "weights": f["weights"],
            "K": f["K"],
            "dimension": dimension,
            "labelsBefore": f["labels_before"],
            "labelsAfter": f.get("labels_after_or_optimum"),
            "poi": f.get("poi_indices"),
            "nuisance": f.get("nuisance_indices"),
            "objectiveBefore": f.get("objective_before"),
            "objectiveAfter": f.get("objective_after"),
            "verification": {
                "method": f.get("verification", {}).get("method"),
                "notes": clean(f.get("verification", {}).get("notes")),
            },
            "consequences": [clean(x) for x in f.get("consequences", [])],
            "date": f.get("date"),
            "url": GITHUB + f"COUNTEREXAMPLES/{fid}.json",
            "citedBy": sorted(cited_by.get(fid, []), key=lambda x: x["claim"]),
            "refusal": (
                {
                    "trigger": refusal_of[fid]["trigger"],
                    "reason": refusal_of[fid]["reason"],
                    "remedy": refusal_of[fid]["remedy"],
                }
                if fid in refusal_of
                else None
            ),
            "note": notes[fid]["markdown"] if fid in notes else None,
        }

    # ---- literature ------------------------------------------------------- #
    traditions_cfg = config["traditions"]
    graph_by_key = {
        p["bibliography_key"]: p for p in export["literature"]["graph"] if p.get("bibliography_key")
    }
    annotation_of: dict[str, dict] = {}
    tradition_notes: dict[str, list[dict]] = {}
    for topic in export["literature"]["topics"]:
        tradition = traditions_cfg.get(topic["file"])
        if tradition is None:
            raise RuntimeError(f"no tradition configured for {topic['file']}")
        for entry in topic["entries"]:
            record = {
                "heading": clean(entry["heading"]),
                "fields": {k: clean(v) for k, v in entry["fields"].items()},
                "links": entry["links"],
                "prose": clean(entry["prose"]),
                "tradition": tradition["slug"],
            }
            if entry["keys"]:
                for key in entry["keys"]:
                    annotation_of[key] = record
            else:
                tradition_notes.setdefault(tradition["slug"], []).append(record)
    cites_from: dict[str, list[str]] = {}
    for edge in edges:
        if edge["type"] == "cites":
            cites_from.setdefault(edge["target"], []).append(edge["source"])
    audits_by_claim: dict[str, list[dict]] = {}
    for audit in export["literature"]["audits"]:
        if audit["claim"] in claims:
            audits_by_claim.setdefault(audit["claim"], []).append(
                {
                    "date": audit["date"],
                    "url": GITHUB + audit["file"],
                    "sources": [
                        {"name": s["name"], "text": clean(s["text"])} for s in audit["sources"]
                    ],
                }
            )
    author_names = config["authorNames"]
    papers: dict[str, dict] = {}
    authors: dict[str, dict] = {}
    for key, entry in sorted(export["bibliography"].items()):
        record = graph_by_key.get(key, {})
        year_match = _YEAR.search(key)
        year = record.get("year") or (int(year_match.group(1)) if year_match else None)
        surnames = _surnames(key)
        names = [author_names.get(s, s) for s in surnames]
        annotation = annotation_of.get(key)
        slug = _slug(key)
        papers[key] = {
            "key": key,
            "slug": slug,
            "title": entry["title"],
            "year": year,
            "authors": [{"slug": _slug(n), "name": n} for n in names],
            "fullAuthors": record.get("authors", []),
            "doi": entry.get("doi"),
            "arxiv": entry.get("arxiv"),
            "pdf": entry.get("pdf"),
            "note": clean(entry.get("note")),
            "tradition": annotation["tradition"] if annotation else None,
            "annotation": annotation,
            "area": record.get("research_area"),
            "citedBy": sorted(cites_from.get(key, [])),
            "relevantTo": sorted(c for c in record.get("relevant_claims", []) if c in claims),
        }
        for name in names:
            author = authors.setdefault(
                _slug(name), {"slug": _slug(name), "name": name, "papers": []}
            )
            author["papers"].append(key)
    for cid, c in public_claims.items():
        c["priorArt"] = audits_by_claim.get(cid, [])

    traditions = [
        {
            "slug": t["slug"],
            "label": t["label"],
            "title": clean(
                next(x["title"] for x in export["literature"]["topics"] if x["file"] == file)
            ),
            "papers": sorted(
                (k for k, p in papers.items() if p["tradition"] == t["slug"]),
                key=lambda k: (papers[k]["year"] or 0, k),
            ),
            "notes": tradition_notes.get(t["slug"], []),
        }
        for file, t in traditions_cfg.items()
    ]

    # ---- chapters, themes, strips ---------------------------------------- #
    chapters = []
    for chapter in export["chapters"]:
        cfg = chapters_cfg.get(chapter["file"])
        if cfg is None:
            raise RuntimeError(f"no public label configured for {chapter['file']}")
        chapters.append(
            {
                "file": chapter["file"],
                "slug": cfg["slug"],
                "label": cfg["label"],
                "url": GITHUB + chapter["file"],
                "sections": [
                    {
                        "label": s["label"],
                        "title": clean(s["title"]),
                        "claims": [cid for cid in s["claims"] if cid in claims],
                    }
                    for s in chapter["sections"]
                ],
            }
        )
    raised_count = {
        cid: sum(1 for e in incoming.get(cid, []) if e["type"] == "raises") for cid in claims
    }
    themes = []
    for programme, cfg in themes_cfg.items():
        members = [cid for cid, c in public_claims.items() if c["theme"] == cfg["slug"]]
        members.sort(key=lambda cid: (-raised_count[cid], cid))
        themes.append({**cfg, "programme": programme, "claims": members})
    strips = [
        {
            "id": strip["id"],
            "label": strip["label"],
            "claims": sorted(cid for cid, c in public_claims.items() if c["strip"] == strip["id"]),
        }
        for strip in strips_cfg
    ]

    # ---- map layout ------------------------------------------------------- #
    nodes = {}
    for cid, c in public_claims.items():
        if c["kind"] == "audit":
            continue
        nodes[cid] = {"level": c["level"], "lane": _lane(c["criterion"], criterion_lane)}
    for fid, f in public_fixtures.items():
        citing = [x["claim"] for x in f["citedBy"] if x["claim"] in nodes]
        lane = nodes[citing[0]]["lane"] if citing else _lane(f["criterion"], criterion_lane)
        nodes[fid] = {"level": f["level"], "lane": lane}
    neighbours: dict[str, set[str]] = {}
    for edge in edges:
        if (
            edge["type"] in ("rests_on", "refuted_by", "bounded_by", "converse_fails")
            and edge["source"] in nodes
            and edge["target"] in nodes
        ):
            neighbours.setdefault(edge["source"], set()).add(edge["target"])
            neighbours.setdefault(edge["target"], set()).add(edge["source"])
    layout = _layout(nodes, neighbours, level_ids, lane_ids)
    for nid, node in nodes.items():
        layout["positions"][nid]["lane"] = node["lane"]

    # ---- formal ----------------------------------------------------------- #
    formal_chain = []
    for module in config["formalChain"]:
        spec = f"formal/ScoreQuantFormal/{module['module']}Spec.lean"
        proof = f"formal/ScoreQuantFormal/{module['module']}.lean"
        formal_chain.append(
            {
                **module,
                "claims": [cid for cid in module.get("claims", []) if cid in claims],
                "spec": {"path": spec, "url": GITHUB + spec}
                if (RESEARCH / spec).is_file()
                else None,
                "proof": {"path": proof, "url": GITHUB + proof}
                if (RESEARCH / proof).is_file()
                else None,
            }
        )

    home = _read_json(ATLAS_CONTENT / "home.json")
    summaries = _read_json(ATLAS_CONTENT / "summaries.json")
    validate_editorial(home, summaries, public_claims)
    for cid, entry in summaries.items():
        public_claims[cid]["editorial"] = entry
    for claim in public_claims.values():
        claim.setdefault("editorial", None)

    return {
        "schemaVersion": SCHEMA_VERSION,
        "vocabulary": {
            "provenance": [
                {"id": key, "label": label, "description": text} for key, label, text in PROVENANCE
            ],
            "edgeTypes": [
                {"id": key, "label": label, "description": text} for key, label, text in EDGE_TYPES
            ],
            "levels": config["levels"],
            "criteria": config["criteria"],
            "lanes": config["lanes"],
        },
        "home": home,
        "headline": [home["central"]["id"], *[entry["id"] for entry in home["results"]]],
        "frontier": [entry["id"] for entry in home["questions"]],
        "claims": public_claims,
        "fixtures": public_fixtures,
        "papers": papers,
        "authors": dict(sorted(authors.items())),
        "traditions": traditions,
        "chapters": chapters,
        "themes": themes,
        "strips": strips,
        "edges": edges,
        "layout": layout,
        "library": {"objects": library_objects, "refusals": library_refusals},
        "formal": {
            "chain": formal_chain,
            "outside": config["formalOutside"],
            "axioms": config["axioms"],
            "readme": GITHUB + "formal/README.md",
        },
    }


def validate_editorial(home: dict, summaries: dict, claims: dict) -> None:
    """Reject stale editorial selections rather than substituting registry prose."""
    required = {cid for cid, claim in claims.items() if claim["kind"] != "audit"}
    if set(summaries) != required:
        raise RuntimeError("summaries.json must cover exactly every non-audit claim")
    for cid, entry in summaries.items():
        for field in ("title", "summary"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise RuntimeError(f"{cid}: missing editorial {field}")
    if not isinstance(home.get("intro"), str) or not home["intro"].strip():
        raise RuntimeError("home.json requires an introduction")
    seen = set()
    for section, count, kind in (
        ("central", 1, "result"),
        ("results", 3, "result"),
        ("boundaries", 2, "counterexample"),
        ("questions", 3, "question"),
    ):
        entries = [home.get(section)] if section == "central" else home.get(section, [])
        if not isinstance(entries, list) or len(entries) != count:
            raise RuntimeError(f"home.json: {section} requires {count} entries")
        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError(f"home.json: invalid {section} entry")
            cid = entry.get("id")
            if cid not in claims or cid in seen:
                raise RuntimeError(f"home.json: missing or duplicate claim {cid}")
            seen.add(cid)
            claim = claims[cid]
            allowed = (
                {"established", "derived", "proved", "proved_new"}
                if kind == "result"
                else {"counterexample" if kind == "counterexample" else "open"}
            )
            if claim["kind"] != kind or claim["provenance"] not in allowed:
                raise RuntimeError(f"home.json: invalid kind/status for {cid}")
            for field in ("title", "summary", "significance", "qualification"):
                if not isinstance(entry.get(field), str) or not entry[field].strip():
                    raise RuntimeError(f"home.json: {cid} requires {field}")


def render() -> str:
    """Return the atlas document as the JSON text that is committed."""
    return json.dumps(build(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> None:
    """Write ``atlas.json``."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
