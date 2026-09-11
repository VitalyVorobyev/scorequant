"""Guard the generated Research Atlas data against drift and leakage.

``website/scripts/generate_atlas.py`` projects the registry export into
``website/src/generated/atlas.json``. This module checks that the committed
document equals a fresh run, that every website-side pointer (notes, headline
config, library map) resolves to a published entity, that the public graph has
no dangling edge, and that no internal work-tracking vocabulary reaches a public
string. The registry itself is validated by ``test_research_registry.py``.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "website" / "scripts" / "generate_atlas.py"
ATLAS = ROOT / "website" / "src" / "generated" / "atlas.json"
NOTES = ROOT / "website" / "content" / "atlas" / "notes"

#: Tokens that name internal work tracking rather than mathematics.
_INTERNAL = re.compile(r"\bP[1-8]\b|\bOP\d+\b|\bpackets?\b|WORK/|\bprogrammes?\b|\bledger\b", re.I)
#: URLs are exempt: a GitHub path may legitimately contain `WORK/artifacts`.
_URL = re.compile(r"https?://\S+")


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_scorequant_generate_atlas", GENERATOR)
    assert spec is not None and spec.loader is not None, GENERATOR
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    assert GENERATOR.is_file(), GENERATOR
    return _load_generator()


@pytest.fixture(scope="module")
def committed() -> dict:
    assert ATLAS.is_file(), "atlas.json is missing; run `pnpm generate:atlas` in website/"
    return json.loads(ATLAS.read_text(encoding="utf-8"))


def test_committed_atlas_matches_a_fresh_run(generator: ModuleType, committed: dict) -> None:
    fresh = json.loads(generator.render())
    assert fresh == committed, (
        "website/src/generated/atlas.json is stale; run `pnpm generate:atlas` in website/"
    )


def test_every_edge_resolves(committed: dict) -> None:
    claims = committed["claims"]
    fixtures = committed["fixtures"]
    papers = committed["papers"]
    for edge in committed["edges"]:
        assert edge["source"] in claims, edge
        target_pool = {
            "refuted_by": fixtures,
            "bounded_by": fixtures,
            "cites": papers,
        }.get(edge["type"], claims)
        assert edge["target"] in target_pool, edge


def test_layout_covers_every_map_node(committed: dict) -> None:
    positions = committed["layout"]["positions"]
    for cid, claim in committed["claims"].items():
        if claim["kind"] == "audit":
            assert cid not in positions, cid
        else:
            assert cid in positions, cid
    for fid in committed["fixtures"]:
        assert fid in positions, fid
    seen: set[tuple[int, int]] = set()
    for nid, position in positions.items():
        key = (position["x"], position["y"])
        assert key not in seen, f"{nid} shares a cell with another node"
        seen.add(key)


def test_notes_and_config_point_at_published_entities(committed: dict) -> None:
    claims = committed["claims"]
    fixtures = committed["fixtures"]
    for path in sorted(NOTES.glob("*.md")):
        assert path.stem in claims or path.stem in fixtures, path.name
    for cid in committed["headline"] + committed["frontier"]:
        assert cid in claims, cid
    for cid in committed["frontier"]:
        assert claims[cid]["kind"] == "question", cid
    for obj in committed["library"]["objects"]:
        for cid in obj["claims"]:
            assert cid in claims, (obj["name"], cid)
    for refusal in committed["library"]["refusals"]:
        assert refusal["code"] in fixtures, refusal["code"]


def _public_strings(value: object, path: str = "") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        return [
            item
            for key, sub in value.items()
            if key not in {"url", "file", "path", "doi", "pdf", "arxiv", "declaration"}
            for item in _public_strings(sub, f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            item for i, sub in enumerate(value) for item in _public_strings(sub, f"{path}[{i}]")
        ]
    return []


def test_no_internal_vocabulary_reaches_the_public_text(committed: dict) -> None:
    hits = []
    for path, text in _public_strings({k: v for k, v in committed.items() if k != "themes"}):
        stripped = _URL.sub("", text)
        for match in _INTERNAL.finditer(stripped):
            hits.append(f"{path}: {match.group(0)!r}")
    # The frontier themes carry the programme id on purpose (a key, never rendered).
    assert not hits, "internal vocabulary in public strings:\n  " + "\n  ".join(hits[:40])


def test_provenance_is_derived_from_status(committed: dict) -> None:
    for cid, claim in committed["claims"].items():
        status, provenance = claim["status"], claim["provenance"]
        if cid.startswith("AUDIT-"):
            assert provenance == "verification", cid
        elif status == "project_proved":
            expected = "proved_new" if claim["searchStatus"] == "search_gap" else "proved"
            assert provenance == expected, cid
        elif status == "literature":
            assert provenance == "established", cid
        elif status == "bridge":
            assert provenance == "derived", cid
        else:
            assert provenance == status, cid
        if claim["machineChecked"] is not None:
            assert status not in {"open", "conjecture", "measured", "counterexample"}, cid


def test_editorial_coverage_and_home_contract(generator: ModuleType, committed: dict) -> None:
    summaries = {
        cid: claim["editorial"]
        for cid, claim in committed["claims"].items()
        if claim["kind"] != "audit"
    }
    generator.validate_editorial(committed["home"], summaries, committed["claims"])
    assert len(summaries) == sum(c["kind"] != "audit" for c in committed["claims"].values())
    for entry in summaries.values():
        assert len(entry["title"].split()) <= 15
        assert len(entry["summary"].split()) <= 65


@pytest.mark.parametrize("failure", ["missing", "duplicate", "count", "kind", "status", "copy"])
def test_editorial_rejects_invalid_featured_content(
    generator: ModuleType, committed: dict, failure: str
) -> None:
    import copy

    home = copy.deepcopy(committed["home"])
    claims = copy.deepcopy(committed["claims"])
    summaries = {cid: c["editorial"] for cid, c in claims.items() if c["kind"] != "audit"}
    if failure == "missing":
        home["central"]["id"] = "NONEXISTENT"
    elif failure == "duplicate":
        home["results"][0]["id"] = home["central"]["id"]
    elif failure == "count":
        home["questions"].pop()
    elif failure == "kind":
        home["results"][0]["id"] = home["questions"][0]["id"]
    elif failure == "status":
        claims[home["central"]["id"]]["provenance"] = "measured"
    else:
        home["central"]["qualification"] = ""
    with pytest.raises(RuntimeError):
        generator.validate_editorial(home, summaries, claims)


def test_editorial_rejects_missing_summary(generator: ModuleType, committed: dict) -> None:
    summaries = {
        cid: c["editorial"] for cid, c in committed["claims"].items() if c["kind"] != "audit"
    }
    summaries.pop(committed["home"]["central"]["id"])
    with pytest.raises(RuntimeError, match="every non-audit"):
        generator.validate_editorial(committed["home"], summaries, committed["claims"])


def test_audits_remain_accessible_outside_the_proof_graph(committed: dict) -> None:
    claims = committed["claims"]
    for edge in committed["edges"]:
        if edge["type"] == "rests_on":
            assert claims[edge["source"]]["kind"] != "audit"
            assert claims[edge["target"]]["kind"] not in {"audit", "evidence"}
    audit = "AUDIT-DS-PRACTICAL-CERTIFIED-SOLVER"
    assert any(e["source"] == audit and e["type"] == "references" for e in committed["edges"])
    assert any(e["target"] == audit and e["type"] == "verified_by" for e in committed["edges"])
