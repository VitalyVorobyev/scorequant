"""Integrity checks for the agenticresearch scientific memory.

These tests never do mathematics; they keep the claim registry, the
counterexample bank, and the human ledgers referentially consistent so that
research agents can trust the graph. If a workspace restructure breaks a
`proof_location`, drops a fixture, or leaves a completed packet's pointer
behind, CI fails here instead of silently corrupting the memory.

The checks themselves live in `agenticresearch/py/registry.py` so that a
bookkeeping session can run exactly what CI runs, without pytest:

    python agenticresearch/py/registry.py validate
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

WORKSPACE = Path(__file__).parents[1] / "agenticresearch"
REGISTRY_MODULE = WORKSPACE / "py" / "registry.py"


def _load_registry_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_scorequant_registry", REGISTRY_MODULE)
    assert spec is not None and spec.loader is not None, REGISTRY_MODULE
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    assert REGISTRY_MODULE.is_file(), (
        f"registry tool not found at {REGISTRY_MODULE}; if the research workspace "
        "moved, update WORKSPACE here and in test_research_claims.py"
    )
    return _load_registry_module()


def test_registry_loads_with_one_file_per_claim(tool: ModuleType) -> None:
    registry = tool.load(WORKSPACE)
    claims = registry["claims"]
    assert claims, "claim registry is empty"
    ids = {claim["id"] for claim in claims}
    assert len(ids) == len(claims), "duplicate claim ids"
    on_disk = {path.stem for path in (WORKSPACE / "claims").glob("*.json")}
    assert ids == on_disk, "claims/ and the loaded registry disagree"


def test_registry_is_referentially_consistent(tool: ModuleType) -> None:
    violations = tool.validate(WORKSPACE)
    assert not violations, "\n".join(["registry integrity violations:", *violations])


def test_programme_claims_display_their_status(tool: ModuleType) -> None:
    registry = {
        "programmes": {
            "P1": {
                "title": "Example programme",
                "rank": 1,
                "readiness": "ready",
            }
        },
        "claims": [
            {
                "id": "EXAMPLE-MEASURED",
                "title": "Measured evidence",
                "status": "measured",
                "programme": "P1",
                "proof_location": {"file": "KNOWN_RESULTS/example.md"},
            },
            {
                "id": "EXAMPLE-OPEN",
                "title": "Open question",
                "status": "open",
                "programme": "P1",
                "proof_location": {"file": "OPEN_PROBLEMS.md"},
            },
        ],
    }

    rendered = tool.render_index(registry)

    assert "`EXAMPLE-MEASURED` — Measured evidence · status: `measured`" in rendered
    assert "`EXAMPLE-OPEN` — Open question · status: `open`" in rendered


def test_generated_indexes_are_current(tool: ModuleType) -> None:
    stale = tool.reindex(WORKSPACE, check=True)
    assert not stale, "\n".join(
        ["stale generated indexes; run `python agenticresearch/py/registry.py reindex`", *stale]
    )


def test_formal_proof_is_attached_only_where_a_spec_is_frozen(tool: ModuleType) -> None:
    """Machine-checked evidence names an audited spec and a unique declaration."""
    registry = tool.load(WORKSPACE)
    index = tool.claims_by_id(registry)

    marked = {
        claim_id: claim["formal_proof"]
        for claim_id, claim in index.items()
        if claim.get("formal_proof") is not None
    }
    assert marked, "no claim carries machine-checked evidence"

    for claim_id, formal in marked.items():
        assert set(formal) == {"system", "spec", "file", "declaration", "statement_audit"}
        for field in ("spec", "file", "statement_audit"):
            assert (WORKSPACE / formal[field]).is_file(), f"{claim_id}: missing {field}"
        assert index[claim_id]["status"] not in {
            "open",
            "conjecture",
            "measured",
            "counterexample",
        }

    declarations = [formal["declaration"] for formal in marked.values()]
    assert len(declarations) == len(set(declarations)), "a Lean theorem is claimed twice"

    expected = {
        "D-EXCHANGE-IMPLIES-VORONOI": "ScoreQuantFormal.exchange_voronoi",
        "D-RANK2-MOVE": "ScoreQuantFormal.rank_two_relocation",
        "D-LOGDET-GAIN": "ScoreQuantFormal.det_relocation_gain",
        "D-LEVERAGE": "ScoreQuantFormal.leverage_inequality",
    }
    for claim_id, declaration in expected.items():
        assert marked[claim_id]["declaration"] == declaration, claim_id

    # The marked declaration's *type* must be the frozen conclusion, not a
    # restatement of it, or the mark can drift from the audited statement while
    # still compiling.
    for claim_id, declaration in expected.items():
        source = (WORKSPACE / marked[claim_id]["file"]).read_text()
        local = declaration.rsplit(".", 1)[-1]
        body = source.split(f"theorem {local}", 1)[1].split(":= by", 1)[0]
        assert "Conclusion" in body, f"{claim_id}: {local} does not name a frozen conclusion"

    # Claims whose statement is only partly formalized must stay unmarked.
    for unmarked in ("D-GLOBAL-GEOMETRIC-REALIZABILITY", "D-EXCHANGE-TERMINATES"):
        assert index[unmarked].get("formal_proof") is None, unmarked

    rendered = tool.render_index(registry)
    assert "machine-checked: `ScoreQuantFormal.exchange_voronoi`" in rendered


def test_formal_proof_validator_rejects_a_dangling_declaration(tool: ModuleType) -> None:
    """A declaration absent from its Lean file is a registry violation."""
    out: list[str] = []
    registry = {
        "claims": [
            {
                "id": "EXAMPLE-FORMAL",
                "status": "project_proved",
                "formal_proof": {
                    "system": "Lean 4.33.1 + Mathlib 4.33.1",
                    "spec": "formal/ScoreQuantFormal/ExchangeVoronoiSpec.lean",
                    "file": "formal/ScoreQuantFormal/ExchangeVoronoi.lean",
                    "declaration": "ScoreQuantFormal.no_such_theorem",
                    "statement_audit": "AUDITS/FORMALIZATION-D-EXCHANGE-IMPLIES-VORONOI-001.md",
                },
            }
        ]
    }
    tool._check_formal_proofs(registry, WORKSPACE, out)
    assert any("declares no theorem named no_such_theorem" in line for line in out)
