"""Guard the portal walkthroughs' numbers against the evidence behind them.

The four walkthrough pages under ``website/walkthroughs/`` may not print a
number that no run produced. ``website/scripts/generate_walkthroughs.py``
enforces the first half of that by resolving every displayed value from a JSON
Pointer into a committed evidence file; this module enforces the second half by
re-resolving every pointer independently and checking the generated file still
agrees with the evidence.

Without this, the generator could compute anything and call it traceable, and a
re-run of any example would silently leave the portal quoting superseded
numbers.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "website" / "scripts" / "generate_walkthroughs.py"
GENERATED = ROOT / "website" / "src" / "generated" / "walkthrough-data.json"


def _load_generator() -> ModuleType:
    name = "_scorequant_generate_walkthroughs"
    spec = importlib.util.spec_from_file_location(name, GENERATOR)
    assert spec is not None and spec.loader is not None, GENERATOR
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because the generator defines a dataclass under
    # ``from __future__ import annotations``; resolving its field types looks the
    # defining module up in ``sys.modules``.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    assert GENERATOR.is_file(), GENERATOR
    return _load_generator()


@pytest.fixture(scope="module")
def generated() -> dict[str, object]:
    return json.loads(GENERATED.read_text(encoding="utf-8"))


def test_generated_file_is_current(generator: ModuleType, generated: dict[str, object]) -> None:
    """The committed facts equal what the generator produces from today's evidence.

    CI builds the portal from the committed generated file rather than
    regenerating it, so an example re-run that is not followed by
    ``pnpm generate`` would otherwise publish stale numbers.
    """
    assert generator.build() == generated, (
        "website/src/generated/walkthrough-data.json is stale; "
        "run `uv run python website/scripts/generate_walkthroughs.py`"
    )


def test_every_fact_resolves_to_its_cited_evidence(generated: dict[str, object]) -> None:
    """Each fact's value is what its own JSON Pointer selects from its own evidence file.

    Resolved here with an independent pointer walk rather than the generator's,
    so a bug in the generator's resolver cannot validate itself.
    """
    documents: dict[str, object] = {}
    pages = generated["pages"]
    assert isinstance(pages, dict) and pages, "no walkthrough facts were generated"
    for page, facts in pages.items():
        assert isinstance(facts, dict)
        for key, fact in facts.items():
            path, _, pointer = str(fact["source"]).partition("#")
            evidence_path = ROOT / path
            assert evidence_path.is_file(), f"{page}.{key} cites missing evidence {path}"
            document = documents.setdefault(
                path, json.loads(evidence_path.read_text(encoding="utf-8"))
            )
            current: object = document
            for raw in pointer.lstrip("/").split("/"):
                token = raw.replace("~1", "/").replace("~0", "~")
                if isinstance(current, list):
                    current = current[int(token)]
                else:
                    assert isinstance(current, dict), f"{page}.{key}: {pointer} hit a scalar"
                    assert token in current, f"{page}.{key}: {pointer} has no {token!r}"
                    current = current[token]
            assert current == fact["value"], f"{page}.{key} no longer matches {fact['source']}"


def test_every_fact_text_renders_its_own_value(generated: dict[str, object]) -> None:
    """The displayed string is a faithful rendering of the value it claims to show.

    A page prints ``text`` and never ``value``, so a text that disagrees with
    its value would put an unbacked number on the site while every pointer
    still resolved.
    """
    pages = generated["pages"]
    assert isinstance(pages, dict)
    for page, facts in pages.items():
        assert isinstance(facts, dict)
        for key, fact in facts.items():
            value, text = fact["value"], str(fact["text"])
            where = f"{page}.{key}"
            if isinstance(value, str):
                assert text == value, where
                continue
            assert isinstance(value, (int, float)), where
            cleaned = text.replace(",", "").rstrip("%")
            shown = float(cleaned)
            if text.endswith("%"):
                shown /= 100.0
            decimals = len(cleaned.partition(".")[2].partition("e")[0])
            tolerance = max(0.5 * 10.0**-decimals, abs(float(value)) * 1e-9)
            assert abs(shown - float(value)) <= tolerance, (
                f"{where}: text {text!r} does not render value {value!r}"
            )


def test_facts_only_cite_declared_evidence(
    generator: ModuleType, generated: dict[str, object]
) -> None:
    """No fact sources a number from a file outside the declared evidence set."""
    allowed = set(generator.EVIDENCE)
    pages = generated["pages"]
    assert isinstance(pages, dict)
    for page, facts in pages.items():
        assert isinstance(facts, dict)
        for key, fact in facts.items():
            path = str(fact["source"]).partition("#")[0]
            assert path in allowed, f"{page}.{key} cites undeclared evidence {path}"


# Numbers a walkthrough may write as a literal, each with the reason it is not a
# measurement. Anything else must come through `factsFor`, so that adding an
# unbacked number to a page is a deliberate, reviewed edit here rather than a
# silent one in prose.
_LITERAL_ALLOWLIST: dict[str, str] = {
    "1": "counting word: one door, one fit, one parameter",
    "2": "counting word: two contracts, two components, two columns",
    "3": "counting word: three doors",
    "4": "counting word: four walkthroughs",
    "10.5281": "the DOI prefix of the HEP fixture's Zenodo record",
    "3.12": "the minimum supported Python version, not a measurement",
    "1881": "the year of Michelson's first interferometer, a date rather than a measurement",
    "1887": "the year of the Michelson–Morley experiment, a date rather than a measurement",
}

_FENCE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
_JSX_EXPRESSION = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
_JSX_TAG = re.compile(r"<[^>]+>", re.DOTALL)
_FRONTMATTER = re.compile(r"\A---.*?^---", re.DOTALL | re.MULTILINE)
#: Displayed and inline TeX. A model's equations carry structural constants
#: (a reference point of zero, a normaliser of one, the period of a cosine),
#: which are not measurements; a measured value still reaches an equation
#: through a fact, so nothing numeric a run produced is typed inside math.
_MATH = re.compile(r"\$\$.*?\$\$|(?<!\\)\$[^$\n]+\$", re.DOTALL)
# MDX `import`/`export` statements are JavaScript, including any continuation
# lines, which are indented or close a bracket opened on the statement's line.
_ESM = re.compile(r"^(?:import|export)\s[^\n]*(?:\n(?:[ \t)\]}][^\n]*|)(?=[^\n]))*", re.MULTILINE)
# A measurement, as distinct from a digit inside a name. The boundary classes
# exclude digits adjacent to letters, hyphens or dots, so flow-cytometry marker
# nomenclature (``CD34``, ``FL5 INT_CD34-PC7``), licence identifiers
# (``CC-BY-4.0``) and record ids (``zenodo.15131565``) are not mistaken for
# numbers a page measured.
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9._\-])\d+(?:[.,]\d+)*(?:e[+-]?\d+)?(?![A-Za-z0-9._\-])",
    re.IGNORECASE,
)


def _walkthrough_prose(text: str) -> str:
    """Return a page's prose, with code, math, JSX expressions and tags removed.

    A fact reaches the page as ``{f("key")}``, which is a JSX expression, so
    everything legitimately numeric disappears here and what remains is
    hand-typed prose.
    """
    for pattern in (_FRONTMATTER, _FENCE, _MATH, _ESM, _JSX_EXPRESSION, _JSX_TAG):
        text = pattern.sub(" ", text)
    return text


#: MDX content roots whose prose is held to the fact contract. A page under one
#: of these may not type a measurement.
_GUARDED_CONTENT_ROOTS = ("walkthroughs", "get-started")

#: Content roots deliberately outside the guard, each with the reason. The meta
#: test below fails if a new docs instance appears in neither collection, so a
#: route cannot be added without someone deciding which side it is on.
_UNGUARDED_CONTENT_ROOTS: dict[str, str] = {}
# The research section is no longer an MDX docs instance: the Research Atlas
# (website/plugins/research-atlas) generates its routes from the registry export,
# and tests/test_atlas_data.py is its own guard (ADR 0033).


def _guarded_prose_pages() -> list[Path]:
    """Every MDX page whose prose the fact contract governs."""
    pages: list[Path] = []
    for root in _GUARDED_CONTENT_ROOTS:
        pages.extend(sorted((ROOT / "website" / root).glob("*.mdx")))
    return pages


def test_every_mdx_content_root_is_classified() -> None:
    """No content route can be added without deciding whether the guard covers it.

    The guard used to be a glob over one directory, so a new route simply went
    unguarded and nothing said so. Here the declared roots are checked against
    the docs instances `docusaurus.config.ts` actually registers.
    """
    config = (ROOT / "website/docusaurus.config.ts").read_text(encoding="utf-8")
    registered = set(re.findall(r'^\s*path:\s*"([^"]+)"', config, flags=re.MULTILINE))
    classified = set(_GUARDED_CONTENT_ROOTS) | set(_UNGUARDED_CONTENT_ROOTS)
    assert registered <= classified, (
        f"content roots {sorted(registered - classified)} are registered in "
        "docusaurus.config.ts but classified in neither _GUARDED_CONTENT_ROOTS nor "
        "_UNGUARDED_CONTENT_ROOTS. Decide which, and record the reason if unguarded."
    )


@pytest.mark.parametrize(
    "path",
    _guarded_prose_pages(),
    ids=lambda path: f"{path.parent.name}/{path.name}",
)
def test_walkthrough_prose_contains_no_hand_typed_numbers(path: Path) -> None:
    """No measurement is typed into a guarded page's prose.

    The generator and `factsFor` make a displayed number traceable; this makes
    *bypassing* them visible. Without it, "every number comes from evidence"
    would hold only for the numbers that happened to go through the helper.
    """
    offenders = sorted(
        {
            match.group(0)
            for match in _NUMBER.finditer(_walkthrough_prose(path.read_text(encoding="utf-8")))
            if match.group(0) not in _LITERAL_ALLOWLIST
        }
    )
    assert not offenders, (
        f"{path.name} writes {offenders} as literals; route each through "
        "factsFor(...) or add it to _LITERAL_ALLOWLIST with the reason it is not a measurement"
    )


#: The home page is TSX, so it gets its own allowlist rather than widening the
#: shared one: a digit that is defensible inside a display equation is not
#: defensible in a walkthrough's prose. Empty since ADR 0035 replaced the home
#: page's displayed loss identity with prose -- the two entries this held were
#: the indicator function and the zero matrix of that identity. Anything added
#: back here needs the same kind of reason: a structural constant of an
#: equation, never a number a run produced.
_HOME_LITERAL_ALLOWLIST: dict[str, str] = {}

_TS_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_TS_LINE_COMMENT = re.compile(r"//[^\n]*")


def test_home_page_contains_no_numeric_literal() -> None:
    """`index.tsx` displays only numbers the fact contract resolved.

    The prose guard above strips JSX expressions and tags, which on a TSX file
    would strip almost everything and prove nothing. So the home page is held to
    a blunter rule instead: after comments, imports, JSX expressions and tags are
    removed, no digit may remain. Every number it shows therefore arrives through
    `factsFor("home")`.
    """
    path = ROOT / "website/src/pages/index.tsx"
    text = path.read_text(encoding="utf-8")
    for pattern in (_TS_BLOCK_COMMENT, _TS_LINE_COMMENT, _ESM, _JSX_EXPRESSION, _JSX_TAG):
        text = pattern.sub(" ", text)
    offenders = sorted(
        {
            match.group(0)
            for match in _NUMBER.finditer(text)
            if match.group(0) not in _HOME_LITERAL_ALLOWLIST
        }
    )
    assert not offenders, (
        f"index.tsx writes {offenders} as literals; route each through "
        'factsFor("home") or add it to _HOME_LITERAL_ALLOWLIST with its reason'
    )


# `flowcyt.mdx` is the one walkthrough that also reads a generated file directly:
# its three charts need series, not scalars, so they consume
# `website/src/generated/showcase-data.json` rather than going through
# `factsFor`. That is a deliberate exception -- a bar chart cannot be driven by
# formatted strings -- but it means those numbers sit outside the fact contract's
# three layers. This closes the gap that matters most: the committed file must
# still agree with the study it was generated from, so re-running the FlowCyt
# example without regenerating the portal data is a test failure rather than a
# quietly stale chart.
SHOWCASE_GENERATOR = ROOT / "website" / "scripts" / "generate_showcase.py"
SHOWCASE_DATA = ROOT / "website" / "src" / "generated" / "showcase-data.json"


def test_showcase_chart_data_is_current() -> None:
    """The committed FlowCyt chart series match what the study now produces."""
    name = "_scorequant_generate_showcase"
    spec = importlib.util.spec_from_file_location(name, SHOWCASE_GENERATOR)
    assert spec is not None and spec.loader is not None, SHOWCASE_GENERATOR
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    committed = json.loads(SHOWCASE_DATA.read_text(encoding="utf-8"))
    assert module.build_narrative() == committed, (
        "website/src/generated/showcase-data.json is stale; "
        "run `uv run python website/scripts/generate_showcase.py`"
    )


# `michelson-sweep.json` is the other file a walkthrough reads directly rather
# than through `factsFor`: drawing the aperture (equal-width segments versus
# the fitted rule's comb) needs every swept bin budget's run-length-encoded
# labels, not a formatted scalar. The same three checks apply as above, plus
# the shape a run-length encoding must hold: the runs tile `[0, uMax]`
# exactly and every label is a valid bin index the labeling actually uses.
MICHELSON_SWEEP_DATA = ROOT / "website" / "src" / "generated" / "michelson-sweep.json"
MICHELSON_EVIDENCE = ROOT / "docs" / "examples" / "assets" / "michelson-phase.json"


@pytest.fixture(scope="module")
def michelson_sweep() -> dict[str, object]:
    return json.loads(MICHELSON_SWEEP_DATA.read_text(encoding="utf-8"))


def test_michelson_sweep_data_is_current(
    generator: ModuleType, michelson_sweep: dict[str, object]
) -> None:
    """The committed sweep aperture runs (and their formatted text) match the generator.

    Equality on the whole payload also covers each row's ``text`` object,
    since a page never formats a value itself.
    """
    assert generator.build_michelson_sweep() == michelson_sweep, (
        "website/src/generated/michelson-sweep.json is stale; "
        "run `uv run python website/scripts/generate_walkthroughs.py`"
    )


def test_michelson_sweep_runs_tile_the_aperture(michelson_sweep: dict[str, object]) -> None:
    """Every row's three labelings' runs tile `[0, uMax]` in valid, exhaustive labels."""
    u_max = float(michelson_sweep["uMax"])  # type: ignore[arg-type]
    rows = michelson_sweep["rows"]
    assert isinstance(rows, list) and rows
    for row in rows:
        assert isinstance(row, dict)
        n_bins = int(row["nBins"])  # type: ignore[call-overload]
        runs_by_labeling = row["runs"]
        assert isinstance(runs_by_labeling, dict)
        for labeling, runs in runs_by_labeling.items():
            assert isinstance(runs, list) and runs, (labeling, n_bins)
            assert float(runs[0][0]) == pytest.approx(0.0, abs=1e-9), (labeling, n_bins)
            assert float(runs[-1][1]) == pytest.approx(u_max, abs=1e-9), (labeling, n_bins)
            for earlier, later in zip(runs, runs[1:], strict=False):
                assert float(earlier[1]) == pytest.approx(float(later[0]), abs=1e-9), (
                    labeling,
                    n_bins,
                )
            labels_seen = set()
            for run in runs:
                label = run[2]
                assert isinstance(label, int)
                assert 0 <= label < n_bins, (labeling, n_bins, label)
                labels_seen.add(label)
            assert len(labels_seen) == n_bins, (labeling, n_bins, sorted(labels_seen))


def test_michelson_sweep_retentions_match_the_evidence(
    michelson_sweep: dict[str, object],
) -> None:
    """Every row's four retentions equal `sweep[i].*` in the committed evidence.

    Compared against the evidence directly rather than against named facts
    such as `equalWidthRetentionAtSix`, which the fact table only defines for
    some bin budgets.
    """
    evidence = json.loads(MICHELSON_EVIDENCE.read_text(encoding="utf-8"))
    evidence_sweep = evidence["sweep"]
    assert isinstance(evidence_sweep, list)
    rows = michelson_sweep["rows"]
    assert isinstance(rows, list)
    assert len(rows) == len(evidence_sweep)
    for row, evidence_row in zip(rows, evidence_sweep, strict=True):
        assert isinstance(row, dict) and isinstance(evidence_row, dict)
        assert row["nBins"] == evidence_row["n_bins"]
        assert row["equalWidth"] == evidence_row["equal_width_retention"]
        assert row["dOptimal"] == evidence_row["d_optimal_retention"]
        assert row["profiled"] == evidence_row["profiled_retention"]
        assert row["ceiling"] == evidence_row["ceiling_retention"]
        assert row["boundGap"] == evidence_row["bound_gap"]
