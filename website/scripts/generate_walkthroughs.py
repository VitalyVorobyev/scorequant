"""Generate the numbers the portal's prose pages are allowed to print.

Every value a walkthrough displays comes from this script, and every value this
script emits carries a JSON Pointer into a committed evidence file. That pairing
is the whole point: a page cannot print a number that no run produced, and
``tests/test_walkthrough_facts.py`` re-resolves every pointer against the
evidence so the generated file cannot drift from the studies behind it.

The fact table below *is* the contract. Adding a number to a page means adding a
row here, which means naming the evidence it comes from.

The table was later extended from the four walkthroughs to the front door: the
``home`` page keys below feed ``website/src/pages/index.tsx``, which may contain
no numeric literal of its own -- one generator, not two.

Run through ``pnpm generate``, or directly::

    uv run python website/scripts/generate_walkthroughs.py
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "website/src/generated/walkthrough-data.json"

#: Committed evidence files the facts may cite. A pointer into anything else is
#: rejected, so a walkthrough cannot quietly start sourcing numbers from a file
#: that nothing pins.
EVIDENCE = (
    "examples/outputs/hep-classifier.json",
    "examples/outputs/michelson-phase.json",
    "examples/outputs/door3-classifier.json",
    "examples/outputs/cell_population.json",
    "examples/data/flowcyt_fixture.json",
    "examples/data/flowcyt_walkthrough.json",
    "examples/data/hep_higgsml_fixture.json",
    "examples/outputs/ideas-toys.json",
    "examples/outputs/lloyd-nonmonotone.json",
)


@dataclass(frozen=True)
class Fact:
    """One number or string a portal page may print.

    Parameters
    ----------
    page
        Page slug the fact belongs to: a walkthrough, or ``home``.
    key
        Name the page looks the fact up by.
    source
        Repo-relative evidence path and a JSON Pointer into it, joined by ``#``.
    render
        Formats the resolved value into the string the page displays. Pages
        never format a value themselves.
    """

    page: str
    key: str
    source: str
    render: Callable[[object], str]


def _fixed(digits: int) -> Callable[[object], str]:
    """Return a renderer printing a float to ``digits`` decimal places."""
    return lambda value: f"{float(value):.{digits}f}"


def _count(value: object) -> str:
    """Render an integer with thousands separators."""
    return f"{int(value):,}"


def _text(value: object) -> str:
    """Render a value as its plain string form."""
    return str(value)


HEP = "examples/outputs/hep-classifier.json"
MICHELSON = "examples/outputs/michelson-phase.json"
DOOR3 = "examples/outputs/door3-classifier.json"
FLOWCYT_STUDY = "examples/outputs/cell_population.json"
FLOWCYT_FIXTURE = "examples/data/flowcyt_fixture.json"
FLOWCYT_WALKTHROUGH = "examples/data/flowcyt_walkthrough.json"
HEP_FIXTURE = "examples/data/hep_higgsml_fixture.json"
IDEAS = "examples/outputs/ideas-toys.json"
LLOYD = "examples/outputs/lloyd-nonmonotone.json"

FACTS: tuple[Fact, ...] = (
    # ---------------------------------------------------------------- flowcyt
    Fact("flowcyt", "bins", f"{FLOWCYT_STUDY}#/operating_partition/n_bins", _count),
    Fact(
        "flowcyt",
        "heldOutPatients",
        f"{FLOWCYT_STUDY}#/unbinned_classifier_ratio/likelihood_convergence/total_patients",
        _count,
    ),
    # The headline: the soft Voronoi rule's held-out error against the unbinned
    # fit it compresses, and against the two marker-space reductions.
    Fact(
        "flowcyt",
        "softVoronoiRmse",
        f"{FLOWCYT_STUDY}#/soft_voronoi:8/target_macro_rmse",
        _fixed(6),
    ),
    Fact(
        "flowcyt",
        "unbinnedMacroRmse",
        f"{FLOWCYT_STUDY}#/unbinned_classifier_ratio/target_macro_rmse",
        _fixed(6),
    ),
    Fact(
        "flowcyt",
        "markerKmeansRmse",
        f"{FLOWCYT_STUDY}#/marker_kmeans:8/target_macro_rmse",
        _fixed(6),
    ),
    Fact(
        "flowcyt",
        "twoDimensionalGridRmse",
        f"{FLOWCYT_STUDY}#/two_dimensional_grid:8/target_macro_rmse",
        _fixed(6),
    ),
    # The bin budget: five bins are structurally short of the five free
    # composition coordinates, and eight is the knee.
    Fact(
        "flowcyt",
        "freeFractions",
        f"{FLOWCYT_STUDY}#/scientific_closure/template_identifiability/soft_voronoi:5/required_rank",
        _count,
    ),
    Fact(
        "flowcyt",
        "fiveBinRankBound",
        f"{FLOWCYT_STUDY}#/scientific_closure/template_identifiability/soft_voronoi:5/rank_bound",
        _count,
    ),
    Fact(
        "flowcyt",
        "fiveBinEfficiency",
        f"{FLOWCYT_STUDY}#/soft_voronoi:5/held_out_d_efficiency",
        _fixed(4),
    ),
    Fact(
        "flowcyt",
        "softVoronoiEfficiency",
        f"{FLOWCYT_STUDY}#/soft_voronoi:8/held_out_d_efficiency",
        _fixed(4),
    ),
    # The study's own sample size, not the fixture's: the walkthrough has to be
    # able to say that its figures come from the full acquisition rather than
    # from the small table its code runs on.
    Fact("flowcyt", "studyCells", f"{FLOWCYT_STUDY}#/source/sample_rows", _count),
    Fact("flowcyt", "license", f"{FLOWCYT_FIXTURE}#/source_license", _text),
    Fact("flowcyt", "sourceRepository", f"{FLOWCYT_FIXTURE}#/source_repository", _text),
    # What the page's own code produces on the committed fixture-scale table,
    # so the reader who runs it is told what to expect and why it differs
    # from the full study.
    Fact("flowcyt", "tableTemplateRows", f"{FLOWCYT_WALKTHROUGH}#/rows/template", _count),
    Fact(
        "flowcyt",
        "tableSoftVoronoiRmse",
        f"{FLOWCYT_WALKTHROUGH}#/chain/soft_voronoi/macro_rmse",
        _fixed(4),
    ),
    Fact(
        "flowcyt",
        "tableUnbinnedRmse",
        f"{FLOWCYT_WALKTHROUGH}#/chain/unbinned_macro_rmse",
        _fixed(4),
    ),
    # -------------------------------------------------------------------- hep
    Fact("hep", "nEvents", f"{HEP}#/fixture/n_events", _count),
    Fact("hep", "bins", f"{HEP}#/n_bins", _count),
    Fact(
        "hep",
        "tesScoreCorrelation",
        f"{HEP}#/classifiers/tes_reliability/weighted_correlation",
        _fixed(2),
    ),
    # Held out: every reusable rule built on one half and scored on the other,
    # averaged over both directions.
    Fact(
        "hep",
        "dsRuleHeldOut",
        f"{HEP}#/cross_evaluation/mean/ds_rule/evaluation_profiled_retention",
        _fixed(4),
    ),
    Fact(
        "hep",
        "dRuleHeldOut",
        f"{HEP}#/cross_evaluation/mean/d_rule/evaluation_profiled_retention",
        _fixed(4),
    ),
    Fact(
        "hep",
        "intervalsHeldOut",
        f"{HEP}#/cross_evaluation/mean/classifier_significance_intervals/evaluation_profiled_retention",
        _fixed(4),
    ),
    Fact(
        "hep",
        "logitHeldOut",
        f"{HEP}#/cross_evaluation/mean/classifier_logit_equal_width/evaluation_profiled_retention",
        _fixed(4),
    ),
    Fact(
        "hep",
        "quantileHeldOut",
        f"{HEP}#/cross_evaluation/mean/classifier_quantile/evaluation_profiled_retention",
        _fixed(4),
    ),
    Fact(
        "hep",
        "evaluationCeilingAB",
        f"{HEP}#/cross_evaluation/directions/0/evaluation_ceiling_retention",
        _fixed(4),
    ),
    # Downstream: the expected relative uncertainty on the signal strength from
    # the count table's own likelihood, with the simulation's actual energy-
    # scale response. Rules fitted on all events.
    Fact("hep", "tesConstraintWidth", f"{HEP}#/downstream/tes_constraint_width", _fixed(2)),
    Fact("hep", "sigmaDsRule", f"{HEP}#/downstream/all_events/rules/ds_rule/sigma_mu", _fixed(3)),
    Fact(
        "hep",
        "sigmaDsRuleFixed",
        f"{HEP}#/downstream/all_events/rules/ds_rule/sigma_mu_tes_fixed",
        _fixed(3),
    ),
    Fact("hep", "sigmaDRule", f"{HEP}#/downstream/all_events/rules/d_rule/sigma_mu", _fixed(3)),
    Fact(
        "hep",
        "sigmaDRuleFixed",
        f"{HEP}#/downstream/all_events/rules/d_rule/sigma_mu_tes_fixed",
        _fixed(3),
    ),
    Fact(
        "hep",
        "sigmaIntervals",
        f"{HEP}#/downstream/all_events/rules/classifier_significance_intervals/sigma_mu",
        _fixed(3),
    ),
    Fact(
        "hep",
        "sigmaIntervalsFixed",
        f"{HEP}#/downstream/all_events/rules/classifier_significance_intervals/sigma_mu_tes_fixed",
        _fixed(3),
    ),
    Fact(
        "hep",
        "sigmaLogit",
        f"{HEP}#/downstream/all_events/rules/classifier_logit_equal_width/sigma_mu",
        _fixed(3),
    ),
    Fact(
        "hep",
        "sigmaLogitFixed",
        f"{HEP}#/downstream/all_events/rules/classifier_logit_equal_width/sigma_mu_tes_fixed",
        _fixed(3),
    ),
    Fact(
        "hep",
        "sigmaQuantile",
        f"{HEP}#/downstream/all_events/rules/classifier_quantile/sigma_mu",
        _fixed(3),
    ),
    Fact(
        "hep",
        "sigmaQuantileFixed",
        f"{HEP}#/downstream/all_events/rules/classifier_quantile/sigma_mu_tes_fixed",
        _fixed(3),
    ),
    # Provenance as three separate facts, from the fixture's own record: the
    # bytes, the code repository they came from, and the archival record the
    # licence claim is made under (S08 decision D9). Never collapsed into one
    # licence line, on the page or here.
    Fact("hep", "datasetName", f"{HEP_FIXTURE}#/dataset", _text),
    Fact("hep", "license", f"{HEP_FIXTURE}#/source_license", _text),
    Fact("hep", "licenseRecordDoi", f"{HEP_FIXTURE}#/license_record_doi", _text),
    Fact("hep", "licenseRecordUrl", f"{HEP_FIXTURE}#/license_record_url", _text),
    # ----------------------------------------------------------------- ratios
    Fact("ratios", "bins", f"{DOOR3}#/n_bins", _count),
    Fact("ratios", "signalFraction", f"{DOOR3}#/mixture/reference_fractions/0", _fixed(2)),
    Fact("ratios", "smallTrainSize", f"{DOOR3}#/ladder/0/n_per_class", _count),
    Fact("ratios", "mediumTrainSize", f"{DOOR3}#/ladder/1/n_per_class", _count),
    Fact("ratios", "largeTrainSize", f"{DOOR3}#/ladder/2/n_per_class", _count),
    Fact("ratios", "smallSurrogate", f"{DOOR3}#/ladder/0/surrogate_retention", _fixed(4)),
    Fact("ratios", "smallTrue", f"{DOOR3}#/ladder/0/true_retention", _fixed(4)),
    Fact("ratios", "largeGap", f"{DOOR3}#/surrogate_gap/rows/2/gap", _fixed(4)),
    Fact("ratios", "smallClosure", f"{DOOR3}#/ladder/0/closure_residual", _fixed(4)),
    Fact("ratios", "largeClosure", f"{DOOR3}#/ladder/2/closure_residual", _fixed(4)),
    # The wrong-measure contrast. Published deliberately: it is what
    # `ratio_closure_report` returns when handed a sample from the wrong
    # measure, it is present for a perfect score, and it is not estimator error.
    Fact(
        "ratios",
        "wrongMeasureResidual",
        f"{DOOR3}#/closure_measure_mismatch/wrong_measure_residual",
        _fixed(4),
    ),
    Fact(
        "ratios",
        "classifierInformationKind",
        f"{DOOR3}#/provenance/classifier/information_kind",
        _text,
    ),
    Fact(
        "ratios",
        "exactInformationKind",
        f"{DOOR3}#/provenance/exact/information_kind",
        _text,
    ),
    # -------------------------------------------------------------- michelson
    Fact("michelson", "bins", f"{MICHELSON}#/headline_bins", _count),
    Fact("michelson", "fringes", f"{MICHELSON}#/fringes", _count),
    Fact("michelson", "visibility", f"{MICHELSON}#/visibility", _fixed(2)),
    Fact("michelson", "correlation", f"{MICHELSON}#/closed_form/correlation", _fixed(4)),
    # sweep[1] is the six-bin row, the headline budget `bins` names. The page
    # quotes only that budget; the rest of the sweep reaches it through the
    # budget explorer, which reads `michelson-sweep.json` below.
    Fact(
        "michelson",
        "equalWidthRetentionAtSix",
        f"{MICHELSON}#/sweep/1/equal_width_retention",
        _fixed(4),
    ),
    Fact(
        "michelson",
        "dOptimalRetentionAtSix",
        f"{MICHELSON}#/sweep/1/d_optimal_retention",
        _fixed(4),
    ),
    Fact(
        "michelson",
        "profiledRetentionAtSix",
        f"{MICHELSON}#/sweep/1/profiled_retention",
        _fixed(4),
    ),
    Fact(
        "michelson",
        "ceilingRetentionAtSix",
        f"{MICHELSON}#/sweep/1/ceiling_retention",
        _fixed(4),
    ),
    # The reusable profiled rule, against the finite partition it cannot be
    # compiled from.
    Fact(
        "michelson",
        "dsRuleProfiledRetention",
        f"{MICHELSON}#/rules/1/profiled_retention",
        _fixed(4),
    ),
    # ------------------------------------------------------------------ ideas
    # The Ideas posts share one page key. Their two toys come from
    # `examples/ideas_toys.py`; the Lloyd post quotes the committed
    # counterexample study. The retention post reuses the Michelson facts.
    Fact("ideas", "fringes", f"{IDEAS}#/fringe/fringes", _count),
    Fact("ideas", "fringeBins", f"{IDEAS}#/fringe/headline_bins", _count),
    Fact("ideas", "fringeEqual", f"{IDEAS}#/fringe/headline_equal", _fixed(2)),
    Fact("ideas", "fringeScore", f"{IDEAS}#/fringe/headline_score", _fixed(2)),
    Fact("ideas", "clusterBins", f"{IDEAS}#/clusters/bins", _count),
    Fact("ideas", "kmeansLow", f"{IDEAS}#/clusters/kmeans/retained_eigenvalues/0", _fixed(2)),
    Fact("ideas", "kmeansHigh", f"{IDEAS}#/clusters/kmeans/retained_eigenvalues/1", _fixed(2)),
    Fact("ideas", "dLow", f"{IDEAS}#/clusters/d_optimal/retained_eigenvalues/0", _fixed(2)),
    Fact("ideas", "dHigh", f"{IDEAS}#/clusters/d_optimal/retained_eigenvalues/1", _fixed(2)),
    Fact(
        "ideas",
        "kmeansArithmetic",
        f"{IDEAS}#/clusters/kmeans/arithmetic_mean_retention",
        _fixed(2),
    ),
    Fact(
        "ideas",
        "dArithmetic",
        f"{IDEAS}#/clusters/d_optimal/arithmetic_mean_retention",
        _fixed(2),
    ),
    Fact(
        "ideas",
        "kmeansGeometric",
        f"{IDEAS}#/clusters/kmeans/geometric_mean_retention",
        _fixed(2),
    ),
    Fact(
        "ideas",
        "dGeometric",
        f"{IDEAS}#/clusters/d_optimal/geometric_mean_retention",
        _fixed(2),
    ),
    Fact("ideas", "lloydMoved", f"{LLOYD}#/counterexample/moved", _count),
    Fact("ideas", "lloydTangent", f"{LLOYD}#/counterexample/tangent_change", _fixed(1)),
    Fact("ideas", "lloydStep", f"{LLOYD}#/counterexample/step", _fixed(2)),
    Fact("ideas", "ledgerRuns", f"{LLOYD}#/ledger_totals/runs", _count),
    Fact("ideas", "ledgerDownhill", f"{LLOYD}#/ledger_totals/downhill_runs", _count),
    Fact("ideas", "ledgerEmptied", f"{LLOYD}#/ledger_totals/emptied_runs", _count),
)


#: Aperture runs for every swept bin budget of the committed Michelson study,
#: read straight from its evidence rather than through the fact table: a
#: walkthrough chart needs the whole sweep's run-length encoding, not a single
#: formatted scalar.
MICHELSON_SWEEP_OUTPUT = ROOT / "website/src/generated/michelson-sweep.json"


def build_michelson_sweep() -> dict[str, object]:
    """Assemble the Michelson sweep's aperture runs for every swept bin budget.

    Read straight from the committed evidence file
    ``examples/outputs/michelson-phase.json`` -- the study's own sweep
    rows, not a re-run -- so a walkthrough can draw the aperture (equal-width
    contiguous segments versus the fitted rule's comb) for every bin budget
    the study compares, and a browser refit can redraw it against the same
    shape.

    Returns
    -------
    dict
        ``{"schemaVersion", "uMax", "fringes", "headlineBins", "visibility",
        "rows"}``, one row per swept bin budget, each row carrying its three retentions, the
        certified ceiling, the bound gap, the three labelings' aperture runs,
        and a ``text`` object formatted with the same renderers the fact
        table uses -- a page never formats a value itself, and a static
        fallback table renders these strings verbatim.
    """
    evidence = json.loads((ROOT / MICHELSON).read_text(encoding="utf-8"))
    retention_text = _fixed(4)
    bound_gap_text = _fixed(6)
    rows = [
        {
            "nBins": row["n_bins"],
            "equalWidth": row["equal_width_retention"],
            "dOptimal": row["d_optimal_retention"],
            "profiled": row["profiled_retention"],
            "ceiling": row["ceiling_retention"],
            "boundGap": row["bound_gap"],
            "text": {
                "equalWidth": retention_text(row["equal_width_retention"]),
                "dOptimal": retention_text(row["d_optimal_retention"]),
                "profiled": retention_text(row["profiled_retention"]),
                "ceiling": retention_text(row["ceiling_retention"]),
                "boundGap": bound_gap_text(row["bound_gap"]),
            },
            "runs": {
                "equalWidth": row["equal_width_runs"],
                "dOptimal": row["d_runs"],
                "profiled": row["profiled_runs"],
            },
        }
        for row in evidence["sweep"]
    ]
    return {
        "schemaVersion": 1,
        "uMax": evidence["u_max"],
        "fringes": evidence["fringes"],
        "headlineBins": evidence["headline_bins"],
        "visibility": evidence["visibility"],
        "rows": rows,
    }


def write_michelson_sweep() -> int:
    """Write the Michelson sweep's aperture runs to committed JSON.

    Returns
    -------
    int
        Byte size of the written file.
    """
    payload = json.dumps(build_michelson_sweep(), indent=2, sort_keys=True) + "\n"
    MICHELSON_SWEEP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MICHELSON_SWEEP_OUTPUT.write_text(payload, encoding="utf-8")
    return len(payload.encode("utf-8"))


def resolve(document: object, pointer: str) -> object:
    """Resolve a JSON Pointer (RFC 6901) against a loaded document.

    Parameters
    ----------
    document
        The parsed evidence file.
    pointer
        A pointer such as ``/partitions/3/label``. The empty pointer selects
        the whole document.

    Returns
    -------
    object
        The referenced value.

    Raises
    ------
    KeyError
        If any token does not resolve, naming the pointer that failed.
    """
    current = document
    if pointer in ("", "/"):
        return current
    for raw in pointer.lstrip("/").split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            if token not in current:
                raise KeyError(f"pointer {pointer!r} failed at token {token!r}")
            current = current[token]
        else:
            raise KeyError(f"pointer {pointer!r} reached a scalar at token {token!r}")
    return current


def build() -> dict[str, object]:
    """Resolve every fact against its evidence and return the generated payload."""
    documents = {name: json.loads((ROOT / name).read_text(encoding="utf-8")) for name in EVIDENCE}
    pages: dict[str, dict[str, object]] = {}
    for fact in FACTS:
        path, _, pointer = fact.source.partition("#")
        if path not in documents:
            raise ValueError(f"{fact.page}.{fact.key} cites {path}, which is not in EVIDENCE")
        value = resolve(documents[path], pointer)
        pages.setdefault(fact.page, {})[fact.key] = {
            "value": value,
            "text": fact.render(value),
            "source": fact.source,
        }
    return {"schemaVersion": 1, "pages": pages}


#: Deterministic per-walkthrough score tables each walkthrough's `LiveFit`
#: experiment fetches on demand. Each slug is written to
#: ``website/static/walkthrough-scores/<slug>.json`` in exactly the shape a
#: ``LiveFit`` problem needs (``website/src/components/liveFit/types.ts``):
#: ``scores``, ``weights``, plus ``schema``, ``label`` and ``detail``.
#: Only the walkthroughs whose pages actually fetch a table are built here;
#: ``get-started.json`` is written by ``generate_snippets.py`` instead.
SCORE_TABLES_OUTPUT = ROOT / "website/static/walkthrough-scores"


def _round(values: object, digits: int | None = 6) -> object:
    """Round nested numeric data so a committed score table has no float noise.

    Parameters
    ----------
    values
        A NumPy array, or a nested Python list/float/int built from one.
    digits
        Decimal places to round every float to. ``None`` keeps full precision:
        a table a browser run must reproduce *bit for bit* against a committed
        study (the Michelson sweep) cannot be rounded, because the exchange
        solver's discrete optimum moves under a 1e-6 perturbation.

    Returns
    -------
    object
        The same nested structure, as plain Python types a JSON encoder
        accepts, with every float rounded.
    """
    if isinstance(values, np.ndarray):
        return _round(values.tolist(), digits)
    if isinstance(values, list):
        return [_round(item, digits) for item in values]
    if isinstance(values, float | np.floating):
        return float(values) if digits is None else round(float(values), digits)
    if isinstance(values, int | np.integer):
        return int(values)
    return values


def _build_michelson_score_table() -> dict[str, object]:
    """Build the Michelson-phase walkthrough's score table.

    Reuses ``examples/michelson_phase.py``'s own analytic provider and
    midpoint-quadrature sample (`build_provider`, `build_train_sample`)
    rather than reimplementing them, at the same full-scale node count the
    committed study runs unless ``SCOREQUANT_EXAMPLE_FAST`` is set.

    Returns
    -------
    dict
        ``{"schema", "label", "detail", "scores", "weights"}``.
    """
    sys.path.insert(0, str(ROOT))
    from examples._env import example_scale
    from examples.michelson_phase import SCHEMA, build_provider, build_train_sample

    n_nodes = example_scale(8_000, 2_000)
    provider = build_provider()
    sample = build_train_sample(provider, n_nodes=n_nodes)
    rows, columns = sample.scores.shape
    return {
        "schema": list(SCHEMA.parameters),
        "label": "Michelson interferometer phase scores",
        "detail": (
            f"{rows:,} midpoint-quadrature nodes × {columns} score dimensions · "
            "analytic ScoreFunction, deterministic seed"
        ),
        # Full precision: the lesson's browser refit reproduces the committed
        # sweep on exactly this table, and `tests/test_browser_lab.py` pins it.
        "scores": _round(sample.scores, digits=None),
        "weights": _round(sample.weights, digits=None),
    }


def write_walkthrough_score_tables() -> dict[str, int]:
    """Write each walkthrough's deterministic score table to committed JSON.

    A walkthrough that carries a ``LiveFit`` experiment needs one small,
    deterministic score table, in exactly the shape a ``LiveFit`` problem
    (``website/src/components/liveFit/types.ts``) needs. Only the Michelson
    page carries one. The FlowCyt and HEP pages run their chains in the
    reader's own Python from committed tables (``examples/data/``), and the
    density-ratio page carries no experiment at all: its argument is the gap
    between a reported and an achieved retention, which a refit of the
    estimated score cannot show, so it makes its point with a comparison and
    a figure instead.

    Returns
    -------
    dict of int
        Byte size of each written file, keyed by slug.
    """
    builders: dict[str, Callable[[], dict[str, object]]] = {
        "michelson": _build_michelson_score_table,
    }
    SCORE_TABLES_OUTPUT.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for slug, builder in builders.items():
        payload = json.dumps(builder(), sort_keys=True) + "\n"
        (SCORE_TABLES_OUTPUT / f"{slug}.json").write_text(payload, encoding="utf-8")
        sizes[slug] = len(payload.encode("utf-8"))
    return sizes


#: Committed example figures the walkthroughs display, copied into the portal's
#: static tree rather than committed twice. ``pnpm build`` and ``pnpm validate``
#: both run this generator, so the copies exist wherever the site is built.
FIGURES = {
    "hep-cells.png": "examples/outputs/hep-cells.png",
    "michelson-d-geometry.png": "examples/outputs/michelson-d-geometry.png",
    "michelson-profiled-ds.png": "examples/outputs/michelson-profiled-ds.png",
    "door3-classifier.png": "examples/outputs/door3-classifier.png",
    "ideas-score-bins.png": "examples/outputs/ideas-score-bins.png",
    "ideas-kmeans-d.png": "examples/outputs/ideas-kmeans-d.png",
    "lloyd-nonmonotone.png": "examples/outputs/lloyd-nonmonotone.png",
}


def copy_walkthrough_figures() -> None:
    """Copy each example's committed figure into ``website/static/walkthrough-figures``.

    The figures live beside the studies that produce them, outside ``website/``,
    where Docusaurus cannot serve them. Copying at generate time keeps one
    committed copy of each image while still letting a walkthrough show it. The
    directory holds only these copies, so a figure dropped from ``FIGURES`` is
    removed rather than left behind for the figure tests to trip over.
    """
    destination = ROOT / "website/static/walkthrough-figures"
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.iterdir():
        if stale.is_file() and stale.name not in FIGURES:
            stale.unlink()
    for name, source in FIGURES.items():
        shutil.copyfile(ROOT / source, destination / name)
    print(f"copied {len(FIGURES)} figures into {destination.relative_to(ROOT)}")


def main() -> None:
    """Write the generated walkthrough facts and score tables."""
    copy_walkthrough_figures()
    payload = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = ", ".join(f"{page} {len(facts)}" for page, facts in sorted(payload["pages"].items()))
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({counts})")

    sizes = write_walkthrough_score_tables()
    for slug, size in sorted(sizes.items()):
        print(f"wrote {(SCORE_TABLES_OUTPUT / f'{slug}.json').relative_to(ROOT)} ({size:,} bytes)")

    sweep_size = write_michelson_sweep()
    print(f"wrote {MICHELSON_SWEEP_OUTPUT.relative_to(ROOT)} ({sweep_size:,} bytes)")


if __name__ == "__main__":
    main()
