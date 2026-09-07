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
    "docs/examples/assets/hep-classifier.json",
    "docs/examples/assets/michelson-phase.json",
    "docs/examples/assets/door3-classifier.json",
    "docs/usecases/assets/cell_population.json",
    "docs/usecases/assets/flowcyt_profiled_ds.json",
    "examples/data/flowcyt_fixture.json",
    "examples/data/hep_higgsml_fixture.json",
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


def _percent(digits: int) -> Callable[[object], str]:
    """Return a renderer printing a fraction as a percentage."""
    return lambda value: f"{100.0 * float(value):.{digits}f}%"


def _scientific(digits: int) -> Callable[[object], str]:
    """Return a renderer printing a float in scientific notation.

    Used where a value is so small that fixed-point rendering would print it
    as a flat ``0.0000`` and hide the fact that it was computed rather than
    asserted.
    """
    return lambda value: f"{float(value):.{digits}e}"


def _count(value: object) -> str:
    """Render an integer with thousands separators."""
    return f"{int(value):,}"


def _text(value: object) -> str:
    """Render a value as its plain string form."""
    return str(value)


HEP = "docs/examples/assets/hep-classifier.json"
MICHELSON = "docs/examples/assets/michelson-phase.json"
DOOR3 = "docs/examples/assets/door3-classifier.json"
FLOWCYT_STUDY = "docs/usecases/assets/cell_population.json"
FLOWCYT_PROFILED = "docs/usecases/assets/flowcyt_profiled_ds.json"
FLOWCYT_FIXTURE = "examples/data/flowcyt_fixture.json"
HEP_FIXTURE = "examples/data/hep_higgsml_fixture.json"

FACTS: tuple[Fact, ...] = (
    # ---------------------------------------------------------------- flowcyt
    Fact("flowcyt", "bins", f"{FLOWCYT_STUDY}#/operating_partition/n_bins", _count),
    # The study's two ScoreQuant methods disagree about which is better, and they
    # disagree in opposite directions on the two metrics: soft Voronoi wins on
    # macro RMSE, the compiled D exchange wins on held-out D-efficiency. Neither
    # is "the" headline, so every key names its method and a page cannot quote a
    # number without saying what produced it.
    Fact(
        "flowcyt",
        "softVoronoiRmse",
        f"{FLOWCYT_STUDY}#/soft_voronoi:8/target_macro_rmse",
        _fixed(6),
    ),
    Fact(
        "flowcyt",
        "softVoronoiEfficiency",
        f"{FLOWCYT_STUDY}#/soft_voronoi:8/held_out_d_efficiency",
        _fixed(4),
    ),
    Fact(
        "flowcyt",
        "dExchangeRmse",
        f"{FLOWCYT_STUDY}#/finite_d_exchange:8/target_macro_rmse",
        _fixed(6),
    ),
    Fact(
        "flowcyt",
        "dExchangeEfficiency",
        f"{FLOWCYT_STUDY}#/finite_d_exchange:8/held_out_d_efficiency",
        _fixed(4),
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
        "oneDimensionalRmse",
        f"{FLOWCYT_STUDY}#/one_dimensional_score:8/target_macro_rmse",
        _fixed(6),
    ),
    Fact(
        "flowcyt",
        "twoDimensionalGridRmse",
        f"{FLOWCYT_STUDY}#/two_dimensional_grid:8/target_macro_rmse",
        _fixed(6),
    ),
    Fact(
        "flowcyt",
        "markerKmeansEfficiency",
        f"{FLOWCYT_STUDY}#/marker_kmeans:8/held_out_d_efficiency",
        _fixed(4),
    ),
    Fact(
        "flowcyt",
        "oneDimensionalEfficiency",
        f"{FLOWCYT_STUDY}#/one_dimensional_score:8/held_out_d_efficiency",
        _fixed(4),
    ),
    Fact(
        "flowcyt",
        "twoDimensionalGridEfficiency",
        f"{FLOWCYT_STUDY}#/two_dimensional_grid:8/held_out_d_efficiency",
        _fixed(4),
    ),
    Fact(
        "flowcyt",
        "informationKind",
        f"{FLOWCYT_STUDY}#/finite_d_exchange:8/information_kind",
        _text,
    ),
    # The study's own sample size, not the fixture's: the walkthrough has to be
    # able to say that its headline numbers come from the full acquisition
    # rather than from the small table CI runs on.
    Fact("flowcyt", "studyCells", f"{FLOWCYT_STUDY}#/source/sample_rows", _count),
    Fact("flowcyt", "license", f"{FLOWCYT_FIXTURE}#/source_license", _text),
    Fact("flowcyt", "sourceRepository", f"{FLOWCYT_FIXTURE}#/source_repository", _text),
    Fact(
        "flowcyt",
        "referencePerPatientClass",
        f"{FLOWCYT_FIXTURE}#/reference_per_patient_class",
        _count,
    ),
    Fact("flowcyt", "testPerPatient", f"{FLOWCYT_FIXTURE}#/test_per_patient", _count),
    Fact(
        "flowcyt",
        "profiledCeilingRetention",
        f"{FLOWCYT_PROFILED}#/fixture_scale/bound/ceiling_retention",
        _fixed(4),
    ),
    Fact(
        "flowcyt",
        "profiledDPartitionRetention",
        f"{FLOWCYT_PROFILED}#/fixture_scale/partitions/0/profiled_retention",
        _fixed(4),
    ),
    # -------------------------------------------------------------------- hep
    Fact("hep", "nEvents", f"{HEP}#/fixture/n_events", _count),
    Fact("hep", "signalEvents", f"{HEP}#/fixture/signal_events", _count),
    Fact("hep", "backgroundEvents", f"{HEP}#/fixture/background_events", _count),
    Fact("hep", "bins", f"{HEP}#/n_bins", _count),
    Fact("hep", "delta", f"{HEP}#/delta", _fixed(3)),
    Fact("hep", "nFolds", f"{HEP}#/n_folds", _count),
    Fact("hep", "signalAuc", f"{HEP}#/classifiers/signal_weighted_auc", _fixed(4)),
    Fact("hep", "signalFraction", f"{HEP}#/classifiers/signal_fraction", _fixed(5)),
    Fact("hep", "tesAuc", f"{HEP}#/classifiers/tes_minus_plus_auc", _fixed(4)),
    Fact("hep", "dFullRetention", f"{HEP}#/partitions/0/full_retention", _fixed(4)),
    Fact("hep", "dProfiledRetention", f"{HEP}#/partitions/0/profiled_retention", _fixed(4)),
    Fact("hep", "dsFullRetention", f"{HEP}#/partitions/1/full_retention", _fixed(4)),
    Fact("hep", "dsProfiledRetention", f"{HEP}#/partitions/1/profiled_retention", _fixed(4)),
    Fact("hep", "quantileProfiledRetention", f"{HEP}#/partitions/2/profiled_retention", _fixed(4)),
    Fact("hep", "logitProfiledRetention", f"{HEP}#/partitions/3/profiled_retention", _fixed(4)),
    Fact("hep", "thresholdProfiledRetention", f"{HEP}#/partitions/4/profiled_retention", _fixed(4)),
    Fact(
        "hep",
        "bestBaselineKey",
        f"{HEP}#/scorequant_vs_classifier_binning/best_baseline_key",
        _text,
    ),
    Fact(
        "hep",
        "bestBaselineLabel",
        f"{HEP}#/partitions/3/label",
        _text,
    ),
    Fact(
        "hep",
        "headlineGap",
        f"{HEP}#/scorequant_vs_classifier_binning/profiled_retention_gap",
        _fixed(4),
    ),
    Fact(
        "hep",
        "gapToEqualFrequency",
        f"{HEP}#/scorequant_vs_classifier_binning/profiled_retention_gap_to_equal_frequency",
        _fixed(4),
    ),
    Fact(
        "hep",
        "baselineSpread",
        f"{HEP}#/scorequant_vs_classifier_binning/baseline_spread",
        _fixed(4),
    ),
    Fact("hep", "ceilingRetention", f"{HEP}#/ceiling/ceiling_retention", _fixed(4)),
    Fact("hep", "gapToCeiling", f"{HEP}#/ceiling/gap_to_ds_partition", _fixed(4)),
    Fact(
        "hep",
        "fullRetentionGivenUp",
        f"{HEP}#/criterion_trade/full_retention_given_up",
        _fixed(4),
    ),
    Fact(
        "hep",
        "profiledRetentionGained",
        f"{HEP}#/criterion_trade/profiled_retention_gained",
        _fixed(4),
    ),
    # Baseline labels are published rather than retyped: two of them carry a
    # bin budget and one an operating threshold, so a page that wrote them out
    # by hand would be printing numbers no run produced.
    Fact("hep", "equalFrequencyLabel", f"{HEP}#/partitions/2/label", _text),
    Fact("hep", "thresholdLabel", f"{HEP}#/partitions/4/label", _text),
    Fact("hep", "thresholdBins", f"{HEP}#/partitions/4/n_bins", _count),
    # Provenance as three separate facts, from the fixture's own record: the
    # bytes, the code repository they came from, and the archival record the
    # licence claim is made under (S08 decision D9). Never collapsed into one
    # licence line, on the page or here.
    Fact("hep", "datasetName", f"{HEP_FIXTURE}#/dataset", _text),
    Fact("hep", "license", f"{HEP_FIXTURE}#/source_license", _text),
    Fact("hep", "licenseRecordDoi", f"{HEP_FIXTURE}#/license_record_doi", _text),
    Fact("hep", "licenseRecordUrl", f"{HEP_FIXTURE}#/license_record_url", _text),
    Fact("hep", "bytesFetchedFrom", f"{HEP_FIXTURE}#/bytes_fetched_from", _text),
    Fact("hep", "upstreamCommit", f"{HEP_FIXTURE}#/upstream_commit", _text),
    # ----------------------------------------------------------------- ratios
    Fact("ratios", "bins", f"{DOOR3}#/n_bins", _count),
    Fact("ratios", "nTrain", f"{DOOR3}#/n_train", _count),
    Fact("ratios", "nTest", f"{DOOR3}#/n_test", _count),
    Fact("ratios", "nClosure", f"{DOOR3}#/n_closure", _count),
    Fact("ratios", "signalFraction", f"{DOOR3}#/mixture/reference_fractions/0", _fixed(2)),
    Fact("ratios", "smallTrainSize", f"{DOOR3}#/ladder/0/n_per_class", _count),
    Fact("ratios", "mediumTrainSize", f"{DOOR3}#/ladder/1/n_per_class", _count),
    Fact("ratios", "largeTrainSize", f"{DOOR3}#/ladder/2/n_per_class", _count),
    Fact("ratios", "smallSurrogate", f"{DOOR3}#/ladder/0/surrogate_retention", _fixed(4)),
    Fact("ratios", "smallTrue", f"{DOOR3}#/ladder/0/true_retention", _fixed(4)),
    Fact("ratios", "mediumSurrogate", f"{DOOR3}#/ladder/1/surrogate_retention", _fixed(4)),
    Fact("ratios", "mediumTrue", f"{DOOR3}#/ladder/1/true_retention", _fixed(4)),
    Fact("ratios", "largeSurrogate", f"{DOOR3}#/ladder/2/surrogate_retention", _fixed(4)),
    Fact("ratios", "largeTrue", f"{DOOR3}#/ladder/2/true_retention", _fixed(4)),
    Fact("ratios", "smallGap", f"{DOOR3}#/surrogate_gap/rows/0/gap", _fixed(4)),
    Fact("ratios", "mediumGap", f"{DOOR3}#/surrogate_gap/rows/1/gap", _fixed(4)),
    Fact("ratios", "largeGap", f"{DOOR3}#/surrogate_gap/rows/2/gap", _fixed(4)),
    Fact("ratios", "largestGap", f"{DOOR3}#/surrogate_gap/largest_gap", _fixed(4)),
    Fact("ratios", "exactCeiling", f"{DOOR3}#/exact_ceiling/retention", _fixed(4)),
    Fact("ratios", "smallClosure", f"{DOOR3}#/ladder/0/closure_residual", _fixed(4)),
    Fact("ratios", "mediumClosure", f"{DOOR3}#/ladder/1/closure_residual", _fixed(4)),
    Fact("ratios", "largeClosure", f"{DOOR3}#/ladder/2/closure_residual", _fixed(4)),
    Fact("ratios", "exactClosure", f"{DOOR3}#/exact_ceiling/closure_residual", _fixed(4)),
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
        "analyticSignalRatioMean",
        f"{DOOR3}#/closure_measure_mismatch/analytic_reference_measure_ratio_means/0",
        _fixed(4),
    ),
    Fact(
        "ratios",
        "analyticBackgroundRatioMean",
        f"{DOOR3}#/closure_measure_mismatch/analytic_reference_measure_ratio_means/1",
        _fixed(4),
    ),
    Fact(
        "ratios",
        "classifierProvenanceKind",
        f"{DOOR3}#/provenance/classifier/provenance_kind",
        _text,
    ),
    Fact(
        "ratios",
        "classifierInformationKind",
        f"{DOOR3}#/provenance/classifier/information_kind",
        _text,
    ),
    Fact("ratios", "exactProvenanceKind", f"{DOOR3}#/provenance/exact/provenance_kind", _text),
    Fact(
        "ratios",
        "exactInformationKind",
        f"{DOOR3}#/provenance/exact/information_kind",
        _text,
    ),
    # The naive one-dimensional binnings of the same estimated score. Published
    # even though they nearly tie the fitted partition -- in one dimension they
    # do, and at the largest training size equal-width cells actually win. That
    # is the honest result and it is what makes the multi-dimensional examples
    # mean something.
    Fact(
        "ratios",
        "smallEqualFrequency",
        f"{DOOR3}#/naive_binning/rows/0/equal_frequency_retention",
        _fixed(4),
    ),
    Fact(
        "ratios",
        "smallEqualWidth",
        f"{DOOR3}#/naive_binning/rows/0/equal_width_retention",
        _fixed(4),
    ),
    Fact(
        "ratios",
        "mediumEqualFrequency",
        f"{DOOR3}#/naive_binning/rows/1/equal_frequency_retention",
        _fixed(4),
    ),
    Fact(
        "ratios",
        "mediumEqualWidth",
        f"{DOOR3}#/naive_binning/rows/1/equal_width_retention",
        _fixed(4),
    ),
    Fact(
        "ratios",
        "largeEqualFrequency",
        f"{DOOR3}#/naive_binning/rows/2/equal_frequency_retention",
        _fixed(4),
    ),
    Fact(
        "ratios",
        "largeEqualWidth",
        f"{DOOR3}#/naive_binning/rows/2/equal_width_retention",
        _fixed(4),
    ),
    Fact(
        "ratios",
        "ceilingEqualWidth",
        f"{DOOR3}#/naive_binning/exact_ceiling/equal_width_retention",
        _fixed(4),
    ),
    Fact(
        "ratios",
        "naiveLargestGap",
        f"{DOOR3}#/naive_binning/largest_gap",
        _fixed(4),
    ),
    Fact(
        "ratios",
        "naiveLargestGapTrainSize",
        f"{DOOR3}#/naive_binning/largest_gap_n_per_class",
        _count,
    ),
    # -------------------------------------------------------------- michelson
    Fact("michelson", "bins", f"{MICHELSON}#/headline_bins", _count),
    Fact("michelson", "fringes", f"{MICHELSON}#/fringes", _count),
    Fact("michelson", "visibility", f"{MICHELSON}#/visibility", _fixed(2)),
    Fact("michelson", "nNodes", f"{MICHELSON}#/n_nodes", _count),
    Fact("michelson", "iPhiPhi", f"{MICHELSON}#/closed_form/i_phiphi", _fixed(4)),
    Fact("michelson", "iPhiEps", f"{MICHELSON}#/closed_form/i_phieps", _fixed(4)),
    Fact("michelson", "correlation", f"{MICHELSON}#/closed_form/correlation", _fixed(4)),
    Fact("michelson", "costOfProfiling", f"{MICHELSON}#/cost_of_profiling", _fixed(4)),
    # sweep[0] is the four-bin row and sweep[1] the six-bin one; `bins` above is
    # the headline budget. Every sweep key names its budget so a page cannot
    # attach the four-bin aliasing result to the six-bin headline.
    Fact("michelson", "sweepBinsLow", f"{MICHELSON}#/sweep/0/n_bins", _count),
    Fact("michelson", "sweepBinsHeadline", f"{MICHELSON}#/sweep/1/n_bins", _count),
    Fact(
        "michelson",
        "equalWidthRetentionAtFour",
        f"{MICHELSON}#/sweep/0/equal_width_retention",
        _scientific(1),
    ),
    Fact(
        "michelson",
        "equalWidthRetentionAtSix",
        f"{MICHELSON}#/sweep/1/equal_width_retention",
        _fixed(4),
    ),
    Fact(
        "michelson",
        "dOptimalRetentionAtFour",
        f"{MICHELSON}#/sweep/0/d_optimal_retention",
        _fixed(4),
    ),
    Fact(
        "michelson",
        "profiledRetentionAtFour",
        f"{MICHELSON}#/sweep/0/profiled_retention",
        _fixed(4),
    ),
    Fact(
        "michelson",
        "ceilingRetentionAtFour",
        f"{MICHELSON}#/sweep/0/ceiling_retention",
        _fixed(4),
    ),
    Fact("michelson", "boundGapAtFour", f"{MICHELSON}#/sweep/0/bound_gap", _fixed(6)),
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
    Fact("michelson", "refusalMessage", f"{MICHELSON}#/compile_bridge/refusal_message", _text),
    Fact(
        "michelson",
        "compiledTestRetention",
        f"{MICHELSON}#/compile_bridge/compiled_test_retention",
        _fixed(4),
    ),
    Fact(
        "michelson", "dRuleProfiledRetention", f"{MICHELSON}#/rules/0/profiled_retention", _fixed(4)
    ),
    Fact(
        "michelson",
        "dsRuleProfiledRetention",
        f"{MICHELSON}#/rules/1/profiled_retention",
        _fixed(4),
    ),
    Fact("michelson", "dRuleOwnCriterion", f"{MICHELSON}#/rules/0/criterion_efficiency", _fixed(4)),
    Fact(
        "michelson", "dsRuleOwnCriterion", f"{MICHELSON}#/rules/1/criterion_efficiency", _fixed(4)
    ),
    # The aliasing signature appears twice in the sweep -- an exact zero at four
    # bins, and equal-width segments getting *worse* from eight bins to ten --
    # so the two upper budgets get keys of their own rather than being described
    # in prose without evidence behind them.
    Fact("michelson", "sweepBinsEight", f"{MICHELSON}#/sweep/2/n_bins", _count),
    Fact("michelson", "sweepBinsTen", f"{MICHELSON}#/sweep/3/n_bins", _count),
    Fact(
        "michelson",
        "equalWidthRetentionAtEight",
        f"{MICHELSON}#/sweep/2/equal_width_retention",
        _fixed(4),
    ),
    Fact(
        "michelson",
        "equalWidthRetentionAtTen",
        f"{MICHELSON}#/sweep/3/equal_width_retention",
        _fixed(4),
    ),
    Fact("michelson", "boundGapAtSix", f"{MICHELSON}#/sweep/1/bound_gap", _scientific(1)),
    # The achieved profiled retention and its certified ceiling agree to four
    # decimals, so both are also published at six: a page must be able to show
    # that the ceiling was approached, not reached.
    Fact(
        "michelson",
        "profiledRetentionAtSixFine",
        f"{MICHELSON}#/sweep/1/profiled_retention",
        _fixed(6),
    ),
    Fact(
        "michelson",
        "ceilingRetentionAtSixFine",
        f"{MICHELSON}#/sweep/1/ceiling_retention",
        _fixed(6),
    ),
    # Act one: the D-optimal partition's detector pull-back and compiled rule.
    Fact("michelson", "narrowRunWidth", f"{MICHELSON}#/narrow_run_width", _fixed(2)),
    Fact("michelson", "dRunsAtSix", f"{MICHELSON}#/d_geometry/n_runs", _count),
    Fact("michelson", "dNarrowRuns", f"{MICHELSON}#/d_geometry/narrow_runs", _count),
    Fact("michelson", "dMinRunWidth", f"{MICHELSON}#/d_geometry/min_run_width", _fixed(2)),
    Fact("michelson", "predictionUOne", f"{MICHELSON}#/compile_bridge/predictions/0/u", _fixed(1)),
    Fact("michelson", "predictionUTwo", f"{MICHELSON}#/compile_bridge/predictions/1/u", _fixed(1)),
    Fact(
        "michelson", "predictionUThree", f"{MICHELSON}#/compile_bridge/predictions/2/u", _fixed(1)
    ),
    Fact("michelson", "dPredictionOne", f"{MICHELSON}#/compile_bridge/predictions/0/bin", _count),
    Fact("michelson", "dPredictionTwo", f"{MICHELSON}#/compile_bridge/predictions/1/bin", _count),
    Fact("michelson", "dPredictionThree", f"{MICHELSON}#/compile_bridge/predictions/2/bin", _count),
    # Act two: what the exchange did to the certified initializer, and what the
    # fragments of the finite profiled partition are worth.
    Fact("michelson", "dsInitialRuns", f"{MICHELSON}#/profiled_diagnostics/initial/n_runs", _count),
    Fact(
        "michelson",
        "dsInitialNarrowRuns",
        f"{MICHELSON}#/profiled_diagnostics/initial/narrow_runs",
        _count,
    ),
    Fact(
        "michelson",
        "dsInitialRetention",
        f"{MICHELSON}#/profiled_diagnostics/initial_retention",
        _fixed(4),
    ),
    Fact("michelson", "dsRunsAtSix", f"{MICHELSON}#/profiled_diagnostics/final/n_runs", _count),
    Fact(
        "michelson", "dsNarrowRuns", f"{MICHELSON}#/profiled_diagnostics/final/narrow_runs", _count
    ),
    Fact(
        "michelson",
        "dsMinRunWidth",
        f"{MICHELSON}#/profiled_diagnostics/final/min_run_width",
        _fixed(3),
    ),
    Fact(
        "michelson",
        "dsMinRunMass",
        f"{MICHELSON}#/profiled_diagnostics/final/min_run_mass",
        _scientific(1),
    ),
    Fact(
        "michelson", "dsAcceptedMoves", f"{MICHELSON}#/profiled_diagnostics/accepted_moves", _count
    ),
    Fact("michelson", "dsScans", f"{MICHELSON}#/profiled_diagnostics/scans", _count),
    Fact(
        "michelson",
        "dsRetentionGain",
        f"{MICHELSON}#/profiled_diagnostics/retention_gain",
        _scientific(1),
    ),
    Fact(
        "michelson",
        "dsRelabelledFraction",
        f"{MICHELSON}#/profiled_diagnostics/relabelled_fraction",
        _percent(1),
    ),
    *(
        Fact(
            "michelson",
            f"smoothing{name}{column}",
            f"{MICHELSON}#/profiled_diagnostics/smoothing/{index}/{pointer}",
            render,
        )
        for index, name in enumerate(("One", "Two", "Three", "Four"))
        for column, pointer, render in (
            ("MinWidth", "min_width", _fixed(2)),
            ("Runs", "n_runs", _count),
            ("BinsUsed", "bins_used", _count),
            ("Retention", "retention", _fixed(4)),
        )
    ),
    # The reusable profiled rule against the finite partition it cannot be
    # compiled from.
    Fact("michelson", "dsRuleRuns", f"{MICHELSON}#/rules/1/n_runs", _count),
    Fact("michelson", "dsRuleHardeningGap", f"{MICHELSON}#/rules/1/hardening_gap", _scientific(1)),
    Fact("michelson", "dsRulePredictionOne", f"{MICHELSON}#/rules/1/predictions/0/bin", _count),
    Fact("michelson", "dsRulePredictionTwo", f"{MICHELSON}#/rules/1/predictions/1/bin", _count),
    Fact("michelson", "dsRulePredictionThree", f"{MICHELSON}#/rules/1/predictions/2/bin", _count),
)


#: Aperture runs for every swept bin budget of the committed Michelson study,
#: read straight from its evidence rather than through the fact table: a
#: walkthrough chart needs the whole sweep's run-length encoding, not a single
#: formatted scalar.
MICHELSON_SWEEP_OUTPUT = ROOT / "website/src/generated/michelson-sweep.json"


def build_michelson_sweep() -> dict[str, object]:
    """Assemble the Michelson sweep's aperture runs for every swept bin budget.

    Read straight from the committed evidence file
    ``docs/examples/assets/michelson-phase.json`` -- the study's own sweep
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
#: FlowCyt is not included: it already has an on-demand table at
#: ``website/static/showcase-data/flowcyt-scores.json``.
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


def _build_hep_score_table() -> dict[str, object]:
    """Build the HEP-classifier walkthrough's score table.

    Reuses ``examples/hep_classifier/experiment.py``'s own out-of-fold
    assembly (`load_fixture`, `event_folds`, `fit_signal_background_oof`,
    `fit_tes_oof`, `assemble_score_sample`) rather than reimplementing it, at
    the same full-scale budget the committed study runs unless
    ``SCOREQUANT_EXAMPLE_FAST`` is set.

    Returns
    -------
    dict
        ``{"schema", "label", "detail", "scores", "weights"}``; 1,000 rows by
        3 columns at full scale.
    """
    sys.path.insert(0, str(ROOT))
    from examples._env import example_scale
    from examples.hep_classifier.data import load_fixture
    from examples.hep_classifier.experiment import FOLD_SEED, HEADLINE_DELTA
    from examples.hep_classifier.scores import (
        SCHEMA,
        assemble_score_sample,
        event_folds,
        fit_signal_background_oof,
        fit_tes_oof,
    )

    max_iter = example_scale(300, 60)
    n_folds = example_scale(5, 3)
    data = load_fixture()
    fold_ids = event_folds(data.is_signal, n_folds=n_folds, seed=FOLD_SEED)
    sigbg = fit_signal_background_oof(
        data, fold_ids=fold_ids, max_iter=max_iter, seed=FOLD_SEED + 100
    )
    tes = fit_tes_oof(
        data, delta=HEADLINE_DELTA, fold_ids=fold_ids, max_iter=max_iter, seed=FOLD_SEED + 500
    )
    sample = assemble_score_sample(data, sigbg, tes)
    rows, columns = sample.scores.shape
    return {
        "schema": list(SCHEMA.parameters),
        "label": "HEP classifier scores",
        "detail": (
            f"{rows:,} events × {columns} score dimensions · "
            "FAIR Universe HiggsML, CC BY 4.0 · cross-fitted out-of-fold classifier scores"
        ),
        "scores": _round(sample.scores),
        "weights": _round(sample.weights),
    }


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
    (``website/src/components/liveFit/types.ts``) needs. Two walkthroughs are
    absent from the table below for different reasons. FlowCyt already has an
    on-demand table at ``website/static/showcase-data/flowcyt-scores.json``.
    The density-ratio page carries no experiment at all: its argument is the
    gap between a reported and an achieved retention, which a refit of the
    estimated score cannot show, so it makes its point with a comparison and a
    figure instead (ADR 0033).

    Returns
    -------
    dict of int
        Byte size of each written file, keyed by slug.
    """
    builders: dict[str, Callable[[], dict[str, object]]] = {
        "hep": _build_hep_score_table,
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
    "hep-classifier.png": "docs/examples/assets/hep-classifier.png",
    "michelson-d-geometry.png": "docs/examples/assets/michelson-d-geometry.png",
    "michelson-profiled-ds.png": "docs/examples/assets/michelson-profiled-ds.png",
    "door3-classifier.png": "docs/examples/assets/door3-classifier.png",
}


def copy_walkthrough_figures() -> None:
    """Copy each example's committed figure into ``website/static/walkthrough-figures``.

    The figures live beside the studies that produce them, outside ``website/``,
    where Docusaurus cannot serve them. Copying at generate time keeps one
    committed copy of each image while still letting a walkthrough show it.
    """
    destination = ROOT / "website/static/walkthrough-figures"
    destination.mkdir(parents=True, exist_ok=True)
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
