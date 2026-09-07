from __future__ import annotations

import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np

ROOT = Path(__file__).parents[1]


def _run_lab() -> Callable[[str], str]:
    namespace = runpy.run_path(ROOT / "website/static/runtime/python/scorequant_browser_lab.py")
    return cast(Callable[[str], str], namespace["run_lab"])


def _flowcyt_table() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The committed fixture-scale FlowCyt walkthrough table's partition rows."""
    with np.load(ROOT / "examples/data/flowcyt_walkthrough.npz") as table:
        scores = np.asarray(table["partition_scores"], dtype=np.float64)
        weights = np.asarray(table["partition_weights"], dtype=np.float64)
        schema = [str(name) for name in table["schema"]]
    return scores, weights / weights.mean(), schema


def _flowcyt_problem(**overrides: object) -> dict[str, object]:
    """Build a request from the committed FlowCyt walkthrough table."""
    scores, weights, schema = _flowcyt_table()
    # A slice keeps the pin fast; the browser envelope is exercised by the shape
    # assertions rather than by re-running seven thousand rows here.
    rows = 600
    problem: dict[str, object] = {
        "scores": scores[:rows].tolist(),
        "weights": weights[:rows].tolist(),
        "schema": schema,
        "nBins": 6,
        "solver": "d_exchange",
        "seed": 28,
        "maxSteps": 120,
        "maxScans": 120,
    }
    problem.update(overrides)
    return problem


def test_the_committed_walkthrough_table_is_within_the_browser_envelope() -> None:
    """The Lab caps a run, and the FlowCyt partition rows must fit under that cap."""
    scores, weights, schema = _flowcyt_table()
    facts = json.loads((ROOT / "examples/data/flowcyt_walkthrough.json").read_text())
    assert scores.ndim == 2
    assert scores.shape[0] <= 8_000, "the browser refuses more rows than this"
    assert scores.shape[1] <= 6, "the browser refuses more score dimensions than this"
    assert scores.shape[1] == len(schema)
    assert weights.shape[0] == scores.shape[0]
    assert np.isfinite(scores).all()
    # The mixture score absorbs one component, so five columns for six
    # populations. A change here means the score construction changed.
    assert scores.shape[1] == 5
    assert facts["license"] == "CC-BY-NC-SA-4.0"
    assert facts["rows"]["partition"] == scores.shape[0]


def test_the_browser_adapter_runs_the_real_five_dimensional_scores() -> None:
    """Five dimensions is what v1's four-column ceiling excluded."""
    result = json.loads(_run_lab()(json.dumps(_flowcyt_problem())))
    assert len(result["labels"]) == 600
    assert np.asarray(result["centers"]).shape == (6, 5)
    assert 0.0 < result["retention"] <= 1.0
    assert result["criterionLabel"] == "D-optimality"
    assert "interest" not in result


def test_a_profiled_run_reports_the_profiled_retention_by_name() -> None:
    """The full-D retention of a D_s fit answers a different question."""
    run_lab = _run_lab()
    profiled = json.loads(
        run_lab(
            json.dumps(
                _flowcyt_problem(
                    solver="soft_voronoi",
                    maxSteps=40,
                    criterion={"name": "profiled_d_optimality", "interest": ["HSPCs"]},
                )
            )
        )
    )
    assert profiled["criterionLabel"] == "Profiled D_s (HSPCs)"
    assert profiled["interest"] == ["HSPCs"]

    plain = json.loads(run_lab(json.dumps(_flowcyt_problem(solver="soft_voronoi", maxSteps=40))))
    # Different objectives, therefore different reported numbers. If these
    # agreed, the profiled label would be decorating a full-D retention.
    assert profiled["retention"] != plain["retention"]


def test_an_unknown_criterion_is_refused_rather_than_defaulted() -> None:
    run_lab = _run_lab()
    for criterion, message in (
        ({"name": "e_optimality"}, "unsupported browser criterion"),
        ({"name": "profiled_d_optimality"}, "at least one parameter of interest"),
    ):
        try:
            run_lab(json.dumps(_flowcyt_problem(criterion=criterion)))
        except ValueError as error:
            assert message in str(error)
        else:  # pragma: no cover - the adapter must not silently accept these
            raise AssertionError(f"expected {criterion} to be refused")


def test_browser_adapter_matches_committed_numpy_fixture() -> None:
    """The browser adapter reproduces the `/get-started` page's own committed fit.

    ``website/scripts/get_started_program.py`` fits its first partition with
    ``sq.optimize_partition(scores, weights=weights, n_bins=5,
    config=sq.DExchangeConfig(seed=21))`` -- every other field at its class
    default, in particular ``initializer_restarts=8`` and
    ``max_scans=None``. ``website/src/generated/snippet-outputs.json``
    (via ``website/src/lib/snippets.ts::firstFitRetention``) pins that run's
    own ``partition.train_report.geometric_mean_retention``. Reproducing that
    recipe through the browser adapter -- ``task: "optimize_partition"``, no
    ``maxScans`` override -- must land on the same retention.
    """
    table = json.loads((ROOT / "website/static/walkthrough-scores/get-started.json").read_text())
    problem = {
        "scores": table["scores"],
        "weights": table["weights"],
        "schema": table["schema"],
        "nBins": table["nBins"],
        "solver": table["solver"],
        "seed": table["seed"],
        "task": "optimize_partition",
    }
    result = json.loads(_run_lab()(json.dumps(problem)))

    snippet_outputs = json.loads((ROOT / "website/src/generated/snippet-outputs.json").read_text())
    committed_retention = snippet_outputs["firstFit"]["retention"]

    np.testing.assert_allclose(result["retention"], committed_retention, rtol=1e-9, atol=1e-9)


def test_the_browser_adapter_runs_optimize_partition_shape() -> None:
    """`optimize_partition` returns a result shaped like `fit_quantizer`, plus `exchangeStable`."""
    result = json.loads(_run_lab()(json.dumps(_flowcyt_problem(task="optimize_partition"))))
    assert len(result["labels"]) == 600
    assert np.asarray(result["centers"]).shape == (6, 5)
    assert 0.0 < result["retention"] <= 1.0
    assert result["criterionLabel"] == "D-optimality"
    assert result["exchangeStable"] is True
    assert "interest" not in result
    assert "profiledRetention" not in result


def _michelson_sweep_rows() -> list[dict[str, object]]:
    fixture = json.loads((ROOT / "docs/examples/assets/michelson-phase.json").read_text())
    return cast(list[dict[str, object]], fixture["sweep"])


def _michelson_score_table() -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads((ROOT / "website/static/walkthrough-scores/michelson.json").read_text()),
    )


def test_optimize_partition_reproduces_the_committed_michelson_sweep() -> None:
    """`optimize_partition` + `report.profiledInterest` should match the committed sweep."""
    run_lab = _run_lab()
    table = _michelson_score_table()
    for row in _michelson_sweep_rows():
        problem = {
            "scores": table["scores"],
            "weights": table["weights"],
            "schema": table["schema"],
            "nBins": row["n_bins"],
            "solver": "d_exchange",
            "seed": 4,
            "task": "optimize_partition",
            "report": {"profiledInterest": ["phase"]},
        }
        result = json.loads(run_lab(json.dumps(problem)))
        assert result["exchangeStable"] is True
        np.testing.assert_allclose(
            result["profiledRetention"], row["d_optimal_retention"], rtol=1e-12, atol=1e-12
        )


def test_bound_seeded_profiled_exchange_reproduces_the_committed_headline() -> None:
    """Profiled D_s seeded from `efficient_score_bound` should match the committed headline."""
    run_lab = _run_lab()
    table = _michelson_score_table()
    for row in _michelson_sweep_rows():
        problem = {
            "scores": table["scores"],
            "weights": table["weights"],
            "schema": table["schema"],
            "nBins": row["n_bins"],
            "solver": "d_exchange",
            "seed": 4,
            "task": "optimize_partition",
            "criterion": {"name": "profiled_d_optimality", "interest": ["phase"]},
            "initialization": "efficient_score_bound",
        }
        result = json.loads(run_lab(json.dumps(problem)))
        np.testing.assert_allclose(
            result["retention"], row["profiled_retention"], rtol=1e-12, atol=1e-12
        )
