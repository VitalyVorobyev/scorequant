"""Locked browser-lab adapter; kept outside the ScoreQuant package."""

from __future__ import annotations

import json

import numpy as np

import scorequant as sq

#: Browser ceiling, mirroring `LAB_LIMITS` in `src/lab/protocol.ts`. Six columns
#: admits the five-dimensional FlowCyt mixture score.
MAX_DIMENSIONS = 6
MAX_ROWS = 8_000


def _configuration(problem: dict[str, object], task: str) -> object:
    seed = int(problem["seed"])
    max_steps = int(problem.get("maxSteps", 120))
    solver = problem["solver"]
    if task == "optimize_partition":
        # `optimize_partition` is the fixed-sample task the committed Michelson
        # sweep was generated from (`examples/michelson_phase.py`), which calls
        # `sq.DExchangeConfig(seed=SEED)` with every other field at its class
        # default (`initializer_restarts=8`). Overriding restarts here, as the
        # `fit_quantizer` path below does, changes the exchange-stable optimum
        # the committed fixture reproduces.
        raw_max_scans = problem.get("maxScans")
        max_scans = None if raw_max_scans is None else int(raw_max_scans)
        if solver == "d_exchange":
            return sq.DExchangeConfig(seed=seed, max_iter=max_scans)
        if solver == "mahalanobis_lloyd":
            return sq.MahalanobisLloydConfig(seed=seed, max_iter=max_steps)
        raise ValueError(f"unsupported browser solver for optimize_partition: {solver}")
    max_scans = int(problem.get("maxScans", 120))
    if solver == "d_exchange":
        return sq.DExchangeConfig(seed=seed, initializer_restarts=2, max_iter=max_scans)
    if solver == "mahalanobis_lloyd":
        return sq.MahalanobisLloydConfig(seed=seed, initializer_restarts=2, max_iter=max_steps)
    if solver == "kmeans":
        return sq.KMeansConfig(seed=seed, solver_restarts=2, max_iter=max_steps)
    if solver == "scalar_dp":
        return sq.ScalarDPConfig(max_rows=MAX_ROWS)
    if solver == "soft_voronoi":
        return sq.SoftVoronoiConfig(
            seed=seed,
            initializer_restarts=2,
            kmeans_max_iter=min(80, max_steps),
            max_iter=max_steps,
            record_every=max(1, max_steps // 10),
        )
    raise ValueError(f"unsupported browser solver: {solver}")


def _criterion(problem: dict[str, object], schema: object) -> object:
    """Build the declared objective, naming parameters of interest when given.

    The schema is what makes ``interest=("HSPCs",)`` possible here: without it
    the browser would have to send column indices and the reader would have to
    know what column four means.
    """
    declared = problem.get("criterion")
    if declared is None:
        return sq.DOptimality()
    name = declared["name"]
    if name == "d_optimality":
        return sq.DOptimality()
    if name == "normalized_trace":
        return sq.NormalizedTrace()
    if name == "profiled_d_optimality":
        interest = tuple(declared.get("interest") or ())
        if not interest:
            raise ValueError("profiled D_s requires at least one parameter of interest")
        if schema is None:
            raise ValueError("profiled D_s by name requires a score schema")
        return sq.ProfiledDOptimality(interest=interest)
    raise ValueError(f"unsupported browser criterion: {name}")


def _centers_skipping_empty_cells(
    scores: np.ndarray, weights: np.ndarray, labels: np.ndarray, n_bins: int
) -> np.ndarray:
    """Return the weighted mean score of every nonempty requested cell.

    ``optimize_partition`` guarantees every requested cell stays nonempty, but
    this reads centers back from the original rows by label rather than from
    the library's own cell statistics, so it stays defensive rather than
    trusting that invariant a second time.
    """
    rows = [
        np.average(scores[mask], axis=0, weights=weights[mask])
        for cell in range(n_bins)
        if (mask := labels == cell).any()
    ]
    return np.vstack(rows) if rows else np.zeros((0, scores.shape[1]))


def _criterion_label(criterion: object, profiled: object) -> str:
    """Name the objective the reported retention is measured against."""
    if isinstance(criterion, sq.ProfiledDOptimality):
        names = getattr(profiled, "interest_names", None)
        shown = ", ".join(names) if names else ", ".join(str(i) for i in criterion.interest)
        return f"Profiled D_s ({shown})"
    if isinstance(criterion, sq.NormalizedTrace):
        return "Normalized trace"
    return "D-optimality"


def run_lab(payload: str) -> str:
    """Run one bounded ScoreQuant problem and return protocol JSON."""
    problem = json.loads(payload)
    scores = np.asarray(problem["scores"], dtype=np.float64)
    weights = np.asarray(problem["weights"], dtype=np.float64)
    if scores.ndim != 2 or not 1 <= scores.shape[1] <= MAX_DIMENSIONS or scores.shape[0] > MAX_ROWS:
        raise ValueError("browser score capacity exceeded")
    names = problem.get("schema")
    schema = None if names is None else sq.ScoreSchema(tuple(str(name) for name in names))
    criterion = _criterion(problem, schema)
    execution = sq.ExecutionConfig(backend="numpy", precision="float64", device="cpu")
    task = problem.get("task", "fit_quantizer")
    n_bins = int(problem["nBins"])

    if task == "optimize_partition":
        initial_labels = None
        if problem.get("initialization") == "efficient_score_bound":
            if schema is None:
                raise ValueError("efficient_score_bound initialization requires a score schema")
            if not isinstance(criterion, sq.ProfiledDOptimality):
                raise ValueError(
                    "efficient_score_bound initialization requires profiled_d_optimality"
                )
            indices = schema.select(*criterion.interest)
            bound = sq.efficient_score_bound(
                scores,
                interest=indices,
                weights=weights,
                n_bins=n_bins,
                execution=execution,
            )
            initial_labels = bound.labels
        # A named ``interest`` (built by ``_criterion`` above) only resolves
        # against a schema carried on the sample itself: the raw
        # ``scores``/``weights`` shorthand of ``optimize_partition`` never
        # attaches one.
        partition_sample = sq.ScoreSample(scores=scores, weights=weights, schema=schema)
        result = sq.optimize_partition(
            partition_sample,
            n_bins=n_bins,
            criterion=criterion,
            config=_configuration(problem, task),
            initial_labels=initial_labels,
            execution=execution,
        )
        labels = np.asarray(result.labels)
        centers = _centers_skipping_empty_cells(scores, weights, labels, n_bins)
        profiled = result.profiled_report
        # A profiled run must report the profiled retention: the full-D retention of
        # a D_s fit is a different number answering a different question, and
        # showing it beside a "profiled" label would be a quiet lie.
        retention = (
            result.train_report.geometric_mean_retention
            if profiled is None
            else profiled.geometric_mean_retention
        )
        payload_out: dict[str, object] = {
            "labels": labels.tolist(),
            "centers": centers.tolist(),
            "retention": retention,
            "objective": float(result.objective),
            "execution": "numpy/float64/cpu",
            "criterionLabel": _criterion_label(criterion, profiled),
            "exchangeStable": bool(result.exchange_stable),
        }
        if profiled is not None and profiled.interest_names is not None:
            payload_out["interest"] = list(profiled.interest_names)
    else:
        source = sq.ScoreSample(scores=scores, weights=weights, schema=schema)
        result = sq.fit_quantizer(
            source,
            n_bins=n_bins,
            criterion=criterion,
            config=_configuration(problem, task),
            execution=execution,
        )
        labels = np.asarray(result.labels)
        centers = np.vstack(
            [
                np.average(scores[labels == cell], axis=0, weights=weights[labels == cell])
                for cell in range(result.n_bins)
            ]
        )
        profiled = result.train_profiled_report
        # A profiled run must report the profiled retention: the full-D retention of
        # a D_s fit is a different number answering a different question, and
        # showing it beside a "profiled" label would be a quiet lie.
        retention = (
            result.train_report.geometric_mean_retention
            if profiled is None
            else profiled.geometric_mean_retention
        )
        payload_out = {
            "labels": labels.tolist(),
            "centers": centers.tolist(),
            "retention": retention,
            "objective": float(result.trace.objective[-1]),
            "execution": "numpy/float64/cpu",
            "criterionLabel": _criterion_label(criterion, profiled),
        }
        if profiled is not None and profiled.interest_names is not None:
            payload_out["interest"] = list(profiled.interest_names)

    report = problem.get("report")
    if isinstance(report, dict):
        interest_names = report.get("profiledInterest")
        if interest_names:
            if schema is None:
                raise ValueError("report.profiledInterest requires a score schema")
            indices = schema.select(*interest_names)
            payload_out["profiledRetention"] = float(
                sq.information.profiled_information_report(
                    scores,
                    labels,
                    interest=indices,
                    weights=weights,
                    n_bins=n_bins,
                    schema=schema,
                    execution=execution,
                ).geometric_mean_retention
            )

    return json.dumps(payload_out)
