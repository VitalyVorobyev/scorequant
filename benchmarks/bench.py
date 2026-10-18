"""Deterministic timing and quality benchmark harness for ScoreQuant solvers.

Every scenario below is seeded and reproducible: two runs on the same machine
with the same flags produce identical quality metrics (objective or
retention) and comparable wall-clock times. This is the harness the CI
regression job and ``benchmarks/baselines.json`` are built from; see
``docs/development.md`` for the usage summary.

Timing measures wall-clock seconds around the public call under test, forcing
JAX to materialize its output with ``jax.block_until_ready`` before stopping
the clock. Peak RSS is process-lifetime (``resource.getrusage``), not a
per-scenario delta, so it only ever grows across a multi-scenario run; treat
it as a rough ceiling, not a precise measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp

import scorequant as sq

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

_DEFAULT_ROWS = "20000,100000"
_DEFAULT_DIMS = 3
_DEFAULT_BINS = "8,64"
_DEFAULT_SEED = 2026
_DEFAULT_TIME_TOLERANCE = 2.5
_DEFAULT_QUALITY_RTOL = 1e-6
_SCENARIO_NAMES = (
    "d_exchange",
    "d_exchange_nobatch",
    "lloyd",
    "kmeans",
    "soft",
    "scalar_dp",
    "profiled_exchange",
    "predict",
    "compile",
    "certify",
)
# Mirrors ScalarDPConfig's own default so a future change to that default is
# picked up here automatically instead of silently drifting.
_SCALAR_DP_DEFAULT_MAX_ROWS = sq.ScalarDPConfig().max_rows
# Single-move acceptance runs one complete scan per accepted move, so its cost
# grows as (accepted moves) x (scan cost) rather than (scans) x (scan cost).
# Above this row count the cell is measured in hours, which is a result worth
# recording in benchmarks/README.md but not worth re-running in a matrix sweep.
_NOBATCH_MAX_ROWS = 20_000
# Hard capacity of the branch-and-bound certifier's recursion depth.
_CERTIFY_MAX_ROWS = 512


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """One matrix cell: a scenario paired with its deterministic data shape.

    ``max_scans`` caps the exchange-family scan budget. ``None`` is the library
    default (run to exchange stability) and is what ``baselines.json`` records;
    a finite cap turns a cell into a fixed-work steady-state probe, which is how
    the profiling campaign measures per-scan cost at row counts whose full
    convergence takes far longer than one measurement window.
    """

    scenario: str
    rows: int
    dims: int
    bins: int
    seed: int
    max_scans: int | None = None


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """One measured (or skipped) benchmark run."""

    config: ScenarioConfig
    elapsed_seconds: float | None
    peak_rss_megabytes: float | None
    quality: float | None
    quality_label: str
    extra: dict[str, JsonValue]
    skipped: bool = False
    skip_reason: str | None = None

    def to_json(self) -> dict[str, JsonValue]:
        """Return a JSON-ready record of this run."""
        return {
            "scenario": self.config.scenario,
            "rows": self.config.rows,
            "dims": self.config.dims,
            "bins": self.config.bins,
            "seed": self.config.seed,
            "max_scans": self.config.max_scans,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_rss_megabytes": self.peak_rss_megabytes,
            "quality_label": self.quality_label,
            "quality": self.quality,
            "extra": self.extra,
        }


def _peak_rss_megabytes() -> float:
    """Return process-lifetime peak RSS in MiB.

    ``ru_maxrss`` is bytes on macOS/BSD and kilobytes on Linux; both platforms
    run this harness in CI, so the unit is resolved explicitly rather than
    assumed.
    """
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return peak / divisor


def _synthetic_sample(seed: int, rows: int, dims: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a deterministic standard-normal score sample with mild weight variation."""
    key = jax.random.PRNGKey(seed)
    score_key, weight_key = jax.random.split(key)
    scores = jax.random.normal(score_key, (rows, dims))
    weights = jax.random.uniform(weight_key, (rows,), minval=0.5, maxval=1.5)
    return scores, weights


def _run_timed[R](
    call: Callable[[], R], primary: Callable[[R], jnp.ndarray], repeats: int
) -> tuple[float, R]:
    """Return the minimum wall time over ``repeats`` calls and the last result."""
    best = math.inf
    result: R | None = None
    for _ in range(max(1, repeats)):
        started = perf_counter()
        result = call()
        jax.block_until_ready(primary(result))
        best = min(best, perf_counter() - started)
    assert result is not None  # the loop always executes at least once
    return best, result


def _partition_extra(result: sq.PartitionResult) -> dict[str, JsonValue]:
    return {
        "accepted_moves": int(result.accepted_moves),
        "scans": int(result.scans),
        "exchange_stable": bool(result.exchange_stable),
        "best_remaining_gain": float(result.best_remaining_gain),
        "lloyd_iterations": int(result.lloyd_iterations),
        "accepted_lloyd_steps": int(result.accepted_lloyd_steps),
    }


def _bench_d_exchange(cfg: ScenarioConfig, repeats: int) -> RunOutcome:
    scores, weights = _synthetic_sample(cfg.seed, cfg.rows, cfg.dims)
    config = sq.DExchangeConfig(seed=cfg.seed, max_iter=cfg.max_scans)
    elapsed, result = _run_timed(
        lambda: sq.optimize_partition(scores, weights=weights, n_bins=cfg.bins, config=config),
        lambda r: r.labels,
        repeats,
    )
    return RunOutcome(
        cfg,
        elapsed,
        _peak_rss_megabytes(),
        float(result.objective),
        "logdet_objective",
        _partition_extra(result),
    )


def _bench_d_exchange_nobatch(cfg: ScenarioConfig, repeats: int) -> RunOutcome:
    scores, weights = _synthetic_sample(cfg.seed, cfg.rows, cfg.dims)
    config = sq.DExchangeConfig(seed=cfg.seed, batch_moves=False, max_iter=cfg.max_scans)
    elapsed, result = _run_timed(
        lambda: sq.optimize_partition(scores, weights=weights, n_bins=cfg.bins, config=config),
        lambda r: r.labels,
        repeats,
    )
    return RunOutcome(
        cfg,
        elapsed,
        _peak_rss_megabytes(),
        float(result.objective),
        "logdet_objective",
        _partition_extra(result),
    )


def _bench_lloyd(cfg: ScenarioConfig, repeats: int) -> RunOutcome:
    scores, weights = _synthetic_sample(cfg.seed, cfg.rows, cfg.dims)
    config = sq.MahalanobisLloydConfig(seed=cfg.seed)
    elapsed, result = _run_timed(
        lambda: sq.optimize_partition(scores, weights=weights, n_bins=cfg.bins, config=config),
        lambda r: r.labels,
        repeats,
    )
    return RunOutcome(
        cfg,
        elapsed,
        _peak_rss_megabytes(),
        float(result.objective),
        "logdet_objective",
        _partition_extra(result),
    )


def _bench_profiled_exchange(cfg: ScenarioConfig, repeats: int) -> RunOutcome:
    scores, weights = _synthetic_sample(cfg.seed, cfg.rows, cfg.dims)
    config = sq.DExchangeConfig(seed=cfg.seed, max_iter=cfg.max_scans)
    criterion = sq.ProfiledDOptimality(interest=(0,))
    elapsed, result = _run_timed(
        lambda: sq.optimize_partition(
            scores, weights=weights, n_bins=cfg.bins, criterion=criterion, config=config
        ),
        lambda r: r.labels,
        repeats,
    )
    return RunOutcome(
        cfg,
        elapsed,
        _peak_rss_megabytes(),
        float(result.objective),
        "profiled_logdet_objective",
        _partition_extra(result),
    )


def _bench_kmeans(cfg: ScenarioConfig, repeats: int) -> RunOutcome:
    scores, weights = _synthetic_sample(cfg.seed, cfg.rows, cfg.dims)
    sample = sq.ScoreSample(scores, weights)
    config = sq.KMeansConfig(seed=cfg.seed)
    elapsed, result = _run_timed(
        lambda: sq.fit_quantizer(
            sample, n_bins=cfg.bins, criterion=sq.NormalizedTrace(), config=config
        ),
        lambda r: r.labels,
        repeats,
    )
    extra: dict[str, JsonValue] = {
        "hardening_gap": None if result.hardening_gap is None else float(result.hardening_gap)
    }
    return RunOutcome(
        cfg,
        elapsed,
        _peak_rss_megabytes(),
        float(result.train_report.geometric_mean_retention),
        "geometric_mean_retention",
        extra,
    )


def _bench_soft(cfg: ScenarioConfig, repeats: int) -> RunOutcome:
    scores, weights = _synthetic_sample(cfg.seed, cfg.rows, cfg.dims)
    sample = sq.ScoreSample(scores, weights)
    config = sq.SoftVoronoiConfig(seed=cfg.seed, max_iter=200)
    elapsed, result = _run_timed(
        lambda: sq.fit_quantizer(
            sample, n_bins=cfg.bins, criterion=sq.DOptimality(), config=config
        ),
        lambda r: r.labels,
        repeats,
    )
    extra: dict[str, JsonValue] = {
        "hardening_gap": None if result.hardening_gap is None else float(result.hardening_gap)
    }
    return RunOutcome(
        cfg,
        elapsed,
        _peak_rss_megabytes(),
        float(result.train_report.geometric_mean_retention),
        "geometric_mean_retention",
        extra,
    )


def _bench_scalar_dp(cfg: ScenarioConfig, repeats: int) -> RunOutcome:
    scores, weights = _synthetic_sample(cfg.seed, cfg.rows, cfg.dims)
    sample = sq.ScoreSample(scores, weights)
    config = sq.ScalarDPConfig()
    elapsed, result = _run_timed(
        lambda: sq.fit_quantizer(
            sample, n_bins=cfg.bins, criterion=sq.DOptimality(), config=config
        ),
        lambda r: r.labels,
        repeats,
    )
    return RunOutcome(
        cfg,
        elapsed,
        _peak_rss_megabytes(),
        float(result.train_report.geometric_mean_retention),
        "geometric_mean_retention",
        {},
    )


def _bench_certify(cfg: ScenarioConfig, repeats: int) -> RunOutcome:
    """Time global branch-and-bound certification of an exchange incumbent.

    Unlike every other cell this one is pure NumPy and pure Python recursion,
    with no JAX kernel in the inner loop, so its cost is measured in nodes per
    second rather than in array throughput. That difference is the whole reason
    it is in the matrix; see ``benchmarks/README.md``.
    """
    scores, weights = _synthetic_sample(cfg.seed, cfg.rows, cfg.dims)
    config = sq.CertificationConfig(max_rows=cfg.rows)
    incumbent = sq.optimize_partition(
        scores, weights=weights, n_bins=cfg.bins, config=sq.DExchangeConfig(seed=cfg.seed)
    )
    elapsed, certificate = _run_timed(
        lambda: sq.certify_partition(
            scores,
            weights=weights,
            n_bins=cfg.bins,
            incumbent=incumbent.labels,
            config=config,
        ),
        lambda c: jnp.asarray(c.labels),
        repeats,
    )
    extra: dict[str, JsonValue] = {
        "nodes_explored": int(certificate.nodes_explored),
        "nodes_per_second": int(certificate.nodes_explored / elapsed) if elapsed > 0 else None,
        "status": str(certificate.status),
        "incumbent_was_optimal": bool(certificate.incumbent_was_optimal),
    }
    return RunOutcome(
        cfg,
        elapsed,
        _peak_rss_megabytes(),
        float(certificate.objective),
        "certified_logdet_objective",
        extra,
    )


def _bench_predict(cfg: ScenarioConfig, repeats: int) -> RunOutcome:
    train_scores, train_weights = _synthetic_sample(cfg.seed, cfg.rows, cfg.dims)
    held_out_scores, held_out_weights = _synthetic_sample(cfg.seed + 1, cfg.rows, cfg.dims)
    quantizer = sq.fit_quantizer(
        sq.ScoreSample(train_scores, train_weights),
        n_bins=cfg.bins,
        criterion=sq.NormalizedTrace(),
        config=sq.KMeansConfig(seed=cfg.seed, solver_restarts=1, max_iter=20),
    )
    elapsed, labels = _run_timed(
        lambda: quantizer.predict_scores(held_out_scores),
        lambda x: x,
        repeats,
    )
    held_out_report = quantizer.evaluate_scores(held_out_scores, held_out_weights)
    extra: dict[str, JsonValue] = {"predicted_rows": int(labels.shape[0])}
    return RunOutcome(
        cfg,
        elapsed,
        _peak_rss_megabytes(),
        float(held_out_report.geometric_mean_retention),
        "held_out_geometric_mean_retention",
        extra,
    )


def _bench_compile(cfg: ScenarioConfig, repeats: int) -> RunOutcome:
    """Time ``compile_quantizer`` on an exchange-stable D partition, then predict.

    The exchange itself is excluded from the clock: this cell measures the
    theorem-backed compilation (which re-predicts every training row to verify
    the Mahalanobis rule reproduces the partition) plus one held-out predict.
    """
    scores, weights = _synthetic_sample(cfg.seed, cfg.rows, cfg.dims)
    held_out_scores, held_out_weights = _synthetic_sample(cfg.seed + 1, cfg.rows, cfg.dims)
    partition = sq.optimize_partition(
        scores,
        weights=weights,
        n_bins=cfg.bins,
        config=sq.DExchangeConfig(seed=cfg.seed, max_iter=cfg.max_scans),
    )
    elapsed, quantizer = _run_timed(
        partition.compile_quantizer,
        # Blocking on the centers rather than on labels: a compiled Quantizer is
        # a rule, and a rule has no labels. Those belong to the fit over a
        # particular sample, which is the distinction the artifact draws.
        lambda q: q.centers,
        repeats,
    )
    held_out_report = quantizer.evaluate_scores(held_out_scores, held_out_weights)
    extra: dict[str, JsonValue] = {"exchange_stable": bool(partition.exchange_stable)}
    return RunOutcome(
        cfg,
        elapsed,
        _peak_rss_megabytes(),
        float(held_out_report.geometric_mean_retention),
        "held_out_geometric_mean_retention",
        extra,
    )


_RUNNERS: dict[str, Callable[[ScenarioConfig, int], RunOutcome]] = {
    "d_exchange": _bench_d_exchange,
    "d_exchange_nobatch": _bench_d_exchange_nobatch,
    "lloyd": _bench_lloyd,
    "kmeans": _bench_kmeans,
    "soft": _bench_soft,
    "scalar_dp": _bench_scalar_dp,
    "profiled_exchange": _bench_profiled_exchange,
    "predict": _bench_predict,
    "compile": _bench_compile,
    "certify": _bench_certify,
}


def _skip_reason(scenario: str, rows: int, dims: int, bins: int) -> str | None:
    """Return why a matrix cell is unrunnable, or ``None`` when it is fine."""
    del dims  # forced per-scenario at matrix-build time; kept for a stable signature
    if scenario == "scalar_dp" and rows > _SCALAR_DP_DEFAULT_MAX_ROWS:
        return f"rows exceeds ScalarDPConfig default max_rows={_SCALAR_DP_DEFAULT_MAX_ROWS}"
    if scenario == "certify" and rows > _CERTIFY_MAX_ROWS:
        return (
            f"global certification is exponential in the atom count; "
            f"CertificationConfig refuses more than {_CERTIFY_MAX_ROWS} atoms"
        )
    if scenario == "d_exchange_nobatch" and rows > _NOBATCH_MAX_ROWS:
        return (
            f"single-move acceptance needs one full scan per accepted move; "
            f"above {_NOBATCH_MAX_ROWS} rows the cell runs for hours"
        )
    if scenario == "profiled_exchange" and bins < 2:
        return "profiled_exchange requires n_bins >= 2 (one interest, one nuisance)"
    return None


def _scenario_dims(scenario: str, dims: int) -> int:
    if scenario == "scalar_dp":
        return 1
    if scenario == "profiled_exchange":
        return max(2, dims)
    return dims


def _build_matrix(
    rows_list: Sequence[int],
    dims: int,
    bins_list: Sequence[int],
    scenarios: Sequence[str],
    seed: int,
    max_scans: int | None = None,
) -> list[ScenarioConfig]:
    matrix: list[ScenarioConfig] = []
    for scenario in scenarios:
        scenario_dims = _scenario_dims(scenario, dims)
        for rows in rows_list:
            for bins in bins_list:
                matrix.append(ScenarioConfig(scenario, rows, scenario_dims, bins, seed, max_scans))
    return matrix


def _run_matrix(matrix: Sequence[ScenarioConfig], repeats: int) -> list[RunOutcome]:
    outcomes: list[RunOutcome] = []
    for cfg in matrix:
        reason = _skip_reason(cfg.scenario, cfg.rows, cfg.dims, cfg.bins)
        if reason is not None:
            outcomes.append(
                RunOutcome(cfg, None, None, None, "", {}, skipped=True, skip_reason=reason)
            )
            continue
        outcomes.append(_RUNNERS[cfg.scenario](cfg, repeats))
    return outcomes


def _run_from_json(record: dict[str, JsonValue], repeats: int) -> RunOutcome:
    """Re-run the exact scenario shape a baseline JSON record describes."""
    scenario = str(record["scenario"])
    recorded_scans = record.get("max_scans")
    cfg = ScenarioConfig(
        scenario=scenario,
        rows=int(record["rows"]),  # type: ignore[arg-type]
        dims=int(record["dims"]),  # type: ignore[arg-type]
        bins=int(record["bins"]),  # type: ignore[arg-type]
        seed=int(record["seed"]),  # type: ignore[arg-type]
        max_scans=None if recorded_scans is None else int(recorded_scans),  # type: ignore[arg-type]
    )
    reason = _skip_reason(cfg.scenario, cfg.rows, cfg.dims, cfg.bins)
    if reason is not None:
        return RunOutcome(cfg, None, None, None, "", {}, skipped=True, skip_reason=reason)
    return _RUNNERS[scenario](cfg, repeats)


def _environment() -> dict[str, JsonValue]:
    """Describe the machine and library versions a run executed under."""
    import os

    return {
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jax_enable_x64": os.environ.get("JAX_ENABLE_X64"),
        "jax_default_backend": jax.default_backend(),
    }


def _format_extra(extra: dict[str, JsonValue]) -> str:
    return " ".join(f"{key}={value}" for key, value in extra.items())


def _format_table(outcomes: Sequence[RunOutcome]) -> str:
    header = (
        f"{'scenario':<18}{'rows':>8}{'dims':>6}{'bins':>6}"
        f"{'elapsed_s':>12}{'peak_rss_mb':>13}{'quality':>18}  extra"
    )
    lines = [header, "-" * len(header)]
    for outcome in outcomes:
        cfg = outcome.config
        prefix = f"{cfg.scenario:<18}{cfg.rows:>8}{cfg.dims:>6}{cfg.bins:>6}"
        if outcome.skipped:
            lines.append(f"{prefix}{'skipped':>12}{'':>13}{'':>18}  {outcome.skip_reason}")
            continue
        assert outcome.elapsed_seconds is not None
        assert outcome.peak_rss_megabytes is not None
        assert outcome.quality is not None
        elapsed = f"{outcome.elapsed_seconds:.4f}"
        rss = f"{outcome.peak_rss_megabytes:.1f}"
        quality = f"{outcome.quality:.6f}"
        lines.append(f"{prefix}{elapsed:>12}{rss:>13}{quality:>18}  {_format_extra(outcome.extra)}")
    return "\n".join(lines)


def _quality_regressed(baseline: float, fresh: float, rtol: float) -> bool:
    scale = max(abs(baseline), 1e-12)
    return abs(fresh - baseline) > rtol * scale


def _format_check_table(rows: Sequence[dict[str, JsonValue]]) -> str:
    header = (
        f"{'scenario':<18}{'rows':>8}{'bins':>6}{'base_s':>10}{'new_s':>10}{'ratio':>8}"
        f"{'base_q':>16}{'new_q':>16}{'status':>18}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['scenario']!s:<18}{row['rows']!s:>8}{row['bins']!s:>6}"
            f"{row['base_s']!s:>10}{row['new_s']!s:>10}{row['ratio']!s:>8}"
            f"{row['base_q']!s:>16}{row['new_q']!s:>16}{row['status']!s:>18}"
        )
    return "\n".join(lines)


def _run_check(
    baseline_path: Path, repeats: int, time_tolerance: float, quality_rtol: float
) -> int:
    baseline_document = json.loads(baseline_path.read_text())
    baseline_runs: list[dict[str, JsonValue]] = baseline_document["runs"]
    table_rows: list[dict[str, JsonValue]] = []
    failed = False
    for record in baseline_runs:
        fresh = _run_from_json(record, repeats)
        record_skipped = bool(record.get("skipped", False))
        if record_skipped or fresh.skipped:
            table_rows.append(
                {
                    "scenario": record["scenario"],
                    "rows": record["rows"],
                    "bins": record["bins"],
                    "base_s": "skip",
                    "new_s": "skip",
                    "ratio": "-",
                    "base_q": "-",
                    "new_q": "-",
                    "status": "SKIPPED",
                }
            )
            continue
        base_elapsed = float(record["elapsed_seconds"])  # type: ignore[arg-type]
        base_quality = float(record["quality"])  # type: ignore[arg-type]
        assert fresh.elapsed_seconds is not None
        assert fresh.quality is not None
        ratio = fresh.elapsed_seconds / base_elapsed if base_elapsed > 0 else math.inf
        slow = ratio > time_tolerance
        quality_regressed = _quality_regressed(base_quality, fresh.quality, quality_rtol)
        stability_regressed = False
        base_stable = record.get("extra", {}).get("exchange_stable")  # type: ignore[union-attr]
        if isinstance(base_stable, bool):
            fresh_stable = fresh.extra.get("exchange_stable")
            stability_regressed = bool(base_stable) and fresh_stable is False
        status = "OK"
        if slow:
            status = "SLOW"
        if quality_regressed:
            status = "QUALITY REGRESSION" if not slow else "SLOW + QUALITY REGRESSION"
        if stability_regressed:
            status = f"{status} + UNSTABLE" if status != "OK" else "UNSTABLE"
        failed = failed or slow or quality_regressed or stability_regressed
        table_rows.append(
            {
                "scenario": record["scenario"],
                "rows": record["rows"],
                "bins": record["bins"],
                "base_s": f"{base_elapsed:.4f}",
                "new_s": f"{fresh.elapsed_seconds:.4f}",
                "ratio": f"{ratio:.2f}",
                "base_q": f"{base_quality:.6f}",
                "new_q": f"{fresh.quality:.6f}",
                "status": status,
            }
        )
    print(_format_check_table(table_rows))
    print()
    if failed:
        print(
            f"FAIL: one or more scenarios exceeded time_tolerance={time_tolerance} "
            f"or quality_rtol={quality_rtol}"
        )
        return 1
    print("PASS: all scenarios within tolerance")
    return 0


def _run_regenerate(baseline_path: Path, repeats: int, destination: Path | None) -> int:
    """Re-measure exactly the cells a baseline file records and rewrite it.

    The cell list in ``benchmarks/baselines.json`` is a curated contract, not a
    matrix product: it is chosen to cover every solver while staying cheap
    enough for a shared CI runner. Regeneration therefore replays that recorded
    list instead of rebuilding a matrix from flags, so an intentional timing
    refresh never silently changes which scenarios CI checks.
    """
    document = json.loads(baseline_path.read_text())
    outcomes = [_run_from_json(record, repeats) for record in document["runs"]]
    print(_format_table(outcomes))
    refreshed: dict[str, JsonValue] = {
        "description": document.get("description"),
        "environment": _environment(),
        "runs": [outcome.to_json() for outcome in outcomes],
    }
    target = baseline_path if destination is None else destination
    target.write_text(json.dumps(refreshed, indent=2) + "\n")
    print(f"\nwrote {target}")
    return 0


def _parse_int_list(raw: str) -> list[int]:
    return [int(token) for token in raw.split(",") if token.strip()]


def _parse_scenario_list(raw: str) -> list[str]:
    names = [token.strip() for token in raw.split(",") if token.strip()]
    for name in names:
        if name not in _SCENARIO_NAMES:
            raise argparse.ArgumentTypeError(
                f"unknown scenario {name!r}; choose from {', '.join(_SCENARIO_NAMES)}"
            )
    return names


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=str, default=_DEFAULT_ROWS)
    parser.add_argument("--dims", type=int, default=_DEFAULT_DIMS)
    parser.add_argument("--bins", type=str, default=_DEFAULT_BINS)
    parser.add_argument("--scenarios", type=str, default=",".join(_SCENARIO_NAMES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--check", type=Path, default=None)
    parser.add_argument("--regenerate", type=Path, default=None)
    parser.add_argument("--time-tolerance", type=float, default=_DEFAULT_TIME_TOLERANCE)
    parser.add_argument("--quality-rtol", type=float, default=_DEFAULT_QUALITY_RTOL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark harness in matrix or ``--check`` regression mode."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.check is not None:
        return _run_check(args.check, args.repeats, args.time_tolerance, args.quality_rtol)

    if args.regenerate is not None:
        return _run_regenerate(args.regenerate, args.repeats, args.json)

    rows_list = _parse_int_list(args.rows)
    bins_list = _parse_int_list(args.bins)
    scenarios = _parse_scenario_list(args.scenarios)
    matrix = _build_matrix(rows_list, args.dims, bins_list, scenarios, args.seed, args.max_scans)
    outcomes = _run_matrix(matrix, args.repeats)
    print(_format_table(outcomes))

    if args.json is not None:
        document: dict[str, JsonValue] = {
            "environment": _environment(),
            "runs": [outcome.to_json() for outcome in outcomes],
        }
        args.json.write_text(json.dumps(document, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
