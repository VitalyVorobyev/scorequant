# System design

## Public task boundary

```text
scores ---------------------------------> optimize_partition() -> PartitionResult

ScoreSample ----------------------------+
ObservationSample + ScoreProvider ------+-> fit_quantizer() ----> QuantizerResult
IntegrationSource + ScoreProvider ------+
```

The provider side is layered: a score provider consumes one of three statistical
representations — exact densities (`LinearComponentScore`), model density ratios
(`DensityRatioScore`, `CentralLogRatioScore`), or scores directly (`ScoreFunction`). Density
ratios are the minimal sufficient representation when absolute densities are unavailable; the
optimizers themselves consume only score rows ([ADR 0001](decisions.md),
[ADR 0017](decisions.md)). Importance ratios for reweighted samples
are source weights, never provider inputs.

`PartitionResult` owns one fixed assignment: labels, cell weights/moments/means, full and retained
information, objective, rank diagnostics, accepted moves, exchange stability, remaining gain, and
provenance. It has no prediction method. `compile_quantizer()` is available only for a stable,
nonsingular D result whose geometry certificate is Voronoi-consistent at the tolerance the
partition was optimized at ([ADR 0016](decisions.md)):
the compiled rule reproduces every training label except boundary rows whose relocation gain sits
inside that tolerance.

`QuantizerResult` owns score-space centers and metric, `predict_scores`, train/validation reports,
hardening gap, trace, criterion/configuration, source kind, and provenance. There is deliberately
no ambiguous `predict` method.

## Criterion and configuration pairs

`api.py` validates every `(config, criterion, task)` combination against one declarative table
instead of scattered isinstance chains. The closed table is [ADR 0011 / 0014](decisions.md); this
is what each supported pair computes.

| Task | Criterion | Configuration | Meaning |
| --- | --- | --- | --- |
| finite assignment | `DOptimality` | `DExchangeConfig` | exact positive-gain relocation |
| finite assignment | `ProfiledDOptimality` | `DExchangeConfig` | exact positive-gain relocation of the same-label profiled objective |
| finite assignment | `DOptimality` | `MahalanobisLloydConfig` | guarded batch Mahalanobis-Lloyd, exact-objective verified per proposal |
| finite assignment | `ProfiledDOptimality` | `MahalanobisLloydConfig` | guarded batch Mahalanobis-Lloyd of the profiled objective |
| reusable quantizer | `DOptimality` | `DExchangeConfig` | finite D assignment followed by explicit verified compilation |
| reusable quantizer | `DOptimality` | `MahalanobisLloydConfig` | guarded batch Lloyd assignment followed by explicit verified compilation |
| reusable quantizer | `DOptimality` | `SoftVoronoiConfig` | direct differentiable soft-D fit and hardening |
| reusable quantizer | `ProfiledDOptimality` | `SoftVoronoiConfig` | direct differentiable soft profiled-D fit and hardening |
| reusable quantizer | `DOptimality` | `ScalarDPConfig` | exact interval dynamic program on one retained score dimension |
| reusable quantizer | `NormalizedTrace` | `KMeansConfig` | Fisher-whitened weighted k-means baseline |

A config type absent from a task's signature raises `TypeError`; a config/criterion pair the task
does not implement raises `ValueError`, before optimization. A profiled finite partition never
compiles into a quantizer: no same-label profiled rule is canonical away from the training rows.

## Execution architecture

The browser runtime is the approved second-runtime use case from [ADR 0018](decisions.md).
Dependency direction:

```text
domain contracts/config/results (canonical NumPy arrays)
        ↓
private execution protocol + JAX/NumPy adapters
        ↓
shared Fisher/geometry/objective kernels
        ↓
solver orchestration and stopping rules
        ↓
public task API
```

Import-boundary tests enforce that domain, execution, kernels, solvers, public API,
visualization, and `website/` cannot reverse this direction; no direct JAX/Optax import exists
outside `_execution.py`, and no backend tensor appears in a public result. One backend-parameterized
conformance suite runs every declared solver family on both backends, compares partitions up to
relabeling, checks the shared analytic soft gradient against autodiff and finite differences, and
imports the package in a subprocess that blocks JAX. Optimized backend kernels are permitted only
behind an adapter and must match the shared reference implementation.

## Module ownership

- `information.py`: Fisher and retained-information algebra, including `retention_uncertainty`.
- `transforms.py`: informative subspace and whitening.
- `partition.py`: the unified exchange engine (D and profiled-\(D_s\) share one determinant-lemma
  scan through a private `_ExchangeObjective` protocol), the guarded batch Mahalanobis-Lloyd solver,
  the standalone stability certificate, and geometry diagnostics.
- `certify.py`: explicit bounded branch-and-bound global certification of D partitions. Its tree
  search is sequential NumPy in float64: per-node JAX dispatch would dominate a search whose nodes
  are one small log determinant each.
- `_execution.py`: the only backend resolver and the private JAX/NumPy primitive adapters.
- `solvers/common.py`: shared assignment, distance, trace, and solver result contracts;
  `solvers/kmeans.py`, `solvers/scalar.py`, and `solvers/soft.py`: responsibility-specific shared
  solver orchestration. `quantizers.py` is a private re-export façade kept for established test
  seams; it is scheduled for removal at the v1.0 API audit (roadmap phase H).
- `sources.py`: empirical and quadrature measures plus provenance (`ScoreProvenance` and the
  nested `RatioProvenance`).
- `providers.py`: framework-neutral observation-to-score adapters, including the ratio-backed
  `DensityRatioScore` and `CentralLogRatioScore`.
- `ratios.py`: density-ratio algebra — posterior-to-ratio prior correction, ratio-to-score maps
  for the mixture and intensity parameterizations, and the ratio-closure diagnostic.
- `components.py`: linear models and the intensity score adapter.
- `reports.py`: diagnostic and certificate dataclasses (`InformationReport`,
  `ProfiledInformationReport`, `GeometryReport`, `ProfiledGeometryReport`, `StabilityReport`,
  `PartitionCertificate`, `EfficientScoreBound`, `RetentionUncertainty`). It depends on nothing
  that depends back on it, which is what lets `result.py` and `information.py` both build on it
  without importing each other.
- `criteria.py`, `config.py`, `result.py`: backend-free public contracts. `api.py`: public
  orchestration. `artifact.py`: the versioned `Quantizer` rule and its non-pickle archive.
  `_predict.py`: the leaf prediction kernel.
- `_binstats.py`, `_chunking.py`, `_validation.py`, `_json.py`, `_typing.py`: private helpers shared
  across modules (weighted per-bin scatter-add statistics, the shared memory-bounded row-chunking
  budget used by exchange scans and assignment kernels, input validation including dtype promotion,
  `to_dict()` JSON conversion, and the shared `ArrayLike`/`JsonValue` type aliases).
- `visualization.py`: optional Matplotlib views over `PartitionResult`/`QuantizerResult`, imported
  lazily so the core package carries no hard visualization dependency.
- `examples/`, `tests/`, `benchmarks/`, `agenticresearch/`: datasets, tuning, counterexample
  search, and application logic. Research exploration is provenance, excluded from the product
  Ruff gate; every relied-upon identity or counterexample is copied into a deterministic
  regression test.

JAX is the default execution backend and Optax supplies its soft-optimizer updates. NumPy is the
portable CPU backend and supplies every declared solver, including an analytic-gradient soft
optimizer with a private Adam state. Mathematical kernels are shared. Optional visualization and
both execution stacks remain lazy at their public boundaries.

## Learning and reference sites

MkDocs at `/docs/` is the exhaustive Python, developer and decision reference; `website/` is an
isolated Docusaurus portal at the site root owning Get started, the walkthroughs, the Research
Atlas and the browser runtime behind walkthrough experiments ([ADR 0031 / 0035](decisions.md)).
Source adapters read canonical Markdown, Griffe API data, benchmark JSON and the registry export.
Browser schemas, workers, plotting and marimo embeds never enter `src/scorequant`.

## Source/provider rules

A score callback without a source has no measure and is rejected. A `ScoreSample` already contains
scores, so supplying a provider with it is also rejected. Observation and integration sources
require a provider. Equivalent source/provider constructions must materialize the same core
result.

`ScoreProvenance.exact_fisher` is derived from provenance kind; an estimated ratio cannot set it
independently. The ratio boundary stores only a ready callback, an explicit declared
parameterization, and structured ratio provenance (estimator, training priors, calibration,
reference component, finite-difference offsets). Estimation frameworks remain outside
dependencies and application splits remain visible. Model density ratios and importance ratios
never share an argument: the former enter through providers, the latter through source weights.

## Complexity and durability

The exact exchange scan is \(O(NKP^2)\) per accepted move and avoids \(O(N^2)\) storage; the
exact rank-one interval DP is the one intentional quadratic exception, capped by
`ScalarDPConfig.max_rows`. Geometric solvers materialize \([N,K]\) distances. Histories store
aggregate metrics and center snapshots, never per-event responsibilities. `to_dict()` is JSON-ready
diagnostic state, not a versioned persistence format; that role belongs to `Quantizer.save`, which
writes a versioned non-pickle artifact holding the rule alone ([ADR 0023](decisions.md),
[ADR 0040](decisions.md)).
