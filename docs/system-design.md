# System design

## Design principles

These are the decisions that constrain the code. Everything else is implementation.

1. **Scores are the contract.** Optimizers take weighted score rows `scores [N, P]` with optional
   `weights [N]`. Densities, density ratios, analytic likelihoods, autodiff and learned estimators
   are adapters that produce scores; they never reach an optimizer directly. Scores are never
   centred, because the origin of score space means something.
2. **Two tasks, two results.** `optimize_partition` labels a fixed weighted sample and has no
   predictor. `fit_quantizer` returns a reusable rule that predicts only from explicit scores
   (`predict_scores`). A finite partition becomes a rule only through `compile_quantizer`, and only
   when the D geometry theorem guarantees the rule reproduces its labels.
3. **The rule is an artifact.** `Quantizer` holds the transform, centres, metric, names and
   provenance and nothing about the training sample. It saves to a versioned, non-pickle archive
   that loads and predicts without JAX on any machine; every 1.x release reads format 1.
4. **Measures and score maps are separate.** A source (`ScoreSample`, `ObservationSample`,
   `IntegrationSource`) is the reference measure; a provider maps observations to scores. Model
   density ratios enter through providers, importance ratios through source weights, and the two
   never share an argument.
5. **Exact and surrogate information are different quantities.** Provenance decides whether
   retained information is exact Fisher information or a supplied-score surrogate; estimated scores
   or ratios can never claim exact semantics.
6. **A closed table of criteria and solvers.** Each `(config, criterion, task)` pair is declared
   once and validated before optimization; there is no plugin system. Certificates are explicit
   calls, use the tolerance the result was optimized at, and never run implicitly.
7. **One mathematical core, two backends.** JAX is the default and NumPy the portable runtime
   (including the browser). Equations and solver flow are shared; backend differences live in one
   private adapter module and one conformance suite. Public arrays are always NumPy, and the
   library never changes global JAX configuration.
8. **Refuse rather than guess.** When a guarantee fails on valid data, the library raises
   `RefusalError` naming the exact counterexample that forces the refusal; a malformed call raises
   `ContractError`. Both derive from `ScoreQuantError`.
9. **Versioning.** Semantic versioning; before 1.0 a minor release may break the API and says so
   under Removed in `CHANGELOG.md`. From 1.0 the inventory in `docs/api.md` is the stable contract;
   reports and `to_dict()` keys are diagnostics that may gain fields but not lose them.

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
optimizers themselves consume only score rows. Importance ratios for reweighted samples
are source weights, never provider inputs.

`PartitionResult` owns one fixed assignment: labels, cell weights/moments/means, full and retained
information, objective, rank diagnostics, accepted moves, exchange stability, remaining gain, and
provenance. It has no prediction method. `compile_quantizer()` is available only for a stable,
nonsingular D result whose geometry certificate is Voronoi-consistent at the tolerance the
partition was optimized at:
the compiled rule reproduces every training label except boundary rows whose relocation gain sits
inside that tolerance.

`QuantizerResult` owns score-space centers and metric, `predict_scores`, train/validation reports,
hardening gap, trace, criterion/configuration, source kind, and provenance. There is deliberately
no ambiguous `predict` method.

## Criterion and configuration pairs

`api.py` validates every `(config, criterion, task)` combination against one declarative table
instead of scattered isinstance chains. This is what each supported pair computes.

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
  solver orchestration.
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
- `examples/`, `tests/`, `benchmarks/`: datasets, tuning and application logic. Every
  relied-upon identity or counterexample is a deterministic regression test.

JAX is the default execution backend and Optax supplies its soft-optimizer updates. NumPy is the
portable CPU backend and supplies every declared solver, including an analytic-gradient soft
optimizer with a private Adam state. Mathematical kernels are shared. Optional visualization and
both execution stacks remain lazy at their public boundaries.

## Learning and reference sites

`website/` is the Docusaurus portal at the site root: the landing page, get-started, the
walkthroughs and the browser runtime that powers their live demos. MkDocs at `/docs/` is the
reference: API, user guide and glossary. Browser schemas, workers and plotting never enter
`src/scorequant`.

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
writes a versioned non-pickle artifact holding the rule alone.
