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

## Stable first-wave combinations

`api.py` validates every `(config, criterion)` pair against one declarative table instead of
scattered isinstance chains; this is the complete current matrix.

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

Unsupported criterion/configuration pairs fail before optimization: a config type absent from a
task's own signature raises `TypeError`, and a config/criterion pair the task does not implement
raises `ValueError`. There is no generic criterion plugin until multiple implementations demonstrate
a stable common contract. A profiled finite partition never compiles into a quantizer: no same-label
profiled rule is canonical away from the training rows. See
[ADR 0014](decisions.md).

## Execution architecture and quality audit

The browser lab is the approved second-runtime use case from
[ADR 0018](decisions.md). The target dependency direction is:

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

The pre-refactor audit found five structural liabilities that are now explicit gates:

| Finding | Risk | Required resolution |
| --- | --- | --- |
| JAX imports in sources, reports, results, JSON conversion, and providers | backend types leak through stable contracts and importing without JAX fails | domain modules use NumPy only; execution imports stay private |
| partition and quantizer modules combine dispatch, geometry, exchange, Lloyd, k-means, DP, soft optimization, and diagnostics | oversized functions accumulate hidden coupling | split by stable responsibility after the backend seam is green |
| scatter, random, JIT, and optimizer choices are embedded in equations | NumPy parity would require conditionals or copied mathematics | adapters own primitives; kernels and solver flow contain no backend-name branches |
| result/config/report state is constructed in multiple paths | serialization and backend provenance can drift | each public state type has one definition and canonicalization path |
| JAX-only tests are organized by implementation module | a second suite would duplicate coverage | parameterize one capability-driven conformance suite over backends |

Import-boundary tests enforce that domain, execution, kernels, solvers, public API,
visualization, and `website/` cannot reverse the declared direction. Architecture review is an
exit gate after the JAX extraction and again after complete NumPy parity. Optimized backend
kernels are permitted only behind an adapter and must match the shared reference implementation.

Both reviews are now recorded. The extraction review found no direct JAX/Optax import outside
`_execution.py`, no backend tensor in recursively inspected public results, and no frontend concern
inside `src/`. The parity review runs every declared partition and quantizer solver family through
one backend-parameterized matrix and compares their partitions up to bin relabeling, checks the
shared analytic soft gradient against JAX autodiff and finite differences, holds the default JAX
path to the committed benchmark quality baselines, and imports the package in a subprocess that
actively blocks JAX and Optax -- with a companion test asserting that the blocker really blocks, so
the claim cannot pass vacuously.

## Module ownership

- `information.py`: Fisher and retained-information algebra.
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
  solver orchestration. `quantizers.py` is now only a thin compatibility façade for established
  private test seams.
- `sources.py`: empirical and quadrature measures plus provenance (`ScoreProvenance` and the
  nested `RatioProvenance`).
- `providers.py`: framework-neutral observation-to-score adapters, including the ratio-backed
  `DensityRatioScore` and `CentralLogRatioScore`.
- `ratios.py`: density-ratio algebra — posterior-to-ratio prior correction, ratio-to-score maps
  for the mixture and intensity parameterizations, and the ratio-closure diagnostic.
- `components.py`: linear models and the intensity score adapter.
- `reports.py`: diagnostic and certificate dataclasses (`InformationReport`,
  `ProfiledInformationReport`, `GeometryReport`, `ProfiledGeometryReport`, `StabilityReport`,
  `PartitionCertificate`, `EfficientScoreBound`). It depends on nothing that depends back on it,
  which is what lets `result.py` and `information.py` both build on it without importing each other.
- `criteria.py`, `config.py`, `result.py`: backend-free public contracts. `api.py`: public
  orchestration. `api.py`
  validates every `(config, criterion, task)` combination against one declarative table instead of
  scattered isinstance chains.
- `_binstats.py`, `_chunking.py`, `_validation.py`, `_json.py`, `_typing.py`: private helpers shared
  across modules (weighted per-bin scatter-add statistics, the shared memory-bounded row-chunking
  budget used by exchange scans and assignment kernels, input validation including dtype promotion,
  `to_dict()` JSON conversion, and the shared `ArrayLike`/`JsonValue` type aliases).
- `visualization.py`: optional Matplotlib views over `PartitionResult`/`QuantizerResult`, imported
  lazily so the core package carries no hard visualization dependency.
- `examples/`, tests, and `research/`: datasets, tuning, counterexample search, and application logic.

JAX is the default execution backend and Optax supplies its soft-optimizer updates. NumPy is the
portable CPU backend and supplies every declared solver, including an analytic-gradient soft
optimizer with a private Adam state. Mathematical kernels are shared. Optional visualization and
both execution stacks remain lazy at their public boundaries. Research exploration is provenance
and is excluded from the product Ruff gate; every relied-upon identity or counterexample is copied
into a deterministic regression test.

## Learning and reference sites

Per [ADR 0019](decisions.md), MkDocs remains the exhaustive Python,
developer, and ADR reference. `website/` is an isolated Docusaurus/React learning portal owning
curated journeys, theory reading, examples, benchmark exploration, public research storytelling,
and the browser Lab. Source adapters read canonical Markdown, Griffe API data, benchmark JSON, and
an explicit research-publication allowlist. Browser schemas, workers, plotting, and marimo embeds
never enter `src/scorequant`.

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

The current exact exchange scan is \(O(NKP^2)\) per accepted move and avoids \(O(N^2)\) storage.
Geometric solvers materialize \([N,K]\) distances. Histories store aggregate metrics and center
snapshots, never per-event responsibilities. `to_dict()` is JSON-ready diagnostic state, not a
versioned persistence format; that role belongs to `Quantizer.save`, which writes a versioned
non-pickle artifact holding the rule alone.

## Pre-1.0 API audit

The current two-function boundary is sound: it prevents a fixed labeling from masquerading as a
rule and keeps observation-to-score conversion visible. The main weaknesses are capability gaps,
not a need for a generic facade:

| Need | Incorrect shortcut | Chosen contract | Status |
| --- | --- | --- | --- |
| same-label nuisance profiling | compile finite labels with an efficient metric | explicit `ProfiledDOptimality`; finite and inductive solvers remain separate | implemented |
| certified profiled ceiling | trust profiled exchange with no upper bound | `efficient_score_bound`/`EfficientScoreBound`, an exact scalar-DP ceiling on the profiled objective | implemented |
| global guarantee | imply exchange stability is global | explicit bounded branch-and-bound certificate | implemented as `certify_partition`/`PartitionCertificate` (D-only) |
| local stability of any labeling | trust a solver's own termination claim | one exact scan via `exchange_stability_report`/`StabilityReport` | implemented |
| estimated density ratios | treat the classifier as the abstraction | named ratio representation: `ratios_from_posteriors`/`mixture_scores_from_ratios`, decomposed providers, `ratio_closure_report`, structured `RatioProvenance` | implemented |
| Voronoi self-consistency of a D result | assume exchange stability implies Voronoi geometry | measured `GeometryReport`/`ProfiledGeometryReport`, judged at the solver's own `gain_tolerance` | implemented |
| Monte Carlo population law | pass an unrecorded callback as a score table | deterministic score/observation sampler source | not yet implemented |
| analytic cell integrals | pretend an oracle contains rows | moment-oracle evaluation of an existing rule | not yet implemented |
| large transported data | call minibatch fitting exact | streaming aggregation for a frozen rule | not yet implemented |
| reuse across processes | treat `to_dict()` as a schema | versioned non-pickle quantizer artifact | implemented as `Quantizer.save`/`Quantizer.load`, a zip of `manifest.json` plus `allow_pickle=False` arrays; loads and predicts with no JAX present |

The revision deliberately does not add `predict`, a generic criterion plugin, classifier training,
or a universal streaming optimizer. See [ADR 0013](decisions.md) and
[ADR 0014](decisions.md) for the complete decisions.
