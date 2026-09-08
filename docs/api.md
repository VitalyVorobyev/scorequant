# API guide

Almost every use of ScoreQuant is built from a dozen names. The rest of the surface is machinery
you reach for when you have a specific question about a result.

**Core.** The two tasks, `optimize_partition` and `fit_quantizer`; the inputs `ScoreSample`,
`ObservationSample`, `IntegrationSource` and `ScoreSchema`; the objectives `DOptimality`,
`ProfiledDOptimality` and `NormalizedTrace`; a solver configuration; a provider such as
`ScoreFunction`, `LinearComponentScore` or `DensityRatioScore`; and `Quantizer`, the rule you
deploy.

**Advanced.** Certificates and bounds (`certify_partition`, `exchange_stability_report`,
`efficient_score_bound`), the information algebra (`fisher_information`,
`binned_fisher_information`, `information_report`, `profiled_information_report`), the held-out
error bar (`retention_uncertainty`), the ratio algebra (`ratios_from_posteriors`,
`mixture_scores_from_ratios`, `ratio_closure_report`), `FisherTransform`, the report types, and the
plotting helpers. Each answers a question you will know you have; none is needed to fit anything.

The sections below follow that order. The generated [reference](symbols/index.md) covers every
public object.

## Which task, which criterion, which solver

<!-- generated: solver-matrix (do not edit by hand; run `pnpm generate:data`) -->
| Configuration | `optimize_partition` | `fit_quantizer` | Contract |
| --- | --- | --- | --- |
| `DExchangeConfig` | `DOptimality`, `ProfiledDOptimality` | `DOptimality` | Exact positive-gain relocations; monotone objective; terminates exchange-stable |
| `MahalanobisLloydConfig` | `DOptimality`, `ProfiledDOptimality` | `DOptimality` | A batch is adopted only if the exactly rebuilt objective improves; optional exact-exchange guard |
| `KMeansConfig` | — | `NormalizedTrace` | Weighted $k$-means in whitened score space |
| `SoftVoronoiConfig` | — | `DOptimality`, `ProfiledDOptimality` | Differentiable soft optimization then hardening, with the hardening gap reported |
| `ScalarDPConfig` | — | `DOptimality` | The exact global interval solution for rank-one score space |
<!-- /generated: solver-matrix -->

A pair that is absent from this table is refused before any optimization starts. Read a result in
this order: the information kind it reports (exact model or supplied-score surrogate), the sample
it was evaluated on, the retained rank, the hard D-efficiency, the worst-direction diagnostics,
then the convergence or certificate status.

---

# Core

## Top-level tasks

### `optimize_partition`

<!-- snippet: skip -->
```python
optimize_partition(
    scores,                       # ScoreSample, or a raw score array
    *,
    weights=None,
    n_bins,
    criterion=None,
    config=None,
    provenance=None,
    initial_labels=None,
    execution=None,
) -> PartitionResult
```

This is fixed-sample assignment. `scores` is either a `ScoreSample` — the same weighted score law
`fit_quantizer` takes, carrying its own weights, schema and provenance — or a raw score array with
`weights` and `provenance` supplied separately. Passing a sample together with either keyword is
rejected rather than silently resolved. An observation source is deliberately not accepted:
converting observations to scores stays an explicit `provider.score(X)`, so the fixed-sample
boundary remains visible. It accepts `DOptimality` or `ProfiledDOptimality` with either
`DExchangeConfig` (exact positive-gain relocation) or `MahalanobisLloydConfig` (guarded
nearest-centroid batches). A batch is adopted only when the exactly rebuilt objective strictly
improves, because the frozen-metric batch step is not monotone on its own; with the default
`guard="exchange"` the labels are then finished by the exchange engine, so the terminal state is
exchange-stable.

`initial_labels` starts the solver from a supplied `[N]` labeling instead of its own seeding, which
is how `efficient_score_bound(...).labels` is used as an initializer. Zero-weight rows carry no
measure and their labels are ignored, identical score rows are merged before the solver runs and
must therefore already agree on their bin, and every requested cell must remain nonempty
afterwards. Supplied labels replace the seeding of the first exchange restart only, so `init` and
`initializer_restarts` still govern any further restart and one call can compare both starts; the guarded
Mahalanobis-Lloyd solver starts from them directly.

### `fit_quantizer`

<!-- snippet: skip -->
```python
fit_quantizer(
    source,
    *,
    provider=None,
    validation=None,
    n_bins,
    criterion=None,
    config=None,
    diagnostics="endpoints",
    execution=None,
) -> QuantizerResult
```

Supported pairs are D exchange, guarded Mahalanobis-Lloyd, soft D, normalized-trace k-means, and
exact scalar interval dynamic programming. The two finite D solvers take the same route: optimize
the labels, then compile the verified rule. `ScoreSample` forbids a provider; observation and
integration sources require one. Validation must use the same score dimension — and, when both
sides declare a `ScoreSchema`, the same parameter names, since a reordering is invisible to a
column count — and remains diagnostic.

`diagnostics` controls how many recorded center snapshots are re-scored into
`trace.train_hard_retention` and `trace.validation_hard_retention`: `"final"` scores only the
terminal snapshot, `"endpoints"` (the default) scores the first and terminal snapshots, and
`"full"` scores every snapshot, matching the historical behavior. Unscored snapshots hold `nan` so
the returned history stays aligned with `trace.steps`; `centers`, `labels`, and both reports are
unaffected.

`ScalarDPConfig` pairs with `DOptimality` only and requires the effective score space to be rank
one after `rank_rtol` projection; a higher rank is rejected by name. On that rank the D-optimal
partition has ordered interval cells, so the weighted interval dynamic program returns the global
optimum rather than a local one, and `max_rows` bounds its exact quadratic work.

## Naming the score coordinates

A score matrix is a table of partial derivatives, one column per model parameter. The order is
meaningful but invisible, which is bearable in a two-parameter toy and dangerous in a real problem
with dozens of components. `ScoreSchema` declares the names:

<!-- snippet: skip -->
```python
schema = sq.ScoreSchema(("T cells", "B cells", "monocytes", "mast cells", "HSPCs"))

sample = sq.ScoreSample(scores, weights, schema=schema)

criterion = sq.ProfiledDOptimality(interest=("HSPCs",))
```

Names are resolved to score columns exactly once, at the public task boundary, so every solver,
report and certificate downstream still works in indices. `ProfiledDOptimality` accepts either
form and refuses a mixture of the two; declaring names against a sample that carries no schema
fails by name rather than guessing. Reports then print `interest: HSPCs` instead of `(4,)`.

A schema answers only *what each coordinate means*. Where the numbers came from and at which
reference point stays with `ScoreProvenance`, and the two are validated against each other rather
than each carrying a reference point that can drift.

Providers may supply the schema for the observation-space routes. `LinearComponentScore` derives
it from the component names the model already declares; `ScoreFunction`, `DensityRatioScore` and
`CentralLogRatioScore` accept `schema=` explicitly. It then reaches `PartitionResult.schema` and
`QuantizerResult.schema`.

## The score-provider contract

`ScoreProvider` is a runtime-checkable protocol, not a closed union. Any object with a
`provenance` and a `score` method is a provider:

<!-- snippet: skip -->
```python
class MyExternalScore:
    provenance = sq.ScoreProvenance(kind="estimated_ratio")

    def score(self, observations):
        return my_package.evaluate(observations)
```

`score` takes observations alone; the execution backend is ambient context established by the
public task, so an external implementation needs no knowledge of `ExecutionConfig`. A provider may
also expose `schema`, which is used when present and is not part of the required contract. The
four built-in providers are convenience implementations of this protocol.

## Sources, providers, and density ratios

Sources carry the measure (`ScoreSample`, `ObservationSample`, `IntegrationSource`); providers
carry the observation-to-score map (`ScoreFunction`, `LinearComponentScore`, `DensityRatioScore`,
`CentralLogRatioScore`). Importance ratios for a sample drawn from a proposal distribution belong
in the source weights, never in a provider.

`DensityRatioScore(ratio, parameterization, *, provenance=None)` pairs a model density-ratio
callback `[N, D] -> [N, K]` with a declared ratio-to-score map: `MixtureParameterization` for a
normalized mixture with one simplex-dependent component (`K - 1` score columns) or
`IntensityParameterization` for an extended linear-intensity model (all `K` columns).
`DensityRatioScore.from_classifier(predict, class_priors, parameterization, ...)` builds the ratio
callback from calibrated posteriors as `ratios_from_posteriors(predict(X), class_priors)` and
always records estimated provenance. `CentralLogRatioScore(predict, deltas, class_priors)` turns
paired minus/plus probabilities from classifiers trained at \(\theta_0\pm\delta\) into central
finite-difference scores.

The underlying array algebra is public: `ratios_from_posteriors(posteriors, class_priors)`,
`mixture_scores_from_ratios(ratios, reference_fractions, reference_component=-1)`, and
`scores_from_components(components, coefficients)` — the latter doubles as the intensity
ratio-to-score map because it is invariant under a common event-wise rescaling of a row.
`ratio_closure_report(ratios, weights)` checks that each ratio column integrates to one under the
declared measure and returns the per-component normalizers with their largest absolute residual.

`ScoreProvenance(kind, ...)` accepts `"unknown"`, `"exact"`, `"autodiff"`, `"estimated_ratio"`,
and `"custom_estimated"`; `exact_fisher` is derived from `kind` and cannot be set independently.
Ratio-derived scores carry a structured `ScoreProvenance.ratio` record (`RatioProvenance`) naming
the estimator, parameterization, reference fractions or coefficients, reference component,
training priors, calibration method, and finite-difference offsets.

## The deployable rule

A fit produces two different things and they now have two different types. `QuantizerResult` is
the record of the fit — labels, reports, trace, solver diagnostics. `result.quantizer` is the rule
itself:

<!-- snippet: skip -->
```python
fit = sq.fit_quantizer(sample, n_bins=8)

fit.train_report  # what the fit measured
fit.trace  # how it got there

rule = fit.quantizer  # what you deploy
rule.predict_scores(future_scores)
rule.save("flowcyt-8bins.sqz")
```

`Quantizer` carries the transform, centers, optional metric, score schema, provenance and
criterion — and nothing about the training sample. `PartitionResult.compile_quantizer()` returns
one too, under exactly the same theorem-backed conditions as before: compilation is bookkeeping on
an exchange-stable D partition, not a second fit, so it hands back a rule rather than something
that impersonates a new training result.

`QuantizerResult.predict_scores` remains the one prediction verb and delegates to the rule;
`centers`, `metric`, `transform` and `schema` read through to it.

### The artifact format

`Quantizer.save` writes a zip archive holding one `manifest.json` and one `.npy` member per array.

- **Versioned.** The manifest declares a `format_version`, and a reader refuses a version it does
  not know by name rather than interpreting a field it has never seen.
- **Not a pickle.** Arrays are written with `allow_pickle=False`, so loading an artifact executes
  no code from the file.
- **Backend-free.** A rule fitted on the JAX backend loads and predicts in a process where JAX is
  not importable, which is the other half of the NumPy backend: fit offline, deploy where the
  fitting stack is not installed.

<!-- snippet: skip -->
```python
rule = sq.Quantizer.load("flowcyt-8bins.sqz")
bins = rule.predict_scores(scores, execution=sq.ExecutionConfig(backend="numpy"))
```

This is a durable format, unlike `to_dict()` on the result types, which stays JSON-ready
diagnostic state.

## Execution

Every public entry point takes `execution=`, an `ExecutionConfig`, and the default is JAX. The
configuration selects the numerical runtime only; it never changes what is being optimized, and
the package never touches global JAX configuration on your behalf (enabling 64-bit arithmetic is
an explicit choice you make in the application, typically through `JAX_ENABLE_X64=1`).

<!-- snippet: skip -->
```python
sq.ExecutionConfig(backend="jax", precision="preserve", device="default")
```

- **`backend`** is `"jax"` (default) or `"numpy"`. The NumPy backend is the portable CPU runtime:
  it runs the same shared mathematics through the same solvers, which is what lets a rule fitted
  here predict where JAX is not installed, including in the browser Lab. It is not a second solver
  tree, so the two backends agree to floating-point tolerance on the same inputs and seeds, and
  the conformance suite holds them to that.
- **`precision`** is `"preserve"` (default: float32 and float64 inputs keep their dtype, integer
  and lower-precision inputs are promoted to float32), `"float32"`, or `"float64"`. A float64
  request on JAX still needs 64-bit mode enabled in the application.
- **`device`** is `"default"`, `"cpu"`, `"gpu"`, or `"tpu"`. NumPy accepts only the default CPU
  device; JAX resolves the first available device of the requested family.

Results remember the execution they were fitted under: `PartitionResult.execution` and
`QuantizerResult.execution` record it, `to_dict()` serializes it, and every follow-up computation
on the result, `predict_scores` included, reuses it unless you pass an override.

<!-- snippet: skip -->
```python
fit = sq.fit_quantizer(sample, n_bins=8, execution=sq.ExecutionConfig(backend="jax"))
fit.predict_scores(scores)  # runs on the recorded JAX execution
fit.predict_scores(scores, execution=sq.ExecutionConfig(backend="numpy"))  # override per call
```

A saved `Quantizer` is backend-free: the artifact stores arrays, not an execution, so `load`
followed by `predict_scores(..., execution=...)` chooses the runtime at prediction time. Public
arrays are always `numpy.ndarray` whichever backend produced them.

## Errors

Every exception the library raises deliberately is a `scorequant.ScoreQuantError`. A
`ContractError` (also a `ValueError`) means change the call: the arguments alone are enough to
see the problem. A `RefusalError` means the library declined a valid request because a
theorem-backed condition fails on your data, not on the call, and `error.counterexample` names
the registry entry that forces the refusal.

<!-- snippet: skip -->
```python
try:
    fit.compile_quantizer()
except sq.RefusalError as error:
    print(error, error.counterexample)
except sq.ContractError as error:
    raise  # the call itself needs to change
```

`compile_quantizer()` and a degenerate profiled `fit_quantizer` raise `RefusalError`, which is
not a `ValueError`.

### Budgets

| Parameter | Class | Default | Meaning |
|---|---|---|---|
| `max_iter` | `KMeansConfig` | 100 | Lloyd iterations per restart |
| `kmeans_max_iter` | `SoftVoronoiConfig` | 100 | Lloyd iterations per initialization restart |
| `max_steps` | `SoftVoronoiConfig` | 1000 | Adam updates |
| `max_scans` | `DExchangeConfig` | `None` (run to stability) | complete candidate scans |
| `max_iter` | `MahalanobisLloydConfig` | 100 | guarded batch iterations |
| `max_nodes` | `CertificationConfig` | 2 000 000 | branch-and-bound nodes before `budget_exhausted` |

These govern how long a solver may run. A separate set of parameters governs how big a problem it
may accept: `solver_restarts` (8 on `KMeansConfig`, 1 on `DExchangeConfig`) and
`initializer_restarts` (8 on the soft, exchange, and Lloyd configs) control how many restarts a
solver tries, while `ScalarDPConfig.max_rows` (20 000) and `CertificationConfig.max_rows` (64,
ceiling 512) cap how many rows a solver will accept before refusing outright.

---

# Advanced

## The efficient-score upper bound

<!-- snippet: skip -->
```python
efficient_score_bound(
    scores,
    *,
    interest,
    weights=None,
    n_bins,
    config=None,
) -> EfficientScoreBound
```

Efficient-score domination bounds the profiled information of every `n_bins`-cell rule of the full
score space by the between-cell information of the full-data efficient score under that rule. For
one parameter of interest the maximizing rule has ordered interval cells and is found exactly, so
`upper_bound` is a certificate rather than an estimate, reported in the same log-determinant
convention as `PartitionResult.objective` under `ProfiledDOptimality`. `gap_to(partition_result)`
returns the remaining slack and is nonnegative up to floating-point error; `labels` doubles as the
`initial_labels` initializer for profiled exchange. More than one interest column raises
`NotImplementedError`, because a multivariate efficient score would need a multivariate solver and
the result would no longer be certified.

## Held-out retention uncertainty

<!-- snippet: skip -->
```python
retention_uncertainty(
    scores,
    assignments,
    *,
    n_bins=None,
    confidence_level=0.95,
    rank_rtol=None,
) -> RetentionUncertainty
```

A retention number is a point estimate. `retention_uncertainty` attaches a standard error and a
two-sided Wald interval to the geometric-mean retention of a *frozen* rule evaluated on rows it never
saw: independent, equally weighted draws from the reference law at which the model's true scores are
evaluated, whose `scores` are those true scores and whose `assignments` come from `predict_scores`.
The estimate is the same uncentred plug-in that `information_report` reports on a full-rank sample;
the standard error is the influence-function estimate of the audited central limit theorem for that
plug-in. The interval covers the sampling variability of the evaluation draw under that theorem's
conditions — positive cell probabilities, finite fourth moments, positive definite full and
between-cell moments, positive asymptotic variance — and none of them can be read off an array: a
positive empirical rank does not certify a population eigenvalue floor. The theorem is asymptotic. An
infinite fourth moment puts a law outside it, and heavy tails, including laws that satisfy every
condition, undercover materially at moderate sample sizes. The reading of the number as the model's
Fisher retention needs the evaluation law to be the reference law, where the true score has mean
zero; on rows from another law the same arithmetic estimates an uncentred determinant ratio, not that
retention. The endpoints are never clipped, so an interval can extend below zero or above one.

Read `status` before the numbers. `ok` gives all three. `insufficient_cells` means the rule declares
at most \(d\) cells, empty ones included: the rank ceiling makes the population retention zero at the
reference law whatever the sample rank, the plug-in is the upward-biased endpoint estimator, and
nothing is reported. `singular_full_information` means the sample lost one of the supplied score
directions, so the retention of all of them is undefined; nothing is reported rather than a silently
projected surrogate, which is what `information_report` would show. `singular_retained_information`
means the between-cell moment is numerically rank deficient on this sample, which the theorem
excludes: the estimate is the plug-in's own value, zero, and the Wald theory does not apply. The
verdict is about the sample and does not identify the population rank; at population rank
\(r < d\) the plug-in is biased upward at order \(n^{-(d-r)/d}\). `degenerate_variance` means the
influence values cancelled to rounding noise, so the standard error is zero and the interval is
withheld. The test is \(\mathrm{rms}(B) \le \tau\,\mathrm{rms}(M)\), with \(B_i\) the bracket of the
influence value, \(M_i\) the sum of the magnitudes of its three terms, and \(\tau\) the dtype's rank
threshold, \(10^{-10}\) in float64 and \(10^{-5}\) in float32; `rank_rtol` moves the two rank guards,
not this threshold. It is a numerical guard: it absorbs a perturbation of a zero-variance law smaller
than \(\tau\), and it is not a finding that the population variance is zero.

Scores from a classifier or any other estimator are not true scores. Their reported retention
carries a separate proxy bias that this error bar never measures; the score-error budget bounds that
bias only under truth-dependent error and conditioning assumptions, and AUC or calibration alone does
not certify it. Weighted samples, rules refitted on the evaluation rows and profiled \(D_s\)
retention are outside this diagnostic.

## Certificates

<!-- snippet: skip -->
```python
exchange_stability_report(
    scores,
    labels,
    *,
    weights=None,
    criterion=None,
    rank_rtol=None,
    gain_tolerance=1e-10,
) -> StabilityReport

certify_partition(
    scores,
    *,
    weights=None,
    n_bins,
    incumbent=None,
    criterion=None,
    rank_rtol=None,
    config=None,
) -> PartitionCertificate
```

`exchange_stability_report` runs exactly one complete exact scan of a supplied labeling and
nothing else, so labels from any source — a `guard="reject"` batch result, an external tool, a
hand edit — can be checked before they are trusted. It reports the exact objective, the best
remaining gain, and the improving `(row, destination)` move when one exists. The cell count comes
from the labels, and every declared cell must hold positive weight. Stability is always a verdict
at a tolerance: match `gain_tolerance` to the configuration that produced the labels, and read it
back from `StabilityReport.gain_tolerance`. Pass `gain_tolerance=0.0` to ask the strict question
instead.

`certify_partition` decides global optimality by branch and bound with the singleton-completion
upper bound: unassigned atoms are left as singleton cells, so refinement monotonicity of the log
determinant bounds every completion of a partial assignment. It starts from an incumbent — normally
`PartitionResult.labels`, otherwise one default exchange — and returns `status="optimal"` only when
the tree was exhausted; a spent node budget returns `status="budget_exhausted"` with the best
outstanding bound and the remaining `gap`. The search is exponential, so `CertificationConfig`
guards both the node count and the number of distinct score atoms, refusing an oversized instance
by name. Certification is `DOptimality` only: the refinement bound uses Loewner monotonicity of
`logdet`, which the profiled Schur objective does not inherit, and a profiled criterion is rejected
rather than approximated. Neither entry point ever runs implicitly during fitting.

## Result semantics

`PartitionResult` has labels, cell statistics, information matrices, `rank`, `accepted_moves`,
`scans`, `lloyd_iterations`, `accepted_lloyd_steps`, `exchange_stable`, and `best_remaining_gain`,
but no prediction method. One scan is one complete evaluation of every admissible relocation; with
the default `batch_moves` a single scan may relocate many rows, so `accepted_moves` normally
exceeds `scans`. The two Lloyd counters stay zero unless the guarded batch solver ran, and
`objective_history` records every accepted step of every phase in order. Its `compile_quantizer()`
raises `RefusalError` on an unstable or geometrically degenerate result.

Geometry diagnostics are criterion-specific and never shared. A `DOptimality` result carries
`geometry`, a `GeometryReport` measuring the largest Mahalanobis-Voronoi violation of the terminal
metric, the guaranteed log-determinant gain such a violation would leave on the table, the largest
exact gain it actually leaves, and the cell-separation residual against the leverage bound; a
`ProfiledDOptimality` result carries `profiled_geometry` instead and leaves `geometry` as `None`.
The two describe different objects — a strict Voronoi rule that exchange stability forces, and an
efficient semimetric whose Voronoi rule a stable profiled partition may violate — so one shared
name would claim an implication that does not hold.

`GeometryReport` is a certificate at a tolerance, recorded as `gain_tolerance` and taken from the
configuration that produced the labels. `voronoi_consistent` means `maximum_violation_gain <=
gain_tolerance`: no Voronoi violation is worth more than the threshold the solver stopped at. It is
not a claim at tolerance zero, and on a large sample it must not be — the Theorem-3 guaranteed gain
falls like \(1/N^2\), so past roughly a million rows a few rows legitimately sit a hair past a cell
boundary inside the default \(10^{-10}\). `compile_quantizer()` reads that certificate rather than
demanding bit-exact label reproduction, so on such a partition `predict_scores` on the training
scores differs from `PartitionResult.labels` on those boundary rows and nowhere else; the
partition's labels stay authoritative for the fixed sample. Assignment itself is unchanged —
`argmin` decides, resolving a tie toward the lowest cell index.

`QuantizerResult.predict_scores(scores)` is the only prediction method. `evaluate_scores` assigns
new scores with the frozen rule and computes supplied-score information. The stored transform,
centers, and optional common metric define its score-space geometry; `rank`, train/validation
reports, hardening gap, solver contract, source kind, and score provenance remain inspectable.
`OptimizationTrace.objective_label` names the units of the recorded objective, because solvers do
not share one convention: `"whitened_sse"` is a minimized weighted within-cell squared error in
Fisher-whitened coordinates, while `"logdet_retained"` and `"profiled_logdet"` are maximized log
determinants. Two traces are comparable only under the same label.
Both result types expose `information_kind`: it is `exact_fisher` only for exact/autodiff
provenance and `supplied_score_surrogate` otherwise.

## Shape and measure contracts

- Scores: finite `[N, P]`, `N > 0`, `P > 0`.
- Observations: finite `[N, D]`.
- Weights: finite nonnegative `[N]` with at least one positive value.
- Classifier central probabilities: positive `[N, P, 2]`, normalized on the final axis.
- Multiclass posteriors: nonnegative `[N, K]`, row-normalized, with positive normalized priors.
- Model density ratios: finite nonnegative `[N, K]`, defined up to a common event-wise factor.
- Integration bounds: finite `[D, 2]` with strictly ordered endpoints and an explicit density.

Numerical null directions are projected out. Scores are never centered. `to_dict()` returns
JSON-ready diagnostic state but is not a durable artifact format.
