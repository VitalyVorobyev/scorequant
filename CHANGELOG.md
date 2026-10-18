# Changelog

All notable changes to ScoreQuant are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html) — with the usual `0.x` caveat that the
public API may still change between minor releases.

An `[Unreleased]` section is created by the first change that lands after a release. The file
deliberately carries no empty one: a standing `unreleased` heading was dated by a later commit
once already, which retroactively asserted that unshipped work had shipped.

## [0.4.0] — 2026-10-18

**Breaking:** the top level is cut to a core plus five public submodules, and iteration caps are
all `max_iter`. The table under "Changed (breaking)" maps every old spelling to its new one.
This release is the candidate for the 1.0 API.

### Added

- A portal benchmark page sets each walkthrough's result beside equal-width, quantile and k-means
  bins at the same budget; `examples/benchmark_baselines.py` computes the baselines the studies lacked.

### Changed

- One singularity rule: `information_report`, `profiled_information_report`, `efficient_scores`
  and the soft solver judge rank by the library's relative eigenvalue threshold instead of a
  log-determinant sign or NumPy's default `matrix_rank`, and an exchange result reports the
  objective rebuilt from its final labels rather than the incrementally updated one.
- Default tolerances follow the working dtype: `gain_tolerance` (now `None` by default in
  `DExchangeConfig`, `MahalanobisLloydConfig` and `exchange_stability_report`), the rank threshold
  and the separation slack keep their float64 values and rise to the float32 noise floor
  (`gain_tolerance` about `1.2e-6`, ten epsilon; the rank threshold about `1.2e-5`), so
  float32 certificates sit above rounding noise; results record the resolved value.
- `RatioProvenance.gauges` declares the measure each ratio column is taken against; a record
  whose declared gauges differ raises `ContractError` naming the one-gauge recipe, and
  `DensityRatioScore.from_classifier` declares `"training_mixture"`. The user guide shows the
  recipe for a classifier signal next to a background.
- A soft fit whose schedule ends on a singular soft Fisher matrix raises `RefusalError`
  (`CE-DS-MARGINS-RANK-VACUITY-001`) instead of returning stalled centres.
- The soft solver and the profiled geometry report work in memory-bounded row chunks, so memory
  no longer grows as \(N \times\) bins \(\times\) dimension; the soft gradient is compiled under JAX.
- Internals only: `partition.py` is split into the exchange engine and Lloyd solver
  (`solvers/`), the exchange algebra and the stability and geometry diagnostics; nearest-centre
  assignment is one routine; `retention_uncertainty` moved to its own module. The split leaves
  the public API alone, and a missing nuisance block reports one shared message.

### Examples

- `examples/hep_classifier/pyhf_fit.py` fits the HEP walkthrough's categories as a standard pyhf
  HistFactory model, and the `examples` extra now includes `pyhf`.

### Changed (breaking)

The public surface is cut from 62 top-level names to 27, plus five public submodules, ahead of
the 1.0 freeze. Every supported object now has exactly one route. A saved quantizer artifact is
unchanged and still loads.

- The top level keeps what a first fit needs: the two tasks, the sources, `ScoreProvider`,
  `ScoreFunction` and `LinearComponentScore`, the criteria and configurations, `Quantizer`,
  `information_report`, `efficient_score_bound`, `retention_uncertainty` and the errors.
- `scorequant.ratios` holds the density-ratio stack: `DensityRatioScore`,
  `CentralLogRatioScore`, `MixtureParameterization`, `IntensityParameterization`,
  `RatioProvenance`, `ratios_from_posteriors`, `ratio_closure_report` and `RatioClosureReport`.
- `scorequant.information` holds the information algebra, `scorequant.certify` the certificates,
  and `scorequant.results` every result and report type, `FisherTransform` included.
- One route each for linear models and ratios. `LinearComponentScore` takes the components and
  coefficients itself, so `LinearComponents` is removed. `scores_from_components` and
  `mixture_scores_from_ratios` duplicated the parameterizations' `scores` methods and are removed.
- The plot helpers move to `scorequant.plotting`, which is outside the stable API.
  `QuantizerResult.plot_summary` is removed.
- Every iteration cap is `max_iter`: `DExchangeConfig.max_scans` and `SoftVoronoiConfig.max_steps`
  are renamed. `SoftVoronoiConfig.tolerance`, which only ever governed its k-means initializer,
  is `kmeans_tolerance`, next to `kmeans_max_iter`.
- `ScalarDPConfig.seed` is removed: the dynamic program consumes no randomness.

| 0.3 | Now |
| --- | --- |
| `sq.LinearComponentScore(sq.LinearComponents(f, c))` | `sq.LinearComponentScore(f, c)` |
| `sq.scores_from_components(values, c)` | `sq.ratios.IntensityParameterization(c).scores(values)` |
| `sq.mixture_scores_from_ratios(r, f, reference_component=k)` | `sq.ratios.MixtureParameterization(f, reference_component=k).scores(r)` |
| `sq.DensityRatioScore`, `sq.CentralLogRatioScore`, `sq.MixtureParameterization`, `sq.IntensityParameterization`, `sq.RatioProvenance`, `sq.ratios_from_posteriors`, `sq.ratio_closure_report`, `sq.RatioClosureReport` | the same names in `sq.ratios` |
| `sq.fisher_information`, `sq.binned_fisher_information`, `sq.fractional_fisher_information`, `sq.efficient_scores`, `sq.profiled_information_report` | the same names in `sq.information` |
| `sq.certify_partition`, `sq.CertificationConfig`, `sq.exchange_stability_report` | the same names in `sq.certify` |
| `sq.PartitionResult`, `sq.QuantizerResult`, `sq.OptimizationTrace`, `sq.FisherTransform` and the report types | the same names in `sq.results` |
| `sq.plot_*`, `result.plot_summary(scores)` | `scorequant.plotting.plot_*`, `plotting.plot_summary(result, scores)` |
| `DExchangeConfig(max_scans=...)` | `DExchangeConfig(max_iter=...)` |
| `SoftVoronoiConfig(max_steps=...)` | `SoftVoronoiConfig(max_iter=...)` |
| `SoftVoronoiConfig(tolerance=...)` | `SoftVoronoiConfig(kmeans_tolerance=...)` |
| `ScalarDPConfig(seed=...)` | `ScalarDPConfig()` |

## [0.3.0] — 2026-10-18

Older entries cite design records as "ADR NNNN"; those records are in the repository history at
the `research-archive-2026-10` tag.

The development status moves from Alpha to Beta.

### Added

- `retention_uncertainty` and `RetentionUncertainty`: the plug-in standard error and Wald interval
  for a frozen rule's `geometric_mean_retention`, from an independent, equally weighted sample of
  true scores from the reference law. Degenerate cases (a rule with at most \(d\) declared cells,
  a rank-deficient moment, influence values that cancel to rounding) return a named status with
  the interval withheld. The interval covers sampling uncertainty only, under the conditions in
  `docs/api.md`.
- A compatibility fixture that loads a saved 0.2.0 quantizer, and a written inventory of the
  public API that 1.x will keep stable.

### Changed

- A loaded `Quantizer` no longer restores the execution it was fitted under, so a rule fitted on a
  GPU, or with float64 on JAX, loads and predicts on a CPU-only or JAX-less machine. It uses JAX
  when importable and NumPy otherwise. `save` still writes a default `execution` object so 0.2
  readers can open the file; the artifact format version is unchanged.
- `precision="preserve"` on JAX without X64 now warns once when it computes float64 inputs in
  float32, instead of truncating silently under tolerances tuned for float64.
- `information_report`'s docstring states its uncentred second-moment convention.

### Documentation

- The reference site is now the user guide, the API guide, the symbol reference and the glossary;
  the learning material lives in the portal at the site root, with the reference under `/docs/`.
  The book chapters, case-study pages, gallery and notebooks are gone.
- D-only global certification is explained as an implementation boundary, not a failure of Schur
  refinement monotonicity.
- The FlowCyt, HEP and Michelson walkthroughs are rewritten as linear scientific walkthroughs.

### Examples

- `examples/hep_classifier` evaluates reusable rules out of sample, with bootstrap envelopes,
  per-half certified ceilings and the downstream signal-strength uncertainty. Example scripts that
  only fed the deleted pages are removed; study outputs live in `examples/outputs`.

### Removed

- The private `scorequant.quantizers` facade; internal imports use the owning
  `scorequant.solvers` modules, which remain private.

## [0.2.0] — 2026-09-04

The consolidation release (roadmap milestone M12). The public surface gains a small error
hierarchy and loses one exported name that no task accepted; the documentation becomes a learning
portal served at the URL the package advertises.

**Breaking:** `LinearProblem` and `LinearComponents.evaluate` are removed — see Removed below.

Everything here sat on `main` under an `[Unreleased]` heading that had to be created before this
release could be described honestly. The entries had been appended while the version heading still
read `unreleased`, and a later commit dated that heading, which retroactively asserted they had
shipped in 0.1.0. Verified at the `v0.1.0` tag: `RefusalError` does not exist anywhere in `src/`
there, and `LinearProblem` is still exported.

### Site

- The learning portal is published at the site root, with the strict MkDocs reference beneath it
  at `reference/`, assembled and deployed by a single workflow (ADR 0026). A pull request builds
  and uploads the whole tree; only a push to `main` deploys.

### Contracts

- Weight and `rank_rtol` validation is single-sourced; the messages are the `ScoreSample` ones
  everywhere.

### Errors

- Every deliberate exception is a `ScoreQuantError`. `ContractError` (a `ValueError`) reports a
  malformed call; `RefusalError` (a `RuntimeError`, deliberately not a `ValueError`) reports a
  theorem-backed refusal and carries `counterexample`, the registry id that forces it.
  `compile_quantizer()` on an unstable, profiled, or geometrically degenerate partition and a
  rank-deficient profiled `fit_quantizer` now raise `RefusalError`.

### Removed

- `LinearProblem` and `LinearComponents.evaluate` — exported and documented but accepted by no
  task; use `LinearComponents.evaluate_components` plus `scores_from_components` to hand an
  evaluated component matrix to `optimize_partition` or a `ScoreSample`.

## [0.1.0] — 2026-08-30

First public release. Everything below describes the shape being released rather than a change
against a previous version, since there is none.

### Two tasks, kept separate

- `optimize_partition(scores, ...) -> PartitionResult` optimizes the labels of one fixed weighted
  score table. It is transductive, and the result has no generic predict method.
- `fit_quantizer(source, provider=..., ...) -> QuantizerResult` fits a reusable rule on score
  space. Prediction is always the explicit `predict_scores`.
- The one crossing between them is a theorem, not a convenience:
  `PartitionResult.compile_quantizer()` returns the Mahalanobis-Voronoi rule that an
  exchange-stable, nonsingular D-optimal partition already is, and refuses otherwise.

### Score access

- Three routes to a score, all reaching the same optimization problem: precomputed vectors
  (`ScoreSample`), an explicit model or oracle (`ScoreFunction`, `LinearComponentScore`), and
  density ratios (`DensityRatioScore`, `CentralLogRatioScore`).
- `ScoreProvider` is a public runtime-checkable protocol, so an external estimator is a provider
  without being wrapped.
- `ScoreSchema` names the score coordinates, so a profiled criterion can declare
  `interest=("HSPCs",)` instead of `interest=(4,)` and reports print names.
- Sources and providers are separate contracts. Model density ratios build scores and enter through
  providers; importance ratios reweight the measure and enter as source weights.

### Objectives, solvers and certificates

- `DOptimality`, `ProfiledDOptimality` and `NormalizedTrace`, paired with a closed set of solver
  configurations. An unsupported pairing is rejected before any optimization runs.
- Exact positive-gain D exchange, guarded Mahalanobis-Lloyd, annealed soft Voronoi, weighted
  k-means, and an exact scalar interval dynamic program.
- Certificates are explicit, separately invoked operations that never run silently during fitting:
  `exchange_stability_report`, `certify_partition`, `efficient_score_bound`, and the measured
  geometry reports.

### Deployment

- `Quantizer` is the deployable rule, separate from the record of the fit. `Quantizer.save` writes
  a versioned, non-pickle artifact that loads and predicts in a process with no JAX installed.
- Every public entry point takes `execution=`. JAX is the default runtime; NumPy is the supported
  portable one, which is what makes the browser and JAX-free deployment possible.

### Contracts

- Public arrays are always `numpy.ndarray`. The package never mutates global JAX configuration at
  import, so 64-bit precision stays an application choice.
- Scores are never centred; numerically singular Fisher directions are projected out rather than
  repaired with a ridge; validation data is diagnostic only.
