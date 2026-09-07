# Changelog

All notable changes to ScoreQuant are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html) — with the usual `0.x` caveat that the
public API may still change between minor releases.

An `[Unreleased]` section is created by the first change that lands after a release. The file
deliberately carries no empty one: a standing `unreleased` heading was dated by a later commit
once already, which retroactively asserted that unshipped work had shipped.

## [Unreleased]

### Site

- The learning portal takes the site root and the hand-written landing page is retired
  (ADR 0033, superseding ADR 0027). Two surfaces remain: the portal at `/`, the MkDocs
  documentation at `/docs/`. `landing/` and its guard test are deleted, `baseUrl` and `SITE_BASE`
  move together, and the assemble script's landing-link parity check becomes a check that every
  `/docs/` link the built portal carries resolves in the assembled tree. The portal's two days at
  `portal/` are not stubbed, for the reason ADR 0027 gave for not stubbing its one day at the root.
- The portal home page is rewritten to the shape ADR 0033 amends into ADR 0031: what the library
  is for, why score space, the two tasks, where scores come from and what is optimised. The
  displayed binning-cost identity and the list of derivations move off it; the primary
  navigation's *Reference* entry is the home page's route into the documentation.
- `/get-started` is shortened to one arc — a table, a reusable quantizer, its prediction, and the
  finite partition of the same table — with the compile bridge, the theorem-backed refusal,
  profiled `D_s`, the scalar dynamic programme and global certification left to the reference
  documentation. Its cells still come from `website/scripts/get_started_program.py` and its output
  is still captured rather than typed.
- The density-ratio walkthrough is retitled *From a classifier to Fisher-preserving bins* and
  reopens on a schematic of its two lanes: the workflow a real analysis runs, and the oracle check
  the synthetic model affords. Its browser refit is removed — the page's argument is the gap
  between a reported and an achieved retention, which refitting the estimated score cannot show.
- The site topology is reversed after the owner's review of the portal front door (ADR 0027,
  superseding ADR 0025). A hand-written landing page owns the site root and links to both
  surfaces; the MkDocs documentation is mounted at `docs/` with *Why ScoreQuant*, *Three doors*
  and *Choosing your workflow* restored to it; the portal is published unchanged at `portal/`.
  The package's Documentation URL now points at `docs/`; every pre-portal URL still redirects.
- The portal's free-form Lab console (solver pickers, file upload, notebook embed) is removed;
  `/portal/lab/` is a lesson index and browser computation is reached only through a lesson's
  experiment (ADR 0029).
- The Michelson walkthrough is the first lesson in the seven-step pattern of ADR 0028: a problem
  contract before any exposition, the model and score as displayed equations, and one experiment
  (the bin budget) whose browser refit reproduces the committed profiled partition on the same
  table. The walkthroughs' links into the reference now resolve through `ReferenceLink`; the
  hand-written `pathname:///reference/…` anchors reached the browser verbatim and were not URLs.
- The portal is reduced to four surfaces after the owner's review of 5 September 2026 (ADR
  0031): the home page is definitions and references in ordinary type, with no measurement and
  no demo; the Lessons index, the API catalogue, the Benchmarks page and the development blog
  are removed; the walkthroughs index is a card per problem with API tags; the Michelson page is
  an article (the instrument, its history, a bench diagram, the fringe law, the readout question)
  rather than a numbered procedure with a contract table; every "who this is for" preface is gone.

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
