# Decisions

The standing decisions that constrain the code, the sites and the research workspace. One
entry per decision, each a few lines: what is decided and the constraint it imposes. The
original architecture decision records (ADR 0001–0033) were narrative histories of individual
pull requests; they are in git history under `docs/adr/`, and their numbers survive here as
labels so that references in prose and code stay resolvable. A new durable decision is a new
entry at the end, not a new file.

## ADR 0001 · Scores are the core data contract

Optimizers take weighted score rows, `scores [N, P]` with optional `weights [N]`. Score
generation is upstream: linear components, analytic likelihoods, autodiff, simulators and
learned estimators are adapters to this contract, never inputs to the optimizers.

## ADR 0003 / 0007 · A small core with generic statistical contracts

The core holds information calculations, quantizers, diagnostics and the linear-component
adapter. Deferred until a real use case demands them: learned score estimation, services,
plugin systems, large experiment schemas, compiled deployment runtimes. API preservation is not
a goal before a stable release. A capability enters the public library only when its name and
semantics refer to no dataset or domain, its invariants and failure modes are stateable
independently of one example, it composes with the existing representation layers, deterministic
tests exercise it without application fixtures, and it removes duplication for plausible callers.
Application orchestration (data access, training, tuning, baselines, downstream likelihoods,
figures) stays in `examples/`.

## ADR 0009 / 0023 / 0024 · Two tasks, the rule as an artifact, two kinds of error

`optimize_partition` returns a `PartitionResult` with no prediction method; `fit_quantizer`
returns a `QuantizerResult` that predicts only from explicit scores through `predict_scores`.
`compile_quantizer` is allowed only from a positive-definite, one-point-exchange-stable D state
whose Mahalanobis rule reproduces every positive-weight training label at the optimizer's
tolerance (ADR 0016); finite profiled-\(D_s\) or E labels never compile.

`Quantizer` is the rule alone (transform, centers, optional metric, schema, provenance,
criterion, execution) and holds nothing about the training sample; `evaluate_scores` lives on
it. Its artifact is a zip of one `manifest.json` plus one `.npy` per array, versioned by
`format_version`, read with `allow_pickle=False`, loadable and predictable without JAX.
`exact_fisher` is derived from `kind`, never read back from a file.

Exceptions: `ScoreQuantError` is the base; `ContractError(ValueError)` is a malformed call;
`RefusalError(RuntimeError)` is a theorem-backed refusal on otherwise acceptable data and carries a
required `counterexample` attribute naming the registry fixture that forces it. `fit_quantizer`
is two stages, a compiled path and a geometric path, with the profiled-degeneracy guard at the
one point between hard assignment and the reports. The prediction kernel is the leaf module
`scorequant._predict`.

## ADR 0010 / 0017 / 0022 · Sources, providers, and density ratios

The reference measure and the observation-to-score map are separate contracts, validated
together. Measures: `ScoreSample`, `ObservationSample`, `IntegrationSource` (bounded integration
needs an explicit density or intensity). Maps: any `ScoreProvider`, a runtime-checkable protocol
with `provenance` and `score(observations)`; the built-ins `ScoreFunction`,
`LinearComponentScore`, `DensityRatioScore`, `CentralLogRatioScore` implement it. A `ScoreSample`
rejects a provider; observation and integration sources require one. The argument is `provider=`.

`ratios.py` owns density-ratio algebra: prior correction, ratio-to-score maps under a declared
parameterization, the closure diagnostic. `DensityRatioScore.from_classifier` composes
posteriors → prior-corrected ratios → scores and can never claim exact provenance; estimated
ratios never claim exact Fisher semantics and closure never upgrades them. Model density ratios
enter through providers; importance ratios are source weights; the two never share an argument.
Ratio estimation, training, calibration and cross-fitting stay outside the core.

## ADR 0011 / 0014 / 0016 · Criterion-specific solvers, one exchange engine, explicit certificates

Criterion and configuration types form a closed, validated table; there is no generic
criterion plugin:

| Config | `optimize_partition` | `fit_quantizer` |
| --- | --- | --- |
| `DExchangeConfig` | `DOptimality`, `ProfiledDOptimality` | `DOptimality` |
| `MahalanobisLloydConfig` | `DOptimality`, `ProfiledDOptimality` | `DOptimality` |
| `KMeansConfig` | — | `NormalizedTrace` |
| `SoftVoronoiConfig` | — | `DOptimality`, `ProfiledDOptimality` |
| `ScalarDPConfig` | — | `DOptimality` |

D and profiled-\(D_s\) exchange run on one engine parameterized by a private
`_ExchangeObjective`; batch moves are proposed by nearest-centroid and accepted only when the
exactly rebuilt objective strictly improves, halving on rejection. Certificates are explicit and
never run implicitly: `exchange_stability_report` (one exact scan over any labeling),
`GeometryReport` and `ProfiledGeometryReport` (measured Voronoi self-consistency and leverage
separation), `certify_partition` (bounded branch-and-bound, `DOptimality` only, `optimal` only
when the tree is exhausted).

Every verification uses the tolerance the partition was optimized at: verdicts are taken in
exact gain units against `config.gain_tolerance`, both certificates record their tolerance,
`compile_quantizer` takes no tolerance argument and delegates to the stamped geometry
certificate, and assignment (`argmin`, ties to the lowest index) is untouched.

## ADR 0015 · The efficient-score bound and solver initialization

`efficient_score_bound` builds the full-data efficient score \(\hat s=s_\psi-B^*s_\lambda\) from
the unbinned information and applies efficient-score domination: the same-label profiled
information of any hard rule with at most `n_bins` cells is bounded by the between-cell
information of \(\hat s\) under that rule. One scalar parameter of interest only; the maximizing
interval rule is found exactly by the scalar dynamic program. Its labels double as
`initial_labels` for restart 0 of `optimize_partition`. `fit_quantizer(diagnostics=...)`
controls how many center snapshots are re-scored into the hard-retention history; the default
scores the endpoints.

## ADR 0018 · Explicit JAX and NumPy execution behind one mathematical core

One public immutable `ExecutionConfig` (backend, precision, device); `execution=None` means the
JAX default at a public boundary and inherits the active scope inside an operation. Dependency
direction: domain contracts → private execution adapters → shared kernels → solver orchestration
→ public API. Equations and solver flow are shared; adapters own conversion, scatter, seeding,
placement, compilation and optimizer updates. All public arrays are `numpy.ndarray`; results
record their execution. One private capability table; no public backend registry or base class.
JAX and Optax stay normal dependencies behind Emscripten markers; no PyTorch; the library never
sets global X64.

## ADR 0021 · Named score coordinates, reference point in provenance

`ScoreSchema` is a frozen tuple of parameter names carried by samples, providers, results and
the profiled report. `ProfiledDOptimality.interest` takes names or indices, never both; names
resolve to columns exactly once, at the public task boundary, and every downstream consumer reads
integers. The schema does not carry the reference point; `ScoreProvenance.reference_point` does,
and the two are validated against each other and the score dimension. Validation samples are
compared by parameter name when both sides declare one.

## ADR 0026 / 0027 / 0035 · Site topology and its one workflow

Two surfaces in one assembled tree since 7 September 2026 (ADR 0035): `/` is the Docusaurus
portal, `/docs/` is the MkDocs documentation and book. The hand-written landing page ADR 0027 put
at the root is deleted with its guards -- the portal's home page is the front door, and its
primary navigation carries the Reference entry into `/docs/` that the landing page used to carry.
The portal's `404.html` therefore serves the whole domain. `website/src/lib/site.ts` and
`mkdocs.yml` state the topology and must move together. One workflow, `site.yml`, runs the strict
MkDocs build, the portal checks, `pnpm assemble:site` and the deploy. Pre-cut URLs are stubbed by
`website/redirects.json`, and the portal must never emit a `docs/` route, which would be
overwritten silently; `assemble-site.mjs` refuses that build and resolves every
`/scorequant/docs/` href in the built portal against the assembled tree, which is the one class of
link Docusaurus's `onBrokenLinks` cannot see. The two days the portal spent at `/portal/` are
deliberately not stubbed, for the reason ADR 0027 gave for not stubbing its one day at the root.

## ADR 0031 / 0032 / 0028 / 0029 · The portal is four surfaces, and its pages are articles

Surfaces: Get started, Walkthroughs, Research, and the Reference link into MkDocs. The home page
is definitions and runs nothing; since it became the site root (ADR 0035) it also quotes no
measurement, and it carries neither the displayed binning-cost identity nor the list of where each
definition is derived, both of which belonged to a page competing with a landing page to be read
first. A walkthrough is an article: the author's
checklist is problem, model and score, admissible labels and criterion, run, evaluation, one
experiment, interpretation, in that order but not headed by step names; it opens with the
subject and the data, states the contract in prose, and puts numbers in sentences. Browser
computation runs only behind a page's explicit action. Figures: `website/static/figures/` is
committed and hand-placed; `website/static/walkthrough-figures/` is derived by
`generate_walkthroughs.py` and gitignored; `website/tests/figures.test.ts` resolves every
figure reference against the lane that owns it.

## ADR 0030 · A bounded formal-verification track

The Lean track covers the finite D chain in general dimension (D2 relocation identity, D3
determinant gain, D4 leverage inequalities, D5 with its quantitative bound, the D7/D8
corollaries) and nothing beyond it until a further decision; profiled \(D_s\), population measure
theory and asymptotics are out. A claim carries `formal_proof` only when its statement is
separately frozen in a `*Spec.lean` and independently audited; the freeze is closed under
definitional dependency, and the marked declaration's type is the frozen conclusion. Counterexample
claims never carry the field. Trust gate: `lake build --wfail` on a pinned toolchain,
`leanchecker`, and an axiom allowlist of `propext`, `Classical.choice`, `Quot.sound` pinned by
`AxiomAudit.lean`. No Lean result certifies the Python/JAX implementation.

## ADR 0033 · The research atlas is generated from the registry

`registry.py export` writes the research graph; `website/scripts/generate_atlas.py` owns every
public addition and writes `website/src/generated/atlas.json`, kept fresh by
`tests/test_atlas_data.py`. Every claim is published unless named in
`website/content/research-public.json`; a published edge into a withheld id fails the build.
Provenance classes derive mechanically from `status` and `literature_search_status`; the words
"novel" and "first" do not appear. Edges are typed for a reader; proof prose is public with
work-tracking vocabulary rewritten on the way out; hand-written content is one note per entity
under `website/content/atlas/notes/`.

## ADR 0034 · One decisions file, a finite research programme, history in git

Adopted 6 September 2026. `docs/decisions.md` replaces `docs/adr/`; a durable decision is a new
entry here. The research workspace's operating contract is `agenticresearch/README.md`,
`OPEN_PROBLEMS.md` (the closure programme, the status, the backlog) and `protocols/`; every
other research file is scientific memory reached through a claim id. Completed packets,
archives and superseded manuscripts are deleted when they close; git history is their record.
The research programme is finite: the closure table in `OPEN_PROBLEMS.md` names the remaining
steps and the end state, one session per step, no retries, and the backlog is worked only if
the owner moves an item into that table.

## ADR 0036 · Research presentation has three levels

Home tells, Atlas explores, Registry proves provenance. The research home is a bounded editorial
argument, with four central results including its lead theorem, two boundaries, and three open
questions. `website/content/atlas/home.json` and `summaries.json` add validated presentation text
keyed to registry claims; exact statements and provenance remain authoritative upstream. Explore
is theme-first, graphs are neighbourhood-first, and detailed evidence uses fragment-aware
disclosure. All entity routes and typed edges survive. Literature coverage is independent of home
selection; proof, publication, and priority-search status remain distinct. Proof and audit status
may appear beside a featured result only as marks derived from the registry, never as editorial
text. Listing titles state a claim, not an instruction. Reader acceptance is a human gate, not
inferred from passing automated checks.

## ADR 0038 · Walkthrough evidence: held-out rules, downstream metrics, one honest headline

An applied walkthrough leads with a quantity its domain reports, measured on data the rule never
saw, and says which of three things every number is: a finite partition of a fixed table (a
labelling of those rows, with no predict method), a reusable rule scored on the rows it was
fitted on, or a reusable rule applied to held-out rows. Local Fisher retention is the
explanation, not the headline; a partition's in-sample retention is a methodological reference,
never quoted as an out-of-sample result. Consequences, in force on the FlowCyt and HEP pages:

- The FlowCyt page's code runs the reference-to-held-out chain (fit, freeze, `predict_scores`,
  per-patient counts, mixture fit, unbinned comparison) on a committed fixture-scale table,
  `examples/data/flowcyt_walkthrough.npz`, whose sidecar records what that run produces so the
  page can state its own outcome next to the full study's. The learned categories are shown
  as \(P(\text{population}\mid\text{bin})\), never as a projection.
- The HEP study splits the events once, stratified, into two halves; every reusable rule is
  built on one half and evaluated on the other, in both directions, with percentile bootstrap
  envelopes and the evaluation half's own certified ceiling. The primary applied result is the
  expected signal-strength uncertainty from each rule's own count-table likelihood, built from the
  simulation's energy-scale-shifted copies, reported with the nuisance fixed and floating. The
  `tes` classifier trains under the Monte Carlo weights (an unweighted one estimates the ratio of
  a one-third-signal population and reverses the downstream conclusion), and the energy scale is
  stated to be unconstrained wherever it is profiled. The two-bin significance cut is reported as
  not identified under three floating parameters, not as zero retention. A label-tuned
  classifier baseline requires a minimum simulated background count per interval.
- A rule's held-out retention is computed outside the fit from `predict_scores` and checked
  against `fit_quantizer`'s validation report; the in-sample finite partition and its ceiling
  stay on the page, labelled in sample.

## Absorbed and superseded records

| Record | Where it lives now |
| --- | --- |
| ADR 0002 NumPy first | superseded by ADR 0004, then ADR 0018 |
| ADR 0004 JAX first | ADR 0018 |
| ADR 0005 explicit representations, `fit`/`fit_components`/`fit_scores` | superseded by ADR 0009 |
| ADR 0006 MkDocs site on GitHub Pages | ADR 0027 / 0026 |
| ADR 0008 classifier-posterior bridge | superseded by ADR 0012, then ADR 0017 |
| ADR 0012 classifier callback boundary | ADR 0017 (ratio layer) |
| ADR 0013 pre-1.0 API boundaries | ADR 0014 (certificates) and ADR 0009 / 0023 / 0024 |
| ADR 0019 React learning portal, ADR 0020 portal blog | ADR 0031 (surfaces) and ADR 0027 (topology); the blog is removed |
| ADR 0025 portal at the site root | superseded by ADR 0027, then restored by ADR 0035 |
| ADR 0027 hand-written landing page at the root | superseded by ADR 0035; `landing/` and `tests/test_landing.py` are deleted |
| ADR 0028 focused research and teaching, ADR 0029 lessons replace the Lab | ADR 0031 and `agenticresearch/README.md` |
