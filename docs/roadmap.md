# ScoreQuant development roadmap

The sole executable development plan. Research work selection is in
`agenticresearch/OPEN_PROBLEMS.md`; proof packets do not create library commitments. Dated
reviews are not kept as files; git history holds them, and a phase cites the pull request that
acted on one.

## M13 — Focused research and teaching, then the road to 1.0

**Current (11 September 2026):** phase F's graph migration and phase H's local engineering
are implemented; all automated gate results are recorded below. Manuscript v10 merged in PR #68, but
publication assurance, owner freeze and the human reader gates remain open. Engineering preparation
may precede 0.3 (ADR 0042). The next research session is the separate, non-blocking HEP
template-likelihood assessment (P5); no new solver or public capability was added.

| Phase | Status and next action | Gate / stop |
| --- | --- | --- |
| A — Geometry and interpretation | Done: the false boundary overlay retired and the optical, periodicity, compilation and information claims corrected (PRs #52, #56). | Equations, chart labels and prose agree; facts/snippets and frontend checks pass. |
| B — Michelson exemplar | Revised 6 September 2026 (PR #56) as a two-act article: instrument → model and analytic score → D-optimal quantization and its compiled rule → nuisance profiling and the profiled \(D_s\) partition → the reusable profiled rule → one experiment. **Human reader gate open.** | A fresh reader explains the physical quantity, nuisance, allowed labels and finite-table certificate. |
| C — Remaining articles | Ratios rewritten 7 September 2026 (PR #62); FlowCyt and HEP rebuilt as held-out-evidence walkthroughs 7 September 2026 (PR #63, ADR 0038). **Human reader gate open.** | Each starts with subject, source and an explanatory figure; one result comparison; one experiment unless the page's argument does not admit one (ratios, by ADR 0035). Facts/snippets, desktop/mobile e2e and links pass. |
| D — Frozen-rule uncertainty | Done: scalar O6 (PR #53), vector O7 (PR #59), score-error budget O8 (PR #65), all independently audited; `retention_uncertainty` shipped 8 September 2026 (PR #66, ADR 0039). The manuscript harvest is closure step 5. | Public estimator under the full contributor gate with seeded coverage and heavy-tail tests. Weighted, refitted, profiled and truth-free uncertainty stay backlog. |
| E — Formal verification | Parked as partial (PR #64): D2–D8 and the compiled predictor are machine-checked under ADR 0037; the residue (positive tolerance, zero weights, D5 duplicate branch, D7 equal-optimum half, D12, one frozen-file import) is listed in `agenticresearch/OPEN_PROBLEMS.md`. | Build, statement correspondence, allowed axioms and checker pass. Reopened only by owner decision. |
| F — Claim graph | Implemented: mathematical `dependencies` are separate from `verified_by` and `references`; cycles are rejected, lookup is proof-only, and Atlas evidence remains navigable (ADR 0041). | Registry, index freshness, fixture, Atlas and portal gates pass; IDs/statuses/formal markers preserved. |
| G — Research Atlas | Editorial redesign delivered (PR #57, ADR 0036). **Human acceptance open.** | A fresh researcher identifies the contribution, boundary and open question, then finds theorem and prior art. |
| H — The 1.0 gate | Local review and engineering implemented before 0.3 (ADR 0042): API inventory, façade removal, historical artifact fixture and audited compilation wording. Release remains pending: manuscript assurance and owner freeze, authorized 0.3 tag with Beta classifier, human gates B/C/G, public manuscript, then 1.0 with Production/Stable classifier. | Findings below; do not mark H complete until owner/publication/human gates close. No version bump, tag, push or publication performed. |

### H — v1.0 review disposition (11 September 2026)

| Issue | Verdict and evidence | Next action / acceptance criterion |
| --- | --- | --- |
| Compilation at positive tolerance | Required correction, applied. Independent `AUDIT-D-COMPILE-TOLERANCE-001.md` and two exact fixtures pin admissible individual gains, singleton refusal and the absence of a batch-change bound. Source/docs/D6/claim and manuscript (4.6) agree. | Keep the new boundary regressions green; no stronger aggregate or population guarantee is claimed. |
| Profiled certification rationale | Required correction, applied. Runtime refusal and API guide now describe missing implementation/singular-block policy, not failure of Schur monotonicity. | D-only behaviour remains unchanged. |
| API and artifact compatibility | Required correction, applied. Complete supported-name/member and diagnostic-key inventories in `docs/api.md`; private façade removed; v0.2.0-written format-1 fixture loads without JAX. | Preserve the inventory and historical-reader checks through 1.x under ADRs 0040/0042. |
| Manuscript dispute provenance | Release/publication blocker. Historical packet at `6a4c7b3` records two disputed rows only as a count, without identities/dispositions. It also identifies unresolved source-read/attribution debt. | Recover the row-level verdicts or independently re-audit the affected ledger scope; resolve or explicitly qualify attribution before publication sign-off. Current disposition is in `manuscripts/README.md`. |
| Formal completeness | Deferred opportunity. Existing Lean build, fresh kernel replay and guarded axiom audit pass; formal markers and frozen statements unchanged. Positive tolerance, zero weights, D5 duplicate alternative, D7 equal-optimum half, D12 and the editable-import residue remain outside completion. | Reopen only by owner selection; no Python correctness claim follows from Lean. |
| Human acceptance and release | Release blocker. B/C/G reader gates, owner research freeze, manuscript publication and release authorization remain outstanding. | Actual human acceptance and explicit release actions; green automation cannot close these gates. |
| HEP template-likelihood programme | Deferred from v1.0. The separately scheduled P5 assessment follows this review. | One go/no-go assessment and finite proposal; no automatic implementation commitment. |

Validation passed: registry validation and nine registry tests (including proof DAG and index
freshness); 96 targeted research/Atlas/compilation/API/artifact tests; full Python suite **641 passed**;
float32 **5 passed**; Ruff, format, type checking, package build and strict MkDocs. One existing
NumPy empty-bin division warning remains in the uncertainty tests. Portal validation passed
**263 tests**, typecheck/lint and build; final browser suite **49 passed, 3 intentional mobile
skips**; assembled site verified **61 redirects and 54 reference links**. All 137 claim IDs,
statuses, proof locations and formal markers were compared with HEAD and preserved; only the
independently audited tolerance statement was narrowed. No numerical solver behaviour changed.

The initial full Python run had 640 passes and one public-prose failure (an internal ADR reference);
that reference was removed and the full rerun passed. Lean `lake build --wfail`,
`lake env leanchecker --fresh ScoreQuantFormal` and `lake env lean ScoreQuantFormal/AxiomAudit.lean`
passed. The guarded axiom allowlist is `propext`, `Classical.choice`, `Quot.sound`; formal statement
markers remain restricted to their frozen scope. The exploratory `leanchecker --help` invocation
was stopped after inspection showed the tool has no help mode; the explicit fresh replay above
is the completed trust check. No formal files changed. Human gates were not run by an agent and
remain open; no release action was taken.

Not needed for 1.0, by decision: samplers, streaming aggregation, moment-oracle evaluation, a
multivariate efficient-score certificate, generic profiled compilation, more solvers or criteria,
a compiled extension.

## Reader gate

Without author hints, ask a fresh reader to explain what is observed and estimated, what K
constrains, which labels can be deployed, what the reference model is, and what the reported
metric and comparison establish. Then reproduce the result and change the experiment control.
Record misunderstandings, not a review transcript. Agent review rehearses this gate; it does not
close it.

## Editorial rules

- README: purpose, install, one example, task choice, limitations, links.
- Documentation index: navigation. Articles: subject and sources before method or API.
- Keep assumptions needed to interpret a result; delete repeated motivation, internal process
  commentary, universal claims about a profession, and duplicate literature/dataset summaries.
- Keep proofs, source citations and reproducible evidence. Move detail once; link instead of
  retelling it. Preserve routes during edits; remove a duplicate page only after its content and
  inbound links are covered elsewhere.

## Session contract and deferrals

One owner, one bounded outcome, one handoff: verdict, evidence, checks, limitation, next action.
Derivation and independent promotion audit use separate contexts. No automatic follow-up tree
and no manuscript update after every result.

Parked: exact \(D_s\) bit complexity, broader calibration/refitting theory, new criteria or
backends, samplers, streaming, signed weights, universal bin-budget selection, generic profiled
compilation, the formal residue. The current API and shared numerical core remain intact until
phase H's audit.

## Delivered milestones

One row per milestone. Nothing here is a standing instruction; the durable decisions are the
entries named in the last column.

| Milestone | Delivered | Durable record |
| --- | --- | --- |
| M1 Canonical contracts and documentation | Consistent naming; population design, empirical quantizer fitting and finite assignment kept distinct; sources separate from providers and exact from surrogate information; the 1D-first book, glossary, bibliography, examples suite and the six-page FlowCyt study; the profiling campaign to \(N=10^6\). | ADR 0001–0010, 0016; `docs/book/`, `docs/usecases/flowcyt/`, `benchmarks/README.md` |
| M2 Exact finite D core | Exact cell statistics, rank-two relocation gain, deterministic monotone exchange, terminal scan, small-instance exhaustive oracle, explicit compilation. | ADR 0009, 0014 |
| M3 Task-explicit API | `optimize_partition`/`PartitionResult` and `fit_quantizer`/`QuantizerResult`; criterion plus solver configuration pairs; old fitting names removed without aliases. | ADR 0009, 0011, 0013 |
| M4 Sources and providers | `ScoreSample`, `ObservationSample`, bounded tensor-quadrature `IntegrationSource`; the provider protocol; named score schema and provenance. | ADR 0010, 0021, 0022 |
| M5 Book and FlowCyt capstone | The task-explicit 600,000-cell workflow with the exact-D reference on the frozen sample; exact rank-two state updates at one million rows; audited data reconstruction. | `docs/usecases/flowcyt/`, ADR 0015 |
| M6 Profiled \(D_s\) | Finite profiled exchange, the soft inductive solver, the efficient-score upper bound, the exact scalar interval dynamic program. | ADR 0014, 0015 |
| M7 Certificates, scale and persistence | Exchange-stability and geometry reports, branch-and-bound `certify_partition`, tolerance-consistent verification, the versioned `Quantizer` artifact. | ADR 0014, 0016, 0023 |
| M8 Density ratios | Exact densities, model density ratios and scores named as the representation layer; `DensityRatioScore`, `CentralLogRatioScore`, ratio provenance and the closure diagnostic. | ADR 0017 |
| M9 Explicit multi-backend execution | `ExecutionConfig`; JAX and NumPy behind one mathematical core; one conformance suite; the browser wheel. | ADR 0018 |
| M10 React portal and browser runtime | The Docusaurus shell, generated data contracts, the Pyodide runtime. | ADR 0031, 0035 |
| M11 First public releases | 0.1.0 on 30 August 2026 and 0.2.0 on 4 September 2026 through PyPI Trusted Publishing, gated on the full handoff gate. | `CHANGELOG.md`, `.github/workflows/release.yml` |
| M12 Consolidation programme | Novelty ledger and manuscript v9; the error hierarchy and one fit pipeline; the HEP classifier showcase; the four walkthroughs; the portal launch. Closed 4 September 2026. | ADR 0024–0027, `CHANGELOG.md` |

## Standing checks

Published code executes; reported numbers trace to committed evidence. Certificates and geometry
state their tolerances. Compilation behaviour, tests and documentation agree with the audited
guarantees for exact and positive-tolerance label agreement. Research
fixtures and registry remain valid; backend conformance remains the numerical contract. The
handoff gate is the command block in `docs/development.md`; portal changes add
`corepack pnpm validate`, `test:e2e` and `assemble:site`. Green automation is not reader
acceptance.
