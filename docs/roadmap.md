# ScoreQuant development roadmap

The sole executable development plan. Research work selection is in
`agenticresearch/OPEN_PROBLEMS.md`; proof packets do not create library commitments. Dated
reviews are not kept as files; git history holds them, and a phase cites the pull request that
acted on one.

## M13 — Focused research and teaching, then the road to 1.0

**Current (10 September 2026):** the research closure programme is at step 5 (manuscript v10);
the human reader gates on the walkthroughs and the Research Atlas are still open; phase H names
what a 1.0 release needs. Two routes, `/` and `/docs/` (ADRs 0031, 0035); no new API, solver or
site stack before phase H's audit.

| Phase | Status and next action | Gate / stop |
| --- | --- | --- |
| A — Geometry and interpretation | Done: the false boundary overlay retired and the optical, periodicity, compilation and information claims corrected (PRs #52, #56). | Equations, chart labels and prose agree; facts/snippets and frontend checks pass. |
| B — Michelson exemplar | Revised 6 September 2026 (PR #56) as a two-act article: instrument → model and analytic score → D-optimal quantization and its compiled rule → nuisance profiling and the profiled \(D_s\) partition → the reusable profiled rule → one experiment. **Human reader gate open.** | A fresh reader explains the physical quantity, nuisance, allowed labels and finite-table certificate. |
| C — Remaining articles | Ratios rewritten 7 September 2026 (PR #62); FlowCyt and HEP rebuilt as held-out-evidence walkthroughs 7 September 2026 (PR #63, ADR 0038). **Human reader gate open.** | Each starts with subject, source and an explanatory figure; one result comparison; one experiment unless the page's argument does not admit one (ratios, by ADR 0035). Facts/snippets, desktop/mobile e2e and links pass. |
| D — Frozen-rule uncertainty | Done: scalar O6 (PR #53), vector O7 (PR #59), score-error budget O8 (PR #65), all independently audited; `retention_uncertainty` shipped 8 September 2026 (PR #66, ADR 0039). The manuscript harvest is closure step 5. | Public estimator under the full contributor gate with seeded coverage and heavy-tail tests. Weighted, refitted, profiled and truth-free uncertainty stay backlog. |
| E — Formal verification | Parked as partial (PR #64): D2–D8 and the compiled predictor are machine-checked under ADR 0037; the residue (positive tolerance, zero weights, D5 duplicate branch, D7 equal-optimum half, D12, one frozen-file import) is listed in `agenticresearch/OPEN_PROBLEMS.md`. | Build, statement correspondence, allowed axioms and checker pass. Reopened only by owner decision. |
| F — Claim graph | Deferred. Separate proof prerequisites from audit/evidence links, preserving ids and statuses. | Mathematical DAG, lookup, generated indexes and fixture checks pass; no bulk proof rewrite. |
| G — Research Atlas | Editorial redesign delivered (PR #57, ADR 0036). **Human acceptance open.** | A fresh researcher identifies the contribution, boundary and open question, then finds theorem and prior art. |
| H — The 1.0 gate | Not started. In order: (1) research closed — closure step 5 merged and the owner's step 6 freeze; (2) 0.3.0 tagged from the Unreleased changelog; (3) one API-surface audit session sorting the public names into stable results, opt-in diagnostics and internals, removing `quantizers.py`, and recording the sort as a table in `docs/api.md`; (4) a committed format-1 `Quantizer` artifact fixture with a load-and-predict test; (5) the classifier moved to Beta at 0.3 and to Production/Stable at 1.0; (6) the human gates of B, C and G accepted and the manuscript public so the docs can cite it. Policy: ADR 0040. | Handoff gate green; no `src/` capability added; every public name has a docstring, an `api.md` row and a stability class. |

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
state their tolerances. A compiled rule reproduces positive-weight training labels. Research
fixtures and registry remain valid; backend conformance remains the numerical contract. The
handoff gate is the command block in `docs/development.md`; portal changes add
`corepack pnpm validate`, `test:e2e` and `assemble:site`. Green automation is not reader
acceptance.
