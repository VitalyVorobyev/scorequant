# ScoreQuant development roadmap

The sole executable development plan. Research is frozen (ADR 0043);
`agenticresearch/OPEN_PROBLEMS.md` holds the backlog and selects nothing. Dated
reviews are not kept as files; git history holds them, and a phase cites the pull request that
acted on one.

## M14 — Software, demonstrations, documentation

**Current (18 October 2026):** research is frozen and releases are decoupled from publication
(ADR 0043). The library's engineering gate is green; the next action is the owner's 0.3.0 tag.
Items come from the October 2026 code review; each is one bounded change with tests.

| Item | Scope | Gate |
| --- | --- | --- |
| 0.3.0 release | Bump to 0.3.0, Beta classifier, `CHANGELOG.md`, `prepare:runtime`; tag by the owner. | Handoff gate; local wheel rehearsal in `docs/development.md`. |
| Precision | Make `gain_tolerance` and rank thresholds scale with dtype so float32 runs are meaningful (the fallback now only warns). | float32 leg covers exchange stability and refusals at larger N. |
| One singularity rule | Replace `slogdet`-sign and NumPy-default rank tests in `information.py` and `solvers/soft.py` with the shared rank threshold; rebuild exact cell statistics before reporting `objective`. | Backend conformance unchanged; degenerate labellings report rather than raise. |
| Memory at scale | Chunk the `[N, B, P]` profiled geometry (`partition.py`) and soft-solver tensors; compile the soft gradient; raise when the soft solver stalls on a singular Fisher matrix. | A 10⁶-row profiled fit runs in bounded memory. |
| `partition.py` split | Move exchange and Lloyd into `solvers/`, the exchange algebra into a private module and the geometry reports into diagnostics; merge the three nearest-centre assignment routines and the repeated nuisance validation. | `test_architecture` layering; no behaviour change. |
| API before 1.0 | Trim the 62 exported names: one route each for linear models and ratios, ratio/provider stack to a submodule, `plot_*` non-stable, one naming for iteration caps and tolerances, remove `ScalarDPConfig.seed`. | `docs/api.md` inventory and ADR 0040 updated in the same change. |
| Demonstrations and docs | One home per walkthrough (portal); fold `three-doors.md` into the book; a HEP template-fit demonstration if P5 is reopened as an application. | Facts/snippets and e2e pass. |
| Test tiers and CI | Move portal/Atlas/research consistency tests out of the library tier; path-filter the site workflow. | CI time per PR falls; nothing stops being run on `main`. |

**Publication and 1.0** stay gated as before: manuscript assurance (`agenticresearch/manuscripts/README.md`),
the B/C/G reader gates, an owner freeze of the manuscript and an authorized 1.0 tag with the
Production/Stable classifier.

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
compilation, the formal residue.

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
| M13 Focused research and teaching | Michelson, ratio, FlowCyt and HEP walkthroughs rebuilt; frozen-rule uncertainty O6–O8 audited and `retention_uncertainty` shipped; D2–D8 machine-checked; claim graph with separate proof and evidence relations; Research Atlas; manuscript v10; the 1.0 engineering contract (API inventory, v0.2.0 artifact fixture). Closed 18 October 2026 with the research freeze. | ADR 0036–0043, `agenticresearch/manuscripts/README.md` |

## Standing checks

Published code executes; reported numbers trace to committed evidence. Certificates and geometry
state their tolerances. Compilation behaviour, tests and documentation agree with the audited
guarantees for exact and positive-tolerance label agreement. Research
fixtures and registry remain valid; backend conformance remains the numerical contract. The
handoff gate is the command block in `docs/development.md`; portal changes add
`corepack pnpm validate`, `test:e2e` and `assemble:site`. Green automation is not reader
acceptance.
