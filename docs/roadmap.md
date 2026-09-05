# ScoreQuant development roadmap

The sole executable development plan. Research work selection is in
`agenticresearch/OPEN_PROBLEMS.md`; proof packets do not create library commitments.

## M13 — Focused research and teaching

**Current:** PR #52 reduces the portal and revises Michelson; reader acceptance is still pending.
[The 5 September review](programme/2026-09-05-user-docs-review.md) corrected factual
overstatements, shortened entry documents, and its remaining editorial remarks (the classifier
ratio statement, the FlowCyt and HEP openings, attribution narration, entry-page overlap, the
walkthrough cards) are addressed in the same PR. Do not replicate the article form until the
revised Michelson explanation passes review. Routes stay `/`, `/docs/`, `/portal/`
(ADRs 0027, 0031); no new API, solver or site stack.

| Phase | Status and next action | Gate / stop |
| --- | --- | --- |
| A — Geometry and interpretation | Geometry overlay retired in PR #52. This review fixes remaining optical, periodicity, compilation and information claims. | Equations, chart labels and prose agree; facts/snippets and frontend checks pass. |
| B — Michelson exemplar | Implemented, reader gate pending. Instrument and sources → model and question → admissible readout → fit and comparison → one experiment. | A fresh reader explains the physical quantity, nuisance, allowed labels and finite-table certificate. Resolve misunderstandings before C. |
| C — Remaining articles | Openings corrected in PR #52: ratios states the mixture and the posterior-odds ratio formula first; FlowCyt starts from this dataset's cells, labels and patient fractions; HEP is stated as a simulation study. The full article form is queued after B acceptance. | Each starts with subject, source and an explanatory figure. One result comparison and one useful interaction; move exhaustive evidence to reference pages. Existing facts/snippets, desktop/mobile e2e and links pass. |
| D — Frozen-rule uncertainty | Next research session, independent of B/C: `agenticresearch/WORK/active/SCORE-ORACLE-ROBUSTNESS.md`. | One conditional scalar retention interval result or precise obstruction, plus one seeded experiment. Independent audit before promotion. No public estimator yet. |
| E — Formal verification | PR #28 superseded: its Lean tree ported onto main, its integration re-authored, ADR numbering reconciled to 0030. Scope widened by that ADR from the scalar core to the finite D chain in general dimension — D2, D3, D4, D5 with its quantitative bound, and the D7/D8 corollaries — with an independent statement audit on D5. | Build, statement correspondence, allowed axioms and checker pass. Profiled `D_s`, population measure theory and asymptotics need a new ADR. |
| F — Claim graph | Deferred until D handoff and E metadata are settled. Separate proof prerequisites from audit/evidence links, preserving IDs and statuses. | Review moved edges; mathematical DAG, lookup, generated indexes and fixture checks pass. No bulk proof rewrite. |

## Reader gate

Without author hints, ask a fresh reader to explain what is observed and estimated, what K
constrains, which labels can be deployed, what the reference model is, and what the reported
metric and comparison establish. Then reproduce the result and change the experiment control.
Record misunderstandings, not a long review transcript. Agent review rehearses this gate;
actual reader feedback remains pending until obtained.

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
Derivation and independent promotion audit use separate contexts. No automatic follow-up tree,
model tiers or manuscript update after every result.

Park exact Ds bit complexity, broader calibration/refitting theory, new criteria/backends,
samplers, streaming, signed weights and universal bin-budget selection. Generic profiled
compilation remains unsupported. Formal expansion and public uncertainty APIs need new evidence.
The current API and shared numerical core remain intact.

## Delivered milestones

One row per milestone. Nothing here is a standing instruction; the durable decisions are the
ADRs named in the last column.

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
| M10 React portal and browser Lab | The Docusaurus shell, generated data contracts, the Pyodide Lab, the development blog. | ADR 0019, 0020 |
| M11 First public releases | 0.1.0 on 30 August 2026 and 0.2.0 on 4 September 2026 through PyPI Trusted Publishing, gated on the full handoff gate. | `CHANGELOG.md`, `.github/workflows/release.yml` |
| M12 Consolidation programme | Novelty ledger and manuscript v9; the error hierarchy and one fit pipeline; the HEP classifier showcase; the four walkthroughs; the portal launch, whose root placement was then reversed. Closed 4 September 2026. | ADR 0024–0027, `CHANGELOG.md` |

## Standing checks

Published code executes; reported numbers trace to committed evidence. Certificates and geometry
state their tolerances. A compiled rule reproduces positive-weight training labels. Research
fixtures and registry remain valid; backend conformance remains the numerical contract.
For portal changes also run pinned-toolchain `pnpm validate`, `pnpm test:e2e` and
`pnpm assemble:site`. Do not treat green automation as reader acceptance.

## Full handoff gate

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check src
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto
JAX_ENABLE_X64=0 MPLBACKEND=Agg uv run pytest tests/test_float32.py
uv build
uv run mkdocs build --strict
```
