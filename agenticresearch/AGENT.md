# ScoreQuant theorem-research agent protocol

**Version:** 4.2 · 5 September 2026

You are a mathematical research agent working on **D- and \(D_s\)-optimal hard
quantization of multivariate score space**. This file holds only the
non-negotiable invariants and the map; detailed recipes live in `protocols/`
and are read when relevant, not up front.

> **Decompose knowledge finely; decompose work only at natural scientific
> boundaries.** Claims are atomic (`claims/`); work is not (`WORK/`).

## Map

| Need | Read |
|---|---|
| What problem are we solving | `PROBLEM.md` |
| What is established / open | `claims/` (graph, via `py/registry.py`) + `KNOWN_RESULTS/` (prose) |
| What falsifies naive generalizations | `COUNTEREXAMPLES/` |
| What to work on | `WORK/active/` packet, then `OPEN_PROBLEMS.md` |
| How to derive / audit / search / measure / build | `protocols/{theorem,audit,literature,numerical,algorithm}.md` |
| How to machine-check a frozen claim | `protocols/formalization.md`, `formal/README.md` |
| Prior art | `LITERATURE/`, `papers/` |
| Measured evidence (never theorem authority) | `NUMERICAL_EVIDENCE.md` |
| Paper snapshots (lagging; do not load bodies) | `manuscripts/README.md` |
| History | `archive/` — canonical files win on any conflict |

The single canonical read order is in `README.md`. Registry integrity is
CI-enforced by `tests/test_research_registry.py`, which runs
`py/registry.py validate`; claim fixtures by `tests/test_research_claims.py`.

## Non-negotiable invariants

1. Primary objectives are **D** and **\(D_s\)**; trace/A/E are controls unless
   explicitly targeted.
2. Every claim names one problem level — finite assignment, empirical
   inductive quantizer, population quantizer — and never silently jumps
   between them.
3. Keep full D, in-bin \(D_s\) (Schur complement, never the POI block alone),
   and the projected full-data efficient-score problem separate.
4. The decision variable is a hard score/score-proxy partition. Never silently
   substitute experimental design, subset selection, within-scatter
   determinant clustering, k-means, scalar thresholding, or soft neural
   categorization.
5. Falsify before proving; a found counterexample is minimized and serialized.
6. `measured` ≠ proved; a search gap ≠ novelty; never write "optimal" without
   its qualifier.
7. Every exact counterexample gets a permanent fixture; every production
   method explains assignment of unseen observations; every result reports
   information loss versus unbinned inference.
8. For estimated scores, distinguish surrogate optimization from true retained
   Fisher information \(I_{\text{true retained}}=\operatorname{Var}(E[s\mid q(\hat s)])\).
9. Two distinct vocabularies, never conflated. A claim's `status` is one of
   `literature`, `bridge`, `project_proved` (internally derived/audited, not
   published), `counterexample`, `measured`, `conjecture`, `open`. Its
   `literature_search_status` is one of `not_searched`, `search_gap`,
   `prior_art_found` — `search_gap` is never a claim status.
10. Do not re-derive a `project_proved` node. If you believe one is wrong,
    open an audit task and try to falsify it; never silently downgrade or
    overwrite it.

## Canonical objects

\[
s(x)=\nabla_\theta\log p(x\mid\theta)|_{\theta_0},\qquad
q:\mathbb R^d\to[K],\qquad
I_q=\sum_bW_b\mu_b\mu_b^\top
\]

\[
\Phi_D=\log\det I_q,\qquad
\Phi_{D_s}=\log\det\bigl(I_{\psi\psi}-I_{\psi\lambda}I_{\lambda\lambda}^{-1}I_{\lambda\psi}\bigr)
\]

\[
\eta_D=(\det I_q/\det I_{\rm full})^{1/d},\qquad
\eta_{D_s}=(\det S_\psi(I_q)/\det S_\psi(I_{\rm full}))^{1/s}
\]

Score-oracle regimes: direct scores; exact/autodiff score; analytic density
ratio; component ratios; learned density-ratio estimator; calibrated
classifier posterior/ratio proxy. Always identify which one applies.

## Claim lookup protocol

Never read `claims/` linearly; it is a theorem dependency graph.

```bash
python py/registry.py show <CLAIM-ID> --deps --proof   # node + closure + prose
python py/registry.py validate                         # before you finish
python py/registry.py reindex                          # after you patch a node
```

1. locate the target node by `id`, or browse `claims/INDEX.md` (generated,
   grouped by programme in queue order) to pick work;
2. `show --deps` expands `dependencies` recursively for you;
3. inspect dependencies with status `project_proved`, `counterexample`,
   `conjecture`, or `open`;
4. `--proof` prints each node's `proof_location` section;
5. check `converse_failures` and `counterexamples` before proposing a stronger
   statement;
6. patch the node's file in `claims/`, then `reindex` and `validate`.

Never hand-edit a generated index (`claims/INDEX.md`,
`COUNTEREXAMPLES/INDEX.md`, `LITERATURE/BIBLIOGRAPHY.md`).

## Unit of work

A session executes one `WORK/active/` packet: **one substantial scientific
question that can change our understanding of the project** — not one claim
node. The agent may create as many internal subproblems as it needs; they do
not become project tasks. Split work only at natural boundaries: parallelable
independent parts, different tool/context needs, a verification boundary
(proof done → independent audit), a genuine scientific branch, or an
oversized context.

## Session and delegation policy (harness-neutral)

Applies whether the operator is Claude Code, Codex, or any other agent
harness; the workspace files, not the harness, are the contract.

- **Derivation is never delegated.** The session that owns the packet does the
  mathematics with the strongest available model/reasoning setting. If the
  harness cannot delegate, everything below simply runs sequentially in the
  main session — the protocol degrades gracefully.
- **Wide reading is delegated when possible.** Summarizing a claim branch,
  scanning literature, or surveying code is handed to subordinate
  agents/sessions that return distilled reports, keeping the research context
  lean. Never load `manuscripts/` article bodies or bulk PDFs into the
  research context.
- **Exhaustive numerical searches run detached** (background task or separate
  session) and report only the summary line plus the serialized artifact —
  never raw enumeration output.
- **Formal evidence is subordinate to this registry.** `formal_proof` marks a claim whose
  statement is separately frozen in a `*Spec.lean` and independently audited; partial or
  unfrozen Lean coverage is recorded in `KNOWN_RESULTS/` prose instead. A Lean build
  certifies a statement, never the Python/JAX implementation of it (ADR 0030).
- **Audits require independence** (`protocols/audit.md`): a fresh session with
  no shared derivation context, only the packet, the registry, and the proof
  artifact. Researcher and auditor must not be the same context.
- **Bookkeeping may run cheap.** Registry patches, index regeneration, fixture
  serialization, and doc edits can use a lighter model/session; the registry
  validator test catches mechanical slips.
- **Escalation ladder:** exploratory lemma — one session; promising theorem —
  researcher + one independent auditor; publication-critical claim —
  researcher + adversarial auditor + independent prior-art search.
