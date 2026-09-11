# ScoreQuant research workspace

**Version:** 5.1 · 11 September 2026. This file is the whole operating contract for research
sessions; everything else here is either scientific memory or a protocol read on demand.

The subject is D- and \(D_s\)-optimal hard quantization of multivariate score space
(definitions and scope: `PROBLEM.md`). The programme is finite and its end is declared in
`OPEN_PROBLEMS.md`. Results are recorded once, in the claim graph; every other file either
points at a claim or is history in git.

## What a session reads

1. This file.
2. `OPEN_PROBLEMS.md` — the closure programme, the status line and the backlog. It alone selects work.
3. The packet in `WORK/active/` if one is named, else the `WORK/TEMPLATE.md` shape for the step chosen.
4. The claims the packet names: `uv run python agenticresearch/py/registry.py show <ID> --deps --proof`.
5. The one protocol that applies: `protocols/theorem.md` (derive), `audit.md`, `literature.md`,
   `numerical.md`, `algorithm.md`, `formalization.md`.

Do not load `PROBLEM.md` beyond the section a definition needs, `NUMERICAL_EVIDENCE.md`,
`KNOWN_RESULTS/` chapters other than the cited section, `AUDITS/`, `LITERATURE/` prose, the
`WORK/artifacts/` JSON, or any manuscript body. They are evidence, reached through a claim.

## Map

| Path | Holds | Loaded |
|---|---|---|
| `claims/*.json`, `registry.json` | one node per claim: status, statement, assumptions, dependencies, proof pointer | by id |
| `KNOWN_RESULTS/` | proof prose behind `proof_location` | the cited section |
| `COUNTEREXAMPLES/` | exact fixtures, pinned by `tests/test_research_claims.py` | by id |
| `AUDITS/` | independent audit reports (16-item contract) and Lean statement audits | never in bulk |
| `LITERATURE/` | bibliography (generated), topics, per-claim prior-art audits, `gaps.md` | per packet |
| `NUMERICAL_EVIDENCE.md` | append-only measurement ledger; never theorem authority | append only |
| `WORK/active/`, `WORK/artifacts/` | the current packet; instrument outputs cited by claims or ledger rows | the packet |
| `py/` | `registry.py` and the exact-arithmetic instruments behind measured rows | by name |
| `formal/` | pinned Lean 4 + Mathlib workspace (`formal/README.md`) | formal packets |
| `manuscripts/` | the v10 article snapshot and its novelty ledger (`manuscripts/README.md`) | paper tasks |

Generated, never hand-edited: `claims/INDEX.md`, `COUNTEREXAMPLES/INDEX.md`,
`LITERATURE/BIBLIOGRAPHY.md`, `website/src/generated/atlas.json`.

## Claim graph relations

`dependencies` contains mathematical proof prerequisites only and must form a DAG.
`verified_by` links a claim to an audit record; `references` carries reviewed inputs or supporting
evidence without a deductive assertion. Audit records use references, not dependencies, for their
inputs. Measured evidence never supplies a theorem prerequisite. `implies` retains result and
question relationships; audit targets belong in `verified_by`. File-valued `audit` pointers and
formal-proof metadata remain unchanged. All relation targets must resolve, duplicate edges are
rejected, and `show --deps` traverses only mathematical prerequisites. The Atlas displays the
evidence relations separately. See ADR 0041.

## Invariants

1. Primary objectives are D and \(D_s\); trace, A and E are controls unless targeted.
2. Every claim names one problem level (finite assignment, empirical inductive quantizer,
   population quantizer) and never jumps between them silently.
3. Full D, in-bin \(D_s\) (Schur complement) and the projected efficient-score problem stay separate.
4. The decision variable is a hard partition of score space. Never substitute experimental
   design, subset selection, k-means, scalar thresholding or soft categorisation.
5. Falsify in exact arithmetic before proving; every counterexample becomes a fixture and a test.
6. `measured` is not proved; a search gap is not novelty; "optimal" always carries its qualifier.
7. For estimated scores, surrogate retention and true retained information
   \(\operatorname{Var}(E[s\mid q(\hat s)])\) are different quantities.
8. Two vocabularies: `status` in {literature, bridge, project_proved, counterexample, measured,
   conjecture, open}; `literature_search_status` in {not_searched, search_gap, prior_art_found}.
9. A `project_proved` node is never re-derived; doubt opens an audit that tries to falsify it.
10. A library guarantee or a publication claim needs an independent audit (`protocols/audit.md`).

## Running a session

- **One session, one step of the closure programme**, taken from `OPEN_PROBLEMS.md`. The step's
  verdict is proved, refuted or reduced; a reduction names the one missing statement and closes the
  step. No second attempt is scheduled by default.
- **Derivation stays in the main session** with the strongest model. Delegate only wide reading,
  literature fetching and bookkeeping, to lighter models; never spawn `fable` subagents. Long
  numerical runs go to the background and return one summary line plus a serialized artifact.
- **Cite before deriving.** When the literature pass finds the method, the write-up names the
  source and reduces it to the project's objects; only the reduction is project content.
- **Record once.** A result changes its claim node(s), the `KNOWN_RESULTS/` section the node
  points at, a fixture and test if exact, and a ledger row if measured. Nothing else restates it.
- **Close with** the packet's Outcome section, then delete the packet (git keeps it), one status
  line in `OPEN_PROBLEMS.md`, and a plain-English report to the owner.

Verification before handoff:

```bash
uv run python agenticresearch/py/registry.py reindex
uv run python agenticresearch/py/registry.py validate
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest tests/test_research_claims.py tests/test_research_registry.py
uv run python website/scripts/generate_atlas.py && JAX_ENABLE_X64=1 uv run pytest tests/test_atlas_data.py
(cd agenticresearch/formal && lake build --wfail)   # only when formal evidence changed
```

Pushes and merges need owner authorization. No `src/` change comes out of a research packet;
a shipped estimator is an engineering step of the programme with its own tests and docs.

## Session prompts

Paste one of these into a fresh session; the packet or claim carries the specifics.

```text
Execute the closure-programme step <N> of agenticresearch/OPEN_PROBLEMS.md in a worktree
branched from origin/main. Follow agenticresearch/README.md. Close with a plain-English report.
```

```text
Independent audit of <CLAIM-ID> (and companions <IDS>) per agenticresearch/protocols/audit.md.
The work under audit is branch <branch> at <commit> (PR #<N>): check it out as a new worktree
under .claude/worktrees/ and stay there. You have not seen the derivation; do not ask for it.
Deliverable: AUDITS/AUDIT-<NAME>-001.md, the audit pointer and hardened assumptions on the claim
nodes, fixtures for any boundary failure, then the verification block of
agenticresearch/README.md. Push to the same branch; do not merge.
```

```text
Literature pass for <CLAIM-ID> per agenticresearch/protocols/literature.md. Return the audit
file under LITERATURE/audits/, new bibliography keys, and a cite-versus-derive table.
```
