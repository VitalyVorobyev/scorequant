# FORMAL-D-CLOSURE — machine-check the compiled D predictor and the exchange-stable remainder

**Programme:** formal verification (roadmap phase E) · **Opened:** 7 September 2026 · **Status:** active

## Goal

Close the finite D chain in Lean, inside the scope ADR 0030 already approves. Concretely:
prove D6 — that an exact zero-tolerance exchange-stable positive-definite D solution compiles
to the Mahalanobis argmin rule, reproducing every merged-atom training label strictly and
without a tie breaker, with original duplicate rows inheriting the merged label — and then the
remainder of the chain that the current freeze leaves out.

"Done" is decidable: `lake build --wfail` green, `#print axioms` inside the three-axiom
allowlist, and each frozen conclusion discharged by a theorem whose *type* is that conclusion.

## Why it matters

`compile_quantizer()` is the library's one theorem-backed capability, and `api.py:341`,
`result.py:304` and `reports.py:156-206` all cite "Theorem 3" for it. Theorem 3 is D5 *plus*
D6. D5 is machine-checked; D6 is explicitly listed as not covered
(`ExchangeVoronoiSpec.lean:64`). The deployable half of the flagship result is the unverified
half.

## Relevant claims

- `D-FINITE-INDUCTIVE-CLOSURE` (D6) — the target.
- `D-EXCHANGE-IMPLIES-VORONOI` (D5) — the dependency, already proved as
  `ScoreQuantFormal.exchange_voronoi`; also the source of the duplicate-branch gap below.
- `D-GLOBAL-GEOMETRIC-REALIZABILITY` (D7), `D-EXCHANGE-TERMINATES` (D8) — partly covered,
  deliberately unmarked; this packet closes the missing halves.
- `D-BB-SINGLETON-BOUND` (D12), `DS-EXCHANGE-TERMINATES`, `A-EXCHANGE-TERMINATES`.

## Step A — selection and normalization

The D6 node bundles three assertions. Protocol step A requires formalizing all of it or
splitting the node. **Decision: split.**

| Part of the registered `statement` | Disposition |
|---|---|
| (a) compiles to `argmin_b (s−μ_b)ᵀÎ⁻¹(s−μ_b)`, reproducing all merged-atom training labels strictly, no tie breaker | formalized here, `ClosureSpec.lean` |
| (b) original duplicate rows inherit the merged atom's label | formalized here, needs the merge-invariance lemma in `MergeSpec.lean` |
| (c) "positive solver tolerance gives only a tolerance-stamped boundary-disagreement guarantee" | **split out** into a new node; it is a different theorem about a weaker hypothesis, and it is the regime every real solver occupies |

Part (b) is not decorative bookkeeping: it only means something once merging is proved to
preserve `cellMass`, `cellSum`, `centroid` and `fisher`. That invariance lemma is also what
D5's unformalized second duplicate branch ("labels constant on every duplicate class") needs,
so the two are one piece of work.

### What this formalization will NOT cover

Written before any Lean, per protocol step B.

- Positive gain tolerances — split out as above.
- Zero-weight rows; capacity, balance or minimum-cell-mass constraints.
- Singular or pseudodeterminant objectives, and projected-subspace variants.
- Any statement about `predict_scores`' lowest-index tie-break, which exists precisely because
  the exact theorem's uniqueness does not survive a positive tolerance.
- Any population, atomless or asymptotic statement.
- Any claim about the Python/JAX implementation: nothing here connects `fisher`, `centroid`
  or the compiled rule to the library's versions of them, nor to floating-point conditioning,
  `rank_rtol`, or the `Quantizer` artifact.

## Known blockers

- The freeze is closed under definitional dependency, so anything D6's conclusion mentions
  must live in a frozen file. The merge relation and the argmin predicate therefore go into
  frozen specs, not into the proof module.
- `Sample.weight_pos` makes a zero-weight row inexpressible. In scope for this packet only as
  a recorded limitation; relaxing it is a separate re-audit of `ConfigSpec.lean`.
- `tests/test_research_registry.py:143-145` currently *asserts* D7 and D8 stay unmarked.
  Closing their missing halves flips that test.

## Recommended starting points

- `ExchangeVoronoiSpec.lean` as the model for a frozen spec that freezes the arrow, not just
  the conclusion; `Corollaries.lean` for how cheap a corollary becomes once D5 is available.
- `StrictVoronoi` already says every row is strictly nearest its own centroid, which is
  uniqueness of the argmin — (a) should be short.
- `Relocation.lean:40-63` for the `cell`-rewriting style the merge lemma will need.

## Required deliverables

Frozen `MergeSpec.lean` and `ClosureSpec.lean`; proof modules; `#guard_msgs` axiom entries;
an independent statement audit for D6 (`AUDITS/FORMALIZATION-D-FINITE-INDUCTIVE-CLOSURE-001.md`);
registry patches including the new tolerance node; `KNOWN_RESULTS/04-d-optimality.md` updated
where coverage is partial; the verification block of `agenticresearch/README.md`.

## Stop conditions

Proved, or reduced with the missing statement named. A mismatch verdict from the statement
audit blocks proof work and returns the packet to step A.

## Next dependency-blocking question

Does merge-invariance of `(cellMass, cellSum, fisher)` hold under the hypothesis D5 actually
needs — labels constant on each duplicate class — or does it need the stronger
`Function.Injective S.score` the current freeze assumes? Claim id:
`D-EXCHANGE-IMPLIES-VORONOI`.

## Outcome (7 September 2026) — partial; packet stays open

**Verdict: proved**, for D6 and D8. Three claims now carry `formal_proof`:

| Claim | Declaration |
|---|---|
| `D-FINITE-INDUCTIVE-CLOSURE` | `ScoreQuantFormal.closure_reproduces_labels` |
| `D-CLOSURE-DUPLICATE-INHERITANCE` | `ScoreQuantFormal.closure_duplicates` |
| `D-EXCHANGE-TERMINATES` | `ScoreQuantFormal.d_exchange_terminates` |

Two new frozen boundaries (`MergeSpec`, `ClosureSpec`, `TerminationSpec`), three proof modules,
14 new axiom-guarded declarations. `lake build --wfail` green; full contributor gate green.

The D6 node was split as protocol step A requires: `D-CLOSURE-DUPLICATE-INHERITANCE` carries
the duplicate half, `D-COMPILE-TOLERANCE-GUARANTEE` the positive-tolerance clause that nothing
formalizes.

### Audits

Three rounds, all finding defects. `FORMALIZATION-D-FINITE-INDUCTIVE-CLOSURE-001`
(`match after hardening`, 4 blocking), `FORMALIZATION-D-EXCHANGE-TERMINATES-001`
(`match after hardening`, 2 required), `-002` fix-verification (`fixes verified with residue`).
The record is now nine audits, nine changed statements.

The two that mattered most:

- **A load-bearing hypothesis was in the Lean and nowhere else.** `∀ c, (cell z c).Nonempty` was
  absent from D6's registry statement and prose. Dropping it makes the theorem false — an empty
  cell has the degenerate centroid `0`, and a row at the origin ties with it. Fixed on the
  registry side, never by weakening the Lean.
- **ADR 0030 did not cover D6.** Its enumeration stops at the D7/D8 corollaries. ADR 0037 now
  sets the scope to the finite theory, puts population and asymptotics permanently out, and adds
  the rule both audits asked for: frozen files import only frozen files.

### Still open in this packet

- Target 2, the ε-tolerance statement (`D-COMPILE-TOLERANCE-GUARANTEE`); §D6 now carries its
  two-line derivation but nothing formalizes it.
- Target 3, zero-weight rows: `Sample.weight_pos` still makes them inexpressible.
- Target 4, D5's second duplicate branch. **Merge invariance did not close this** —
  `ClosureDuplicateConclusion` assumes the inherited labeling rather than deriving constancy.
- Target 5, D7's equal-optimum-value half.
- Target 8, D12's singleton-refinement bound.

### Owed elsewhere

- `ExchangeVoronoiSpec.lean` imports the editable `Leverage.lean` — the one remaining freeze
  leak, flagged by both audits. Fixing it edits a frozen file, so it needs its own audit.
- `D-COMPILE-TOLERANCE-GUARANTEE` is `project_proved` on a two-line derivation; audit 002 notes
  its statement is close to `compile_quantizer`'s docstring contract and may want a different
  status.
- `FORMALIZATION-D-LEVERAGE-001`'s hardenings are still unapplied while `D-LEVERAGE` is marked;
  `FORMALIZATION-D-LOGDET-GAIN-001` is superseded but still present.

## Next dependency-blocking question

Does the ε-tolerance statement admit a frozen conclusion that a real solver can discharge —
"no geometric disagreement has exact gain above ε" — or does the degradation of strictness make
the deployable claim weaker than `GeometryReport` currently implies? Claim id:
`D-COMPILE-TOLERANCE-GUARANTEE`.
