# Formal statement audit — D-EXCHANGE-IMPLIES-VORONOI (001)

**Audited object:** `formal/ScoreQuantFormal/ExchangeVoronoiSpec.lean`, the frozen
statement boundary for `D-EXCHANGE-IMPLIES-VORONOI` (manuscript v9 Theorem 2) and
`D-EXCHANGE-VIOLATION-LOWER-BOUND`.

**Canonical source:** `claims/D-EXCHANGE-IMPLIES-VORONOI.json`,
`claims/D-EXCHANGE-VIOLATION-LOWER-BOUND.json`,
`KNOWN_RESULTS/04-d-optimality.md` §D5, `AUDITS/AUDIT-D-EXCHANGE-VORONOI-001.md`.

**Date:** 5 September 2026. **Verdict: match after hardening.** No blocking
finding; every hardening below has been applied and the spec re-frozen.

**Independence.** The audit ran in a session with no shared derivation context:
it was given the claim nodes, the prose proof, the prior D5 audit, the boundary
counterexample and the Lean files, and no part of the formalizing session's
transcript.

## Correspondence

| # | Registry clause | Lean | Status |
|---|---|---|---|
| H1 | strictly positive atom weights | `Sample.weight_pos` | carried by the type, so a zero-weight row is inexpressible |
| H2 | coincident rows merged into distinct atoms | `VoronoiAssumptions`, conjunct 1 | frozen |
| H3 | exactly `K` nonempty cells | `VoronoiAssumptions`, conjunct 2 | frozen |
| H4 | `I_q ≻ 0` | `VoronoiAssumptions`, conjunct 3 | frozen |
| H5 | no move constraint beyond preserving nonempty cells | `Admissible` | frozen |
| H6 | exact zero-gain-tolerance stability | `ExchangeStable` | frozen, deliberately on `det` (see F2) |
| C1 | tied-or-worse comparison has strictly positive exact gain | `ViolationLowerBound`, `ViolationStrictGain` | frozen |
| C2 | stability forces distinct centroids | `centroid_ne_of_stable` (proof module) | derived, not assumed |
| C3 | singleton rows strictly nearest their own centroid | subsumed by `StrictVoronoi` | frozen |
| C4 | strict self-consistent D-Voronoi | `StrictVoronoi` | frozen |
| — | the implication itself | `ExchangeVoronoiConclusion` | frozen |

Nothing appears formally that is absent informally.

## Findings and dispositions

**F1 — the frozen boundary did not contain the theorem (hardening; fixed).**
The audited version froze `Admissible`, `ExchangeStable`, `StrictVoronoi` and
`ViolationLowerBound`, but left the three standing hypotheses and the arrow
between hypotheses and conclusion in docstring prose and in the *editable* proof
module. A prover could therefore have added a hypothesis — for instance the
distinct centroids that D5 makes a point of deriving — and weakened the theorem
without touching an audited file. Secondary effect: detached from H3,
`Admissible` ranges over relocations into empty cells and `StrictVoronoi`
compares against the `0` centroid of an empty cell.
*Fixed:* `VoronoiAssumptions`, `ExchangeVoronoiConclusion` and
`ViolationStrictGain` were added to the spec, mirroring the `ScalarExchangeSpec`
pattern, and the exported theorems `exchange_voronoi`, `violation_lower_bound`
and `violation_strict_gain` now state those frozen Props.

**F2 — "the two objectives agree" was false (hardening; fixed).**
`ExchangeStableLogDet → ExchangeStable` holds under `det I > 0`, but the
converse fails: the auditor exhibited `d = K = 2`, `N = 3`, scores `(0,1)`,
`(1,0)`, `(0,-1)`, unit weights, `z = (0,0,1)`, where `I` is positive definite
with `det I = 1/2`, the scores are injective, both cells are nonempty,
`ExchangeStable` holds, and `ExchangeStableLogDet` fails on the move whose
candidate determinant is `0`, because `Real.log 0 = 0 > log(1/2)`.
The *direction chosen was the sound one* — the determinant form is the weaker
hypothesis, hence the stronger theorem — but the stated justification was wrong,
and a reader who believed it might have "simplified" the spec to the log form,
which would be an unsound weakening. *Fixed:* the docstring now states the
one-directional relation and why the weaker hypothesis is deliberate.

**F3 — the non-coverage list was incomplete (hardening; fixed).** Now also
disclaims: the registry's second duplicate branch ("labels constant on every
duplicate class", which `Function.Injective` cannot express and which the claim
asserts as *true*); positive gain tolerances; capacity, balance and
minimum-mass constraints; the population/atomless statement; score-estimation
error and the observation-to-score step; and the information-loss consequences.
D7 and D8 were moved from "not covered" to covered, since `Corollaries.lean`
proves them.

**F4 — the strict half of the quantitative claim was not frozen (cosmetic;
fixed).** `D-EXCHANGE-VIOLATION-LOWER-BOUND` asserts `ΔF_D ≥ log(1+·) > 0`; the
audited spec froze only the `≤` form, with strictness derived inline inside the
main theorem. *Fixed:* `ViolationStrictGain` freezes the strict half, and
`violation_strict_gain` proves it; the main theorem now calls it rather than
re-deriving the arithmetic.

**F5 — no automatic check (hardening; fixed).** `.github/workflows/` contained
no Lean job at all, although `formal/README.md` asserted a CI trust gate. Also
noted: `lake_lib` builds only the root module and its import closure, so the new
modules were outside the default target until `AxiomAudit.lean` was extended to
import them. *Fixed:* `AxiomAudit.lean` imports `Corollaries` and guards every
exported theorem, so the whole chain is in the default target; and a `formal`
job running `lake build --wfail`, `leanchecker` and a namespace-wide axiom audit
was added to the PR and release workflows.

## Independently confirmed correct

- **The Fisher matrix.** `cellBlock W T = W⁻¹ • (T Tᵀ)` with the unnormalized
  `T_c = W_c μ_c` is exactly `W_c μ_c μ_cᵀ`, so `fisher = Σ_c W_c μ_c μ_cᵀ`.
  Empty cells contribute `0` under Lean's `(0:ℝ)⁻¹ = 0`, matching the informal
  convention of summing over occupied cells. Checked against a direct
  implementation on 2,965 configurations in exact rational arithmetic, zero
  mismatches.
- **Unnormalized weights are a generalization, not a defect.** The statement is
  invariant under `w ↦ t·w`.
- **`Admissible`** is exactly `|cell| ≥ 2` plus a different destination, and
  imposes nothing else. Omitting a destination-nonempty clause is sound: under
  H3 every cell is nonempty, and the omission only enlarges the set of moves
  stability must survive.
- **`StrictVoronoi`** has the right metric, strictness and quantifier range, and
  its junk behaviour is safe: at a singular `fisher`, `Matrix.inv` returns `0`
  and the Prop becomes false rather than vacuously true.
- **Injectivity is the right reading of "merged", and is necessary.** The
  auditor reconstructed `CE-D-UNMERGED-DUPLICATES-001` inside the Lean encoding:
  every hypothesis except injectivity holds and the conclusion fails, so
  dropping it makes the frozen statement false. It is used only in the
  both-singletons branch — the same place the prose uses it.
- **Degenerate dimensions.** `d = 0` forces `N ≤ 1` through injectivity, hence
  `K ≤ 1`, making the conclusion vacuous; `K = 0` forces `N = 0`.
- **The quantitative bound is faithful.** `centroidSeparation` is `q_δ`,
  verified as `q_aa + q_bb − 2q_ab` on all 2,965 moves; `alpha` and `beta` carry
  source and destination masses in the right slots; the determinant-ratio
  phrasing is equivalent to the `ΔF_D` phrasing under `PosDef`.
- **The tied-or-worse orientation is load-bearing and correct.** The bound holds
  on 761/761 tied-or-worse moves and fails on 1,759 of 2,296 strictly correctly
  assigned ones, so a flipped inequality would have produced a false statement.
- **The supporting statements** D2, D3 and D4 were each verified exactly on the
  same 2,965 moves, and `ScalarExchangeAssumptions` was walked conjunct by
  conjunct against what the matrix layer supplies.

## What remains uncovered

The frozen statement concerns one fixed finite labeling of finitely many
distinct, strictly-positive-weight atoms in exact real arithmetic, with the
retained information already known to be nonsingular. It says nothing about
positive gain tolerances — the regime every real solver is in — nor about
singular or pseudodeterminant objectives, capacity or balance constraints,
atomless population limits, score-estimation error, or any information-loss
guarantee. It is one-directional: the converse `D-VORONOI-NOT-EXCHANGE` and the
compiled predictor `D-FINITE-INDUCTIVE-CLOSURE` are not formalized. And it is a
statement about mathematics: nothing connects `fisher`, `centroid` or `relocate`
to the library's implementations of them, nor to floating-point conditioning.
