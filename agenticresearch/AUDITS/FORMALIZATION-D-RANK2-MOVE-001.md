# Formal statement audit — D-RANK2-MOVE (001)

**Audited object:** `formal/ScoreQuantFormal/RelocationSpec.lean`, the frozen
statement boundary for `D-RANK2-MOVE`, together with its frozen definitional
dependencies in `ConfigSpec.lean` and `ScalarExchangeSpec.lean`.

**Canonical source:** `claims/D-RANK2-MOVE.json`,
`KNOWN_RESULTS/04-d-optimality.md` §D2, `AUDITS/AUDIT-D-EXCHANGE-VORONOI-001.md`
§7.

**Date:** 6 September 2026. **Verdict: match after hardening.** The frozen Lean
statement needs no change. One hardening is owed on the claim node, which lists
two of the four hypotheses the Lean carries; until it is applied, the registry
mark would advertise a slightly wider theorem than Lean proves.

**Independence.** The audit ran in a session with no shared derivation context:
it was given the claim node, the prose proof, the prior D5 algebra audit, the
boundary counterexamples and the Lean files, and no part of the formalizing
session's transcript. No Lean file, claim node or other workspace file was
modified.

## Informal source

D2 moves one weighted point \((s,w)\) out of a non-singleton source cell \(a\)
into a destination cell \(b\), and asserts the exact identity

\[
\Delta I=\alpha u_au_a^\top-\beta u_bu_b^\top,\qquad
u_c=s-\mu_c,\qquad
\alpha=\frac{wW_a}{W_a-w},\qquad
\beta=\frac{wW_b}{W_b+w},
\]

with \(W_c,\mu_c\) the cell mass and centroid **before** the move and
\(\Delta I=I_{\text{after}}-I_{\text{before}}\). The retained information is
\(I_q=\sum_c W_c\mu_c\mu_c^\top=\sum_c m_cm_c^\top/W_c\) (`PROBLEM.md` §2).

## Correspondence

| # | Registry clause | Lean | Status |
|---|---|---|---|
| H1 | `assumptions`: positive weights | `Sample.weight_pos` in `ConfigSpec.lean` | carried by the type; a zero or negative weight is inexpressible |
| H2 | `assumptions`: source remains nonempty / `statement`: "non-singleton `a`" | `RelocationAssumptions`, conjunct 2: `∃ j ∈ cell z (z i), j ≠ i` | exact — `i` is always in its own cell, so this is `|cell| ≥ 2`, which is both "non-singleton" and "nonempty after the move" |
| H3 | `statement`: "moved **from** `a` **to** `b`" (a relocation, so `a ≠ b`) | `RelocationAssumptions`, conjunct 1: `b ≠ z i` | exact, and load-bearing (F2) |
| H4 | not stated in the claim node | `RelocationAssumptions`, conjunct 3: `(cell z b).Nonempty` | **stronger than the claim** — narrowing, disclosed in the spec's non-coverage list; see F1 |
| D1 | \(I_q=\sum_c W_c\mu_c\mu_c^\top\) | `fisher = ∑ c, cellBlock (cellMass) (cellSum)`, `cellBlock W T = W⁻¹ • vecMulVec T T` | exact, identical to `PROBLEM.md`'s \(\sum_c m_cm_c^\top/W_c\) |
| D2 | \(W_a,W_b\) are the pre-move masses | `cellMass S z (z i)`, `cellMass S z b` — both at the **old** labeling `z` | exact |
| D3 | \(\mu_a,\mu_b\) are the pre-move centroids | `centroid S z (z i)`, `centroid S z b` | exact |
| D4 | \(u_c=s-\mu_c\) | `centroid S z c - S.score i` \(=\mu_c-s\) | sign-flipped, immaterial in a symmetric outer product; see F3 |
| D5 | \(\alpha=wW_a/(W_a-w)\), \(\beta=wW_b/(W_b+w)\) | `alpha`, `beta` in `ScalarExchangeSpec.lean`, applied as `alpha (S.weight i) (cellMass S z (z i))`, `beta (S.weight i) (cellMass S z b)` | exact, with source and destination masses in the right slots |
| D6 | the move itself | `relocate z i b = Function.update z i b` | exact: one row relabelled, all others fixed |
| C1 | \(\Delta I=\alpha u_au_a^\top-\beta u_bu_b^\top\) | `RelocationConclusion`, right-hand side | exact, including the `after − before` orientation |
| — | the implication itself | `RelocationConclusion` freezes hypotheses, conclusion and the arrow | frozen |

Nothing appears in the conclusion that is absent informally. The one direction
of asymmetry is H4, in the safe direction (fewer moves covered, not more).

## Findings and dispositions

**F1 — the claim node lists two hypotheses; the Lean carries four (hardening;
not applied, owed on the claim node).** `claims/D-RANK2-MOVE.json` gives
`assumptions: ["positive weights", "source remains nonempty"]`. The Lean adds
`b ≠ z i` and `(cell z b).Nonempty`. The first is implicit in the word "moved"
and is genuinely required (F2). The second is a real narrowing: read literally,
the claim node asserts the identity for a move into a cell that happens to be
empty, and the Lean does not. The narrowing is *disclosed* — the spec's
"Not part of this specification" section leads with the destination-empty case —
so this is an honesty-of-the-registry issue rather than a silent weakening.
*Recommended disposition:* patch the claim node's `assumptions` to
`["positive weights", "source remains nonempty", "destination is a distinct,
nonempty cell"]` before attaching `formal_proof`. Do **not** instead delete the
conjunct from the spec: see F2.

**F2 — exactly one of the three conjuncts is load-bearing (informational).**
I re-implemented `fisher`, `cellBlock`, `centroid`, `alpha`, `beta` and
`relocate` in exact rational arithmetic, reproducing Lean's `x⁻¹ = 0` and
`x / 0 = 0` conventions, and checked the frozen identity on 4,000 random
configurations with \(d\in\{1,2,3\}\), \(N\in\{2,\dots,5\}\), \(K\in\{1,2,3\}\),
rational scores and rational positive weights, stratified by which conjunct
holds:

| stratum | identity holds |
|---|---|
| all three conjuncts (the frozen statement) | 657 / 657 |
| `b ≠ z i` dropped (destination = source) | **32 / 2113** |
| source-non-singleton dropped | 417 / 417 |
| destination-nonempty dropped | 388 / 388 |
| both of the latter dropped | 104 / 104 |

So `b ≠ z i` is essential — without it the frozen statement is false, since a
no-op move has \(\Delta I=0\) while the right-hand side becomes
\((\alpha-\beta)u_au_a^\top\) with
\(\alpha-\beta=2w^2W_a/(W_a^2-w^2)>0\). (The 32 survivors are the cases where
\(u_a=0\) or \(d=0\).)

The other two conjuncts are *not* load-bearing, and it is worth recording why,
because the spec's stated reason is not quite the real one and a future reader
might "simplify" in the wrong direction. The spec says the destination-empty
case is excluded because "`W_b = 0` makes `β = 0` and `μ_b = 0` by the
junk-value convention rather than by the algebra". Only `μ_b = 0` is junk;
`β = w·0/(0+w) = 0` is ordinary arithmetic, since `w > 0` makes the denominator
nonzero. The identity therefore does hold with an empty destination — but it
holds because a genuine zero coefficient annihilates a meaningless `u_b = −s`.
Symmetrically, in the singleton-source case `α = w·w/0` *is* junk, yet the term
vanishes for a real reason: a singleton's centroid is its own score, so
`u_a = 0` and `α • (u_a u_aᵀ) = 0` whatever `α` is.
*Disposition:* keeping both conjuncts is the correct call — dropping either
would buy generality that rests on a junk value rather than on the algebra, and
would make the statement's truth depend on Lean's division convention. The
docstring justification for the destination-empty bullet should be corrected to
say this. No change to the logical content of the spec.

**F3 — `u` is oriented as `μ − s`, the prose as `s − μ` (informational).**
`RelocationSpec.lean` writes `centroid S z c - S.score i`; §D2 and
`AUDIT-D-EXCHANGE-VORONOI-001.md` §7 write \(u_c=s-\mu_c\). Both appear only
inside `vecMulVec u u`, which is symmetric under `u ↦ −u`, so the two agree
identically. Downstream this stays safe: D3's \(q_{ab}=u_a^\top Hu_b\) flips
both arguments at once and is likewise invariant. The spec's correspondence
table states its own convention (`u_c = μ_c − s`) but does not flag the
difference from the prose; a half-line would remove the reader's double-take.

**F4 — the non-coverage list is honest but has three small gaps (informational).**
The five listed exclusions are all real and correctly described (modulo F2's
justification wording). Missing:

* *No positivity or rank statement.* `ScalarExchangeSpec.lean` documents `alpha`
  and `beta` as "positive rank-one coefficient", and the claim is titled
  "rank-two", but the frozen conclusion asserts only the matrix equality —
  neither `0 < alpha`, `0 < beta`, nor `rank ΔI ≤ 2`. This matches the claim's
  `statement` field, which is the identity alone, so it is not a mismatch; but
  downstream consumers should know that D5 re-derives coefficient positivity
  from `ScalarExchangeAssumptions` rather than importing it from D2.
* *One row, not one merged atom.* `relocate` moves a single row. In the merged
  encoding D5 uses, one atom is one row, so this is faithful; on an unmerged
  sample, relocating an entire duplicate class is a different (multi-row) move
  and is not covered. D2 needs no injectivity hypothesis, so this is a
  strengthening relative to D5's setting, not a gap — but it is worth naming.
* *No simultaneous or iterated moves, and no population/atomless analogue.* The
  claim is `level: finite_assignment`, so nothing is owed here; the omission is
  only from the disclaimer list.

**F5 — no blocking finding.** No hypothesis was found that makes a clause
vacuous or trivially true; no quantifier is misplaced; no editable definition
appears in the frozen statement.

## Independently confirmed correct

- **Closure under definitional dependency is total, and stronger than required.**
  Every name in `RelocationConclusion` — `RelocationAssumptions`, `relocate`,
  `fisher`, `cellMass`, `cellSum`, `cellBlock`, `cell`, `centroid`,
  `Sample.score`, `Sample.weight`, `alpha`, `beta` — resolves to
  `RelocationSpec.lean`, `ConfigSpec.lean` or `ScalarExchangeSpec.lean`, all
  frozen. More than that, the *import closure* of `RelocationSpec.lean` is
  exactly `ConfigSpec` + `ScalarExchangeSpec` + Mathlib: no proof module is even
  in scope, so no prover edit can reach the statement. `Config.lean` was checked
  declaration by declaration and contains theorems only — the definitional layer
  is entirely in `ConfigSpec.lean`. Hosting `relocate` in the spec rather than in
  `Relocation.lean` is necessary for the same reason and also protects
  `ExchangeStable` in `ExchangeVoronoiSpec.lean`, which mentions it.
- **The exported theorem states the frozen Prop.**
  `rank_two_relocation (S) (z) (i) (b) : RelocationConclusion S z i b` is
  universally quantified over `S, z, i, b` and implicitly over `d, N, K`, adds no
  hypothesis, and does not restate the identity; its body is a destructuring of
  `RelocationAssumptions` into `fisher_relocate_sub`. The registry mark should
  name `ScoreQuantFormal.rank_two_relocation` for that reason, even though D5
  calls the uncurried `fisher_relocate_sub` (`ExchangeVoronoi.lean:112`).
  `AxiomAudit.lean` `#guard_msgs`-pins `#print axioms` for both.
- **The retained information is the right object.** `cellBlock W T = W⁻¹ • T Tᵀ`
  with `T_c = W_c μ_c` is `W_c μ_c μ_cᵀ`, so `fisher = ∑_c W_c μ_c μ_cᵀ`, which
  is `PROBLEM.md`'s \(I_q\) verbatim. Empty cells contribute `0` under
  `(0:ℝ)⁻¹ = 0`, matching the informal convention of summing over occupied cells.
  The statement is homogeneous of degree one in the weights, so using
  unnormalized masses instead of probabilities is a generalization.
- **The pre/post bookkeeping is right.** All four of \(W_a,W_b,\mu_a,\mu_b\) are
  evaluated at `z`, not at `relocate z i b`; the left-hand side is
  `after − before`. Reversing the subtraction, or evaluating a mass after the
  move, breaks the identity on the first random configuration.
- **The proof-side decomposition matches the human audit.** `cellBlock_erase`
  and `cellBlock_insert` state exactly the two displayed identities of
  `AUDIT-D-EXCHANGE-VORONOI-001.md` §7, `α u_au_a^\top-wss^\top` and
  `wss^\top-\beta u_bu_b^\top`, whose sum telescopes the `w ssᵀ` term away.
  (Proof-side, so out of scope for this audit, but it confirms the statement is
  the one the prose derives.)
- **The hypotheses are satisfiable, so the statement is not vacuous.**
  `Counterexamples.VoronoiConverse` supplies a concrete instance: `labeling =
  ![0,0,0,1]`, `i = 2`, `b = 1` gives `b ≠ z i`, a source cell `{0,1,2}` with a
  second row, and a nonempty destination `{3}`.
- **Degenerate dimensions are harmless.** `K ≤ 1` makes `b ≠ z i` unsatisfiable
  and the conclusion vacuous; `d = 0` makes it a true `0 × 0` equality; neither
  can be reached from a nondegenerate instance.
- **The extra hypothesis is dischargeable downstream.**
  `RelocationAssumptions z i b` is exactly `Admissible z i b ∧ (cell z b).Nonempty`,
  and `VoronoiAssumptions` conjunct 2 supplies the second factor for every `b`,
  so H4 costs D5 nothing.

## What remains uncovered

The frozen statement is one exact matrix identity about one single-row move in
one fixed finite labeling, in exact real arithmetic. It says nothing about
determinants, log-determinants or any ordering of objectives (those are D3 and
downstream, frozen separately in `DetGainSpec.lean`); nothing about the sign or
size of the coefficients; nothing about moves that empty a cell or fill an empty
one; nothing about simultaneous or iterated relocations; nothing about the
atomless population limit. And it is a statement about mathematics: nothing
connects `fisher`, `centroid` or `relocate` to `src/scorequant/partition.py`'s
versions of them, nor to floating-point conditioning.

## Trust audit

Not performed here — this audit is on the statement, and the brief excluded
running `lake`. What is visible statically: `AxiomAudit.lean` carries
`#guard_msgs`-pinned `#print axioms` lines for both `fisher_relocate_sub` and
`rank_two_relocation`, each expecting exactly `[propext, Classical.choice,
Quot.sound]`, and `AxiomAudit.lean` sits in the default target's import closure
through `Counterexamples`. The toolchain is `leanprover/lean4:v4.33.1` with
Mathlib pinned at `v4.33.1`, matching the `system` string other `formal_proof`
blocks record. A green `lake build --wfail` on the `formal` CI job is still a
precondition for the mark.
