# Formal statement audit — D-LOGDET-GAIN (001)

**Audited object:** `formal/ScoreQuantFormal/DetGainSpec.lean`, the frozen
statement boundary for `D-LOGDET-GAIN` (D3, the exact finite D determinant
gain), together with the frozen definitional layer it imports,
`formal/ScoreQuantFormal/ConfigSpec.lean`.

**Canonical source:** `claims/D-LOGDET-GAIN.json`,
`KNOWN_RESULTS/04-d-optimality.md` §D3, `AUDITS/AUDIT-D-EXCHANGE-VORONOI-001.md`
§§7–8.

**Date:** 6 September 2026. **Verdict: match after hardening.** The frozen
statement expresses the claim node's `statement` field exactly and under
strictly weaker hypotheses. One finding (F1) must be resolved before
`formal_proof` is attached; it is a registry-level finding about the claim node,
not a defect in the Lean.

**Independence.** This audit ran in a session with no shared derivation context.
It was given the claim node, the prose proof, the prior human-algebra audit of
the chain, the boundary counterexamples and the Lean files, and no part of the
formalizing session's transcript. Every algebraic assertion below was
re-derived, and the load-bearing checks were re-run in exact rational
arithmetic rather than taken from the spec docstring.

## Informal source

D3 in `KNOWN_RESULTS/04-d-optimality.md` states, for \(H=I^{-1}\),
\(q_{aa}=u_a^\top Hu_a\), \(q_{bb}=u_b^\top Hu_b\), \(q_{ab}=u_a^\top Hu_b\),

\[
\Delta F_D=\log\!\left[(1+\alpha q_{aa})(1-\beta q_{bb})+\alpha\beta q_{ab}^2\right],
\]

where \(\alpha\), \(\beta\) and the rank-two update
\(\Delta I=\alpha u_au_a^\top-\beta u_bu_b^\top\) come from D2
(`D-RANK2-MOVE`, the node's sole dependency). The claim node's `statement`
field carries the bracket only:

> With `H = I⁻¹`, `q_aa = u_aᵀ H u_a`, `q_bb = u_bᵀ H u_b`, `q_ab = u_aᵀ H u_b`,
> the determinant ratio is `(1 + α q_aa)(1 − β q_bb) + α β q_ab²`.

Its `assumptions` field is the single line "current and candidate information
positive definite".

## Correspondence

| # | Registry clause | Lean | Status |
|---|---|---|---|
| H1 | current information positive definite | `DetGainAssumptions`: `I.IsSymm ∧ IsUnit I.det` | weakened, see F2 — `PosDef I` implies both conjuncts, so the frozen hypothesis is strictly weaker |
| H2 | candidate information positive definite | — | dropped entirely, see F2; the identity does not use it and §8 of the prior audit *derives* it rather than assuming it |
| D1 | `H = I⁻¹` | `qform I⁻¹ _ _`, with `I⁻¹` the genuine inverse under H1 | exact |
| D2 | `q_aa = u_aᵀ H u_a` | `qform I⁻¹ ua ua`, `qform H u v = u ⬝ᵥ H.mulVec v` | exact |
| D3 | `q_bb = u_bᵀ H u_b` | `qform I⁻¹ ub ub` | exact |
| D4 | `q_ab = u_aᵀ H u_b` | `qform I⁻¹ ua ub` | exact, argument order included |
| D5 | the rank-two update `α u_a u_aᵀ − β u_b u_bᵀ` of D2 | `α • vecMulVec ua ua - β • vecMulVec ub ub` added to `I` | exact; `vecMulVec u v = u vᵀ`, and the `+α`/`−β` signs match `RelocationConclusion` |
| C1 | the ratio equals `(1+α q_aa)(1−β q_bb)+αβ q_ab²` | `detRatio`, and the product form `det (I + Δ) = det I * detRatio …` | equivalent under H1, since `IsUnit I.det` makes the division legitimate |
| — | the identity itself | `DetGainConclusion` | frozen, hypotheses and arrow included |
| — | `ΔF_D = log(…)` (node title, prose boxed display) | — | **not covered**, see F1 |
| — | `role`: "exact O(d²)-type candidate evaluation" | — | not covered, see F5 |

`α`, `β`, `u_a` and `u_b` are universally quantified reals and vectors in the
Lean, where the claim intends the specific D2 relocation coefficients and
centroid offsets. That is a strengthening, not a slip: the claim's instance is
obtained by instantiation, and the spec discloses it. See F6.

Nothing appears formally that is absent informally.

## Findings and dispositions

**F1 — the node promises a logarithm the frozen statement does not contain
(required before the mark; registry-level).** The node is titled "Exact finite D
log-determinant gain", its `proof_location` points at a section titled "Exact
log-det relocation gain", and that section's boxed result is
\(\Delta F_D=\log[\cdot]\). `DetGainConclusion` contains no logarithm. The
gap is real, and it is not an accident of drafting: it is the exact price of
F2. Dropping positive definiteness is what makes `det I > 0` and
`detRatio > 0` unavailable, and without those two facts \(\log R\) is not
\(\log\det(I+\Delta I)-\log\det I\) — at a singular candidate Lean's
`Real.log 0 = 0` breaks the subtraction, which is the same trap
`AUDITS/FORMALIZATION-D-EXCHANGE-IMPLIES-VORONOI-001.md` F2 documents one layer
up. So the weakening in F2 and the omission here are two sides of one
deliberate trade, and the spec's non-coverage section says so.

Whether that blocks the mark turns on what `formal_proof` asserts. ADR 0030 is
explicit — "the field asserts that the claim's own `statement` is
machine-checked" — and the node's `statement` field is the bracket, with no
logarithm anywhere in it. On that reading the mark is honest and this finding is
a naming defect. But ADR 0030 is equally explicit that "partly covered" claims
stay unmarked, and a reader who takes the node's title or its cited prose
display as the claim will read the mark as covering \(\Delta F_D\), which it
does not.

*Required:* reconcile the node before attaching `formal_proof`. Either
(a) retitle the node to name the determinant ratio, matching its own `statement`
field, and let the log form live where it is actually frozen — in
`ExchangeVoronoiSpec.ViolationLogGain`, as a downstream inequality; or (b) keep
the title and extend the freeze with a second Prop, e.g. a
`DetLogGainConclusion` that assumes `0 < det I` and `0 < detRatio …` (or
`PosDef` on both matrices) and concludes
`Real.log (det (I + Δ)) - Real.log (det I) = Real.log (detRatio …)`, discharged
alongside `det_relocation_gain`. Option (b) must be *added*, never substituted:
replacing `DetGainConclusion` with a log form would reintroduce the positivity
hypotheses and weaken the theorem. Option (a) is the cheaper and, in this
auditor's judgement, the better one — the identity is where the content is, and
D3's `implies` edges are all consumed through the ratio.

Until (a) or (b) lands, the honest record is the one D3 already has:
`KNOWN_RESULTS` prose naming `det_add_rank_two` and saying the logarithm is not
formalized.

**F2 — the hypothesis substitution is a genuine strengthening, and both
conjuncts are load-bearing (confirmed sound; no change).** This was the question
the audit was asked to decide independently, so it was decided by
re-derivation, not by reading the docstring.

The general rank-two determinant law, for invertible `I` with no symmetry
assumption, is

\[
\det(I+\alpha u_au_a^\top-\beta u_bu_b^\top)
=\det I\left[(1+\alpha q_{aa})(1-\beta q_{bb})+\alpha\beta\,q_{ab}q_{ba}\right],
\qquad q_{ba}=u_b^\top Hu_a .
\]

Symmetry of `I` is exactly and only what collapses \(q_{ab}q_{ba}\) to
\(q_{ab}^2\), because \((I^{-1})^\top=(I^\top)^{-1}\), so `I⁻¹` is symmetric iff
`I` is. `IsSymm` is therefore not a convenience: without it the frozen statement
is **false**. Witness, checked in exact rationals: `d = 2`,
`I = [[1,1],[0,1]]` (so `det I = 1`, `I⁻¹ = [[1,-1],[0,1]]`), `u_a = e₁`,
`u_b = e₂`, `α = β = 1`. Then `q_aa = 1`, `q_bb = 1`, `q_ab = -1`, `q_ba = 0`;
the true determinant is `0`, the frozen right-hand side is `1`, and the
general `q_ab q_ba` form gives `0`.

`IsUnit I.det` is load-bearing for a second, independent reason: Mathlib's
`Matrix.inv` is `Ring.inverse (det A) • adjugate A`, so `I⁻¹ = 0` whenever
`det I` is not a unit. At a singular `I` all three `qform`s would be `0`,
`detRatio` would collapse to `1`, and the right-hand side to `det I * 1 = 0`,
while the left-hand side is generally nonzero — at `d = 1`, `I = 0`, `α = 1`,
`β = 0`, `u_a = 1` it is `1`. Dropping the hypothesis makes the statement false,
which is the good direction. Note that the junk value works *against* the
statement here rather than for it; there is no vacuity to exploit.

That the substitution is a strengthening rather than a quiet drop is then
immediate. `Matrix.PosDef` over ℝ gives `IsHermitian`, hence `IsSymm`, and
`PosDef.det_pos` gives `IsUnit det`; so `PosDef I → DetGainAssumptions I` and
`DetGainConclusion → (claim's identity under the claim's hypotheses)`. The
converse fails, so the strengthening is strict: `I = (-1)` at `d = 1` satisfies
`DetGainAssumptions` and not `PosDef`. It is also exercised rather than
notional — 400 exact-rational trials at `d ∈ {1,2,3}` produced 367 nonsingular
symmetric configurations, of which 295 were indefinite and 253 used a negative
`α` or `β`; the frozen identity held on every one, zero failures. The claim
node's *second* hypothesis, candidate positive definiteness, is not needed at
all: §8 of `AUDIT-D-EXCHANGE-VORONOI-001` derives it from the cell-moment
representation and `R > 1`, and it plays no role in an identity between two
reals. So nothing the claim needs has been dropped, with the single exception
recorded as F1.

**F3 — the non-coverage pointer for the logarithm is inaccurate (hardening;
prose only).** The spec says the `F_D` phrasing "is frozen downstream, in
`ExchangeVoronoiSpec.ViolationLogGain`". `ViolationLogGain` freezes D5's
*inequality* \(\Delta F_D\ge\log(1+\alpha\beta q_\delta^2/4)>0\), which is a
different proposition from D3's *identity* \(\Delta F_D=\log R\). D3's log form
is frozen nowhere. As written the sentence invites a reader to conclude that
the logarithm is covered somewhere in the tree; it is not. *Required:* say that
D3's log form is not formalized anywhere, and that the downstream `F_D`
statement is D5's bound, not this identity.

**F4 — an editable-module theorem is named inside the frozen boundary without
that being said (cosmetic).** The non-coverage list points at
`detRatio_eq_one_add_exchangeExcess` as the record of the join between `α`, `β`
and a relocation. That theorem lives in `DetGain.lean`, which the prover may
edit, so it is outside the audited boundary; the sentence should say so.
Substantively the join is fine — the equation `detRatio = 1 + exchangeExcess`
was checked against `ScalarExchangeSpec.exchangeExcess` and is a `ring` identity
in the six scalars — but its being unfrozen is exactly the kind of fact the
non-coverage section exists to state.

**F5 — the node's `role` is not disclaimed (cosmetic).** `role` reads "exact
O(d²)-type candidate evaluation". The identity supports that use but establishes
no complexity bound: nothing in the frozen statement says the three `qform`s can
be evaluated in `O(d²)` given a cached factorization, and nothing could, since
the statement has no computational content. One line in the non-coverage list
closes it.

**F6 — the claim node's `assumptions` field is over-strong for its own
`statement` (observation; fold into the F1 patch).** "Current and candidate
information positive definite" is not what the identity needs, and the prose
does not assume the candidate half either. Left as it stands, every future
reader of this chain will re-open the same question F2 answers — whether the
Lean quietly dropped something. If the node is patched for F1 anyway, correct
the assumptions field in the same change: symmetric nonsingular current
information, nothing about the candidate.

## Independently confirmed correct

- **Closure under definitional dependency.** Every symbol in
  `DetGainConclusion` resolves either to Mathlib (`Matrix.det`, `Matrix.inv`,
  `Matrix.IsSymm`, `IsUnit`, `vecMulVec`, `dotProduct`, `mulVec`, `+`, `-`, `•`)
  or to `DetGainSpec.lean` itself (`qform`, `detRatio`, `DetGainAssumptions`).
  `DetGainSpec.lean` imports only `ConfigSpec.lean`, which is frozen. No
  definition reachable from the frozen conclusion lives in `Config.lean`,
  `Relocation.lean`, `DetGain.lean`, `Leverage.lean`, `ExchangeVoronoi.lean`,
  `Corollaries.lean` or `Counterexamples.lean`. No blocking finding here. The
  `ConfigSpec` import is in fact unused by this file's own definitions — it
  supplies the Mathlib imports transitively — which is harmless because the
  file it reaches is itself frozen.
- **The exported theorem states the frozen Prop.**
  `det_relocation_gain (I) (α β) (ua ub) : DetGainConclusion I α β ua ub` in
  `DetGain.lean` has the frozen proposition as its type, not a restatement, and
  is universally quantified over `{d : ℕ}` and over all five explicit
  arguments; it discharges the hypotheses by `rintro` from
  `DetGainAssumptions`. `det_add_rank_two` is the unfolded working form and
  `det_relocation_gain` is the declaration the registry should name.
  `AxiomAudit.lean` pins both with `#guard_msgs`, on the allowlist
  `propext`, `Classical.choice`, `Quot.sound`.
- **The polynomial is the claim's polynomial.** `detRatio α β qaa qbb qab =
  (1 + α * qaa) * (1 - β * qbb) + α * β * qab ^ 2`, character for character the
  registry bracket, with no rearrangement that could hide a sign.
- **`qform` is the right form and the right way round.**
  `qform H u v = u ⬝ᵥ H.mulVec v = uᵀ H v`, so `qform I⁻¹ ua ub` is `q_ab` and
  not `q_ba`. Under H1 the two coincide, but the argument order still matches
  the claim, so the correspondence does not depend on symmetry to be read.
- **No vacuity and no junk-value escape.** `DetGainAssumptions` is satisfiable
  (any symmetric nonsingular `I`, e.g. `1`), so the implication is not vacuous;
  under it `I⁻¹` is the genuine inverse; and the conclusion contains no other
  inverse, no division, and no `Real.log`, so there is no second junk site.
  Both hypotheses fail *false* when removed rather than remaining true, which is
  the test that distinguishes a load-bearing hypothesis from decoration.
- **The abstraction from `fisher` costs nothing.** The frozen statement is about
  an arbitrary matrix `I`, not about `fisher S z`. Because it is universally
  quantified, the D-chain instance is a specialization, and the generality is
  what lets D5 apply it to `fisher S (relocate z i b) - fisher S z` through
  `rank_two_relocation`.
- **Degenerate dimension.** At `d = 0` both sides are `1` and the statement is
  true but empty; no hypothesis is needed to exclude it and none is claimed.
- **The prose derivation.** §7 of `AUDIT-D-EXCHANGE-VORONOI-001` writes
  `u_a = s − μ_a` where `RelocationSpec` writes `μ_a − s`. The outer products
  are invariant under that sign flip, so the two conventions agree; `q_ab` is
  likewise unaffected, since both vectors flip together.

## What remains uncovered

The frozen statement is one algebraic identity between two real numbers, for a
symmetric nonsingular real matrix and an arbitrary rank-≤2 symmetric update. It
asserts no sign, no positivity, and no ordering: that `detRatio > 0`, that
`det I > 0`, or that `I + ΔI` is itself positive definite is claimed nowhere,
and the logarithm the claim is named after is claimed nowhere either (F1). It
says nothing about which `α`, `β`, `u_a`, `u_b` arise from a relocation — that
tie is D2's, and the theorem joining them, `detRatio_eq_one_add_exchangeExcess`,
sits in an editable module (F4). It establishes no complexity bound (F5). It is
exact real arithmetic: nothing here concerns floating-point conditioning of
`I⁻¹`, rank tolerances, or any part of the Python/JAX implementation.

## Verdict and conditions

**Match after hardening.** Clause for clause, `DetGainConclusion` expresses the
claim node's `statement` field exactly, with hypotheses that are strictly
weaker in the sound direction and both load-bearing, with the conclusion no
weaker than the claim's, with no misplaced quantifier, with no vacuity, and with
its definitional dependencies closed inside frozen files. The exported theorem
states the frozen Prop.

Conditions on the `formal_proof` mark:

1. **F1 must be resolved first.** Either retitle the node to match its own
   `statement` (preferred), or add a separate frozen log-form Prop under
   positivity. Without one of the two, D3's Lean coverage stays in
   `KNOWN_RESULTS` prose, naming `det_relocation_gain` and saying the logarithm
   is missing — which is what ADR 0030 already prescribes for partly covered
   claims.
2. **F3 must be corrected** in the spec docstring, and F4 and F5 should ride
   along. These are prose corrections inside the audited file; this report is
   the statement audit of record for the file so amended, on the condition that
   the Lean declarations — `qform`, `detRatio`, `DetGainAssumptions`,
   `DetGainConclusion` — are not touched. Any change to those requires a new
   audit.
3. **F6** should be folded into the same registry patch, so the record does not
   keep suggesting that a needed hypothesis was dropped.

If conditions 1–3 are met, the mark should name `spec`
`formal/ScoreQuantFormal/DetGainSpec.lean`, `file`
`formal/ScoreQuantFormal/DetGain.lean`, `declaration`
`ScoreQuantFormal.det_relocation_gain`, and this report as `statement_audit`.
