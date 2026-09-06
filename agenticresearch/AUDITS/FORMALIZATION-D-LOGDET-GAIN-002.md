# Formal statement audit — D-LOGDET-GAIN (002)

**Claim:** `D-LOGDET-GAIN` (D3, the exact finite D determinant and
log-determinant relocation gain).

**Date:** 6 September 2026.

**Result: match after hardening.** No blocking finding. The frozen proposition
now covers the claim node whole — statement field, title and cited prose box —
under hypotheses that are exactly the node's amended `assumptions`. Two prose
corrections are required inside the frozen file; neither touches a Lean
declaration, and neither blocks the `formal_proof` mark once made.

**Audited object:** `formal/ScoreQuantFormal/DetGainSpec.lean` **as amended**,
together with the frozen definitional layer it imports,
`formal/ScoreQuantFormal/ConfigSpec.lean`, and the exported declaration
`ScoreQuantFormal.det_relocation_gain` in the editable
`formal/ScoreQuantFormal/DetGain.lean`.

**Canonical source:** `claims/D-LOGDET-GAIN.json` as amended,
`KNOWN_RESULTS/04-d-optimality.md` §D3,
`AUDITS/AUDIT-D-EXCHANGE-VORONOI-001.md` §§7–8.

**This is a second-round audit.** It supersedes
`AUDITS/FORMALIZATION-D-LOGDET-GAIN-001.md` as the statement audit of record
for `D-LOGDET-GAIN`, and is the report the `formal_proof.statement_audit` field
should name. Round one audited a `DetGainConclusion` that contained the
determinant identity only; it returned `match after hardening` with one
blocking finding, F1 — the node promised a log-determinant gain the frozen
statement did not contain — and it stated explicitly that any change to the
Lean declarations `qform`, `detRatio`, `DetGainAssumptions` or
`DetGainConclusion` would require a new audit. `DetGainConclusion` has since
been changed, so this is that audit. Round one is not withdrawn: it remains the
record of the pre-amendment file, and where its conclusions were disturbed by
the amendment they are re-derived below rather than carried over.

**Independence.** This audit ran in a session with no shared derivation
context. It was given the amended claim node, the prose proof, the prior
human-algebra audit of the chain, the first-round formalization audit and the
Lean files, and no part of any formalizing session's transcript. The identity
was re-checked in exact rational arithmetic over 3,718 nonsingular symmetric
configurations; the logarithmic component was re-checked over the 1,070 of them
satisfying its positivity hypotheses; every hypothesis was probed for necessity
by explicit witness rather than taken from the spec docstring.

## What changed since round one

1. `DetGainConclusion` is now a conjunction. The first conjunct is the
   determinant identity round one audited, unchanged in the form 001 records.
   The second is the `F_D = log det I` gain, guarded by two positivity
   hypotheses that sit **inside** the conclusion, after the
   `DetGainAssumptions` arrow.
2. `DetGainSpec.lean` gained `import Mathlib.Analysis.SpecialFunctions.Log.Basic`.
3. The node's `assumptions` field was rewritten from the single line "current
   and candidate information positive definite" to
   `["current information symmetric and nonsingular", "positive current and
   candidate determinants for the log-determinant form"]`.
4. The spec docstring was rewritten in response to round one's F3, F4 and F5.
5. `det_relocation_gain` now discharges both conjuncts.

`qform`, `detRatio` and `DetGainAssumptions` are unchanged, character for
character, from the definitions round one quotes. The node's `statement`,
`title`, `role`, `proof_location` and `dependencies` fields are unchanged.

## Informal source

Chapter 4 of `KNOWN_RESULTS` opens by defining the objective,
\(F_D(I)=\log\det I\). §D3 then states, for \(H=I^{-1}\),
\(q_{aa}=u_a^\top Hu_a\), \(q_{bb}=u_b^\top Hu_b\), \(q_{ab}=u_a^\top Hu_b\),

\[
\Delta F_D=\log\!\left[(1+\alpha q_{aa})(1-\beta q_{bb})+\alpha\beta q_{ab}^2\right],
\]

where \(\alpha\), \(\beta\) and the update \(\Delta I=\alpha u_au_a^\top-\beta
u_bu_b^\top\) come from D2 (`D-RANK2-MOVE`, the node's sole dependency).
\(\Delta F_D\) is the gain of the move, \(F_D(I+\Delta I)-F_D(I)\); the same
orientation is used one layer up in `ExchangeVoronoiSpec.ViolationLogGain`,
which writes `Real.log (fisher S (relocate z i b)).det - Real.log (fisher S z).det`.

The claim node's `statement` field carries the bracket only; its `title` and
`proof_location` carry the logarithm. Both are now formalized.

## Correspondence

| # | Registry clause | Lean | Status |
|---|---|---|---|
| H1 | `assumptions[0]`, "current information symmetric" | `DetGainAssumptions`, first conjunct: `I.IsSymm` | exact; load-bearing for **both** components (F3 of this report's confirmations) |
| H2 | `assumptions[0]`, "nonsingular" | `DetGainAssumptions`, second conjunct: `IsUnit I.det` | exact; load-bearing for the identity |
| H3 | `assumptions[1]`, positive current determinant | `0 < I.det`, inside the second component | exact |
| H4 | `assumptions[1]`, positive candidate determinant | `0 < detRatio …`, inside the second component | exact **given the first component**, which makes `det (I+ΔI) = det I · detRatio`; under H3 the two positivity statements are equivalent |
| D1 | `H = I⁻¹` | `qform I⁻¹ _ _`, the genuine inverse under H2 | exact |
| D2 | `q_aa = u_aᵀ H u_a` | `qform I⁻¹ ua ua`, with `qform H u v = u ⬝ᵥ H.mulVec v` | exact |
| D3 | `q_bb = u_bᵀ H u_b` | `qform I⁻¹ ub ub` | exact |
| D4 | `q_ab = u_aᵀ H u_b` | `qform I⁻¹ ua ub` | exact, argument order included |
| D5 | the D2 update `α u_a u_aᵀ − β u_b u_bᵀ` | `α • vecMulVec ua ua - β • vecMulVec ub ub`, added to `I` | exact; `vecMulVec u v = u vᵀ`, and the `+α`/`−β` signs match `RelocationConclusion` |
| C1 | `statement`: "the determinant ratio is `(1+α q_aa)(1−β q_bb)+αβ q_ab²`" | first component: `(I + Δ).det = I.det * detRatio …` | equivalent under H2, which makes the division legitimate |
| C2 | `title` "log-determinant gain"; §D3's boxed `ΔF_D = log[·]` with `F_D = log det I` | second component: `Real.log (I + Δ).det - Real.log I.det = Real.log (detRatio …)` | exact, orientation included |
| — | the implication itself | `DetGainConclusion` | frozen: hypotheses, conclusion and arrow |
| — | `role`: "exact O(d²)-type candidate evaluation" | — | not covered; disclaimed in the spec's non-coverage list |
| — | `dependencies`: `D-RANK2-MOVE` supplies `α`, `β`, `u_a`, `u_b` | universally quantified reals and vectors | strengthening by instantiation; disclosed in the non-coverage list |

Nothing appears formally that is absent informally. Every clause of the node
that states a proposition is now on the covered side of this table.

## Findings and dispositions

**F1 — the spec docstring still describes the pre-amendment claim node
(required before the mark; prose only, inside the frozen file).** Two passages
quote a registry that no longer exists. The paragraph headed "**Hypotheses are
deliberately weaker than the registry's**" says "The registry assumes 'current
and candidate information positive definite'", and the doc comment on
`DetGainAssumptions` repeats it as "Strictly weaker than the registry's
'current and candidate information positive definite'". The node's
`assumptions` field was rewritten in this same amendment and now reads
"current information symmetric and nonsingular", which is `DetGainAssumptions`
exactly, not something strictly stronger.

The consequence is not cosmetic. After the amendment the relation between the
frozen hypotheses and the node's is **equality**, not strict weakening, and the
docstring's closing warning — "An audit that finds this direction reversed is a
mismatch, not a hardening" — now guards a comparison that no longer has any
slack in it. A future reader who trusts the docstring will believe there is
headroom between the Lean and the registry that they can spend; there is none.
The two sentences also make the frozen file's own correspondence record
factually wrong about the object it corresponds to, which is the same class of
defect as round one's F3.

*Required:* restate the hypothesis paragraph against the amended node — the
identity's hypotheses now match `assumptions[0]` exactly, the second
component's match `assumptions[1]` exactly, and what is dropped relative to the
node's *former* wording is candidate positive definiteness, which the identity
never needed. The load-bearing argument in the same paragraph, including both
counterexamples, is correct as written and should be kept (see the
confirmations below).

**F2 — the node's `statement` field still contains no logarithm, while its
`assumptions` field now references one (recommended; registry prose, not
blocking).** `assumptions[1]` reads "positive current and candidate
determinants for the log-determinant form", but no "log-determinant form"
appears anywhere in the node's `statement`. A reader of the node alone cannot
tell what that hypothesis qualifies; they must reach the title, or §D3, to find
it. This does not endanger the mark — ADR 0030 ties `formal_proof` to the
claim's own `statement`, and the frozen Prop covers that statement and more, so
the failure mode the protocol guards against (partial coverage) is absent in
both directions. But the node is now internally incomplete in a way it was not
before, because the amendment added an assumption for a conclusion the
`statement` field does not state.

*Recommended:* extend `statement` with the second clause it now proves, e.g.
"…the determinant ratio is `(1+αq_aa)(1−βq_bb)+αβq_ab²`; where the current and
candidate determinants are positive, `ΔF_D = log` of that ratio for
`F_D = log det I`." That change is a pure addition to the node and does not
disturb this audit, since it names exactly the second component already
frozen. If it is made, this report stands as the audit of record for the node
so extended; a change to the node's *first* clause would not.

**F3 — the second component's positivity hypotheses are stronger than Lean
requires and weaker than the claim's (observation; no change recommended).**
This is the question the task asked to be decided independently, so it was
decided by probing, not by reading.

*Not too weak.* Under `0 < I.det` and `0 < detRatio` all three logarithms are
taken of positive reals: the two hypotheses directly, and
`det (I+ΔI) = det I · detRatio > 0` derived from the first component. No
`Real.log` in the statement can reach Mathlib's junk region under its own
hypotheses. Dropping the ratio hypothesis makes the statement false, not
vacuous: at `d = 1`, `I = (2)`, `u_a = u_b = 1`, `α = 0`, `β = 2` the ratio is
`1 + (α−β)q_aa = 0`, so the left side is `Real.log 0 − Real.log 2 = −log 2` and
the right side is `Real.log 0 = 0`.

*Stronger than Lean needs.* Mathlib defines `Real.log x = Real.log |x|`
(`Real.log_abs`, `Real.log_neg_eq_log`) and `Real.log_mul` needs only
`x ≠ 0`, `y ≠ 0`. So the equation is in fact true under `detRatio ≠ 0` alone,
with `I.det ≠ 0` already supplied by `IsUnit I.det`; `0 < I.det` is redundant
for provability. Probes confirm it: at `det I = −2` with ratio `3/2`, and at
`det I = 2` with ratio `−3/2`, both sides agree.

*Disposition: keep them.* The extra strength is fidelity, not slack. At a
negative determinant `Real.log (det I)` is not \(F_D\) in any sense §D3 would
recognize — it is `log |det I|`, and a `≠ 0` version of this theorem would be a
true Lean statement exporting a false-looking claim about the D objective.
Both hypotheses are also strictly weaker than what the claim supplies:
`PosDef I → 0 < det I` and, per §8 of `AUDIT-D-EXCHANGE-VORONOI-001`,
candidate positive definiteness is *derived* in the D chain from the cell-moment
representation and `R > 1`, so the claim's regime implies both. The theorem is
therefore not weakened below the claim, which is the test that matters here.

**F4 — the correspondence table's two identity rows are asymmetric
(cosmetic).** The table maps the log form to "`DetGainConclusion`, second
component" but maps "the identity itself" to bare "`DetGainConclusion`". It
should read "first component", parallel to the log row; as written it can be
read as saying the whole conclusion is the identity, which is what the
amendment stopped being true.

**F5 — the non-coverage list does not disclaim the statistical reading of `I`
(cosmetic; completeness).** `I` here is a bare symmetric matrix. Nothing in the
frozen statement identifies it with `fisher S z`, still less with a Fisher
information; the third non-coverage bullet disclaims the tie between `α`, `β`,
`u_a`, `u_b` and a relocation but says nothing about `I`. The sibling
`LeverageSpec.lean` spells this out at length ("**That `fisher` is a Fisher
information**"), and `FORMALIZATION-D-LEVERAGE-001` treated it as worth
recording. One bullet here would make the two specs consistent and would close
the last route by which a reader could over-read the frozen Prop.

**F6 — §D3's prose must be updated when the mark lands (registry-level
condition; not a Lean defect).** `KNOWN_RESULTS/04-d-optimality.md` §D3 still
reads "Machine-checked … as `ScoreQuantFormal.det_add_rank_two`" and "Statement
not separately frozen, so no `formal_proof` field yet". Both sentences are
false once the mark is attached. §D2 and §D4 were rewritten when their audits
closed and give the pattern: name the frozen spec, name the exported
declaration (`det_relocation_gain`, not `det_add_rank_two`, which is the
unfolded working form and lives in an editable module), name this audit, and
say that both the determinant and the log form are covered while the `O(d²)`
evaluation claim is not.

## Independently confirmed correct

- **Round one's F1 is genuinely resolved, and resolved in the stronger of the
  two ways it offered.** Round one allowed either retitling the node to drop
  the logarithm (its preferred option (a)) or extending the freeze with a log
  form under positivity (option (b)). Option (b) was taken. The result is that
  the ambiguity F1 turned on — whether the mark is read against the node's
  `statement` field or against its title and cited prose — no longer needs
  deciding, because the frozen Prop covers both readings. The protocol's
  "partly covered claims stay unmarked" rule no longer bites: the only node
  fields not covered are `role`, which asserts no proposition and is
  explicitly disclaimed, and the `dependencies` edge, which is covered by
  universal quantification.
- **The conjunction was added, not substituted, and the positivity is scoped
  correctly.** This is the single most important structural check, because
  round one warned that hoisting the positivity into the hypotheses would
  reintroduce what F2 of that report had removed. `DetGainAssumptions` is still
  `I.IsSymm ∧ IsUnit I.det`; the two positivity hypotheses sit inside the
  second conjunct, after the arrow, and are invisible to the first. The
  identity is therefore available at an indefinite `I` and at a negative
  determinant exactly as before. Verified by re-deriving the identity over
  3,718 exact-rational nonsingular symmetric configurations at
  `d ∈ {1,2,3,4}`, of which 3,192 were not positive definite, 1,937 had
  `det I < 0` and 2,579 used a negative `α` or `β`: zero failures.
- **Bundling the two components under one arrow is sound.** The second
  component needs `IsSymm` and needs it genuinely, so inheriting it is
  correct; it needs no nonsingularity beyond what `0 < I.det` would give it,
  since over ℝ `x ≠ 0 ↔ IsUnit x`, so inheriting `IsUnit I.det` is redundant
  rather than weakening. Formally the bundled form and the unbundled form
  `I.IsSymm → 0 < I.det → 0 < detRatio → …` imply each other, so nothing is
  lost by the bundle and nothing needed is missing from it. Bundling is also
  what the protocol asks for — step A requires a node carrying an identity and
  its corollary to be formalized whole or split — and it matches the precedent
  set by `LeverageSpec.LeverageConclusion`, which freezes two components under
  one `LeverageAssumptions` arrow and is discharged by one exported theorem.
  One theorem per node also satisfies the registry validator, which requires a
  single `declaration` per claim and rejects a declaration already claimed
  elsewhere; `det_relocation_gain` is claimed by no other node.
- **`IsSymm` is load-bearing for the second component too, not only the
  first.** Round one established this for the identity. It is not inherited
  automatically for the log form, because a false identity could still yield a
  true log equation if the two ratios happened to have equal logarithms, so it
  was re-checked: with the docstring's non-symmetric `I = ![![1,1],![0,1]]`,
  `u_a = e₁`, `u_b = e₂`, `α = 1`, `β = −1`, both determinants are positive
  (`det I = 1`, `det (I+ΔI) = 4`) and the frozen ratio is `3`, so the left side
  is `log 4` and the right side `log 3`. The second component is false without
  symmetry, in the region where its own hypotheses hold.
- **The docstring's two counterexamples are correct as stated.** Non-symmetric
  `I = ![![1,1],![0,1]]`, `u_a = e₁`, `u_b = e₂`, `α = β = 1`: `q_ab = −1`,
  `q_ba = 0`, true determinant `0`, frozen right-hand side `1`, and the general
  `q_ab q_ba` law gives `0`. Singular `d = 1`, `I = 0`, `α = 1`, `β = 0`,
  `u_a = 1`: left side `1`, right side `0` because Mathlib's `I⁻¹` is the junk
  value `0`. Both re-computed in exact rationals.
- **The log component is non-vacuous.** `I = 1`, `α = β = 0` satisfies every
  hypothesis simultaneously; 1,070 of the random trials did.
- **Closure under definitional dependency, re-verified including the new
  import.** Every symbol in the amended `DetGainConclusion` resolves either to
  Mathlib (`Matrix.det`, `Matrix.inv`, `Matrix.IsSymm`, `IsUnit`, `vecMulVec`,
  `dotProduct`, `mulVec`, `Real.log`, `+`, `-`, `•`) or to `DetGainSpec.lean`
  itself (`qform`, `detRatio`, `DetGainAssumptions`). No definition reachable
  from the frozen conclusion lives in `Config.lean`, `Relocation.lean`,
  `DetGain.lean`, `Leverage.lean`, `ExchangeVoronoi.lean`, `Corollaries.lean`
  or `Counterexamples.lean`. The file's only project import is the frozen
  `ConfigSpec.lean`, and none of that file's definitions appears in the Prop —
  the import supplies Mathlib's matrix vocabulary transitively.
  `Mathlib.Analysis.SpecialFunctions.Log.Basic` introduces nothing editable: it
  is a pinned Mathlib module under `.lake/packages/`, at the manifest revision
  `0df444a3` for `v4.33.1`, and it is already part of the frozen surface
  through `ExchangeVoronoiSpec.lean`, which imports it directly. Mathlib cannot
  depend on `ScoreQuantFormal`, so no cycle back into an editable module is
  possible. The only symbol it contributes is `Real.log`.
  One consequence worth recording: `LeverageSpec.lean` imports
  `DetGainSpec.lean`, so this file is now inside D4's audited boundary as well,
  and a future edit here needs both audits reopened.
- **The exported theorem states the frozen Prop and nothing narrower.**
  `theorem det_relocation_gain (I : Matrix (Fin d) (Fin d) ℝ) (α β : ℝ)
  (ua ub : Fin d → ℝ) : DetGainConclusion I α β ua ub` — the type is the frozen
  proposition applied to its five explicit arguments in the spec's own order,
  universally quantified over those and over the implicit `{d : ℕ}`, with no
  side condition and no restatement. It proves both conjuncts: `rintro` takes
  the assumptions apart, `hid` is the identity from `det_add_rank_two`, and the
  second is closed by rewriting with `hid` and applying `Real.log_mul` to the
  two nonzero factors obtained from the component's own hypotheses. Nothing in
  the proof term needs a hypothesis the spec does not grant.
  `AxiomAudit.lean` pins it with `#guard_msgs` on the allowlist `propext`,
  `Classical.choice`, `Quot.sound`.
- **The polynomial and the quadratic form are unchanged and still right.**
  `detRatio α β qaa qbb qab = (1 + α * qaa) * (1 - β * qbb) + α * β * qab ^ 2`,
  character for character the registry bracket; `qform H u v = u ⬝ᵥ H.mulVec v`,
  so `qform I⁻¹ ua ub` is `q_ab` and not `q_ba` independently of symmetry.
- **Round one's F3, F4 and F5 were addressed.** F3: the inaccurate pointer
  claiming the `F_D` form was "frozen downstream in
  `ExchangeVoronoiSpec.ViolationLogGain`" is gone, and it is gone in the best
  way — the log form is now frozen here, so the misleading sentence has no
  successor to be inaccurate about. F4: the non-coverage list now says
  explicitly that `detRatio_eq_one_add_exchangeExcess` "lives in the editable
  `DetGain.lean` and is therefore *not* part of this frozen boundary … a reader
  must not treat it as audited". The join itself was re-checked and is a `ring`
  identity: expanding `detRatio` gives
  `1 + [α q_aa − β q_bb − αβ(q_aa q_bb − q_ab²)]`, whose bracket is
  `ScalarExchangeSpec.exchangeExcess` verbatim; it held in every trial. F5: the
  `role` field is now disclaimed in its own bullet. Round one's remaining
  finding, F6, was the `assumptions` rewrite, which is done and is accurate —
  see the next item.
- **The amended `assumptions` field is accurate for the node.** Clause one,
  "current information symmetric and nonsingular", is `DetGainAssumptions`
  exactly: neither over-strong (the identity holds at indefinite and
  negative-determinant `I`, checked on 3,192 non-positive-definite
  configurations) nor under-strong (both conjuncts fail when removed, by the
  two witnesses above). Clause two, "positive current and candidate
  determinants for the log-determinant form", is the second component's
  hypotheses exactly, once the first component is used to turn positivity of
  the candidate determinant into positivity of `detRatio`; the qualifier "for
  the log-determinant form" correctly scopes it away from the identity. The
  relocation-specific hypotheses — non-singleton source, distinct nonempty
  destination — are absent, correctly: they belong to `D-RANK2-MOVE`, which is
  this node's declared dependency and whose own `assumptions` were narrowed to
  record them when `FORMALIZATION-D-RANK2-MOVE-001` closed. Because the
  amendment weakened a *proved* node's assumptions, the five claims in
  `implies` are unaffected: they consume D3 in the positive-definite regime,
  which is contained in the new one.
- **Orientation and sign.** The second component's left side is
  `log det (candidate) − log det (current)`, matching `ΔF_D` as §D3 uses it and
  matching `ViolationLogGain` one layer up. A reversed sign would have been
  invisible in the identity and is the kind of slip this component could have
  carried; it does not.

## What remains uncovered

The frozen statement is one algebraic identity between two real numbers,
together with the logarithmic form that identity yields where both determinants
are positive. It asserts no sign or positivity of its own: the second component
*assumes* `0 < det I` and `0 < detRatio` and says nothing when either fails, and
that `I + ΔI` is positive definite is claimed nowhere. It says nothing about
which `α`, `β`, `u_a`, `u_b` arise from a relocation — that tie is D2's, and
the theorem joining the two layers,
`detRatio_eq_one_add_exchangeExcess`, sits in an editable module and is not
audited. It says nothing about `I` being a retained information or a Fisher
information; `I` is an arbitrary symmetric matrix here, and even in the
downstream chain `fisher` is a definition rather than a statistical theorem
(F5). It establishes no complexity bound, so the node's `role` is unsupported
by it. It is exact real arithmetic: nothing here concerns floating-point
conditioning of `I⁻¹`, rank tolerances, or any part of the Python/JAX
implementation.

## Verdict and conditions

**Match after hardening.** Clause for clause, the amended `DetGainConclusion`
expresses the whole of the amended claim node: the `statement` field's
determinant ratio as its first component, the title's and §D3's log-determinant
gain as its second, under hypotheses that match the node's amended
`assumptions` exactly, with the positivity correctly confined to the component
that needs it, with every hypothesis load-bearing, with no vacuity, no
reachable junk value, no misplaced quantifier, and with definitional
dependencies closed inside frozen files and pinned Mathlib. The exported
theorem states that Prop and proves both of its components without adding a
hypothesis. Round one's blocking finding F1 is discharged.

Conditions on the `formal_proof` mark:

1. **F1 must be corrected** — the two passages in `DetGainSpec.lean` that
   describe the registry's hypotheses as "current and candidate information
   positive definite" no longer describe the node. F4 and F5 should ride along.
   These are prose corrections inside the audited file; this report is the
   statement audit of record for the file so corrected, **on the condition that
   the Lean declarations `qform`, `detRatio`, `DetGainAssumptions` and
   `DetGainConclusion` are not touched.** Any change to those requires a third
   audit.
2. **F6 must be done in the same change** — §D3's prose still says the
   statement is not frozen and names the wrong declaration.
3. **F2 is recommended, not required.** The node's `statement` field may be
   extended to name the log form; the mark is honest either way, because the
   frozen Prop covers that field and more.

None of these blocks the mark. On completion it should name `system`
`Lean 4.33.1 + Mathlib 4.33.1`, `spec`
`formal/ScoreQuantFormal/DetGainSpec.lean`, `file`
`formal/ScoreQuantFormal/DetGain.lean`, `declaration`
`ScoreQuantFormal.det_relocation_gain`, and `statement_audit`
`AUDITS/FORMALIZATION-D-LOGDET-GAIN-002.md`.
