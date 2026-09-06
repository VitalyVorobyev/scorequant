# Formal statement audit — D-LEVERAGE (001)

**Audited object:** `formal/ScoreQuantFormal/LeverageSpec.lean`, the frozen
statement boundary for `D-LEVERAGE` (D4, the leverage inequality), together with
the definitions it inherits from `ConfigSpec.lean` and `DetGainSpec.lean`.

**Canonical source:** `claims/D-LEVERAGE.json`, its dependency
`claims/FI-QUANT-IDENTITY.json`, `KNOWN_RESULTS/04-d-optimality.md` §D4, and
`AUDITS/AUDIT-D-EXCHANGE-VORONOI-001.md` §7.

**Date:** 6 September 2026. **Verdict: match after hardening.** No blocking
finding. Both halves of the bundled claim are frozen, the arrow is frozen, the
statement is closed under definitional dependency, and the exported theorem has
the frozen type. The hardenings below are docstring-level and are *not yet
applied*; one optional item (F2) would change the frozen proposition and is
pre-analyzed here so that it does not require a second audit.

**Independence.** This audit ran in a session with no shared derivation context.
It was given the two claim nodes, the §D4 prose, the prior D5 human-algebra
audit, the two prior formalization audits as format templates, and the Lean
files — no part of the formalizing session's transcript. Nothing was built and
`lake` was not run; the trust gate is step E's business, not this report's.

## Informal source

§D4 states two inequalities and proves both by one projector argument. With
\(A=[\sqrt{W_1}\mu_1,\ldots,\sqrt{W_K}\mu_K]\) and \(I=AA^\top\), the matrix
\(P=A^\top(AA^\top)^{-1}A\) is an orthogonal projector, so \(v^\top Pv\le
v^\top v\) for every \(v\). Taking \(v\) supported on one cell with entry
\(1/\sqrt{W_c}\) gives \(Av=\mu_c\) and the single-centroid bound
\(\mu_c^\top I^{-1}\mu_c\le 1/W_c\); taking \(v_a=1/\sqrt{W_a}\),
\(v_b=-1/\sqrt{W_b}\) gives \(Av=\mu_a-\mu_b\) and the centroid-difference
bound \(\delta^\top I^{-1}\delta\le 1/W_a+1/W_b\). The construction requires
\(W_c>0\) (it divides by \(\sqrt{W_c}\)) and \(I^{-1}\) to exist. §D4 is where
the hypotheses of `D-LEVERAGE` actually live: the claim node itself carries no
`assumptions` field (F5).

## Correspondence

| # | Registry clause | Lean | Status |
|---|---|---|---|
| H1 | `I` is the retained information of the labeling | `fisher S z` | definitional, inherited from `FI-QUANT-IDENTITY` (F4a) |
| H2 | `I⁻¹` exists | `LeverageAssumptions`, conjunct 2 (`PosDef`) | frozen; equivalent to nonsingularity here, not stronger |
| H3 | every `W_c` is a genuine positive cell mass | `LeverageAssumptions`, conjunct 1 (`∀ c, (cell z c).Nonempty`) | frozen; global where the claim is per-centroid (F3) |
| H4 | — (unwritten in the claim) | `Sample.weight_pos` | carried by the type; costs nothing (see confirmations) |
| H5 | — (unwritten in the claim) | `a ≠ b` in the second component | stronger than the registry statement; excludes a contentless instance (F2) |
| C1 | `μ_cᵀ I⁻¹ μ_c ≤ 1/W_c` for every centroid | `LeverageConclusion`, first component | exact |
| C2 | `δᵀ I⁻¹ δ ≤ 1/W_a + 1/W_b` for `δ = μ_a − μ_b` | `LeverageConclusion`, second component | exact, modulo H5 |
| — | the implication itself | `LeverageConclusion` | frozen |

Term by term: `qform H u v` is `u ⬝ᵥ H.mulVec v`, instantiated at `u = v`, so
`qform (fisher S z)⁻¹ μ μ` is `μᵀ I⁻¹ μ`; `centroid S z a - centroid S z b` is
`Pi.sub`, i.e. `δ` componentwise; `(cellMass S z c)⁻¹` is `1/W_c` in `ℝ`; and
`∀ c : Fin K` ranges over exactly the labels the claim's "every centroid" ranges
over. Nothing appears formally that is absent informally, apart from H4 and H5.

## Findings and dispositions

**F1 — neither hypothesis is load-bearing; the frozen statement is
unconditionally true (report only; no change recommended).** This is the answer
to the junk-value question, and it runs the *opposite* way to the D5 spec, where
the prior audit found that a singular `fisher` makes `StrictVoronoi` false
rather than vacuous. Here:

* *Empty cells do not falsify.* For an empty cell, `cellSum` is a sum over `∅`,
  so `cellSum = 0`, `cellMass = 0`, `centroid = (0:ℝ)⁻¹ • 0 = 0`, and
  `(cellMass)⁻¹ = 0⁻¹ = 0`. The first component reads `0 ≤ 0`. The prompt's
  worry that `1/W_c = 0` would make the inequality false is not realized,
  because the left side collapses to `0` at the same time and by the same
  convention.
* *A singular `fisher` does not falsify either.* Mathlib's `Matrix.inv` is
  `A.det⁻¹ʳ • A.adjugate`, and `nonsing_inv_apply_not_isUnit` gives `A⁻¹ = 0`
  when `det A` is not a unit. Then `qform 0 u u = 0`, while the right side is a
  sum of `(cellMass)⁻¹ ≥ 0`. Both components read `0 ≤ (nonneg)`.
* *And the hypotheses are not needed in the good case either.* The
  factorization `fisher = A Aᵀ` holds with empty cells included, since an empty
  cell contributes `cellBlock 0 0 = 0` on the left and a zero column
  `√0 · centroid = 0` on the right. The test vectors degenerate harmlessly:
  `(√W_c)⁻¹ = 0` when `W_c = 0`, so `v = 0`, `A v = 0 = centroid`, and
  `v ⬝ᵥ v = 0 = (cellMass)⁻¹`.

Consequently `LeverageAssumptions S z → …` and `True → …` are the same
proposition up to provability: dropping either conjunct leaves a statement that
is still true. I checked this by exhaustively evaluating both components in
exact rational arithmetic under Lean's junk conventions on 4,000 random
configurations (`d ∈ {1,2,3}`, `K ≤ 4`, `N ≤ 5`, integer scores, positive
integer weights), 10,034 instances in total, of which 1,832 had a singular
`fisher` and 2,064 had at least one empty cell: zero violations.

Disposition: **not a defect, and no change recommended.** The hypotheses match
the claim's implicit presuppositions — `1/W_c` presupposes `W_c > 0` and
`I^{-1}` presupposes nonsingularity — and a formalization that states the
claim's scope is more faithful than one that exploits junk values to prove a
formally more general proposition. But two consequences must be recorded because
a later reader will otherwise draw the wrong conclusion from them:

1. The usual sanity check — delete a hypothesis and watch the statement break —
   is unavailable for `D-LEVERAGE`. Its unavailability is not evidence that the
   hypotheses are decorative padding added to make a proof go through; it is a
   property of Lean's total-function conventions on this particular statement.
2. The statement is therefore *not* self-certifying against a junk-value proof.
   I checked the proof module for this specifically: `qform_rootFactor_le` uses
   `hpd` genuinely, through `Matrix.nonsing_inv_mul` in the idempotence step, so
   `leverage_inequality` is not discharged by the `I⁻¹ = 0` branch. That check
   is outside the audited boundary and would have to be repeated if the proof is
   rewritten.

**F2 — `a ≠ b` is a hypothesis the registry statement does not have (hardening;
optional).** The registry writes the second inequality for `δ = μ_a − μ_b` with
no restriction on `a` and `b`. The Lean adds `a ≠ b`, which makes the frozen
theorem formally weaker than the claim read literally. The excluded instance is
`a = b`, where `δ = 0`, the left side is `0` and the right side is `2/W_a > 0`:
true, and empty of content. So no content is lost, and the spec's non-coverage
section does disclose the exclusion and does say the case is "true and vacuous".

Two smaller points. First, the stated justification — "admitting it would let
`2/W_a` stand where the claim intends two distinct cells" — is a rationalization
rather than a reason: admitting `a = b` would let `0 ≤ 2/W_a` stand, which is
harmless. The honest reading is that `a ≠ b` is a fidelity convention matching
the claim's two-cell intent, and that it also happens to be what the proof's
`sum_pair_of_support` wants. Second, `a ≠ b` costs nothing downstream:
`ExchangeVoronoi.lean:158` invokes `leverage_bound` with `Ne.symm hb`, an
already-distinct pair.

Disposition: **harmless restriction, non-blocking.** Either (a) keep it and
replace the justification sentence with the plain statement that the excluded
case is trivially true and is excluded for fidelity, or (b) drop `a ≠ b` from
`LeverageConclusion`, which makes the frozen statement literally the registry
statement at the cost of a two-line `rcases eq_or_ne a b` in the proof. Option
(b) changes the frozen proposition; it is analyzed here and does not need a
second statement audit. Any *other* change to `LeverageAssumptions` or
`LeverageConclusion` does.

**F3 — nonemptiness is global where the claim is per-centroid (report only).**
`LeverageAssumptions` requires `∀ c, (cell z c).Nonempty`, so a labeling with
one empty cell falls outside the theorem even for the bound on a different,
nonempty cell. This is formally stronger than "the cell in question is
nonempty". It is nonetheless the faithful reading: the claim quantifies over
every centroid `μ_c`, and a labeling with an empty cell has no `μ_c` there, so
the claim's own scope is the all-cells-occupied one — the same convention D5
uses ("exactly `K` nonempty cells"). The class excluded is not empty: in a
3,000-configuration sweep, 1,034 of the 1,786 nonsingular-`fisher` cases had at
least one empty cell, so `PosDef` does not imply `hne` in general. (It does when
`K ≤ d`: `fisher` has rank at most the number of occupied cells.) Disposition:
**faithful, non-blocking**; worth one sentence in the docstring so a reader does
not mistake the global form for an oversight.

**F4 — the non-coverage list is honest but incomplete (hardening).** Three
omissions, in decreasing order of substance.

*(a) The dependency is imported definitionally and is not disclaimed.* The
correspondence table's first row asserts "`I` is the retained information of the
labeling | `fisher S z`". `fisher` is *defined* in `ConfigSpec.lean` as
`∑ c, cellBlock (cellMass) (cellSum) = ∑_c m_c m_cᵀ / W_c`, which is precisely
the right-hand side of `FI-QUANT-IDENTITY`. That identity — that this algebraic
object is the Fisher information of the quantized model, `Var(E[S|Z])` — is not
formalized here and is not formalizable in this vocabulary: `Var(E[S|Z])` equals
`∑_c W_c μ_c μ_cᵀ` only when `E[S] = 0` and the weights are a probability
measure, neither of which the Lean assumes. `D-LEVERAGE` is a `bridge` node
whose sole dependency is `FI-QUANT-IDENTITY` (status `literature`), so the
formal result covers D4 *given* that dependency, and the reader should be told
so. The spec should say: nothing here proves that `fisher` is the retained
Fisher information of a statistical model; the identification is inherited from
`FI-QUANT-IDENTITY` and the regularity conditions it carries.

*(b) Weights are unnormalized.* `Sample.weight` is any strictly positive
function; `∑_i w_i` need not be `1`, so `cellMass` is a mass, not `P(Z = c)`,
and the frozen `1/W_c` is not `1/P(Z=c)` unless the sample is normalized. This
is a generalization rather than a defect — the statement is invariant under
`w ↦ t·w`, since scaling `w` by `t` scales `fisher` by `t`, hence
`μᵀ I⁻¹ μ` by `t⁻¹`, and `1/W` by `t⁻¹` as well — but it is a place where the
frozen object is not literally the claim's object, and the D5 spec's practice
(and the prior audit's) is to record such things.

*(c) Zero-weight rows are inexpressible.* `Sample.weight_pos` is a field of the
type, so a labeling with a zero-weight row cannot even be stated. The D5 spec
disclaims zero-weight rows explicitly; this one does not.

Disposition: **hardening, non-blocking.** All three are docstring additions to
the "Not part of this specification" section; none touches the frozen
proposition.

**F5 — the claim node carries no `assumptions` field (hardening on the registry,
not the Lean).** `D-LEVERAGE.json` has `statement`, `dependencies`,
`proof_location`, `role` and `implies`, and no `assumptions`. There is therefore
no written hypothesis list for a statement audit to check `LeverageAssumptions`
against; this audit had to reconstruct the hypotheses from §D4's prose, where
they are visible only in the construction (`1/\sqrt{W_a}` needs `W_a > 0`,
`(AA^\top)^{-1}` needs nonsingularity). Every other formalized node in this
chain — `D-EXCHANGE-SCALAR-CORE`, `D-EXCHANGE-IMPLIES-VORONOI` — carries an
explicit `assumptions` list, and the two prior formalization audits were able to
walk it conjunct by conjunct.

Disposition: **hardening.** Before the `formal_proof` mark is attached, add to
`D-LEVERAGE.json` an `assumptions` list matching `LeverageAssumptions` —
"all `K` cells nonempty, hence every `W_c > 0`" and "retained information `I`
nonsingular (equivalently, positive definite)". This makes explicit what §D4
already requires; it narrows nothing that was ever true without it. Per the
protocol, the canonical node is patched by the audit, never silently by the
prover.

## Independently confirmed correct

- **Closure under definitional dependency holds; there is no blocking finding
  here.** `LeverageSpec.lean` imports `DetGainSpec` only, which imports
  `ConfigSpec` only. Every name in `LeverageAssumptions` and
  `LeverageConclusion` resolves into a frozen file or Mathlib: `Sample`, `cell`,
  `cellMass`, `cellSum`, `centroid`, `fisher` (and `cellBlock`, reached through
  `fisher`) are in `ConfigSpec.lean`; `qform` is in `DetGainSpec.lean`;
  `Finset.Nonempty`, `Matrix.PosDef`, `Matrix.inv` and `Pi.sub` are Mathlib. No
  definition in the frozen statement lives in `Config.lean`, `Leverage.lean` or
  any other editable module. The proof module's own definitions — `rootFactor`,
  and the lemmas `qform_le_of_symm_idem`, `fisher_eq_rootFactor_mul`,
  `qform_rootFactor_le`, `sum_pair_of_support`, `inv_sqrt_sq` — are all outside
  the statement, which is where they belong. `Config.lean`'s `cellMass_pos`,
  `cellBlock_eq_centroid` and `fisher_isSymm` are used only inside proofs.
- **The exported theorem has the frozen type.** `leverage_inequality (S) (z) :
  LeverageConclusion S z` states the frozen proposition itself, not a
  restatement of it, and destructures the frozen `LeverageAssumptions` rather
  than taking its conjuncts as separate arguments. `AxiomAudit.lean` already
  guards `leverage_inequality`, `centroid_leverage_bound` and `leverage_bound`.
  A registry mark naming `ScoreQuantFormal.leverage_inequality` would point at
  the audited statement.
- **The bundled node is formalized whole, and neither half is weakened.** Both
  the single-centroid bound and the centroid-difference bound are components of
  one frozen `LeverageConclusion`, discharged by one exported theorem. The
  protocol's rule — formalize the bundle or split the node, never mark on one
  half — is satisfied. Neither component is a specialization: the first is
  `∀ c`, the second `∀ a b` (modulo F2), with the claim's exact right-hand
  sides. Only the second half is consumed downstream
  (`ExchangeVoronoi.lean:158`); the first is present because the claim bundles
  it, which is the correct reason.
- **The direction and the non-strictness are both load-bearing.** Both bounds
  are attained: `d = K = 1` with score `1` and weight `1` gives
  `μᵀ I⁻¹ μ = 1 = 1/W`; `d = 1`, `K = 2` with scores `1, −1` and unit weights
  gives `δᵀ I⁻¹ δ = 2 = 1/W_a + 1/W_b`. So a strict `<` would be false, and the
  spec is right to disclaim strictness and the equality cases. The reverse
  inequality is false — equal centroids give `0 ≤ 2` — so a flipped `≤` would
  not have survived.
- **`PosDef` is not a strengthening of "`I⁻¹` exists".** `fisher` is
  `∑_c (cellMass)⁻¹ • (T_c T_cᵀ)` with `cellMass ≥ 0`, hence a nonnegative
  combination of rank-one PSD blocks, hence positive semidefinite
  unconditionally (confirmed on 60,000 rational test vectors across 3,000
  configurations: no negative quadratic form). A positive semidefinite symmetric
  real matrix is positive definite exactly when it is nonsingular, so
  `(fisher S z).PosDef ↔ IsUnit (fisher S z).det`, and the second conjunct of
  `LeverageAssumptions` is precisely the claim's "`I⁻¹` exists" — not more. The
  spec's line "`PosDef` is assumed, not derived" is true (it does not follow
  from `hne`) but undersells the point; the docstring would be improved by
  saying that under `hne` it is *equivalent* to nonsingularity, so no strength
  is being added. `PosDef` is also the form the downstream `VoronoiAssumptions`
  uses, so the two specs compose without a conversion.
- **`Sample.weight_pos` costs nothing.** It is formally stronger than anything
  the claim states, but a zero-weight row can be deleted without changing any
  `cellMass`, `centroid` or `fisher` entry, so the inequality for a sample with
  zero-weight rows is the same real-number inequality as for the sample with
  them removed. The only case not recovered this way is a cell whose rows all
  have weight zero, which has `W_c = 0` and is outside the claim regardless.
- **The Mahalanobis form and the metric are the right ones.** `qform` is reused
  from `DetGainSpec.lean` — the same `qform` that appears in the frozen
  `centroidSeparation` of `ExchangeVoronoiSpec.lean` — so the metric in D4 and
  the metric in D5's `q_δ` are the same expression, and the discharge of
  `ScalarExchangeSpec`'s `qDelta ≤ 1/sourceMass + 1/destinationMass` is a
  syntactic match rather than a re-derivation.
- **Degenerate shapes are safe.** `K = 0` makes both components vacuous and
  `fisher = 0`, so `PosDef` fails for `d > 0` and the implication is idle;
  `d = 0` makes every quadratic form `0` and the bounds trivial. Neither
  produces a false statement.

## What remains uncovered

The frozen statement is a fact of exact real linear algebra about one fixed
finite labeling of a strictly-positive-weight sample with all cells occupied and
`fisher` already known nonsingular. It says nothing about strictness or the
equality cases, nothing about singular or pseudodeterminant objectives, nothing
about the population/atomless setting, and nothing about the `FI-QUANT-IDENTITY`
step that licenses reading `fisher` as a Fisher information. It certifies
nothing about the Python/JAX implementation, in particular not about the rank
tolerance by which `information.py` decides `I` is nonsingular, nor about
floating-point conditioning of `I⁻¹`. Its role — discharging
`ScalarExchangeSpec`'s leverage hypothesis — is correctly listed as outside the
statement.

## Verdict

**Match after hardening.** No blocking finding: the mark may be attached once
F4 and F5 are applied. Specifically, before `D-LEVERAGE` carries `formal_proof`
with `spec: formal/ScoreQuantFormal/LeverageSpec.lean`, `file:
formal/ScoreQuantFormal/Leverage.lean`, `declaration:
ScoreQuantFormal.leverage_inequality` and `statement_audit:
AUDITS/FORMALIZATION-D-LEVERAGE-001.md`:

1. Add the three missing disclaimers of F4 to the spec's "Not part of this
   specification" section, and the `assumptions` list of F5 to
   `D-LEVERAGE.json`.
2. Optionally apply F2(a) or F2(b), and the two docstring clarifications noted
   under F3 and under the `PosDef` confirmation.
3. Update `KNOWN_RESULTS/04-d-optimality.md` §D4, which currently says
   "Statement not separately frozen, so no `formal_proof` field yet" and names
   only `centroid_leverage_bound` and `leverage_bound`; it should name
   `leverage_inequality`, the spec, and this audit.

None of these changes the mathematics of the frozen proposition except the
optional F2(b), which is analyzed above and does not require a further statement
audit. Every other change to `LeverageAssumptions` or `LeverageConclusion` does.
