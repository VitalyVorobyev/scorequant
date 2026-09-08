import ScoreQuantFormal.ExchangeVoronoiSpec
import ScoreQuantFormal.MergeSpec

/-!
# The compiled-predictor specification

This file is the reviewed statement boundary for

* registry claim `D-FINITE-INDUCTIVE-CLOSURE` (§D6, manuscript v9 Theorem 3's deployable half),

proved in `KNOWN_RESULTS/04-d-optimality.md` §D6. A prover may change `Closure.lean`, but must
not change this file without a new statement audit.

The registry statement reads:

> Every exact zero-tolerance one-point-exchange-stable positive-definite finite D solution on
> merged distinct positive-weight score atoms compiles to
> `q_hat(s)=argmin_b (s-mu_b)^T I_hat^{-1}(s-mu_b)`, reproducing all merged-atom training
> labels strictly; original duplicate rows inherit the merged label. Positive solver tolerance
> gives only a tolerance-stamped boundary-disagreement guarantee.

The correspondence is:

| registry clause | this file |
|---|---|
| merged distinct positive-weight score atoms | `VoronoiAssumptions` (first conjunct) and `Sample.weight_pos` |
| exactly `K` nonempty cells | `VoronoiAssumptions` (**second** conjunct) |
| positive-definite | `VoronoiAssumptions` (third conjunct) |
| exact zero-tolerance one-point exchange stability | `ExchangeStable` |
| `argmin_b (s-mu_b)^T I_hat^{-1}(s-mu_b)` | `IsNearestCentroidRule` |
| "reproducing all merged-atom training labels" | `ClosureConclusion`, second conjunct |
| "strictly", i.e. without a tie breaker | quantification over *every* rule satisfying `IsNearestCentroidRule`: no tie-breaking convention appears, and the conclusion holds for all of them |
| the compiled rule exists at all | `ClosureConclusion`, first conjunct |
| "original duplicate rows inherit the merged label" | `ClosureDuplicateConclusion`, whose rule is built from the *unmerged* sample |

**The nonempty-cells hypothesis is load-bearing, not bookkeeping.** Without it an empty cell
`c` has `cellMass = 0` and therefore the junk centroid `(0 : ℝ)⁻¹ • 0 = 0`, which a training
row sitting at the origin is exactly as near to as it is to its own centroid. Then the argmin
is not a singleton and a conforming rule may return the empty cell: `d = 1`, `N = 2`, `K = 3`,
scores `(-2, 0)`, unit weights, `z = (0, 1)` is stable, positive definite and injective, yet
row 1 ties between cells `1` and `2`. The registry statement omitted this hypothesis and has
been amended to carry it, as `D-EXCHANGE-IMPLIES-VORONOI`'s node already does.

**Why the rule is quantified rather than constructed.** A definition of `argmin` must break
ties somehow, and a theorem about one particular tie-break would not say what the claim says.
Stating the conclusion for *every* nearest-centroid rule is strictly stronger and is the exact
content of "without a tie breaker": at an exactly stable state no training row is ever tied,
so no convention can matter.

`IsNearestCentroidRule` is deliberately stated with `≤`, the weaker requirement on a rule,
which makes the theorem about it stronger.

**Why existence is a conjunct rather than its own proposition.** `formal_proof.declaration`
names one theorem whose type is the frozen conclusion, so a node's coverage cannot be spread
over several frozen `Prop`s. Existence and reproduction are therefore one `Prop`. Existence is
also what stops the `∀ q` from being satisfied by having no rules at all, and it is guarded by
`0 < d` because it is *false* without it: at `K = 0` there is no function into `Fin 0`, and
`VoronoiAssumptions` is satisfiable at `d = K = N = 0`. For `0 < d` the guard is discharged
rather than assumed — `K = 0` would make `fisher` the zero matrix, whose determinant vanishes,
contradicting positive-definiteness — so `K ≥ 1` and a minimiser exists.

**Not part of this specification.**

* The registry statement's third sentence, on positive solver tolerance. That is a different
  theorem with a weaker hypothesis, and it is now carried by its own claim node,
  `D-COMPILE-TOLERANCE-GUARANTEE`, which nothing here formalizes. Every real solver lives in
  that regime, and `predict_scores`' lowest-index tie-break exists precisely because
  uniqueness does not survive it.
* D5's *second* duplicate branch — "labels constant on every duplicate class" — remains
  unformalized. `ClosureDuplicateConclusion` **assumes** the inherited labeling `z ∘ map`,
  which is constant on each class by construction; it does not derive constancy from
  stability, and nothing here says an arbitrary labeling of duplicated rows must be constant
  on a class.
* Equivalence of merged and unmerged exchange stability. `ExchangeStable S z` is a hypothesis
  on the merged side only; nothing states that the unmerged configuration is stable, or that
  stability transfers in either direction.
* Any behaviour of the rule off the training scores. `IsNearestCentroidRule` constrains `q` at
  every `s`, which is more than the conclusion needs, and no conclusion here describes `q` at
  a score that is not a training row's. In particular no uniqueness is claimed off the sample:
  two conforming rules may disagree at any non-training `s`.
* The *shape* of the induced decision regions — nothing here says they are an affine-max or
  Voronoi partition of score space, which is D1's reading — and nothing about measurability
  of `q`.
* Zero-weight rows, capacity, balance, or minimum-cell-mass constraints.
* Singular or pseudodeterminant objectives, and projected-subspace variants.
* Any claim that the compiled rule is *optimal*, or that it agrees with `z` off the sample.
  The conclusion is about the `N` training scores only; nothing here constrains a fresh `s`.
* Population, atomless and asymptotic statements, and score-estimation error.
* Any statement about the Python/JAX implementation. Nothing connects `fisher`, `centroid` or
  `IsNearestCentroidRule` to `compile_quantizer`, `predict_scores`, the `Quantizer` artifact,
  floating-point conditioning, or `rank_rtol`.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N N' K : ℕ}

/-- A deterministic label rule that always returns *a* nearest centroid in the `I⁻¹` metric.
Ties may be broken arbitrarily, or inconsistently; nothing here constrains that choice. -/
def IsNearestCentroidRule (S : Sample d N) (z : Fin N → Fin K)
    (q : (Fin d → ℝ) → Fin K) : Prop :=
  ∀ s c, mahalanobis (fisher S z)⁻¹ s (centroid S z (q s))
    ≤ mahalanobis (fisher S z)⁻¹ s (centroid S z c)

/-- **The frozen statement of `D-FINITE-INDUCTIVE-CLOSURE`.** At an exactly stable state a
nearest-centroid rule exists, and every such rule, however it breaks ties, reproduces every
training label. -/
def ClosureConclusion (S : Sample d N) (z : Fin N → Fin K) : Prop :=
  VoronoiAssumptions S z → ExchangeStable S z →
    (0 < d → ∃ q, IsNearestCentroidRule S z q) ∧
      ∀ q, IsNearestCentroidRule S z q → ∀ i, q (S.score i) = z i

/-- **The frozen duplicate half of `D-FINITE-INDUCTIVE-CLOSURE`:** a rule built from the
*unmerged* sample and its inherited labeling assigns every original row the label its atom
carries.

The rule is quantified over `IsNearestCentroidRule S' (z ∘ map) q`, not
`IsNearestCentroidRule S z q`. This is the whole content of the clause. `VoronoiAssumptions S'
(z ∘ map)` is false whenever a duplicate exists — the scores are not injective — so the
stability and nonsingularity hypotheses can only live on the merged side, and reaching the
unmerged rule from them requires knowing that merging changed neither the centroids nor the
retained information. That is `MergeConclusion`. Stating the rule on the merged side instead
would make this clause follow from `score_eq` alone and assert nothing. -/
def ClosureDuplicateConclusion (S' : Sample d N') (S : Sample d N) (map : Fin N' → Fin N)
    (z : Fin N → Fin K) : Prop :=
  IsMerge S' S map → VoronoiAssumptions S z → ExchangeStable S z →
    ∀ q, IsNearestCentroidRule S' (z ∘ map) q → ∀ j, q (S'.score j) = (z ∘ map) j

end

end ScoreQuantFormal
