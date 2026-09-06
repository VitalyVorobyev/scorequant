import ScoreQuantFormal.ConfigSpec
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# D3 specification: the exact determinant ratio of a rank-two relocation

This file is the reviewed statement boundary for registry claim `D-LOGDET-GAIN`,
proved in `KNOWN_RESULTS/04-d-optimality.md` §D3. A prover may change
`DetGain.lean`, but must not change this file without a new statement audit.

The registry statement reads:

> With `H = I⁻¹`, `q_aa = u_aᵀ H u_a`, `q_bb = u_bᵀ H u_b`,
> `q_ab = u_aᵀ H u_b`, the determinant ratio is
> `(1 + α q_aa)(1 − β q_bb) + α β q_ab²`.

The correspondence is:

| registry clause | this file |
|---|---|
| `H = I⁻¹` | `qform I⁻¹ _ _` |
| `q_aa`, `q_bb`, `q_ab` | the three `qform` arguments of `detRatio` |
| the rank-two update `α u_a u_aᵀ − β u_b u_bᵀ` | the matrix added to `I` |
| the ratio `(1 + α q_aa)(1 − β q_bb) + α β q_ab²` | `detRatio` |
| the boxed `ΔF_D = log[·]` of §D3, `F_D = log det I` | `DetGainConclusion`, second component |
| the identity itself | `DetGainConclusion`, first component |

`qform` and `detRatio` are defined here rather than in `DetGain.lean` because
both appear inside this conclusion, and `qform` also appears inside the frozen
`centroidSeparation` of `ExchangeVoronoiSpec.lean`.

**Hypotheses match the registry's, which an audit corrected to match these.**
The node once assumed "current and candidate information positive definite";
the identity needs only that `I` is symmetric with a unit determinant, and needs
nothing at all of the candidate, so the node's `assumptions` field now records
that weaker pair. The relation is therefore equality, not headroom: this
statement is neither stronger nor weaker than the claim it is frozen against.
An audit that finds the Lean *stronger* than the node is a mismatch, not a
hardening.

Both conjuncts are load-bearing, and the statement is false without either.
Dropping `IsSymm`: the general law is `(1 + α q_aa)(1 − β q_bb) + α β q_ab q_ba`,
and symmetry is exactly what collapses `q_ab q_ba` to `q_ab²`; for
`I = ![![1, 1], ![0, 1]]`, `u_a = e₁`, `u_b = e₂`, `α = β = 1` the true
determinant is `0` while the right-hand side is `1`. Dropping `IsUnit I.det`:
Mathlib's `I⁻¹` is the junk value `0` at a singular `I`, and `d = 1`, `I = 0`,
`α = 1`, `β = 0`, `u_a = 1` gives left side `1`, right side `0`.

**The `F_D` form is the second component,** because the claim is named for a
log-determinant gain and §D3 boxes `ΔF_D = log[·]`. It carries the positivity
that taking a logarithm needs, and only that; the identity itself needs none.
Freezing the determinant form alone would have left the claim covered in part,
which the protocol does not allow to carry `formal_proof`.

**Not part of this specification.**

* Any sign or positivity assertion. The first component is an identity between
  two real numbers; that `detRatio > 0`, that `det I > 0`, or that the candidate
  is itself positive definite is not claimed. The second component *assumes*
  both positivities rather than establishing either, and says nothing when one
  fails.
* Any computational content, despite the claim node's `role` naming an
  "`O(d²)`-type candidate evaluation". This is an identity between real numbers;
  nothing here is an algorithm, a cost model, or a claim that either side is the
  cheaper one to evaluate.
* Any connection between `α`, `β` and a relocation. This statement holds for
  arbitrary real `α`, `β` and arbitrary vectors `u_a`, `u_b`; that the D2
  coefficients are what get substituted is D2's business. The join to the
  audited scalar core is `detRatio_eq_one_add_exchangeExcess`, which lives in
  the editable `DetGain.lean` and is therefore *not* part of this frozen
  boundary — it is convenience, and a reader must not treat it as audited.
* Any tie between `I` and the retained information of a labeling. `I` here is
  an arbitrary symmetric nonsingular matrix; nothing connects it to `fisher`,
  and nothing claims either is a statistical Fisher information — see
  `LeverageSpec.lean`, which carries the same disclaimer for `fisher` itself.
* Rank-two updates of a non-symmetric or singular `I`.
* Any statement about the Python/JAX implementation, or about floating-point
  conditioning of the inverse `I⁻¹`.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d : ℕ}

/-- The quadratic form `uᵀ H v`. -/
def qform (H : Matrix (Fin d) (Fin d) ℝ) (u v : Fin d → ℝ) : ℝ :=
  u ⬝ᵥ H.mulVec v

/-- The determinant ratio of a rank-two update, as a function of the two
relocation coefficients and the three quadratic products. -/
def detRatio (α β qaa qbb qab : ℝ) : ℝ :=
  (1 + α * qaa) * (1 - β * qbb) + α * β * qab ^ 2

/-- The hypotheses of D3: `I` is symmetric and nonsingular — the registry's
`assumptions` field verbatim. Positive definiteness of the candidate is not
needed and is not assumed. -/
def DetGainAssumptions (I : Matrix (Fin d) (Fin d) ℝ) : Prop :=
  I.IsSymm ∧ IsUnit I.det

/-- **The frozen statement of `D-LOGDET-GAIN`,** both components of the claim:
the determinant identity, and the `F_D = log det I` gain it yields wherever the
logarithm is defined. -/
def DetGainConclusion (I : Matrix (Fin d) (Fin d) ℝ) (α β : ℝ) (ua ub : Fin d → ℝ) :
    Prop :=
  DetGainAssumptions I →
    (I + (α • vecMulVec ua ua - β • vecMulVec ub ub)).det
        = I.det * detRatio α β (qform I⁻¹ ua ua) (qform I⁻¹ ub ub) (qform I⁻¹ ua ub) ∧
      (0 < I.det →
        0 < detRatio α β (qform I⁻¹ ua ua) (qform I⁻¹ ub ub) (qform I⁻¹ ua ub) →
          Real.log (I + (α • vecMulVec ua ua - β • vecMulVec ub ub)).det
              - Real.log I.det
            = Real.log
                (detRatio α β (qform I⁻¹ ua ua) (qform I⁻¹ ub ub) (qform I⁻¹ ua ub)))

end

end ScoreQuantFormal
