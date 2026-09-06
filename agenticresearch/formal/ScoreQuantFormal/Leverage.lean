import ScoreQuantFormal.DetGain
import ScoreQuantFormal.LeverageSpec
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
# D4: the centroid leverage inequality

Writing `A` for the `d × K` matrix with columns `√W_c · μ_c`, the retained
information factors as `I = A Aᵀ`, so `P = Aᵀ I⁻¹ A` is a symmetric idempotent
— an orthogonal projector. For any `v`,

```
vᵀ P v ≤ vᵀ v,
```

because `Q = 1 − P` is symmetric idempotent too, hence `vᵀ Q v = ‖Q v‖² ≥ 0`.
Choosing `v` supported on the two cells `a, b` with entries `1/√W_a` and
`−1/√W_b` gives `A v = μ_a − μ_b` and `vᵀ v = 1/W_a + 1/W_b`, so

```
(μ_a − μ_b)ᵀ I⁻¹ (μ_a − μ_b) ≤ 1/W_a + 1/W_b.
```

This is precisely the hypothesis `qDelta ≤ 1/sourceMass + 1/destinationMass`
that `ScalarExchangeSpec.lean` assumes, so this file is what discharges it.
Both halves of the claim are frozen in `LeverageSpec.lean`.

Registry claim: `D-LEVERAGE`, `KNOWN_RESULTS/04-d-optimality.md` §D4.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N K : ℕ}

/-- A symmetric idempotent matrix contracts every quadratic form. -/
theorem qform_le_of_symm_idem {P : Matrix (Fin K) (Fin K) ℝ}
    (hsymm : Pᵀ = P) (hidem : P * P = P) (v : Fin K → ℝ) :
    v ⬝ᵥ P.mulVec v ≤ v ⬝ᵥ v := by
  set Q : Matrix (Fin K) (Fin K) ℝ := 1 - P with hQ
  have hQsymm : Qᵀ = Q := by
    rw [hQ, Matrix.transpose_sub, Matrix.transpose_one, hsymm]
  have hQidem : Q * Q = Q := by
    rw [hQ, Matrix.sub_mul, Matrix.mul_sub, Matrix.mul_sub, Matrix.one_mul,
      Matrix.mul_one, Matrix.one_mul, hidem]
    abel
  have hnonneg : 0 ≤ v ⬝ᵥ Q.mulVec v := by
    have hstep : v ⬝ᵥ Q.mulVec v = (Q.mulVec v) ⬝ᵥ (Q.mulVec v) := by
      conv_lhs => rw [← hQidem]
      rw [← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, hQsymm]
    rw [hstep]
    exact Finset.sum_nonneg fun i _ => mul_self_nonneg _
  have hsplit : v ⬝ᵥ Q.mulVec v = v ⬝ᵥ v - v ⬝ᵥ P.mulVec v := by
    rw [hQ, Matrix.sub_mulVec, Matrix.one_mulVec, dotProduct_sub]
  rw [hsplit] at hnonneg
  linarith

/-- The `d × K` factor whose columns are `√W_c · μ_c`. -/
def rootFactor (S : Sample d N) (z : Fin N → Fin K) : Matrix (Fin d) (Fin K) ℝ :=
  Matrix.of fun i c => Real.sqrt (cellMass S z c) * centroid S z c i

/-- The retained information factors as `A Aᵀ`. -/
theorem fisher_eq_rootFactor_mul (S : Sample d N) (z : Fin N → Fin K)
    (hne : ∀ c, (cell z c).Nonempty) :
    fisher S z = rootFactor S z * (rootFactor S z)ᵀ := by
  ext i j
  rw [fisher, Matrix.sum_apply, Matrix.mul_apply]
  refine Finset.sum_congr rfl fun c _ => ?_
  have hW : 0 < cellMass S z c := cellMass_pos (hne c)
  have hroot : Real.sqrt (cellMass S z c) * Real.sqrt (cellMass S z c) = cellMass S z c :=
    Real.mul_self_sqrt hW.le
  rw [cellBlock_eq_centroid (cellMass S z c) (ne_of_gt hW) (cellSum S z c)]
  simp only [rootFactor, Matrix.of_apply, Matrix.transpose_apply, Matrix.smul_apply,
    smul_eq_mul, vecMulVec_apply]
  calc cellMass S z c * (centroid S z c i * centroid S z c j)
      = (Real.sqrt (cellMass S z c) * Real.sqrt (cellMass S z c)) *
          (centroid S z c i * centroid S z c j) := by rw [hroot]
    _ = Real.sqrt (cellMass S z c) * centroid S z c i *
          (Real.sqrt (cellMass S z c) * centroid S z c j) := by ring

/-- Sums supported on two distinct indices collapse to two terms. -/
private theorem sum_pair_of_support {a b : Fin K} (hab : a ≠ b) (f : Fin K → ℝ)
    (hzero : ∀ c, c ≠ a → c ≠ b → f c = 0) :
    ∑ c, f c = f a + f b := by
  classical
  rw [← Finset.sum_subset (Finset.subset_univ ({a, b} : Finset (Fin K))) ?_,
    Finset.sum_pair hab]
  intro c _ hc
  simp only [Finset.mem_insert, Finset.mem_singleton, not_or] at hc
  exact hzero c hc.1 hc.2

/-- The projector bound behind D4: any direction realized as `A v` has quadratic
form at most `vᵀ v` in the `I⁻¹` metric. -/
theorem qform_rootFactor_le (S : Sample d N) (z : Fin N → Fin K)
    (hne : ∀ c, (cell z c).Nonempty) (hpd : (fisher S z).PosDef) (v : Fin K → ℝ) :
    qform (fisher S z)⁻¹ ((rootFactor S z).mulVec v) ((rootFactor S z).mulVec v)
      ≤ v ⬝ᵥ v := by
  classical
  set I := fisher S z with hI
  set A := rootFactor S z with hA
  have hfac : I = A * Aᵀ := fisher_eq_rootFactor_mul S z hne
  have hunit : IsUnit I.det := (Matrix.isUnit_iff_isUnit_det I).mp hpd.isUnit
  have hIsymm : (I⁻¹)ᵀ = I⁻¹ := by
    rw [Matrix.transpose_nonsing_inv, hI, fisher_isSymm S z]
  set P : Matrix (Fin K) (Fin K) ℝ := Aᵀ * I⁻¹ * A with hP
  have hPsymm : Pᵀ = P := by
    rw [hP, Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose, hIsymm,
      Matrix.mul_assoc]
  have hPidem : P * P = P := by
    rw [hP,
      show Aᵀ * I⁻¹ * A * (Aᵀ * I⁻¹ * A) = Aᵀ * I⁻¹ * (A * Aᵀ) * I⁻¹ * A by
        simp only [Matrix.mul_assoc],
      ← hfac,
      show Aᵀ * I⁻¹ * I * I⁻¹ * A = Aᵀ * (I⁻¹ * I) * I⁻¹ * A by simp only [Matrix.mul_assoc],
      Matrix.nonsing_inv_mul I hunit, Matrix.mul_one]
  have hquad : qform I⁻¹ (A.mulVec v) (A.mulVec v) = v ⬝ᵥ P.mulVec v := by
    -- `qform` stays folded so the rewrites land on the right-hand side only.
    rw [hP, show Aᵀ * I⁻¹ * A = Aᵀ * (I⁻¹ * A) by rw [Matrix.mul_assoc],
      ← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec,
      Matrix.vecMul_transpose]
    rfl
  rw [hquad]
  exact qform_le_of_symm_idem hPsymm hPidem v

/-- Reciprocal of a square root, used to turn test-vector norms into cell masses. -/
private theorem inv_sqrt_sq {x : ℝ} (hx : 0 < x) :
    (Real.sqrt x)⁻¹ * (Real.sqrt x)⁻¹ = x⁻¹ := by
  rw [← mul_inv, Real.mul_self_sqrt hx.le]

/-- **D4, first half: the single-centroid leverage inequality** `μ_cᵀ I⁻¹ μ_c ≤ 1/W_c`. -/
theorem centroid_leverage_bound (S : Sample d N) (z : Fin N → Fin K)
    (hne : ∀ c, (cell z c).Nonempty) (hpd : (fisher S z).PosDef) (c : Fin K) :
    qform (fisher S z)⁻¹ (centroid S z c) (centroid S z c) ≤ (cellMass S z c)⁻¹ := by
  classical
  have hWc : 0 < cellMass S z c := cellMass_pos (hne c)
  have hsc : Real.sqrt (cellMass S z c) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hWc)
  set v : Fin K → ℝ :=
    Function.update (fun _ => (0 : ℝ)) c (Real.sqrt (cellMass S z c))⁻¹ with hv
  have hvc : v c = (Real.sqrt (cellMass S z c))⁻¹ := by rw [hv, Function.update_self]
  have hvo : ∀ e, e ≠ c → v e = 0 := fun e he => by
    rw [hv, Function.update_of_ne he]
  have hAv : (rootFactor S z).mulVec v = centroid S z c := by
    funext i
    rw [Matrix.mulVec, dotProduct,
      Finset.sum_eq_single c (fun e _ he => by rw [hvo e he, mul_zero])
        (fun h => absurd (Finset.mem_univ c) h)]
    rw [hvc]
    simp only [rootFactor, Matrix.of_apply]
    rw [show Real.sqrt (cellMass S z c) * centroid S z c i * (Real.sqrt (cellMass S z c))⁻¹
        = centroid S z c i *
          (Real.sqrt (cellMass S z c) * (Real.sqrt (cellMass S z c))⁻¹) by ring,
      mul_inv_cancel₀ hsc, mul_one]
  have hvv : v ⬝ᵥ v = (cellMass S z c)⁻¹ := by
    rw [dotProduct,
      Finset.sum_eq_single c (fun e _ he => by rw [hvo e he, mul_zero])
        (fun h => absurd (Finset.mem_univ c) h),
      hvc, inv_sqrt_sq hWc]
  rw [← hAv, ← hvv]
  exact qform_rootFactor_le S z hne hpd v

/-- **D4, second half: the centroid-difference leverage inequality.** -/
theorem leverage_bound (S : Sample d N) (z : Fin N → Fin K)
    (hne : ∀ c, (cell z c).Nonempty) (hpd : (fisher S z).PosDef) (a b : Fin K)
    (hab : a ≠ b) :
    qform (fisher S z)⁻¹ (centroid S z a - centroid S z b) (centroid S z a - centroid S z b)
      ≤ (cellMass S z a)⁻¹ + (cellMass S z b)⁻¹ := by
  classical
  have hWa : 0 < cellMass S z a := cellMass_pos (hne a)
  have hWb : 0 < cellMass S z b := cellMass_pos (hne b)
  have hsa : Real.sqrt (cellMass S z a) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hWa)
  have hsb : Real.sqrt (cellMass S z b) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hWb)
  -- The test vector supported on cells `a` and `b`.
  set v : Fin K → ℝ :=
    Function.update (Function.update (fun _ => (0 : ℝ)) a (Real.sqrt (cellMass S z a))⁻¹)
      b (-(Real.sqrt (cellMass S z b))⁻¹) with hv
  have hva : v a = (Real.sqrt (cellMass S z a))⁻¹ := by
    rw [hv, Function.update_of_ne hab, Function.update_self]
  have hvb : v b = -(Real.sqrt (cellMass S z b))⁻¹ := by
    rw [hv, Function.update_self]
  have hvc : ∀ c, c ≠ a → c ≠ b → v c = 0 := by
    intro c hca hcb
    rw [hv, Function.update_of_ne hcb, Function.update_of_ne hca]
  have hAv : (rootFactor S z).mulVec v = centroid S z a - centroid S z b := by
    funext i
    rw [Matrix.mulVec, dotProduct,
      sum_pair_of_support hab (fun c => rootFactor S z i c * v c)
        (fun c hca hcb => by rw [hvc c hca hcb, mul_zero])]
    rw [hva, hvb]
    simp only [rootFactor, Matrix.of_apply, Pi.sub_apply]
    rw [show Real.sqrt (cellMass S z a) * centroid S z a i * (Real.sqrt (cellMass S z a))⁻¹
          = centroid S z a i *
            (Real.sqrt (cellMass S z a) * (Real.sqrt (cellMass S z a))⁻¹) by ring,
      show Real.sqrt (cellMass S z b) * centroid S z b i * -(Real.sqrt (cellMass S z b))⁻¹
          = -(centroid S z b i *
            (Real.sqrt (cellMass S z b) * (Real.sqrt (cellMass S z b))⁻¹)) by ring,
      mul_inv_cancel₀ hsa, mul_inv_cancel₀ hsb, mul_one, mul_one]
    ring
  have hvv : v ⬝ᵥ v = (cellMass S z a)⁻¹ + (cellMass S z b)⁻¹ := by
    rw [dotProduct,
      sum_pair_of_support hab (fun c => v c * v c)
        (fun c hca hcb => by rw [hvc c hca hcb, mul_zero])]
    rw [hva, hvb, neg_mul_neg, inv_sqrt_sq hWa, inv_sqrt_sq hWb]
  rw [← hAv, ← hvv]
  exact qform_rootFactor_le S z hne hpd v

/-- **D4 discharges its frozen statement,** both halves of the bundled claim.
This is the declaration that `D-LEVERAGE` carries as its `formal_proof`. -/
theorem leverage_inequality (S : Sample d N) (z : Fin N → Fin K) :
    LeverageConclusion S z := by
  rintro ⟨hne, hpd⟩
  refine ⟨fun c => centroid_leverage_bound S z hne hpd c, fun a b => ?_⟩
  rcases eq_or_ne a b with rfl | hab
  · -- The registry statement puts no restriction on `a` and `b`; the coincident
    -- case is `0 ≤ 2/W_a`, true but empty of content.
    have hW : (0 : ℝ) ≤ (cellMass S z a)⁻¹ := le_of_lt (inv_pos.mpr (cellMass_pos (hne a)))
    simpa [qform, sub_self] using add_nonneg hW hW
  · exact leverage_bound S z hne hpd a b hab

end

end ScoreQuantFormal
