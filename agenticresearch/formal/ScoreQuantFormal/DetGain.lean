import ScoreQuantFormal.Relocation
import ScoreQuantFormal.DetGainSpec
import Mathlib.LinearAlgebra.Matrix.SchurComplement
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

/-!
# D3: the exact determinant ratio of a rank-two relocation

For an invertible symmetric `I` with `H = I⁻¹`, and the three quadratic
products `q_aa = u_aᵀ H u_a`, `q_bb = u_bᵀ H u_b`, `q_ab = u_aᵀ H u_b`,

```
det(I + α u_a u_aᵀ − β u_b u_bᵀ) = det I · [(1 + α q_aa)(1 − β q_bb) + α β q_ab²].
```

`qform`, `detRatio` and the identity itself are frozen in `DetGainSpec.lean`.
The bracket is exactly `1 + exchangeExcess` for the `exchangeExcess` frozen in
`ScalarExchangeSpec.lean`; `detRatio_eq_one_add_exchangeExcess` records that
join, which is what lets the audited scalar bound be applied to the matrix
layer.

The proof is the Weinstein–Aronszajn identity `Matrix.det_one_add_mul_comm`,
which turns the `d × d` determinant into a `2 × 2` one.

Registry claim: `D-LOGDET-GAIN`, `KNOWN_RESULTS/04-d-optimality.md` §D3.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d : ℕ}

@[simp]
theorem qform_smul_left (H : Matrix (Fin d) (Fin d) ℝ) (c : ℝ) (u v : Fin d → ℝ) :
    qform H (c • u) v = c * qform H u v := by
  simp [qform, dotProduct, Finset.mul_sum, mul_assoc]

/-- A symmetric metric gives a symmetric quadratic form. -/
theorem qform_comm {H : Matrix (Fin d) (Fin d) ℝ} (hH : Hᵀ = H) (u v : Fin d → ℝ) :
    qform H u v = qform H v u := by
  simp only [qform, dotProduct, Matrix.mulVec, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun y _ => ?_
  have : H y x = H x y := by
    have := congrFun (congrFun hH x) y
    simpa [Matrix.transpose_apply] using this
  simp [this]
  ring

theorem qform_sub_right (H : Matrix (Fin d) (Fin d) ℝ) (u v w : Fin d → ℝ) :
    qform H u (v - w) = qform H u v - qform H u w := by
  simp [qform, Matrix.mulVec_sub, dotProduct_sub]

theorem qform_sub_left (H : Matrix (Fin d) (Fin d) ℝ) (u v w : Fin d → ℝ) :
    qform H (u - v) w = qform H u w - qform H v w := by
  simp [qform, sub_dotProduct]

/-- Expansion of the quadratic form of a difference, in a symmetric metric. -/
theorem qform_sub_self {H : Matrix (Fin d) (Fin d) ℝ} (hH : Hᵀ = H) (u v : Fin d → ℝ) :
    qform H (u - v) (u - v) = qform H u u - 2 * qform H u v + qform H v v := by
  rw [qform_sub_left, qform_sub_right, qform_sub_right, qform_comm hH v u]
  ring

/-- Entries of a `2 × 2` Gram-type product are quadratic forms of the rows. -/
theorem gram_entry (H : Matrix (Fin d) (Fin d) ℝ) (P Q : Matrix (Fin 2) (Fin d) ℝ)
    (k l : Fin 2) : (P * (H * Qᵀ)) k l = qform H (P k) (Q l) := rfl

/-- **D3, the exact determinant ratio of a rank-two relocation.** -/
theorem det_add_rank_two (I : Matrix (Fin d) (Fin d) ℝ) (hsymm : I.IsSymm)
    (hunit : IsUnit I.det) (α β : ℝ) (ua ub : Fin d → ℝ) :
    (I + (α • vecMulVec ua ua - β • vecMulVec ub ub)).det
      = I.det * detRatio α β (qform I⁻¹ ua ua) (qform I⁻¹ ub ub) (qform I⁻¹ ua ub) := by
  classical
  have hHsymm : (I⁻¹)ᵀ = I⁻¹ := by rw [Matrix.transpose_nonsing_inv, hsymm.eq]
  set A : Matrix (Fin 2) (Fin d) ℝ := Matrix.of ![ua, ub] with hA
  set B : Matrix (Fin 2) (Fin d) ℝ := Matrix.of ![α • ua, (-β) • ub] with hB
  have hprod : α • vecMulVec ua ua - β • vecMulVec ub ub = Aᵀ * B := by
    ext i j
    simp [hA, hB, Matrix.mul_apply, Fin.sum_univ_two, vecMulVec_apply]
    ring
  have hfactor : I + Aᵀ * B = I * (1 + I⁻¹ * (Aᵀ * B)) := by
    rw [Matrix.mul_add, Matrix.mul_one, ← Matrix.mul_assoc,
      Matrix.mul_nonsing_inv I hunit, Matrix.one_mul]
  rw [hprod, hfactor, Matrix.det_mul]
  congr 1
  have hcomm : (1 + I⁻¹ * (Aᵀ * B)).det = (1 + B * (I⁻¹ * Aᵀ)).det := by
    simpa [Matrix.mul_assoc] using Matrix.det_one_add_mul_comm (I⁻¹ * Aᵀ) B
  rw [hcomm, Matrix.det_fin_two]
  have hrowB0 : B 0 = α • ua := rfl
  have hrowB1 : B 1 = (-β) • ub := rfl
  have hrowA0 : A 0 = ua := rfl
  have hrowA1 : A 1 = ub := rfl
  simp only [Matrix.add_apply, Matrix.one_apply_eq,
    Matrix.one_apply_ne (show (0 : Fin 2) ≠ 1 by decide),
    Matrix.one_apply_ne (show (1 : Fin 2) ≠ 0 by decide),
    gram_entry, hrowB0, hrowB1, hrowA0, hrowA1, qform_smul_left,
    qform_comm hHsymm ub ua, detRatio]
  ring

/-- The determinant ratio and the frozen scalar `exchangeExcess` are the same
quantity: `detRatio = 1 + exchangeExcess`. This is the join between the matrix
layer and the audited scalar core. -/
theorem detRatio_eq_one_add_exchangeExcess
    (sourceWeight sourceMass destinationMass qaa qbb qab : ℝ) :
    detRatio (alpha sourceWeight sourceMass) (beta sourceWeight destinationMass) qaa qbb qab
      = 1 + exchangeExcess sourceWeight sourceMass destinationMass qaa qbb qab := by
  unfold detRatio exchangeExcess
  ring

/-- **D3 discharges its frozen statement.** This is the declaration that
`D-LOGDET-GAIN` carries as its `formal_proof`. -/
theorem det_relocation_gain (I : Matrix (Fin d) (Fin d) ℝ) (α β : ℝ)
    (ua ub : Fin d → ℝ) : DetGainConclusion I α β ua ub := by
  rintro ⟨hsymm, hunit⟩
  have hid := det_add_rank_two I hsymm hunit α β ua ub
  refine ⟨hid, fun hdet hratio => ?_⟩
  rw [hid, Real.log_mul (ne_of_gt hdet) (ne_of_gt hratio)]
  ring

end

end ScoreQuantFormal
