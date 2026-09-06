import ScoreQuantFormal.ConfigSpec
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
# Lemmas about finite weighted configurations

The definitions themselves — sample, cells, masses, sums, centroids, retained
information, Mahalanobis form — are frozen in `ConfigSpec.lean`, because the
frozen statements of D2, D3, D4 and D5 are written in them. This file holds
everything provable *about* them.

Cells are represented by their *unnormalized* weighted sum `T_c` rather than by
their centroid `μ_c = W_c⁻¹ • T_c`. The two descriptions agree — `cellBlock`
is `W⁻¹ • (T Tᵀ) = W • (μ μᵀ)` — but the unnormalized one turns the relocation
algebra of `Relocation.lean` into a two-variable identity in `(W, T)` instead of
centroid bookkeeping.

Nothing here is a theorem about ScoreQuant; these are the two rank-one update
identities that `Relocation.lean` assembles into D2, plus bookkeeping.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N K : ℕ}

@[simp]
theorem cellBlock_apply (W : ℝ) (T : Fin d → ℝ) (i j : Fin d) :
    cellBlock W T i j = W⁻¹ * (T i * T j) := by
  simp [cellBlock, vecMulVec_apply]

/-- `cellBlock` in centroid form: `W⁻¹ • (T Tᵀ) = W • (μ μᵀ)`. -/
theorem cellBlock_eq_centroid (W : ℝ) (hW : W ≠ 0) (T : Fin d → ℝ) :
    cellBlock W T = W • vecMulVec (W⁻¹ • T) (W⁻¹ • T) := by
  ext i j
  simp only [Matrix.smul_apply, smul_eq_mul, cellBlock_apply, vecMulVec_apply,
    Pi.smul_apply]
  field_simp

/-- Removing weight `w` carried by score `s` from a cell of mass `W` and sum `T`
changes that cell's block by a positive rank-one term along `μ - s` minus the
row's own rank-one term. -/
theorem cellBlock_erase (W w : ℝ) (hw : 0 < w) (hWw : w < W) (T s : Fin d → ℝ) :
    cellBlock (W - w) (T - w • s) - cellBlock W T
      = (w * W / (W - w)) • vecMulVec (W⁻¹ • T - s) (W⁻¹ • T - s) - w • vecMulVec s s := by
  have hW : (0 : ℝ) < W := lt_trans hw hWw
  have hWne : W ≠ 0 := ne_of_gt hW
  have hDiff : W - w ≠ 0 := ne_of_gt (sub_pos.mpr hWw)
  ext i j
  simp only [Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, cellBlock_apply,
    vecMulVec_apply, Pi.sub_apply, Pi.smul_apply]
  field_simp
  ring

/-- Inserting weight `w` carried by score `s` into a cell of mass `W` and sum `T`
changes that cell's block by the row's own rank-one term minus a positive
rank-one term along `μ - s`. -/
theorem cellBlock_insert (W w : ℝ) (hw : 0 < w) (hW : 0 < W) (T s : Fin d → ℝ) :
    cellBlock (W + w) (T + w • s) - cellBlock W T
      = w • vecMulVec s s - (w * W / (W + w)) • vecMulVec (W⁻¹ • T - s) (W⁻¹ • T - s) := by
  have hWne : W ≠ 0 := ne_of_gt hW
  have hSum : W + w ≠ 0 := ne_of_gt (add_pos hW hw)
  ext i j
  simp only [Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul,
    cellBlock_apply, vecMulVec_apply, Pi.sub_apply, Pi.add_apply, Pi.smul_apply]
  field_simp
  ring

@[simp]
theorem mem_cell {z : Fin N → Fin K} {c : Fin K} {i : Fin N} :
    i ∈ cell z c ↔ z i = c := by
  simp [cell]

/-- A cell with at least one row has strictly positive mass. -/
theorem cellMass_pos {S : Sample d N} {z : Fin N → Fin K} {c : Fin K}
    (h : (cell z c).Nonempty) : 0 < cellMass S z c := by
  rw [cellMass]
  exact Finset.sum_pos (fun i _ => S.weight_pos i) h

/-- Removing row `i` from its own cell leaves a mass smaller by `w i`. -/
theorem cellMass_erase {S : Sample d N} {z : Fin N → Fin K} {i : Fin N} :
    ∑ j ∈ (cell z (z i)).erase i, S.weight j = cellMass S z (z i) - S.weight i := by
  have hi : i ∈ cell z (z i) := by simp
  rw [eq_sub_iff_add_eq, cellMass, ← Finset.sum_erase_add _ _ hi]

/-- The retained information is symmetric: each cell block is. -/
theorem fisher_isSymm (S : Sample d N) (z : Fin N → Fin K) :
    (fisher S z).IsSymm := by
  ext i j
  simp only [Matrix.transpose_apply, fisher, Matrix.sum_apply, cellBlock_apply]
  exact Finset.sum_congr rfl fun c _ => by ring

end

end ScoreQuantFormal
