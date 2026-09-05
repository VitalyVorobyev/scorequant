import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
# Finite weighted configurations and their retained information

This file sets up the finite objects the D chain talks about: a weighted score
sample, a labeling into `K` cells, the per-cell mass and weighted sum, and the
retained information matrix

```
I(z) = ∑ c, W_c • (μ_c μ_cᵀ)
```

Cells are represented by their *unnormalized* weighted sum `T_c` rather than by
their centroid `μ_c = W_c⁻¹ • T_c`. The two descriptions agree — `cellBlock`
below is `W⁻¹ • (T Tᵀ) = W • (μ μᵀ)` — but the unnormalized one turns the
relocation algebra of `Relocation.lean` into a two-variable identity in
`(W, T)` instead of centroid bookkeeping.

Nothing here is a theorem about ScoreQuant; these are definitions plus the two
rank-one update identities that `Relocation.lean` assembles into D2.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N K : ℕ}

/-- A finite weighted score sample: `N` rows of `d`-dimensional scores with
strictly positive weights. -/
structure Sample (d N : ℕ) where
  /-- The score attached to each row. -/
  score : Fin N → (Fin d → ℝ)
  /-- The weight attached to each row. -/
  weight : Fin N → ℝ
  /-- Weights are strictly positive. -/
  weight_pos : ∀ i, 0 < weight i

/-- The contribution of one cell to the retained information, written through the
cell's unnormalized weighted sum `T` and mass `W`. Equal to `W • (μ μᵀ)` for
`μ = W⁻¹ • T`. -/
def cellBlock (W : ℝ) (T : Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  W⁻¹ • vecMulVec T T

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

/-- The rows carrying label `c`. -/
def cell (z : Fin N → Fin K) (c : Fin K) : Finset (Fin N) :=
  Finset.univ.filter fun i => z i = c

@[simp]
theorem mem_cell {z : Fin N → Fin K} {c : Fin K} {i : Fin N} :
    i ∈ cell z c ↔ z i = c := by
  simp [cell]

/-- Total weight carried by cell `c`. -/
def cellMass (S : Sample d N) (z : Fin N → Fin K) (c : Fin K) : ℝ :=
  ∑ i ∈ cell z c, S.weight i

/-- Weighted sum of the scores in cell `c`. -/
def cellSum (S : Sample d N) (z : Fin N → Fin K) (c : Fin K) : Fin d → ℝ :=
  ∑ i ∈ cell z c, S.weight i • S.score i

/-- The weighted mean score of cell `c`. -/
def centroid (S : Sample d N) (z : Fin N → Fin K) (c : Fin K) : Fin d → ℝ :=
  (cellMass S z c)⁻¹ • cellSum S z c

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

/-- The retained information of a labeling: `∑ c, W_c • (μ_c μ_cᵀ)`. -/
def fisher (S : Sample d N) (z : Fin N → Fin K) : Matrix (Fin d) (Fin d) ℝ :=
  ∑ c, cellBlock (cellMass S z c) (cellSum S z c)

/-- The Mahalanobis form `(s - μ)ᵀ H (s - μ)` used throughout the D chain. -/
def mahalanobis (H : Matrix (Fin d) (Fin d) ℝ) (s μ : Fin d → ℝ) : ℝ :=
  (s - μ) ⬝ᵥ H.mulVec (s - μ)

end

end ScoreQuantFormal
