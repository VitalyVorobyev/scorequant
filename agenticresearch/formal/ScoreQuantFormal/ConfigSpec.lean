import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Algebra.BigOperators.Fin

/-!
# The frozen definitional layer of the finite D chain

Every `*Spec.lean` statement in this project is written in the vocabulary
defined here: the weighted sample, its cells, their masses, sums and centroids,
the retained information, and the Mahalanobis form. This file is therefore part
of the reviewed statement boundary and carries the same rule as the `Spec`
files: **a prover may not change it without a new statement audit.**

Splitting these definitions out of `Config.lean` is what makes the freeze
closed under definitional dependency. A frozen conclusion that mentions
`fisher` means nothing if `fisher` itself lives in a module the prover may
edit — redefining `fisher` as `0` would make several statements trivially true
without touching an audited file.

Definitions only. Every lemma about them, including the two rank-one update
identities that D2 assembles, stays in `Config.lean`.
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

/-- The rows carrying label `c`. -/
def cell (z : Fin N → Fin K) (c : Fin K) : Finset (Fin N) :=
  Finset.univ.filter fun i => z i = c

/-- Total weight carried by cell `c`. -/
def cellMass (S : Sample d N) (z : Fin N → Fin K) (c : Fin K) : ℝ :=
  ∑ i ∈ cell z c, S.weight i

/-- Weighted sum of the scores in cell `c`. -/
def cellSum (S : Sample d N) (z : Fin N → Fin K) (c : Fin K) : Fin d → ℝ :=
  ∑ i ∈ cell z c, S.weight i • S.score i

/-- The weighted mean score of cell `c`. -/
def centroid (S : Sample d N) (z : Fin N → Fin K) (c : Fin K) : Fin d → ℝ :=
  (cellMass S z c)⁻¹ • cellSum S z c

/-- The retained information of a labeling: `∑ c, W_c • (μ_c μ_cᵀ)`. -/
def fisher (S : Sample d N) (z : Fin N → Fin K) : Matrix (Fin d) (Fin d) ℝ :=
  ∑ c, cellBlock (cellMass S z c) (cellSum S z c)

/-- The Mahalanobis form `(s - μ)ᵀ H (s - μ)` used throughout the D chain. -/
def mahalanobis (H : Matrix (Fin d) (Fin d) ℝ) (s μ : Fin d → ℝ) : ℝ :=
  (s - μ) ⬝ᵥ H.mulVec (s - μ)

end

end ScoreQuantFormal
