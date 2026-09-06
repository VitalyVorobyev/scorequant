import ScoreQuantFormal.Corollaries
import Mathlib.Algebra.Order.Star.Real

/-!
# Machine-checked boundary witnesses for D5

Two exact-rational configurations from `COUNTEREXAMPLES/`, verified here rather
than only measured. Both are `d = 1`, where the `I⁻¹` metric is a positive
scalar and Voronoi geometry is Euclidean.

* `CE-D-UNMERGED-DUPLICATES-001` — every hypothesis of `exchange_voronoi`
  except injectivity of the score map holds, and the conclusion fails. So the
  merged-atoms hypothesis is load-bearing, not decorative.
* `CE-D-VORONOI-CONVERSE-001` — a strictly Voronoi labeling that is *not*
  exchange stable. So the implication is strictly one-directional.

These are statements about explicit finite configurations; they carry no
`formal_proof` marker, because a counterexample claim's `statement` is the
proposition being refuted (ADR 0030).
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

/-! ## CE-D-UNMERGED-DUPLICATES-001

Scores `(1, 1, -1)`, weights `(1/4, 1/4, 1/2)`, `K = 3`, each row in its own
cell. All three cells are singletons, so no relocation is admissible and
stability is vacuous; the retained information is `1`, hence positive definite;
but rows `0` and `1` are equidistant from the centroids of cells `0` and `1`.
-/

namespace UnmergedDuplicates

/-- The three split duplicate atoms. -/
def sample : Sample 1 3 where
  score := ![![1], ![1], ![-1]]
  weight := ![1 / 4, 1 / 4, 1 / 2]
  weight_pos := by intro i; fin_cases i <;> norm_num

@[simp] theorem weight_zero : sample.weight 0 = 1 / 4 := rfl
@[simp] theorem weight_one : sample.weight 1 = 1 / 4 := rfl
@[simp] theorem weight_two : sample.weight 2 = 1 / 2 := rfl
@[simp] theorem score_zero : sample.score 0 0 = 1 := rfl
@[simp] theorem score_one : sample.score 1 0 = 1 := rfl
@[simp] theorem score_two : sample.score 2 0 = -1 := rfl

/-- Each row sits in its own cell. -/
def labeling : Fin 3 → Fin 3 := id

theorem cell_eq (c : Fin 3) : cell labeling c = {c} := by
  ext j
  simp [mem_cell, labeling, eq_comm]

theorem cells_nonempty : ∀ c, (cell labeling c).Nonempty := fun c => by
  rw [cell_eq]; exact ⟨c, Finset.mem_singleton_self c⟩

/-- A singleton cell has its own row as centroid. -/
theorem centroid_eq (c : Fin 3) : centroid sample labeling c = sample.score c := by
  have hc : labeling c = c := rfl
  have := centroid_of_singleton sample labeling c (by rw [hc]; exact cell_eq c)
  rwa [hc] at this

/-- The retained information is the identity. -/
theorem fisher_eq : fisher sample labeling = 1 := by
  ext i j
  fin_cases i
  fin_cases j
  rw [fisher, Matrix.sum_apply, Fin.sum_univ_three]
  simp only [cellMass, cellSum, cell_eq, Finset.sum_singleton, cellBlock_apply,
    Pi.smul_apply, smul_eq_mul, Matrix.one_apply_eq]
  norm_num

theorem posDef : (fisher sample labeling).PosDef := by
  rw [fisher_eq]; exact Matrix.PosDef.one

/-- Stability is vacuous: every source cell is a singleton. -/
theorem exchangeStable : ExchangeStable sample labeling := by
  rintro i b ⟨-, j, hj, hji⟩
  rw [cell_eq] at hj
  exact absurd (Finset.mem_singleton.mp hj) hji

/-- **The witness.** Every hypothesis but injectivity holds, and strict Voronoi
geometry fails: row `0` is exactly as close to centroid `1` as to its own. -/
theorem not_strictVoronoi : ¬ StrictVoronoi sample labeling := by
  intro h
  have := h 0 1 (by decide)
  rw [centroid_eq, centroid_eq] at this
  simp only [mahalanobis, dotProduct, Fin.sum_univ_one, Pi.sub_apply,
    Matrix.mulVec, labeling, id_eq] at this
  norm_num at this

/-- Injectivity of the score map is exactly what fails here. -/
theorem score_not_injective : ¬ Function.Injective sample.score := by
  intro h
  exact absurd (h (show sample.score 0 = sample.score 1 from rfl)) (by decide)

/-- Every hypothesis of `exchange_voronoi` except injectivity is met here. -/
theorem all_but_injectivity :
    (∀ c, (cell labeling c).Nonempty) ∧ (fisher sample labeling).PosDef ∧
      ExchangeStable sample labeling :=
  ⟨cells_nonempty, posDef, exchangeStable⟩

end UnmergedDuplicates

/-! ## CE-D-VORONOI-CONVERSE-001

Scores `(-3/4, -3/4, 1/4, 5/4)`, uniform weights `1/4`, `K = 2`, labels
`(0, 0, 0, 1)`. Every row is strictly nearest its own centroid, yet moving row
`2` to cell `1` raises the retained information from `25/48` to `9/16`.
-/

namespace VoronoiConverse

/-- The four rows of the converse witness. -/
def sample : Sample 1 4 where
  score := ![![-3 / 4], ![-3 / 4], ![1 / 4], ![5 / 4]]
  weight := ![1 / 4, 1 / 4, 1 / 4, 1 / 4]
  weight_pos := by intro i; fin_cases i <;> norm_num

@[simp] theorem weight_eq (i : Fin 4) : sample.weight i = 1 / 4 := by
  fin_cases i <;> rfl
@[simp] theorem score_zero : sample.score 0 0 = -3 / 4 := rfl
@[simp] theorem score_one : sample.score 1 0 = -3 / 4 := rfl
@[simp] theorem score_two : sample.score 2 0 = 1 / 4 := rfl
@[simp] theorem score_three : sample.score 3 0 = 5 / 4 := rfl

def labeling : Fin 4 → Fin 2 := ![0, 0, 0, 1]

@[simp] theorem labeling_zero : labeling 0 = 0 := rfl
@[simp] theorem labeling_one : labeling 1 = 0 := rfl
@[simp] theorem labeling_two : labeling 2 = 0 := rfl
@[simp] theorem labeling_three : labeling 3 = 1 := rfl

theorem cell_zero : cell labeling 0 = {0, 1, 2} := by decide

theorem cell_one : cell labeling 1 = {3} := by decide

/-- The move that refutes stability: row `2` joins cell `1`. -/
theorem admissible : Admissible labeling 2 1 :=
  ⟨by decide, ⟨0, by decide, by decide⟩⟩

theorem cell_moved_zero : cell (relocate labeling 2 1) 0 = {0, 1} := by decide

theorem cell_moved_one : cell (relocate labeling 2 1) 1 = {2, 3} := by decide

theorem det_before : (fisher sample labeling).det = 25 / 48 := by
  rw [Matrix.det_unique, fisher, Matrix.sum_apply, Fin.sum_univ_two]
  simp only [cellMass, cellSum, cell_zero, cell_one, cellBlock_apply]
  rw [Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_singleton,
    Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_singleton,
    Finset.sum_singleton, Finset.sum_singleton]
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, weight_eq]
  norm_num

theorem det_after : (fisher sample (relocate labeling 2 1)).det = 9 / 16 := by
  rw [Matrix.det_unique, fisher, Matrix.sum_apply, Fin.sum_univ_two]
  simp only [cellMass, cellSum, cell_moved_zero, cell_moved_one, cellBlock_apply]
  rw [Finset.sum_insert (by decide), Finset.sum_singleton,
    Finset.sum_insert (by decide), Finset.sum_singleton,
    Finset.sum_insert (by decide), Finset.sum_singleton,
    Finset.sum_insert (by decide), Finset.sum_singleton]
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, weight_eq]
  norm_num

/-- The retained information is `25/48` times the identity. -/
theorem fisher_eq : fisher sample labeling = (25 / 48 : ℝ) • 1 := by
  ext i j
  fin_cases i
  fin_cases j
  rw [fisher, Matrix.sum_apply, Fin.sum_univ_two]
  simp only [cellMass, cellSum, cell_zero, cell_one, cellBlock_apply]
  rw [Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_singleton,
    Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_singleton,
    Finset.sum_singleton, Finset.sum_singleton]
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, weight_eq,
    Matrix.smul_apply, Matrix.one_apply_eq]
  norm_num

theorem posDef : (fisher sample labeling).PosDef := by
  rw [fisher_eq]
  exact Matrix.PosDef.one.smul (by norm_num)

theorem inv_eq : (fisher sample labeling)⁻¹ = (48 / 25 : ℝ) • 1 := by
  refine Matrix.inv_eq_right_inv ?_
  rw [fisher_eq, smul_mul_smul_comm, Matrix.one_mul]
  norm_num

/-- In one dimension the metric is a positive scalar, so Voronoi geometry is
Euclidean. -/
theorem mahalanobis_eq (s μ : Fin 1 → ℝ) :
    mahalanobis (fisher sample labeling)⁻¹ s μ = 48 / 25 * (s 0 - μ 0) ^ 2 := by
  rw [mahalanobis, inv_eq]
  simp only [Matrix.mulVec, dotProduct, Fin.sum_univ_one, Matrix.smul_apply,
    Matrix.one_apply_eq, smul_eq_mul, Pi.sub_apply]
  ring

theorem centroid_zero : centroid sample labeling 0 0 = -5 / 12 := by
  rw [centroid, cellMass, cellSum, cell_zero]
  rw [Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_singleton,
    Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_singleton]
  simp only [Pi.smul_apply, Pi.add_apply, smul_eq_mul, weight_eq, score_zero, score_one,
    score_two]
  norm_num

theorem centroid_one : centroid sample labeling 1 0 = 5 / 4 := by
  rw [centroid, cellMass, cellSum, cell_one, Finset.sum_singleton, Finset.sum_singleton]
  simp only [Pi.smul_apply, smul_eq_mul, weight_eq, score_three]
  norm_num

/-- **The first half of the witness.** Every row is strictly nearest its own
centroid, so this labeling *is* a strict Voronoi fixed point. -/
theorem strictVoronoi : StrictVoronoi sample labeling := by
  intro i c hc
  rw [mahalanobis_eq, mahalanobis_eq]
  fin_cases i <;> fin_cases c <;>
    simp_all [centroid_zero, centroid_one] <;> norm_num

/-- **The second half.** A single admissible relocation strictly increases the
retained information, so the same labeling is *not* exchange stable. Hence the
converse of `exchange_voronoi` fails. -/
theorem not_exchangeStable : ¬ ExchangeStable sample labeling := by
  intro h
  have := h 2 1 admissible
  rw [det_before, det_after] at this
  norm_num at this

end VoronoiConverse

end

end ScoreQuantFormal
