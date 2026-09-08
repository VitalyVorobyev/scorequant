import ScoreQuantFormal.MergeSpec
import ScoreQuantFormal.Config

/-!
# Merge invariance

Merging duplicate rows into weighted atoms changes none of the quantities the D objective and
the compiled rule are built from. The argument is one fibrewise regrouping: the cells of the
inherited labeling `z ∘ map` are the disjoint unions of the fibres of `map` over the cells of
`z`, so each cell statistic is a sum of atom statistics.

Discharges `MergeSpec.MergeConclusion`. Nothing here is a theorem about ScoreQuant beyond
that; the geometric consequences are in `Closure.lean`.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N N' K : ℕ}

/-- Inside a cell of the inherited labeling, filtering by the atom is the atom's whole fibre:
if `z i = c`, every row merging into `i` already lies in cell `c`. -/
theorem cell_comp_filter (map : Fin N' → Fin N) (z : Fin N → Fin K) {c : Fin K} {i : Fin N}
    (hi : z i = c) :
    (cell (z ∘ map) c).filter (fun j => map j = i) = cell map i := by
  ext j
  simp only [Finset.mem_filter, mem_cell, Function.comp_apply]
  constructor
  · rintro ⟨-, h⟩
    exact h
  · intro h
    exact ⟨by rw [h]; exact hi, h⟩

/-- Regrouping a sum over a cell of the inherited labeling into the fibres of `map` over the
corresponding cell of `z`. -/
theorem sum_merge {M : Type*} [AddCommMonoid M] (map : Fin N' → Fin N) (z : Fin N → Fin K)
    (f : Fin N' → M) (c : Fin K) :
    ∑ i ∈ cell z c, ∑ j ∈ cell map i, f j = ∑ j ∈ cell (z ∘ map) c, f j := by
  classical
  rw [← Finset.sum_fiberwise_of_maps_to
    (g := map) (t := cell z c) (f := f)
    (fun j hj => mem_cell.mpr (by simpa using mem_cell.mp hj))]
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [cell_comp_filter map z (mem_cell.mp hi)]

theorem cellMass_merge {S' : Sample d N'} {S : Sample d N} {map : Fin N' → Fin N}
    (h : IsMerge S' S map) (z : Fin N → Fin K) (c : Fin K) :
    cellMass S' (z ∘ map) c = cellMass S z c := by
  simp only [cellMass]
  rw [← sum_merge map z S'.weight c]
  exact (Finset.sum_congr rfl fun i _ => h.weight_eq i).symm

theorem cellSum_merge {S' : Sample d N'} {S : Sample d N} {map : Fin N' → Fin N}
    (h : IsMerge S' S map) (z : Fin N → Fin K) (c : Fin K) :
    cellSum S' (z ∘ map) c = cellSum S z c := by
  simp only [cellSum]
  rw [← sum_merge map z (fun j => S'.weight j • S'.score j) c]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [h.weight_eq i]
  simp only [cellMass]
  rw [Finset.sum_smul]
  refine Finset.sum_congr rfl fun j hj => ?_
  rw [h.score_eq j, mem_cell.mp hj]

theorem centroid_merge {S' : Sample d N'} {S : Sample d N} {map : Fin N' → Fin N}
    (h : IsMerge S' S map) (z : Fin N → Fin K) (c : Fin K) :
    centroid S' (z ∘ map) c = centroid S z c := by
  rw [centroid, centroid, cellMass_merge h, cellSum_merge h]

theorem fisher_merge {S' : Sample d N'} {S : Sample d N} {map : Fin N' → Fin N}
    (h : IsMerge S' S map) (z : Fin N → Fin K) :
    fisher S' (z ∘ map) = fisher S z := by
  simp only [fisher]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [cellMass_merge h, cellSum_merge h]

/-- **Merge invariance discharges its frozen statement.** -/
theorem merge_invariance (S' : Sample d N') (S : Sample d N) (map : Fin N' → Fin N)
    (z : Fin N → Fin K) : MergeConclusion S' S map z :=
  fun h => ⟨cellMass_merge h z, cellSum_merge h z, centroid_merge h z, fisher_merge h z⟩

end

end ScoreQuantFormal
