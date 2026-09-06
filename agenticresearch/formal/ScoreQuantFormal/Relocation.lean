import ScoreQuantFormal.Config
import ScoreQuantFormal.RelocationSpec

/-!
# D2: the exact weighted rank-two relocation identity

Moving one weighted row from a non-singleton source cell to a different
destination cell changes the retained information by an exact rank-two term

```
ΔI = α · u_a u_aᵀ − β · u_b u_bᵀ,   u_c = μ_c − s,
α = w W_a / (W_a − w),   β = w W_b / (W_b + w).
```

`alpha` and `beta` are the coefficients already frozen in
`ScalarExchangeSpec.lean`; `relocate` and the statement itself are frozen in
`RelocationSpec.lean`. This file is where the matrix layer meets them.

Registry claim: `D-RANK2-MOVE`, `KNOWN_RESULTS/04-d-optimality.md` §D2.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N K : ℕ}

@[simp]
theorem relocate_self (z : Fin N → Fin K) (i : Fin N) (b : Fin K) :
    relocate z i b i = b := by
  simp [relocate]

theorem relocate_of_ne (z : Fin N → Fin K) (i : Fin N) (b : Fin K) {j : Fin N}
    (hj : j ≠ i) : relocate z i b j = z j := by
  simp [relocate, Function.update_of_ne hj]

/-- The source cell loses exactly row `i`. -/
theorem cell_relocate_source (z : Fin N → Fin K) (i : Fin N) {b : Fin K}
    (hb : b ≠ z i) : cell (relocate z i b) (z i) = (cell z (z i)).erase i := by
  ext j
  by_cases hj : j = i
  · subst hj
    simp [hb]
  · simp [mem_cell, relocate_of_ne z i b hj, hj]

/-- The destination cell gains exactly row `i`. -/
theorem cell_relocate_dest (z : Fin N → Fin K) (i : Fin N) (b : Fin K) :
    cell (relocate z i b) b = insert i (cell z b) := by
  ext j
  by_cases hj : j = i
  · subst hj; simp
  · simp [mem_cell, relocate_of_ne z i b hj, hj]

/-- Cells other than source and destination are untouched. -/
theorem cell_relocate_other (z : Fin N → Fin K) (i : Fin N) (b : Fin K) {c : Fin K}
    (hca : c ≠ z i) (hcb : c ≠ b) : cell (relocate z i b) c = cell z c := by
  ext j
  by_cases hj : j = i
  · subst hj; simp [hca.symm, hcb.symm]
  · simp [mem_cell, relocate_of_ne z i b hj]

/-- A source cell holding a second row has mass strictly above that row's weight. -/
theorem weight_lt_cellMass (S : Sample d N) (z : Fin N → Fin K) (i : Fin N)
    (h : ∃ j ∈ cell z (z i), j ≠ i) : S.weight i < cellMass S z (z i) := by
  obtain ⟨j, hjmem, hji⟩ := h
  have hi : i ∈ cell z (z i) := by simp
  have hrest : 0 < ∑ k ∈ (cell z (z i)).erase i, S.weight k :=
    Finset.sum_pos (fun k _ => S.weight_pos k) ⟨j, Finset.mem_erase.mpr ⟨hji, hjmem⟩⟩
  have := cellMass_erase (S := S) (z := z) (i := i)
  linarith

/-- Mass and weighted sum of the source cell after the move. -/
theorem source_after (S : Sample d N) (z : Fin N → Fin K) (i : Fin N) {b : Fin K}
    (hb : b ≠ z i) :
    cellMass S (relocate z i b) (z i) = cellMass S z (z i) - S.weight i ∧
      cellSum S (relocate z i b) (z i) = cellSum S z (z i) - S.weight i • S.score i := by
  have hi : i ∈ cell z (z i) := by simp
  constructor
  · rw [cellMass, cell_relocate_source z i hb]
    exact cellMass_erase
  · rw [cellSum, cell_relocate_source z i hb, eq_sub_iff_add_eq, cellSum,
      ← Finset.sum_erase_add _ _ hi]

/-- Mass and weighted sum of the destination cell after the move. -/
theorem dest_after (S : Sample d N) (z : Fin N → Fin K) (i : Fin N) {b : Fin K}
    (hb : b ≠ z i) :
    cellMass S (relocate z i b) b = cellMass S z b + S.weight i ∧
      cellSum S (relocate z i b) b = cellSum S z b + S.weight i • S.score i := by
  have hi : i ∉ cell z b := by simp [mem_cell]; exact fun h => hb h.symm
  constructor
  · rw [cellMass, cell_relocate_dest z i b, Finset.sum_insert hi, cellMass, add_comm]
  · rw [cellSum, cell_relocate_dest z i b, Finset.sum_insert hi, cellSum, add_comm]

/-- **D2, the exact weighted rank-two relocation identity.**

Moving row `i` from its own cell to a different, nonempty cell `b`, with the
source retaining at least one other row, changes the retained information by
exactly `α u_a u_aᵀ − β u_b u_bᵀ`. -/
theorem fisher_relocate_sub (S : Sample d N) (z : Fin N → Fin K) (i : Fin N) {b : Fin K}
    (hb : b ≠ z i)
    (hsource : ∃ j ∈ cell z (z i), j ≠ i)
    (hdest : (cell z b).Nonempty) :
    fisher S (relocate z i b) - fisher S z
      = alpha (S.weight i) (cellMass S z (z i)) •
          vecMulVec (centroid S z (z i) - S.score i) (centroid S z (z i) - S.score i)
        - beta (S.weight i) (cellMass S z b) •
          vecMulVec (centroid S z b - S.score i) (centroid S z b - S.score i) := by
  classical
  set a := z i with ha
  set w := S.weight i with hw
  have hwpos : 0 < w := S.weight_pos i
  have hWa : w < cellMass S z a := weight_lt_cellMass S z i hsource
  have hWb : 0 < cellMass S z b := cellMass_pos hdest
  obtain ⟨hMassA, hSumA⟩ := source_after S z i hb
  obtain ⟨hMassB, hSumB⟩ := dest_after S z i hb
  -- Only the source and destination blocks change.
  have hzero : ∀ c ∈ (Finset.univ : Finset (Fin K)), c ∉ ({a, b} : Finset (Fin K)) →
      cellBlock (cellMass S (relocate z i b) c) (cellSum S (relocate z i b) c)
        - cellBlock (cellMass S z c) (cellSum S z c) = 0 := by
    intro c _ hc
    simp only [Finset.mem_insert, Finset.mem_singleton, not_or] at hc
    have hcell := cell_relocate_other z i b hc.1 hc.2
    rw [cellMass, cellMass, cellSum, cellSum, hcell, sub_self]
  have hsplit :
      fisher S (relocate z i b) - fisher S z
        = (cellBlock (cellMass S (relocate z i b) a) (cellSum S (relocate z i b) a)
            - cellBlock (cellMass S z a) (cellSum S z a))
          + (cellBlock (cellMass S (relocate z i b) b) (cellSum S (relocate z i b) b)
            - cellBlock (cellMass S z b) (cellSum S z b)) := by
    rw [fisher, fisher, ← Finset.sum_sub_distrib,
      ← Finset.sum_subset (Finset.subset_univ ({a, b} : Finset (Fin K))) hzero,
      Finset.sum_pair (Ne.symm hb)]
  rw [hsplit, hMassA, hSumA, hMassB, hSumB,
    cellBlock_erase (cellMass S z a) w hwpos hWa (cellSum S z a) (S.score i),
    cellBlock_insert (cellMass S z b) w hwpos hWb (cellSum S z b) (S.score i)]
  rw [show centroid S z a - S.score i = (cellMass S z a)⁻¹ • cellSum S z a - S.score i from rfl,
    show centroid S z b - S.score i = (cellMass S z b)⁻¹ • cellSum S z b - S.score i from rfl]
  unfold alpha beta
  abel

/-- **D2 discharges its frozen statement.** This is the declaration that
`D-RANK2-MOVE` carries as its `formal_proof`: it names the audited
`RelocationConclusion`, so no hypothesis can be added or conclusion weakened
here without changing an audited file. -/
theorem rank_two_relocation (S : Sample d N) (z : Fin N → Fin K) (i : Fin N)
    (b : Fin K) : RelocationConclusion S z i b := by
  rintro ⟨hb, hsource, hdest⟩
  exact fisher_relocate_sub S z i hb hsource hdest

end

end ScoreQuantFormal
