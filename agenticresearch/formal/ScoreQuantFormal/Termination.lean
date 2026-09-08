import ScoreQuantFormal.TerminationSpec
import Mathlib.Data.Fintype.EquivFin

/-!
# Finite termination of exact positive-gain exchange

One argument, three claims. The objective is an arbitrary `F : (Fin N → Fin K) → ℝ`; nothing
below inspects it.

Strict ascent and the absence of an infinite run are the easy half: a strictly increasing
sequence of reals indexed by `ℕ` forces the labelings to be pairwise distinct, and there are
finitely many. Termination *at a stable state* is the half `Corollaries.lean` does not prove,
and it needs a measure: the number of labelings strictly better than the current one. An
accepted move strictly decreases it, because the new state is better than the old one and no
longer counts itself.

Registry claim: `D-EXCHANGE-TERMINATES`. The generic theorems also cover §DS3's and §A1's
termination sentences, but neither objective is defined in this tree, so that coverage is
recorded in `KNOWN_RESULTS/` prose rather than as a registry marker.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N K : ℕ}

/-- Prepending a move to a run. `Reaches` appends at the tail, so the induction runs the other
way and this is the lemma that composes a step with a run after it. -/
theorem Reaches.head {F : (Fin N → Fin K) → ℝ} {z z' w : Fin N → Fin K}
    (h : ImprovingMove F z z') (hr : Reaches F z' w) : Reaches F z w := by
  induction hr with
  | refl => exact Reaches.tail (Reaches.refl z) h
  | tail _ hstep ih => exact Reaches.tail ih hstep

/-- **Strict ascent.** A run either ends where it started or strictly improved. Absence of
cycles is `no_cycle` below; it does not follow from this statement alone, because a cycle
satisfies the `z = w` disjunct. -/
theorem ascent_strict (F : (Fin N → Fin K) → ℝ) : AscentStrict F := by
  intro z w hr
  induction hr with
  | refl => exact Or.inl rfl
  | tail _ hstep ih =>
    rcases ih with h | h
    · exact Or.inr (h ▸ hstep.2)
    · exact Or.inr (lt_trans h hstep.2)

/-- **No infinite run of accepted moves.** -/
theorem no_infinite_run (F : (Fin N → Fin K) → ℝ) : NoInfiniteRun F := by
  rintro ⟨seq, hseq⟩
  have hmono : StrictMono fun n => F (seq n) :=
    strictMono_nat_of_lt_succ fun n => (hseq n).2
  have hinj : Function.Injective seq := by
    intro m n hmn
    by_contra hne
    rcases lt_or_gt_of_ne hne with h | h
    · exact absurd (congrArg F hmn) (ne_of_lt (hmono h))
    · exact absurd (congrArg F hmn.symm) (ne_of_lt (hmono h))
  exact not_injective_infinite_finite seq hinj

/-- The measure: how many labelings are strictly better than `z`. -/
private def betterCount (F : (Fin N → Fin K) → ℝ) (z : Fin N → Fin K) : ℕ :=
  open Classical in (Finset.univ.filter fun v => F z < F v).card

private theorem terminates_aux (F : (Fin N → Fin K) → ℝ) :
    ∀ n z, betterCount F z ≤ n → ∃ w, Reaches F z w ∧ StableFor F w := by
  classical
  intro n
  induction n with
  | zero =>
    intro z hz
    refine ⟨z, Reaches.refl z, fun i b _ => ?_⟩
    by_contra hcon
    push Not at hcon
    have hmem : relocate z i b ∈ Finset.univ.filter fun v => F z < F v := by
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      exact hcon
    rw [Finset.card_eq_zero.mp (Nat.le_zero.mp hz)] at hmem
    exact absurd hmem (Finset.notMem_empty _)
  | succ n ih =>
    intro z hz
    by_cases hstable : StableFor F z
    · exact ⟨z, Reaches.refl z, hstable⟩
    · rw [StableFor] at hstable
      push Not at hstable
      obtain ⟨i, b, hab, hlt⟩ := hstable
      have hstep : ImprovingMove F z (relocate z i b) := ⟨⟨i, b, hab, rfl⟩, hlt⟩
      have hsub : (Finset.univ.filter fun v => F (relocate z i b) < F v)
          ⊆ Finset.univ.filter fun v => F z < F v := by
        intro v hv
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hv ⊢
        exact lt_trans hlt hv
      have hmem : relocate z i b ∈ Finset.univ.filter fun v => F z < F v := by
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        exact hlt
      have hnot : relocate z i b ∉ Finset.univ.filter fun v => F (relocate z i b) < F v := by
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, lt_self_iff_false,
          not_false_eq_true]
      have hdrop : betterCount F (relocate z i b) < betterCount F z :=
        Finset.card_lt_card ((Finset.ssubset_iff_of_subset hsub).mpr ⟨_, hmem, hnot⟩)
      obtain ⟨w, hreach, hw⟩ := ih (relocate z i b) (by omega)
      exact ⟨w, Reaches.head hstep hreach, hw⟩

/-- **Termination at a stable state.** -/
theorem terminates (F : (Fin N → Fin K) → ℝ) : TerminationConclusion F :=
  fun z => terminates_aux F (betterCount F z) z le_rfl

/-- Stability for the determinant objective is exactly the frozen `ExchangeStable`. -/
theorem stableFor_det_iff (S : Sample d N) (z : Fin N → Fin K) :
    StableFor (fun z : Fin N → Fin K => (fisher S z).det) z ↔ ExchangeStable S z :=
  Iff.rfl

/-- **No cycles.** Once a move is accepted the run cannot return: coming back would need the
objective to be both strictly larger and no larger. -/
theorem no_cycle (F : (Fin N → Fin K) → ℝ) : NoCycle F := by
  intro z w hmove hback
  rcases ascent_strict F w z hback with h | h
  · exact absurd (h ▸ hmove.2) (lt_irrefl _)
  · exact absurd (lt_trans hmove.2 h) (lt_irrefl _)

/-- **`D-EXCHANGE-TERMINATES` discharges its frozen statement.** -/
theorem d_exchange_terminates (S : Sample d N) (K : ℕ) : DTerminationConclusion S K :=
  ⟨ascent_strict _, no_cycle _, no_infinite_run _, terminates _⟩

/-- **The `F_D = log det` run terminates too.** A different run from the determinant one, and
neither is a corollary of the other; the shared generic argument covers both. -/
theorem d_log_exchange_terminates (S : Sample d N) (K : ℕ) : DLogTerminationConclusion S K :=
  ⟨ascent_strict _, no_cycle _, no_infinite_run _, terminates _⟩

end

end ScoreQuantFormal
