import ScoreQuantFormal.ClosureSpec
import ScoreQuantFormal.ExchangeVoronoi
import ScoreQuantFormal.Merge

/-!
# D6: the compiled predictor

`D-FINITE-INDUCTIVE-CLOSURE`, the deployable half of manuscript v9 Theorem 3.

Part (a) is a short consequence of D5: `StrictVoronoi` already says every training row is
*strictly* nearest its own centroid, so no rule that returns a nearest centroid has any freedom
at a training score, whatever it does with ties elsewhere. The strictness is what makes the
quantification over all such rules possible, and it is exactly the claim's "without a tie
breaker".

Part (b) is not a restatement. The rule there is built from the unmerged sample, whose scores
are not injective and to which D5 therefore does not apply; reaching it needs `Merge.lean`.

Registry claim: `D-FINITE-INDUCTIVE-CLOSURE`.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N N' K : ℕ}

/-- **D6, part (a).** At an exactly stable state a nearest-centroid rule has no choice at a
training score: it must return that row's label. -/
theorem nearestCentroidRule_eq_label (S : Sample d N) (z : Fin N → Fin K)
    (hmerged : Function.Injective S.score) (hne : ∀ c, (cell z c).Nonempty)
    (hpd : (fisher S z).PosDef) (hstable : ExchangeStable S z)
    (q : (Fin d → ℝ) → Fin K) (hq : IsNearestCentroidRule S z q) (i : Fin N) :
    q (S.score i) = z i := by
  by_contra hqi
  have hsv := exchangeStable_implies_strictVoronoi S z hmerged hne hpd hstable
  exact absurd (hsv i (q (S.score i)) hqi) (not_lt.mpr (hq (S.score i) (z i)))

/-- Positive-definiteness rules out an empty label set: with no cells the retained information
is the zero matrix, whose determinant vanishes in positive dimension. -/
theorem nonempty_labels_of_posDef {S : Sample d N} {z : Fin N → Fin K}
    (hpd : (fisher S z).PosDef) (hd : 0 < d) : Nonempty (Fin K) := by
  by_contra hempty
  rw [not_nonempty_iff] at hempty
  have hzero : fisher S z = 0 := by
    simp [fisher, Finset.univ_eq_empty]
  have hdne : Nonempty (Fin d) := ⟨⟨0, hd⟩⟩
  have hdet := hpd.det_pos
  rw [hzero, Matrix.det_zero] at hdet
  exact lt_irrefl _ hdet

/-- A nearest-centroid rule exists whenever there is at least one label. -/
theorem exists_nearestCentroidRule (S : Sample d N) (z : Fin N → Fin K)
    (hK : Nonempty (Fin K)) : ∃ q, IsNearestCentroidRule S z q := by
  classical
  have hmin : ∀ s : Fin d → ℝ, ∃ b : Fin K, ∀ c : Fin K,
      mahalanobis (fisher S z)⁻¹ s (centroid S z b)
        ≤ mahalanobis (fisher S z)⁻¹ s (centroid S z c) := by
    intro s
    obtain ⟨b, -, hb⟩ := Finset.exists_min_image (Finset.univ : Finset (Fin K))
      (fun c => mahalanobis (fisher S z)⁻¹ s (centroid S z c))
      ⟨Classical.choice hK, Finset.mem_univ _⟩
    exact ⟨b, fun c => hb c (Finset.mem_univ c)⟩
  exact ⟨fun s => (hmin s).choose, fun s c => (hmin s).choose_spec c⟩

/-- **D6 discharges its frozen statement.** -/
theorem closure_reproduces_labels (S : Sample d N) (z : Fin N → Fin K) :
    ClosureConclusion S z := by
  rintro ⟨hmerged, hne, hpd⟩ hstable
  exact ⟨fun hd => exists_nearestCentroidRule S z (nonempty_labels_of_posDef hpd hd),
    fun q hq i => nearestCentroidRule_eq_label S z hmerged hne hpd hstable q hq i⟩

/-- **D6, part (b), discharges its frozen statement.** Merge invariance turns a rule built
from the unmerged sample into one built from the atoms, where D5 applies. -/
theorem closure_duplicates (S' : Sample d N') (S : Sample d N) (map : Fin N' → Fin N)
    (z : Fin N → Fin K) : ClosureDuplicateConclusion S' S map z := by
  rintro hm ⟨hmerged, hne, hpd⟩ hstable q hq j
  have hq' : IsNearestCentroidRule S z q := by
    intro s c
    have h := hq s c
    rwa [fisher_merge hm z, centroid_merge hm z, centroid_merge hm z] at h
  rw [hm.score_eq j]
  exact nearestCentroidRule_eq_label S z hmerged hne hpd hstable q hq' (map j)

end

end ScoreQuantFormal
