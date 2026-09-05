import ScoreQuantFormal.ExchangeVoronoiSpec
import ScoreQuantFormal.ScalarExchange
import Mathlib.Analysis.Matrix.PosDef

/-!
# D5: exchange stability implies strict D-Voronoi geometry

Proof module for the specification frozen in `ExchangeVoronoiSpec.lean`. The
argument follows `KNOWN_RESULTS/04-d-optimality.md` §D5 clause by clause:

1. the determinant of a relocation is `det I · detRatio` (D2 + D3);
2. the scalar hypotheses of `ScalarExchangeSpec.lean` are discharged by
   positive-definiteness and the leverage bound D4;
3. distinct centroids follow from stability rather than being assumed;
4. a tied-or-worse comparison from a non-singleton source has strictly positive
   gain, and a singleton source is strictly nearest its own centroid for free.

Registry claims: `D-EXCHANGE-IMPLIES-VORONOI`, `D-EXCHANGE-VIOLATION-LOWER-BOUND`.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N K : ℕ}

/-! ## Coefficient positivity -/

theorem alpha_pos {w W : ℝ} (hw : 0 < w) (hW : w < W) : 0 < alpha w W :=
  div_pos (mul_pos hw (lt_trans hw hW)) (sub_pos.mpr hW)

theorem beta_pos {w W : ℝ} (hw : 0 < w) (hW : 0 < W) : 0 < beta w W :=
  div_pos (mul_pos hw hW) (add_pos hW hw)

/-- The removal coefficient always exceeds the insertion coefficient: `β < w < α`. -/
theorem beta_lt_alpha {w Wa Wb : ℝ} (hw : 0 < w) (hWa : w < Wa) (hWb : 0 < Wb) :
    beta w Wb < alpha w Wa := by
  have h1 : beta w Wb < w := by
    rw [beta, div_lt_iff₀ (by linarith)]
    nlinarith
  have h2 : w < alpha w Wa := by
    rw [alpha, lt_div_iff₀ (by linarith)]
    nlinarith
  linarith

/-! ## Geometry helpers -/

theorem mahalanobis_eq_qform (H : Matrix (Fin d) (Fin d) ℝ) (s μ : Fin d → ℝ) :
    mahalanobis H s μ = qform H (μ - s) (μ - s) := by
  unfold mahalanobis qform
  rw [show μ - s = -(s - μ) by abel, Matrix.mulVec_neg, neg_dotProduct, dotProduct_neg,
    neg_neg]

/-- A positive-definite real matrix has strictly positive determinant. -/
theorem posDef_det_pos {I : Matrix (Fin d) (Fin d) ℝ} (hpd : I.PosDef) : 0 < I.det :=
  lt_of_le_of_ne hpd.posSemidef.det_nonneg
    (Ne.symm (IsUnit.ne_zero ((Matrix.isUnit_iff_isUnit_det I).mp hpd.isUnit)))

/-- Positive definiteness transfers to the quadratic form of the inverse. -/
theorem qform_inv_pos {I : Matrix (Fin d) (Fin d) ℝ} (hpd : I.PosDef) {u : Fin d → ℝ}
    (hu : u ≠ 0) : 0 < qform I⁻¹ u u := by
  have hinv : (I⁻¹).PosDef := Matrix.posDef_inv_iff.mpr hpd
  simpa [qform] using hinv.dotProduct_mulVec_pos hu

theorem qform_inv_nonneg {I : Matrix (Fin d) (Fin d) ℝ} (hpd : I.PosDef) (u : Fin d → ℝ) :
    0 ≤ qform I⁻¹ u u := by
  rcases eq_or_ne u 0 with rfl | hu
  · simp [qform]
  · exact (qform_inv_pos hpd hu).le

/-- A singleton cell has its own row as centroid. -/
theorem centroid_of_singleton (S : Sample d N) (z : Fin N → Fin K) (i : Fin N)
    (hsingle : cell z (z i) = {i}) : centroid S z (z i) = S.score i := by
  have hw : S.weight i ≠ 0 := ne_of_gt (S.weight_pos i)
  rw [centroid, cellMass, cellSum, hsingle, Finset.sum_singleton, Finset.sum_singleton,
    smul_smul, inv_mul_cancel₀ hw, one_smul]

/-- A cell holding two rows holds a row away from its centroid, because merged
atoms have distinct scores. -/
theorem exists_score_ne_centroid (S : Sample d N) (z : Fin N → Fin K)
    (hmerged : Function.Injective S.score) {c : Fin K}
    (hcard : 1 < (cell z c).card) :
    ∃ i ∈ cell z c, S.score i ≠ centroid S z c := by
  obtain ⟨i, hi, j, hj, hij⟩ := Finset.one_lt_card.mp hcard
  by_cases h : S.score i = centroid S z c
  · refine ⟨j, hj, ?_⟩
    intro hj'
    exact hij (hmerged (h.trans hj'.symm))
  · exact ⟨i, hi, h⟩

/-! ## The determinant of a relocation -/

/-- D2 and D3 combined: the exact determinant after an admissible relocation. -/
theorem det_relocate (S : Sample d N) (z : Fin N → Fin K)
    (hne : ∀ c, (cell z c).Nonempty) (hpd : (fisher S z).PosDef)
    (i : Fin N) {b : Fin K} (hadm : Admissible z i b) :
    (fisher S (relocate z i b)).det
      = (fisher S z).det *
          detRatio (alpha (S.weight i) (cellMass S z (z i)))
            (beta (S.weight i) (cellMass S z b))
            (qform (fisher S z)⁻¹ (centroid S z (z i) - S.score i)
              (centroid S z (z i) - S.score i))
            (qform (fisher S z)⁻¹ (centroid S z b - S.score i)
              (centroid S z b - S.score i))
            (qform (fisher S z)⁻¹ (centroid S z (z i) - S.score i)
              (centroid S z b - S.score i)) := by
  obtain ⟨hb, hsource⟩ := hadm
  have hunit : IsUnit (fisher S z).det :=
    (Matrix.isUnit_iff_isUnit_det _).mp hpd.isUnit
  have hdiff := fisher_relocate_sub S z i hb hsource (hne b)
  have hsum : fisher S (relocate z i b)
      = fisher S z +
        (alpha (S.weight i) (cellMass S z (z i)) •
            vecMulVec (centroid S z (z i) - S.score i) (centroid S z (z i) - S.score i)
          - beta (S.weight i) (cellMass S z b) •
            vecMulVec (centroid S z b - S.score i) (centroid S z b - S.score i)) := by
    rw [← hdiff]; abel
  rw [hsum, det_add_rank_two (fisher S z) (fisher_isSymm S z) hunit]

/-! ## The scalar hypotheses -/

/-- The frozen scalar assumptions hold at any admissible tied-or-worse move. -/
theorem scalar_assumptions (S : Sample d N) (z : Fin N → Fin K)
    (hne : ∀ c, (cell z c).Nonempty) (hpd : (fisher S z).PosDef)
    (i : Fin N) {b : Fin K} (hadm : Admissible z i b)
    (hviol : qform (fisher S z)⁻¹ (centroid S z b - S.score i) (centroid S z b - S.score i)
      ≤ qform (fisher S z)⁻¹ (centroid S z (z i) - S.score i)
        (centroid S z (z i) - S.score i)) :
    ScalarExchangeAssumptions (S.weight i) (cellMass S z (z i)) (cellMass S z b)
      (qform (fisher S z)⁻¹ (centroid S z (z i) - S.score i)
        (centroid S z (z i) - S.score i))
      (qform (fisher S z)⁻¹ (centroid S z b - S.score i) (centroid S z b - S.score i))
      (qform (fisher S z)⁻¹ (centroid S z (z i) - S.score i)
        (centroid S z b - S.score i)) := by
  obtain ⟨hb, hsource⟩ := hadm
  have hHsymm : ((fisher S z)⁻¹)ᵀ = (fisher S z)⁻¹ := by
    rw [Matrix.transpose_nonsing_inv, fisher_isSymm S z]
  -- `q_δ` is the quadratic form of the centroid difference.
  have hdelta : qDelta
      (qform (fisher S z)⁻¹ (centroid S z (z i) - S.score i)
        (centroid S z (z i) - S.score i))
      (qform (fisher S z)⁻¹ (centroid S z b - S.score i) (centroid S z b - S.score i))
      (qform (fisher S z)⁻¹ (centroid S z (z i) - S.score i)
        (centroid S z b - S.score i))
      = qform (fisher S z)⁻¹ (centroid S z (z i) - centroid S z b)
          (centroid S z (z i) - centroid S z b) := by
    have hsplit : centroid S z (z i) - centroid S z b
        = (centroid S z (z i) - S.score i) - (centroid S z b - S.score i) := by abel
    conv_rhs => rw [hsplit, qform_sub_self hHsymm]
    unfold qDelta
    ring
  refine ⟨S.weight_pos i, weight_lt_cellMass S z i hsource, cellMass_pos (hne b),
    qform_inv_nonneg hpd _, qform_inv_nonneg hpd _, hviol, ?_, ?_⟩
  · rw [hdelta]; exact qform_inv_nonneg hpd _
  · rw [hdelta, one_div, one_div]
    exact leverage_bound S z hne hpd (z i) b (Ne.symm hb)

/-! ## The quantitative gain, and strict Voronoi geometry -/

/-- **`D-EXCHANGE-VIOLATION-LOWER-BOUND`.** A tied-or-worse nearest-centroid
comparison from a non-singleton source has determinant ratio at least
`1 + (α β / 4) q_δ²`. -/
theorem violation_lower_bound (S : Sample d N) (z : Fin N → Fin K)
    (hne : ∀ c, (cell z c).Nonempty) (hpd : (fisher S z).PosDef) :
    ViolationLowerBound S z := by
  intro i b hadm hviol
  rw [mahalanobis_eq_qform, mahalanobis_eq_qform] at hviol
  have hassum := scalar_assumptions S z hne hpd i hadm hviol
  have hbound := scalarExchangeLowerBound hassum
  have hdetpos : 0 < (fisher S z).det := posDef_det_pos hpd
  have hHsymm : ((fisher S z)⁻¹)ᵀ = (fisher S z)⁻¹ := by
    rw [Matrix.transpose_nonsing_inv, fisher_isSymm S z]
  have hdelta : qDelta
      (qform (fisher S z)⁻¹ (centroid S z (z i) - S.score i)
        (centroid S z (z i) - S.score i))
      (qform (fisher S z)⁻¹ (centroid S z b - S.score i) (centroid S z b - S.score i))
      (qform (fisher S z)⁻¹ (centroid S z (z i) - S.score i)
        (centroid S z b - S.score i))
      = centroidSeparation S z (z i) b := by
    have hsplit : centroid S z (z i) - centroid S z b
        = (centroid S z (z i) - S.score i) - (centroid S z b - S.score i) := by abel
    rw [centroidSeparation]
    conv_rhs => rw [hsplit, qform_sub_self hHsymm]
    unfold qDelta
    ring
  rw [det_relocate S z hne hpd i hadm, detRatio_eq_one_add_exchangeExcess]
  rw [hdelta] at hbound
  have := mul_le_mul_of_nonneg_left (add_le_add_left hbound 1) hdetpos.le
  linarith [this]

/-- Stability forces distinct centroids: no separate hypothesis is needed. -/
theorem centroid_ne_of_stable (S : Sample d N) (z : Fin N → Fin K)
    (hmerged : Function.Injective S.score) (hne : ∀ c, (cell z c).Nonempty)
    (hpd : (fisher S z).PosDef) (hstable : ExchangeStable S z) {a b : Fin K}
    (hab : a ≠ b) : centroid S z a ≠ centroid S z b := by
  intro hcent
  have hdetpos : 0 < (fisher S z).det := posDef_det_pos hpd
  -- Moving a non-centroid row between two cells with equal centroids gains
  -- `(α − β) q_aa > 0`, so at least one of the two cells must be a singleton.
  have key : ∀ p q : Fin K, p ≠ q → centroid S z p = centroid S z q →
      1 < (cell z p).card → False := by
    intro p q hpq hcpq hcard
    obtain ⟨i, hi, hscore⟩ := exists_score_ne_centroid S z hmerged hcard
    have hzi : z i = p := mem_cell.mp hi
    subst hzi
    have hadm : Admissible z i q := by
      refine ⟨Ne.symm hpq, ?_⟩
      obtain ⟨x, hx, y, hy, hxy⟩ := Finset.one_lt_card.mp hcard
      by_cases hxi : x = i
      · exact ⟨y, hy, fun h => hxy (hxi.trans h.symm)⟩
      · exact ⟨x, hx, hxi⟩
    -- The two directions coincide, so the excess collapses to `(α − β) q_aa`.
    have hdir : centroid S z q - S.score i = centroid S z (z i) - S.score i := by rw [hcpq]
    have hq : (0 : ℝ) < qform (fisher S z)⁻¹ (centroid S z (z i) - S.score i)
        (centroid S z (z i) - S.score i) :=
      qform_inv_pos hpd fun h => hscore (sub_eq_zero.mp h).symm
    have hw := S.weight_pos i
    have hWa : S.weight i < cellMass S z (z i) := weight_lt_cellMass S z i hadm.2
    have hWb : 0 < cellMass S z q := cellMass_pos (hne q)
    have hlt := beta_lt_alpha hw hWa hWb
    have hdet := det_relocate S z hne hpd i hadm
    rw [hdir] at hdet
    have hratio : 1 < detRatio (alpha (S.weight i) (cellMass S z (z i)))
        (beta (S.weight i) (cellMass S z q))
        (qform (fisher S z)⁻¹ (centroid S z (z i) - S.score i)
          (centroid S z (z i) - S.score i))
        (qform (fisher S z)⁻¹ (centroid S z (z i) - S.score i)
          (centroid S z (z i) - S.score i))
        (qform (fisher S z)⁻¹ (centroid S z (z i) - S.score i)
          (centroid S z (z i) - S.score i)) := by
      unfold detRatio
      nlinarith [mul_pos (sub_pos.mpr hlt) hq]
    have hstab := hstable i q hadm
    rw [hdet] at hstab
    have hgain := mul_lt_mul_of_pos_left hratio hdetpos
    rw [mul_one] at hgain
    linarith
  -- Both cells must therefore be singletons, and merged atoms make that impossible.
  have hcarda : ¬ 1 < (cell z a).card := fun h => key a b hab hcent h
  have hcardb : ¬ 1 < (cell z b).card := fun h => key b a (Ne.symm hab) hcent.symm h
  obtain ⟨ia, hia⟩ := hne a
  obtain ⟨ib, hib⟩ := hne b
  have hsa : cell z a = {ia} := Finset.eq_singleton_iff_unique_mem.mpr
    ⟨hia, fun x hx => Finset.card_le_one.mp (not_lt.mp hcarda) x hx ia hia⟩
  have hsb : cell z b = {ib} := Finset.eq_singleton_iff_unique_mem.mpr
    ⟨hib, fun x hx => Finset.card_le_one.mp (not_lt.mp hcardb) x hx ib hib⟩
  have hza : z ia = a := mem_cell.mp hia
  have hzb : z ib = b := mem_cell.mp hib
  have hca : centroid S z a = S.score ia := by
    have := centroid_of_singleton S z ia (by rw [hza]; exact hsa)
    rwa [hza] at this
  have hcb : centroid S z b = S.score ib := by
    have := centroid_of_singleton S z ib (by rw [hzb]; exact hsb)
    rwa [hzb] at this
  rw [hca, hcb] at hcent
  exact hab (by rw [← hza, ← hzb, hmerged hcent])

/-- **`D-EXCHANGE-IMPLIES-VORONOI` (manuscript v9 Theorem 2).**

On merged distinct positive-weight score atoms with exactly `K` nonempty cells
and positive-definite retained information, zero-tolerance one-point exchange
stability implies strict self-consistent `I⁻¹`-Voronoi geometry. -/
theorem exchangeStable_implies_strictVoronoi (S : Sample d N) (z : Fin N → Fin K)
    (hmerged : Function.Injective S.score) (hne : ∀ c, (cell z c).Nonempty)
    (hpd : (fisher S z).PosDef) (hstable : ExchangeStable S z) :
    StrictVoronoi S z := by
  intro i c hc
  rw [mahalanobis_eq_qform, mahalanobis_eq_qform]
  have hdetpos : 0 < (fisher S z).det := posDef_det_pos hpd
  have hcent : centroid S z (z i) ≠ centroid S z c :=
    centroid_ne_of_stable S z hmerged hne hpd hstable (Ne.symm hc)
  by_cases hcard : 1 < (cell z (z i)).card
  · -- Non-singleton source: a violation would be an improving admissible move.
    have hadm : Admissible z i c := by
      refine ⟨hc, ?_⟩
      obtain ⟨x, hx, y, hy, hxy⟩ := Finset.one_lt_card.mp hcard
      by_cases hxi : x = i
      · exact ⟨y, hy, fun h => hxy (hxi.trans h.symm)⟩
      · exact ⟨x, hx, hxi⟩
    by_contra hle
    push Not at hle
    have hbound := violation_lower_bound S z hne hpd i c hadm
      (by rw [mahalanobis_eq_qform, mahalanobis_eq_qform]; exact hle)
    have hstab := hstable i c hadm
    -- The separation is strictly positive because the centroids differ.
    have hsep : 0 < centroidSeparation S z (z i) c := by
      refine qform_inv_pos hpd ?_
      exact sub_ne_zero.mpr hcent
    have hw := S.weight_pos i
    have hWa : S.weight i < cellMass S z (z i) := weight_lt_cellMass S z i hadm.2
    have hWb : 0 < cellMass S z c := cellMass_pos (hne c)
    have ha := alpha_pos hw hWa
    have hb := beta_pos hw hWb
    have hQ : 0 < alpha (S.weight i) (cellMass S z (z i)) *
        beta (S.weight i) (cellMass S z c) / 4 * centroidSeparation S z (z i) c ^ 2 :=
      mul_pos (div_pos (mul_pos ha hb) (by norm_num)) (pow_pos hsep 2)
    have hexpand : (fisher S z).det *
        (1 + alpha (S.weight i) (cellMass S z (z i)) *
          beta (S.weight i) (cellMass S z c) / 4 * centroidSeparation S z (z i) c ^ 2)
        = (fisher S z).det + (fisher S z).det *
          (alpha (S.weight i) (cellMass S z (z i)) *
            beta (S.weight i) (cellMass S z c) / 4 *
            centroidSeparation S z (z i) c ^ 2) := by ring
    have hpos := mul_pos hdetpos hQ
    linarith [hbound, hstab]
  · -- Singleton source: its own distance is zero and every other centroid differs.
    have hsingle : cell z (z i) = {i} := by
      have hi : i ∈ cell z (z i) := by simp
      exact Finset.eq_singleton_iff_unique_mem.mpr
        ⟨hi, fun x hx => Finset.card_le_one.mp (not_lt.mp hcard) x hx i hi⟩
    have hci : centroid S z (z i) = S.score i := centroid_of_singleton S z i hsingle
    rw [hci, sub_self]
    have hzero : qform (fisher S z)⁻¹ (0 : Fin d → ℝ) 0 = 0 := by simp [qform]
    rw [hzero]
    refine qform_inv_pos hpd ?_
    rw [← hci]
    exact sub_ne_zero.mpr (Ne.symm hcent)

/-- The same theorem with stability written on the D objective `F_D = log det I`. -/
theorem exchangeStableLogDet_implies_strictVoronoi (S : Sample d N) (z : Fin N → Fin K)
    (hmerged : Function.Injective S.score) (hne : ∀ c, (cell z c).Nonempty)
    (hpd : (fisher S z).PosDef) (hstable : ExchangeStableLogDet S z) :
    StrictVoronoi S z := by
  refine exchangeStable_implies_strictVoronoi S z hmerged hne hpd ?_
  intro i b hadm
  by_contra hlt
  push Not at hlt
  have hdetpos : 0 < (fisher S z).det := posDef_det_pos hpd
  have hafter : 0 < (fisher S (relocate z i b)).det := lt_trans hdetpos hlt
  have := hstable i b hadm
  have hmono : Real.log (fisher S z).det < Real.log (fisher S (relocate z i b)).det :=
    Real.log_lt_log hdetpos hlt
  linarith

end

end ScoreQuantFormal
