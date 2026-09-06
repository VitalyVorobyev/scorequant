import ScoreQuantFormal.ScalarExchangeSpec
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

namespace ScoreQuantFormal

/-- The relocation coefficients reproduce the reciprocal cell-mass identity. -/
theorem weightCoefficientIdentity
    {sourceWeight sourceMass destinationMass : ℝ}
    (hWeight : 0 < sourceWeight)
    (hSource : sourceWeight < sourceMass)
    (hDestination : 0 < destinationMass) :
    alpha sourceWeight sourceMass - beta sourceWeight destinationMass =
      alpha sourceWeight sourceMass * beta sourceWeight destinationMass *
        (1 / sourceMass + 1 / destinationMass) := by
  have hWeightNe : sourceWeight ≠ 0 := ne_of_gt hWeight
  have hSourceNe : sourceMass ≠ 0 := ne_of_gt (lt_trans hWeight hSource)
  have hDestinationNe : destinationMass ≠ 0 := ne_of_gt hDestination
  have hSourceDifferenceNe : sourceMass - sourceWeight ≠ 0 := ne_of_gt (sub_pos.mpr hSource)
  have hDestinationSumNe : destinationMass + sourceWeight ≠ 0 :=
    ne_of_gt (add_pos hDestination hWeight)
  unfold alpha beta
  field_simp
  ring

/-- The scalar core of the D-exchange violation lower bound. -/
theorem scalarExchangeStrengthenedLowerBound
    {sourceWeight sourceMass destinationMass qaa qbb qab : ℝ}
    (h : ScalarExchangeAssumptions
      sourceWeight sourceMass destinationMass qaa qbb qab) :
    ScalarExchangeConclusion sourceWeight sourceMass destinationMass qaa qbb qab := by
  rcases h with
    ⟨hWeight, hSource, hDestination, hQaa, hQbb, hViolation, hQDelta, hLeverage⟩
  have hAlpha : 0 < alpha sourceWeight sourceMass := by
    unfold alpha
    exact div_pos (mul_pos hWeight (lt_trans hWeight hSource)) (sub_pos.mpr hSource)
  have hBeta : 0 < beta sourceWeight destinationMass := by
    unfold beta
    exact div_pos (mul_pos hWeight hDestination) (add_pos hDestination hWeight)
  have hIdentity := weightCoefficientIdentity hWeight hSource hDestination
  have hCoefficient :
      alpha sourceWeight sourceMass - beta sourceWeight destinationMass ≥
        alpha sourceWeight sourceMass * beta sourceWeight destinationMass *
          qDelta qaa qbb qab := by
    rw [hIdentity]
    have hProduct : 0 < alpha sourceWeight sourceMass * beta sourceWeight destinationMass :=
      mul_pos hAlpha hBeta
    nlinarith
  unfold ScalarExchangeConclusion exchangeExcess qDelta at *
  nlinarith [sq_nonneg (qaa - qbb), sq_nonneg (qaa + qbb - 2 * qab)]

/-- The strengthened result implies the advertised quadratic lower bound. -/
theorem scalarExchangeLowerBound
    {sourceWeight sourceMass destinationMass qaa qbb qab : ℝ}
    (h : ScalarExchangeAssumptions
      sourceWeight sourceMass destinationMass qaa qbb qab) :
    exchangeExcess sourceWeight sourceMass destinationMass qaa qbb qab ≥
      alpha sourceWeight sourceMass * beta sourceWeight destinationMass / 4 *
        qDelta qaa qbb qab ^ 2 := by
  have hStrengthened := scalarExchangeStrengthenedLowerBound h
  have hData := h
  rcases hData with
    ⟨hWeight, hSource, hDestination, _hQaa, _hQbb, _hViolation, _hQDelta, _hLeverage⟩
  have hAlpha : 0 < alpha sourceWeight sourceMass := by
    unfold alpha
    exact div_pos (mul_pos hWeight (lt_trans hWeight hSource)) (sub_pos.mpr hSource)
  have hBeta : 0 < beta sourceWeight destinationMass := by
    unfold beta
    exact div_pos (mul_pos hWeight hDestination) (add_pos hDestination hWeight)
  have hCoefficient :
      0 ≤ alpha sourceWeight sourceMass * beta sourceWeight destinationMass / 4 := by
    positivity
  unfold ScalarExchangeConclusion at hStrengthened
  calc
    exchangeExcess sourceWeight sourceMass destinationMass qaa qbb qab ≥
        alpha sourceWeight sourceMass * beta sourceWeight destinationMass / 4 *
          (qDelta qaa qbb qab ^ 2 + (qaa - qbb) ^ 2) := hStrengthened
    _ ≥ alpha sourceWeight sourceMass * beta sourceWeight destinationMass / 4 *
        qDelta qaa qbb qab ^ 2 := by
      nlinarith [sq_nonneg (qaa - qbb)]

/-- Positive centroid separation makes the determinant-ratio excess positive. -/
theorem scalarExchangeExcessPositive
    {sourceWeight sourceMass destinationMass qaa qbb qab : ℝ}
    (h : ScalarExchangeAssumptions
      sourceWeight sourceMass destinationMass qaa qbb qab)
    (hDistinct : 0 < qDelta qaa qbb qab) :
    0 < exchangeExcess sourceWeight sourceMass destinationMass qaa qbb qab := by
  have hBound := scalarExchangeLowerBound h
  have hData := h
  rcases hData with
    ⟨hWeight, hSource, hDestination, _hQaa, _hQbb, _hViolation, _hQDelta, _hLeverage⟩
  have hAlpha : 0 < alpha sourceWeight sourceMass := by
    unfold alpha
    exact div_pos (mul_pos hWeight (lt_trans hWeight hSource)) (sub_pos.mpr hSource)
  have hBeta : 0 < beta sourceWeight destinationMass := by
    unfold beta
    exact div_pos (mul_pos hWeight hDestination) (add_pos hDestination hWeight)
  have hCoefficient :
      0 < alpha sourceWeight sourceMass * beta sourceWeight destinationMass / 4 := by
    positivity
  have hSquare : 0 < qDelta qaa qbb qab ^ 2 := sq_pos_of_pos hDistinct
  exact lt_of_lt_of_le (mul_pos hCoefficient hSquare) hBound

end ScoreQuantFormal
