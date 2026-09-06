import Mathlib.Data.Real.Basic

/-!
# Scalar exchange specification

This file is the reviewed statement boundary for the scalar core of the finite
D-exchange theorem. A prover may change `ScalarExchange.lean`, but must not
change this file without a new statement audit.

The variables are the scalar quantities from D5:

* `sourceWeight`, `sourceMass`, and `destinationMass` define the relocation
  coefficients `alpha` and `beta`;
* `qaa`, `qbb`, and `qab` are the three quadratic products;
* `qDelta = qaa + qbb - 2 * qab` is the squared centroid separation;
* `exchangeExcess` is the determinant ratio minus one.

No matrix, determinant, logarithm, singleton, or duplicate-atom statement is
part of this specification.
-/

namespace ScoreQuantFormal

noncomputable section

/-- Positive rank-one coefficient for removing an atom from its source cell. -/
def alpha (sourceWeight sourceMass : ℝ) : ℝ :=
  sourceWeight * sourceMass / (sourceMass - sourceWeight)

/-- Positive rank-one coefficient for adding an atom to its destination cell. -/
def beta (sourceWeight destinationMass : ℝ) : ℝ :=
  sourceWeight * destinationMass / (destinationMass + sourceWeight)

/-- Squared centroid separation expressed through the three quadratic products. -/
def qDelta (qaa qbb qab : ℝ) : ℝ :=
  qaa + qbb - 2 * qab

/-- The determinant ratio minus one after the scalar rank-two reduction. -/
def exchangeExcess
    (sourceWeight sourceMass destinationMass qaa qbb qab : ℝ) : ℝ :=
  let a := alpha sourceWeight sourceMass
  let b := beta sourceWeight destinationMass
  a * qaa - b * qbb - a * b * (qaa * qbb - qab ^ 2)

/-- Exact hypotheses used by the scalar lower-bound pilot. -/
def ScalarExchangeAssumptions
    (sourceWeight sourceMass destinationMass qaa qbb qab : ℝ) : Prop :=
  0 < sourceWeight ∧
    sourceWeight < sourceMass ∧
    0 < destinationMass ∧
    0 ≤ qaa ∧
    0 ≤ qbb ∧
    qbb ≤ qaa ∧
    0 ≤ qDelta qaa qbb qab ∧
    qDelta qaa qbb qab ≤ 1 / sourceMass + 1 / destinationMass

/-- The strengthened scalar lower bound audited against D5. -/
def ScalarExchangeConclusion
    (sourceWeight sourceMass destinationMass qaa qbb qab : ℝ) : Prop :=
  exchangeExcess sourceWeight sourceMass destinationMass qaa qbb qab ≥
    alpha sourceWeight sourceMass * beta sourceWeight destinationMass / 4 *
      (qDelta qaa qbb qab ^ 2 + (qaa - qbb) ^ 2)

end

end ScoreQuantFormal
