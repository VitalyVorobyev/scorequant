import ScoreQuantFormal.ExchangeVoronoi

/-!
# Axiom audit

`#guard_msgs` makes the dependency list part of the checked source. Any new
axiom dependency changes the message and fails the build until it is reviewed.

The allowlist is exactly `propext`, `Classical.choice`, `Quot.sound`. A `sorry`
anywhere in a proof would show up here as `sorryAx`.
-/

namespace ScoreQuantFormal

/-! ## Scalar core (the audited pilot) -/

/-- info: 'ScoreQuantFormal.weightCoefficientIdentity' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms weightCoefficientIdentity

/-- info: 'ScoreQuantFormal.scalarExchangeStrengthenedLowerBound' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms scalarExchangeStrengthenedLowerBound

/-- info: 'ScoreQuantFormal.scalarExchangeLowerBound' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms scalarExchangeLowerBound

/-- info: 'ScoreQuantFormal.scalarExchangeExcessPositive' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms scalarExchangeExcessPositive

/-! ## The finite D chain in general dimension -/

/-- info: 'ScoreQuantFormal.fisher_relocate_sub' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms fisher_relocate_sub

/-- info: 'ScoreQuantFormal.det_add_rank_two' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms det_add_rank_two

/-- info: 'ScoreQuantFormal.detRatio_eq_one_add_exchangeExcess' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms detRatio_eq_one_add_exchangeExcess

/-- info: 'ScoreQuantFormal.leverage_bound' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms leverage_bound

/-- info: 'ScoreQuantFormal.violation_lower_bound' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms violation_lower_bound

/-- info: 'ScoreQuantFormal.centroid_ne_of_stable' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms centroid_ne_of_stable

/-- info: 'ScoreQuantFormal.exchangeStable_implies_strictVoronoi' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms exchangeStable_implies_strictVoronoi

/-- info: 'ScoreQuantFormal.exchangeStableLogDet_implies_strictVoronoi' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms exchangeStableLogDet_implies_strictVoronoi

end ScoreQuantFormal
