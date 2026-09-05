import ScoreQuantFormal.ScalarExchange

/-!
# Axiom audit

`#guard_msgs` makes the dependency list part of the checked source. Any new
axiom dependency changes the message and fails the build until it is reviewed.
-/

namespace ScoreQuantFormal

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

end ScoreQuantFormal
