import ScoreQuantFormal.Counterexamples

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

/-- info: 'ScoreQuantFormal.violation_strict_gain' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms violation_strict_gain

/-- info: 'ScoreQuantFormal.exchange_voronoi' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms exchange_voronoi

/-- info: 'ScoreQuantFormal.violation_log_gain' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms violation_log_gain

/-- info: 'ScoreQuantFormal.centroid_leverage_bound' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms centroid_leverage_bound

/-- info: 'ScoreQuantFormal.centroid_ne_of_stable' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms centroid_ne_of_stable

/-- info: 'ScoreQuantFormal.exchangeStable_implies_strictVoronoi' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms exchangeStable_implies_strictVoronoi

/-- info: 'ScoreQuantFormal.exchangeStableLogDet_implies_strictVoronoi' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms exchangeStableLogDet_implies_strictVoronoi

/-! ## Corollaries D7 and D8 -/

/-- info: 'ScoreQuantFormal.globalOptimum_strictVoronoi' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms globalOptimum_strictVoronoi

/-- info: 'ScoreQuantFormal.no_infinite_strict_ascent' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms no_infinite_strict_ascent

/-- info: 'ScoreQuantFormal.exists_exchangeStable' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms exists_exchangeStable

/-! ## Boundary witnesses -/

/-- info: 'ScoreQuantFormal.UnmergedDuplicates.all_but_injectivity' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms UnmergedDuplicates.all_but_injectivity

/-- info: 'ScoreQuantFormal.UnmergedDuplicates.not_strictVoronoi' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms UnmergedDuplicates.not_strictVoronoi

/-- info: 'ScoreQuantFormal.UnmergedDuplicates.score_not_injective' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms UnmergedDuplicates.score_not_injective

/-- info: 'ScoreQuantFormal.VoronoiConverse.strictVoronoi' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms VoronoiConverse.strictVoronoi

/-- info: 'ScoreQuantFormal.VoronoiConverse.not_exchangeStable' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms VoronoiConverse.not_exchangeStable

end ScoreQuantFormal
