---
entity: CE-DS-GLOBAL-GEOMETRY-001
kind: fixture
---
This is the fixture behind `DS-GLOBAL-NONGEOMETRIC`: an exact eight-row, two-score-dimension, three-cell example, enumerated exhaustively, whose unique global profiled optimum violates its own first-order rule by margins of 2862/3239 and 618/3239. Calling `compile_quantizer()` on a result fitted under `ProfiledDOptimality` is refused, citing this fixture, because a profiled result has no canonical rule to compile into; the remedy is to fit a reusable rule directly with `fit_quantizer` rather than convert a fixed-sample partition into one.
