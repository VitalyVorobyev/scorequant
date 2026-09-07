---
entity: DS-GLOBAL-NONGEOMETRIC
kind: claim
---
Worse than the first-order failure above, a globally optimal profiled partition can violate its own first-order rule. The witness is exact: eight rows, two score dimensions, one parameter of interest, three cells, enumerated exhaustively, with the unique global optimum violating its own rule by margins of 2862/3239 and 618/3239 (`CE-DS-GLOBAL-GEOMETRY-001`).

The consequence is that a profiled result has no canonical rule to compile into. Calling `compile_quantizer()` on one is refused, citing this fixture, and the remedy is to fit an explicit quantizer rather than convert a fixed-sample profiled partition into one. Verifying this result authorized no change to the library's existing refusal behaviour — it confirmed the behaviour was already correct.
