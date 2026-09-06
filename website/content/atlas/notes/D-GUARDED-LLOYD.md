---
entity: D-GUARDED-LLOYD
kind: claim
---
Batch reassignment is available as a proposal generator, but every proposal it produces is accepted only on the exact objective, never taken on faith, because the unguarded step is not monotone (`D-LLOYD-NONMONOTONE`). This guard is part of the solver contract rather than an optional safeguard. Setting `guard="exchange"` hands the proposed labels to the exact exchange engine, so the reported state stays compilable afterward.
