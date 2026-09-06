---
entity: D-LLOYD-NONMONOTONE
kind: claim
---
The obvious algorithm — freeze the metric, reassign every row to its nearest centre, recompute — is not monotone for the determinant objective. One such step can decrease the log determinant. A committed eight-row, two-dimensional, three-cell example loses 0.136521 nat in a single step, and a seeded census recorded 57 decreasing steps out of 300.

The reason is ordinary: the tangent of a concave function is an upper bound, not a lower one, so the usual monotonicity argument for Lloyd-type iteration does not transfer to the determinant criterion. Consequently, the batch iteration is only usable as a proposal generator whose proposals are accepted on the exact objective (`D-GUARDED-LLOYD`).

No prior-art search has been recorded for this specific counterexample; it is possible that adaptive-metric Lloyd non-monotonicity already appears in the determinant-clustering computational literature.
