---
entity: D-UNMERGED-DUPLICATES-FAIL
kind: claim
---
The exchange-implies-Voronoi theorem assumes coincident score rows have been merged into one weighted atom. Without that assumption it is false: identical score rows sitting in different singleton cells can make a partition vacuously stable — there is no move that changes anything — while strict nearest-centre assignment and deterministic label reproduction both fail. The witness is one-dimensional with three singleton cells and two rows equidistant from their competing centres (`CE-D-UNMERGED-DUPLICATES-001`).

Merging duplicates is therefore part of the contract, not an optimization step. The solver refuses a terminal state whose own Mahalanobis rule would relabel a row for more than the gain tolerance, citing the same fixture, and the same code appears if a compile attempt hits the same condition.
