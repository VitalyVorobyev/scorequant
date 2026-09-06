---
entity: CE-D-UNMERGED-DUPLICATES-001
kind: fixture
---
This is the fixture behind `D-UNMERGED-DUPLICATES-FAIL`: a one-dimensional example with three singleton cells and two rows equidistant from their competing centres, which is vacuously exchange-stable while strict nearest-centre assignment and deterministic label reproduction both fail. The solver refuses a terminal state whose own Mahalanobis rule would relabel a row by more than the gain tolerance, citing this fixture; the same code appears if a compile attempt hits the same condition.
