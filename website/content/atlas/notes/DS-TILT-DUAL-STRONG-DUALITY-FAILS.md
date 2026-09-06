---
entity: DS-TILT-DUAL-STRONG-DUALITY-FAILS
kind: claim
---
Strong duality between the tilt primal and dual is false in general, and the gap can be order one rather than a rounding artifact: an exact four-row, three-cell table has a gap above 0.68, and the smallest possible witness — three rows, two cells — has an exact gap of 1/6.

This should not be read as surprising: minimax interchange on a finite nonconvex feasible set is expected to fail in general, and the contribution is the exact witnesses rather than the phenomenon itself. Separately, a reported non-closing bracket is not evidence that a duality gap exists — a tie in the dynamic program can let a deterministic policy return a labelling that hides an available closure — so certification requires exhibiting the specific closing labelling rather than merely reporting an interval.
