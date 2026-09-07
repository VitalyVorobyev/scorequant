---
entity: FI-RANK-CEILING
kind: claim
---
With `K` categories and a `d`-dimensional parameter, the binned information has rank at most `min(d, K-1)`, so a nonsingular answer needs at least `d + 1` categories. Asking three bins to measure four parameters is not a hard optimization problem to solve poorly; it is an impossible one, and the library refuses such a configuration rather than returning a degenerate answer.
