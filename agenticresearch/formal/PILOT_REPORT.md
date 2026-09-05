# Scalar D-exchange formalization pilot

**Date:** 31 August 2026  
**Claim:** `D-EXCHANGE-SCALAR-CORE`  
**Verdict:** pilot accepted; the next formal layer may be proposed separately

## Result

Lean checks the reciprocal cell-mass identity, the strengthened scalar lower
bound, its advertised quadratic corollary, and strict positivity under positive
centroid separation. The statement audit deliberately leaves the matrix,
determinant, logarithm, and finite Voronoi boundary arguments outside the
machine-checked marker.

## What the pilot learned

- Separating `ScalarExchangeSpec.lean` from the proof made the
  informal-to-formal boundary reviewable and prevented proof search from
  silently reshaping the claim.
- Real division required a `noncomputable` specification section; this affects
  executable code generation, not theorem soundness.
- The coefficient identity needed explicit denominator nonzero facts. After
  that, `field_simp`, `ring`, `positivity`, and `nlinarith` were sufficient;
  no specialized prover or custom tactic was needed.
- The checked proof depends only on `propext`, `Classical.choice`, and
  `Quot.sound`. Guarded axiom messages ensure this list cannot drift silently.
- The bundled `leanchecker` accepts the environment. Nanoda was also exercised,
  but its current parser rejected the Lean 4.33.1 export with the known
  `invalid digit found in string` incompatibility; required CI therefore uses
  `leanchecker` plus namespace-wide `axiom-audit` instead of a knowingly broken
  nanoda gate.

## Go/no-go

Go for a separately reviewed next step covering inner-product identities and
the scalar-to-geometric interpretation. Do not start population measure theory,
profiled \(D_s\), or a specialized prover integration. The finite D dependency
chain remains the only approved expansion path, and this report does not itself
authorize that next implementation.
