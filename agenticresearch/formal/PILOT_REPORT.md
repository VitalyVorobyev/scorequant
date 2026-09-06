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

---

## Follow-up, 5 September 2026: the finite D chain in general dimension

The go/no-go above was taken up under [ADR 0030](../../docs/adr/0030-formal-verification-track.md),
which widened the approved track from "inner-product identities" to the whole finite D chain in
arbitrary dimension. D2, D3, D4, D5 and the D7/D8 corollaries are now machine-checked.

What transferred, and what the second round learned:

- **The frozen scalar core plugged in unmodified.** `ScalarExchangeAssumptions` turned out to be
  exactly what the matrix layer discharges: the violation hypothesis is `q_bb ≤ q_aa`, the
  positivity facts come from `PosDef`, and `q_δ ≤ 1/W_a + 1/W_b` is precisely D4. Freezing the
  right six scalars in August was what made September cheap.
- **Representing a cell by its unnormalized weighted sum** rather than its centroid reduced the
  D2 algebra to two standalone rank-one identities in `(W, T)` plus a cancellation, and kept
  Finset bookkeeping out of the matrix proofs.
- **`Matrix.det_one_add_mul_comm`** — Weinstein–Aronszajn — turns the `d × d` determinant into a
  `2 × 2` one in a single rewrite. No specialized tactic was needed here either.
- **The projector argument for D4 needs no Cauchy–Schwarz**: `1 − P` is symmetric idempotent, so
  `vᵀ(1−P)v` is a sum of squares.
- **The statement audit earned its keep.** It found that the frozen file contained the conclusion
  but neither the hypotheses nor the implication — a prover could have weakened the theorem
  without touching an audited file — and a false docstring claim that the determinant and
  log-determinant objectives are interchangeable, with an explicit witness. Both were fixed
  before the marker went into the registry.

Nanoda remains disabled for the same upstream reason.
