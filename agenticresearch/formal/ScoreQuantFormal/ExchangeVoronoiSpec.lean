import ScoreQuantFormal.Leverage
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Exchange-to-Voronoi specification

This file is the reviewed statement boundary for the finite D theorem

* registry claim `D-EXCHANGE-IMPLIES-VORONOI` (manuscript v9 Theorem 2),
* registry claim `D-EXCHANGE-VIOLATION-LOWER-BOUND`,

proved in `KNOWN_RESULTS/04-d-optimality.md` §D5. A prover may change
`ExchangeVoronoi.lean`, but must not change this file without a new statement
audit.

The registry statement reads:

> For distinct positive-weight score atoms obtained by merging duplicate rows,
> positive-definite retained information, exactly `K` nonempty cells, no move
> constraints beyond preserving nonempty cells, and zero-tolerance exact
> stability, every tied-or-worse nearest-centroid comparison from a
> non-singleton source has strictly positive exact D gain. Stability forces
> distinct centroids; singleton rows are then strictly nearest to their own
> centroids. Hence one-point exchange stability implies strict self-consistent
> D-Voronoi geometry.

The correspondence is:

| registry hypothesis | this file |
|---|---|
| strictly positive atom weights | `Sample.weight_pos` |
| coincident rows merged into distinct atoms | `Function.Injective S.score` |
| `I_q ≻ 0` | `(fisher S z).PosDef` |
| exactly `K` nonempty cells | `∀ c, (cell z c).Nonempty` |
| no move constraint beyond preserving nonempty cells | `Admissible` |
| exact zero-gain-tolerance stability | `ExchangeStable` |
| strict self-consistent D-Voronoi | `StrictVoronoi` |

**Objective convention.** `ExchangeStable` is stated on `det I` rather than on
`F_D = log det I`. Under `(fisher S z).PosDef` the determinant is strictly
positive and `log` is strictly monotone there, so the two agree; the
determinant form is used because `Real.log` is junk-valued at a nonpositive
determinant and that junk value would otherwise silently change the strength of
the hypothesis. `ExchangeStableLogDet` records the `F_D` phrasing, and
`ExchangeVoronoi.lean` proves the theorem for it too.

**Not part of this specification.** Positive gain tolerances, zero-weight rows,
singular or pseudodeterminant objectives, capacity or mass constraints,
unmerged duplicate atoms, the converse implication, the compiled predictor, or
any statement about the Python/JAX implementation.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N K : ℕ}

/-- A relocation is admissible when it sends row `i` to a different cell and
leaves the source cell nonempty. This is the only move constraint. -/
def Admissible (z : Fin N → Fin K) (i : Fin N) (b : Fin K) : Prop :=
  b ≠ z i ∧ ∃ j ∈ cell z (z i), j ≠ i

/-- Zero-tolerance one-point exchange stability: no admissible relocation
strictly increases the retained information determinant. -/
def ExchangeStable (S : Sample d N) (z : Fin N → Fin K) : Prop :=
  ∀ i b, Admissible z i b → (fisher S (relocate z i b)).det ≤ (fisher S z).det

/-- The same stability written on the D objective `F_D = log det I`. -/
def ExchangeStableLogDet (S : Sample d N) (z : Fin N → Fin K) : Prop :=
  ∀ i b, Admissible z i b →
    Real.log (fisher S (relocate z i b)).det ≤ Real.log (fisher S z).det

/-- Strict self-consistent D-Voronoi geometry: every row is strictly nearer its
own centroid than any competing centroid, in the `I⁻¹` metric. -/
def StrictVoronoi (S : Sample d N) (z : Fin N → Fin K) : Prop :=
  ∀ i c, c ≠ z i →
    mahalanobis (fisher S z)⁻¹ (S.score i) (centroid S z (z i))
      < mahalanobis (fisher S z)⁻¹ (S.score i) (centroid S z c)

/-- The squared centroid separation in the `I⁻¹` metric, as it appears in the
quantitative bound. -/
def centroidSeparation (S : Sample d N) (z : Fin N → Fin K) (a b : Fin K) : ℝ :=
  qform (fisher S z)⁻¹ (centroid S z a - centroid S z b) (centroid S z a - centroid S z b)

/-- The quantitative violation lower bound of `D-EXCHANGE-VIOLATION-LOWER-BOUND`:
a tied-or-worse nearest-centroid comparison from a non-singleton source has
determinant ratio at least `1 + (α β / 4) q_δ²`. -/
def ViolationLowerBound (S : Sample d N) (z : Fin N → Fin K) : Prop :=
  ∀ i b, Admissible z i b →
    mahalanobis (fisher S z)⁻¹ (S.score i) (centroid S z b)
        ≤ mahalanobis (fisher S z)⁻¹ (S.score i) (centroid S z (z i)) →
      (fisher S z).det *
          (1 + alpha (S.weight i) (cellMass S z (z i)) *
              beta (S.weight i) (cellMass S z b) / 4 *
              centroidSeparation S z (z i) b ^ 2)
        ≤ (fisher S (relocate z i b)).det

end

end ScoreQuantFormal
