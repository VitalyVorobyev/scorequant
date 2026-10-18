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
| coincident rows merged into distinct atoms | `VoronoiAssumptions`, first conjunct |
| exactly `K` nonempty cells | `VoronoiAssumptions`, second conjunct |
| `I_q ≻ 0` | `VoronoiAssumptions`, third conjunct |
| no move constraint beyond preserving nonempty cells | `Admissible` |
| exact zero-gain-tolerance stability | `ExchangeStable` |
| strict self-consistent D-Voronoi | `StrictVoronoi` |
| the implication itself | `ExchangeVoronoiConclusion` |

Hypotheses, conclusion *and the arrow between them* are all frozen here, as they
are one layer down in `ScalarExchangeSpec.lean`. A prover that could add a
hypothesis in the proof module could weaken the theorem — for instance by
assuming the distinct centroids that D5 makes a point of deriving — without
touching an audited file.

**Objective convention.** `ExchangeStable` is stated on `det I` rather than on
`F_D = log det I`. The two are *not* interchangeable: `ExchangeStableLogDet`
implies `ExchangeStable` whenever `det I > 0`, but the converse fails at a
singular candidate, where `Real.log 0 = 0` can exceed `Real.log (det I) < 0`.
The determinant form is therefore deliberately the weaker hypothesis, hence the
stronger theorem; the `F_D` phrasing is proved as a corollary in
`ExchangeVoronoi.lean` rather than assumed here.

**Not part of this specification.**

* The registry's *second* duplicate branch — "labels constant on every duplicate
  class" — which the claim also asserts and which `Function.Injective S.score`
  cannot express. Only the merged-atoms branch is formalized.
* Positive gain tolerances. Every real solver is in that regime, where the
  guarantee degrades to "no geometric disagreement has exact gain above ε".
* Zero-weight rows, capacity, balance, or minimum-mass constraints, any of which
  can make a genuine geometric violation inadmissible.
* Singular or pseudodeterminant objectives, and projected-subspace variants.
* The compiled predictor (`D-FINITE-INDUCTIVE-CLOSURE`). The converse
  (`D-VORONOI-NOT-EXCHANGE`) is not stated in general; only its explicit
  witness is checked, in `Counterexamples.lean`. The two downstream corollaries
  D7 and D8 *are* formalized, in `Corollaries.lean`.
* The population/atomless statement (`D-POP-VORONOI`) and any empirical-to-
  population transfer.
* Score-estimation error and the observation-to-score step.
* Information-loss consequences: no D-efficiency bound, no bound on the worst
  normalized retention eigenvalue, no held-out guarantee.
* Any statement about the Python/JAX implementation. Nothing here connects
  `fisher`, `centroid` or `relocate` to the library's versions of them, nor to
  floating-point conditioning or rank tolerances.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N K : ℕ}

/-- The standing hypotheses of the finite D theorem: merged distinct atoms,
exactly `K` nonempty cells, and nonsingular retained information. Positivity of
the weights is carried by `Sample` itself. -/
def VoronoiAssumptions (S : Sample d N) (z : Fin N → Fin K) : Prop :=
  Function.Injective S.score ∧ (∀ c, (cell z c).Nonempty) ∧ (fisher S z).PosDef

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

/-- **The frozen statement of `D-EXCHANGE-IMPLIES-VORONOI`.** -/
def ExchangeVoronoiConclusion (S : Sample d N) (z : Fin N → Fin K) : Prop :=
  VoronoiAssumptions S z → ExchangeStable S z → StrictVoronoi S z

/-- The squared centroid separation in the `I⁻¹` metric, as it appears in the
quantitative bound. -/
def centroidSeparation (S : Sample d N) (z : Fin N → Fin K) (a b : Fin K) : ℝ :=
  qform (fisher S z)⁻¹ (centroid S z a - centroid S z b) (centroid S z a - centroid S z b)

/-- **The frozen quantitative half of `D-EXCHANGE-VIOLATION-LOWER-BOUND`:** a
tied-or-worse nearest-centroid comparison from a non-singleton source has
determinant ratio at least `1 + (α β / 4) q_δ²`. -/
def ViolationLowerBound (S : Sample d N) (z : Fin N → Fin K) : Prop :=
  VoronoiAssumptions S z →
    ∀ i b, Admissible z i b →
      mahalanobis (fisher S z)⁻¹ (S.score i) (centroid S z b)
          ≤ mahalanobis (fisher S z)⁻¹ (S.score i) (centroid S z (z i)) →
        (fisher S z).det *
            (1 + alpha (S.weight i) (cellMass S z (z i)) *
                beta (S.weight i) (cellMass S z b) / 4 *
                centroidSeparation S z (z i) b ^ 2)
          ≤ (fisher S (relocate z i b)).det

/-- **The frozen strict half of `D-EXCHANGE-VIOLATION-LOWER-BOUND`:** for
distinct centroids the same comparison has strictly positive exact D gain. -/
def ViolationStrictGain (S : Sample d N) (z : Fin N → Fin K) : Prop :=
  VoronoiAssumptions S z →
    ∀ i b, Admissible z i b → centroid S z (z i) ≠ centroid S z b →
      mahalanobis (fisher S z)⁻¹ (S.score i) (centroid S z b)
          ≤ mahalanobis (fisher S z)⁻¹ (S.score i) (centroid S z (z i)) →
        (fisher S z).det < (fisher S (relocate z i b)).det

/-- **The frozen `F_D` phrasing of `D-EXCHANGE-VIOLATION-LOWER-BOUND`:**
`ΔF_D ≥ log(1 + α β q_δ² / 4) > 0` for distinct centroids, with
`F_D = log det I`. This is the registry statement verbatim; the determinant
forms above are the same fact without `Real.log`. -/
def ViolationLogGain (S : Sample d N) (z : Fin N → Fin K) : Prop :=
  VoronoiAssumptions S z →
    ∀ i b, Admissible z i b → centroid S z (z i) ≠ centroid S z b →
      mahalanobis (fisher S z)⁻¹ (S.score i) (centroid S z b)
          ≤ mahalanobis (fisher S z)⁻¹ (S.score i) (centroid S z (z i)) →
        0 < Real.log (1 + alpha (S.weight i) (cellMass S z (z i)) *
              beta (S.weight i) (cellMass S z b) / 4 *
              centroidSeparation S z (z i) b ^ 2) ∧
          Real.log (1 + alpha (S.weight i) (cellMass S z (z i)) *
              beta (S.weight i) (cellMass S z b) / 4 *
              centroidSeparation S z (z i) b ^ 2)
            ≤ Real.log (fisher S (relocate z i b)).det - Real.log (fisher S z).det

end

end ScoreQuantFormal
