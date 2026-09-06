import ScoreQuantFormal.DetGainSpec

/-!
# D4 specification: the centroid leverage inequality

This file is the reviewed statement boundary for registry claim `D-LEVERAGE`,
proved in `KNOWN_RESULTS/04-d-optimality.md` §D4. A prover may change
`Leverage.lean`, but must not change this file without a new statement audit.

The registry statement reads:

> For every centroid `μ_c`, `μ_cᵀ I⁻¹ μ_c ≤ 1/W_c`, and for `δ = μ_a − μ_b`,
> `δᵀ I⁻¹ δ ≤ 1/W_a + 1/W_b`.

The correspondence is:

| registry clause | this file |
|---|---|
| `I` is the retained information of the labeling | `fisher S z` |
| `I⁻¹` exists | `LeverageAssumptions`, second conjunct |
| every `W_c` is a genuine positive cell mass | `LeverageAssumptions`, first conjunct |
| `μ_cᵀ I⁻¹ μ_c ≤ 1/W_c` | `LeverageConclusion`, first component |
| `δᵀ I⁻¹ δ ≤ 1/W_a + 1/W_b`, for any `a`, `b` | `LeverageConclusion`, second component |
| the implication itself | `LeverageConclusion` |

The claim node bundles two inequalities, so both are frozen here and both are
discharged by the single exported theorem: the protocol's rule is that a
bundled node is formalized whole or split, never marked on one half.

The nonemptiness hypothesis is global — `∀ c` — where the claim reads
per-centroid. That is the faithful form, not an oversight: the claim quantifies
over every centroid `μ_c`, and a labeling with an empty cell has no `μ_c` there,
so the claim's own scope is the all-cells-occupied one, exactly as D5's
"exactly `K` nonempty cells". The class this excludes is not empty; `PosDef`
does not imply it, except when `K ≤ d`.

**Not part of this specification.**

* **That `fisher` is a Fisher information.** `fisher` is *defined* in
  `ConfigSpec.lean` as `∑_c m_c m_cᵀ / W_c`, which is the right-hand side of
  `FI-QUANT-IDENTITY`. That this algebraic object is the retained information
  `Var(E[S∣Z])` of a statistical model is not proved here, and is not statable
  in this vocabulary: the equality needs `E[S] = 0` and weights forming a
  probability measure, neither of which the Lean assumes. `D-LEVERAGE` is a
  `bridge` node; the identification is inherited from `FI-QUANT-IDENTITY` and
  the regularity conditions that claim carries.
* **Normalized weights.** `Sample.weight` is any strictly positive function, so
  `cellMass` is a mass and `1/W_c` is not `1/P(Z = c)`. The statement is
  invariant under `w ↦ t · w`, so this is a generalization rather than a defect,
  but the frozen object is not literally the claim's object.
* **Zero-weight rows,** which `Sample.weight_pos` makes inexpressible.
* Empty cells. `LeverageAssumptions` requires every cell nonempty. Nothing is
  claimed about a labeling that leaves a cell empty — and note that Lean's
  `0⁻¹ = 0` would make such a case read `0 ≤ 0` rather than fail, so the
  hypothesis is a fidelity convention, not a guard against falsity.
* Singular or merely positive-semidefinite `I`, where `I⁻¹` is a junk value and
  the inequality has no content. `PosDef` is assumed, not derived — though for
  this `fisher` it is equivalent to nonsingularity, since a sum of rank-one PSD
  blocks is PSD unconditionally.
* Any strictness. Both inequalities are non-strict, and the equality cases are
  not characterized here.
* Any consequence for exchange stability or the D objective — those are D5 and
  downstream. What this statement is *for* is discharging the hypothesis
  `qDelta ≤ 1/sourceMass + 1/destinationMass` of `ScalarExchangeSpec.lean`, but
  that use is not part of the statement.
* Any statement about the Python/JAX implementation, or about the rank
  tolerance by which it decides that `I` is nonsingular.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N K : ℕ}

/-- The hypotheses of D4: every cell is nonempty, and the retained information
is positive definite. -/
def LeverageAssumptions (S : Sample d N) (z : Fin N → Fin K) : Prop :=
  (∀ c, (cell z c).Nonempty) ∧ (fisher S z).PosDef

/-- **The frozen statement of `D-LEVERAGE`,** both halves of the bundled claim:
the single-centroid bound and the centroid-difference bound. -/
def LeverageConclusion (S : Sample d N) (z : Fin N → Fin K) : Prop :=
  LeverageAssumptions S z →
    (∀ c, qform (fisher S z)⁻¹ (centroid S z c) (centroid S z c)
        ≤ (cellMass S z c)⁻¹) ∧
      (∀ a b, qform (fisher S z)⁻¹ (centroid S z a - centroid S z b)
            (centroid S z a - centroid S z b)
          ≤ (cellMass S z a)⁻¹ + (cellMass S z b)⁻¹)

end

end ScoreQuantFormal
