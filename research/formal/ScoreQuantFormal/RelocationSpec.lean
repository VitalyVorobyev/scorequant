import ScoreQuantFormal.ConfigSpec
import ScoreQuantFormal.ScalarExchangeSpec

/-!
# D2 specification: the exact weighted rank-two relocation identity

This file is the reviewed statement boundary for registry claim `D-RANK2-MOVE`,
proved in `KNOWN_RESULTS/04-d-optimality.md` §D2. A prover may change
`Relocation.lean`, but must not change this file without a new statement audit.

The registry statement reads:

> For a weighted point moved from non-singleton `a` to `b`,
> `ΔI = α u_a u_aᵀ − β u_b u_bᵀ` with `α = w W_a / (W_a − w)`,
> `β = w W_b / (W_b + w)`.

The correspondence is:

| registry clause | this file |
|---|---|
| positive weights | `Sample.weight_pos`, in `ConfigSpec.lean` |
| moved to a different cell | `RelocationAssumptions`, first conjunct |
| source non-singleton, so it stays nonempty | `RelocationAssumptions`, second conjunct |
| `W_b` is a genuine cell mass | `RelocationAssumptions`, third conjunct |
| — (narrowing; see `assumptions` in the claim node) | the destination cell is nonempty |
| `u_c = μ_c − s` | `centroid S z c - S.score i` |
| `α`, `β` | `alpha`, `beta`, frozen in `ScalarExchangeSpec.lean` |
| `ΔI = α u_a u_aᵀ − β u_b u_bᵀ` | `RelocationConclusion` |
| the implication itself | `RelocationConclusion` |

`relocate` is defined here rather than in `Relocation.lean` for the same reason
the vocabulary of `ConfigSpec.lean` is frozen: it appears inside this
conclusion, and inside the frozen `ExchangeStable` of `ExchangeVoronoiSpec.lean`.
A prover who could redefine `relocate` as the identity would make both
statements vacuous without touching an audited file.

**Not part of this specification.**

* The destination-empty case. `RelocationAssumptions` requires `cell z b`
  nonempty; the identity is not asserted for a move into an empty cell. There
  `β = w · 0 / (0 + w) = 0` by ordinary arithmetic, but `μ_b = 0` only by the
  junk-value convention `0⁻¹ = 0`, so whatever holds there holds for a reason
  outside the algebra this claim is about.
* The singleton-source case, excluded by the second conjunct: emptying a cell
  changes the number of cells and is a different move.
* Any determinant, log-determinant or ordering consequence — those are D3 and
  downstream.
* Any positivity or rank statement about `α` and `β`. This is an identity; that
  `α > 0`, `β > 0`, or that `ΔI` has rank two, is not claimed here.
* Zero or negative weights, which `Sample` excludes by construction.
* Merged-atom moves. One row of the sample moves; relocating a whole duplicate
  class, several rows at once, or iterating the move is not covered.
* Any population or atomless analogue of the identity.
* Any statement about the Python/JAX implementation. Nothing here connects
  `fisher`, `centroid` or `relocate` to the library's versions of them.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N K : ℕ}

/-- Relabel row `i` to cell `b`, leaving every other row alone. -/
def relocate (z : Fin N → Fin K) (i : Fin N) (b : Fin K) : Fin N → Fin K :=
  Function.update z i b

/-- The hypotheses of D2: the move is to a different cell, the source keeps at
least one other row, and the destination is a nonempty cell. Strict positivity
of the weights is carried by `Sample` itself. -/
def RelocationAssumptions (z : Fin N → Fin K) (i : Fin N) (b : Fin K) : Prop :=
  b ≠ z i ∧ (∃ j ∈ cell z (z i), j ≠ i) ∧ (cell z b).Nonempty

/-- **The frozen statement of `D-RANK2-MOVE`.** Under `RelocationAssumptions`,
the retained information changes by exactly `α u_a u_aᵀ − β u_b u_bᵀ`. -/
def RelocationConclusion (S : Sample d N) (z : Fin N → Fin K) (i : Fin N) (b : Fin K) :
    Prop :=
  RelocationAssumptions z i b →
    fisher S (relocate z i b) - fisher S z
      = alpha (S.weight i) (cellMass S z (z i)) •
          vecMulVec (centroid S z (z i) - S.score i) (centroid S z (z i) - S.score i)
        - beta (S.weight i) (cellMass S z b) •
          vecMulVec (centroid S z b - S.score i) (centroid S z b - S.score i)

end

end ScoreQuantFormal
