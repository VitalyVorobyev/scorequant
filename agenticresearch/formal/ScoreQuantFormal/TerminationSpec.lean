import ScoreQuantFormal.ExchangeVoronoiSpec

/-!
# The frozen termination layer

This file is the reviewed statement boundary for `D-EXCHANGE-TERMINATES` (§D8).

Its registry statement asserts three things about accepting only exact positive gains: strict
ascent, absence of cycles, and finite termination at a one-point exchange-stable state. Nothing
in that argument uses the determinant — only that the objective is real-valued and the labeling
set is finite — so it is stated once for an arbitrary `F : (Fin N → Fin K) → ℝ` and then
instantiated at the D objective.

`DS-EXCHANGE-TERMINATES` (§DS3) and `A-EXCHANGE-TERMINATES` (§A1) are the same argument at
`F_s` and `F_A`. This file does **not** cover them and they must not be marked from it: neither
objective exists as a Lean object anywhere in the tree — there is no Schur complement and no
trace of an inverse — and there is no frozen `Prop` for either. ADR 0037 admits the finite
profiled `D_s` core to the track, so the obstacle is the missing objects and conclusions, not
the scope. Their coverage is recorded in `KNOWN_RESULTS/` prose per protocol step E, naming the
generic declarations and saying what is missing.

`DLogTerminationConclusion` is **auxiliary**: no claim node states it, and no
`formal_proof.declaration` may name `d_log_exchange_terminates`. It exists so that the
determinant and `log det` runs cannot be quietly conflated.

`Corollaries.lean` already proves two fragments of D8 — that no infinite strictly-ascending
sequence of labelings exists, and that a maximizer exists. Neither is the registered claim,
which is about a *process* reaching a stable state; that is `TerminationConclusion` here, and
it is why `D-EXCHANGE-TERMINATES` was left unmarked.

A prover may change `Termination.lean` but must not change this file without a new statement
audit.

## Not part of this specification

* Positive gain tolerances. A run that accepts gains above `ε > 0` still terminates, but at an
  `ε`-stable state, which is a weaker conclusion and a separate claim.
* Any bound on the number of moves. Finiteness here is non-quantitative; nothing is claimed
  about how the run length scales with `N` or `K`.
* Any claim that the terminal state is a global optimum, or that different runs agree.
* Move neighbourhoods other than the one-point relocation `Admissible` names — no swaps, no
  merge-split, no boundary perturbation.
* Whether the objective is well defined at a singular candidate. `F` is total by assumption;
  the determinant instance below inherits `Real` totality.
* Agreement between the `det` and `log det` runs. Both terminate, and their terminal states
  differ: the accepted-move sets are not the same at a singular candidate, so a `det`-stable
  state can still admit a `log det`-improving move. Neither instance is a corollary of the
  other.
* The `D_s` and A objectives, which do not exist as Lean objects in this tree at all — no Schur
  complement, no trace of an inverse.
* Any link from a terminal stable state back to `VoronoiAssumptions` or `StrictVoronoi`. §D8's
  closing bullet, "by D5/D6, a canonical deployable D quantizer", is *not* chained here: nothing
  below carries nonempty cells or positive-definiteness along a run.
* Anything about the Python/JAX solver loop, its stopping rule, or its tolerance.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N K : ℕ}

/-- One exact positive-gain one-point move: an admissible relocation that strictly increases
the objective. -/
def ImprovingMove (F : (Fin N → Fin K) → ℝ) (z w : Fin N → Fin K) : Prop :=
  (∃ i b, Admissible z i b ∧ w = relocate z i b) ∧ F z < F w

/-- Stability for an arbitrary objective: no admissible relocation strictly improves it. -/
def StableFor (F : (Fin N → Fin K) → ℝ) (z : Fin N → Fin K) : Prop :=
  ∀ i b, Admissible z i b → F (relocate z i b) ≤ F z

/-- Reachability by finitely many exact positive-gain moves. This is the run itself: `Reaches
F z w` holds exactly when some finite sequence of accepted moves carries `z` to `w`. -/
inductive Reaches (F : (Fin N → Fin K) → ℝ) : (Fin N → Fin K) → (Fin N → Fin K) → Prop
  | refl (z : Fin N → Fin K) : Reaches F z z
  | tail {z w v : Fin N → Fin K} : Reaches F z w → ImprovingMove F w v → Reaches F z v

/-- **The frozen strict-ascent clause:** a run either ends where it started, or strictly
increased the objective.

This does *not* on its own exclude cycles: a cycle back to `z` satisfies the left disjunct
`z = w` and yields no contradiction. Absence of cycles is `NoCycle`, stated separately. -/
def AscentStrict (F : (Fin N → Fin K) → ℝ) : Prop :=
  ∀ z w, Reaches F z w → z = w ∨ F z < F w

/-- **The frozen no-cycle clause:** once a move is accepted, the run can never return to where
it came from. -/
def NoCycle (F : (Fin N → Fin K) → ℝ) : Prop :=
  ∀ z w, ImprovingMove F z w → ¬ Reaches F w z

/-- **The frozen no-infinite-run clause.** -/
def NoInfiniteRun (F : (Fin N → Fin K) → ℝ) : Prop :=
  ¬ ∃ seq : ℕ → (Fin N → Fin K), ∀ n, ImprovingMove F (seq n) (seq (n + 1))

/-- **The frozen termination clause:** from any starting labeling, exact positive-gain moves
reach a stable state in finitely many steps. This is the clause `Corollaries.lean` does not
prove and `D-EXCHANGE-TERMINATES` is registered as. -/
def TerminationConclusion (F : (Fin N → Fin K) → ℝ) : Prop :=
  ∀ z, ∃ w, Reaches F z w ∧ StableFor F w

/-- **The frozen D instance**, at the determinant objective, matching `ExchangeStable`'s
determinant convention: `StableFor` at this objective *is* `ExchangeStable`. -/
def DTerminationConclusion (S : Sample d N) (K : ℕ) : Prop :=
  AscentStrict (fun z : Fin N → Fin K => (fisher S z).det) ∧
    NoCycle (fun z : Fin N → Fin K => (fisher S z).det) ∧
      NoInfiniteRun (fun z : Fin N → Fin K => (fisher S z).det) ∧
        TerminationConclusion (fun z : Fin N → Fin K => (fisher S z).det)

/-- **The frozen `F_D = log det` instance.** Stated separately because it is a *different run*,
not a rephrasing. `ExchangeVoronoiSpec.lean`'s objective convention records that the two
stability notions are not interchangeable at a singular candidate, and the same gap moves the
accepted-move set: a state can be `det`-stable while a `log det`-improving move remains, so the
two runs stop in different places. Both terminate, which is what is claimed here. -/
def DLogTerminationConclusion (S : Sample d N) (K : ℕ) : Prop :=
  AscentStrict (fun z : Fin N → Fin K => Real.log (fisher S z).det) ∧
    NoCycle (fun z : Fin N → Fin K => Real.log (fisher S z).det) ∧
      NoInfiniteRun (fun z : Fin N → Fin K => Real.log (fisher S z).det) ∧
        TerminationConclusion (fun z : Fin N → Fin K => Real.log (fisher S z).det)

end

end ScoreQuantFormal
