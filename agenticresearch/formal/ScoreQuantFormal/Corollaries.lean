import ScoreQuantFormal.ExchangeVoronoi
import Mathlib.Data.Fintype.EquivFin

/-!
# D7 and D8: global realizability and finite termination

Two corollaries of `exchangeStable_implies_strictVoronoi`:

* **D7** (`D-GLOBAL-GEOMETRIC-REALIZABILITY`) — a finite global D optimum is
  exchange stable, hence strictly D-Voronoi, hence geometrically realizable.
* **D8** (`D-EXCHANGE-TERMINATES`) — accepting only exact positive gains gives
  strict ascent, and the labeling set is finite, so no run is infinite and a
  terminal exchange-stable labeling always exists.

D7 here is only the realizability half of the registry claim. The further
statement that unrestricted assignment and optimization over realizable
affine-max labelings share the same optimum value is *not* formalized.

Registry claims: `D-GLOBAL-GEOMETRIC-REALIZABILITY`, `D-EXCHANGE-TERMINATES`.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N K : ℕ}

/-- A labeling is a finite global D optimum when no relabeling has a larger
retained-information determinant. -/
def GlobalOptimum (S : Sample d N) (z : Fin N → Fin K) : Prop :=
  ∀ z' : Fin N → Fin K, (fisher S z').det ≤ (fisher S z).det

/-- A global optimum is in particular exchange stable: relocations are a
special case of relabeling. -/
theorem globalOptimum_exchangeStable (S : Sample d N) (z : Fin N → Fin K)
    (hopt : GlobalOptimum S z) : ExchangeStable S z :=
  fun i b _ => hopt (relocate z i b)

/-- **D7, geometric realizability of finite global optima.** A finite global D
optimum with `K` nonempty cells and nonsingular retained information is
strictly D-Voronoi. -/
theorem globalOptimum_strictVoronoi (S : Sample d N) (z : Fin N → Fin K)
    (hmerged : Function.Injective S.score) (hne : ∀ c, (cell z c).Nonempty)
    (hpd : (fisher S z).PosDef) (hopt : GlobalOptimum S z) :
    StrictVoronoi S z :=
  exchangeStable_implies_strictVoronoi S z hmerged hne hpd
    (globalOptimum_exchangeStable S z hopt)

/-- **D8, finite termination.** Exact positive gains give strict ascent, and the
labeling set is finite, so there is no infinite run of improving moves. -/
theorem no_infinite_strict_ascent (S : Sample d N) (seq : ℕ → (Fin N → Fin K))
    (hgain : ∀ n, (fisher S (seq n)).det < (fisher S (seq (n + 1))).det) : False := by
  have hmono : StrictMono fun n => (fisher S (seq n)).det :=
    strictMono_nat_of_lt_succ hgain
  have hinj : Function.Injective seq := by
    intro m n hmn
    by_contra hne
    rcases lt_or_gt_of_ne hne with hlt | hlt
    · exact absurd (congrArg (fun z => (fisher S z).det) hmn) (ne_of_lt (hmono hlt))
    · exact absurd (congrArg (fun z => (fisher S z).det) hmn.symm) (ne_of_lt (hmono hlt))
  exact not_injective_infinite_finite seq hinj

/-- A terminal state always exists: the labeling set is a nonempty finite type,
so the determinant attains a maximum, and that maximizer is exchange stable. -/
theorem exists_exchangeStable (S : Sample d N) [Nonempty (Fin N → Fin K)] :
    ∃ z : Fin N → Fin K, ExchangeStable S z := by
  classical
  obtain ⟨z, -, hz⟩ :=
    Finset.exists_max_image (Finset.univ : Finset (Fin N → Fin K))
      (fun z => (fisher S z).det) Finset.univ_nonempty
  exact ⟨z, globalOptimum_exchangeStable S z fun z' => hz z' (Finset.mem_univ z')⟩

end

end ScoreQuantFormal
