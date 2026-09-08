import ScoreQuantFormal.ConfigSpec

/-!
# The frozen merge layer

`D-FINITE-INDUCTIVE-CLOSURE` (§D6) is stated on *merged distinct positive-weight score atoms*
and then extended to the original rows: "original duplicate rows inherit the merged atom's
label". That extension is only meaningful once merging is known to preserve the objects the
compiled rule is built from — the cell masses, the cell sums, the centroids and the retained
information. This file freezes that statement.

The same invariance is what `D-EXCHANGE-IMPLIES-VORONOI`'s *second* duplicate branch would
need. `ExchangeVoronoiSpec.lean` records that branch as unformalized because
`Function.Injective S.score` cannot express it; `IsMerge` can, by naming the merge map
instead of asserting injectivity.

This file is part of the reviewed statement boundary: **a prover may not change it without a
new statement audit.** Definitions only; every lemma about them lives in `Merge.lean`.

## Correspondence to the registry statement

| registry clause | this file |
|---|---|
| "merged … score atoms" | `IsMerge`, whose `score_eq` and `weight_eq` say the atom carries its rows' common score and their total weight |
| "distinct" | **not here** — supplied by `VoronoiAssumptions`' injectivity in `ClosureSpec.lean` |
| "positive-weight" | **not here** — carried by `Sample.weight_pos` |
| "original duplicate rows" | the domain `Fin N'` of `map` |
| "inherit the merged atom's label" | the labeling `z ∘ map`, used in `MergeConclusion` |

`MergeConclusion` is **auxiliary**: no claim node states it, and no
`formal_proof.declaration` may ever name `merge_invariance`. It exists to be used by
`ClosureSpec.lean`.

## Not part of this specification

* That the merged scores are *distinct*. `IsMerge` does not require
  `Function.Injective S.score`; a merge that leaves two equal-score atoms unmerged still
  satisfies it. Distinctness is a hypothesis of the D5/D6 conclusions, carried there, not here.
  (Under `ClosureDuplicateConclusion`'s hypotheses the merge is nonetheless forced to be
  complete: two equal-score rows of `S'` landing in different atoms would make `S.score` fail
  injectivity.)
* Any minimality or canonicity of the merge: nothing says `map` merges *all* duplicates, only
  that whatever it merges, it merges consistently.
* Commutation with anything but the statistics of **one fixed labeling**. Merging is not proved
  to commute with `relocate`, and not with stability — which is exactly why `ClosureSpec.lean`
  must put `ExchangeStable` on the merged side. A one-point move on `S'` can split a duplicate
  class, which no move on `S` can express, so the two stability notions are not equivalent.
* Any labeling of `S'` other than the inherited `z ∘ map`.
* D5's *second* duplicate branch, "labels constant on every duplicate class". `IsMerge` can
  name a merge map, but **no frozen statement here derives constancy** — `z ∘ map` is constant
  on fibres by construction. `ExchangeVoronoiSpec.lean`'s disclaimer of that branch stands.
* Zero-weight rows: `Sample.weight_pos` makes them inexpressible. Surjectivity of `map` is
  therefore *not* a field — it is derivable, since an empty fibre would make `weight_eq` read
  `S.weight i = 0`, contradicting `Sample.weight_pos i`. `IsMerge` is kept minimal so that every
  field it carries is one the audit can justify.
* Any determinant, ordering, stability or geometric consequence. Those are `ClosureSpec.lean`.
* Anything about the Python/JAX implementation's duplicate handling.
-/

namespace ScoreQuantFormal

open Matrix

noncomputable section

variable {d N N' K : ℕ}

/-- `map` merges the rows of `S'` into the atoms of `S`: every original row carries the score
of its atom, and an atom's weight is the total weight of its rows. Those are the two fields;
inhabitedness of every atom is a consequence, not an assumption (see the non-coverage list).

The fibre of `map` over an atom `i` is `cell map i`, so an atom's weight is `cellMass S' map i`
— the merge is written in the same vocabulary as a labeling, because that is what it is. -/
structure IsMerge (S' : Sample d N') (S : Sample d N) (map : Fin N' → Fin N) : Prop where
  /-- Every original row carries the score of the atom it merges into. -/
  score_eq : ∀ j, S'.score j = S.score (map j)
  /-- An atom's weight is the total weight of the rows merging into it. -/
  weight_eq : ∀ i, S.weight i = cellMass S' map i

/-- **The frozen statement of merge invariance.** Labelling the original rows by inheritance,
`z ∘ map`, leaves every quantity the compiled D rule is built from unchanged. -/
def MergeConclusion (S' : Sample d N') (S : Sample d N) (map : Fin N' → Fin N)
    (z : Fin N → Fin K) : Prop :=
  IsMerge S' S map →
    (∀ c, cellMass S' (z ∘ map) c = cellMass S z c) ∧
      (∀ c, cellSum S' (z ∘ map) c = cellSum S z c) ∧
        (∀ c, centroid S' (z ∘ map) c = centroid S z c) ∧
          fisher S' (z ∘ map) = fisher S z

end

end ScoreQuantFormal
