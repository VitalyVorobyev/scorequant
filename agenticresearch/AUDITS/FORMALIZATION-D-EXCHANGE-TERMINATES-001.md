# Formal statement audit — D-EXCHANGE-TERMINATES (001)

**Audited object:** `formal/ScoreQuantFormal/TerminationSpec.lean`, the newly frozen
statement boundary offered simultaneously for `D-EXCHANGE-TERMINATES`,
`DS-EXCHANGE-TERMINATES` and `A-EXCHANGE-TERMINATES`. The freeze it depends on —
`ConfigSpec.lean`, `RelocationSpec.lean`, `ScalarExchangeSpec.lean`,
`DetGainSpec.lean`, `LeverageSpec.lean`, `ExchangeVoronoiSpec.lean` — was read as
part of the boundary. `Termination.lean` and `Corollaries.lean` were read only to
check that the frozen statements are what the exported theorems have as types.

**Canonical source:** `claims/D-EXCHANGE-TERMINATES.json`,
`claims/DS-EXCHANGE-TERMINATES.json`, `claims/A-EXCHANGE-TERMINATES.json`;
`KNOWN_RESULTS/04-d-optimality.md` §D8, `KNOWN_RESULTS/05a-ds-core.md` §DS3,
`KNOWN_RESULTS/07-a-optimality.md` §A1; scope from ADR 0030 in `docs/decisions.md`;
trust policy from `formal/README.md`; prior boundary audit
`AUDITS/FORMALIZATION-D-EXCHANGE-IMPLIES-VORONOI-001.md`.

**Date:** 7 September 2026. **Verdict: match after hardening.**

For `D-EXCHANGE-TERMINATES` alone the mathematics is right and the hardenings F1
and F2 are additive; proof work is not blocked. **F3 is BLOCKING for
`DS-EXCHANGE-TERMINATES` and `A-EXCHANGE-TERMINATES`:** this spec contains no
frozen conclusion for either, so neither can carry `formal_proof` on the strength
of it. None of the hardenings below has been applied; the spec must be re-frozen
and this audit re-run only if F1 or F2 changes a statement.

**Independence.** This audit ran in a session with no shared derivation context.
It was given the three claim nodes, the three prose sections, the Lean files, the
prior formalization audits, ADR 0030 and the formal README, and no part of the
formalizing session's transcript. `lake` was not run (see "Not checked by this
audit").

## Correspondence

### Hypotheses

`TerminationSpec.lean` states three propositions with no explicit hypothesis
binders. Its hypotheses are carried by the types, so the table below records
where each lives and what removing it does.

| # | Hypothesis | Lean | Removing it |
|---|---|---|---|
| H1 | the label set is finite | `Fin K` in `F : (Fin N → Fin K) → ℝ` | **FALSE** — with an infinite label set an infinite strictly improving run exists |
| H2 | the row set is finite | `Fin N` in the same type | **FALSE** — flip rows `0,1,2,…` of an infinite sample under `F(z)=Σ_{z i = 1} 2^{-i}` and every move is strictly improving |
| H3 | only exact positive gains are accepted | `F z < F w`, second conjunct of `ImprovingMove` | **FALSE** — with `≤`, `AscentStrict` fails and a constant `F` gives an infinite run |
| H4 | the objective is real-valued, so its order is irreflexive and transitive | codomain `ℝ` | **FALSE** — `AscentStrict` is exactly transitivity plus irreflexivity of `<` |
| H5 | moves are one-point relocations preserving a nonempty source | `Admissible`, `relocate` (frozen in `ExchangeVoronoiSpec.lean` / `RelocationSpec.lean`) | **VACUOUS** if narrowed toward unsatisfiable (`StableFor` becomes trivially true and `refl` discharges `TerminationConclusion`); **still true, and stronger,** if widened to arbitrary relabelings. This is the only vacuity lever, and it is inside the freeze. |
| H6 | strictly positive weights, dimension `d` | `Sample.weight_pos`, `Sample d N` | **still TRUE** — the argument never inspects the weights. Inherited by the `D` instance, not used by it. |
| H7 | *(absent by design)* continuity, monotonicity, or well-definedness of `F` | — | correctly absent; adding any of them would weaken the statement |

H7 is the point of the file and it is sound. `TerminationConclusion` for an
arbitrary `F : (Fin N → Fin K) → ℝ` is not suspiciously strong: it is the standard
finite local-search termination lemma, where finiteness of the domain and
irreflexive-transitive strict order on the codomain are the entire content. The
proof in `Termination.lean` confirms this — its measure is
`#{v | F z < F v}`, and nothing about `F` beyond `<` is touched.

### Conclusions

| # | Registry clause | Claim | Lean | Status |
|---|---|---|---|---|
| C1 | "strictly increases the objective" (per move) | all three | `ImprovingMove`, second conjunct | frozen, definitional |
| C2 | "strictly increases the objective" (along a run) | all three | `AscentStrict` | frozen, but under-states the intent — see **F1** |
| C3 | "cannot cycle" | D | *no frozen proposition* | semantically entailed by `NoInfiniteRun`; **not** by `AscentStrict`, contrary to its docstring — see **F1** |
| C4 | "terminates finitely" (every run) | all three | `NoInfiniteRun` | frozen, exact |
| C5 | "terminates finitely" (a run exists) | all three | `Reaches` is an inductive type, so every inhabitant is a finite chain | frozen, exact |
| C6 | "at a one-point exchange-stable state" | all three | `TerminationConclusion` = `∀ z, ∃ w, Reaches F z w ∧ StableFor F w` | frozen, exact |
| C7 | "one-point" | all three | `relocate` + `Admissible` | frozen, exact |
| C8 | the objective is the D objective `F_D = log det I` | D | `fun z => (fisher S z).det` in `DTerminationConclusion` | frozen, but a different run at singular candidates — see **F2** |
| C9 | terminal state is `ExchangeStable` in D5's sense | D | `StableFor (fun z => (fisher S z).det) z ↔ ExchangeStable S z` | **exact, definitionally** — `stableFor_det_iff` is `Iff.rfl` |
| C10 | the objective is the profiled `F_s = log det I − log det I_λλ` | Ds | *absent from the whole Lean development* | **F3, BLOCKING for that claim** |
| C11 | the objective is `F_A = −tr(I⁻¹)` | A | *absent from the whole Lean development* | **F3, BLOCKING for that claim** |
| C12 | "on a finite labeling set" | Ds, A | `Fin N → Fin K` | frozen, exact |

Nothing appears formally that is absent informally.

## Findings and dispositions

**F1 — "cannot cycle" is asserted in a docstring and attributed to the wrong
proposition (hardening; required).**

`AscentStrict F : ∀ z w, Reaches F z w → z = w ∨ F z < F w` is documented as
"a run either has not moved, or strictly increased the objective. Absence of
cycles is the immediate consequence." Both sentences are wrong as written.
`z = w` is not "has not moved" — it is *exactly the cycle case*, the endpoints
coinciding. Applying `AscentStrict` to a nonempty cycle `Reaches F z z` yields
`z = z ∨ F z < F z`, whose left disjunct is true, so no contradiction follows.
`AscentStrict` alone therefore does **not** exclude cycles.

The mathematical content is nonetheless present in the freeze, twice over:

- from `AscentStrict` plus the `Reaches.tail` constructor, in two lines — given
  `ImprovingMove F z z'` and `Reaches F z' z`, `AscentStrict` gives `z' = z` (then
  `F z < F z' = F z`) or `F z' < F z` (contradicting `F z < F z'`);
- from `NoInfiniteRun`, by iterating a cycle into a periodic sequence — a longer
  but routine induction.

So C3 is entailed, not merely asserted. What is missing is a proposition whose
*type* says it. The registry `statement` for `D-EXCHANGE-TERMINATES` names three
clauses; `DTerminationConclusion` conjoins two of them plus a docstring. Under the
README rule that `formal_proof.declaration` names a theorem whose type is the
frozen conclusion, marking the claim with `d_exchange_terminates` today would mark
a three-clause statement with a two-clause conclusion. This is the same shape of
defect the prior audit recorded as F1 against `ExchangeVoronoiSpec.lean`.

*Required hardening:* add to the frozen spec

```lean
/-- **The frozen no-cycle clause:** no nonempty run returns to its start. -/
def NoCycle (F : (Fin N → Fin K) → ℝ) : Prop :=
  ∀ z w, ImprovingMove F z w → ¬ Reaches F w z
```

conjoin it into `DTerminationConclusion`, prove it in `Termination.lean`, guard it
in `AxiomAudit.lean`, and correct the `AscentStrict` docstring to say that its
first disjunct is the coinciding-endpoint case rather than "has not moved".

**F2 — the frozen D instance is the determinant run, and the determinant run is
provably not the `F_D` run (hardening; required).**

`DTerminationConclusion` instantiates at `fun z => (fisher S z).det`. The ledger's
D objective is `F_D = log det I`, and `D-LOGDET-GAIN` is this claim's declared
dependency, so "exact positive D gains" in the registry `statement` means
`Δ log det I > 0`.

For the *terminal state* the determinant convention is the right choice and I
confirm it: `StableFor` at `det` is `ExchangeStable` up to beta reduction
(`stableFor_det_iff := Iff.rfl`), which is the weaker stability hypothesis D5
takes, so the terminal state feeds `exchangeStable_implies_strictVoronoi`
directly. Choosing `ExchangeStableLogDet` here would have been the unsound
simplification the prior audit's F2 warned against.

For the *accepted moves* the choice is not neutral, and the two runs demonstrably
differ. Reusing the witness verified in
`AUDITS/FORMALIZATION-D-EXCHANGE-IMPLIES-VORONOI-001.md` §F2 — `d = K = 2`,
`N = 3`, scores `(0,1)`, `(1,0)`, `(0,-1)`, unit weights, `z = (0,0,1)`, where
`det I = 1/2`, `ExchangeStable` holds, and some admissible candidate has
determinant `0` — at that `z`:

- `StableFor (fun z => (fisher S z).det) z` **holds**, so the frozen run stops;
- `StableFor (fun z => Real.log (fisher S z).det) z` **fails**, because
  `Real.log 0 = 0 > Real.log (1/2)`, so the `F_D` run accepts a move and continues.

Neither acceptance rule implies the other in general: they coincide exactly when
both candidates are nonsingular, and diverge at `det = 0`. The spec's non-coverage
list does disclose that the two phrasings are not interchangeable, but it frames
this as a question about whether the objective is *well defined*, not as the fact
that the *set of accepted moves* differs — a reader can finish the file believing
the `F_D` process has been formalized.

*Required hardening:* either add the free second instance

```lean
AscentStrict (fun z : Fin N → Fin K => Real.log (fisher S z).det) ∧ …
```

to `DTerminationConclusion` — `terminates _` discharges it unchanged, and its
terminal state is `ExchangeStableLogDet`, which under `det I > 0` is the stronger
stability — or state plainly in the non-coverage list and in §D8 that the
formalized acceptance rule is determinant increase, that it coincides with
`ΔF_D > 0` only when both candidates are nonsingular, and cite the witness above.
Adding the instance is the better fix, because it makes the registry sentence
literally true of the marked type.

**F3 — the spec claims three claims and freezes a conclusion for one (BLOCKING
for `DS-EXCHANGE-TERMINATES` and `A-EXCHANGE-TERMINATES`).**

The docstring opens "This file is the reviewed statement boundary for the
finite-ascent half of `D-EXCHANGE-TERMINATES`, `DS-EXCHANGE-TERMINATES`,
`A-EXCHANGE-TERMINATES` … It is therefore stated once, for an arbitrary `F`, and
instantiated." It is instantiated **once**, at the determinant. There is no
`DsTerminationConclusion`, no `ATerminationConclusion`, and — more fundamentally —
neither `F_s = log det I − log det I_λλ` nor `F_A = −tr(I⁻¹)` is defined anywhere
in the Lean development. The Schur block `I_λλ` does not exist as a Lean object;
`Matrix.trace` of an inverse appears nowhere.

The generic argument does of course cover them mathematically: nothing in the
proof inspects the objective. But the README's rule is that a claim's
`formal_proof.declaration` names a theorem *whose type is the frozen conclusion*.
The only candidates here are `ascent_strict`, `no_infinite_run` and `terminates`,
whose types quantify over an arbitrary `F`. A reader checking a `DS` mark against
`terminates : ∀ F, TerminationConclusion F` would find no `D_s` objective anywhere
in the type, and so no way to confirm that the marked statement is the registered
one. That is precisely the drift the rule exists to prevent.

*On the ADR question, specifically:*

- ADR 0030 bounds the Lean track to "the finite D chain in general dimension …
  and nothing beyond it until a further decision; profiled `D_s`, population
  measure theory and asymptotics are out." `DS-EXCHANGE-TERMINATES` is named out
  of scope by that sentence. `A-EXCHANGE-TERMINATES` is out by the "nothing beyond
  it" clause — A-optimality is not part of the D chain.
- **Attaching `formal_proof` to either claim node requires a new ADR**, plus
  frozen `DsTerminationConclusion` / `ATerminationConclusion` definitions written
  in a frozen `F_s` / `F_A`, plus a fresh statement audit of those instances.
  Nothing in this spec substitutes for that.
- **No new ADR is needed to record the coverage in prose.** Protocol step E has
  the exact escape hatch: "Lean results whose statements are not separately frozen
  and audited, or that cover a claim only in part, are recorded in
  `KNOWN_RESULTS/` prose instead, naming the declaration and saying what is
  missing." So §DS3 and §A1 may say that the objective-agnostic finite-ascent and
  termination argument is machine-checked as `ScoreQuantFormal.terminates`,
  `ascent_strict` and `no_infinite_run`, and that the profiled and A objectives
  are not defined in Lean, hence the claims carry no `formal_proof`. That is the
  disposition I recommend.
- **This is not blocking for `D-EXCHANGE-TERMINATES`.** D8 may be marked once F1
  and F2 are settled.

*Required hardening either way:* narrow the docstring's opening sentence to
`D-EXCHANGE-TERMINATES`, and move the `DS`/`A` mention into a sentence that says
the argument is objective-agnostic and *would* serve them once their objectives
are frozen under a new ADR — not that this file is their statement boundary.

**F4 — the non-coverage list is accurate but incomplete (hardening; required).**

Every one of its six bullets is true and I confirm each: no positive gain
tolerances (the spec uses strict `<` with zero tolerance), no move-count bound
(`Reaches` is non-quantitative, and the `betterCount` bound of `K^N` in the proof
module is deliberately not exported), no global-optimality or run-agreement claim,
no move neighbourhood beyond `Admissible`, `F` total by assumption, nothing about
the Python/JAX loop. Missing:

1. **That the `D_s` and A objectives have no Lean definition** (F3). The list
   should say so directly, since the docstring's opening sentence invites the
   opposite reading.
2. **That the accepted-move set at `det` differs from the one at `log det`** (F2),
   not merely that the two phrasings are "not interchangeable".
3. **That nothing links a terminal stable state back to `VoronoiAssumptions` or
   `StrictVoronoi`.** §D8's fifth prose bullet is "by D5/D6, a canonical deployable
   D quantizer"; that chaining is not formalized. It is in fact close at hand —
   `Admissible` preserves `∀ c, (cell z c).Nonempty` (the source keeps a row, the
   destination gains one), `Function.Injective S.score` does not depend on `z`, and
   along a determinant-improving run from a `PosDef` start `det` only increases,
   so a PSD `fisher` with `det > 0` stays `PosDef`. None of that is stated or
   proved, and `fisher`'s positive semidefiniteness is not proved anywhere in the
   tree. Until it is, "the run ends at a deployable quantizer" is not machine-
   checked, and the list should say so. (§D8's fifth bullet is outside the
   registry `statement` field, so this is a disclosure item, not a mismatch.)
4. **That no-cycle is not a frozen proposition** (F1), if F1's `NoCycle` is not
   added.

**F5 — the frozen import chain runs through four editable modules (hygiene;
recommended, not blocking).**

The freeze *is* closed under definitional dependency — see the confirmation below
— but `TerminationSpec` imports `ExchangeVoronoiSpec`, which imports `Leverage`
(editable), which pulls in `DetGain`, `Relocation` and `Config`, all editable.
`relocate`, a frozen definition that both `ExchangeStable` and `ImprovingMove`
depend on, reaches the frozen layer only by passing through three editable proof
modules. No definition leaks, and a prover cannot redefine a name that already
exists, but the frozen statement layer elaborates inside an environment a prover
controls (instances, `simp` attributes, local notation). `ExchangeVoronoiSpec`
needs nothing from `Leverage.lean` itself: `import ScoreQuantFormal.LeverageSpec`
plus `import ScoreQuantFormal.RelocationSpec` supplies every definition it uses.
Making the `*Spec` chain import only `*Spec` modules would make the freeze closed
under *import*, not just under definitional dependency.

**F6 — `Reaches.head` is exported but not axiom-guarded (cosmetic).**

`formal/README.md` promises "a guarded `#print axioms` per exported theorem" and
`AxiomAudit.lean` guards `ascent_strict`, `no_infinite_run`, `terminates`,
`stableFor_det_iff` and `d_exchange_terminates`. `Reaches.head` in
`Termination.lean` is public and unguarded. Either guard it or make it `private`
(`betterCount` and `terminates_aux` correctly are). CI's namespace-wide
`axiom-audit` presumably covers it, but the file-level promise does not hold as
written.

**F7 — `formal/README.md`'s coverage table omits the new modules (cosmetic).**

The table stops at `Counterexamples.lean` / `AxiomAudit.lean`. `MergeSpec`,
`Merge`, `ClosureSpec`, `Closure`, `TerminationSpec` and `Termination` are all
absent, and the "Not covered" paragraph still lists the compiled predictor
`D-FINITE-INDUCTIVE-CLOSURE`, which `ClosureSpec.lean` now appears to address.
A step-E recording item, not a statement defect.

## Independently confirmed correct

- **`Reaches` is the right formalization of a run, and the `refl` case is
  required for the statement to be true.** `Reaches` is literally Mathlib's
  `Relation.ReflTransGen (ImprovingMove F)` — same two constructors, same tail
  orientation — restated locally so the freeze does not depend on a Mathlib
  definition. `Reaches F z z` by `refl` means a zero-length run, and
  `TerminationConclusion` being satisfied at an already-stable `z` by the empty
  run is correct behaviour, not a weakening: a local-search loop started at a
  stable point does terminate there, and demanding at least one move would make
  the statement **false** at every stable start. The `refl` case only becomes a
  weakening if `StableFor` is trivially true, which is H5's vacuity lever, and
  that lever is frozen (see the next item).
- **`TerminationConclusion` is not vacuous, and there is a witness in the repo.**
  `VoronoiConverse.not_exchangeStable : ¬ ExchangeStable sample labeling` in
  `Counterexamples.lean` exhibits a concrete `(S, z)` where `StableFor` at the
  determinant fails, so the terminal condition is a real constraint and the
  `refl` run does not discharge it in general.
- **`TerminationConclusion` is genuinely stronger than `exists_exchangeStable`,
  and the new spec earns its existence.** `exists_exchangeStable :
  ∃ z, ExchangeStable S z` produces one stable labeling — in fact a *global
  maximizer*, since `globalOptimum_exchangeStable` is how it is obtained — with no
  statement that it is reachable from anywhere. `TerminationConclusion` quantifies
  over every starting labeling and asserts that a stable state is reachable *by
  accepted moves* from that start. It implies `exists_exchangeStable` on a
  nonempty domain; the converse fails, because a global maximizer sitting in a
  different basin is not `Reaches`-reachable. That the two are different theorems
  is visible in the proofs: `exists_exchangeStable` is `Finset.exists_max_image`,
  while `terminates` runs a strictly decreasing measure `#{v | F z < F v}` along
  actual moves and never mentions a maximum. This is the process claim, and the
  spec's framing of why D8 was left unmarked is accurate.
- **The universal and existential readings of "terminates" are both present.**
  `NoInfiniteRun` is the universal half — *no* run is infinite — and
  `TerminationConclusion` the existential half. The registry sentence is fairly
  rendered by the pair; neither alone would be.
- **`StableFor` at the determinant is `ExchangeStable` exactly.** Unfolding
  `StableFor (fun z => (fisher S z).det) z` gives `∀ i b, Admissible z i b →
  (fisher S (relocate z i b)).det ≤ (fisher S z).det`, character for character
  `ExchangeStable`. `stableFor_det_iff` being provable by `Iff.rfl` is the machine
  confirmation. The terminal state of the frozen D run is therefore literally the
  hypothesis of `exchangeStable_implies_strictVoronoi`.
- **The freeze is closed under definitional dependency; no leak.**
  `TerminationSpec.lean` is written in `Admissible` (`ExchangeVoronoiSpec`),
  `relocate` (`RelocationSpec`), `fisher` and `Sample` (`ConfigSpec`),
  `Matrix.det`, `Fin`, `ℝ` and its own six definitions. Every project definition
  it mentions lives in a `*Spec.lean`. Transitively, `ExchangeVoronoiSpec` adds
  `cell`, `cellMass`, `centroid`, `mahalanobis` (`ConfigSpec`), `qform`
  (`DetGainSpec`) and `alpha`, `beta` (`ScalarExchangeSpec`) — all frozen. Nothing
  in the audited statements resolves to a definition in `Config.lean`,
  `Relocation.lean`, `DetGain.lean`, `Leverage.lean`, `ExchangeVoronoi.lean`,
  `Corollaries.lean`, `Merge.lean`, `Closure.lean` or `Termination.lean`. See F5
  for the import-level, as opposed to definition-level, observation.
- **Exported theorem types are the frozen conclusions.** `ascent_strict F :
  AscentStrict F`, `no_infinite_run F : NoInfiniteRun F`, `terminates F :
  TerminationConclusion F`, `d_exchange_terminates S K : DTerminationConclusion
  S K` — each type is the frozen `Prop` and nothing else, with no added
  hypothesis. `betterCount` and `terminates_aux` are `private`, so the measure
  argument is not part of the exported surface. `d_exchange_terminates` is the
  declaration a `formal_proof` mark on `D-EXCHANGE-TERMINATES` should name, once
  F1 and F2 are settled; `DTerminationConclusion` taking `K` explicitly is right,
  because it puts the label count in the marked type.
- **Every new module is inside the default build target.**
  `ScoreQuantFormal.lean` imports `AxiomAudit`, which imports `Termination`
  (and `Closure`, and `Counterexamples`), so `lean_lib ScoreQuantFormal` builds
  them. This was F5 in the prior audit and it has stayed fixed.
- **Degenerate instances are harmless.** At `K = 0` with `N > 0` the labeling
  type is empty and the `∀ z` statements are vacuously true; at `N = 0` the type
  is a singleton and `Admissible` is unsatisfiable, so `StableFor` is vacuous.
  Both are true instances of a statement that is non-vacuous in general.
- **The weights are inert, correctly.** The D instance carries
  `Sample.weight_pos` and dimension `d` without using them (H6). This makes the
  instance weaker than the mathematics supports, which is safe, and it is why
  stating the argument once for an arbitrary `F` is the right structure.

## What remains uncovered

The frozen statements concern an abstract real-valued objective on a finite
labeling set, moved by one-point relocations that keep the source cell nonempty,
with zero gain tolerance. Nothing is claimed about how long a run is, which stable
state it reaches, whether two runs agree, whether the terminal state is globally
optimal, whether it is nonsingular or has `K` nonempty cells, or whether it is
strictly D-Voronoi — so §D8's "by D5/D6, a canonical deployable D quantizer" is
not machine-checked. Nothing is claimed for positive gain tolerances, which is the
regime every real solver is in. The `D_s` and A objectives are not Lean objects.
And nothing connects `fisher`, `relocate`, `Admissible` or the run itself to
`optimize_partition`, its move ordering, its stopping rule, its tolerance, or to
floating-point arithmetic.

## Not checked by this audit

`lake build --wfail` was not run, by instruction. This audit is about the frozen
statements and the *types* of the exported theorems; whether `Termination.lean`
compiles, and whether the `#guard_msgs` axiom messages in `AxiomAudit.lean` match,
remains for the trust gate. The `Merge`/`Closure` modules added on the same branch
were read only far enough to confirm they do not participate in the audited
statements.
