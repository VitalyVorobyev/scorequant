# Formal statement audit — D-FINITE-INDUCTIVE-CLOSURE / D-EXCHANGE-TERMINATES (002)

**Kind:** fix-verification pass over the two round-001 audits. Not a fresh audit:
the mathematics is not re-derived from scratch, but every statement whose text
changed was re-checked in full.

**Audited object:** the frozen boundary after hardening —
`formal/ScoreQuantFormal/MergeSpec.lean`, `ClosureSpec.lean`,
`TerminationSpec.lean`; the editable modules `Closure.lean`, `Merge.lean`,
`Termination.lean`, `AxiomAudit.lean`; the claim nodes
`D-FINITE-INDUCTIVE-CLOSURE`, `D-CLOSURE-DUPLICATE-INHERITANCE`,
`D-COMPILE-TOLERANCE-GUARANTEE`, `D-EXCHANGE-TERMINATES`;
`KNOWN_RESULTS/04-d-optimality.md` §D6 and §D8; `formal/README.md`; ADR 0037 in
`docs/decisions.md`.

**Checklist:** `AUDITS/FORMALIZATION-D-FINITE-INDUCTIVE-CLOSURE-001.md` (F1–F11)
and `AUDITS/FORMALIZATION-D-EXCHANGE-TERMINATES-001.md` (F1–F7).

**Date:** 7 September 2026. **Verdict: fixes verified with residue.**

Every blocking finding from both round-001 audits is addressed. Two findings are
partially addressed and are recorded below as accepted, documented residue. Six
new items surfaced from the changed text; none of them makes a frozen statement
false, one of them (N1) should be settled before marking, and one (N3) is a stale
citation now sitting inside a frozen file.

**Independence.** This pass ran with no shared derivation context.
`WORK/active/FORMAL-D-CLOSURE.md` was deliberately left unopened, as in round 001.
`lake` was not run, by instruction. `python3 py/registry.py validate` *was* run —
it is a read-only registry check and finding D6-F3 asks for it explicitly. It
reports `registry clean`.

## Per-finding verdicts

### D-FINITE-INDUCTIVE-CLOSURE (001)

| # | Finding | Verdict |
|---|---|---|
| F1 | nonempty-cells hypothesis missing from statement, prose, table | **ADDRESSED** — in the node `statement`, in `assumptions`, in §D6 prose with the refuting configuration, and as its own row in `ClosureSpec.lean`'s table plus a bolded paragraph |
| F2 | docstring asserts a claim split that had not happened | **ADDRESSED** — `claims/D-COMPILE-TOLERANCE-GUARANTEE.json` exists, §D6's `**Claims:**` line lists it, and `ClosureSpec.lean` now says the node exists *and* that nothing here formalizes it. Wording matches reality. See N2 |
| F3 | one `declaration` cannot name three theorems | **ADDRESSED** — three frozen conclusions became two nodes and two Props; `NearestCentroidRuleExists` folded into `ClosureConclusion`. Declarations named below; `registry.py` uniqueness and `theorem <name>` both pass. See N1 |
| F4 | `IsMerge.surjective` redundant, docstring wrong | **ADDRESSED** — field deleted; no `surjective` token remains anywhere in `ScoreQuantFormal/`; docstring now states the derivation; nothing depended on it |
| F5 | non-vacuity chain had an unstated link | **ADDRESSED** — folded into `ClosureConclusion` as `0 < d → ∃ q …`, with the discharge argument in the docstring and `nonempty_labels_of_posDef` in `Closure.lean`. Scrutinised below; the conjunct is true at every corner |
| F6 | a frozen file imports an editable one | **PARTIALLY ADDRESSED** — `ClosureSpec.lean` fixed; `ExchangeVoronoiSpec.lean:1` still imports `ScoreQuantFormal.Leverage`. Named as the one remaining exception in `formal/README.md` |
| F7 | definitional closure holds | still holds — `MergeSpec` → `ConfigSpec`; `ClosureSpec` → `ExchangeVoronoiSpec`, `MergeSpec`; every definition reachable from a frozen conclusion is in a `*Spec.lean` |
| F8 | ADR 0030 does not cover D6 | **ADDRESSED** — ADR 0037 supersedes the scope clause with "the finite theory", naming D6 explicitly; the supersession table records it; `formal/README.md`'s *Not covered* paragraph no longer lists the compiled predictor |
| F9 | exported theorem types are the frozen conclusions | still holds for the survivors; re-verified below |
| F10 | non-coverage lists accurate but incomplete | **PARTIALLY ADDRESSED** — `ClosureSpec.lean` gained four of the five requested disclaimers; `MergeSpec.lean` gained none of its three. Detail below |
| F11 | free strengthening available (observation) | no action was required and none was taken; still correct |

### D-EXCHANGE-TERMINATES (001)

| # | Finding | Verdict |
|---|---|---|
| F1 | "cannot cycle" asserted in a docstring, attributed to the wrong Prop | **ADDRESSED** — `NoCycle` frozen, conjoined into `DTerminationConclusion`, proved as `no_cycle`, guarded; `AscentStrict`'s docstring corrected. See N6 for one surviving copy of the old sentence in an editable module |
| F2 | the frozen D instance is the determinant run, not the `F_D` run | **ADDRESSED** — via the audit's second option *and* the free instance: `DLogTerminationConclusion` frozen, `d_log_exchange_terminates` proved and guarded, and both `TerminationSpec.lean` and §D8 now state that the accepted-move sets differ at a singular candidate |
| F3 | spec claimed three claims, froze a conclusion for one | **PARTIALLY ADDRESSED** — the blocking half is fully honoured (docstring narrowed to D8; explicit "must not be marked from it"; DS/A nodes unmarked; §D8 records the generic coverage). The recording sits in §D8 only — §DS3 and §A1 are untouched. See N3 for a stale reason |
| F4 | non-coverage list incomplete | **ADDRESSED** — all four requested items present (`D_s`/A have no Lean object; det vs log-det accepted-move sets; no chain back to `VoronoiAssumptions`/`StrictVoronoi`; no-cycle now frozen so item 4 is moot) |
| F5 | frozen import chain runs through editable modules | **PARTIALLY ADDRESSED** — same single exception as D6-F6; `TerminationSpec` → `ExchangeVoronoiSpec` → `Leverage` |
| F6 | `Reaches.head` exported but unguarded | **ADDRESSED** — guarded in `AxiomAudit.lean` |
| F7 | README coverage table omits the new modules | **ADDRESSED** — `MergeSpec`/`Merge`, `ClosureSpec`/`Closure`, `TerminationSpec`/`Termination` all listed; `Corollaries.lean` re-described as "the earlier weaker D8 fragments" |

## Scrutiny of the changed statements

### D6-F5: the `0 < d` guard and the new existence conjunct

`ClosureConclusion` is now

```lean
VoronoiAssumptions S z → ExchangeStable S z →
  (0 < d → ∃ q, IsNearestCentroidRule S z q) ∧
    ∀ q, IsNearestCentroidRule S z q → ∀ i, q (S.score i) = z i
```

The second conjunct is textually unchanged from the version audit 001 swept
exhaustively, so only the first needed real scrutiny.

**`nonempty_labels_of_posDef` is sound.** `fisher S z = ∑ c : Fin K, cellBlock …`
(`ConfigSpec.lean:66`). With `Fin K` empty the sum is over `∅`, so `fisher = 0`.
With `0 < d`, `Fin d` is nonempty, so `Matrix.det_zero` gives `det 0 = 0`, while
`PosDef.det_pos` gives `0 < det`. Contradiction; hence `Nonempty (Fin K)`. The
implication is true. (Mechanically the proof also looks right: `hempty : IsEmpty
(Fin K)` and `hdne : Nonempty (Fin d)` are Prop-classes, so Lean 4 registers them
as local instances, which is exactly what `Finset.univ_eq_empty` and the
`[Nonempty n]` on `Matrix.det_zero` need. `Matrix.PosDef.det_pos` lives in
`Mathlib/Analysis/Matrix/PosDef.lean`, reachable because `Closure.lean` imports
`ExchangeVoronoi.lean`, which imports `Mathlib.Analysis.Matrix.PosDef`.)

**The new conjunct is not false in any corner.** `∃ q, IsNearestCentroidRule S z q`
fails exactly when `Fin K` is empty, because `Fin d → ℝ` is inhabited for every
`d` (at `d = 0` by the empty function) and there is no function into `Fin 0`. So
the conjunct can only fail at `0 < d ∧ K = 0` — and there `fisher = 0`, whose
determinant is `0`, so `VoronoiAssumptions` is unsatisfiable. The guard is
discharged rather than assumed, exactly as the docstring says.

**The guard is necessary, and `ClosureConclusion` is true at `d = 0`.** Enumerating
the `d = 0` corner under the hypotheses: `Function.Injective S.score` forces
`N ≤ 1` (all scores are the empty function), and `∀ c, (cell z c).Nonempty` forces
`K ≤ N`.
- `N = 1`: every cell must be occupied by the single row, so `K = 1`. `Fin 1` is a
  singleton, so any `q` returns `0 = z 0`. Conclusion true; existence also true.
- `N = 0`: all cells are empty, so `K = 0`. `fisher` is the empty matrix, `PosDef`
  holds vacuously, so the hypotheses *are* satisfiable — and `∃ q` is **false**
  there. Without the guard `ClosureConclusion` would be false at `d = K = N = 0`.
  With the guard, `0 < 0` is false and the conjunct is vacuous; the second
  conjunct quantifies over `Fin 0` and is vacuous too.

So the guard is both necessary and sufficient, and no corner refutes the amended
statement. The correspondence-table row "the compiled rule exists at all →
`ClosureConclusion`, first conjunct" does not mention the guard; the paragraph
below it does. Worth one parenthetical in the row for the same reason F1 existed
(see N4).

### D8-F1: is `NoCycle` really "cannot cycle"?

```lean
def NoCycle (F) : Prop := ∀ z w, ImprovingMove F z w → ¬ Reaches F w z
```

A cycle is a run of length ≥ 1 from a state back to itself. `Reaches` appends at
the tail, so any `Reaches F z z` with at least one move decomposes as
`Reaches F z v` together with `ImprovingMove F v z`; instantiating `NoCycle` at
`(v, z)` contradicts it directly. Transitivity of `Reaches` extends this to any
repeated state along a run. `NoCycle` is therefore equivalent to "no run returns
to a state it has left", which is what the registry clause means. It is frozen in
`TerminationSpec.lean:86`, proved in `Termination.lean:115` (`ascent_strict` in
both branches, sound), and guarded.

`DTerminationConclusion` now conjoins `AscentStrict ∧ NoCycle ∧ NoInfiniteRun ∧
TerminationConclusion`. That covers all three clauses of the D8 statement —
strict ascent, cannot cycle, terminates finitely at a one-point exchange-stable
state — with `NoInfiniteRun` as the universal half of the third. `AscentStrict`'s
docstring now correctly says its left disjunct is the coinciding-endpoint case
and points to `NoCycle`.

### D8-F2: the log-det instance

`DLogTerminationConclusion` (`TerminationSpec.lean:112`) is the same four-clause
conjunction at `fun z => Real.log (fisher S z).det`, discharged by
`d_log_exchange_terminates` and guarded. It is correct: the generic theorems are
objective-agnostic, so `⟨ascent_strict _, no_cycle _, no_infinite_run _,
terminates _⟩` types at either objective.

It was added as a *separate* frozen Prop rather than conjoined into
`DTerminationConclusion`, which is the audit's second option, not its preferred
one. Consequence: the type marked for `D-EXCHANGE-TERMINATES` covers the
determinant run only. That is acceptable here because `ExchangeVoronoiSpec.lean`'s
objective convention already fixes `det` as the project's frozen D-stability
form, and `D-EXCHANGE-IMPLIES-VORONOI` is already marked on it — but the D8 node
carries no `assumptions` array saying so, and its statement says "D gains". See N5.

### D6-F3: exactly one theorem per frozen conclusion

| frozen Prop | file | exported theorem | other theorem of that type |
|---|---|---|---|
| `ClosureConclusion` | `ClosureSpec.lean:108` | `closure_reproduces_labels` (`Closure.lean:69`) | none |
| `ClosureDuplicateConclusion` | `ClosureSpec.lean:124` | `closure_duplicates` (`Closure.lean:77`) | none |
| `MergeConclusion` | `MergeSpec.lean:64` | `merge_invariance` (`Merge.lean:81`) | none — auxiliary, not claim-bearing |
| `DTerminationConclusion` | `TerminationSpec.lean:101` | `d_exchange_terminates` (`Termination.lean:122`) | none |
| `DLogTerminationConclusion` | `TerminationSpec.lean:112` | `d_log_exchange_terminates` (`Termination.lean:127`) | none — auxiliary, not claim-bearing |

`nearestCentroidRule_eq_label`, `nonempty_labels_of_posDef` and
`exists_nearestCentroidRule` are helpers with their own unbundled types and must
not be what any `declaration` names; `ascent_strict`, `no_cycle`,
`no_infinite_run`, `terminates` and `stableFor_det_iff` likewise.

`registry.py` checks that would apply:
- `declaration` non-empty and unique — the six existing marks are
  `exchange_voronoi`, `scalarExchangeStrengthenedLowerBound`,
  `leverage_inequality`, `det_relocation_gain`, `rank_two_relocation`,
  `violation_log_gain`. No collision with the three proposed names.
- `\btheorem\s+<local>\b` present in `formal_proof.file` — matches for
  `closure_reproduces_labels` and `closure_duplicates` in `Closure.lean`, and for
  `d_exchange_terminates` in `Termination.lean` (the regex does not spuriously
  match `d_log_exchange_terminates`).
- `spec`, `file`, `statement_audit` must resolve — all three do.
- status must not be in `{open, conjecture, measured, counterexample}` — all three
  nodes are `project_proved`.

### F6 / D8-F5: the one remaining import exception

Frozen import graph today:

```
ConfigSpec, ScalarExchangeSpec  → Mathlib only
DetGainSpec → ConfigSpec        RelocationSpec → ConfigSpec, ScalarExchangeSpec
LeverageSpec → DetGainSpec      MergeSpec → ConfigSpec
ClosureSpec → ExchangeVoronoiSpec, MergeSpec
TerminationSpec → ExchangeVoronoiSpec
ExchangeVoronoiSpec → ScoreQuantFormal.Leverage        ← the exception
```

`ClosureSpec.lean` was fixed (its `ExchangeVoronoi` import moved into
`Closure.lean`, which is where `exchangeStable_implies_strictVoronoi` is used).
`ExchangeVoronoiSpec.lean:1` still pulls in the editable `Leverage.lean`, and both
`ClosureSpec` and `TerminationSpec` inherit it, so the D6 and D8 frozen statements
still elaborate inside an environment a prover controls. Nothing leaks at the
definition level. The residue is disclosed in `formal/README.md`'s trust policy
and is owed a fix under `ExchangeVoronoiSpec`'s own audit; ADR 0037 states the
rule the tree does not yet satisfy. Accepted as documented, not silent.

### F10 / D8-F4: what the non-coverage lists still omit

`ClosureSpec.lean` added items 1–4 of the eight (the nonempty-cells hypothesis as
a bolded paragraph rather than a bullet, which is better; D5's second duplicate
branch; merged-vs-unmerged stability; off-training-set behaviour). Missing:

- item 5 — nothing is claimed about the *shape* of the induced decision regions
  (D1's affine-max reading), nor about measurability of `q`. `grep` for `affine`
  and `measur` in `ClosureSpec.lean` returns nothing.

`MergeSpec.lean`'s three items are all still open:

- item 6 — merging is proved to commute with the statistics of one fixed
  labeling, not with relocation or with stability. The bullet "Any determinant,
  ordering, stability or geometric consequence. Those are `ClosureSpec.lean`" is a
  pointer, not that disclaimer;
- item 7 — nothing is claimed for labelings of `S'` other than `z ∘ map`;
- item 8 — the correspondence row still maps "merged **distinct** positive-weight
  score atoms" onto `score_eq` and `weight_eq`, which supply neither distinctness
  nor positivity. The first non-coverage bullet now disclaims distinctness, so the
  row and the list still disagree with each other.
- the offered refinement in the other direction (under
  `ClosureDuplicateConclusion`'s hypotheses the merge is forced to be complete,
  because two equal-score `S'` rows in different atoms would break
  `VoronoiAssumptions`' injectivity) was not added; the "nothing says `map` merges
  *all* duplicates" bullet stands unqualified.
- the header sentence "`IsMerge` can, by naming the merge map instead of asserting
  injectivity" still stops short of saying plainly that no frozen statement does.
  `ClosureSpec.lean` does say it plainly, so the fact is recorded — just not here.

D8's list is complete as requested.

## New items from this round

**N1 — `D-CLOSURE-DUPLICATE-INHERITANCE`'s statement bundles two frozen Props
(settle before marking).** Its second sentence — "Merging preserves cell masses,
cell sums, centroids and retained information, which is what carries the
merged-side stability to a rule defined by unmerged data" — is `MergeConclusion`,
discharged by `merge_invariance`, and is *not* part of `closure_duplicates`'s
type. This is D6-F3's defect one level down, and ADR 0037 states the rule it
trips ("a node whose statement bundles several theorems is split before marking").
Cheapest fixes, either is fine: drop the second sentence from `statement` (it is
the proof mechanism, and `assumptions` already carries the two `IsMerge` fields),
or name `ScoreQuantFormal.merge_invariance` alongside the other two declarations
in §D6's **Machine-checked.** paragraph so the prose covers what the marked type
does not. Marking as it stands would attach a one-theorem declaration to a
two-theorem sentence.

**N2 — `D-COMPILE-TOLERANCE-GUARANTEE` is `project_proved` on no derivation.** The
node exists (which is what F2 asked for) and correctly carries no `formal_proof`,
and `ClosureSpec.lean` correctly says nothing here formalizes it. But its
`statement` is quantitatively specific — "reproduces every training label except
on rows whose relocation gain is at most that tolerance" — and §D6 gives no
argument for it, only a pointer to `GeometryReport`. It is in fact a near-verbatim
restatement of `PartitionResult.compile_quantizer`'s docstring contract
(`src/scorequant/result.py:309-317`). Under the registry's own vocabulary
(`project_proved` = "derived in the project") this wants either two lines of
derivation in §D6 or a status that matches what it is. Not blocking anything here,
because the node must never be marked.

**N3 — a frozen file cites a scope clause ADR 0037 has just superseded.**
`TerminationSpec.lean:17` says "ADR 0030 places profiled `D_s` outside the track".
ADR 0037 replaces that clause and explicitly admits "the finite profiled \(D_s\)
core" into scope. Nothing false follows — the docstring's *operative* reason for
refusing a DS mark is the right one and is unchanged (`F_s` is not a Lean object;
there is no frozen `DsTerminationConclusion`) — but the citation is now stale, and
it sits in a file a prover may not edit without a new statement audit. Fold the
correction into the next re-freeze rather than opening one for it.

**N4 — D6's `statement` asserts existence unconditionally; the Lean guards it.**
"a nearest-centroid rule … exists" has no dimension condition in the sentence,
while `ClosureConclusion`'s first conjunct is `0 < d → ∃ q …`. The guard *is*
recorded, in `assumptions` ("positive score dimension, for the existence half
only"), and at `d = K = N = 0` the unguarded reading is genuinely false — so the
sentence read alone over-claims by exactly the corner the guard exists for. F1's
fix put "exactly K nonempty cells" into the sentence; the same treatment here
("in positive score dimension, a nearest-centroid rule … exists") would close it.
Recommended, not blocking.

**N5 — two frozen Props have no registry counterpart and nothing says so.**
`MergeConclusion`/`merge_invariance` and
`DLogTerminationConclusion`/`d_log_exchange_terminates` are auxiliary: no claim
node states either, and neither may be named by a `declaration`. Audit 001 said
this of `merge_invariance` in its own text; the tree itself still does not. One
sentence in each spec docstring would make the intent survive this audit's
lifetime. Relatedly, `D-EXCHANGE-TERMINATES` has no `assumptions` array at all, so
the determinant-versus-`log det` convention behind its marked type lives only in
§D8's caution paragraph and in `ExchangeVoronoiSpec.lean`'s objective convention.

**N6 — cosmetic.**
- `Termination.lean:38` still labels `ascent_strict` "**Strict ascent, hence no
  cycles.**" — the exact misattribution D8-F1 flagged, surviving in the editable
  module while the frozen docstring was corrected.
- `MergeSpec.lean:51-52`'s structure docstring lists three properties ("every
  original row carries the score of its atom, every atom is inhabited, and an
  atom's weight is …") for a two-field structure. True as a consequence list,
  misleading as a field list.
- §D8's new paragraph writes `` `\(\det\)` `` — LaTeX inside a code span, which
  renders as literal backslashes.
- §D6 and §D8's **Machine-checked.** paragraphs cite the `-001` audits, which
  audited statement text that has since changed. See the `statement_audit`
  recommendation below.

## Decision: what may now carry `formal_proof`

ADR 0037 is the scope authority and admits all of this: the finite theory, "the
whole finite D chain including D6, the compiled predictor". Three nodes qualify.

**1. `D-FINITE-INDUCTIVE-CLOSURE`** — may be marked.

```json
"formal_proof": {
  "system": "Lean 4.33.1 + Mathlib 4.33.1",
  "spec": "formal/ScoreQuantFormal/ClosureSpec.lean",
  "file": "formal/ScoreQuantFormal/Closure.lean",
  "declaration": "ScoreQuantFormal.closure_reproduces_labels",
  "statement_audit": "AUDITS/FORMALIZATION-D-FINITE-INDUCTIVE-CLOSURE-002.md"
}
```

Type is `ClosureConclusion S z` at full generality in `d N K`, which is the frozen
conclusion and nothing else. Consider N4 first — it is a one-line edit to the
`statement`, not a re-freeze.

**2. `D-CLOSURE-DUPLICATE-INHERITANCE`** — may be marked **after N1 is settled.**

```json
"declaration": "ScoreQuantFormal.closure_duplicates",
"spec": "formal/ScoreQuantFormal/ClosureSpec.lean",
"file": "formal/ScoreQuantFormal/Closure.lean"
```

Type is `ClosureDuplicateConclusion S' S map z`. `MergeSpec.lean` is the
supporting freeze but `spec` takes one path, and `ClosureDuplicateConclusion`
lives in `ClosureSpec.lean`, so that is the right value. The node's second
sentence is `MergeConclusion`, not this type — trim it or record `merge_invariance`
in §D6 prose before attaching the field.

**3. `D-EXCHANGE-TERMINATES`** — may be marked.

```json
"declaration": "ScoreQuantFormal.d_exchange_terminates",
"spec": "formal/ScoreQuantFormal/TerminationSpec.lean",
"file": "formal/ScoreQuantFormal/Termination.lean"
```

Type is `DTerminationConclusion S K`, whose four conjuncts cover all three
registry clauses. The marked run is the determinant run; §D8 and
`TerminationSpec.lean` both now state that the `log det` run is a different run.

**On `statement_audit`.** The `-001` audits no longer audit the current text:
`ClosureSpec.lean` lost `NearestCentroidRuleExists` and gained the guarded
conjunct, `MergeSpec.lean` lost a field, and `TerminationSpec.lean` gained
`NoCycle` and `DLogTerminationConclusion`. This file audits the current text of
both boundaries, so it is the honest value for all three nodes. If a per-claim
filename is preferred, copy the D8 sections above into
`AUDITS/FORMALIZATION-D-EXCHANGE-TERMINATES-002.md` and point that node there.

### Must not be marked

- **`D-COMPILE-TOLERANCE-GUARANTEE`** — nothing in the tree formalizes a positive
  gain tolerance. `ClosureSpec.lean` and `formal/README.md` both say so correctly.
- **`DS-EXCHANGE-TERMINATES`, `A-EXCHANGE-TERMINATES`** — ADR 0037 no longer puts
  them out of *scope*, but the freeze rule still refuses them: neither `F_s` nor
  `F_A` exists as a Lean object, and no `DsTerminationConclusion` /
  `ATerminationConclusion` is frozen. Marking either would need those definitions
  plus a fresh statement audit. Their coverage is prose-only, currently recorded
  in §D8; §DS3 and §A1 themselves still say nothing, which is where a reader of
  those claims would look.
- **Any node on `merge_invariance`, `d_log_exchange_terminates`,
  `nearestCentroidRule_eq_label`, `exists_nearestCentroidRule`,
  `nonempty_labels_of_posDef`, `terminates`, `ascent_strict`, `no_cycle`,
  `no_infinite_run` or `stableFor_det_iff`.** The first two are auxiliary frozen
  Props with no registry counterpart; the rest are helpers or generic lemmas whose
  types are not any node's statement.

## Not checked by this audit

`lake build --wfail` was not run, by instruction. Whether `Closure.lean`,
`Merge.lean` and `Termination.lean` compile, and whether the `#guard_msgs` axiom
messages match, remains for the trust gate — including the three new proofs
(`nonempty_labels_of_posDef`, `exists_nearestCentroidRule`, `no_cycle`) and the
two new `AxiomAudit.lean` entries for `d_log_exchange_terminates` and
`Reaches.head`. The mathematics of each was verified by hand above; only
elaboration is unverified. `WORK/active/FORMAL-D-CLOSURE.md` was not read. No file
was edited by this pass.
