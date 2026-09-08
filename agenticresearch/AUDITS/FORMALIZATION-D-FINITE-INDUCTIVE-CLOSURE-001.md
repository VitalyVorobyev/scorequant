# Formal statement audit — D-FINITE-INDUCTIVE-CLOSURE (001)

**Audited object:** `formal/ScoreQuantFormal/MergeSpec.lean` and
`formal/ScoreQuantFormal/ClosureSpec.lean`, the frozen statement boundary for
`D-FINITE-INDUCTIVE-CLOSURE` (§D6, manuscript v9 Theorem 3's deployable half),
together with the definitional layer they are written in (`ConfigSpec.lean`,
`RelocationSpec.lean`, `ExchangeVoronoiSpec.lean`).

**Canonical source:** `claims/D-FINITE-INDUCTIVE-CLOSURE.json`,
`KNOWN_RESULTS/04-d-optimality.md` §D6, `claims/D-EXCHANGE-IMPLIES-VORONOI.json`
(the dependency), `claims/D-UNMERGED-DUPLICATES-FAIL.json` and
`CE-D-UNMERGED-DUPLICATES-001` (the boundary witness),
`AUDITS/FORMALIZATION-D-EXCHANGE-IMPLIES-VORONOI-001.md` (the prior audit).

**Date:** 7 September 2026. **Verdict: match after hardening.**
Nothing frozen is false, and nothing frozen is vacuous. The hardening is on the
*registry* side and on the two docstrings; four items (F1, F2, F3, F8) are
BLOCKING for protocol step E — that is, `formal_proof` must not be attached
until they are fixed. None of them blocks step D: the frozen propositions are
true as stated and the proofs may proceed.

**Independence.** This audit ran in a session with no shared derivation
context. It was given the claim nodes, the §D6 prose, the prior D5
formalization audit, the boundary counterexamples and the Lean files. It did
not read the formalizing session's transcript or its work note
(`WORK/active/FORMAL-D-CLOSURE.md` was deliberately left unopened), and it did
not run `lake`. Every necessity claim below is backed by an explicit
configuration checked in exact rational arithmetic against a direct
reimplementation of the frozen definitions, including Lean's junk conventions
(`(0:ℝ)⁻¹ = 0` for an empty cell's centroid; `Matrix.inv A = 0` when
`A.det = 0`).

## Correspondence

Registry statement, clause by clause. `H` rows are hypotheses, `C` rows are
conclusions.

| # | Registry clause | Lean | Status |
|---|---|---|---|
| H1 | "exact zero-tolerance one-point-exchange-stable" | `ExchangeStable` (frozen in `ExchangeVoronoiSpec.lean`, on `det`) | frozen; inherited from D5 |
| H2 | "positive-definite" | `VoronoiAssumptions`, third conjunct | frozen |
| H3 | "merged … score atoms" | `IsMerge` (`score_eq`, `weight_eq`) for the duplicate half; nothing needed for the merged half | frozen |
| H4 | "distinct" | `VoronoiAssumptions`, first conjunct (`Function.Injective S.score`) | frozen |
| H5 | "positive-weight" | `Sample.weight_pos` | carried by the type; a zero-weight row is inexpressible |
| H6 | — **no registry counterpart** | `VoronoiAssumptions`, **second conjunct** (`∀ c, (cell z c).Nonempty`) | frozen, load-bearing, **undocumented** — see F1 |
| H7 | — no registry counterpart | `Nonempty (Fin K)` in `NearestCentroidRuleExists` | necessary for truth; see F5 |
| C1 | "compiles to `q̂(s) = argmin_b (s−μ_b)ᵀ Î⁻¹ (s−μ_b)`" | `IsNearestCentroidRule` (the shape) + `NearestCentroidRuleExists` (that there is one) | frozen |
| C2 | "reproducing all merged-atom training labels" | `ClosureConclusion` | frozen |
| C3 | "strictly", i.e. without a tie breaker | quantification over **every** `q` satisfying `IsNearestCentroidRule` | frozen; equivalent to "the argmin at each training score is a singleton", given C1's existence half |
| C4 | "original duplicate rows inherit the merged label" | `ClosureDuplicateConclusion`, whose rule is built from the unmerged sample; `MergeConclusion` is the machinery | frozen |
| C5 | "Positive solver tolerance gives only a tolerance-stamped boundary-disagreement guarantee." | **nothing** | not covered; the claimed split node does not exist — see F2 |

Two frozen objects have no registry counterpart at all and are auxiliary rather
than claim-bearing: `MergeConclusion` (a supporting invariance, not a claim —
`merge_invariance` must never be marked on a claim node) and `IsMerge.surjective`
(redundant — see F4).

## Findings and dispositions

**F1 — a load-bearing hypothesis is missing from the registry statement and
from the correspondence table (BLOCKING for step E).**
`VoronoiAssumptions`' second conjunct, `∀ c, (cell z c).Nonempty`, appears in
the hypotheses of `ClosureConclusion` and `ClosureDuplicateConclusion`. The D6
registry `statement` does not mention it, the §D6 prose does not mention it, and
`ClosureSpec.lean`'s correspondence table skips it while *explicitly naming the
first and third conjuncts of the same definition* — which reads as an exhaustive
decomposition and is not one.

Removing it makes the frozen statement **false**, not vacuous:

> `d = 1`, `N = 2`, `K = 3`, scores `(−2, 0)`, unit weights, `z = (0, 1)`.
> Cell `2` is empty, so `centroid = (0)⁻¹ • 0 = 0`. `fisher = [4]`, positive
> definite; the scores are injective; both occupied cells are singletons so no
> relocation is `Admissible` and `ExchangeStable` holds vacuously. Row `1` has
> score `0`, is at Mahalanobis distance `0` from its own centroid `μ₁ = 0` *and*
> from the empty cell's junk centroid `μ₂ = 0`. The argmin set is `{1, 2}`, so a
> rule with `q 0 = 2 ≠ z 1` satisfies `IsNearestCentroidRule` and the conclusion
> fails.

The hypothesis is also needed informally — `argmin_b` over `b = 1..K` presumes
every `μ_b` exists — and D5, which D6 depends on, states it. So the fix is on
the registry side, never by deleting the conjunct: patch
`claims/D-FINITE-INDUCTIVE-CLOSURE.json`'s `statement` (and the §D6 prose) to
carry "exactly `K` nonempty cells" as D5's node does, and add the missing row to
the `ClosureSpec.lean` table. Do not weaken the Lean.

**F2 — the docstring asserts a claim split that has not happened (BLOCKING for
step E).** `ClosureSpec.lean` disclaims the registry statement's third sentence
with "That is a different theorem with a weaker hypothesis, split into its own
claim node per protocol step A." No such node exists: `claims/` contains no
tolerance-stamped boundary-disagreement claim, and grepping for
`tolerance-stamped` / `boundary-disagreement` across `claims/` returns only
`D-FINITE-INDUCTIVE-CLOSURE.json` itself. As things stand the node bundles a
formalized part and an unformalized part, which protocol step A forbids and step
E turns into "record in `KNOWN_RESULTS/` prose instead". Either create the split
node before marking, or drop the "split into its own claim node" wording and
record the Lean result in prose.

**F3 — one `formal_proof.declaration` cannot name three theorems (BLOCKING for
step E).** `py/registry.py` requires `formal_proof.declaration` to be a single
non-empty name, unique across the registry, and `formal/README.md` requires its
*type* to be the frozen conclusion. D6 as frozen has three exported theorems
with three different frozen types — `closure_reproduces_labels :
ClosureConclusion`, `closure_duplicates : ClosureDuplicateConclusion`, and
`nearestCentroidRule_exists : NearestCentroidRuleExists` — so no single name
covers the node. Resolve by splitting the node (a merged-atoms half and a
duplicate-rows half, as D5/D-EXCHANGE-VIOLATION-LOWER-BOUND are split), or by
freezing one conjoined Prop and exporting one theorem of that type. Marking one
of the three and calling the node covered would be exactly the drift the README
rule exists to prevent.

**F4 — `IsMerge.surjective` is redundant, and its docstring says otherwise
(hardening).** `weight_eq` says `S.weight i = cellMass S' map i`. If the fibre
over `i` were empty the sum would be `0`, contradicting `Sample.weight_pos i`.
So `surjective` is derivable from the other field plus the type, and dropping it
changes neither the truth nor the vacuity of `MergeConclusion` or
`ClosureDuplicateConclusion` (a search over 10,440 admissible `(S, S', map)`
triples in `d = 1` found no configuration that `surjective` alone excludes).
`MergeSpec.lean` says "`surjective` **additionally** forbids empty fibers
outright", which is not so. A redundant field is harmless — it cannot weaken a
statement it does not constrain — but a frozen structure should not carry a
field the audit cannot justify. Either delete it (preferred: it makes `IsMerge`
minimal) or change the docstring to say it is derivable and kept for
readability.

**F5 — the non-vacuity chain has an unstated link (hardening).**
`NearestCentroidRuleExists` is conditioned on `Nonempty (Fin K)`, and that
hypothesis is necessary: at `K = 0` there is no function from the inhabited type
`Fin d → ℝ` into `Fin 0`, so dropping it makes the Prop *false*, not merely
vacuous. It is also strong enough — but only via a step that appears nowhere in
the frozen files or the proof modules: under `VoronoiAssumptions` with `d ≥ 1`,
`K = 0` would make `fisher` the zero matrix, which is not `PosDef` for `d ≥ 1`;
hence `K ≥ 1` and a rule exists. Without that observation a reader cannot tell
that `ClosureConclusion`'s `∀ q` is inhabited under its own hypotheses. Add the
argument to the `ClosureSpec.lean` docstring, or freeze the sharper
`VoronoiAssumptions S z → 0 < d → ∃ q, IsNearestCentroidRule S z q`.
For completeness, the `d = 0` corner is genuinely empty rather than false:
injectivity forces `N ≤ 1`, nonempty cells force `N ≥ K`, so either `K = N = 1`
(conclusion trivially true) or `K = N = 0` (both quantifiers empty).

**F6 — a frozen file imports an editable one (hardening).**
`ClosureSpec.lean` opens with `import ScoreQuantFormal.ExchangeVoronoi`, the
*proof* module, although everything it uses (`VoronoiAssumptions`,
`ExchangeStable`) lives in `ExchangeVoronoiSpec.lean`. Definitional closure is
intact (see F7), but the elaboration environment of a frozen file then includes
declarations a prover may add: an `instance`, `scoped notation` or `macro_rules`
introduced in `ExchangeVoronoi.lean` can change how a frozen statement
elaborates without touching an audited file — the same failure mode the
`ConfigSpec.lean` split exists to close, one level up. Change the import to
`ScoreQuantFormal.ExchangeVoronoiSpec` and move the `ExchangeVoronoi` import
into `Closure.lean`, which is where `exchangeStable_implies_strictVoronoi` is
actually used. Note the same pattern pre-exists in `ExchangeVoronoiSpec.lean`
(`import ScoreQuantFormal.Leverage`); it was not caught by audit 001 and should
be fixed with it. Every other `*Spec.lean` imports only `*Spec.lean` files.

**F7 — definitional closure holds; no leak (confirmed).** Every definition
reachable from the three frozen conclusions lives in a frozen file:
`Sample`, `cell`, `cellMass`, `cellSum`, `cellBlock`, `centroid`, `fisher`,
`mahalanobis` (`ConfigSpec.lean`); `relocate` (`RelocationSpec.lean`, reached
through `ExchangeStable`); `Admissible`, `ExchangeStable`, `VoronoiAssumptions`
(`ExchangeVoronoiSpec.lean`); `IsMerge`, `MergeConclusion` (`MergeSpec.lean`);
`IsNearestCentroidRule`, `ClosureConclusion`, `NearestCentroidRuleExists`,
`ClosureDuplicateConclusion` (`ClosureSpec.lean`). `Merge.lean` and
`Closure.lean` introduce **no** definitions at all, only theorems — checked by
grep. (Incidental, outside this audit's object: `Corollaries.lean` defines
`GlobalOptimum` and `Leverage.lean` defines `rootFactor` in editable modules.
`rootFactor` is proof-internal and harmless; `GlobalOptimum` appears in D7's
conclusion and would be a leak if D7 were ever marked.)

**F8 — ADR 0030 does not cover D6 (BLOCKING for step E).** The ADR scopes the
Lean track to "the finite D chain in general dimension (D2 relocation identity,
D3 determinant gain, D4 leverage inequalities, D5 with its quantitative bound,
the D7/D8 corollaries) **and nothing beyond it until a further decision**". D6 is
not in that enumeration, and `formal/README.md` still lists "the compiled
predictor `D-FINITE-INDUCTIVE-CLOSURE`" under *Not covered*. The protocol makes
ADR 0030 the scope authority, so D6 needs an ADR amendment (and a README update)
before it can carry `formal_proof` — a prover's discretion is explicitly not
enough. The mathematics is squarely in the finite D chain, so this should be a
short amendment, not a re-scoping debate.

**F9 — the exported theorems' types are the frozen conclusions (confirmed).**
`merge_invariance : MergeConclusion S' S map z`,
`closure_reproduces_labels : ClosureConclusion S z`,
`nearestCentroidRule_exists : NearestCentroidRuleExists S z`,
`closure_duplicates : ClosureDuplicateConclusion S' S map z` — each is stated at
full generality in `d N N' K` with the frozen Prop as its type, not a restatement.
`nearestCentroidRule_eq_label` *is* an unbundled restatement, but it is a helper,
not a candidate for the mark; it must not be what `declaration` names. All four
exported theorems plus the four merge lemmas are guarded in `AxiomAudit.lean`,
which the README's "a module nothing guards is a module CI does not build" rule
requires.

**F10 — the non-coverage lists are accurate but incomplete (hardening).**
Everything the two lists *assert* is correct — including the aside that
`predict_scores` breaks ties toward the lowest cell index, which
`src/scorequant/result.py:318-320` confirms — with the two exceptions already
recorded as F2 (the non-existent split node) and F4 (`surjective`
"additionally"). What is missing:

*`ClosureSpec.lean` should also disclaim:*
1. the "exactly `K` nonempty cells" hypothesis (F1), which is silently *assumed*
   rather than silently omitted, and is the one thing a reader of the registry
   statement would not expect;
2. that nothing here formalizes D5's *second* duplicate branch — "labels
   constant on every duplicate class". `ClosureDuplicateConclusion` **assumes**
   the labeling is `z ∘ map`, which is constant on fibres by construction; it
   does not derive constancy. `ExchangeVoronoiSpec.lean`'s disclaimer of that
   branch therefore still stands, and `MergeSpec.lean`'s remark that `IsMerge`
   "can" express it should say plainly that no frozen statement does;
3. that `ClosureDuplicateConclusion` says nothing about the *unmerged*
   configuration — not that `z ∘ map` is exchange-stable for `S'`, not that
   `S'` is D-optimal, not that the unmerged and merged stability notions agree
   (they do not: a single-row move on `S'` can split a duplicate class, which no
   move on `S` can express);
4. that no uniqueness is claimed for the compiled rule off the training set —
   two `IsNearestCentroidRule` rules may disagree at any non-training `s`;
5. that nothing is claimed about the *shape* of the induced decision regions
   (D1's "affine-max partition" reading), nor about measurability of `q`.

*`MergeSpec.lean` should also disclaim:*
6. that merging is proved to commute only with the *statistics* of one fixed
   labeling, not with relocation or with stability (the reason `ClosureSpec.lean`
   must place `ExchangeStable` on the merged side);
7. that nothing is claimed for labelings of `S'` other than `z ∘ map`;
8. its correspondence row maps "merged **distinct** positive-weight score atoms"
   onto `score_eq` and `weight_eq`, which supply neither distinctness (disclaimed
   two paragraphs later) nor positivity (`Sample.weight_pos`). The row overclaims
   against its own non-coverage list.

One genuine refinement in the other direction, worth adding rather than
disclaiming: `MergeSpec.lean` says "nothing says `map` merges *all* duplicates".
True of `IsMerge` alone, but under `ClosureDuplicateConclusion`'s hypotheses the
merge is forced to be complete — two equal-score rows of `S'` landing in
different atoms would give `S.score i = S.score i'` for `i ≠ i'`, contradicting
`VoronoiAssumptions`' injectivity.

**F11 — an available free strengthening (observation, no action required).**
`IsNearestCentroidRule` requires `q` to be an argmin at **every** `s : Fin d → ℝ`.
That is the stronger requirement on `q`, hence the *smaller* set quantified over
in `ClosureConclusion`, hence the weaker theorem — the opposite direction from
the `≤`-versus-`<` choice the docstring correctly defends. `Closure.lean`'s proof
uses only the instance `hq (S.score i) (z i)`, so restricting the premise to
training scores would strengthen both conclusions at zero proof cost. This is not
a defect: quantifying over globally-defined rules is the honest reading of
"compiles to `q̂(s) = argmin_b …`" as a deployable predictor, and combined with
`NearestCentroidRuleExists` the two forms are equivalent in content. Recorded so
the choice is visible rather than accidental.

## Independently confirmed correct

- **The frozen statement is true.** Exhaustive exact-rational sweeps over every
  configuration meeting all hypotheses found no failure: 139,984 configurations
  in `d = 1` (`N ≤ 4`, `K ≤ 3`, scores in `{−2,…,2}`, weights in `{1/2, 1, 2}`)
  and 224,144 in `d = 2` (`N ≤ 3`, `K ≤ 3`), plus 10,440 `(S, S', map)` triples
  for `ClosureDuplicateConclusion`. The hypotheses are satisfiable
  non-vacuously — e.g. `d = 2`, scores `(1,0)` and `(0,1)`, unit weights,
  `z = (0,1)`: `fisher = I₂`, injective, both cells occupied, no admissible move,
  and the conclusion is a true non-trivial statement.
- **Every hypothesis of `ClosureConclusion` prevents falsity, not merely
  vacuity.** Each was dropped in turn and refuted by an explicit configuration:
  - *injectivity* — `d=1`, `N=2`, `K=2`, scores `(−2, −2)`, unit weights,
    `z = (0,1)`: `fisher = [8]`, cells occupied, no admissible move, and no
    *function* `q` can return both `0` and `1` at score `−2`. The
    already-machine-checked `CE-D-UNMERGED-DUPLICATES-001` (scores `(1,1,−1)`,
    weights `(¼,¼,½)`, `K=3`, `z = id`) refutes it identically, so D5's boundary
    witness transfers to D6 verbatim;
  - *nonempty cells* — F1 above;
  - *positive definiteness* — `d=2`, `N=2`, `K=2`, scores `(1,1)` and `(2,2)`,
    unit weights, `z = (0,1)`: `fisher = [[5,5],[5,5]]` has `det = 0`, so Lean's
    `Matrix.inv` returns `0`, every Mahalanobis value is `0`, every cell ties,
    and a rule returning `1` at `(1,1)` exists. Both cells are singletons so
    stability is vacuous and the scores are injective. Worth recording that this
    hypothesis is **not** independently load-bearing at `d = 1`: a 4,762-configuration
    sweep found no `d = 1` counterexample, because there `det fisher ≥ 0` and no
    singular, injective, all-cells-occupied, stable configuration exists. The
    hypothesis earns its place only in `d ≥ 2`;
  - *exchange stability* — `d=1`, `N=3`, `K=2`, scores `(−2,−1,0)`, unit weights,
    `z = (0,1,0)`: `fisher = [3]` is positive definite, scores injective, cells
    occupied, but `μ₀ = μ₁ = −1`, so every row ties and every label is
    reproducible only by luck. Moving row `0` to cell `1` raises `det` from `3`
    to `9/2`, so `ExchangeStable` is exactly what fails;
  - *`IsNearestCentroidRule q`* — trivially, a constant `q`.
- **`ClosureDuplicateConclusion` is strictly stronger than a restatement of
  `IsMerge.score_eq`.** The rule is quantified over
  `IsNearestCentroidRule S' (z ∘ map) q` — built from the unmerged sample's own
  `fisher` and centroids — and `Closure.lean` reaches the merged side only by
  rewriting with `fisher_merge` and `centroid_merge`. Had the rule been stated on
  the merged side, `score_eq` plus part (a) would give the clause immediately and
  it would assert nothing. The docstring's account of this is exactly right.
- **The merged-side hypotheses are needed.** `IsMerge S S id` holds
  (`weight_eq` is `cellMass S id i = S.weight i`), so
  `ClosureDuplicateConclusion S S id z` specializes to `ClosureConclusion S z`
  and all four counterexamples above lift unchanged. `VoronoiAssumptions` cannot
  be moved to the unmerged side at all — a genuine merge makes `S'.score`
  non-injective by `score_eq`, as the docstring says.
- **`IsMerge`'s remaining two fields are each load-bearing for the duplicate
  half.**
  - *`score_eq`* — `S` = scores `(−2,−1)`, unit weights, `z = (0,1)` (stable,
    positive definite, injective, cells occupied); `S'` = scores `(−2,−2)`, unit
    weights, `map = id`. `surjective` and `weight_eq` hold, `score_eq` fails, and
    `z ∘ map` puts two coincident `S'`-scores in different cells, so no `q` works.
  - *`weight_eq`* — dropping it, with `score_eq` and `surjective` kept, is exactly
    "re-weight the merged sample by arbitrary fibre masses", and stability of one
    weighting says nothing about the geometry of another. Take `S` = scores
    `(−2,−1,0,2)`, weights `(2,1,1,1)`, `z = (0,0,1,1)` — positive definite,
    exchange-stable, injective, both cells occupied, conclusion true. Let `S'`
    have six unit-weight rows with fibre masses `(1,1,1,3)`. Then
    `centroid S' (z∘map) = (−3/2, 3/2)` and the row with score `0` and label `1`
    is at distance `1/6` from *both* centroids, so a nearest-centroid rule for
    `S'` may return `0`. The conclusion fails.
- **`IsMerge` neither forces nor forbids anything the claim does not intend,**
  with the one exception in F4. It correctly does **not** require the merged
  scores to be distinct: that would be a *hypothesis* on `MergeConclusion` and
  `ClosureDuplicateConclusion`, weakening both, and it is unnecessary because
  merge invariance is true without it and because
  `ClosureDuplicateConclusion` already carries distinctness through
  `VoronoiAssumptions S z`. Omitting it is the right call, not a defect. It also
  correctly permits `map = id` (so the duplicate half generalizes the merged
  half), permits unequal weights within a fibre, and — through `score_eq` —
  forbids merging rows whose scores differ.
- **`ClosureConclusion` is not satisfiable trivially.** Two triviality routes
  were checked and both are closed: the hypotheses are satisfiable (witness
  above), and the `∀ q` is inhabited whenever `Nonempty (Fin K)`, which
  `VoronoiAssumptions` forces for `d ≥ 1` (F5). `IsNearestCentroidRule` is
  satisfiable for any finite nonempty `Fin K` because a minimum over a finite
  nonempty type always exists — which is precisely what
  `nearestCentroidRule_exists` proves, without needing `PosDef`.
- **`MergeConclusion` is sound and its hypotheses are minimal up to F4.**
  Dropping `score_eq` breaks `cellSum`, `centroid` and `fisher` (`d=1`, one row,
  atom score `1` against row score `2`); dropping `weight_eq` breaks `cellMass`
  (same shape, weight `1` against `2`); dropping `surjective` breaks nothing.
- **The junk conventions are safe in both directions.** An empty cell's
  `centroid` is `0` and its `cellBlock` is `0`, which matches "sum over occupied
  cells"; a singular `fisher` gives `Matrix.inv = 0`, which makes
  `IsNearestCentroidRule` *satisfiable by everything* and therefore makes
  `ClosureConclusion` false rather than vacuously true — the safe direction, and
  the reason the `PosDef` counterexample above is a real refutation.

## What remains uncovered

The frozen statement concerns one fixed finite labeling of finitely many
strictly-positive-weight rows in exact real arithmetic, with distinct merged
scores, every cell occupied, retained information already known to be
nonsingular, and stability at exactly zero gain tolerance. It says nothing about
the regime every real solver is in — the registry statement's own third sentence,
the tolerance-stamped boundary-disagreement guarantee, is not formalized and, as
of this audit, not even split out as a claim. It says nothing about capacity,
balance or minimum-mass constraints, singular or pseudodeterminant objectives,
projected-subspace variants, population or atomless limits, score-estimation
error, or the observation-to-score step. It does not claim the compiled rule is
optimal, unique off the training set, or in agreement with `z` at any fresh `s`;
the conclusion ranges over the training scores only. It does not formalize D5's
alternative duplicate branch. And it is a statement about mathematics: nothing
connects `fisher`, `centroid` or `IsNearestCentroidRule` to `compile_quantizer`,
`predict_scores`, the `Quantizer` artifact, `rank_rtol`, or floating-point
conditioning.
