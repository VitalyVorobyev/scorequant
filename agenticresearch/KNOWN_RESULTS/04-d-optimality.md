# 4. Full D-optimality

> Part of the ScoreQuant known-results ledger. Read `PROBLEM.md` first and
> `KNOWN_RESULTS/index.md` for the status vocabulary and the chapter map.
> Resolve any claim id with `python py/registry.py show <ID> --deps --proof`.

\[
F_D(I)=\log\det I,
\qquad G_D=I^{-1}.
\]

## D1. Population stationary geometry — [PROJECT-PROVED/BRIDGE]

**Claims:** D-POP-VORONOI

A regular atomless stationary D quantizer satisfies

\[
\boxed{
q(s)\in\arg\min_b
(s-\mu_b)^\top I_q^{-1}(s-\mu_b)
\quad\text{a.e.}
}
\]

Thus cells form a self-consistent common-metric Mahalanobis Voronoi / affine-max partition.

This is a first-order stationarity result, not a global-optimality theorem.

## D2. Exact weighted rank-two relocation — [PROJECT-PROVED]

**Claims:** D-RANK2-MOVE

Move weighted point \((s,w)\) from a non-singleton source \(a\) to destination \(b\). Let

\[
u_a=s-\mu_a,\quad u_b=s-\mu_b,
\]

\[
\alpha=\frac{wW_a}{W_a-w},\qquad
\beta=\frac{wW_b}{W_b+w}.
\]

Then

\[
\boxed{
\Delta I
=\alpha u_au_a^\top-
\beta u_bu_b^\top.
}
\]

**Machine-checked.** Verified for arbitrary \(d\) in Lean 4.33.1 + Mathlib. The
statement is frozen in `formal/ScoreQuantFormal/RelocationSpec.lean` and the
proof is `ScoreQuantFormal.rank_two_relocation` in
`formal/ScoreQuantFormal/Relocation.lean`, whose type *is* the frozen
`RelocationConclusion`; the underlying identity is `fisher_relocate_sub`.
Independent statement audit:
`AUDITS/FORMALIZATION-D-RANK2-MOVE-001.md`, verdict `match after hardening`.
That audit narrowed the claim node: the formalization requires the destination
to be a distinct, nonempty cell, and the node's `assumptions` now record it.

## D3. Exact log-det relocation gain — [PROJECT-PROVED]

**Claims:** D-LOGDET-GAIN

With \(H=I^{-1}\) and

\[
q_{aa}=u_a^\top Hu_a,\quad
q_{bb}=u_b^\top Hu_b,\quad
q_{ab}=u_a^\top Hu_b,
\]

\[
\boxed{
\Delta F_D
=\log\!
[(1+\alpha q_{aa})(1-\beta q_{bb})+\alpha\beta q_{ab}^2].
}
\]

**Machine-checked.** Verified for arbitrary \(d\) in Lean 4.33.1 + Mathlib,
through the Weinstein-Aronszajn identity. The statement is frozen in
`formal/ScoreQuantFormal/DetGainSpec.lean` and the proof is
`ScoreQuantFormal.det_relocation_gain` in `formal/ScoreQuantFormal/DetGain.lean`;
the underlying determinant identity is `det_add_rank_two`, and
`detRatio_eq_one_add_exchangeExcess` joins it to the frozen scalar core — that
join lives in the editable proof module and is *not* itself audited. Independent
statement audit: `AUDITS/FORMALIZATION-D-LOGDET-GAIN-002.md`, verdict `match
after hardening`, superseding the first round.

The frozen statement has two components, because this claim is named for a
log-determinant gain: the determinant identity, which needs only that \(I\) is
symmetric and nonsingular and nothing at all of the candidate, and the boxed
\(\Delta F_D\) form above, which carries the positivity that taking a
logarithm needs. The first round found the identity frozen alone, which left the
claim covered only in part; the node's `assumptions` field, which had asked for
positive definiteness the identity never uses, was corrected to match.

This supports exact \(O(d^2)\)-type candidate evaluation with cached factorizations.

## D4. Leverage inequality — [STANDARD/BRIDGE]

**Claims:** D-LEVERAGE

For every cell centroid,

\[
\mu_c^\top I^{-1}\mu_c\le 1/W_c,
\]

and for centroid difference \(\delta=\mu_a-\mu_b\),

\[
\boxed{
\delta^\top I^{-1}\delta
\le
1/W_a+1/W_b.
}
\]

This is a standard projection/leverage inequality; its role here is to bridge infinitesimal D geometry to exact finite gains.

Explicitly, with

\[
A=[\sqrt{W_1}\mu_1,\ldots,\sqrt{W_K}\mu_K],
\qquad I=AA^\top,
\]

the matrix \(P=A^\top(AA^\top)^{-1}A\) is an orthogonal projector. Taking
\(v_a=1/\sqrt{W_a}\), \(v_b=-1/\sqrt{W_b}\), and all other coordinates zero
gives \(Av=\mu_a-\mu_b\), hence

\[
(\mu_a-\mu_b)^\top I^{-1}(\mu_a-\mu_b)
=v^\top Pv\le v^\top v=1/W_a+1/W_b.
\]

**Machine-checked.** Both halves are verified for arbitrary \(d\) in Lean
4.33.1 + Mathlib, via the same projector argument. The bundled statement is
frozen in `formal/ScoreQuantFormal/LeverageSpec.lean` and the proof is
`ScoreQuantFormal.leverage_inequality` in `formal/ScoreQuantFormal/Leverage.lean`,
assembling `centroid_leverage_bound` and `leverage_bound`. Independent statement
audit: `AUDITS/FORMALIZATION-D-LEVERAGE-001.md`, verdict `match after
hardening`. Two things that audit established are worth keeping in view. The
frozen inequality is true with *no* hypotheses, because Lean's \(0^{-1}=0\)
collapses both sides together for an empty cell and a singular \(I\) — the
hypotheses are fidelity conventions, not guards against falsity, and the proof
uses positive definiteness genuinely. And nothing formal connects \(I\) to a
statistical Fisher information: `fisher` is *defined* as
\(\sum_c m_cm_c^\top/W_c\), and the identification is inherited from
`FI-QUANT-IDENTITY`, not proved.

## D5. Exchange stability implies strict D-Voronoi geometry — [PROJECT-PROVED; audited]

**Claims:** D-EXCHANGE-IMPLIES-VORONOI, D-EXCHANGE-VIOLATION-LOWER-BOUND, D-EXCHANGE-SCALAR-CORE

Let coincident score rows be merged into distinct atoms with positive weights,
and partition those atoms into exactly \(K\) nonempty cells. Assume \(I\succ0\),
that the only relocation constraint is preservation of nonempty cells, and that
exchange stability means no **exact positive-gain** relocation (zero gain
tolerance). Then a point in a non-singleton source that is tied with or farther
from its own centroid than a competing centroid under \(I^{-1}\) has

\[
\boxed{
\Delta F_D
\ge
\log\left(1+\frac{\alpha\beta}{4}q_\delta^2\right)>0
}
\]

when the two centroids are distinct. Here
\(q_\delta=(\mu_a-\mu_b)^\top I^{-1}(\mu_a-\mu_b)\).

The exact algebra is

\[
\frac{\det(I+\Delta I)}{\det I}=1+E,
\qquad
E\ge\frac{\alpha\beta}{4}
\left[q_\delta^2+(q_{aa}-q_{bb})^2\right].
\]

Distinct centroids follow from stability rather than needing a separate
assumption. If \(\mu_a=\mu_b\) and either cell is non-singleton, moving a
non-centroid atom between them gives determinant ratio
\(1+(\alpha-\beta)q_{aa}>1\). If both are singletons, equality would mean the
two score atoms are duplicates, excluded by merging. A singleton atom is then
strictly nearest to its own centroid because its own distance is zero and every
other centroid is distinct.

Hence

\[
\boxed{
\text{one-point exchange stable}
\Rightarrow
\text{strict self-consistent D-Voronoi}.
}
\]

The converse fails.

Exact ties between distinct centroids are therefore ruled out, not left as an
unresolved degeneracy. Split duplicate atoms are a genuine boundary failure:
see `COUNTEREXAMPLES/CE-D-UNMERGED-DUPLICATES-001.json`, machine-checked as
`ScoreQuantFormal.UnmergedDuplicates.not_strictVoronoi` — every hypothesis but
injectivity of the score map holds there, and the conclusion fails. Zero-weight rows,
singular/pseudodeterminant objectives, extra capacity or mass constraints, and
nonzero solver gain tolerances are outside the theorem. At tolerance
\(\varepsilon>0\), the implementation certifies only that no geometric
disagreement has exact gain exceeding \(\varepsilon\); strict training-label
reproduction need not hold.

**Machine-checked.** The whole of D5 is verified in Lean 4.33.1 + Mathlib for
arbitrary \(d\): the frozen statement boundary is
`formal/ScoreQuantFormal/ExchangeVoronoiSpec.lean` and the proof is
`ScoreQuantFormal.exchange_voronoi`, with the quantitative bound as
`ScoreQuantFormal.violation_log_gain` and its determinant forms
`violation_lower_bound` / `violation_strict_gain`. Distinct centroids are
*derived* there, as here, by `centroid_ne_of_stable`. Independent statement
audit: `AUDITS/FORMALIZATION-D-EXCHANGE-IMPLIES-VORONOI-001.md`. The
formalization covers the merged-atoms branch of the duplicate hypothesis only,
and certifies nothing about the Python/JAX implementation.

Publication-grade audit and proof: `AUDITS/AUDIT-D-EXCHANGE-VORONOI-001.md`.
Exact-rational regression: `py/audit_d_exchange_voronoi.py`.

## D6. Exact finite inductive closure / compiler — [PROJECT-PROVED]

**Claims:** D-FINITE-INDUCTIVE-CLOSURE, D-CLOSURE-DUPLICATE-INHERITANCE, D-COMPILE-TOLERANCE-GUARANTEE

Every one-point-exchange-stable positive-definite finite D solution with exactly
\(K\) nonempty cells can be compiled to

\[
\boxed{
\hat q_D(s)=
\arg\min_b(s-\mu_b)^\top\widehat I^{-1}(s-\mu_b).
}
\]

For merged distinct positive-weight score atoms at exact zero-tolerance
stability, this predictor reproduces **all training labels exactly**, without a
tie breaker. Therefore an exact terminal D exchange solution is not merely
transductive: it has a canonical deployable extension.

The nonempty-cell hypothesis is load-bearing, not bookkeeping. An empty cell has
zero mass, hence the degenerate centroid \(0\), and a training row at the origin
is exactly as near to it as to its own centroid: at \(d=1\), \(N=2\), \(K=3\)
with scores \((-2,0)\), unit weights and \(z=(0,1)\), the configuration is
stable, positive definite and has injective scores, yet row 1 ties between its own
cell and the empty one, so a conforming rule may return the empty cell. The
statement of `D-FINITE-INDUCTIVE-CLOSURE` was amended on 7 September 2026 to
carry this hypothesis, which D5's node already had.

Original duplicate rows inherit the merged atom's label
(`D-CLOSURE-DUPLICATE-INHERITANCE`): merging preserves cell masses, cell sums,
centroids and the retained information, so a rule built from the *unmerged* data
is the same rule. That direction is the content — the unmerged scores are not
distinct, so D5 does not apply to them directly.

A finite numerical solver using a positive gain tolerance has the weaker,
explicitly tolerance-stamped compiler guarantee documented by `GeometryReport`
(`D-COMPILE-TOLERANCE-GUARANTEE`). The derivation is D5's quantitative bound read
backwards: a nearest-centroid disagreement from a non-singleton source has exact
gain at least \(\log(1+\alpha\beta q_\delta^2/4)>0\), so a solver that
accepted no move of gain above \(\varepsilon\) leaves only disagreements whose
guaranteed gain is at most \(\varepsilon\). Strictness is gone with it, which is
why prediction keeps a deterministic tie-break. That regime is where every real
solver lives, and it is not formalized.

**Machine-checked.** The exact zero-tolerance statements are
`ScoreQuantFormal.closure_reproduces_labels` and
`ScoreQuantFormal.closure_duplicates`, resting on
`ScoreQuantFormal.merge_invariance` (`formal/ScoreQuantFormal/Merge.lean`) for the
preservation of masses, sums, centroids and information under merging — which is
the proof mechanism, carried by no claim node of its own. All in
`formal/ScoreQuantFormal/Closure.lean`,
frozen in `ClosureSpec.lean` and `MergeSpec.lean` and audited in
`AUDITS/FORMALIZATION-D-FINITE-INDUCTIVE-CLOSURE-001.md`. Not covered: the
positive-tolerance guarantee; D5's second duplicate branch, since
`ClosureDuplicateConclusion` *assumes* the inherited labeling rather than
deriving that labels are constant on a duplicate class; and any behaviour of the
rule away from the training scores.

## D7. Every finite global D optimum is geometrically realizable — [PROJECT-PROVED COROLLARY]

**Claims:** D-GLOBAL-GEOMETRIC-REALIZABILITY

A finite global optimum is exchange stable, hence strict D-Voronoi. Therefore unrestricted finite D assignment optimization and global optimization over the corresponding realizable D-Voronoi/affine-max labelings have the same optimum value.

This does **not** say every D-Voronoi fixed point is globally optimal.

The realizability half is machine-checked as
`ScoreQuantFormal.globalOptimum_strictVoronoi` in
`formal/ScoreQuantFormal/Corollaries.lean`. The equal-optimum-value half is
not formalized, so this claim carries no `formal_proof` field.

## D8. Monotone exact one-point exchange — [PROJECT-PROVED]

**Claims:** D-EXCHANGE-TERMINATES

Accepting only exact positive D gains gives:

- strict objective ascent;
- no cycles;
- finite termination because the labeling set is finite;
- a terminal one-point exchange-stable solution;
- by D5/D6, a canonical deployable D quantizer.

**Machine-checked.** All three clauses are now frozen in
`formal/ScoreQuantFormal/TerminationSpec.lean` and discharged by
`ScoreQuantFormal.d_exchange_terminates`, audited in
`AUDITS/FORMALIZATION-D-EXCHANGE-TERMINATES-001.md`. The argument is generic in
the objective, so `ScoreQuantFormal.terminates`, `ascent_strict` and `no_cycle`
also cover the termination sentences of DS3 and A1 — but neither \(F_s\) nor
\(F_A\) exists as a Lean object in that tree, so those two claims carry no
`formal_proof` and this note is their record.

Two cautions. The \(\det\) and \(\log\det\) runs are different runs: a
\(\det\)-stable state can still admit a \(\log\det\)-improving move, so both
instances are stated and proved separately. And nothing formal chains a terminal
stable state back to D5 — the fifth bullet above is prose, not a machine-checked
consequence. `ScoreQuantFormal.no_infinite_strict_ascent` and
`exists_exchangeStable` in `Corollaries.lean` remain as the earlier, weaker
fragments.

## D9. Adaptive Mahalanobis Lloyd is not monotone — [COUNTEREXAMPLE]

**Claims:** D-GUARDED-LLOYD, D-LLOYD-NONMONOTONE

The batch iteration “compute \(I^{-1}\) → nearest-centroid reassignment → recompute \(I\)” can decrease \(\log\det I\).

Reason:

\[
\log\det J
\le
\log\det I+\operatorname{tr}(I^{-1}(J-I)),
\]

so the fixed-metric tangent is an **upper** bound, not a minorizer.

Measured suite: decreasing steps occurred in 57/300 instances; one explicit example loses about 0.137 nat.

## D10. Voronoi fixed point does not imply exchange stability — [COUNTEREXAMPLE]

**Claims:** D-VORONOI-NOT-EXCHANGE

Measured suite: 35/100 Lloyd/Voronoi fixed points still admitted an exact improving one-point move, with improvements up to about 1.033 nat.

The exact witness `CE-D-VORONOI-CONVERSE-001` is machine-checked as
`ScoreQuantFormal.VoronoiConverse.strictVoronoi` together with
`VoronoiConverse.not_exchangeStable` in
`formal/ScoreQuantFormal/Counterexamples.lean`: every row is strictly nearest
its own centroid, yet moving row 2 raises the retained information from
\(25/48\) to \(9/16\). The claim's `formal_proof` field stays empty because
its `statement` is the proposition being refuted (ADR 0030).

## D11. Exact global enumeration for fixed \((d,K)\) — [PROJECT-PROVED]

**Claims:** D-GLOBAL-XP

D7 restricts candidate global labelings to affine-max-realizable partitions. Arrangement enumeration gives an exact XP algorithm of form

\[
\boxed{N^{O(Kd)}}
\]

for fixed \((d,K)\), with an effective affine parameter count of order \((K-1)(d+1)\).

This is an application of a known computational-geometry template; hardness/FPT status remains open.

## D12. Singleton-refinement branch-and-bound bound — [PROJECT-PROVED]

**Claims:** D-BB-SINGLETON-BOUND

For a partial assignment and unassigned point set \(U\), singleton refinement yields

\[
\boxed{
I_{\rm completion}
\preceq
I_{\rm partial}+
\sum_{i\in U}w_is_is_i^\top.
}
\]

Thus

\[
\log\det\left(
I_{\rm partial}+\sum_{i\in U}w_is_is_i^\top
\right)
\]

is a valid D upper bound for every completion.

Measured implementation: exact agreement with exhaustive search on small instances and certificates through \(N=40\) for reported \(d=2,K=3\) tests. The hardest reported \(N=40\) instance visited 131,799 nodes in 8.3 s; this is instance-dependent evidence, not a worst-case claim.

---
