# 7. A-optimality control theory

> Part of the ScoreQuant known-results ledger. Read `PROBLEM.md` first and
> `KNOWN_RESULTS/index.md` for the status vocabulary and the chapter map.
> Resolve any claim id with `python py/registry.py show <ID> --deps --proof`.

\[
F_A(I)=-\operatorname{tr}(I^{-1}),
\qquad
G_A=I^{-2}.
\]

## A1. Exact finite move oracle — [PROJECT-PROVED]

**Claims:** A-EXACT-MOVE-ORACLE, A-EXCHANGE-TERMINATES

The universal rank-two \(\Delta I\) plus Woodbury gives an exact \(O(d^2)\)-type A move oracle; exact positive-gain exchange is monotone and finitely terminating.
The termination sentence is machine-checked generically, as
`ScoreQuantFormal.terminates` with `ascent_strict` and `no_cycle`, in
`formal/ScoreQuantFormal/Termination.lean`: the argument uses only that the
objective is real-valued and the labeling set finite. `A-EXCHANGE-TERMINATES` carries no
`formal_proof` all the same, because \(F_A\) exists as no Lean object in that tree
and no frozen conclusion instantiates it — see
`AUDITS/FORMALIZATION-D-EXCHANGE-TERMINATES-001.md`.

## A2. D-style finite geometry theorem fails — [COUNTEREXAMPLE]

**Claims:** A-FINITE-GEOMETRY-FAILS

The reported search found 443 A moves violating the would-be D-style implication. Therefore finite exchange stability does not generally collapse to the first-order \(I^{-2}\) Voronoi rule.

## A3. Concavity screening remains valid — [PROJECT-PROVED]

**Claims:** A-TANGENT-SCREENING

G2 supplies a sound tangent rejection rule for A.

## A4. Quantitative A necessity bound — [OPEN]

No analogue of the \(D_s\) Prop.-17-style \(O(w)\) violation bound has yet been derived.

---
