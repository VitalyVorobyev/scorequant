# 5. \(D_s\)-optimality

> Part of the ScoreQuant known-results ledger. Read `PROBLEM.md` first and
> `KNOWN_RESULTS/index.md` for the status vocabulary and the chapter map.
> Resolve any claim id with `python py/registry.py show <ID> --deps --proof`.

## DS0. Profiled objective and Schur notation — [LIT]

**Claims:** DS-SCHUR

Let \(\theta=(\psi,\lambda)\) and

\[
F_s(I)
=\log\det S_\psi(I)
=\log\det I-\log\det I_{\lambda\lambda}
\]

in the nonsingular block regime.

## DS1. Classical \(D_s\) design theory — [LIT]

**Claims:** DS-CLASSICAL-DESIGN-THEORY

Wynn, Whittle, Näther–Reinsch, Kiefer/general equivalence theory provide classical subset/nuisance-parameter optimal design, sensitivity, and singular-case tools. They do not directly solve the quantizer feasible set.

## DS2. Gradient / efficient semimetric — [PROJECT-PROVED/BRIDGE]

**Claims:** DS-GRADIENT-EFFICIENT-SEMIMETRIC

For regular nonsingular blocks,

\[
\boxed{
G_s
=
I^{-1}
-E_\lambda I_{\lambda\lambda}^{-1}E_\lambda^\top
\succeq0,
}
\]

of rank \(d_\psi\). Population first-order stationarity therefore induces an efficient-score semi-metric / affine-max geometry.

## DS3. Exact finite one-point objective oracle — [PROJECT-PROVED]

**Claims:** DS-EXACT-MOVE-ORACLE, DS-EXCHANGE-TERMINATES

The same rank-two full-information update applies. The exact profiled gain is

\[
\boxed{
\Delta F_s
=
\Delta\log\det I
-
\Delta\log\det I_{\lambda\lambda},
}
\]

with each term evaluable by low-rank determinant algebra when blocks remain nonsingular. Exact positive-gain \(D_s\) exchange is therefore monotone and finitely terminating on a finite sample.
The termination sentence is machine-checked generically, as
`ScoreQuantFormal.terminates` with `ascent_strict` and `no_cycle`, in
`formal/ScoreQuantFormal/Termination.lean`: the argument uses only that the
objective is real-valued and the labeling set finite. `DS-EXCHANGE-TERMINATES` carries no
`formal_proof` all the same, because \(F_s\) exists as no Lean object in that tree
and no frozen conclusion instantiates it — see
`AUDITS/FORMALIZATION-D-EXCHANGE-TERMINATES-001.md`.

## DS4. D-style finite geometry theorem fails — [COUNTEREXAMPLE]

**Claims:** DS-FINITE-GEOMETRY-FAILS

A positive first-order efficient-Voronoi margin need not imply positive exact finite \(D_s\) gain. Therefore

\[
\text{exchange stable}\not\Rightarrow\text{exact }G_s\text{-Voronoi}
\]

in general.

## DS5. A global finite \(D_s\) optimum can be non-geometric — [EXACT COUNTEREXAMPLE]

**Claims:** DS-GLOBAL-NONGEOMETRIC

There is a centered equal-weight \(N=8,d=2,d_\psi=1,K=3\) example for which exhaustive enumeration of all 966 unlabeled nonempty partitions gives a unique global \(D_s\) optimum that violates its own efficient-semi-metric nearest-cell rule.

Canonical fixture is stored in `COUNTEREXAMPLES/CE-DS-GLOBAL-GEOMETRY-001.json`.

This proves that unrestricted finite \(D_s\) assignment and deployable self-consistent geometric \(D_s\) fitting are genuinely different finite problems.

## DS6. Approximate finite efficient-Voronoi bound — [PROJECT-PROVED]

**Claims:** DS-OKN-BOUND

At a \(D_s\) one-point exchange-stable state, the relative first-order violation obeys a bound of the form

\[
\boxed{
\frac{s_{aa}-s_{bb}}{q_{aa}}
\le
w_i\left(\frac1{W_a}+\frac1{W_b}\right).
}
\]

For equal weights and balanced cells this is \(O(K/N)\).

Measured suite: the observed maximum violation shrank from roughly 0.18 to 0.029 as \(N\) increased from 8 to 64 in the reported experiment.

## DS7. Full-data efficient-score domination — [PROJECT-PROVED; see also DS11(a)]

**Claims:** DS-EFFICIENT-SCORE-DOMINATION, DS-EFFICIENT-SCORE-GLOBAL-UPPER

Let

\[
\widehat S=S_\psi-B^*S_\lambda,
\qquad
B^*=I^{\rm full}_{\psi\lambda}(I^{\rm full}_{\lambda\lambda})^{-1}.
\]

For **every** quantizer \(q\),

\[
\boxed{
S_\psi(I_q)
\preceq
\operatorname{Var}(E[\widehat S\mid q(S)]).
}
\]

Interpretation: profiling nuisance information from the bins cannot beat first projecting to the full-data efficient score and then measuring the retained between-cell information of that projection.

Taking log-determinants and suprema gives

\[
\boxed{
\sup_q F_s(q)
\le
\text{best D-optimal K-bin value for the }d_\psi\text{-dimensional efficient score}.
}
\]

For the last equality to deterministic quantization of \(\widehat S\), atomlessness is required for the **efficient-score law itself**.

## DS8. Scalar efficient-score upper problem is exactly solvable by DP — [PROJECT-PROVED/BRIDGE]

**Claims:** DS-SCALAR-EFFICIENT-DP

For \(d_\psi=1\), deterministic D-optimal quantization of a scalar atomless efficient score has ordered interval cells. On a finite sample the optimal interval partition can be solved exactly by dynamic programming.

Consequences:

- cheap globally optimal upper bound for in-bin \(D_s\);
- strong initializer;
- explicit optimality-gap certificate for a found \(D_s\) solution.

Measured certificate gaps were 0.003–0.118 nat on reported \(d=3,d_\psi=1,K=4,N=60\) tests and 0.011–0.19 nat on reported \(d=4,d_\psi=2,K=5,N=80\) tests using the corresponding projected D upper problems.

## DS9. The \(K\le d\) feasibility split — [PROJECT-PROVED/STRUCTURAL]

**Claims:** DS-FULL-PROFILE-K-LE-D-SINGULAR, DS-PROJECTED-K-REQUIREMENT

For the **full in-bin profiled formulation**, \(I_q\) is singular when \(K\le d\), so ordinary full-block \(D_s\) profiling is not identifiable.

The projected efficient-score problem only needs enough categories to identify the \(d_\psi\)-dimensional efficient score, i.e. structurally \(K\ge d_\psi+1\), if nuisance information is supplied externally.

These are different statistical problems and must be exposed separately.

## DS10. Finite-to-population bridge: resolution map — [SUMMARY]

The bridge programme (OP4/OP5, packet packet `DS-POPULATION-BRIDGE` (git history),
28 Aug 2026) is resolved as follows:

- DS11 — variational form of the profiled objective, \(\Phi\)-neutral splits,
  and the exact equality condition for efficient-score domination (OP6, fixed
  \(q\));
- DS12 — rigorous population stationary \(D_s\) geometry and its deployability
  characterization (OP5);
- DS13 — exact finite leverage stability bound (the finite half of the bridge);
- DS14 — conditional finite\(\to\)population bridge theorem (OP4);
- DS15 — margins dichotomy at global finite \(D_s\) optima (OP28, resolved
  for conditionally centered laws at \(d_\psi=d_\lambda=1\); audited).

Residual open conditions live in `OPEN_PROBLEMS.md` OP29 (margins beyond
conditional centering — non-centered laws, \(d_\psi>1\), \(d_\lambda\ge2\),
exchange-stable non-global sequences) and C2 (unrestricted population
attainment).
