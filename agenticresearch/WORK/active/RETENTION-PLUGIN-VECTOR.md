# RETENTION-PLUGIN-VECTOR — error bars for the vector geometric-mean retention under a frozen rule

**Programme:** P4 (OP27) · **Opened:** 5 September 2026 · **Status:** active
(literature-first pass done 5 September 2026; derivation not started)
**Source:** branch `main` after PR #53 (O6 proved and audited) and the
literature commit that rewrote this packet

## Goal

Extend O6 (`RETENTION-PLUGIN-CLT-FROZEN-SCALAR`, audited
`AUDITS/AUDIT-SCORE-ORACLE-ROBUSTNESS-001.md`) from one score coordinate to
a \(d\)-dimensional true score. The target is the number the library
actually reports for a frozen rule on an oracle-score evaluation sample,
`information_report(...).geometric_mean_retention`:

\[
\eta_D=\Big(\frac{\det I_Z}{\det V}\Big)^{1/d},\qquad
V=E[SS^\top],\qquad I_Z=\sum_b\frac{m_bm_b^\top}{p_b},\qquad
m_b=E[S\,\mathbf 1_{Z=b}],
\]

with the plug-in \(\hat\eta_D\) built from the same cell moments
(\(0/0:=0\) on empty cells). The question: **is \(\sqrt n(\hat\eta_D-\eta_D)\)
asymptotically normal with an explicit, consistently estimable variance, and
exactly where does the log-determinant degenerate?** This is the matrix
influence function O6.8 named as the missing derivation.

## Prior art located before the derivation (5 September 2026)

`LITERATURE/audits/RETENTION-PLUGIN-CLT-FROZEN-VECTOR-5-September-2026.md`
(round 11). The reframing that made the search work: with
\(M=E[S\mathbf 1_Z^\top]\) and \(P=\operatorname{diag}(p)\),
\(\det I_Z/\det V=\det(V^{-1}MP^{-1}M^\top)=\prod_{i\le d}\rho_i^2\), so
\(\eta_D\) is the **geometric mean of the squared uncentred canonical
correlations between \(S\) and the cell indicator**; at the evaluation law
(\(E[S]=0\)) it is the MANOVA between/total determinant ratio and
\(\Lambda=\prod(1-\rho_i^2)\) is Wilks' criterion for the grouping \(Z\).

| Component of O7 | Status | Cite |
|---|---|---|
| CLT for a smooth function of the sample mean of \(T=(\mathbf 1_{Z=b},S\mathbf 1_{Z=b},\operatorname{vech}SS^\top)_b\) | prior art | van der Vaart Thm 3.1 (`vanderVaart-1998`) |
| \(d\log\det A=\operatorname{tr}A^{-1}dA\), \(dA^{-1}=-A^{-1}(dA)A^{-1}\) | prior art | Magnus & Neudecker §8.3–8.4 (`Magnus-Neudecker-1999`) |
| Influence function of \(\Lambda\)-type and noncentrality-matrix parameters | prior art (centred, several populations) | `Radhakrishnan-Kshirsagar-1981`; per-coefficient form `Romanazzi-1992` |
| Asymptotic normality under finite fourth moments, no normality | prior art (continuous blocks, distinct \(\rho_i\)) | `Muirhead-Waternaux-1980`; general eigenvalue functions `Fang-Krishnaiah-1982` |
| Fixed-alternative asymptotics of a determinant ratio | prior art (normal theory) | `Sugiura-Fujikoshi-1969`; nonnormal companion Fujikoshi (2002) |
| Singular \(I_Z\) as a dimensionality hypothesis | prior art (null theory) | `Seo-Kanda-Fujikoshi-1995` |
| Geometric-mean form, fixed partition of a *different* variable, uncentred moments, \(0/0:=0\) | **search gap** | — |
| Plug-in \(\hat\sigma^2=n^{-1}\sum\hat\psi_i^2\) in cell moments, its a.s. consistency | **search gap** | — |
| Endpoints \(\eta_D\in\{0,1\}\), the \(\sigma^2=0\) set, the library's rank projection | **search gap** | — |

Consequence: **cite the first six rows and reduce them to the cell
moments; spend the session on the last three.** Two of the cited texts are
known by abstract only (`gaps.md`); read the statements before citing them
for more than the method.

## Independence and scope

- Build on O6 as audited; do not re-prove it. Reuse its normalisation
  (frozen rule, iid equally weighted evaluation sample from the reference
  law, scores never centred) and its assumptions (A1)–(A3) in matrix form.
- Do not re-derive the delta method or the determinant differential; the
  contribution is the reduction and the endpoints.
- Falsify before proving (`protocols/numerical.md`): exact rationals, \(d=2,3\),
  smallest rank-feasible \(K\) (\(I_Z\) has rank \(\le K\), and \(\le K-1\)
  when \(E[S]=0\): `FI-RANK-CEILING`), duplicate atoms, singleton and empty
  cells, near-singular \(V\) and near-singular \(I_Z\).
- Out of scope, flag only: refitted rules (the boundary non-smoothness of
  OP27), weights, \(D_s\)/profiled retention, bootstrap comparisons, any
  public uncertainty API. No `src/` change.

## Attack plan (reordered after the literature pass)

1. **Finite identity and library agreement.** Show \(\hat\eta_D\) equals
   `geometric_mean_retention` whenever \(\hat V\succ0\) and no direction is
   projected out (the whitening \(\hat V^{-1/2}\) cancels in the determinant
   ratio). Name exactly what `rank_rtol` does to the plug-in when \(\hat V\)
   is numerically singular (a discontinuity of the estimator, outside the
   CLT). Generalise the RSS identity: \(\hat V-\hat I_Z\) is the within-cell
   scatter about cell means, so \(\hat I_Z\preceq\hat V\) and
   \(0\le\hat\eta_D\le1\); note this is Kendall & Stuart's (26.50) in matrix
   form (`Kendall-Stuart-1961`).
2. **Endpoints first (the O6 lesson).** Before any limit theorem: (a)
   singular \(I_Z\) (\(\eta_D=0\), \(\log\det=-\infty\)) — the plug-in is
   \(O_p(1/n)\)-biased upward from 0, the Wald interval is unsupported, and
   the classical reading is the dimensionality test
   (`Seo-Kanda-Fujikoshi-1995`); (b) \(\eta_D=1\) (\(S\) a function of \(Z\)
   a.s.); (c) the \(\sigma^2=0\) set: \(\psi=0\) a.s. is a quadratic variety
   per cell — state which laws sit on it and check the endpoints
   separately, since at \(d=1\), \(\eta=0\) every law with zero cell means
   has \(\psi\equiv0\).
3. **The reduction (cite, then compute).** \(g(p,M,V)=(\det\sum_b
   m_bm_b^\top/p_b/\det V)^{1/d}\) is \(C^\infty\) on \(\{p_b>0,\ I_Z\succ0,\
   V\succ0\}\); assumptions (A1) \(p_b>0\); (A2) \(E\|S\|^4<\infty\); (A3)
   \(V\succ0\); **(A3′) \(I_Z\succ0\)** (needs \(K\ge d+1\) under
   \(E[S]=0\)); (A4) \(\sigma^2>0\). Apply van der Vaart Thm 3.1 to the
   estimator's own everywhere-defined functional \(\phi\) (the O6-audit
   \(\phi\)-route), with the differentials of Magnus & Neudecker §8.3–8.4,
   and **derive \(\psi\) in cell moments**:
   \[
   \psi(S,Z)=\frac{\eta_D}{d}\Big[\,2S^\top I_Z^{-1}c_Z-c_Z^\top I_Z^{-1}c_Z-S^\top V^{-1}S\,\Big],
   \quad c_b=m_b/p_b\ (\text{conjectured; verify}).
   \]
   Check \(E[\psi]=0\) exactly (\(\sum_bp_bc_b^\top I_Z^{-1}c_b=d\),
   \(E[S^\top V^{-1}S]=d\)); check \(d=1\) recovers O6.2's
   \(\psi=((1-\eta)S^2-(S-c_Z)^2)/v\); check that at the evaluation law it
   agrees with the \(\Lambda\)/noncentrality-matrix influence function of
   `Radhakrishnan-Kshirsagar-1981` (read the statement first) and with the
   average of Romanazzi's per-coefficient functions divided by \(\rho_i^2\)
   when the \(\rho_i\) are simple. Give the covariance form
   (numerator/denominator log-determinant influences).
4. **Consistent variance.** \(\hat\sigma^2=n^{-1}\sum\hat\psi_i^2\) as a
   continuous function of within-cell moments of order \(\le4\)
   (\(E[S_iS_jS_kS_l\mathbf 1_{Z=b}]\)); a.s. consistency under (A1)–(A3′)
   with the Borel–Cantelli remark for empty cells.
5. **Closed-form references and falsification.** A \(d=2\) law with
   closed-form cell moments: the two-parameter Gaussian location–scale
   score \(S=(x-\mu,\,((x-\mu)^2-\sigma^2)/\sigma)\) under a fixed interval
   partition of \(x\) (moments are Gaussian-CDF/PDF differences), or the
   door2 three-component mixture (\(d=2\), `examples/door2_mixture_densities.py`)
   with a frozen D-exchange rule and quadrature references. Coverage at
   \(n=100,300,1000,3000\), \(\ge2000\) replicates, fresh seeds, deterministic
   revision and script hash recorded under `WORK/artifacts/RETENTION-PLUGIN-VECTOR/`.
   Read the second-order picture against `Sugiura-Fujikoshi-1969` and
   Ogasawara (2007) rather than re-deriving it.
6. **Self-adversarial pass (protocol G):** ties and duplicates; singleton and
   empty cells; \(K\le d\) (singular \(I_Z\)); numerically singular \(V\)
   (the library projects, the theorem excludes); atomic laws; heavy tails
   (only \(E\|S\|^2<\infty\)); estimated scores enter only through the label
   map; the \(D\)/\(D_s\) distinction (this packet is full-\(D\) only);
   repeated canonical correlations (the determinant route must not need
   simple \(\rho_i\); the per-coefficient route does).

## Required deliverables

- `KNOWN_RESULTS/10-oracle.md` section **O7** in the O6 format (normalised
  target, identity, CLT, variance, Wald interval and its unsupported cases,
  what it measures, protocol-G pass, measured table, verdict), with the
  canonical-correlation reading stated once.
- Claim nodes `RETENTION-PLUGIN-CLT-FROZEN-VECTOR` (status `bridge` if
  proved, with `assumptions` fully explicit, `dependencies` on the O6 node,
  `FI-QUANT-IDENTITY`, `FI-RANK-CEILING`, `INFO-D-EFFICIENCY`, and
  `literature` carrying the round-11 keys) and a measured companion; patch
  `OPEN-RETENTION-UNCERTAINTY` to name the remainder.
- Instrument `py/retention_plugin_vector.py` (exact `fractions.Fraction`
  identities in \(d=2,3\); closed-form or quadrature references; coverage with
  recorded seeds); ledger rows `N-VECTOR-RETENTION-*` in `NUMERICAL_EVIDENCE.md`;
  a CI pin in `tests/test_research_claims.py` (influence function against
  Gateaux finite differences in \(d=2\), and the determinant identity against
  `information_report`).
- Literature: **extend** `LITERATURE/audits/RETENTION-PLUGIN-CLT-FROZEN-VECTOR-5-September-2026.md`
  (do not start a new file) with the post-derivation claim-by-claim check:
  the statements of `Radhakrishnan-Kshirsagar-1981` and
  `Fang-Krishnaiah-1982` read and compared with the derived \(\psi\), and the
  second citation hop named in `gaps.md`.
- `manuscripts/README.md` note; packet moved to `WORK/completed/` with an
  Outcome section; `reindex` and `validate` clean; the research-claims tests
  green.

## Stop conditions

1. **Proved:** the vector CLT, the influence function, the consistent
   variance and the Wald statement hold under explicit (A1)–(A4), with the
   \(d=1\) reduction to O6 checked exactly, the agreement with the cited
   influence functions checked at the evaluation law, and coverage
   replicating.
2. **Reduced:** a named missing step (most likely the \(\sigma^2=0\)
   characterisation or the singular-\(I_Z\) boundary); record what is proved
   and open the gap as its own node.
3. **Refuted:** an exact counterexample to the identity or the influence
   function, serialised in `COUNTEREXAMPLES/` and pinned.

Do not close on coverage alone; the verdict is about the derivation. A
search gap is not novelty: the O7 section must say that the method is
textbook and name what is project-level.

## Next dependency-blocking question

Rules **refitted on the evaluation sample** (the genuine OP27): the
evaluation sample enters the cell boundaries, the pairs \((S_i,Z_i)\) are no
longer iid, and the hard-assignment non-smoothness appears. First target: the
population D-exchange/Voronoi stationary rule of `D-POP-VORONOI` with an
empirical-process argument under a margin condition, or an exact
counterexample showing the plug-in is not \(\sqrt n\)-normal without one.
