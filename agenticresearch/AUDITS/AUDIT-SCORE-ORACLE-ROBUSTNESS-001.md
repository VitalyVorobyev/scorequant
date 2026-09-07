# Publication-grade audit of the frozen-rule scalar retention plug-in CLT

**Claims:** `RETENTION-PLUGIN-CLT-FROZEN-SCALAR` (bridge), `RETENTION-PLUGIN-COVERAGE-DOOR3` (measured), and the 5 September patch of `OPEN-RETENTION-UNCERTAINTY`
**Audit:** `AUDIT-SCORE-ORACLE-ROBUSTNESS`
**Date:** 5 September 2026
**Source:** branch `score-oracle-robustness` at `a5f905e` (packet handoff `b9a42c3`)
**Result:** verified with hardened assumptions — O6.1, O6.2, O6.3 and the Wald statement of O6.4 hold as stated; the \(\sigma^2=0\) characterisation is corrected at \(\eta=0\); population references and coverage replicate independently; no re-attribution to prior art.

The auditor did not produce O6 and did not read the researcher's instrument
`py/score_oracle_retention_uncertainty.py`, the completed packet or any
transcript. The independent instrument is
`py/audit_score_oracle_retention_uncertainty.py`; its four provenance-stamped
records live under `AUDITS/artifacts/AUDIT-SCORE-ORACLE-ROBUSTNESS-001/`.
The researcher's `WORK/artifacts/SCORE-ORACLE-ROBUSTNESS/popref.json` and the
5 September literature audit were opened only after the audit's own numbers
and triangulation table existed, as comparison material.

## 1. Target statement

Frozen: training data, provider \(\hat s\), rule \(q\), reference point
\(\theta_0\), \(K\). Random: \(X_1,\dots,X_n\) iid from the reference law \(P\),
equally weighted. Observables \(S_i=s(X_i)\in\mathbb R\) (true score) and
\(Z_i=q(\hat s(X_i))\in\{1,\dots,K\}\). With \(p_b=P(Z=b)\), \(m_b=E[S\mathbf 1_{Z=b}]\),
\(c_b=m_b/p_b\), \(v=E[S^2]\),

\[
\eta=\frac{\sum_b m_b^2/p_b}{v},\qquad
\hat\eta=\frac{\sum_b \hat m_b^2/\hat p_b}{\hat v}\ (0/0:=0).
\]

Claimed: (O6.1) \(\hat\eta=1-\mathrm{RSS}/\mathrm{TSS}\) for every sample and
equals the library's scalar `geometric_mean_retention`; (O6.2) under (A1)
\(p_b>0\), (A2) \(E S^4<\infty\), (A3) \(v>0\):
\(\sqrt n(\hat\eta-\eta)\Rightarrow N(0,\sigma^2)\) with
\(\sigma^2=E\psi^2\), \(\psi=((1-\eta)S^2-(S-c_Z)^2)/v\), equivalently
\([\operatorname{Var}N_1-2\eta\operatorname{Cov}(N_1,S^2)+\eta^2\operatorname{Var}S^2]/v^2\)
with \(N_1=2c_ZS-c_Z^2\); (O6.3) \(\hat\sigma^2=n^{-1}\sum\hat\psi_i^2\to\sigma^2\)
a.s. and \(\sum_i\hat\psi_i=0\); (O6.4) the Wald interval has asymptotic
level \(1-\alpha\) iff (A4) \(\sigma^2>0\), and \(\sigma^2=0\) iff \(S\mid Z=b\) is
supported on the roots of \((s-c_b)^2=(1-\eta)s^2\) ("at most two atoms per
cell"; an atomless cell of positive probability implies (A4)); (O6.5) the
interval is for the true retention, the proxy value \(\tilde\eta\) is bias
excluded with probability \(\to1\).

## 2. Criterion and problem level

- Criterion: the scalar retention ratio; for one score coordinate the D-, A-
  and E-efficiencies coincide with it.
- Level: `information_accounting`, conditional on a frozen
  `empirical_inductive_quantizer`.
- Decision variable: none (the rule is frozen); the random object is the
  evaluation sample.
- Score-oracle regime: an exact oracle score on the evaluation sample, an
  estimated (classifier-ratio) score inside the frozen label map.

## 3. Status before the audit

`RETENTION-PLUGIN-CLT-FROZEN-SCALAR`: `bridge`, `prior_art_found`,
"not independently audited". `RETENTION-PLUGIN-COVERAGE-DOOR3`: `measured`.
`OPEN-RETENTION-UNCERTAINTY`: `open`, with the special case recorded as
settled. Under `protocols/audit.md` nothing of O6 may be used by the library
or cited before this audit closes.

## 4. Dependencies rechecked

- `FI-QUANT-IDENTITY` (literature). Its hypotheses are a regular model and
  a quantizer fixed with respect to \(\theta\). Here \(q\circ\hat s\) is a fixed
  measurable map of \(x\) — \(\hat s\) was fitted once, at \(\theta_0\), on
  training data that is frozen — so \(P_\theta(Z=b)=\int_{\{q(\hat s(x))=b\}}p(x\mid\theta)\,dx\)
  and the categorical score at \(\theta_0\) is \(E[S\mid Z=b]=c_b\) under the same
  differentiation-under-the-integral regularity as the full model. Hence
  \(I_Z=\sum_bp_bc_b^2=\sum_bm_b^2/p_b\) and, with \(E[S]=0\), \(v=I_{\rm full}\).
  Checked, not re-proved; the hypothesis is met only when the evaluation law
  is \(P_{\theta_0}\) (see §9, attack 5).
- `PROXY-TRUE-RETAINED-FI` (bridge): \(\eta\) is
  \(\operatorname{Var}(E[s\mid q(\hat s)])/\operatorname{Var}(s)\), the true
  retained fraction, exactly because the numerator uses \(s\) and the label map
  uses \(\hat s\). Consistent with O6.0; nothing further imported.
- Imported theorems: the multivariate CLT for iid vectors with finite second
  moments and the delta method for a map differentiable at the limit point.
  Their feasible-set assumptions are the whole of §8.

## 5. Nearest literature and transfer boundary

Fresh triangulation recorded in
`LITERATURE/audits/AUDIT-SCORE-ORACLE-ROBUSTNESS-5-September-2026.md`; the
table there carries the six fields per source and the verification label of
every theorem number. Summary:

- **Method — prior art, confirmed.** O6.2 is the multivariate delta method
  applied to the sample-mean vector of \((\mathbf 1_{Z=b},S\mathbf 1_{Z=b},S^2)_b\);
  O6.3 is the plug-in variance principle for a polynomial functional; the
  form \(\sigma^2=E[\psi^2]\) is the influence-function variance formula.
  Sources: van der Vaart (1998) Theorem 3.1 (delta method; primary text)
  and Example 3.2 (sample variance under \(EX^4<\infty\)); Cramér (1946)
  Chapter 28 §28.4 "Functions of moments" (primary text); Serfling (1980)
  Chapter 3 §3.3 (functions of asymptotically normal vectors — the
  Theorem A/B labels could not be read and stay unresolved); Hampel et al.
  (1986) §2.1, p. 85 (\(V(T,F)=\int\mathrm{IF}^2dF\); secondary). Two of the
  researcher's four "unverified" notes are now confirmed in primary text,
  two remain unresolved; Example 3.2 and Chapter 20 (functional delta
  method) are added.
- **The exact statement — search gap, not novelty.** No located source states
  the uncentred, fixed-partition correlation ratio with \(0/0:=0\), the RSS
  identity, the closed-form \(\psi\) and the root-set characterisation of
  \(\sigma^2=0\). The correlation-ratio sampling theory (Kendall & Stuart;
  Wishart 1932) is centred and normal-theory; the effect-size interval
  literature (Kelley 2007) is noncentral-\(F\) under normality; the
  delta-method/influence treatments of \(R^2\) and intraclass correlations
  under non-normality are the nearest kin but use a random regressor or a
  random-effects law, not a fixed measurable partition of an oracle score.
- **What transfers / does not.** The delta method, the fourth-moment
  condition and the influence-variance formula transfer verbatim. Nothing
  normal-theory transfers. Edgeworth theory for studentized functions of
  sample moments (Bhattacharya–Ghosh 1978; Hall 1992) explains the measured
  positive skew of the plug-in studentized statistic and the \(O(1/n)\) bias
  as the standard second-order picture — it is context for O6.7, not a
  theorem O6 relies on.

## 6. Counterexample search

Instrument stage `exact` (`exact.json`), all in `fractions.Fraction`:

- Six adversarial samples — ties and duplicate scores, a singleton cell, an
  empty declared cell (\(K\) larger than the occupied cells), an empty cell
  next to a zero-mean cell, \(\hat\eta=0\), \(\hat\eta=1\). On every one the
  moment form and the RSS form of \(\hat\eta\) coincide exactly,
  \(\sum_i\hat\psi_i=0\), the order-four moment expansion of \(\hat\sigma^2\)
  equals the direct sum, and the library's `geometric_mean_retention` and
  `binned_fisher_information` agree with the exact values to
  \(\le2.2\cdot10^{-16}\). All-zero scores are refused by the library
  (`ContractError`: no positive informative direction), matching "target
  undefined".
- Sixty random rational atomic laws (\(K=1,2,3\), unequal cell masses,
  duplicate atoms): \(E[\psi]=0\), \(E[N_1]=\eta v\), the covariance form of
  \(\sigma^2\), and the atom-wise gradient reduction \(\nabla g^\top(T-\theta)=\psi\)
  hold exactly on 60/60.
- Degenerate laws: the two-atom law of §8 (\(\eta=3/4\), \(\sigma^2=0\)); the
  double-root law (a cell with \(S\equiv0\), \(\eta=3/4\), \(\sigma^2=0\)); the
  \(\eta=0\) law with four atoms in one cell (\(\sigma^2=0\)). Exhaustive
  enumeration of every sample composition (494, 164 and 923 compositions up
  to \(n=8,8,6\)): in the two-atom law \(\hat\eta\ge\eta\) always and
  \(\hat\eta=\eta\) only with \(\hat\sigma=0\).

**Boundary failure found:** the \(\eta=0\) law is an exact counterexample to
the sentences "at most two atoms per cell" and "an atomless cell of positive
probability implies (A4)". Serialised as
`COUNTEREXAMPLES/CE-O6-ETA-ZERO-MULTIATOM-VARIANCE-001.json` and pinned. No
counterexample to O6.1, O6.2, O6.3 or the Wald statement of O6.4 was found.

## 7. Algebraic reduction

Let \(T=(\mathbf 1_{Z=b},S\mathbf 1_{Z=b},S^2)_{b}\in\mathbb R^{2K+1}\),
\(\theta=E[T]=(p_b,m_b,v)_b\), \(g(p,m,v)=\sum_bm_b^2/p_b\,/\,v\).

**Gradient.** \(\partial_{p_b}g=-m_b^2/(p_b^2v)\), \(\partial_{m_b}g=2m_b/(p_bv)\),
\(\partial_vg=-\eta/v\). At an atom \((s,b)\):

\[
\nabla g^\top(T-\theta)
=\frac{2c_bs-c_b^2}{v}-\frac{\eta s^2}{v}
+\underbrace{\sum_{b'}\Big[\frac{m_{b'}^2}{p_{b'}v}-\frac{2m_{b'}^2}{p_{b'}v}\Big]+\eta}_{\eta-2\eta+\eta=0}
=\frac{(1-\eta)s^2-(s-c_b)^2}{v}=\psi(s,b),
\]

using \(2c_bs-c_b^2=s^2-(s-c_b)^2\). Verified exactly atom by atom on 60 laws.

**Covariance form.** \(\psi=(N_1-\eta S^2)/v\) with \(N_1=2c_ZS-c_Z^2\), so
\(\operatorname{Var}\psi=[\operatorname{Var}N_1-2\eta\operatorname{Cov}(N_1,S^2)+\eta^2\operatorname{Var}S^2]/v^2\);
the cross-term sign is negative, as recorded. \(E[N_1]=\sum_bp_b(2c_b^2-c_b^2)=\eta v\)
and \(E[\psi]=0\), so \(\sigma^2=E\psi^2\).

**Order-four expansion (O6.3).** With \(\hat\psi_i=(-\hat\eta S_i^2+2\hat c_{Z_i}S_i-\hat c_{Z_i}^2)/\hat v\),

\[
\hat\sigma^2=\frac1{\hat v^2}\sum_b\Big[\hat\eta^2\hat M_{b,4}-4\hat\eta\hat c_b\hat M_{b,3}+(4+2\hat\eta)\hat c_b^2\hat M_{b,2}-4\hat c_b^3\hat M_{b,1}+\hat c_b^4\hat M_{b,0}\Big],
\qquad \hat M_{b,k}=\tfrac1n\sum_iS_i^k\mathbf 1_{Z_i=b},
\]

a polynomial in \((\hat\eta,\hat v^{-1},\hat c_b)\) with coefficients in
moments of order \(0\ldots4\) only, and \(\hat\eta,\hat c_b,\hat v\) are
themselves rational in \(\hat M_{b,0},\hat M_{b,1},\hat M_{\cdot,2}\). Verified
exactly on every sample of §6.

**Root set of \(\sigma^2=0\).** \(\psi=0\) at \((s,b)\) iff \((s-c_b)^2=(1-\eta)s^2\).
For \(0<\eta<1\) the two roots are \(s_\pm=c_b/(1\mp\sqrt{1-\eta})\), both of
the sign of \(c_b\), with \(|s_-|<|c_b|<|s_+|\); if \(c_b=0\) the only root is
\(s=0\). On the root set \(\eta s^2=2c_bs-c_b^2\), so a law supported on it has
\(E[S^2\mid Z=b]=c_b^2/\eta\) in every cell, and the mean constraint fixes the
upper-root weight \(w=(c_b-s_-)/(s_+-s_-)=\eta/(2(1+\sqrt{1-\eta}))\in(0,1)\).
For \(\eta=1\) the single root is \(s=c_b\). **For \(\eta=0\) every \(c_b=0\) and
the equation reads \(s^2=s^2\): the root set is all of \(\mathbb R\).**

## 8. Proof, hardening and the corrected statement

**O6.1.** \(\sum_i(S_i-\hat c_{Z_i})^2=\sum_iS_i^2-\sum_bn_b\hat c_b^2\) over the
occupied cells, and \(n_b\hat c_b^2=n\hat m_b^2/\hat p_b\); empty cells
contribute to neither side. Dividing by \(\sum_iS_i^2=n\hat v>0\) gives the
identity; \(0\le\mathrm{RSS}\le\mathrm{TSS}\) gives \(0\le\hat\eta\le1\). The
library computes \(\sum_bW_b\mu_b^2\) with an empty cell's mean set to \(0\) and
weight \(0\) (`scatter_bin_statistics`), whitens by \(\hat v\), and returns
\(0\) when the retained eigenvalue is \(\le0\), i.e. exactly \(\hat\eta\) with
\(0/0:=0\). Verified exactly (§6). **Holds as stated.**

**O6.2.** Cleaner than the recorded transfer: define the estimator's own
functional on all of \(\mathbb R^{2K+1}\),
\(\phi(p,m,v)=\sum_{b:p_b>0}m_b^2/p_b\,/\,v\) for \(v>0\), \(\phi:=0\) for
\(v\le0\). Then \(\hat\eta=\phi(\bar T)\) for every sample, and \(\phi=g\) on the
open neighbourhood \(\{p_b>0\ \forall b,\ v>0\}\ni\theta\) (A1, A3), so
\(\phi\) is differentiable at \(\theta\) with \(\nabla\phi(\theta)=\nabla g(\theta)\).
By (A2) \(T\) has finite second moments (the largest is \(ES^4\)), the
multivariate CLT gives \(\sqrt n(\bar T-\theta)\Rightarrow N(0,\Sigma)\), and the
delta method (a map differentiable at the point, random vectors with values
in its domain — here the whole space) gives
\(\sqrt n(\phi(\bar T)-\phi(\theta))\Rightarrow N(0,\nabla g^\top\Sigma\nabla g)=N(0,\operatorname{Var}\psi)\).
No "two sequences agree with probability \(\to1\)" step is needed. The
recorded transfer is nevertheless complete: on \(A_n=\{\text{all cells nonempty}\}\),
\(P(A_n^c)\le\sum_b(1-p_b)^n\to0\), \(\hat\eta-g(\bar T)\to0\) in probability
(with \(g(\bar T)\) extended arbitrarily off its domain), and Slutsky transfers
the limit. **Holds as stated; the \(\phi\)-route is recorded in the node.**

**O6.3.** By §7 \(\hat\sigma^2\) is a continuous function of the finitely many
averages \(\hat M_{b,k}\), \(k\le4\), at any point with \(M_{b,0}>0\) and
\(M_{\cdot,2}>0\); each average has a finite mean by (A2)
(\(E|S|^k\mathbf 1_{Z=b}\le1+ES^4\)) and converges a.s. by the strong law; (A1),
(A3) place the limit in the continuity set; and the value at the limit is
\(E[\psi^2]\) expanded in the same moments. The empty-cell convention
\(\hat c_b=0\) is used only on \(A_n^c\), which is a.s. transient because
\(\sum_n(1-p_b)^n<\infty\) (Borel–Cantelli). **Almost-sure is correct as
stated.** \(\sum_i\hat\psi_i=n(1-\hat\eta)-\mathrm{RSS}/\hat v=0\) by O6.1.

**O6.4, Wald statement.** From O6.2, O6.3 and Slutsky, given (A4). **Holds.**

**O6.4, \(\sigma^2=0\) characterisation — hardened (H1).** "\(\psi=0\) a.s. iff
each positive-probability cell is supported on the root set" is correct.
The gloss "at most two atoms per cell" holds for \(0<\eta\le1\) and fails at
\(\eta=0\); the sentence "if \(S\mid Z=b\) is atomless on one cell of positive
probability, (A4) holds automatically" is **false at \(\eta=0\)**: with all
cell means zero \(\psi\equiv0\) for every law. Exact witness
`CE-O6-ETA-ZERO-MULTIATOM-VARIANCE-001` (cell 0: \(\{-3,-1,1,3\}\), cell 1:
\(\{-2,2\}\), \(\eta=0\), four atoms in one cell); atomless witness in
`coverage.json` (\(S\sim N(0,1)\), cells split at \(|S|=0.6745\)). Corrected
statement, now in the node and in O6.4: *(A4) holds automatically iff
\(\eta>0\) and some cell of positive probability is atomless; for
\(0<\eta<1\) the \(\sigma^2=0\) laws are exactly the two-atom-per-cell laws
with the forced weights, and they exist* — the explicit law: cells of
probability \(1/2\), \(S\in\{2/3,2\}\) with weights \((3/4,1/4)\) and its mirror
image, \(\eta=3/4\), \(E[S]=0\), \(\sigma^2=0\), \(w=\eta/(2(1+\tfrac12))=1/4\)
reproduced exactly. In that law \(\hat c_b^2/\hat M_{b,2}\ge3/4\) on every
sample, hence \(\hat\eta\ge\eta\) always (exhaustive to \(n=8\)): the
degenerate case is one-sided, not merely non-normal.

**O6.5 (H2).** The interval is for \(\eta\) under the evaluation law. The
retained-information reading needs the evaluation law to be \(P_{\theta_0}\):
then \(E[S]=0\), \(v=I_{\rm full}\), \(\sum m_b^2/p_b=I_Z\). Under any other law
the same interval estimates the uncentred second-moment ratio of \(S\) on
\(Z\), which is not a Fisher retention of anything; O6.2–O6.4 never use
\(E[S]=0\), so the CLT survives, the interpretation does not. Recorded in the
node's assumptions.

## 9. Adversarial audit (protocol G and the packet's ten attacks)

1. *Empty-cell transfer:* complete; superseded by the \(\phi\)-route (§8).
   Library agreement on empty cells exact. **Verified.**
2. *Gradient algebra:* recomputed by hand and exactly at every atom of 60
   laws; cross-term sign confirmed. **Verified.**
3. *Consistency:* order four suffices (§7); continuity at the limit under
   (A1), (A3); finitely many strong laws; a.s. is right. **Verified.**
4. *\(\sigma^2=0\):* the \(\eta=0\) case was wrong; explicit two-atom law
   constructed; double root at \(c_b=0\) handled. **Hardened (H1).**
5. *Retention reading:* evaluation law must be \(P_{\theta_0}\); fixed-in-\(\theta\)
   hypothesis met by the frozen map; no hidden centring. **Verified (H2).**
6. *Population references:* the four cuts in \(\hat s\)-space are midpoints of
   the sorted rule centres (nearest-centre prediction with a positive scalar
   whitening); \(\hat s=(u-1)/(0.3u+0.7)\), \(u=e^{-\ell}\), \(\ell=a+bx+cx^2\) the
   fitted logit, so each cut is a quadratic in \(x\); the rung 15 has six
   roots and seven \(x\)-pieces with labels \(2,1,3,0,3,1,2\). With
   \(s(x)f(x)=\phi_{\rm sig}-\phi_{\rm bkg}\), \(p_b\) and \(m_b\) are
   Gaussian-CDF differences; \(v,ES^4,\sigma^2,\tilde\eta\) by QUADPACK and by
   \(200\times64\) composite Gauss–Legendre agree to \(4\cdot10^{-15}\):
   \(\eta=0.8936629669\), \(\sigma=0.2354100747\), \(\tilde\eta=0.9670644809\),
   \(p=(0.49533,0.12149,0.21935,0.16383)\), \(ES^4=2.7335806\). The recorded
   values are reproduced to rounding and the researcher's full-precision
   artifact to \(\le2\cdot10^{-15}\); the rule anchor (surrogate 0.9658175,
   true 0.8847075 on door3's test sample) matches bit for bit. Closed-form
   \(\hat s\) and \(s\) agree with the providers to \(2.5\cdot10^{-15}\); zero
   label disagreements on \(10^6\) draws. **Verified.**
7. *Coverage:* fresh `SeedSequence(20260906)`, 4000 replicates per size.
   Rung 15: coverage 0.918, 0.937, 0.952, 0.946 (SE 0.004), SD\((\hat\eta)\)
   matching \(\sigma/\sqrt n\), \(n\cdot\)bias 0.30, 0.31, 0.12, 0.22, no
   interval outside \([0,1]\), no empty cell, covers \(\tilde\eta\) 0.046 then
   0. **Replicates.** Rung 300 (\(\eta=0.9716\), \(\sigma=0.0532\)): 0.896,
   0.935, 0.940, 0.947 — the small-\(n\) shortfall is larger at the edge, and
   the proxy gap (0.0022) sits inside the interval with frequency 0.94 at
   \(n=100\) and 0.38 at \(n=3000\): O6.5's exclusion is asymptotic, not a
   small-sample fact. Two-atom \(\sigma^2=0\) law: \(\hat\eta\ge\eta\) in every
   replicate, \(n\cdot\)bias 0.50, plug-in studentized statistic with median
   0.60 and 95% quantile 1.25, coverage 0.994–0.9995 with width \(O(1/n)\):
   the unsupported interval is *conservative*. Atomless \(\eta=0\) law:
   \(n\hat\eta\) has mean 2.0 (a \(\chi^2_2\)-type limit), coverage 0.999, the
   interval extends below 0 in 3996/4000 replicates. **Recorded.**
8. *Second-order readings:* population-studentized skew \(-0.30\) at \(n=100\)
   (recorded \(-0.30\)); plug-in studentized skew \(+0.86\) (recorded
   \(+1.06\); sample skewness of 4000 draws is noisy, the sign and order agree);
   \(n\cdot\)bias \(\approx0.3\). **Consistent.**
9. *Literature:* §5 and the dated file. **Method prior art; statement search gap.**
10. *Registry hygiene:* `criterion`, `level`, `dependencies`, `literature` and
    `implies` are consistent with the prose; the `implies` edge into
    `OPEN-RETENTION-UNCERTAINTY` follows the workspace convention for
    "feeds into" and does not overclaim; the `OPEN-RETENTION-UNCERTAINTY`
    patch lists the remainders correctly and now also names the degenerate
    \(\sigma^2=0\) limit. The A4 parenthetical was the one wrong field. **Patched.**

Protocol G items not covered above: *strictness and ties* — irrelevant, the
functional is smooth in moments; *singular information* — excluded by (A3),
refused by the library; *nuisance singularity* — out of scope (scalar);
*hidden compactness* — none; *first-order-to-finite jumps* — none claimed,
measured \(O(1/n)\) bias; *empirical-to-population jumps* — none, the rule
is frozen; *score-estimation error* — enters only through the label map;
*new-event extension* — the label map is the frozen rule's own prediction.
*Heavy tails* — with only \(ES^2<\infty\) the CLT for \(\hat v\) fails; not
relaxed.

## 10. Algorithmic consequence

None for optimisation. For evaluation: a frozen rule plus an oracle-score
evaluation sample from the reference law yields \(\hat\eta\), \(\hat\sigma\) and
a Wald interval from cell counts, cell score sums and score second and
fourth moments — \(O(n)\), no resampling. The interval must not be reported
when \(\hat\eta\in\{0,1\}\) or \(\hat\sigma=0\), and is untrustworthy below
\(n\approx1000\) at the 5% level (0.90–0.92 measured at \(n=100\)).

## 11. Deployability consequence

The audit authorizes removing the "not independently audited" notes. It does
**not** authorize a library uncertainty API: the theorem is conditional on a
frozen rule, requires an oracle score on the evaluation sample, needs the
evaluation law to be the reference law for the retention reading, and is
first-order only. The surrogate plug-in carries the same interval around
\(\tilde\eta\), a different number; the proxy gap is bias, and at a good rung
the gap can sit inside the interval for practical \(n\).

## 12. Information-loss consequence

O6 bounds nothing. It equips the true scalar retention (the D-, A- or
E-efficiency of a one-parameter model) with a conditional \(n^{-1/2}\) error
bar on held-out oracle-score data, and separates evaluation uncertainty from
proxy bias only once an oracle sample exists. No statement about training
retention, vector retention, refitted rules or weighted samples.

## 13. Updated status

- `RETENTION-PLUGIN-CLT-FROZEN-SCALAR`: remains `bridge`, now audited; A4
  and the \(\sigma^2=0\) statement hardened (H1); evaluation-law and
  \(\phi\)-domain assumptions made explicit (H2); `literature_search_status`
  stays `prior_art_found` for the method, the exact statement recorded as a
  search gap in the warning.
- `RETENTION-PLUGIN-COVERAGE-DOOR3`: remains `measured`, now with the
  independent references, the fresh-seed replication, the rung-300 and
  degenerate-law probes.
- `OPEN-RETENTION-UNCERTAINTY`: remains `open`; remainder list gains the
  degenerate \(\sigma^2=0\) limit.
- `AUDIT-SCORE-ORACLE-ROBUSTNESS`: new node, `project_proved`.

## 14. Registry patch

`claims/RETENTION-PLUGIN-CLT-FROZEN-SCALAR.json`: `audit:` pointer,
`boundary_counterexamples`, rewritten `assumptions` (A1–A4 hardened, frozen
map, evaluation law, \(\phi\)-domain, a.s. remark, first-order caveat),
corrected `statement`, warning without "not independently audited".
`claims/RETENTION-PLUGIN-COVERAGE-DOOR3.json`: `audit:` pointer, replication
and probe assumptions, proxy-exclusion caveat.
`claims/OPEN-RETENTION-UNCERTAINTY.json`: remainder list and audit pointer.
`claims/AUDIT-SCORE-ORACLE-ROBUSTNESS.json`: new. `KNOWN_RESULTS/10-oracle.md`
O6 preamble, O6.4 and O6.6 carry the audit notes.

## 15. Regression artifacts

- `py/audit_score_oracle_retention_uncertainty.py` (stages `exact`,
  `popref`, `coverage`, `fixtures`) and
  `AUDITS/artifacts/AUDIT-SCORE-ORACLE-ROBUSTNESS-001/{exact,popref,coverage,fixtures}.json`
  with git revision, script hash, Python and platform.
- `COUNTEREXAMPLES/CE-O6-ETA-ZERO-MULTIATOM-VARIANCE-001.json` and its
  catalogue entry.
- `tests/test_research_claims.py::test_o6_audit_eta_zero_law_has_zero_influence_variance_with_many_atoms`
  and `::test_o6_audit_two_atom_law_has_zero_variance_and_upward_biased_plugin`.
- `NUMERICAL_EVIDENCE.md` rows `N-ORACLE-AUDIT-EXACT`, `-DEGENERATE`,
  `-POPREF`, `-COVERAGE`.

## 16. Next dependency-blocking question

Verified, so the packet's stated successor applies: the vector case of
`OPEN-RETENTION-UNCERTAINTY` — the geometric-mean retention
\((\det I_Z/\det V)^{1/d}\) under a frozen rule as a smooth matrix functional
of the same cell moments, its matrix influence function, the new hypothesis
\(I_Z\succ0\) (which needs \(K\ge d+1\) under \(E[S]=0\)), and the endpoints
\(\eta_D\in\{0,1\}\) treated separately from the start. Packet:
`WORK/completed/RETENTION-PLUGIN-VECTOR.md`.
