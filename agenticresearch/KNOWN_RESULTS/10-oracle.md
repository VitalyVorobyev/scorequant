# 10. Score/density-ratio/classifier access

> Part of the ScoreQuant known-results ledger. Read `PROBLEM.md` first and
> `KNOWN_RESULTS/index.md` for the status vocabulary and the chapter map.
> Resolve any claim id with `python py/registry.py show <ID> --deps --proof`.

## O1. Density ratios suffice for local scores — [BRIDGE]

**Claims:** RATIO-LOCAL-SCORE

\[
s(x)=\left.\nabla_\theta\log\frac{p(x\mid\theta)}{p(x\mid\theta_0)}\right|_{\theta_0}.
\]

Full absolute densities are not required if the relevant local density ratio is available.

## O2. Linear-mixture component ratios suffice — [BRIDGE]

**Claims:** MIXTURE-RATIO-SCORE

For

\[
p(x\mid\theta)=\sum_\alpha\theta_\alpha\phi_\alpha(x),
\]

score coordinates depend on \(\phi_\alpha(x)/\sum_\beta\theta_{0\beta}\phi_\beta(x)\). Ratios to one reference component therefore suffice exactly after algebraic reconstruction.

## O3. Calibrated classifier posteriors provide ratios — [LIT/BRIDGE]

**Claims:** CLASSIFIER-MIXTURE-SCORE-FORMULA, CLASSIFIER-RATIO-ORACLE

With class priors \(\pi_\alpha\), posterior odds recover component density ratios. In the mixture parameterization,

\[
\boxed{
s_\alpha(x)=
\frac{\eta_\alpha(x)/\pi_\alpha}
{\sum_\beta\theta_{0\beta}\eta_\beta(x)/\pi_\beta}.
}
\]

Estimated classifiers solve the exact score problem only to the extent that they recover calibrated ratios.

## O4. True retained FI under an estimated score — [BRIDGE]

**Claims:** PROXY-TRUE-RETAINED-FI

If the quantizer uses \(\hat s\), the actual retained Fisher information is

\[
\boxed{
\operatorname{Var}(E[s\mid q(\hat s)]),
}
\]

not \(\operatorname{Var}(E[\hat s\mid q(\hat s)])\) unless \(\hat s=s\) in the relevant sense.

## O5. Representation loss and quantization loss separate — [BRIDGE]

**Claims:** REPRESENTATION-QUANTIZATION-LOSS

For a representation \(R(X)\),

\[
I_R=\operatorname{Var}(E[s\mid R]),
\qquad
I_q\preceq I_R\preceq I_{\rm full}.
\]

This separates oracle/representation loss from hard-quantization loss whenever truth scores are available for validation.

---

## O6. Frozen-rule scalar retention: plug-in asymptotics and a consistent variance — [BRIDGE]

**Claims:** RETENTION-PLUGIN-CLT-FROZEN-SCALAR, RETENTION-PLUGIN-COVERAGE-DOOR3

*Recorded 5 September 2026 by the SCORE-ORACLE-ROBUSTNESS session (packet
packet `SCORE-ORACLE-ROBUSTNESS` (git history)). Verdict: **proved**, as a direct
corollary of the delta method; the special case of OP27 it settles is stated
exactly below and the broad claim `OPEN-RETENTION-UNCERTAINTY` stays open.
Instrument `py/score_oracle_retention_uncertainty.py`; artifacts under
`WORK/artifacts/SCORE-ORACLE-ROBUSTNESS/`. **Independently audited 5 September 2026**
(`AUDITS/AUDIT-SCORE-ORACLE-ROBUSTNESS-001.md`, instrument
`py/audit_score_oracle_retention_uncertainty.py`): O6.1–O6.3 and the Wald statement of
O6.4 verified; the \(\sigma^2=0\) characterisation hardened at \(\eta=0\) (see the
audit notes in O6.4); population references and coverage replicated with fresh seeds.*

### O6.0 Normalized target (protocol A)

- **Level:** `information_accounting`, conditional on a frozen
  `empirical_inductive_quantizer`. **Criterion:** the scalar retention ratio
  (for one score coordinate the D-, A- and E-efficiencies coincide with it).
- **Frozen:** training data, provider \(\hat s\), rule \(q\), reference point
  \(\theta_0\), finite \(K\). **Random:** an independent evaluation sample
  \(X_1,\dots,X_n\) iid from the reference law \(P\), equally weighted.
- **Observables:** \(S_i=s(X_i)\) (true scalar score) and
  \(Z_i=q(\hat s(X_i))\in\{1,\dots,K\}\). Because \(q\circ\hat s\) is a fixed
  measurable map, the pairs \((S_i,Z_i)\) are iid with a fixed joint law: the
  boundary non-smoothness of OP27 never enters this special case.
- **Population target:**
  \[
  p_b=P(Z=b),\qquad m_b=E[S\,\mathbf 1_{Z=b}],\qquad v=E[S^2],\qquad
  \eta=\frac{\sum_b m_b^2/p_b}{v},\qquad c_b=\frac{m_b}{p_b}.
  \]
  Under a regular model with \(E[S]=0\), \(v=I_{\rm full}\) and
  \(\sum_b m_b^2/p_b=I_Z\) (U1, `FI-QUANT-IDENTITY`, with \(q\) fixed in
  \(\theta\)), so \(\eta=\operatorname{Var}(E[s\mid q(\hat s)])/\operatorname{Var}(s)\)
  is exactly the *true* retained fraction of O4 (`PROXY-TRUE-RETAINED-FI`).
- **Estimator:** the ordinary plug-in ratio on the same evaluation sample,
  \(\hat p_b=n_b/n\), \(\hat m_b=n^{-1}\sum_i S_i\mathbf 1_{Z_i=b}\),
  \(\hat v=n^{-1}\sum_i S_i^2\), \(\hat\eta=\sum_b\hat m_b^2/\hat p_b\,/\,\hat v\),
  with \(0/0:=0\) for an empty cell. Scores are never centred.
- **Excluded (packet scope):** importance weights, growing \(K\), refitting the
  rule on the evaluation sample, boundary stability, \(D_s\), classifier
  calibration theory, bootstrap comparisons, any public uncertainty API.

**Assumptions.**
(A1) \(p_b>0\) for every \(b\);
(A2) \(E[S^4]<\infty\);
(A3) \(v=E[S^2]>0\);
(A4) \(\sigma^2>0\) with \(\sigma^2\) defined in O6.2.
(A2) is automatic for bounded scores, e.g. every mixture-fraction score at an
interior reference point (\(|s|\le\max_k 1/\theta_{0k}\)); (A3) is the
nonsingular unbinned-information hypothesis the library already imposes.

### O6.1 Finite-sample identity — [BRIDGE]

For every sample, with \(\hat c_b=\hat m_b/\hat p_b\) on nonempty cells,

\[
\boxed{\;
\hat\eta \;=\; 1-\frac{\sum_i\big(S_i-\hat c_{Z_i}\big)^2}{\sum_i S_i^2}
\;}
\]

so \(\hat\eta\) is one minus the within-cell residual sum of squares about the
cell means over the *uncentred* total sum of squares, and \(0\le\hat\eta\le1\)
always. Empty cells contribute nothing to either side, which is what the
\(0/0:=0\) convention encodes. For scalar scores \(\hat\eta\) coincides with the
public `information_report(...).geometric_mean_retention` (the whitening
divides by \(\hat v\); selftest agreement \(5\cdot10^{-17}\)).

*Proof.* \(\sum_i(S_i-\hat c_{Z_i})^2=\sum_b\big[\sum_{i\in b}S_i^2-n_b\hat c_b^2\big]
=\sum_iS_i^2-\sum_b n_b\hat c_b^2\) and \(n_b\hat c_b^2=(\sum_{i\in b}S_i)^2/n_b
=n\,\hat m_b^2/\hat p_b\). Divide by \(\sum_iS_i^2=n\hat v\). \(\square\)

### O6.2 Conditional asymptotic normality — [BRIDGE]

Under (A1)–(A3), conditionally on the frozen rule,

\[
\boxed{\;
\sqrt n\,(\hat\eta-\eta)\;\Rightarrow\;N(0,\sigma^2),\qquad
\sigma^2=E[\psi(S,Z)^2],\qquad
\psi(S,Z)=\frac{(1-\eta)\,S^2-(S-c_Z)^2}{v}.
\;}
\]

Equivalently, writing the numerator influence \(N_1=2c_ZS-c_Z^2\) (so that
\(E[N_1]=\eta v\)) and the denominator influence \(S^2\),

\[
\sigma^2=\frac{\operatorname{Var}(N_1)-2\eta\operatorname{Cov}(N_1,S^2)+\eta^2\operatorname{Var}(S^2)}{v^2},
\]

which is the numerator–denominator covariance form the packet asked for.

*Proof.* Let \(T_i=(\mathbf 1_{Z_i=b},\,S_i\mathbf 1_{Z_i=b},\,S_i^2)_{b=1..K}\in\mathbb R^{2K+1}\),
\(\theta=E[T_1]=(p_b,m_b,v)_b\). By (A2) \(T_1\) has finite second moments (the
largest is \(E[S^4]\)), so the multivariate CLT gives
\(\sqrt n(\bar T-\theta)\Rightarrow N(0,\Sigma)\), \(\Sigma=\operatorname{Cov}(T_1)\).
The map \(g(p,m,v)=\sum_b m_b^2/p_b\,/\,v\) is \(C^\infty\) on
\(\{p_b>0\ \forall b,\ v>0\}\), which contains \(\theta\) by (A1), (A3). The delta
method (van der Vaart 1998, Thm 3.1) yields
\(\sqrt n(g(\bar T)-g(\theta))\Rightarrow N(0,\nabla g^\top\Sigma\nabla g)\), and
\(\nabla g^\top\Sigma\nabla g=\operatorname{Var}(\nabla g^\top T_1)\). With

\[
\frac{\partial g}{\partial p_b}=-\frac{m_b^2}{p_b^2v},\qquad
\frac{\partial g}{\partial m_b}=\frac{2m_b}{p_bv},\qquad
\frac{\partial g}{\partial v}=-\frac{\eta}{v},
\]

\[
\nabla g^\top(T_1-\theta)
=\sum_b\Big[-\frac{m_b^2}{p_b^2v}(\mathbf 1_{Z=b}-p_b)+\frac{2m_b}{p_bv}(S\mathbf 1_{Z=b}-m_b)\Big]-\frac{\eta}{v}(S^2-v)
=\frac{2c_ZS-c_Z^2-\eta S^2}{v}+(\eta-2\eta+\eta)=\psi(S,Z),
\]

and \(2c_ZS-c_Z^2=S^2-(S-c_Z)^2\) gives the boxed form; \(E[\psi]=0\) is the
vanishing of the constant terms, so \(\operatorname{Var}(\psi)=E[\psi^2]\).
Finally \(\hat\eta=g(\bar T)\) on the event that every cell is nonempty, whose
complement has probability at most \(\sum_b(1-p_b)^n\to0\) by (A1); hence
\(\hat\eta\) and \(g(\bar T)\) share the limit law. \(\square\)

### O6.3 A consistent implementable variance — [BRIDGE]

Let \(\hat\psi_i=\big((1-\hat\eta)S_i^2-(S_i-\hat c_{Z_i})^2\big)/\hat v\) and
\(\hat\sigma^2=n^{-1}\sum_i\hat\psi_i^2\). Then \(\sum_i\hat\psi_i=0\) exactly
(by O6.1), and under (A1)–(A3) \(\hat\sigma^2\to\sigma^2\) almost surely.

*Proof.* Expanding, \(n^{-1}\sum_i\hat\psi_i^2\) is a fixed polynomial in
\((\hat\eta,\hat v^{-1},\hat c_1,\dots,\hat c_K)\) whose coefficients are the
within-cell empirical moments \(\hat M_{b,k}=n^{-1}\sum_iS_i^k\mathbf 1_{Z_i=b}\),
\(k=0,\dots,4\). By (A2) and the strong law each \(\hat M_{b,k}\to M_{b,k}=E[S^k\mathbf 1_{Z=b}]\)
a.s.; by (A1) \(\hat p_b=\hat M_{b,0}\to p_b>0\) so \(\hat c_b\to c_b\); by (A3)
\(\hat v\to v>0\); hence \(\hat\eta\to\eta\). The polynomial is continuous at the
limit point, and its value there is \(E[\psi^2]\) expanded in the same moments.
\(\square\)

### O6.4 Wald interval — [BRIDGE]

Under (A1)–(A4), \(P\big(\eta\in\hat\eta\pm z_{1-\alpha/2}\,\hat\sigma/\sqrt n\big)\to1-\alpha\)
(O6.2, O6.3 and Slutsky). **Unsupported cases and degeneracies:**

- *(A4) fails* iff \(\psi=0\) a.s., i.e. iff for every \(b\) with \(p_b>0\) the
  conditional law of \(S\) given \(Z=b\) is supported on the roots of
  \((s-c_b)^2=(1-\eta)s^2\). **[Audit 5 Sep 2026, hardened.]** For \(0<\eta<1\)
  the roots are \(s_\pm=c_b/(1\mp\sqrt{1-\eta})\), so at most two atoms per cell
  (one, \(s=0\), when \(c_b=0\)); the mean constraint fixes the upper-root weight
  \(w=\eta/(2(1+\sqrt{1-\eta}))\) and forces \(E[S^2\mid Z=b]=c_b^2/\eta\), and such
  laws exist (cells \(\{2/3,2\}\) and \(\{-2/3,-2\}\) with weights \((3/4,1/4)\),
  \(p_b=1/2\): \(\eta=3/4\), \(E[S]=0\), \(\sigma^2=0\)). At \(\eta=1\) the single
  root is \(s=c_b\) (\(S=c_Z\) a.s.). At \(\eta=0\) every \(c_b=0\) and the equation is
  the identity \(s^2=s^2\): *every* law with vanishing cell means has \(\psi\equiv0\),
  with any number of atoms or none (`CE-O6-ETA-ZERO-MULTIATOM-VARIANCE-001`; the
  atomless probe in the audit's `coverage.json`). Hence: an atomless cell of positive
  probability implies (A4) **iff \(\eta>0\)**; the original sentence omitted the
  qualifier. Where (A4) fails the first-order limit is degenerate, \(n(\hat\eta-\eta)\)
  has a non-normal quadratic-form limit (not derived here), and the Wald interval is
  not supported: measured, it is *conservative* there (coverage \(\to1\) with width
  \(O(1/n)\), \(\hat\eta\ge\eta\) exactly in the two-atom law), not liberal.
- *Empty evaluation cells:* handled by \(0/0:=0\); their probability vanishes
  exponentially under (A1). No uniformity over small \(p_b\) is claimed: a cell
  with \(p_b\) of order \(1/n\) is outside the theorem.
- *Edges:* the interval may leave \([0,1]\) near \(\eta\in\{0,1\}\); no
  transformation was attempted (one estimator, one interval, per packet).
- *No finite-sample guarantee:* coverage below is evidence of the first-order
  statement only.

### O6.5 What the interval does and does not measure — [BRIDGE]

The interval is for the **true** retention \(\eta\), conditional on the frozen
rule. The library's self-reported surrogate replaces \(S\) by \(\hat s\); its
population value \(\tilde\eta=\sum_b\tilde m_b^2/p_b/\tilde v\) with
\(\tilde m_b=E[\hat s\mathbf 1_{Z=b}]\), \(\tilde v=E[\hat s^2]\) is a *different
number* (O4). The same theorem applies verbatim to the surrogate plug-in as an
estimator of \(\tilde\eta\), so the surrogate carries a valid interval around the
wrong target. The gap \(\tilde\eta-\eta\) is bias of the proxy, of order one in
\(n\), and the \(O(n^{-1/2})\) interval excludes \(\tilde\eta\) with probability
\(\to1\): evaluation uncertainty and proxy discrepancy separate cleanly once an
oracle-score evaluation sample exists. Without an oracle sample nothing here
applies (OP27's remainder, and OP17/OP18).

### O6.6 Self-adversarial pass (protocol G)

- *Ties, duplicate scores:* irrelevant; the functional is smooth in moments.
- *Singleton/empty cells:* O6.4; under (A1) transient.
- *Singular information:* excluded by (A3); with \(v=0\) the target is
  undefined, matching the library's refusal.
- *Nuisance singularity, \(D_s\):* out of scope (scalar, no nuisance).
- *Atomic laws:* the only route to (A4) failure with \(0<\eta<1\); recorded above.
  [Audit: at \(\eta=0\) atomless laws also fail (A4); see O6.4.]
- *Hidden compactness:* none used.
- *First-order-to-finite jumps:* none claimed; measured \(O(1/n)\) bias below.
- *Empirical-to-population jumps:* none — the rule is frozen, the sample is
  independent of it; a rule refitted on the evaluation sample voids the iid
  structure and is exactly what OP27 still owes.
- *Score-estimation error:* enters only through the label map and the
  target's definition; the theorem is conditional on \(\hat s\).
- *Heavy tails:* (A2) is used for the CLT of \(S^2\) and the SLLN of fourth
  within-cell moments; with only \(E[S^2]<\infty\) the denominator CLT fails.

### O6.7 Measured (protocol D, run before the proof was trusted) — [MEASURED]

Frozen rule: door3 rung `n_per_class = 15` (classifier seed 101, four cells,
D-exchange seed 7, the rung with the largest published proxy gap). The
instrument reproduces the published ladder on door3's own test sample
(surrogate 0.9658, true 0.8847). Population references by two independent
composite Gauss–Legendre routes (disagreement \(1.1\cdot10^{-16}\), tail
truncation \(<10^{-18}\), closed-form score vs the exact provider
\(9\cdot10^{-16}\)); \(E[S]=-4.5\cdot10^{-17}\) confirms the zero-mean score.

| quantity | value |
|---|---|
| \(\eta\) (true retention) | 0.893663 |
| \(\sigma\) | 0.235410 |
| \(\tilde\eta\) (proxy population value) | 0.967064 |
| proxy gap \(\tilde\eta-\eta\) | 0.073402 |
| cell probabilities | 0.4953, 0.1215, 0.2193, 0.1638 |
| \(E[S^4]\) | 2.7336 |

Coverage of the 95% Wald interval, 2000 independent evaluation samples per
size, `SeedSequence(20260905)`:

| \(n\) | coverage ± SE | SD\((\hat\eta)\) | \(\sigma/\sqrt n\) | mean \(\hat\sigma/\sqrt n\) | rel. RMSE \(\hat\sigma/\sigma\) | \(n\cdot\)bias | studentized skew | covers \(\tilde\eta\) |
|---|---|---|---|---|---|---|---|---|
| 100 | 0.913 ± 0.006 | 0.02335 | 0.02354 | 0.02255 | 0.167 | 0.33 | +1.06 | 0.046 |
| 300 | 0.934 ± 0.006 | 0.01356 | 0.01359 | 0.01342 | 0.088 | 0.35 | +0.20 | 0.000 |
| 1000 | 0.953 ± 0.005 | 0.00738 | 0.00744 | 0.00742 | 0.049 | 0.18 | +0.11 | 0.000 |
| 3000 | 0.949 ± 0.005 | 0.00433 | 0.00430 | 0.00429 | 0.029 | 0.40 | +0.12 | 0.000 |

Reading: the sampling spread matches \(\sigma/\sqrt n\) at every size and
\(\hat\sigma\) converges to \(\sigma\); coverage reaches nominal by
\(n=1000\). The \(n=100\) shortfall is a second-order effect — the statistic
studentized with the *population* \(\sigma\) has skew \(-0.30\) and SD 0.99,
while the plug-in studentized statistic has skew \(+1.06\): \(\hat\sigma\) is
small precisely when \(\hat\eta\) is high. The measured bias is \(O(1/n)\)
(\(n\cdot\)bias \(\approx0.3\)). No interval left \([0,1]\); no evaluation cell
was empty. The interval for \(\eta\) essentially never covers the proxy value
\(\tilde\eta\), as O6.5 predicts. None of this is a proof of O6.2–O6.4, and
nothing is claimed about coverage below \(n=1000\) beyond the numbers shown.

### O6.8 Information-loss implication and verdict (protocol H)

O6 bounds nothing; it equips the true scalar retention \(\eta\) (equivalently
the D-, A- or E-efficiency of a one-parameter model) with a conditional
\(n^{-1/2}\) error bar on held-out oracle-score data. **Verdict: proved**, as a
bridge from the delta method, with the following left open and rerouted to
`OPEN-RETENTION-UNCERTAINTY`: vector scores and the geometric-mean retention
\((\det R)^{1/d}\) (a smooth matrix functional of the same moments — the
matrix influence function is the missing derivation); refitted rules, where
the evaluation sample enters the boundaries; weighted samples; and any
statement without an oracle score.

## O7. Frozen-rule vector retention: plug-in asymptotics of the geometric-mean retention — [BRIDGE]

**Claims:** RETENTION-PLUGIN-CLT-FROZEN-VECTOR, RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE, RETENTION-PLUGIN-COVERAGE-VECTOR

*Recorded 6 September 2026 by the RETENTION-PLUGIN-VECTOR session (packet
packet `RETENTION-PLUGIN-VECTOR` (git history)). Verdict: **proved**, as a bridge
from the vector delta method and the determinant differential; the vector
special case of OP27 it settles is stated exactly below and
`OPEN-RETENTION-UNCERTAINTY` stays open for refitted rules, weights and the
no-oracle case. Instrument `py/retention_plugin_vector.py`; artifacts under
`WORK/artifacts/RETENTION-PLUGIN-VECTOR/`. **Independently audited 6 September 2026**
(`AUDITS/AUDIT-RETENTION-PLUGIN-VECTOR-001.md`): verified with hardened assumptions —
O7.1–O7.3, the Wald statement under (A4), the ellipsoid characterisation and the
endpoint rate hold; H1 corrects the \(\eta_D=1\) sentence (singular \(\hat V\) samples,
`CE-O7-UNIT-RETENTION-SINGULAR-SAMPLE-001`), H2 reduces the Wald "iff" to "if", H3 marks
the endpoint-coverage convergence as measured, H4 names the degenerate ellipsoid.*

**Cite versus derive.** The literature-first pass of 5 September 2026
(`LITERATURE/audits/RETENTION-PLUGIN-CLT-FROZEN-VECTOR-5-September-2026.md`)
found the *method* published: the vector delta method (van der Vaart 1998,
Thm 3.1), the determinant and inverse differentials (Magnus & Neudecker
§8.3–8.4), influence functions of \(\Lambda\)-type and canonical-correlation
parameters (Radhakrishnan & Kshirsagar 1981; Romanazzi 1992) and asymptotic
normality of eigenvalue functionals under finite fourth moments, repeated
eigenvalues allowed (Muirhead & Waternaux 1980; Fang & Krishnaiah 1982). None
of it is re-proved here. What O7 adds is project algebra inside that method:
the uncentred fixed-partition form on the cell moments the library actually
computes, the explicit \(\psi\) and its \(d=1\) reduction, the plug-in variance
and its consistency, and the endpoints — where the vector case genuinely
differs from O6. A search gap, not novelty.

### O7.0 Normalized target (protocol A)

- **Level:** `information_accounting`, conditional on a frozen
  `empirical_inductive_quantizer`. **Criterion:** D (the geometric-mean
  retention `INFO-D-EFFICIENCY`).
- **Frozen:** training data, provider \(\hat s\), rule \(q\), reference point
  \(\theta_0\), finite \(K\). **Random:** an independent evaluation sample
  \(X_1,\dots,X_n\) iid from the evaluation law \(P\), equally weighted.
- **Observables:** \(S_i=s(X_i)\in\mathbb R^d\) (true score) and
  \(Z_i=q(\hat s(X_i))\in\{1,\dots,K\}\); as in O6 the pairs \((S_i,Z_i)\) are
  iid because \(q\circ\hat s\) is a fixed measurable map.
- **Population target:**
  \[
  p_b=P(Z=b),\quad m_b=E[S\mathbf 1_{Z=b}],\quad c_b=\frac{m_b}{p_b},\quad
  V=E[SS^\top],\quad I_Z=\sum_b\frac{m_bm_b^\top}{p_b},\quad
  \eta_D=\Big(\frac{\det I_Z}{\det V}\Big)^{1/d}.
  \]
  Under a regular model at the reference law (\(E[S]=0\)), \(V=I_{\rm full}\)
  and \(I_Z=\operatorname{Var}(E[s\mid q(\hat s)])\) (U1 with \(q\) fixed in
  \(\theta\), `FI-QUANT-IDENTITY`; O4, `PROXY-TRUE-RETAINED-FI`), so
  \(\eta_D=(\det R)^{1/d}\) with \(R=V^{-1/2}I_ZV^{-1/2}\) is exactly the true
  geometric-mean retention I1 (`INFO-D-EFFICIENCY`). Both \(\eta_D\) and the
  influence function below are invariant under \(S\mapsto AS\), \(A\) nonsingular
  (`D-REPARAM-INVARIANCE`), so nothing here depends on the parametrisation of \(\theta\).
- **Classical reading (stated once).** With \(M=E[S\mathbf 1_Z^\top]\) and
  \(P=\operatorname{diag}(p)\), \(I_Z=MP^{-1}M^\top\) and
  \(\det I_Z/\det V=\det(V^{-1}MP^{-1}M^\top)=\prod_{i\le d}\rho_i^2\): \(\eta_D\)
  is the geometric mean of the squared *uncentred* canonical correlations
  between \(S\) and the cell indicator \(\mathbf 1_Z\). At the reference law
  the uncentred and centred objects coincide, \(I_Z\) is the between-cell
  matrix, \(V\) the total matrix, and \(\prod_i(1-\rho_i^2)=\det(V-I_Z)/\det V\)
  is Wilks' \(\Lambda\) for the grouping \(Z\). O7 is therefore the
  fixed-alternative first-order theory of a MANOVA determinant ratio under
  finite fourth moments and no normality, with the grouping a fixed map of a
  different variable.
- **Estimator:** the plug-in on the same evaluation sample,
  \(\hat p_b=n_b/n\), \(\hat m_b=n^{-1}\sum_iS_i\mathbf 1_{Z_i=b}\),
  \(\hat V=n^{-1}\sum_iS_iS_i^\top\), \(\hat I_Z=\sum_{b:\,n_b>0}\hat m_b\hat m_b^\top/\hat p_b\),
  \(\hat\eta_D=(\det\hat I_Z/\det\hat V)^{1/d}\) with \(0/0:=0\) on empty
  cells and \(\hat\eta_D:=0\) when \(\det\hat V=0\). Scores are never centred.
- **Excluded (packet scope):** importance weights, growing \(K\), rules
  refitted on the evaluation sample, \(D_s\)/profiled retention, classifier
  calibration, bootstrap comparisons, any public uncertainty API.

**Assumptions.**
(A1) \(p_b>0\) for every \(b\);
(A2) \(E\|S\|^4<\infty\);
(A3) \(V\succ0\);
(A3′) \(I_Z\succ0\);
(A4) \(\sigma^2>0\) with \(\sigma^2\) defined in O7.2.
(A2) is automatic for bounded scores (mixture-fraction scores at an
interior reference point); (A3) is the library's nonsingular-information
hypothesis; (A3′) is new to the vector case — by U4 (`FI-RANK-CEILING`) it
needs \(K\ge d+1\) at the reference law, and it fails exactly at the
singular endpoint treated in O7.4(b).

### O7.1 Finite-sample identity and library agreement — [BRIDGE]

For every sample, with \(\hat c_b=\hat m_b/\hat p_b\) on nonempty cells,

\[
\boxed{\;
\hat V-\hat I_Z=\frac1n\sum_i\big(S_i-\hat c_{Z_i}\big)\big(S_i-\hat c_{Z_i}\big)^\top\succeq0,
\qquad 0\preceq\hat I_Z\preceq\hat V,\qquad 0\le\hat\eta_D\le1.
\;}
\]

*Proof.* \(\sum_i(S_i-\hat c_{Z_i})(S_i-\hat c_{Z_i})^\top=\sum_iS_iS_i^\top-\sum_bn_b\hat c_b\hat c_b^\top\)
because the cross terms in cell \(b\) sum to \(n_b\hat c_b\hat c_b^\top\), and
\(n_b\hat c_b\hat c_b^\top=n\,\hat m_b\hat m_b^\top/\hat p_b\). Empty cells
contribute to neither side (the \(0/0:=0\) convention). \(\det\) is monotone on
the positive-semidefinite order, so \(0\le\det\hat I_Z\le\det\hat V\). This is
Kendall & Stuart's decomposition (26.50) in uncentred matrix form. \(\square\)

**Rank.** \(\operatorname{rank}\hat I_Z\le\min(d,K)\) on any sample, and
\(\le\min(d,K-1)\) on an exactly centred sample (\(\sum_b\hat m_b=0\)); at the
reference law \(\operatorname{rank}I_Z\le\min(d,K-1)\) (U4). So \(K\le d-1\)
gives \(\hat\eta_D=0\) exactly, while \(K=d\) gives a population value \(0\)
with a generically positive plug-in — O7.4(b).

**Library agreement.** `information_report(S, Z).geometric_mean_retention`
whitens by \(\hat V^{-1/2}\) on the retained eigen-subspace of \(\hat V\) and
returns \(\exp(\log\det/\mathrm{rank})\) (`reports.py`, `_report_from_fishers`).
When \(\hat V\succ0\) and `rank_rtol` retains every direction the whitening
cancels in the ratio, \(\det(\hat V^{-1/2}\hat I_Z\hat V^{-1/2})=\det\hat I_Z/\det\hat V\),
and the library number *is* \(\hat\eta_D\) (measured agreement \(1.1\cdot10^{-16}\) at
\(d=2\), \(4\cdot10^{-17}\) at \(d=3\)). Two caveats outside the theorem: (i) when
`rank_rtol` projects a direction out, the library reports the ratio on the
retained subspace with exponent \(1/\mathrm{rank}\) — a different, discontinuous
estimator whose target (A3) excludes; (ii) the \(d\)-th root amplifies
rounding at an exactly singular \(\hat I_Z\): the exact plug-in is \(0\) but the
library returns \(\approx\varepsilon^{1/d}\) (measured \(1.6\cdot10^{-9}\) at
\(d=2\) for an exact zero).

### O7.2 Conditional asymptotic normality — [BRIDGE]

Under (A1)–(A3′), conditionally on the frozen rule,

\[
\boxed{\;
\sqrt n\,(\hat\eta_D-\eta_D)\;\Rightarrow\;N(0,\sigma^2),\qquad
\sigma^2=E[\psi(S,Z)^2],\qquad
\psi(S,Z)=\frac{\eta_D}{d}\Big[\,2S^\top I_Z^{-1}c_Z-c_Z^\top I_Z^{-1}c_Z-S^\top V^{-1}S\,\Big].
\;}
\]

Writing \(N_1=2S^\top I_Z^{-1}c_Z-c_Z^\top I_Z^{-1}c_Z\) (the influence of
\(\log\det I_Z\), \(E[N_1]=d\)) and \(N_2=S^\top V^{-1}S\) (the influence of
\(\log\det V\), \(E[N_2]=d\)),

\[
\sigma^2=\frac{\eta_D^2}{d^2}\Big[\operatorname{Var}N_1-2\operatorname{Cov}(N_1,N_2)+\operatorname{Var}N_2\Big],
\]

the numerator–denominator covariance form. At \(d=1\), \(I_Z=\eta v\) gives
\(\psi=(2c_ZS-c_Z^2-\eta S^2)/v=((1-\eta)S^2-(S-c_Z)^2)/v\), exactly O6.2.

*Proof.* Let \(T=(\mathbf 1_{Z=b},\,S\mathbf 1_{Z=b},\,\operatorname{vech}SS^\top)_{b\le K}\)
and \(\theta=E[T]=(p,M,V)\). By (A2) \(T\) has finite second moments, so
\(\sqrt n(\bar T-\theta)\Rightarrow N(0,\operatorname{Cov}T)\). Define the
estimator's own functional on the whole space,
\(\phi(p,M,V)=\big(\det\sum_{b:p_b>0}m_bm_b^\top/p_b\,/\det V\big)^{1/d}\) with
\(\phi:=0\) where \(\det V=0\) or the determinant ratio is \(\le0\); then
\(\hat\eta_D=\phi(\bar T)\) for *every* sample, empty cells included. On the
open set \(\{p_b>0\ \forall b,\ V\succ0,\ I_Z\succ0\}\), which contains \(\theta\)
by (A1), (A3), (A3′), \(\phi\) coincides with the \(C^\infty\) function
\(g=\exp\{d^{-1}[\log\det I_Z-\log\det V]\}\), so \(\phi\) is differentiable at
\(\theta\) and the delta method (van der Vaart Thm 3.1) gives
\(\sqrt n(\phi(\bar T)-\phi(\theta))\Rightarrow N(0,\operatorname{Var}(\nabla g^\top T))\).
The gradient: by \(d\log\det A=\operatorname{tr}A^{-1}dA\) (Magnus & Neudecker
§8.3) and \(dI_Z=\sum_b[(dm_b)m_b^\top+m_b(dm_b)^\top]/p_b-\sum_bm_bm_b^\top dp_b/p_b^2\),

\[
d\log\det I_Z=\sum_b\big[2c_b^\top I_Z^{-1}dm_b-c_b^\top I_Z^{-1}c_b\,dp_b\big],\qquad
d\log\det V=\operatorname{tr}(V^{-1}dV).
\]

Substituting the increments \(dp_b=\mathbf 1_{Z=b}-p_b\), \(dm_b=S\mathbf 1_{Z=b}-m_b\),
\(dV=SS^\top-V\) of one observation,

\[
\nabla g^\top(T-\theta)=\frac{\eta_D}{d}\Big[2S^\top I_Z^{-1}c_Z-c_Z^\top I_Z^{-1}c_Z-S^\top V^{-1}S
-\big(2\textstyle\sum_bm_b^\top I_Z^{-1}c_b-\sum_bp_bc_b^\top I_Z^{-1}c_b-\operatorname{tr}V^{-1}V\big)\Big],
\]

and the bracketed constant is \(2d-d-d=0\) because
\(\sum_bm_bc_b^\top=\sum_bp_bc_bc_b^\top=I_Z\). Hence \(\nabla g^\top(T-\theta)=\psi\),
\(E[\psi]=0\) by the same three identities, and \(\operatorname{Var}\psi=E[\psi^2]\).
\(\square\)

**Canonical-correlation form (a check, not a second proof).** With
\(V\)-orthonormal canonical directions \(a_i\) of \(S\) and \(P\)-orthonormal
directions \(b_i\) of \(\mathbf 1_Z\) (\(Mb_i=\rho_iVa_i\)), put \(u_i=a_i^\top S\),
\(v_i=b_{i,Z}\). Then \(S^\top V^{-1}S=\sum_iu_i^2\), \(c_Z^\top I_Z^{-1}c_Z=\sum_iv_i^2\)
and \(S^\top I_Z^{-1}c_Z=\sum_iu_iv_i/\rho_i\), so

\[
\psi=\frac{\eta_D}{d}\sum_{i\le d}\frac{-\rho_i^2u_i^2+2\rho_iu_iv_i-\rho_i^2v_i^2}{\rho_i^2}
=\frac{\eta_D}{d}\sum_i\frac{\operatorname{IF}(\rho_i^2)}{\rho_i^2},
\]

the average of the per-coefficient influence functions of Romanazzi (1992)
in the form restated by his citers (`LITERATURE/audits/…VECTOR-5-September-2026.md`,
post-derivation check). The per-coefficient pieces need simple \(\rho_i\);
the determinant route above does not, and the sum is well defined with
repeated \(\rho_i\) (Fang & Krishnaiah 1982 treat exactly that case).

### O7.3 A consistent implementable variance — [BRIDGE]

Let \(\hat\psi_i=\frac{\hat\eta_D}{d}\big[2S_i^\top\hat I_Z^{-1}\hat c_{Z_i}-\hat c_{Z_i}^\top\hat I_Z^{-1}\hat c_{Z_i}-S_i^\top\hat V^{-1}S_i\big]\)
and \(\hat\sigma^2=n^{-1}\sum_i\hat\psi_i^2\). Then \(\sum_i\hat\psi_i=0\)
exactly whenever \(\hat I_Z\succ0\) and \(\hat V\succ0\), and under (A1)–(A3′)
\(\hat\sigma^2\to\sigma^2\) almost surely.

*Proof.* The three sample identities \(\sum_iS_i^\top\hat I_Z^{-1}\hat c_{Z_i}=n\operatorname{tr}(\hat I_Z^{-1}\sum_b\hat m_b\hat c_b^\top)=nd\),
\(\sum_i\hat c_{Z_i}^\top\hat I_Z^{-1}\hat c_{Z_i}=nd\), \(\sum_iS_i^\top\hat V^{-1}S_i=nd\)
give the exact zero. \(\hat\sigma^2\) is a fixed polynomial in \(\hat\eta_D\), the
entries of \(\hat I_Z^{-1}\), \(\hat V^{-1}\) and \(\hat c_1,\dots,\hat c_K\), whose
coefficients are within-cell empirical moments \(n^{-1}\sum_iS_{i,j}S_{i,k}S_{i,l}S_{i,m}\mathbf 1_{Z_i=b}\)
of order \(\le4\). By (A2) and the strong law every such moment converges a.s.;
by (A1) \(\hat p_b\to p_b>0\), so \(\hat c_b\to c_b\); by (A3), (A3′) \(\hat V\to V\succ0\)
and \(\hat I_Z\to I_Z\succ0\), so the inverses and \(\hat\eta_D\) converge; the
polynomial is continuous at the limit and its value there is \(E[\psi^2]\).
Empty cells are a.s. transient by Borel–Cantelli under (A1). \(\square\)

### O7.4 Wald interval and its endpoints — [BRIDGE]

Under (A1)–(A4), \(P\big(\eta_D\in\hat\eta_D\pm z_{1-\alpha/2}\hat\sigma/\sqrt n\big)\to1-\alpha\)
(O7.2, O7.3, Slutsky). *Audit note (H2):* this is an "if"; when \(\sigma^2=0\) the
first-order theory gives no level and the interval is unsupported (measured
conservative, O7.7). The endpoints are where the vector case differs from O6.

**(a) The \(\sigma^2=0\) set is an ellipsoid per cell.** Under (A1)–(A3′),
\(\sigma^2=0\) iff for every cell \(b\) the conditional law of \(S\) given
\(Z=b\) is supported on

\[
\boxed{\;
\mathcal E_b=\Big\{s:\ (s-VI_Z^{-1}c_b)^\top V^{-1}(s-VI_Z^{-1}c_b)=c_b^\top I_Z^{-1}(V-I_Z)I_Z^{-1}c_b\Big\},
\;}
\]

an ellipsoid in the \(V^{-1}\) metric centred at \(VI_Z^{-1}c_b\) (the right-hand
side is \(\ge0\) because \(V\succeq I_Z\); when it is \(0\) the ellipsoid is that single
point — audit note H4). The identity behind it holds for anisotropic \(V\):
`test_o7_audit_ellipsoid_identity_holds_on_an_anisotropic_rational_law`.

*Proof.* \(\psi=0\) a.s. iff \(s^\top V^{-1}s-2s^\top I_Z^{-1}c_b+c_b^\top I_Z^{-1}c_b=0\)
on the support of each conditional law; completing the square in the
\(V^{-1}\) metric turns the left side into
\((s-VI_Z^{-1}c_b)^\top V^{-1}(s-VI_Z^{-1}c_b)-c_b^\top I_Z^{-1}VI_Z^{-1}c_b+c_b^\top I_Z^{-1}c_b\). \(\square\)

Consequences.

- *Absolutely continuous cells force (A4).* If \(S\mid Z=b\) has a Lebesgue
  density on some cell of positive probability, \(\sigma^2>0\): \(\psi\) is a
  quadratic polynomial in \(s\) with quadratic part \(-(\eta_D/d)\,s^\top V^{-1}s\ne0\),
  so its zero set is Lebesgue-null.
- *Atomless is not enough for \(d\ge2\)* — the naive lift of O6.4's
  "an atomless cell of positive probability implies \(\sigma^2>0\) when
  \(\eta>0\)" is **false**: `CE-O7-ELLIPSOID-ZERO-VARIANCE-001`. Four cells
  related by quarter turns, \(S\mid Z=0\) uniform on \(\{(3,4),(3,-4)\}\),
  \(p_b=1/4\): \(E[S]=0\), \(V=\tfrac{25}2I\), \(I_Z=\tfrac92I\),
  \(\eta_D=9/25\) (det ratio \(81/625\)), and \(\psi=0\) at every atom. The
  cell-0 ellipsoid is the circle \(|s-(25/3,0)|^2=(20/3)^2\) and, as a
  polynomial, \(\psi_r(s)=-(2r/25)\,(s_1^2+s_2^2-\tfrac{50}3s_1+25)\) with
  \(r=\eta_D^2\) — it vanishes on the whole circle. Every law on that circle
  with mean \((3,0)\) therefore has \(\sigma^2=0\) at \(\eta_D=9/25\), including
  the *atomless* law uniform on the arc of half-angle \(\alpha\) about the far
  point, \(\sin\alpha/\alpha=4/5\) (\(\alpha\approx1.1311\)); measured on it,
  the Wald interval is conservative (O7.7). At \(d=1\) the ellipsoid is the
  two-point set \(\{c_b/(1\mp\sqrt{1-\eta})\}\) of O6.4, which is why O6's
  atomless remark was true there and only there.
- *\(\eta_D=1\)* iff \(V=I_Z\) iff \(S=c_Z\) a.s. (zero within-cell scatter);
  then \(\psi\equiv0\), the ellipsoid degenerates to the point \(c_b\), and
  \(\hat\eta_D=1\) for every sample **with \(\hat V\succ0\)**, i.e. whenever the
  occupied cells' means span \(\mathbb R^d\). *Audit hardening H1:* a sample confined
  to fewer spanning cells has \(\hat V\) singular, \(\hat\eta_D=0\) by the estimator's
  own convention, and library value \(1\) (projection) —
  `CE-O7-UNIT-RETENTION-SINGULAR-SAMPLE-001`; the event has probability \(\to0\)
  exponentially under (A1).
- Where (A4) fails the first-order limit is degenerate and the interval is
  unsupported; measured (O7.7) it is conservative on the arc law with width
  \(O(1/n)\). The quadratic-form limit of \(n(\hat\eta_D-\eta_D)\) is not derived.

**(b) Singular \(I_Z\): the plug-in is biased upward at rate \(n^{-(d-r)/d}\).**
Let (A1), (A3) hold with only \(E\|S\|^2<\infty\), and let
\(r=\operatorname{rank}I_Z<d\) (so \(\eta_D=0\)), \(U\in\mathbb R^{d\times(d-r)}\)
an orthonormal basis of the null space of \(I_Z\). Then

\[
\boxed{\;\hat\eta_D=O_p\big(n^{-(d-r)/d}\big),\;}
\]

and if moreover \(K\ge d\) and \(E[U^\top SS^\top U\,\mathbf 1_{Z=b}]\succ0\) for
every cell, \(n^{(d-r)/d}\hat\eta_D\) converges in law to a strictly positive
random variable. Registered as `RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE`.

*Proof.* Let \(R\) span the range of \(I_Z\) and write \(a_b=R^\top\hat m_b\),
\(u_b=U^\top\hat m_b\). Since \(U^\top m_b=0\) for every \(b\) (\(I_Z=\sum_bm_bm_b^\top/p_b\succeq m_bm_b^\top/p_b\)),
\(\sqrt n\,u_b\Rightarrow G_b\) jointly, with \(G_b\sim N(0,E[U^\top SS^\top U\mathbf 1_{Z=b}])\)
and \(G_b\perp G_{b'}\) for \(b\ne b'\) (the cross-covariance
\(E[U^\top S\mathbf 1_{Z=b}(U^\top S\mathbf 1_{Z=b'})^\top]-0=0\)). In the basis
\([R\ U]\), \(\hat I_Z\) has blocks \(A_n=\sum_ba_ba_b^\top/\hat p_b\to A=R^\top I_ZR\succ0\),
\(B_n=\sum_ba_bu_b^\top/\hat p_b=O_p(n^{-1/2})\), \(C_n=\sum_bu_bu_b^\top/\hat p_b=O_p(n^{-1})\),
and \(\det\hat I_Z=\det A_n\det S_n\) with the Schur complement
\(S_n=C_n-B_n^\top A_n^{-1}B_n=\sum_b(u_b-\Gamma_na_b)(u_b-\Gamma_na_b)^\top/\hat p_b\),
\(\Gamma_n=B_n^\top A_n^{-1}\). By the CLT, the strong law and Slutsky,
\(nS_n\Rightarrow\sum_bW_bW_b^\top/p_b\) with \(W=G(I-\Lambda)\),
\(\Lambda_{b'b}=a_{b'}^{\infty\top}A^{-1}a_b^\infty/p_{b'}\), \(a_b^\infty=R^\top m_b\)
— \(\Lambda\) is a rank-\(r\) projection. Hence
\(n^{d-r}\hat\eta_D^d=n^{d-r}\det\hat I_Z/\det\hat V\Rightarrow\det A\cdot\det(WP^{-1}W^\top)/\det V\),
which is \(O_p(1)\): the upper bound. Under the nondegeneracy hypothesis the
\(G_b\) are independent nondegenerate Gaussians, so \(G\) has a Lebesgue
density on \(\mathbb R^{(d-r)\times K}\) and \(W=G(I-\Lambda)\) one on the
\((d-r)(K-r)\)-dimensional image space; when \(K-r\ge d-r\) the rank-deficient
matrices form a proper algebraic subvariety of that space, so
\(\operatorname{rank}W=d-r\) a.s. and \(\det(WP^{-1}W^\top)>0\) a.s. \(\square\)

Consequences. At \(d=1\) (\(r=0\)) this is O6's measured \(O(1/n)\) bias. At
\(d\ge2\) with \(r=d-1\) the bias is \(n^{-1/d}\), *slower* than the CLT
scale: the plug-in and its Wald half-width \(\hat\sigma/\sqrt n\) are then of the
same order (measured, O7.7), so coverage of the true value \(0\) is *measured* to stabilise at a
law-dependent constant, not at \(1-\alpha\); it happened to be \(\approx0.95\) on
the \(d=2\), \(K=2\) law and \(\approx0.80\) on the \(d=3\), \(K=3\) law. *Audit note
(H3):* the joint limit of \((n^{(d-r)/d}\hat\eta_D,\,n^{(d-r)/d}\hat\sigma/\sqrt n)\) is not
derived, so neither the half-width order nor the coverage convergence is a theorem;
the audit confirmed the limit law of \(n^{(d-r)/d}\hat\eta_D\) itself against its
Gaussian ingredients (closed form \(E=2/\pi\) at \(d=2\), \(K=2\)). This is the
classical dimensionality problem (Seo, Kanda & Fujikoshi 1995): the right
tool at the singular endpoint is a rank test, not this interval. The
reference-law instance is \(K=d\): U4 forces \(\eta_D=0\), the sample plug-in
is positive of order \(n^{-1/d}\), and the library number is that plug-in.

**(c) Other unsupported cases.** Empty evaluation cells are handled by
\(0/0:=0\) and vanish exponentially under (A1); no uniformity over small
\(p_b\). The interval may leave \([0,1]\) near the endpoints (measured: never,
on the regular laws; at the singular endpoint its lower end is negative in
most replicates, which is how it covers \(0\)). No transformation was
attempted — one estimator, one interval, per packet. No finite-sample
guarantee: O7.7 shows how far from first order a heavy-tailed score can be.

### O7.5 What the interval does and does not measure — [BRIDGE]

Verbatim from O6.5 with \(\eta\) replaced by \(\eta_D\): the interval is for
the **true** geometric-mean retention conditional on the frozen rule; the
library's self-reported surrogate replaces \(S\) by \(\hat s\), its population
value \(\tilde\eta_D\) is a different number (O4), the same theorem gives the
surrogate plug-in a valid interval around that wrong target, and the
\(O(n^{-1/2})\) interval excludes \(\tilde\eta_D\) with probability \(\to1\)
whenever \(\tilde\eta_D\ne\eta_D\). Without an oracle-score evaluation sample
nothing here applies. Since the D-efficiency is the geometric mean of the
retention spectrum, the interval says nothing about the worst retained
direction (`OPEN-D-DIRECTIONAL-BOUND`).

### O7.6 Self-adversarial pass (protocol G)

- *Ties, duplicate scores:* irrelevant; the functional is smooth in moments
  (selftest carries a duplicate atom).
- *Singleton/empty cells:* O7.4(c); under (A1) transient.
- *Singular information:* \(V\) singular is excluded by (A3) — the library
  projects instead (O7.1, a different estimator); \(I_Z\) singular is O7.4(b).
- *Nuisance singularity, \(D_s\):* out of scope; the profiled retention is a
  Schur-complement functional and needs its own influence function.
- *Atomic laws:* the ellipsoid characterisation; two atoms per cell suffice
  for \(\sigma^2=0\) in every \(d\), and in \(d\ge2\) atomless singular laws do too.
- *Hidden compactness:* none used.
- *First-order-to-finite jumps:* none claimed. Measured: \(n\cdot\)bias
  \(\approx2\)–\(3\) on the \(d=2\) laws, but still growing at \(n=3000\) on the
  \(d=3\) Hermite law (\(E\|S\|^4=4259\)) — pre-asymptotic, see O7.7.
- *Empirical-to-population jumps:* none — the rule is frozen; refitted rules
  are OP27's remainder.
- *Score-estimation error:* enters only through the label map.
- *Heavy tails:* (A2) is used for the CLT of \(SS^\top\) and the strong law of
  the fourth within-cell moments. It is a real constraint in practice: the
  polynomial scores of O7.7 satisfy (A2) yet need \(n\gg3000\) for nominal
  coverage, while bounded mixture-fraction scores are nominal at \(n=100\).
- *Repeated canonical correlations:* the determinant route needs none.
- *New-event extension:* the rule is frozen, so every evaluation event is
  assigned by \(q\circ\hat s\); nothing new.

### O7.7 Measured (protocol D, run before the proof was trusted) — [MEASURED]

**Exact identities** (`selftest`, `fractions.Fraction`): on integer samples in
\(d=2\) (\(n=12\), \(K=5\) with one declared-empty cell and a duplicate atom) and
\(d=3\) (\(n=14\), \(K=4\)): the within-scatter identity, \(0\le\det\)-ratio\(\le1\)
and \(\sum_i\hat\psi_i=0\) hold exactly; the exact Gateaux difference quotients
of the rational functional \(r=\hat\eta_D^d\) at \(\varepsilon=2^{-10},2^{-11},2^{-12}\)
approach \(\psi_r=d\eta_D^{d-1}\psi\) with a remainder that halves per halving
of \(\varepsilon\) (ratio within \(0.02\) of \(2\)); the exactly centred \(K=d=2\)
sample has \(\det\)-ratio exactly \(0\); the \(d=1\) reduction agrees with O6's
\(\psi\) to \(1.5\cdot10^{-15}\); library agreement \(\le1.1\cdot10^{-16}\).

**Population references** (`popref`): interval partitions of \(x\sim N(0,1)\)
with the Hermite scores \(S=(x,x^2-1)\) (\(d=2\), cuts \(-1,0,1\)) and
\(S=(x,x^2-1,x^3-3x)\) (\(d=3\), cuts \(\pm1.5,\pm0.5\)); every cell moment up to
order \(4d+2\) is a truncated-Gaussian moment (three-term recursion), so
\(\eta_D\) and \(\sigma^2\) are closed-form; a composite Gauss–Legendre route
with pieces aligned to the cuts agrees to \(1.5\cdot10^{-14}\) (\(d=2\)) and
\(1.0\cdot10^{-13}\) (\(d=3\)); \(E[S]=0\), \(E[\psi]\le1.3\cdot10^{-16}\),
\(E\|S\|^4=83\) and \(4259\) (the hand value \(3+20+60=83\) checks). A
three-component Gaussian location mixture with two interior fractions
(\(0.35,0.25\); bounded scores, \(|s|\le2.5\), \(E\|S\|^4=22.2\); cuts \(-2,-0.5,1\))
by two aligned quadrature routes agreeing to \(2.3\cdot10^{-15}\). A
**library-fitted rule**: `optimize_partition` (D-exchange, \(K=4\)) on 4000
training draws of the \(d=2\) Hermite score, compiled to its Mahalanobis rule;
its cells in \(x\) are three intervals located by bisection
(cuts \(-1.3918,\,0.0138,\,1.3753\)), and the closed-form moments per interval
agree with a quadrature that asks the compiled rule for every node's label to
\(5.3\cdot10^{-15}\).

| law | \(d\) | \(K\) | \(p_b\) | \(\eta_D\) | \(\sigma\) |
|---|---|---|---|---|---|
| Hermite \(d=2\) | 2 | 4 | 0.159, 0.341, 0.341, 0.159 | 0.690666 | 0.673175 |
| Hermite \(d=3\) | 3 | 5 | 0.067, 0.242, 0.383, 0.242, 0.067 | 0.629969 | 1.624099 |
| bounded mixture | 2 | 4 | 0.178, 0.305, 0.254, 0.263 | 0.330317 | 0.621795 |
| library rule on Hermite \(d=2\) | 2 | 4 | 0.424, 0.085, 0.410, 0.082 | 0.743363 | 0.557001 |
| Hermite \(d=2\), \(K=2\) (cut \(0\)) | 2 | 2 | 0.5, 0.5 | 0 (rank 1) | — |
| Hermite \(d=3\), \(K=3\) (cuts \(\pm0.6\)) | 3 | 3 | 0.274, 0.451, 0.274 | 0 (rank 2) | — |
| arc law (fixture, atomless) | 2 | 4 | 1/4 each | 9/25 | 0 |

**Coverage of the 95% Wald interval**, 2000 independent samples per size,
`SeedSequence([20260906, law, n])`:

| law | \(n\) | coverage ± SE | SD\((\hat\eta_D)\) | \(\sigma/\sqrt n\) | mean \(\hat\sigma/\sqrt n\) | rel. RMSE \(\hat\sigma/\sigma\) | \(n\cdot\)bias | plug-in skew | oracle SD / skew |
|---|---|---|---|---|---|---|---|---|---|
| bounded mixture | 100 | 0.950 ± 0.005 | 0.0627 | 0.0622 | 0.0675 | 0.168 | 2.80 | +0.16 | 1.01 / +0.20 |
| | 300 | 0.945 ± 0.005 | 0.0363 | 0.0359 | 0.0370 | 0.083 | 2.46 | −0.12 | 1.01 / −0.02 |
| | 1000 | 0.956 ± 0.005 | 0.0191 | 0.0197 | 0.0198 | 0.040 | 1.70 | +0.00 | 0.97 / +0.07 |
| | 3000 | 0.945 ± 0.005 | 0.0116 | 0.0114 | 0.0114 | 0.022 | 2.51 | −0.02 | 1.02 / +0.01 |
| Hermite \(d=2\) | 100 | 0.681 ± 0.010 | 0.0554 | 0.0673 | 0.0366 | 0.480 | 3.16 | +0.60 | 0.82 / −0.52 |
| | 300 | 0.813 ± 0.009 | 0.0342 | 0.0389 | 0.0284 | 0.362 | 3.80 | +0.63 | 0.88 / −0.43 |
| | 1000 | 0.892 ± 0.007 | 0.0204 | 0.0213 | 0.0186 | 0.278 | 2.86 | +0.59 | 0.96 / −0.26 |
| | 3000 | 0.937 ± 0.005 | 0.0118 | 0.0123 | 0.0117 | 0.198 | 2.90 | +0.49 | 0.96 / −0.14 |
| library rule | 100 | 0.816 ± 0.009 | 0.0405 | 0.0557 | 0.0343 | 0.403 | 2.01 | +0.04 | 0.73 / −0.63 |
| | 300 | 0.861 ± 0.008 | 0.0276 | 0.0322 | 0.0233 | 0.346 | 2.03 | +0.30 | 0.86 / −0.63 |
| | 1000 | 0.889 ± 0.007 | 0.0166 | 0.0176 | 0.0150 | 0.295 | 2.70 | +0.50 | 0.94 / −0.47 |
| | 3000 | 0.927 ± 0.006 | 0.0097 | 0.0102 | 0.0095 | 0.242 | 2.80 | +0.51 | 0.96 / −0.25 |
| Hermite \(d=3\) | 100 | 0.399 ± 0.011 | 0.0706 | 0.1624 | 0.0394 | 0.761 | 8.9 | +0.69 | 0.43 / −0.55 |
| | 300 | 0.556 ± 0.011 | 0.0579 | 0.0938 | 0.0329 | 0.656 | 12.6 | +0.43 | 0.62 / −0.52 |
| | 1000 | 0.733 ± 0.010 | 0.0414 | 0.0514 | 0.0290 | 0.480 | 15.0 | +0.82 | 0.81 / −0.50 |
| | 3000 | 0.827 ± 0.008 | 0.0272 | 0.0297 | 0.0222 | 0.362 | 19.3 | +0.83 | 0.92 / −0.45 |

Reading. On the bounded mixture scores — the library's mixture-fraction
use case — the interval is nominal from \(n=100\): the spread of \(\hat\eta_D\)
equals \(\sigma/\sqrt n\), \(\hat\sigma\) is within 8% of \(\sigma\) at \(n=100\) and
within 3% beyond, and the studentized statistic has no skew. On the Hermite scores every first-order quantity
converges as the theorem says (SD\(\to\sigma/\sqrt n\), \(\hat\sigma\to\sigma\),
oracle-studentized SD \(\to1\)) but slowly: \(\hat\sigma\) *under*-estimates
\(\sigma\) by 46% at \(n=100\) and 5% at \(n=3000\) in \(d=2\), by 76% and 25% in
\(d=3\), because \(\hat\sigma^2\) is a sample eighth (\(d=2\)) or twelfth
(\(d=3\)) moment of \(x\). The interval is therefore liberal there at every size
tested — a second-order, heavy-tail effect governed by \(E\|S\|^4\) (83 and
4259 against 22 for the mixture), not a counterexample to O7.2–O7.4. Bias
is \(O(1/n)\) in \(d=2\) and still pre-asymptotic in \(d=3\) at \(n=3000\). No
regular-law interval left \([0,1]\); empty cells occurred in 4 of 32,000
replicates (all at \(n=100\), in cells with \(p_b\le0.085\)).

**Endpoints** (same seeds):

| law | \(n\) | covers \(0\) | mean \(\hat\eta_D\) | \(n^{1/d}\,\)mean \(\hat\eta_D\) | median half-width / \(\hat\eta_D\) |
|---|---|---|---|---|---|
| Hermite \(d=2\), \(K=2\) (\(r=1\)) | 100 | 0.926 | 0.0704 | 0.704 | 2.77 |
| | 300 | 0.946 | 0.0379 | 0.656 | 2.90 |
| | 1000 | 0.954 | 0.0199 | 0.631 | 3.03 |
| | 3000 | 0.956 | 0.0114 | 0.627 | 2.95 |
| Hermite \(d=3\), \(K=3\) (\(r=2\)) | 100 | 0.720 | 0.1319 | 0.612 | 1.65 |
| | 300 | 0.753 | 0.0853 | 0.571 | 1.82 |
| | 1000 | 0.788 | 0.0546 | 0.546 | 1.81 |
| | 3000 | 0.802 | 0.0365 | 0.526 | 1.94 |

\(n^{1/d}\hat\eta_D\) stabilises (\(0.63\) at \(d=2\); \(0.53\) and still drifting
at \(d=3\)), the half-width stays a fixed multiple of \(\hat\eta_D\), and
coverage of the true \(0\) tends to a constant — \(0.95\) by coincidence at
\(d=2\), \(0.80\) at \(d=3\) — exactly the behaviour O7.4(b) predicts. On the
**arc law** (\(\sigma^2=0\), \(\eta_D=9/25\)): coverage \(0.996\), \(0.999\), \(0.999\),
\(0.999\); \(n\cdot\)bias \(\approx1.4\) and \(n\cdot\)SD \(\approx1.55\) at every
size (width \(O(1/n)\)); \(\hat\eta_D\ge\eta_D\) in 85% of replicates —
conservative, never liberal, but unlike O6's two-atom law not one-sided.

None of this is a proof of O7.2–O7.4; nothing is claimed about coverage on
heavy-tailed scores at the sizes shown beyond the numbers themselves.

### O7.8 Information-loss implication and verdict (protocol H)

O7 bounds nothing; it equips the library's reported number, the true
geometric-mean D-retention \(\eta_D\) of a frozen rule, with a conditional
\(n^{-1/2}\) error bar on held-out oracle-score data, computable in one extra
pass from the same cell moments. **Verdict: proved**, as a bridge — the CLT
is the vector delta method and the determinant differential, the influence
function is a sum of published canonical-correlation influence functions, and
what is project-level is the uncentred fixed-partition form on cell moments,
the plug-in variance with its consistency, the ellipsoid characterisation of
\(\sigma^2=0\) with its atomless counterexample, and the \(n^{-(d-r)/d}\) rate at
the singular endpoint. Practical reading: for bounded (mixture-fraction)
scores the interval is usable at \(n\) in the hundreds; for unbounded
polynomial-type scores it is liberal until \(n\) is far larger than the fourth
moment suggests, and at \(K\le d\) or any rank-deficient \(I_Z\) it must not be
used at all — the plug-in is then biased upward at a rate that is *slower*
than \(n^{-1/2}\) for \(d\ge2\). Left open and rerouted to
`OPEN-RETENTION-UNCERTAINTY`: rules refitted on the evaluation sample (the
boundary non-smoothness OP27 names); weighted samples; the no-oracle case;
the degenerate limits at \(\sigma^2=0\) and at the singular endpoint; and the
profiled \(D_s\) retention.
