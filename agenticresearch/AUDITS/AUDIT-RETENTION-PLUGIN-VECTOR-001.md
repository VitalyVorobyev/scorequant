# Publication-grade audit of the frozen-rule vector retention plug-in CLT

**Claims:** `RETENTION-PLUGIN-CLT-FROZEN-VECTOR` (bridge), `RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE` (project_proved), `RETENTION-PLUGIN-COVERAGE-VECTOR` (measured), `CE-O7-ELLIPSOID-ZERO-VARIANCE-001`, and the 6 September patch of `OPEN-RETENTION-UNCERTAINTY`
**Audit:** `AUDIT-RETENTION-PLUGIN-VECTOR`
**Date:** 6 September 2026
**Source:** branch `worktree-retention-plugin-vector` at `b8a61ab` (PR #59)
**Result:** verified with hardened assumptions — O7.1, O7.2, O7.3, the Wald statement of O7.4 under (A4), the ellipsoid characterisation O7.4(a) and the rate and limit law of O7.4(b) hold as stated; two endpoint sentences and one "iff" are hardened; the fixture is verified; every population reference and every coverage number replicates on fresh seeds; the singular-endpoint limit law is confirmed against its own Gaussian ingredients; no re-attribution to prior art.

The auditor did not produce O7 and did not read the researcher's instrument
`py/retention_plugin_vector.py`, the completed packet or any transcript before the
audit's own numbers existed. The independent instrument is
`py/audit_retention_plugin_vector.py`; its six provenance-stamped records live under
`AUDITS/artifacts/AUDIT-RETENTION-PLUGIN-VECTOR-001/`. After the audit's Hermite references
were computed, the researcher's `popref.json` and then two definitions in the researcher's
module (the mixture law's parameters and the rule-fitting recipe) were opened, because
neither the claim node nor the artifact records them (§9, attack 10); they are copied into
`extra_laws.json` of this audit.

## 1. Target statement

Frozen: training data, provider \(\hat s\), rule \(q\), reference point \(\theta_0\), \(K\).
Random: \(X_1,\dots,X_n\) iid from the evaluation law, equally weighted. Observables
\(S_i=s(X_i)\in\mathbb R^d\) and \(Z_i=q(\hat s(X_i))\in\{1,\dots,K\}\). With \(p_b\), \(m_b=E[S\mathbf 1_{Z=b}]\),
\(c_b=m_b/p_b\), \(V=E[SS^\top]\), \(I_Z=\sum_bm_bm_b^\top/p_b\),

\[
\eta_D=\Big(\frac{\det I_Z}{\det V}\Big)^{1/d},\qquad
\hat\eta_D=\Big(\frac{\det\hat I_Z}{\det\hat V}\Big)^{1/d}\ (0/0:=0;\ \hat\eta_D:=0\text{ if }\det\hat V=0).
\]

Claimed: (O7.1) \(\hat V-\hat I_Z\) is the within-cell scatter, \(0\le\hat\eta_D\le1\), and the
library's `geometric_mean_retention` is \(\hat\eta_D\) when \(\hat V\succ0\) and nothing is
projected; (O7.2) under (A1) \(p_b>0\), (A2) \(E\|S\|^4<\infty\), (A3) \(V\succ0\), (A3′) \(I_Z\succ0\):
\(\sqrt n(\hat\eta_D-\eta_D)\Rightarrow N(0,\sigma^2)\), \(\sigma^2=E\psi^2\),
\(\psi=\frac{\eta_D}{d}[2S^\top I_Z^{-1}c_Z-c_Z^\top I_Z^{-1}c_Z-S^\top V^{-1}S]\); (O7.3)
\(\hat\sigma^2=n^{-1}\sum\hat\psi_i^2\to\sigma^2\) a.s., \(\sum_i\hat\psi_i=0\); (O7.4) the Wald
interval has level \(1-\alpha\) iff (A4) \(\sigma^2>0\); \(\sigma^2=0\) iff every cell's
conditional law sits on its ellipsoid \(\mathcal E_b\); at \(\eta_D=1\) the plug-in is 1 on
every sample; at \(\operatorname{rank}I_Z=r<d\), \(\hat\eta_D=O_p(n^{-(d-r)/d})\) with a strictly
positive limit of \(n^{(d-r)/d}\hat\eta_D\) when \(K\ge d\) and the null-direction cell second
moments are nondegenerate, and coverage of \(0\) converges to a law-dependent constant;
(O7.5) the interval is for the true retention and excludes the proxy value with probability
\(\to1\).

## 2. Criterion and problem level

- Criterion: D — the geometric-mean retention \((\det R)^{1/d}\), `INFO-D-EFFICIENCY`.
- Level: `information_accounting`, conditional on a frozen `empirical_inductive_quantizer`.
- Decision variable: none; the random object is the evaluation sample.
- Score-oracle regime: exact oracle score on the evaluation sample, estimated score inside
  the frozen label map.

## 3. Status before the audit

`RETENTION-PLUGIN-CLT-FROZEN-VECTOR`: `bridge`, `prior_art_found`, "not yet independently
audited". `RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE`: `project_proved`, `search_gap`, not
audited. `RETENTION-PLUGIN-COVERAGE-VECTOR`: `measured`. `OPEN-RETENTION-UNCERTAINTY`: open,
with the vector special case recorded as settled but unaudited. Under `protocols/audit.md`
nothing of O7 may be shipped or cited before this audit closes; closure step 1
(`OPEN_PROBLEMS.md`) waits on it.

## 4. Dependencies rechecked

- `RETENTION-PLUGIN-CLT-FROZEN-SCALAR` (O6, audited 5 Sep). O7 at \(d=1\) must be O6: with
  \(I_Z=\eta v\), \(\psi=(2c_ZS-c_Z^2-\eta S^2)/v=((1-\eta)S^2-(S-c_Z)^2)/v\). Verified exactly on
  the \(d=1\) sample of `exact.json` (`d1_reduction_to_O6_exact`).
- `FI-QUANT-IDENTITY`, `PROXY-TRUE-RETAINED-FI`: as in the O6 audit, \(q\circ\hat s\) is a fixed
  measurable map, so \(I_Z=\operatorname{Var}(E[s\mid Z])\) at \(\theta_0\) and \(V=I_{\rm full}\)
  under the reference law; the retention reading needs the evaluation law to be
  \(P_{\theta_0}\). Rechecked, not re-proved.
- `FI-RANK-CEILING` (U4): at the reference law \(\sum_bm_b=E[S]=0\), so
  \(\operatorname{rank}I_Z\le\min(d,K-1)\); (A3′) needs \(K\ge d+1\). Verified on the endpoint
  laws: \(K=2\), \(d=2\) has rank 1; \(K=3\), \(d=3\) has rank 2 (`popref.json`).
- `INFO-D-EFFICIENCY`, `D-REPARAM-INVARIANCE`: \(\eta_D\) and \(\psi\) invariant under
  \(S\mapsto AS\); verified exactly with a rational \(A\) in \(d=2,3\)
  (`affine_invariance_exact`).
- Imported theorems: multivariate CLT with finite second moments of
  \(T=(\mathbf 1_{Z=b},S\mathbf 1_{Z=b},\operatorname{vech}SS^\top)_b\) (needs (A2)), the delta method
  (van der Vaart Thm 3.1), \(d\log\det A=\operatorname{tr}A^{-1}dA\).

## 5. Nearest literature and transfer boundary

Fresh triangulation in `LITERATURE/audits/AUDIT-RETENTION-PLUGIN-VECTOR-6-September-2026.md`
(nine sources, six fields each, verification labels). Summary:

- **Method — prior art, confirmed.** Delta method and matrix differentials (verified
  texts). Romanazzi's per-coefficient influence function
  \(\operatorname{IF}(\rho_i^2)=2\rho_iu_iv_i-\rho_i^2(u_i^2+v_i^2)\) is now seen restated
  (Taskinen et al. 2006 via arXiv:1705.04194); O7.2's \(\psi\) is
  \((\eta_D/d)\sum_i\operatorname{IF}(\rho_i^2)/\rho_i^2\). Muirhead & Waternaux (1980): abstract
  verified — finite fourth moments, nonnormal, per-root. Fang & Krishnaiah (1982), Seo, Kanda
  & Fujikoshi (1995): citation-level only, as the researcher recorded in `gaps.md`.
- **The exact statement — search gap, not novelty.** No source states the uncentred
  fixed-partition determinant ratio on the library's cell moments with its \(\psi\), plug-in
  variance, ellipsoid zero set or the \(n^{-(d-r)/d}\) rate. The rank-test literature (Robin &
  Smith 2000, abstract verified: \(n\)-scaled smallest roots \(\to\) weighted \(\chi^2\) sums) has
  the same shape of limit as O7.4(b) at \(d-r=1\); the determinant form and the bias reading
  are O7's.
- **No re-attribution.** Two keys added: `Robin-Smith-2000`,
  `Taskinen-Croux-Kankainen-Ollila-Oja-2006`.

## 6. Counterexample search

Stage `exact` (`exact.json`), all in `fractions.Fraction` with an own Gauss–Jordan inverse:

- Six adversarial samples — \(d=2\) with ties, a duplicate atom, a singleton cell and a
  declared-empty cell (\(K=5\)); \(d=3\) with the same score in two cells; \(K=d=2\) on an
  exactly centred sample; \(K=2<d=3\); \(d=2\) with \(\hat V\) singular; \(d=1\) with ties and an
  empty cell. On every sample the within-scatter identity \(\hat V-\hat I_Z=\frac1n\sum(S_i-\hat c_{Z_i})(\cdot)^\top\)
  and \(0\le\)ratio\(\le1\) hold exactly; on the regular ones \(\sum_i\hat\psi_i=0\), the
  order-four moment expansion of \(\hat\sigma^2\) equals the direct sum, the exact Gateaux
  quotients of \(r=\hat\eta_D^d\) approach \(r\,B\) with remainder halving per halving of
  \(\varepsilon\) (all atoms), and the library agrees to \(\le3.3\cdot10^{-16}\). The centred
  \(K=d\) sample has ratio exactly \(0\) and library \(3.5\cdot10^{-9}\); \(K<d\) gives
  \(0\) and \(8.3\cdot10^{-7}\) — the \(\varepsilon^{1/d}\) rounding caveat of O7.1(ii),
  reproduced and larger in \(d=3\). Singular \(\hat V\): exact \(0\) by convention, library
  \(0.868\) with effective rank 1 — caveat (i).
- Thirty random rational atomic laws (\(d=2,3\), \(K=3,4,5\), unequal masses): \(E\psi=0\),
  \(E N_1=E N_2=d\), the covariance form of \(\sigma^2\), the population Gateaux reduction
  and the ellipsoid polynomial identity (on the support and at generic points, every cell)
  hold exactly on 30/30.

Stage `degenerate` (`degenerate.json`):

- The fixture's two-atom law: \(V=\frac{25}2I\), \(I_Z=\frac92I\), ratio \(81/625\), \(\psi=0\) at
  all eight atoms, cell-0 ellipsoid centre \((25/3,0)\) with \(V^{-1}\)-radius\(^2\) \(32/9\),
  library \(0.36\). The atomless arc law (\(\sin\alpha/\alpha=4/5\), \(\alpha=1.13110\)) by
  4000-node Gauss–Legendre in the angle: cell mean \((3,0)\), second moment \(25\),
  \(V=\frac{25}2I\), \(I_Z=\frac92I\) to \(10^{-11}\), \(\max|B|\) on the support \(1.1\cdot10^{-12}\),
  \(\sigma^2=8\cdot10^{-27}\). **The fixture is verified.**
- A \(d=3\) law of the same kind (\(K=6\) cells \(\pm e_i\), two atoms \(3\sigma e_i\pm4e_j\) per
  cell, weights \(1/12\)): \(V=\frac{25}3I\), \(I_Z=3I\), \(\eta_D=9/25\), \(\psi=0\) at every atom,
  and the circle \(\{s_1=3,\ s_2^2+s_3^2=16\}\) lies on the cell-0 ellipsoid — the atomless
  obstruction to (A4) exists in every \(d\ge2\) by one construction.
- **Boundary failure found (H1):** the \(\eta_D=1\) law \(S=c_Z\) a.s., \(c_0=(1,0)\),
  \(c_1=(0,1)\), \(p_b=\frac12\). Exhaustive compositions to \(n=6\): the 12 one-cell samples
  have \(\det\hat V=0\), exact plug-in \(0\) and library \(1\) (effective rank 1); every sample
  with both cells occupied has ratio exactly \(1\). Serialised as
  `COUNTEREXAMPLES/CE-O7-UNIT-RETENTION-SINGULAR-SAMPLE-001.json` and pinned.
- Two-atom law, exhaustive compositions to \(n=6\) (3002 compositions, 2824 regular): the
  plug-in falls **below** \(\eta_D\) in 320 of them (first at \(n=4\), ratio \(121/5625\)) — unlike
  O6's two-atom law the vector \(\sigma^2=0\) plug-in is not one-sided, as O7.7 remarked.

No counterexample to O7.1, O7.2, O7.3, the ellipsoid characterisation or the endpoint rate
was found.

## 7. Algebraic reduction

Write \(B(s,b)=2s^\top I_Z^{-1}c_b-c_b^\top I_Z^{-1}c_b-s^\top V^{-1}s\), so \(\psi=\frac{\eta_D}dB\) and the
influence of \(r=\eta_D^d\) is \(\psi_r=rB\).

**Gradient.** With \(g=\exp\{d^{-1}[\log\det I_Z-\log\det V]\}\) and one observation's
increments \(dp_b=\mathbf 1_{Z=b}-p_b\), \(dm_b=S\mathbf 1_{Z=b}-m_b\), \(dV=SS^\top-V\):
\(d\log\det I_Z=\sum_b[2c_b^\top I_Z^{-1}dm_b-c_b^\top I_Z^{-1}c_b\,dp_b]\) and
\(d\log\det V=\operatorname{tr}V^{-1}dV\). The constants are
\(2\operatorname{tr}(I_Z^{-1}\sum_bm_bc_b^\top)=2d\), \(\operatorname{tr}(I_Z^{-1}\sum_bp_bc_bc_b^\top)=d\),
\(\operatorname{tr}(V^{-1}V)=d\), and \(2d-d-d=0\). Hence \(\nabla g^\top(T-\theta)=\psi\). Verified
by exact Gateaux quotients on every atom of the six samples and on 30 laws (the
population tilt \((1-\varepsilon)\theta+\varepsilon\delta_{\rm atom}\)).

**Zero mean.** \(E[N_1]=\sum_bp_b(2c_b^\top I_Z^{-1}c_b-c_b^\top I_Z^{-1}c_b)=d\) and
\(E[N_2]=\operatorname{tr}V^{-1}V=d\), so \(E\psi=0\) and \(\sigma^2=E\psi^2\). Exact on 30/30.

**Sample identities (O7.3).** \(\sum_iS_i^\top\hat I_Z^{-1}\hat c_{Z_i}=n\operatorname{tr}(\hat I_Z^{-1}\sum_b\hat p_b\hat c_b\hat c_b^\top)=nd\),
likewise the other two, so \(\sum_i\hat\psi_i=0\) whenever \(\hat I_Z,\hat V\succ0\). Exact.

**Order-four expansion (O7.3).** With \(a_b=\hat I_Z^{-1}\hat c_b\), \(k_b=\hat c_b^\top\hat I_Z^{-1}\hat c_b\),
\(Q=\hat V^{-1}\): \(B^2=4(s^\top a)^2-4ks^\top a-4(s^\top a)(s^\top Qs)+k^2+2k\,s^\top Qs+(s^\top Qs)^2\),
whose cell averages are contractions of the within-cell moment tensors of orders 0–4
with \(a_b,k_b,Q\). Verified exactly on every sample and law; the strong law under (A2)
then gives \(\hat\sigma^2\to\sigma^2\) a.s. as O7.3 says.

**Ellipsoid (O7.4a).** Completing the square in the \(V^{-1}\) metric,
\(B(s,b)=-\big[(s-VI_Z^{-1}c_b)^\top V^{-1}(s-VI_Z^{-1}c_b)-c_b^\top I_Z^{-1}(V-I_Z)I_Z^{-1}c_b\big]\) as a
polynomial identity in \(s\). Verified exactly for anisotropic \(V\) on 30 laws and pinned in
`test_o7_audit_ellipsoid_identity_holds_on_an_anisotropic_rational_law`. The right-hand
side is \(\ge0\) because \(V-I_Z\succeq0\) (O7.1); when it is \(0\) the ellipsoid is the single
point \(VI_Z^{-1}c_b\) (H4). Since the quadratic part \(-s^\top V^{-1}s\) is a nonzero polynomial,
the zero set is Lebesgue-null: an absolutely continuous cell forces \(\sigma^2>0\).

**Endpoint (O7.4b).** In the basis \([R\ U]\) of range and null space of \(I_Z\), \(U^\top m_b=0\)
for every \(b\) (\(I_Z\succeq m_bm_b^\top/p_b\)); \(\sqrt n\,U^\top\hat m_b\Rightarrow G_b\sim N(0,U^\top E[SS^\top\mathbf 1_{Z=b}]U)\)
jointly, independent across cells because the indicators are disjoint. The Schur complement
\(S_n=\sum_b(u_b-\Gamma_na_b)(\cdot)^\top/\hat p_b\) satisfies \(nS_n\Rightarrow WP^{-1}W^\top\) with
\(W=G(I-\Lambda)\), \(\Lambda_{b'b}=a_{b'}^\top A^{-1}a_b/p_{b'}\) an idempotent of rank \(r\) (checked:
\(\Lambda^2=\Lambda\), \(\operatorname{tr}\Lambda=r\), on both endpoint laws). Hence
\(n^{d-r}\hat\eta_D^d\Rightarrow\det A\det(WP^{-1}W^\top)/\det V\). On `endpoint2` (\(d=2\), \(K=2\),
cut \(0\)): \(A=2/\pi\), \(\Lambda=\frac12\begin{pmatrix}1&-1\\-1&1\end{pmatrix}\), \(G_b\sim N(0,1)\),
\(\det V=2\), so \(\sqrt n\,\hat\eta_D\Rightarrow|G_0+G_1|/\sqrt\pi\) with mean \(2/\pi=0.6366\) — a closed
form the recorded \(0.63\) was approaching.

## 8. Proof, hardening and the corrected statements

**O7.1.** The cross terms in cell \(b\) sum to \(n_b\hat c_b\hat c_b^\top=n\hat m_b\hat m_b^\top/\hat p_b\);
empty cells contribute to neither side. \(\det\) is monotone on the PSD order. Library
agreement follows from \(\det(\hat V^{-1/2}\hat I_Z\hat V^{-1/2})=\det\hat I_Z/\det\hat V\) when nothing
is projected. **Holds as stated**, with the two caveats reproduced (§6).

**O7.2.** \(\phi(p,M,V)=(\det\sum_{b:p_b>0}m_bm_b^\top/p_b/\det V)^{1/d}\), \(\phi:=0\) off
\(\{\det V>0,\ \text{ratio}>0\}\), equals \(\hat\eta_D\) on every sample (an empty cell has
\(\hat m_b=0\) too) and equals the smooth \(g\) on the open set
\(\{p_b>0,\ V\succ0,\ I_Z\succ0\}\ni\theta\), so it is differentiable at \(\theta\) with
\(\nabla\phi=\nabla g\); (A2) makes \(T\) square-integrable (its largest entries are
\(S_jS_k\)). Delta method. **Holds as stated.**

**O7.3.** §7: \(\hat\sigma^2\) is a fixed polynomial in \(\hat\eta_D\), \(\hat I_Z^{-1}\), \(\hat V^{-1}\),
\(\hat c_b\) and within-cell moments of order \(\le4\), each with a finite mean under (A2)
(\(E|S_jS_kS_lS_m|\le E\|S\|^4\)); finitely many strong laws; (A1), (A3), (A3′) put the limit in
the continuity set; empty cells are a.s. transient. **Almost-sure is right as stated.**

**O7.4, Wald statement.** From O7.2, O7.3 and Slutsky **given (A4). Holds.** The recorded
"level \(1-\alpha\) **iff** \(\sigma^2>0\)" overstates: nothing in O7 derives the coverage when
\(\sigma^2=0\) (no limit law for \(n(\hat\eta_D-\eta_D)\) or for \(n\hat\sigma^2\)), so "only if" is
unproved. **Hardened (H2):** the interval has asymptotic level \(1-\alpha\) *if* (A4); when
(A4) fails it is unsupported by the theorem and measured conservative (§9, attack 7).

**O7.4(a), \(\sigma^2=0\) characterisation.** Correct as an iff on the support, with (H4)
recorded. The fixture and its arc law verified; the \(d=3\) construction shows the obstruction
is not a planar accident. **Holds.**

**O7.4(a), \(\eta_D=1\) — hardened (H1).** "\(\hat\eta_D=1\) for every sample" is false: a sample
whose occupied cells' means do not span \(\mathbb R^d\) has \(\hat V\) singular and
\(\hat\eta_D=0\) by the estimator's own convention, while the library projects and returns
\(1\). Exact witness `CE-O7-UNIT-RETENTION-SINGULAR-SAMPLE-001` (probability \(2^{1-n}\) of the
event at every \(n\)). Corrected statement: *at \(\eta_D=1\), \(\hat\eta_D=1\) on every sample
with \(\hat V\succ0\), i.e. whenever the occupied cells' means span \(\mathbb R^d\); the
complementary event has probability \(\to0\) exponentially under (A1), so O7.2–O7.3 are
unaffected and \(\psi\equiv0\) stands.*

**O7.4(b), rate and limit law.** §7 confirms the derivation; the lower bound needs
\(K\ge d\) (so \(K-r\ge d-r\)) and a nondegenerate null-direction second moment in every cell,
both stated. Numerically (§9, attack 8) the limit law simulated from \(G\), \(\Lambda\), \(A\),
\(\det V\) matches the finite-\(n\) scaled plug-in in mean and quantiles. **Holds as stated.**
The consequence "coverage of \(0\) converges to a law-dependent constant" is a *measured*
statement — the joint limit of \((n^{(d-r)/d}\hat\eta_D,\ n^{(d-r)/d}\hat\sigma/\sqrt n)\) is not
derived, and the order heuristic \(\hat\sigma/\sqrt n=O_p(n^{-(d-r)/d})\) (from the null block of
\(\hat I_Z^{-1}\) being \(O_p(n)\)) is not proved. **Hardened (H3):** measured stable, not
proved to converge.

**O7.5.** Verbatim O6.5 with \(\eta_D\); the evaluation-law requirement (O6 audit H2)
carries over unchanged. **Holds.**

## 9. Adversarial audit (protocol G and the packet's attacks)

1. *Ties, duplicates, singleton and empty cells:* exact on all six samples; library agreement
   at rounding. **Verified.**
2. *Gradient algebra:* recomputed by hand (§7) and by exact Gateaux quotients on every atom
   of six samples and 30 laws. **Verified.**
3. *Consistency:* order four suffices; a.s. is right. **Verified.**
4. *Singular information:* \(\hat V\) singular — the library's projected estimator differs
   (caveat i), now also at \(\eta_D=1\) (H1, fixture). \(\hat I_Z\) exactly singular — the
   \(\varepsilon^{1/d}\) rounding (caveat ii), \(3.5\cdot10^{-9}\) in \(d=2\) and \(8.3\cdot10^{-7}\) in
   \(d=3\). **Hardened (H1), recorded.**
5. *Atomic and atomless laws:* fixture verified; \(d=3\) sphere law; ellipsoid identity for
   anisotropic \(V\); (H4). The vector \(\sigma^2=0\) plug-in is two-sided (§6). **Verified.**
6. *Population references:* independent route — QUADPACK per cell and per moment against
   cut-aligned composite Gauss–Legendre (\(600\times32\) and \(900\times40\)), no moment
   recursion. Hermite \(d=2\): \(\eta_D=0.6906662616\), \(\sigma=0.6731754001\), \(E\|S\|^4=83\);
   Hermite \(d=3\): \(0.6299693752\), \(1.6240992547\), \(4259\); mixture: \(0.3303169996\),
   \(0.6217950265\), \(22.2425\); library rule: \(0.7433628059\), \(0.5570005384\). Routes agree to
   \(\le10^{-14}\); the recorded six-decimal values are reproduced to rounding and the
   researcher's full-precision artifact to \(\le2\cdot10^{-15}\); \(E[S]=0\), \(E\psi=0\),
   \(E N_1=d\) to \(10^{-15}\). The library rule refitted from the recipe (training seed 11,
   \(n=4000\), D-exchange seed 7, 8 initializer restarts) reproduces the three recorded
   \(x\)-cuts bit for bit, with zero label disagreements on 9600 nodes. **Verified.**
7. *Coverage:* fresh `SeedSequence([20260906, 1001, law, n])`, 4000 replicates per size.
   Bounded mixture 0.9455, 0.953, 0.950, 0.9495 (recorded 0.950, 0.945, 0.956, 0.945),
   \(\mathrm{SD}(\hat\eta_D)=\sigma/\sqrt n\), \(\hat\sigma/\sigma\) 1.08 → 1.003, no skew, \(n\cdot\)bias
   ≈ 2.5. Hermite \(d=2\): 0.689, 0.816, 0.887, 0.925 (0.681, 0.813, 0.892, 0.937); library rule:
   0.819, 0.851, 0.900, 0.920 (0.816, 0.861, 0.889, 0.927); Hermite \(d=3\): 0.400, 0.5645,
   0.731, 0.8465 (0.399, 0.556, 0.733, 0.827), with \(\hat\sigma/\sigma\) 0.55 → 0.95 (\(d=2\)) and
   0.24 → 0.75 (\(d=3\)) — the recorded under-estimation, reproduced. Arc law: 0.996, 0.9985,
   0.998, 0.999, \(n\cdot\)bias 1.40–1.42, \(n\cdot\)SD 1.55–1.61, \(\hat\eta_D\ge\eta_D\) in 85–87%.
   No regular interval left \([0,1]\); empty cells in 7 of 64,000 regular replicates (all
   \(n=100\)). **Replicates.**
8. *Singular endpoint:* 4000 replicates at \(n=300,1000,3000,10000\) against \(10^6\) draws of
   the limit law. `endpoint2` (\(r=1\)): \(\sqrt n\,\)mean\(\hat\eta_D\) 0.633, 0.650, 0.636 at
   \(n\ge1000\) vs limit mean 0.6373 (closed form \(2/\pi=0.6366\)); quantiles (10/50/90) at
   \(n=10^4\) 0.104, 0.537, 1.313 vs 0.100, 0.539, 1.314; coverage of \(0\) 0.946, 0.9455, 0.951.
   `endpoint3` (\(r=2\)): \(n^{1/3}\)mean 0.569, 0.535, 0.527, 0.535 vs limit 0.524; quantiles at
   \(n=10^4\) 0.170, 0.517, 0.919 vs 0.164, 0.502, 0.911; coverage 0.755, 0.794, 0.810, 0.798.
   The scaled median half-width \(n^{(d-r)/d}z\hat\sigma/\sqrt n\) is constant (1.56; 0.97), so
   the "same order" statement is measured, and the half-width is 2.9 resp. 1.86 times
   \(\hat\eta_D\). **Limit law confirmed; H3 recorded.**
9. *Literature:* §5 and the dated file. **Method prior art; statement search gap; no
   re-attribution.**
10. *Registry hygiene:* `criterion`, `level`, `dependencies`, `literature`, `implies` consistent
    with the prose. **One defect:** `RETENTION-PLUGIN-COVERAGE-VECTOR` and `popref.json` name
    the bounded mixture and the library rule without the parameters that define them
    (component means \((-1.5,1,0)\), SDs \((0.8,0.6,2.5)\), fractions \((0.35,0.25)\); training
    seed 11, D-exchange seed 7, 8 restarts) — the laws were not reproducible from the record.
    **Patched** into the node's assumptions and into `extra_laws.json`. The "not yet
    independently audited" notes are removed by this audit.

Protocol G items not covered above: *strictness* — irrelevant, smooth in moments; *nuisance
singularity* — out of scope (\(D_s\) is a Schur-complement functional); *hidden compactness* —
none; *first-order-to-finite jumps* — none claimed; the measured \(n\cdot\)bias is still growing
at \(n=3000\) on Hermite \(d=3\) (9 → 16), pre-asymptotic as recorded; *empirical-to-population
jumps* — none, the rule is frozen; *score-estimation error* — enters only through the label
map; *new-event extension* — the frozen rule's own prediction. *Heavy tails* — (A2) is used
twice and is a real constraint in practice; nothing relaxed.

## 10. Algorithmic consequence

None for optimisation. For evaluation: a frozen rule plus an oracle-score evaluation sample
from the reference law yields \(\hat\eta_D\), \(\hat\sigma\) and a Wald interval from cell counts,
cell score sums, and score second and fourth moments — one \(O(nd^2)\) pass, no resampling.
The interval must not be reported when \(\hat I_Z\) is rank-deficient or \(\hat V\) is
singular (the library's projected number is then a different estimator), when \(\hat\sigma=0\),
or when \(K\le d\); for unbounded polynomial-type scores it is liberal at every \(n\le3000\)
tested; for bounded mixture-fraction scores it is nominal from \(n=100\).

## 11. Deployability consequence

The audit authorizes removing the "not independently audited" notes and lets closure step 1
of `OPEN_PROBLEMS.md` (shipping the error bar in `information_report`) proceed as an
engineering step. It does **not** itself change `src/`. Whatever ships must: require an
oracle score on an evaluation sample independent of the fit; refuse the interval at
rank-deficient \(\hat I_Z\), singular \(\hat V\), \(\hat\sigma=0\) or \(K\le d\); state that the
reading as a Fisher retention needs the evaluation law to be the reference law; and carry the
heavy-tail caveat.

## 12. Information-loss consequence

O7 bounds nothing. It equips the library's reported number — the true geometric-mean
D-retention of a frozen rule — with a conditional \(n^{-1/2}\) error bar on held-out
oracle-score data, and separates evaluation uncertainty from proxy bias once an oracle
sample exists. Nothing about the worst retained direction, training retention, refitted
rules, weights or \(D_s\).

## 13. Updated status

- `RETENTION-PLUGIN-CLT-FROZEN-VECTOR`: remains `bridge`, now audited; H1 (fixture), H2, H4
  in the statement and assumptions; `prior_art_found` for the method, statement a search
  gap in the warning.
- `RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE`: remains `project_proved`, now audited; the
  limit law confirmed numerically with a closed form at \(d=2\); H3 in the statement.
- `RETENTION-PLUGIN-COVERAGE-VECTOR`: remains `measured`, now with the fresh-seed
  replication, the limit-law comparison and the law parameters it lacked.
- `CE-O7-ELLIPSOID-ZERO-VARIANCE-001`: verified.
- `OPEN-RETENTION-UNCERTAINTY`: remains `open`; the vector case is now audited.
- `AUDIT-RETENTION-PLUGIN-VECTOR`: new node, `project_proved`.

## 14. Registry patch

`claims/RETENTION-PLUGIN-CLT-FROZEN-VECTOR.json`: `audit:` pointer, second boundary
counterexample, corrected `statement` (H1, H2, H4), hardened `assumptions`, warning without
"not yet independently audited". `claims/RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE.json`:
`audit:` pointer, H3 in statement and assumptions. `claims/RETENTION-PLUGIN-COVERAGE-VECTOR.json`:
`audit:` pointer, law parameters, replication and limit-law comparison.
`claims/OPEN-RETENTION-UNCERTAINTY.json`: vector case marked audited.
`claims/AUDIT-RETENTION-PLUGIN-VECTOR.json`: new. `KNOWN_RESULTS/10-oracle.md`: O7 preamble
and O7.4 carry the audit notes.

## 15. Regression artifacts

- `py/audit_retention_plugin_vector.py` (stages `exact`, `degenerate`, `popref`, `coverage`,
  `singular`, `fixtures`) and
  `AUDITS/artifacts/AUDIT-RETENTION-PLUGIN-VECTOR-001/{exact,degenerate,popref,coverage,singular,fixtures,extra_laws}.json`
  with git revision, script hash, Python and platform.
- `COUNTEREXAMPLES/CE-O7-UNIT-RETENTION-SINGULAR-SAMPLE-001.json` and its catalogue entry.
- `tests/test_research_claims.py::test_o7_audit_unit_retention_singular_sample_fixture` and
  `::test_o7_audit_ellipsoid_identity_holds_on_an_anisotropic_rational_law`.
- `NUMERICAL_EVIDENCE.md` rows `N-VECTOR-AUDIT-EXACT`, `-DEGENERATE`, `-POPREF`, `-COVERAGE`,
  `-SINGULAR`.

## 16. Next dependency-blocking question

None new: the closure programme's next step is engineering (step 1, ship the error bar),
not research. Within `OPEN-RETENTION-UNCERTAINTY` the remainders stand as recorded — refitted
rules, weights, no oracle, \(D_s\), the degenerate limits (now including the joint endpoint
limit of H3) and a second-order correction for heavy-tailed scores.
