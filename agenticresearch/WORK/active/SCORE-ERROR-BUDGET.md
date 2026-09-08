# SCORE-ERROR-BUDGET — what an estimated score costs a frozen rule, to first order

**Programme:** P2 (closure step 3 of `OPEN_PROBLEMS.md`) · **Opened:** 8 September 2026 · **Status:** completed

## Goal

Population level, frozen rule, one atomic or continuous law. True score \(s\) with
\(V=E[ss^\top]\succ0\); proxy \(\hat s=s+e\); **Fisher-whitened score error**
\(\varepsilon^2=E[e^\top V^{-1}e]\). Frozen rule \(q\) on score space; labels
\(\hat Z=q(\hat s)\). Decide, each with an exact fixture:

- **B1** (directional reporting budget, no margin condition): for every direction \(a\),
  \(|a^\top(\tilde I_Z-I_Z)a - a^\top E_Za|\le2\sqrt{(a^\top I_Za)(a^\top E_Za)}\) with
  \(E_Z=E[e_Ze_Z^\top]\preceq E[ee^\top]\), \(e_Z=E[e\mid \hat Z]\); the same for \(\tilde V-V\).
- **B2** (D-retention reporting budget): with \(\varepsilon_R^2=E[e_Z^\top I_Z^{-1}e_Z]\),
  \(|\log\tilde\eta_D-\log\eta_D|\le\frac{2}{\sqrt d}(\varepsilon_R+\varepsilon)+\) an explicit
  second-order remainder; first-order constant sharp; \(\varepsilon_R\le\varepsilon/\sqrt{\rho_{\min}}\)
  and the \(\rho_{\min}\) dependence cannot be removed; the bound holds in both directions
  (truth from the reported \(\tilde\eta_D\), \(\tilde\rho_{\min}\), \(\tilde\varepsilon\)).
- **B3** (rule transfer under a margin condition): \(Z=q(s)\) versus \(\hat Z=q(\hat s)\), both
  scored by \(s\): mislabel mass \(\pi\le\inf_t\{M(t)+\varepsilon^2/t^2\}\) with
  \(M(t)=P(\operatorname{dist}_V(s,\partial q)\le t)\), and the Loewner sandwich
  \(R_{\hat Z}\succeq R_Z-\Gamma\), \(R_Z\succeq R_{\hat Z}-\hat\Gamma\) with
  \(\Gamma=2E[(ww^\top+c_{\hat Z}c_{\hat Z}^\top)\mathbf 1_{Z\ne\hat Z}]\); no margin, no budget.
- **B4** (OP18, one inequality): the mixture-fraction score is Lipschitz in the posterior on
  the simplex with explicit \(L(\theta_0,\pi)\), so \(\varepsilon\) is bounded by the \(L^2\)
  posterior error, which is the excess Brier score; measured reliability lower-bounds it.
- **B5** (AUC is not enough): a monotone distortion of a scalar score leaves every threshold
  rule, its labels and the true retention unchanged and moves the reported retention.
- **B6** (deployment corollary): the true retention of the rule fitted on the proxy against the
  oracle rule, as a product of B2 budgets and one B3 budget.

"Done" is decidable: each of B1–B6 is proved with an exact-arithmetic check on the default
falsification search, refuted with a minimised fixture, or reduced with the missing statement
named.

## Why it matters

The number the library prints for a classifier provider is the *proxy* retention
(`information_kind == "supplied_score_surrogate"`); O4 says the true retained information is a
different quantity. This packet gives the gap a formula in terms of a score-error scale the user
can bound from a classifier's calibration, and says which rules are fragile (small
\(\rho_{\min}\)). It is the "classifier-quality requirement beyond AUC" of the closure programme,
and it fixes what closure step 1's docs sentence must say (sampling error bar from O7 *and*
proxy bias from here).

## Relevant claims

`PROXY-TRUE-RETAINED-FI` (O4), `REPRESENTATION-QUANTIZATION-LOSS` (O5),
`CLASSIFIER-MIXTURE-SCORE-FORMULA`, `CLASSIFIER-RATIO-ORACLE` (O3), `FI-QUANT-IDENTITY`,
`INFO-RETENTION-SPECTRUM`, `INFO-D-EFFICIENCY`, `D-REPARAM-INVARIANCE`,
`RETENTION-PLUGIN-CLT-FROZEN-VECTOR` (O7, the sampling companion); targets
`OPEN-SCORE-PERTURBATION` (OP17), `OPEN-CLASSIFIER-CALIBRATION-FI` (OP18).

## Known blockers

- \(\rho_{\min}\) and \(\varepsilon\) both need the truth; the deployable form must be stated
  from the proxy side (reported \(\tilde\rho_{\min}\), error in the proxy metric).
- The excess Brier score needs the Bayes posterior; only the reliability term is observable
  without truth. OP18 can only be half-closed.
- The library's rank projection (`rank_rtol`) is outside every statement, as in O7.
- Unbounded scores: the mislabel-to-retention step loses a square root.

## Recommended starting points

Conditional Cauchy–Schwarz in the \(I_Z^{-1}\) inner product (the identity
\(E[c_Z^\top I_Z^{-1}c_Z]=d\)); the rank-one Loewner inequality
\((c+e)(c+e)^\top\succeq(1-t)cc^\top-\frac{1-t}{t}ee^\top\); Pythagoras for conditional
expectations in matrix form; the audit instrument
`py/audit_score_oracle_retention_uncertainty.py` for the door3 closed forms.

## Required deliverables

Instrument `py/score_error_budget.py` (exact selftest, fixtures, door3, synthetic2d); fixtures
`CE-SCORE-ERROR-RHO-MIN-NECESSARY-001`, `CE-SCORE-ERROR-BOUNDARY-ATOM-001`,
`CE-AUC-INVARIANT-PROXY-RETENTION-001` pinned in `tests/test_research_claims.py`; claims
`SCORE-ERROR-RETENTION-BUDGET`, `SCORE-ERROR-RULE-TRANSFER`,
`CLASSIFIER-CALIBRATION-SCORE-LIPSCHITZ`, `SCORE-ERROR-DOOR3-MEASURED`; section O8 of
`KNOWN_RESULTS/10-oracle.md`; ledger rows; literature gaps; `OPEN_PROBLEMS.md` status.
Audit prompt for the next session (closure step 3 gate is "verdict plus audit").

## Stop conditions

All of B1–B6 decided, or a counterexample to B1/B2 on the default search (then minimise and
stop), or a reduction naming one missing statement.

## Outcome (8 September 2026)

**Verdict: proved**, at the first order asked for; audit owed (the step's gate).

- **B1** proved (conditional Cauchy–Schwarz; O8.1). A direction retaining \(\rho\) is
  misreported by at most \(2\sqrt\rho\,\varepsilon_Z+\varepsilon_Z^2\).
- **B2** proved, and sharper than planned (O8.2): the reporting gap
  \(\log(\tilde\eta_D/\eta_D)\) is an *exact* two-term expansion with a curvature remainder —
  an alignment term bounded by \(\frac2d\sqrt{d-\operatorname{tr}R}\,(\dots)\) and a
  spurious-information term \((\varepsilon_R^2-\varepsilon^2)/d\) with
  \(\varepsilon_R^2\le\varepsilon_Z^2/\rho_{\min}\). The first-order constant \(2\sqrt d\) is
  attained; no bound in \(\varepsilon\) alone exists (`CE-SCORE-ERROR-RHO-MIN-NECESSARY-001`);
  the budget is free of the proxy's affine class (\(\varepsilon_{\rm aff}^2=\sum(1-r_i^2)\),
  canonical correlations); the bracket also runs from the reported side. Two facts not in
  the plan: the \(\sqrt{\text{trace loss}}\) factor (a nearly lossless rule reports
  correctly whatever the proxy, at first order) and the affine reduction (found when the
  door3 classifier's 50% score error produced only an 8% reporting error).
- **B3** proved (O8.3): Markov-plus-margin mislabel bound, exact Loewner sandwich
  \(I_{\hat Z}\succeq I_Z-\Gamma\); the margin is necessary
  (`CE-SCORE-ERROR-BOUNDARY-ATOM-001`). Weak in practice: informative only for
  \(\varepsilon\lesssim0.1\).
- **B4** one inequality plus the Brier identity (O8.4, bridge): calibration error is a
  *lower* bound on the score error; the excess Brier score on simulation is the requirement;
  the resolution gap is not certifiable without truth. OP18 half-closed.
- **B5** fixture (`CE-AUC-INVARIANT-PROXY-RETENTION-001`): same ROC, different report.
- **B6** deployment corollary (O8.6), needing proxy-global optimality of \(\hat q\).
- **Measured** (O8.7): door3 rungs — crude budget over-covers 24×, the sharper alignment
  bound 3–6×, \(T_1\) tracks the gap; synthetic \(d=2\) — the weak-direction proxy inflates
  the report 3–9× more than the strong-direction one at equal \(\varepsilon\); every row inside
  the bracket. Exact selftest: 22 942 checks, 0 failures.

Records: claims `SCORE-ERROR-RETENTION-BUDGET`, `SCORE-ERROR-RULE-TRANSFER`,
`CLASSIFIER-CALIBRATION-SCORE-LIPSCHITZ`, `SCORE-ERROR-DOOR3-MEASURED`; patches to
`OPEN-SCORE-PERTURBATION`, `OPEN-CLASSIFIER-CALIBRATION-FI`, `PROXY-TRUE-RETAINED-FI`;
section O8; three fixtures with tests; four ledger rows; `LITERATURE/gaps.md`;
`OPEN_PROBLEMS.md` status and step 3; `manuscripts/README.md`. No `src/` change.

**Product consequence for closure step 1.** The docs sentence must carry two sources of
uncertainty: the O7 sampling bar, and the O8 proxy bias — first order
\(\frac2d\sqrt{d-\operatorname{tr}R}(\dots)+\varepsilon_Z^2/(d\rho_{\min})\) in the classifier's
whitened score error, not controlled by AUC. **For step 4 (OP23):** a score shift under the
reference law is an O8.2 instance; the law shift is what remains.

## Next dependency-blocking question

`OPEN-SCORE-PERTURBATION`: a **uniform-over-rules** reporting budget — a bound on
\(\sup_{q\in\mathcal Q}|\log\tilde\eta_D(q)-\log\eta_D(q)|\) over the Mahalanobis rule class
with a floor on \(\rho_{\min}\) — which is what turns O8.6 from a statement about the
proxy-global optimum into one about the library's exchange-stable solution. Not in the
closure programme; backlog unless the owner moves it.
