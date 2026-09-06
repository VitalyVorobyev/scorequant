# AUDIT-SCORE-ORACLE-ROBUSTNESS — independent audit of O6

**Programme:** P2 (target claim belongs to P4) · **Opened:** 5 September 2026 · **Closed:** 5 September 2026 · **Status:** completed
**Source:** branch `score-oracle-robustness` at `a5f905e` (unmerged; check it out before starting)

## Goal

Independently verify, harden, refute, or reduce the O6 complex recorded in
`KNOWN_RESULTS/10-oracle.md`: `RETENTION-PLUGIN-CLT-FROZEN-SCALAR` (status
`bridge`) and its measured companion `RETENTION-PLUGIN-COVERAGE-DOOR3`, together
with the same commit's patch of `OPEN-RETENTION-UNCERTAINTY`. The auditor did not
produce O6 and will not trust its instrument, its protocol-G notes, or its
literature round.

O6 is a **positive** result with a product consequence: it says a user with an
independent oracle-score evaluation sample can put a valid \(n^{-1/2}\) Wald
error bar on the true retention of a frozen rule, and that the proxy
surrogate's discrepancy is bias outside that bar. It was born at `bridge`
(delta method). It is the first result of programme P2 and the intended basis
for any future uncertainty surface in the library, so `protocols/audit.md`
applies: no library use, no manuscript citation and no promotion before this
audit closes.

The deliverable is the 16-item report `AUDITS/AUDIT-SCORE-ORACLE-ROBUSTNESS-001.md`
plus the registry, numerical, literature and manuscript consequences the
protocol requires.

## Independence contract

- Source frozen at `a5f905e`. The audit session has no access to the
  researcher's transcript or plan file.
- Do **not** import, extend or copy `py/score_oracle_retention_uncertainty.py`.
  Build `py/audit_score_oracle_retention_uncertainty.py` independently:
  exact `fractions.Fraction` arithmetic where a claim is a finite identity,
  closed-form Gaussian algebra where it exists (see attack 6), an independent
  quadrature route where it does not, and fresh Monte Carlo seeds.
- Recheck the imported hypotheses of `FI-QUANT-IDENTITY` (status
  `literature`) and `PROXY-TRUE-RETAINED-FI` (status `bridge`) without
  re-proving them; open a separate audit task if either appears wrong.
- `LITERATURE/audits/RETENTION-PLUGIN-CLT-FROZEN-SCALAR-5-September-2026.md`
  is comparison material only. Run a fresh theorem-level prior-art pass.
- Scope is O6.1–O6.5 and the two claim nodes. Vector scores, refitted rules,
  weights, bootstrap comparisons and any public API are out of scope; flag
  them, do not chase them.
- Do not edit `src/`.

## Attack plan

1. **The empty-cell convention versus the delta method.** O6.2 proves the CLT
   for \(g(\bar T)\) and transfers it to \(\hat\eta\) with \(0/0:=0\) through
   the event that every cell is nonempty. Verify that the transfer argument
   is complete (weak convergence of two sequences that agree with probability
   \(\to1\)), that O6.1 and the library's `information_report` really return
   the same number when a cell is empty (check `scatter_bin_statistics` for
   zero-count cells), and that the identity \(\hat\eta=1-\mathrm{RSS}/\mathrm{TSS}\)
   holds in exact rational arithmetic on small samples with ties, duplicates,
   a singleton cell and an empty cell.
2. **The gradient algebra.** Recompute \(\nabla g\) and the reduction
   \(\nabla g^\top(T-\theta)=\psi\), including the cancellation
   \(\eta-2\eta+\eta=0\) and the identity \(2c_ZS-c_Z^2=S^2-(S-c_Z)^2\).
   Verify the covariance form
   \(\sigma^2=[\operatorname{Var}N_1-2\eta\operatorname{Cov}(N_1,S^2)+\eta^2\operatorname{Var}S^2]/v^2\)
   against \(E[\psi^2]\) symbolically and on an exact atomic law. Confirm the
   sign of the cross term.
3. **Consistency of \(\hat\sigma^2\) (O6.3).** The proof asserts that
   \(n^{-1}\sum\hat\psi_i^2\) is a continuous function of within-cell
   empirical moments up to order four. Write the expansion out; check that no
   moment above order four is needed, that \(\hat v^{-2}\) and \(\hat c_b\)
   are continuous at the limit under (A1), (A3), and that the strong law is
   applied to finitely many iid averages only. Decide whether the result is
   a.s. or in probability as stated.
4. **The \(\sigma^2=0\) characterisation (O6.4).** Verify "\(\psi=0\) a.s. iff
   \(S\mid Z=b\) is supported on the roots of \((s-c_b)^2=(1-\eta)s^2\)". Treat
   separately \(\eta=0\) (a linear equation), \(c_b=0\) with \(\eta>0\) (a
   double root at \(0\)), and \(\eta=1\). Check that the root set is
   consistent with \(E[S\mid Z=b]=c_b\) — construct an explicit two-atom law
   with \(0<\eta<1\) and \(\sigma^2=0\) in exact arithmetic, or prove none
   exists and correct the statement. Test the claim "any atomless cell of
   positive probability implies \(\sigma^2>0\)".
5. **Retention reading and conditioning.** O6.0 identifies \(\eta\) with
   \(\operatorname{Var}(E[s\mid q(\hat s)])/\operatorname{Var}(s)\) via
   \(E[S]=0\) and `FI-QUANT-IDENTITY` "with \(q\) fixed in \(\theta\)". Check
   that the evaluation law must be \(P_{\theta_0}\), that a frozen rule built
   from \(\hat s\) (itself fitted at \(\theta_0\)) satisfies the fixed-in-\(\theta\)
   hypothesis, and that nothing in O6.2–O6.4 secretly uses \(E[S]=0\). State
   what the interval estimates when the evaluation sample comes from a
   different law than the score's reference point.
6. **Independent population references.** For the door3 rung the cuts in
   \(\hat s\)-space are monotone images of logit thresholds, and the logit is
   a quadratic in \(x\); the cell boundaries in \(x\) are therefore roots of
   quadratics, and \(p_b\) and \(m_b\) are closed-form differences of Gaussian
   CDFs (note \(s(x)f(x)=\phi_{\rm sig}(x)-\phi_{\rm bkg}(x)\)). Derive the
   boundaries and \(p_b,m_b\) in closed form, compute \(v\) and \(\sigma^2\)
   by an independent quadrature, and compare with the recorded
   \(\eta=0.893663\), \(\sigma=0.235410\), \(\tilde\eta=0.967064\). Any
   disagreement beyond \(10^{-10}\) is a finding.
7. **Independent coverage replication.** Rebuild the frozen rule from the
   example module's public functions, draw fresh seeds, and replicate the
   coverage table at \(n=100,300,1000,3000\) with at least 2000 replicates.
   Then probe two boundaries the researcher did not: the rung
   `n_per_class = 300` (\(\eta\approx0.966\), near the edge, small
   \(\sigma\)) and a synthetic law with two atoms per cell built in attack 4
   (\(\sigma^2=0\)): report what the Wald interval does there. Record every
   seed, revision and script hash under
   `AUDITS/artifacts/AUDIT-SCORE-ORACLE-ROBUSTNESS-001/`.
8. **Second-order claims in O6.7.** The researcher attributes the \(n=100\)
   shortfall to the correlation between \(\hat\sigma\) and \(\hat\eta\) and
   reports \(n\cdot\)bias \(\approx0.3\). Check these readings against your
   replication; they are measured notes, so the outcome is "consistent" or
   "not reproduced", not a theorem verdict.
9. **Literature.** Fresh triangulation on: the asymptotic variance of the
   (centred and uncentred) sample correlation ratio under non-normality
   (Kendall & Stuart Vol. 2 was not located to a section); delta-method
   intervals for ANOVA effect sizes without normality; influence functions
   of ratio-of-quadratic-form statistics. Verify or correct the theorem and
   chapter numbers the researcher recorded as unverified (van der Vaart
   Thm 3.1, Serfling §3.3 Thm B, Cramér Ch. 28, Hampel et al. section). If a
   direct statement of O6.2 exists, re-attribute; if not, keep
   `prior_art_found` for the method and say so in the report.
10. **Registry hygiene.** Check the two nodes' `criterion`, `level`,
    `dependencies`, `implies`, `literature`, `assumptions` and `warning`
    fields against the prose; check the `OPEN-RETENTION-UNCERTAINTY` patch
    lists the remainders correctly and that no `implies` edge overclaims.

## Required deliverables

- `AUDITS/AUDIT-SCORE-ORACLE-ROBUSTNESS-001.md` with all 16 items of
  `protocols/audit.md` in order.
- `py/audit_score_oracle_retention_uncertainty.py` and provenance-complete
  artifacts under `AUDITS/artifacts/AUDIT-SCORE-ORACLE-ROBUSTNESS-001/`;
  ledger rows `N-ORACLE-AUDIT-*` in `NUMERICAL_EVIDENCE.md`.
- Registry patches: `audit:` pointer and hardened `assumptions` on both
  nodes; any status change justified in the report; `reindex` and
  `validate` clean.
- Any boundary failure minimised into `COUNTEREXAMPLES/` with a fixture and a
  pin in `tests/test_research_claims.py`.
- A dated literature audit under `LITERATURE/audits/` and registry entries for
  any new source; corrected verification notes on existing keys.
- A `manuscripts/README.md` staleness note if the audit re-attributes or
  contradicts anything.
- The packet moved to `WORK/completed/` with an Outcome section: verdict,
  changed claim ids, evidence, limitations, one next action.

## Stop conditions

1. **Verified:** O6.1–O6.4 hold as stated (or with hardened assumptions
   recorded in the nodes); the population references and coverage replicate;
   the bridge status stands and the nodes carry the `audit:` pointer.
2. **Refuted:** an exact counterexample to O6.1, O6.2, O6.3 or the
   \(\sigma^2=0\) characterisation, serialised and pinned; the node is
   downgraded and the dependent patch of `OPEN-RETENTION-UNCERTAINTY` reverted.
3. **Reduced:** a precise missing step or assumption is named; the statement
   is narrowed to what is proved and the gap is recorded as a new open node.

Do not close on a coverage replication alone: the verdict is about the proof.

## Next dependency-blocking question

If verified: `OPEN-RETENTION-UNCERTAINTY`, vector case — the geometric-mean
retention \((\det\hat R)^{1/d}\) under a frozen rule as a smooth matrix
functional of the same cell moments, its matrix influence function, and where
\(\log\det\) degenerates. If reduced or refuted: whichever step failed, as its
own packet.

## Outcome

**Verified with hardened assumptions.** Every component received an
independent decision (`AUDITS/AUDIT-SCORE-ORACLE-ROBUSTNESS-001.md`, sixteen
items); the instrument `py/audit_score_oracle_retention_uncertainty.py` was
built without reading the researcher's.

- **O6.1** — verified exactly (ties, duplicates, singleton, empty declared
  cell, \(\hat\eta\in\{0,1\}\)); library agreement to \(2\cdot10^{-16}\);
  all-zero scores refused.
- **O6.2** — verified; the empty-cell transfer is complete and is replaced
  by applying the delta method to the estimator's own everywhere-defined
  functional \(\phi\); gradient reduction and covariance form exact on 60
  random rational laws.
- **O6.3** — verified; the order-four expansion written out and checked
  exactly; a.s. is correct (Borel–Cantelli makes empty cells transient).
- **O6.4** — Wald statement verified. **H1:** the \(\sigma^2=0\)
  characterisation was wrong at \(\eta=0\): the root equation is the
  identity there, every law with vanishing cell means has \(\psi\equiv0\),
  atomless or not (`CE-O6-ETA-ZERO-MULTIATOM-VARIANCE-001`, atomless probe).
  For \(0<\eta<1\) the two-atom laws exist explicitly (roots
  \(c_b/(1\mp\sqrt{1-\eta})\), forced weight \(\eta/(2(1+\sqrt{1-\eta}))\);
  \(\eta=3/4\) witness) and there \(\hat\eta\ge\eta\) always.
- **O6.5** — verified with **H2:** the retention reading needs the evaluation
  law to be \(P_{\theta_0}\); the CLT never uses \(E[S]=0\).
- **Population references** — closed-form boundaries (quadratic roots of the
  logit) and Gaussian-CDF cell moments reproduce the recorded values to
  rounding and the researcher's artifact to \(2\cdot10^{-15}\).
- **Coverage** — fresh seeds replicate O6.7 (0.918, 0.937, 0.952, 0.946);
  rung 300 under-covers to 0.896 at \(n=100\) and its interval contains the
  proxy value at practical \(n\); both \(\sigma^2=0\) laws give a
  *conservative* interval (coverage \(\to1\), width \(O(1/n)\)).
- **Literature** — method is textbook prior art (delta method, influence
  variance); the exact fixed-partition uncentred statement is a search gap,
  not novelty; no re-attribution. Theorem-number verification recorded in
  `LITERATURE/audits/AUDIT-SCORE-ORACLE-ROBUSTNESS-5-September-2026.md`.

No `src/`, public API or example change; the audit removes the "not
independently audited" notes and nothing more.

## Artifacts

- `AUDITS/AUDIT-SCORE-ORACLE-ROBUSTNESS-001.md`; `claims/AUDIT-SCORE-ORACLE-ROBUSTNESS.json`
- `py/audit_score_oracle_retention_uncertainty.py`;
  `AUDITS/artifacts/AUDIT-SCORE-ORACLE-ROBUSTNESS-001/{exact,popref,coverage,fixtures}.json`
- `COUNTEREXAMPLES/CE-O6-ETA-ZERO-MULTIATOM-VARIANCE-001.json` (+ catalogue entry)
- `tests/test_research_claims.py`: two new exact regressions
- patched: `claims/{RETENTION-PLUGIN-CLT-FROZEN-SCALAR,RETENTION-PLUGIN-COVERAGE-DOOR3,OPEN-RETENTION-UNCERTAINTY}.json`,
  `KNOWN_RESULTS/10-oracle.md`, `OPEN_PROBLEMS.md` (OP27), `manuscripts/README.md`,
  `NUMERICAL_EVIDENCE.md` (four `N-ORACLE-AUDIT-*` rows),
  `LITERATURE/{audits,topics/08-plug-in-asymptotics.md,graph.json,reviewed.md}`

## Validation

`registry.py reindex` and `validate` clean; `tests/test_research_claims.py`
and `tests/test_research_registry.py` green; `ruff check` and `ruff format
--check` clean on the gated tree.

## Next dependency-blocking question

The vector case: `WORK/completed/RETENTION-PLUGIN-VECTOR.md`.

