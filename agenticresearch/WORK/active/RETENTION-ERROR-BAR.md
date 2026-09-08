# RETENTION-ERROR-BAR — ship held-out oracle-score retention uncertainty

**Programme:** closure step 1 (engineering; P4 evidence) · **Opened:** 8 September 2026 · **Status:** closed 8 September 2026

## Goal

Expose the audited O7 plug-in standard error and two-sided Wald interval for
`geometric_mean_retention` on an independent, equally weighted evaluation sample
of true scores with frozen labels. Done means the public contract, numerical guards,
documentation and full contributor gate are complete. This packet changes library code;
it is the explicitly selected engineering step, not another theorem investigation.

## Why it matters

A retention estimate needs sampling uncertainty. An interval for proxy moments is not
an interval for true Fisher retention. The O8 audit makes that distinction explicit
and removes any suggestion that calibration or AUC alone certifies the missing bias.

## Relevant claims

- `RETENTION-PLUGIN-CLT-FROZEN-VECTOR`, `RETENTION-PLUGIN-CLT-FROZEN-SCALAR`.
- `RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE`, `CE-O7-ELLIPSOID-ZERO-VARIANCE-001`.
- `OPEN-RETENTION-UNCERTAINTY`.
- `SCORE-ERROR-RETENTION-BUDGET`, `SCORE-ERROR-RULE-TRANSFER`,
  `CLASSIFIER-CALIBRATION-SCORE-LIPSCHITZ` and their audit boundaries.

Read each through `py/registry.py show <ID> --deps --proof`, then the cited O7/O8
sections and audits. Do not rederive O7 or reopen optional closure steps 2 or 4.

## Known blockers

The oracle, iid and held-out conditions are caller obligations that an array cannot
prove. Positive empirical rank cannot certify a population eigenvalue floor. Finite
fourth moments and positive asymptotic variance are required for Wald coverage;
heavy tails can cause material finite-sample undercoverage. Singular retained information
and zero variance require an explicit unavailable-interval outcome, not a zero-width
confidence claim. Weighted, refitted, D_s and truth-free proxy guarantees are out of scope.
No O8 audit defect blocks this oracle-only engineering step.

## Recommended starting points

`src/scorequant/information.py` already owns statistics and `information_report`;
`src/scorequant/reports.py` owns immutable report types. Reuse their validation,
Fisher/rank conventions and JSON contracts. Use existing private shared mathematics;
do not add backend branches, a registry, a copied solver, or a new dependency.

Implementation contract for this packet:

- Add `retention_uncertainty(scores, assignments, *, n_bins=None,
  confidence_level=0.95, rank_rtol=None, execution=None)` and an immutable
  `RetentionUncertainty` report, exported through the package facade. No weights
  argument: this version is for iid equally weighted observations only. Match existing
  `execution` handling for diagnostic computations and canonical public array outputs.
- Report fields: `estimate: float | None`, `standard_error: float | None`,
  `confidence_interval: tuple[float, float] | None`, `confidence_level: float`,
  `n_observations: int`, and `status` from `ok`, `singular_full_information`,
  `singular_retained_information`, `degenerate_variance`. Supply NumPy-style docstrings
  and a JSON-safe `to_dict()` using existing contracts; no NaN/Infinity sentinels.
- Validate N>=2, finite scores, hard labels and bin count through existing helpers,
  finite confidence level strictly between zero and one, and existing rank-rtol rules.
  Empty bins are permitted and contribute zero; preserve uncentred moments.
- If the full moment matrix loses any of the supplied d directions at the selected
  rank threshold, return `singular_full_information` with no estimate, SE or interval.
  Do not silently switch the target to a projected subspace.
- If the full matrix passes but the retained matrix is numerically rank deficient,
  return `singular_retained_information`, estimate zero, no SE or interval. Apply the
  guard before determinant roots/inverses so roundoff cannot invent a regular endpoint.
- Otherwise compute the same full-rank estimate as `information_report`. With empirical
  cell means c_z and moments I,V, evaluate
  ψ_i=(η/d)[2 s_iᵀI⁻¹c_z−c_zᵀI⁻¹c_z−s_iᵀV⁻¹s_i],
  σ_hat²=mean(ψ_i²), SE=sqrt(σ_hat²/N). Use stable solves and existing private
  numerical primitives. Do not center the score rows or fit anything on evaluation data.
- Suppress Wald coverage when σ_hat² vanishes numerically: use a scale-aware rounding
  guard on cancellation in the influence expression, document the chosen threshold,
  and return `degenerate_variance`, SE zero and interval None. Keep this an explicit
  numerical guard, not a theorem that empirical degeneracy proves population degeneracy.
- For `ok`, return the untruncated two-sided Wald interval η±z_(1−alpha/2)SE.
  Do not clip to [0,1], bootstrap, invent bias correction, or attach this automatically
  to training/validation histories. Explicitly document possible out-of-range endpoints.

## Required deliverables

Public diagnostic and report; deterministic tests; `docs/api.md` usage/limitations;
one held-out walkthrough sentence and example call; one concise entry in
`docs/decisions.md`; update the existing `docs/roadmap.md` gate. No parallel planning files.
Correct the existing proxy-moment `information_report` docstring's variance shorthand
as part of these documentation changes: the library uses uncentred second moments.

Required tests: scalar agreement with O6; vector agreement with the audited O7
influence calculation; agreement of full-rank point estimates with `information_report`;
reparameterization, event-order and bin-relabel invariance; empty bins; malformed
inputs/confidence levels; singular full and retained information; the audited
zero-variance ellipsoid/lossless examples; JSON serialization; available execution
configurations under the shared conformance pattern. Do not require split-row-duplication
invariance of the SE: duplicating rows changes the asserted iid sample size.
Use a seeded coverage experiment with an MC-error-based tolerance for a regular bounded
law, and reproduce a heavy-tail limitation without a flaky universal coverage assertion.

The guide must say, in plain language: this interval covers sampling variability for
a frozen rule evaluated on independent true-score data under O7's conditions. Proxy
scores add a separate reporting bias; O8 can bound it only with truth-dependent error
and conditioning assumptions. AUC or calibration alone does not certify that bias.

Run the research README verification block for any research-record changes, then all
contributor commands: locked uv sync, Ruff check/format check, ty, full parallel float64
pytest, float32 tests, strict MkDocs, and package build. No Lean changes are planned.

## Stop conditions

Delivered and verified, or reduced to a precisely named implementation/contract blocker.
Do not ship an interval for unsupported cases to avoid an unavailable result. Record the
Outcome, retire this packet when closed, and update the programme's status and step-1 gate.
Pushes need explicit owner authorization; do not merge, tag, publish or deploy by default.

## Next dependency-blocking question

After step 1: can manuscript v10 incorporate the audited O6–O8 results without reviving
refuted calibration, affine-invariance, singular-Wald or novelty claims?
This is closure step 5; `OPEN-RETENTION-UNCERTAINTY` still holds the deferred statistical
extensions, which belong in future work rather than the active queue.

## Outcome

**Delivered.** `retention_uncertainty(scores, assignments, *, n_bins=None, confidence_level=0.95,
rank_rtol=None, execution=None)` and the immutable `RetentionUncertainty` report, exported through
the package facade, on branch `codex/retention-error-bar`. The estimate is the uncentred O7 plug-in
(identical to `information_report` on a full-rank sample); the standard error is the O7.3
influence-function estimate evaluated in the \(\hat V\)-whitened coordinates
(\(\hat V^{-1}=WW^\top\), \(\hat I_Z^{-1}=WR^{-1}W^\top\), one symmetric solve per cell);
the interval is the untruncated two-sided Wald interval with the standard-library normal quantile.

**Guards, in order.** (1) `fisher_transform` on \(\hat V\) at the selected `rank_rtol`: any lost
direction returns `singular_full_information` with no estimate, standard error or interval
(`CE-O7-UNIT-RETENTION-SINGULAR-SAMPLE-001` is the pinned witness where `information_report`
projects and reports 1). (2) The library's scale-free eigenvalue rank test on
\(R=W^\top\hat I_Z W\), taken before any determinant root or solve, returns
`singular_retained_information` with estimate 0 and no interval; testing \(R\) rather than
\(\hat I_Z\) keeps the verdict reparameterization-invariant. (3) A cancellation guard on the
influence bracket: with \(B_i\) the bracket and \(M_i\) the sum of the absolute values of its three
terms, `degenerate_variance` (standard error 0, no interval) is returned when
\(\mathrm{rms}(B)\le\tau\,\mathrm{rms}(M)\) with \(\tau\) the dtype rank threshold (1e-10
float64, 1e-5 float32). The guard is documented as numerical, not as a population statement;
`CE-O7-ELLIPSOID-ZERO-VARIANCE-001` (eight equal-weight atoms) and the lossless three-cell law both
land there on both backends.

**Tests** (`tests/test_retention_uncertainty.py`, plus conformance and float32 cases): O6 closed
form at d = 1; exact rational O7 influence values (helper moved to `tests/_oracles.py`) on a sample
with ties, a duplicate atom, a singleton and an empty declared cell, with \(\sum_i\hat\psi_i=0\);
agreement with `information_report` in d = 2, 3 on both backends; reparameterization, row-order and
relabel invariance, with row duplication pinned as *not* invariant; empty bins; malformed inputs
and confidence levels; both singular statuses; both zero-variance examples; strict JSON for every
status. Seeded coverage on a bounded nine-atom law (N = 200, 400 replicates, NumPy backend): 0.9475
against 0.95 within three Monte Carlo standard errors. Heavy-tail limitation on Student-t(3) under
the sign rule, exact \(\eta=4/\pi^2\): coverage 0.545 at N = 200 (0.46 at N = 100), asserted as
below nominal by more than three Monte Carlo standard errors and below 0.75 at the pinned seed.

**Limitations (unchanged from the packet).** Oracle scores, iid equally weighted rows and a frozen
rule are caller obligations. No weights, no refitted rules, no profiled \(D_s\), no truth-free
statement, no bias correction, no bootstrap; endpoints are not clipped. A positive empirical rank
does not certify a population eigenvalue floor. Proxy scores add a reporting bias the interval
never measures (`SCORE-ERROR-RETENTION-BUDGET`); AUC or calibration alone does not certify it.

**Records.** `docs/api.md` section "Held-out retention uncertainty"; one sentence and an executed
call in `docs/user-workflow.md`; ADR 0039; roadmap row D and its deferral sentence; `CHANGELOG.md`;
the `information_report` docstring now states the uncentred convention. No claim node changed;
`OPEN-RETENTION-UNCERTAINTY` keeps the deferred extensions. No Lean change.
