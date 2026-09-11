# Open problems — the closure programme and the backlog

**Version:** 5.3 · 11 September 2026
**Rule:** this file selects work. It holds the finite programme that ends the research, a
three-line status, and the backlog of unresolved questions. Established results live in the
claim graph and `KNOWN_RESULTS/`; they are not restated here.

## Status (11 September 2026)

- Active: none. Next: the separate non-blocking HEP template-likelihood assessment (P5).
  Step 5a's review and local corrections are delivered; manuscript dispute/attribution assurance,
  owner freeze and human gates remain open in development phase H. Steps 2 and 4 remain dropped.
- Done: step 5a (11 September 2026, `AUDITS/AUDIT-D-COMPILE-TOLERANCE-001.md`,
  two exact fixtures, narrowed guarantee and publication disposition in development phase H); step 0 (O7 audit, PR #59); step 3's corrected verdict plus independent audit
  (`AUDITS/AUDIT-SCORE-ERROR-BUDGET-001.md`); step 1 shipped on 8 September 2026 as
  `retention_uncertainty` (PR #66, ADR 0039); step 5, manuscript v10 (merged PR #68, 10 September 2026,
  `manuscripts/README.md`), harvesting O6–O8 and both audits, with §10.2 absorbing this backlog and
  the bounded novelty pass `LITERATURE/audits/NOVELTY-STEP5-10-September-2026.md`. The formal-D
  packet is retired as partial, with its unresolved obligations preserved under the
  formal-verification backlog below.
- Parked: OP31 and `WORK/active/DS-TILT-DUAL-EXACT-COMPLEXITY.md`; formal residue below,
  the informal positive-tolerance contract audit is completed, while its Lean proof remains parked.
  Neither is selected work. P1 deployment verdict closed on 1 September 2026
  (`KNOWN_RESULTS/05b-ds-bridge.md`).

## Closure programme

The current research programme ends when the steps below are done. Each theorem step is one
session and one verdict (proved, refuted, or reduced with the missing statement named); no step
is retried. Every theorem step names its product consequence before work starts. Step 5a reviews
assurance and release obligations, without reopening completed theorem steps. Outside this table,
only the explicitly scheduled P5 assessment is selected follow-up work; it proposes a separate
programme and does not delay closure. Other questions remain backlog.

| Step | Packet | Programme | Product consequence | Gate |
|---|---|---|---|---|
| 0 | Independent audit of O7 (`RETENTION-PLUGIN-CLT-FROZEN-VECTOR`, `RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE`, `CE-O7-ELLIPSOID-ZERO-VARIANCE-001`) | P4 | the reported retention may carry an error bar | audit verdict; PR #59 merged |
| 1 | Ship the error bar: standard error and Wald interval for `geometric_mean_retention` on a held-out oracle-score sample, rank guard at singular \(\hat I_Z\), heavy-tail warning; tests, `docs/api.md`, one walkthrough sentence, one decision record | engineering | users get the number with its uncertainty | **shipped** (8 Sep 2026, `retention_uncertainty`, ADR 0039; full contributor gate passed) |
| 2 | Refitted rules (OP27 remainder): is \(\sqrt n(\hat\eta_n-\eta^*)\) normal when the rule is fitted on the evaluation sample, under a margin condition, or is there an exact counterexample; order of the in-sample optimism | P4 | the docs sentence "evaluate on held-out data" gains a theorem or a counterexample | verdict; optional, drop first |
| 3 | Score-error budget (OP17, first order only): bound the retention loss of a frozen rule under an \(L^2\) score error, one classifier example; OP18 only if it falls out as one inequality | P2 | a classifier-quality requirement beyond AUC | **verdict: proved after correction; audited** (8 Sep 2026, `AUDITS/AUDIT-SCORE-ERROR-BUDGET-001.md`); core bounds retained, unconditional calibration consequences refuted |
| 4 | Parameter mismatch (OP23): second-order local expansion of a frozen rule's retention at \(\theta_0+\delta\), as a diagnostic | P4 | a story for the reference-point assumption | verdict; optional, drop second |
| 5 | Manuscript v10: harvest O6, O7, both audits and step 3; the future-work section absorbs the backlog below | publication | the paper | **merged PR #68** (10 September 2026, `manuscripts/score_space_quantization_article_v10.md`); owner review pending |
| 5a | Bounded v1.0 readiness review, the same session as `docs/roadmap.md` phase H: compilation contract and publication assurance | assurance | verified scope of shipped guarantees and an explicit publication disposition | reviewed 11 September 2026; independent audit and local corrections delivered; manuscript publication blocker remains in phase H |
| 6 | Freeze the current programme: retain backlog and the separate P5 assessment, mark the current research programme closed in the roadmap; release tag under phase H's sequence | — | — | owner decision after step 5a disposition; release actions separately authorized |

Steps 2 and 4 remain dropped. Bin-count theory (P3), the D-core spine (P6), foundations (P7)
and the literature graph (P8) receive no further work in this programme. HEP specialisation (P5)
has a separate assessment below; its open questions are not prerequisites for v1.0.

### Step 5a — pre-freeze assurance (review delivered 11 September 2026)

Review `D-COMPILE-TOLERANCE-GUARANTEE` against the implementation and regression fixtures:
exact versus tolerated agreement, individual relocation gains versus simultaneous reassignment,
duplicate atoms, zero-weight rows and singular directions. The informal contract audit is complete
(`AUDITS/AUDIT-D-COMPILE-TOLERANCE-001.md`); completing its Lean proof is not selected. Any strengthened guarantee needs an independent audit
under `protocols/audit.md` before promotion. Record any missing audit as an unresolved obligation.

Recover the inherited v9 manuscript dispute verdicts from audit history and reconcile their v10
dispositions and remaining attribution qualifications before publication sign-off. Correct the
profiled-certification refusal rationale without adding a solver. Classify each issue as release
blocker, required correction or deferred opportunity, and record actionable outcomes in development
phase H plus the research status here. The session runs the development handoff gate and inspects
formal-proof trust evidence, reporting passed, failed and unrun checks. It does not infer human
acceptance, change claim status, declare closure or authorize publication.

## Backlog

Unresolved questions, grouped by programme. OP numbers are stable ids; claim
`proof_location`s point at these headings. A backlog item is worked only if the owner moves it
into a selected programme. The P5 assessment below scopes a proposal only; it does not select
its underlying theorem or implementation work.

# P2 · SCORE-ORACLE-ROBUSTNESS — estimated scores and classifiers

## OP17. Perturbation theory for estimated scores

Assume \(\|\hat s-s\|_{L^2}\le\varepsilon\) or a stronger bound. Control cell moments, \(I_q\),
the D/\(D_s\) objective, efficiency, and geometric boundaries under a margin condition.
Settled at first order for a frozen rule (closure step 3, 8 Sep 2026, O8): the reporting gap is
an exact two-term expansion — alignment \(\propto\sqrt{d-\operatorname{tr}R}\), spurious
information \(\varepsilon_Z^2/\rho_{\min}\) — with the numerator first-order constant attained and no uniform relative/log bound
from an upper bound on \(\varepsilon\) alone (`SCORE-ERROR-RETENTION-BUDGET`); the boundary displacement is a
Markov-plus-margin mislabel bound with an exact Loewner sandwich (`SCORE-ERROR-RULE-TRANSFER`).
Remaining: uniform-over-rules error control plus a proxy optimization-gap certificate
(uniform control alone cannot make an exchange-stable solution globally optimal), \(D_s\), refitted rules, sharp
second-order constants. Target claim: `OPEN-SCORE-PERTURBATION`.

## OP18. Classifier calibration error → Fisher loss

Relate posterior/ratio calibration error to density-ratio error, score error, pre-quantization
representation loss, and final D/\(D_s\) efficiency. The target is a classifier-quality
requirement beyond AUC. Half settled (8 Sep 2026, O8.4, `CLASSIFIER-CALIBRATION-SCORE-LIPSCHITZ`):
the mixture score is Lipschitz in the posterior, so the whitened score error is bounded by the
excess Brier score, which is reliability plus
the resolution gap (needs truth). The audit requires an injective chart and explicit
constants for a reverse score-error bound; calibration error alone does not imply
retention distortion. Reliability is a population quantity estimated from labelled data,
not an exact finite-data certificate. AUC does not determine reporting error (`CE-AUC-INVARIANT-PROXY-RETENTION-001`).
Remaining: certifying the resolution gap without truth scores (with OP19). Target claim:
`OPEN-CLASSIFIER-CALIBRATION-FI`.

## OP19. Estimating representation versus quantization loss

When truth scores exist only on simulation, devise cross-fitted estimators and uncertainty bars
for \(I_R=\operatorname{Var}(E[s\mid R])\) and \(I_q=\operatorname{Var}(E[s\mid q(R)])\).

# P3 · INFORMATION-BUDGET — how many bins does a target need

## OP14. Sharp D-efficiency versus K

Study \(\eta_D(K)=\sup_{|q|=K}(\det I_q/\det I_{\rm full})^{1/d}\). Find distribution-dependent
or distribution-free bounds and inversion formulas for "how many bins for a target efficiency".

## OP15. High-rate \(K\to\infty\) asymptotics

With \(L=I_{\rm full}-I_q\), expand \(\log\det(I_{\rm full}-L)\) to second order. Working
hypothesis: first order is Fisher-whitened quadratic quantization; D-specific cell-shape effects
appear at second order. Connect to Zador–Gersho high-rate theory.

## OP16. Direction-wise guarantees from determinant efficiency

Given \(\eta_D\), bound \(\lambda_{\min}(R)\) and conversely; find assumptions under which
determinant retention controls worst-direction loss tightly.

# P4 · DEPLOYMENT-ROBUSTNESS — away from the reference point, with error bars

## OP23. Parameter-mismatch degradation

For a quantizer optimized at \(\theta_0\), bound the D/\(D_s\) loss at \(\theta_0+\delta\):
local second-order perturbation results and practical validation metrics.

## OP24. Multi-reference / robust quantization

Expected or minimax objectives over a parameter region: does affine/common-metric geometry
survive, or become a mixture of local metrics?

## OP27. Finite-sample uncertainty for retention estimates

Every retention number the library reports is a point estimate. Develop influence-function or
bootstrap intervals for retention functionals, handling the non-smoothness of hard assignment
at cell boundaries. Settled for a frozen rule on an iid oracle-score sample: scalar O6
(`RETENTION-PLUGIN-CLT-FROZEN-SCALAR`, audited) and vector O7
(`RETENTION-PLUGIN-CLT-FROZEN-VECTOR`, audited 6 Sep 2026, `AUDIT-RETENTION-PLUGIN-VECTOR`), both in `KNOWN_RESULTS/10-oracle.md`.
Shipped as the library diagnostic `retention_uncertainty` (closure step 1, 8 Sep 2026, ADR 0039).
Remaining: rules refitted on the evaluation sample (closure step 2); weights; no oracle; the
profiled \(D_s\) retention; the degenerate limits when \(\sigma^2=0\); a second-order-corrected
interval for heavy-tailed scores. Target claim: `OPEN-RETENTION-UNCERTAINTY`.

# P5 · HEP-SPECIALIZATION — template fits made mathematically explicit

## Scheduled follow-up — HEP template-likelihood research assessment

After the v1.0 readiness review (step 5a / development phase H), run one bounded assessment,
independent of the freeze and release schedule. Start from OP20–22, their claim nodes, the current
HEP study and its committed evidence, and primary literature under `protocols/literature.md`.
Assess four connected questions:

1. Extended-Poisson count/shape accounting and stable parameterization (OP20–21), including
   yield scaling, intensity versus probability scores, and fixed selection assumptions.
2. Auxiliary constraints and profiled bin optimization (OP22): which existing exchange,
   geometry and bound contracts survive adding actual auxiliary information.
3. Finite-template statistics, split/merge decisions and bin-budget tradeoffs under a named
   likelihood; distinguish information loss from changes in the nuisance model between binnings.
4. A controlled benchmark with known scores, separate training/design/evaluation samples,
   and downstream profile-likelihood uncertainty, bias and coverage. Compare existing D and
   profiled rules with classifier baselines before proposing a larger HEP dataset study.

Stop with a **go/no-go recommendation** and a finite programme proposal recorded here under P5:
each proposed session names its product consequence, dependencies, evidence, acceptance criterion
and stop condition. Separate established identities, literature adaptations, unresolved questions
and engineering integration; a search gap is not novelty. A no-go verdict records the reason and
parks the proposal. A go verdict still requires owner selection before derivation or implementation.
This assessment adds no library capabilities, dependencies or automatic v1.0 blockers, and creates
no parallel planning file.

## OP20. Canonical parameterization for linear mixtures

For mixture fractions and extended yields, derive numerically stable score coordinates under
simplex constraints, reference-component coordinates, unconstrained local coordinates, and yield
parameterization. Clarify D invariance and \(D_s\) POI/nuisance transformations.

## OP21. Count + shape information in extended fits

Formalize \(I_{\rm total}=I_{\rm count}+I_{\rm shape}\) for the relevant extended-likelihood
conventions and specify exactly what event hard quantization changes.

## OP22. Systematic template morphing and nuisance scalability

Score and efficient-score construction for calibration, template-shape, normalization,
MC-statistical and correlated nuisance parameters without an impractical score dimension.

# P6 · D-CORE-COMPLETION — the paper's remaining spine

## OP8. Unrestricted D global consistency

Restricted compact affine-max consistency is established. Open: do unrestricted empirical global
D optima converge in value and decision to population global D quantizers under natural
assumptions? Finite geometric realizability may reduce this to a controlled geometric class, but
the metric and centroids are data-dependent and singular boundaries must be controlled.

## OP9. Consistency of exchange-stable D solutions

Do one-point-exchange-stable empirical D quantizers converge to the population stationary set?
What prevents spurious local branches from persisting? Tools: set-valued M-estimation, stability
margins, uniform convergence of move gains.

## OP10. Unrestricted \(D_s\)/E consistency

Finite global optima can be non-geometric for \(D_s\) and E. Does the non-geometric discrepancy
vanish asymptotically, and do global finite values converge to the population values? The
programme also carries the empirical half of the paper story: the controlled D-versus-trace and
k-means benchmark establishing when D differs.

## OP29. Margins beyond conditional centering

Two vector academic branches after DS19: for \(d_\psi>1\), the uniqueness and rigidity theory
for vector-D quantization of the efficient score before transferring DS15's degenerate-attainer
dichotomy; for \(d_\lambda\ge2\) above the centered-sample threshold
\(K\ge d_\psi+d_\lambda+1\), construct or refute a vector-(R) steering mechanism spanning all
nuisance directions (`CE-DS-MARGINS-RANK-VACUITY-001` covers \(K=d_\psi+d_\lambda\)). Do not
reopen the audited scalar DS15, DS18 or DS19 claims. Target claim: `OPEN-DS-MARGINS-NONCENTERED`.

# P7 · FOUNDATIONS — why D is special, complexity, randomization

## OP1. Which concave matrix criteria have finite exchange ⇒ first-order geometry?

For concave \(F(I)\), characterize when one-point exchange stability implies pointwise
first-order assignment under a common \(G\). Anchors: true for full D; false for A, for
\(D_s\), and naively for E; the screening direction follows from concavity for all four. Desired:
a curvature or operator inequality, a useful subclass, or an impossibility theorem showing
log-det is exceptional.

## OP2. Quantitative finite-geometry bound for A

Derive or disprove an A analogue of the \(D_s\) \(O(w_i(1/W_a+1/W_b))\) necessity bound. Known:
exact \(O(d^2)\) A move oracle, concavity screening, and `CE-A-DSTYLE-001` refuting the exact
D-style geometry theorem.

## OP3. Quantitative E necessity bound under a spectral gap

For a simple \(\lambda_{\min}\) separated by \(\gamma>0\), does exchange stability imply an
approximate rank-one Voronoi rule with an explicit \(O(w/\gamma)\) bound? Population companion:
`OPEN-E-COMMON-SUPERGRADIENT` (`KNOWN_RESULTS/06-e-optimality.md` § E6).

## OP11. Parameterized complexity

NP-hardness for fixed \(d=2\) and variable \(K\); for \(K=d+1\) and variable \(d\); FPT in
\(K+d\); W[1]/ETH bounds; tightness of the \(N^{O(Kd)}\) exact route. No imported k-means or
subset-selection hardness without a valid reduction.

## OP12. Stronger local neighborhoods

Two-point swaps, move-two, merge-split, boundary perturbations, rank-\(r\) determinant updates:
approximation guarantees, stronger geometry from 2-swap stability, reduced multistart dependence.

## OP13. Stronger branch-and-bound upper bounds

Improve singleton-refinement bounds via moment relaxations, SDP/convex envelopes,
affine-realizability pruning or minimum-cell-mass constraints, to certify larger instances (the
mathematical alternative to a compiled port of `certify.py`).

## OP25. Atomic randomization gap

For atomic score laws, can splitting an atom among labels strictly improve D or \(D_s\)? Smallest
exact counterexample, or conditions for no gap.

## OP26. Soft-to-hard zero-temperature limit

When do stationary points or optima of a temperature-softened affine/Voronoi family converge to
hard stationary points or optima as \(\tau\to0\)? Separate objective convergence along a
parameter path, convergence of global optima, and convergence of local branches.

## OP30. Inhabitation and selection of margin-retaining stable states

After DS17: (M5)-free tracking of coincident-projected-centroid wasted-cell configurations by
empirical exchange-stable sequences; attainment and one-sided continuity of \(v^*(\kappa)\) and
\(v^{*+}(\kappa)\) under their DS16 conventions (the Gaussian sign-split family proves
nonemptiness for \(\kappa\le1/\pi\), not attainment). No wasted-cell state becomes deployable by
retaining a nuisance floor. Target claim: `OPEN-DS-STABLE-BASINS`.

## OP31. Exact bit complexity of the tilt-DP dual

For positive rational weights and a rational score table, with \(K\) and \(d_\lambda\) in the
input, does \(\min_\beta\hat v_K(S_\psi-\beta S_\lambda)\) admit exact optimization in polynomial
bit complexity with the fixed-tilt interval-DP oracle, without materializing the parametric-DP
envelope? DS19 proves exact fixed-tilt evaluation, polynomial certified-\(\varepsilon\)
minimization, exact polynomial-bit minimization at \(d_\lambda=1\), and polynomially many
arithmetic operations for fixed \(d_\lambda\ge2\). Remaining: a polynomial bit bound for fixed
\(d_\lambda\ge2\) and any statement for variable \(d_\lambda\). Target claim:
`OPEN-DS-TILT-DUAL-EXACT-COMPLEXITY`; parked packet `WORK/active/DS-TILT-DUAL-EXACT-COMPLEXITY.md`.

# P8 · LITERATURE-GRAPH — coverage you can defend

Bidirectional citation snowballing from `LITERATURE/seeds.md` to saturation, per
`protocols/literature.md`. The claim-by-claim adversarial novelty search runs once, against
frozen statements, inside closure step 5.

## Formal-verification residue — parked, not a closure step

The `FORMAL-D-CLOSURE` packet (git history) was retired on 8 September 2026 as **partial**,
not as a completed finite-D formal programme. D6, duplicate inheritance and D8 have
formal markers. The duplicate node was trimmed as audit 002 required, but inherited
labels do not establish D5's second duplicate branch. Standing evidence:
`AUDITS/FORMALIZATION-D-FINITE-INDUCTIVE-CLOSURE-002.md` and `formal/README.md`.

Completed separately: the informal `D-COMPILE-TOLERANCE-GUARANTEE` contract audit
(`AUDITS/AUDIT-D-COMPILE-TOLERANCE-001.md`) narrowed the guarantee and added two boundary
fixtures in step 5a. Its positive-tolerance Lean proof remains parked.

Unresolved formal obligations: positive-tolerance `D-COMPILE-TOLERANCE-GUARANTEE`;
zero-weight samples; D5 duplicate-label constancy; D7's equal-optimum
half; D12 singleton-refinement bound; frozen `ExchangeVoronoiSpec` importing editable
`Leverage`; unapplied leverage statement hardenings and the superseded log-det audit's
bookkeeping. Generic helper proofs do not supply additional registry formal markers.
These obligations remain parked pending explicit reopening; this housekeeping changes
no frozen statement, Lean proof, or formal marker. ADR 0037 is the scope authority.
