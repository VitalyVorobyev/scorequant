# Independent audit of the score-error budget

**Target revision:** `22af4866d9b4eaf7e2772fcdfb9bbed7c8b19af0`, branch `score-error-budget`, PR #65.
**Date:** 8 September 2026. **Independence:** this session did not produce the derivation
and received no prover-session context; it inspected the committed statements and evidence.
**Verdict:** the reporting and transfer budgets survive after corrections. Two classifier
consequences and several invariance/order readings are false as originally written.
The corrected claims are audited; this is no certification of Python/JAX implementations.

## 1. Target statement

Targets: `SCORE-ERROR-RETENTION-BUDGET`, `SCORE-ERROR-RULE-TRANSFER`,
`CLASSIFIER-CALIBRATION-SCORE-LIPSCHITZ`, `SCORE-ERROR-DOOR3-MEASURED`, and
`CE-SCORE-ERROR-RHO-MIN-NECESSARY-001`, `CE-SCORE-ERROR-BOUNDARY-ATOM-001`,
`CE-AUC-INVARIANT-PROXY-RETENTION-001`.

For one law, true score s, proxy h=s+e and fixed labels Z, compare uncentred
cell information I=E[c_Z c_Zᵀ] with proxy cell information Ĩ and full moments
V=E[ssᵀ], Ṽ. Bound log(η̃/η), η=(det I/det V)^(1/d), using the whitened error
ε²=tr(V⁻¹E[eeᵀ]), its cell-average ε_Z², and ε_R²=tr(I⁻¹E[e_Ze_Zᵀ]).
Separately compare labels q(s) and q(h), then convert mixture posterior error to ε.

## 2. Criterion and problem level

Full D/geometric-mean retention, with directional and trace controls.
`information_accounting` under a frozen measurable label map; classifier conversion is
`score_oracle`, and door3 is `application`. Labels are the decision variable, not
subset selection or experimental-design weights. No profiled D_s result is claimed.
An empirical law gives algebraic sample-wise inequalities, not population certification.

## 3. Status before attempt

The reporting and transfer nodes were `project_proved` and unaudited, classifier conversion
was `bridge`, and door3 was `measured`. All four had no targeted literature pass.
The three negative fixtures already had deterministic tests. Closure step 3 owed its audit.

## 4. Dependencies rechecked

- `FI-QUANT-IDENTITY`: differentiating a fixed cell probability gives its conditional
  true-score mean, under the regular-model interchange assumptions. Mean-zero truth
  makes its second moment Fisher information. An estimated score is not substituted.
- `FI-LOSS-DECOMPOSITION`, `REPRESENTATION-QUANTIZATION-LOSS`: expand
  E[(s−E[s|Z])(s−E[s|Z])ᵀ]; orthogonality removes the cross terms. For a representation,
  apply the tower property before conditioning on labels.
- `INFO-RETENTION-SPECTRUM`, `INFO-D-EFFICIENCY`: 0≼I≼V implies 0≼R≼Id by
  congruence, and det R=det I/det V. No centering is needed for this algebra.
- `D-REPARAM-INVARIANCE`: I and V both transform by A(·)Aᵀ, so det(A)² cancels
  for nonsingular linear A. This proves no translation invariance.
- `CLASSIFIER-RATIO-ORACLE`, `CLASSIFIER-MIXTURE-SCORE-FORMULA`: Bayes' rule gives
  η_a/π_a=f_a/p_π, hence Φ_a=f_a/p_θ. A normalized-fraction score applies the
  parameter tangent chart to Φ. Merely reliable probabilities need not be P(Y|X);
  this prerequisite's wording is hardened, not silently assumed.
- `PROXY-TRUE-RETAINED-FI`: apply the first identity to the fixed observation map
  q(h(X)). Varying its input representation does not replace its categorical score.

No unresolved dependency blocks the corrected elementary results. Unread literature
is not used as a proof premise.

## 5. Nearest literature

The five-source table, inspected locations, queries and cite-versus-derive decisions
are in `LITERATURE/audits/SCORE-ERROR-RETENTION-BUDGET-2026-09-08.md`.
[Boyd–Vandenberghe](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf),
[Audibert–Tsybakov](https://imagine.enpc.fr/~audibert/Mes%20articles/plugin_v3.pdf),
[Bröcker](https://arxiv.org/pdf/0806.0813),
[Barnes–Han–Özgür](https://web.stanford.edu/~aozgur/FisherAllerton.pdf), and
[Cranmer–Pavez–Louppe](https://arxiv.org/pdf/1506.02169) provide the imported ingredients.
The specific reporting expansion has a targeted search gap. Neither that gap nor these
boundary repairs support novelty wording.

## 6. Counterexample search

`py/audit_score_error_budget.py exact`, seed 650022, enumerated all K=d+1 partitions
of 12 constructed rational laws in d=1,2,3, N=6 or 7: **1,884 partitions**, including
1,856 with positive retained determinant. There were 9,364 exact projection checks,
3,768 exact sandwich checks, 180 rational simplex pairs (including vertices and small
priors), and 2,629 60-digit Decimal log-remainder checks. The latter are numerical
screening, not exact proof. No corrected-core failure was found.

Coverage includes centered scores, unequal/tiny weights, duplicates, singletons,
near-singular coordinate scaling and empty labels after transfer. This is not an
exhaustive enumeration of all rational laws or all N≤10. Independent targeted witnesses
below found defects missed by the original random selftest. Elementary matrix primitives
were reused after inspection; conditional-moment assembly and projection checks were
written independently. The audit does not count replaying the original implementation
as independent numerical verification of its own formulas.

## 7. Algebraic reduction

Write C=E[c_Ze_Zᵀ], E_Z=E[e_Ze_Zᵀ]. Then Ĩ−I=C+Cᵀ+E_Z exactly.
Conditional Jensen gives E_Z≼E[eeᵀ], and directional Cauchy–Schwarz gives B1.
For any A=E[ccᵀ]≻0, whiten c,e by A and put x²=E||e'||². Then
tr(A⁻¹(Ã−A))=2E[e'ᵀc']+x², |E[e'ᵀc']|≤√d x, and
||A⁻¹/²(Ã−A)A⁻¹/²||_F≤U_d(x)=2√d x+x².
The rank-one inequality at t=x gives Ã≽(1−x)²A for 0<x<1;
x=0 is equality. These are the only matrix ingredients needed for the curvature bound.

## 8. Proof / counterexample / conditional result

### O8.2: curvature survives a false intermediate scalar assertion

The original lemma asserted log(1+λ)≥λ−λ²/[2(1+λ_min)] for every λ≥λ_min>−1.
This is false for λ_min>0. With c=±2 and proxy ±3, λ=λ_min=5/4;
the claimed lower bound is 65/72, whereas log(9/4)<65/72. The regression proves
this strictly using a rational upper Riemann sum, not floating logarithms.

The repair retains the original final bound. Exactly,

\[
\log(1+\lambda)-\lambda=-\lambda^2\int_0^1\frac{t}{1+t\lambda}\,dt.
\]

Put m=min(0,λ_min). On the entire interpolation, 1+tλ≥1+m≥(1−x)².
Thus the remainder lies in [−λ²/(2(1−x)²),0]. Sum over eigenvalues and use the
Frobenius bound. Subtract denominator from numerator to obtain precisely the published
r₃ bracket. A generic Hessian supremum would instead give the looser fourth-power
denominator; the integral identity is why the stated square is valid.

A finite logarithmic expansion needs Ĩ≻0 as well as I,V,Ṽ≻0; ε_R,ε<1 imply
both proxy conditions. r₃ is second order, despite its name, and T₂ alone is not a
complete second-order Taylor term. The e_b=κc_b attainer proves numerator sharpness,
not sharpness of the log-retention ratio (global score rescaling cancels there).

### Alignment: the trace-loss factor is valid; the ordering is false

Let w=V⁻¹/²s, u=V⁻¹/²e, c=E[w|Z], u_Z=E[u|Z], R=E[ccᵀ]. Then

\[
\frac d2T_1=E[u_Z^\top(R^{-1}-I)c]
-E[(u-u_Z)^\top(w-c)].
\]

For H=R⁻¹−I≽0, E[cᵀHc]=d−tr R. Cauchy–Schwarz bounds the first term by
√(E[u_ZᵀHu_Z])√(d−tr R), and the second by
√(ε²−ε_Z²)√(d−tr R). Bounding H by its largest eigenvalue gives the printed
trace-loss expression. Independently applying Cauchy–Schwarz before subtracting
bounds |T₁| by 2(ε_R+ε)/√d. Use the minimum of these two bounds.

They cannot be chained. Four atoms (−1,−9),(1,−9),(−1,1),(1,1), weights
(1,1,9,9)/20 and labels (0,1,2,2) have R=diag(1/10,1).
For e=(0,s₂/10), ε²=ε_Z²=ε_R²=1/100 and T₁=0. The squared trace-loss bound
is 81/1000, exceeding the squared crude bound 2/25. Four atoms are minimal for
centered d=2, positive retained rank and positive within-cell loss at K=3.
At R=Id the trace-loss factor vanishes exactly; projection orthogonality explains it.

### O8.3: exponent and constants pass, with explicit domains

For disagreement, the segment from s to h meets the original cell boundary. Hence
π≤M_q(t)+ε²/t². With M_q(t)≤Ct^α, differentiate Ct^α+ε²/t² to get

\[
t_*=(2\varepsilon^2/(\alpha C))^{1/(\alpha+2)},\quad
\pi\le(1+\alpha/2)(2/\alpha)^{\alpha/(\alpha+2)}
C^{2/(\alpha+2)}\varepsilon^{2\alpha/(\alpha+2)}.
\]

At α=1 the coefficient is 3/2^(2/3)<1.89. The error is an L² **norm**;
if using its square the exponent is α/(α+2). Require C,α>0 and t_* in the
margin's valid range; otherwise minimize on that range and cap by one.
Distinct centres and a positive-definite G justify the displayed Voronoi distance
in the G norm. A Fisher-norm distance uses G=V⁻¹ there, or the corresponding dual
norm for a different G. Coincident centres are not handled by division by zero.

For the sandwich, use g=c_{Zhat} as an alternative Zhat-measurable predictor.
Projection minimizes matrix squared error, g=c_Z off the disagreement event, and
(s−g)(s−g)ᵀ≼2(ssᵀ+ggᵀ). This gives the two claimed Loewner inequalities, with
zero-valued conditional centroids on empty cells. Trace follows immediately.
Γ≼γ_R I_Z gives the determinant lower bound when γ_R<1.

Bounded whitened scores give tr(V⁻¹Γ)≤4B²π. With fourth moment M₄ and a fixed
finite base partition of minimum positive mass p*, the forward bound is at most
2√M₄√π+2π max||c_b'||². On original occupied labels and π<p*/2, the reverse
centroids satisfy ||ĉ_b'||²≤2d/p*, giving 2√M₄√π+4dπ/p*.
Originally empty labels contribute no reverse cross-centroid term. Uniform rates
need uniform M₄,p*, and determinant rates a retained eigenvalue floor.
The former atom at π^(−1/2) has divergent fourth moment: it does not prove sharpness.

### O8.4: forward constants pass; reverse and deployment readings fail

The quotient identity ΔΦ=[diag(1/π)Δη−ΦΔD]/D' proves the stated coordinate
bound and L=(1/π_min+Q√m||1/θ₀||₂)/D_min. The law-change factor Q is correct.
Brier excess and its reliability/resolution split follow by conditional orthogonality.

For h=πᵀΦ≥1/Q, Δeta=[diag(π)ΔΦ−eta Δh]/h' gives
L_inv=Q(π_max+||π||₂). This relates posterior error to **full Φ**, not necessarily
TΦ. Only if T is injective on θ₀-perp, with restricted minimum singular value σ_T,

\[
\varepsilon^2\ge\frac{D_{\min}\sigma_T^2}{L_{\rm inv}^2\|V\|_{op}}
E_{P_\pi}\|\hat\eta-\eta\|^2\ge
\frac{D_{\min}\sigma_T^2}{L_{\rm inv}^2\|V\|_{op}}\operatorname{reliability}.
\]

Two equiprobable observations with posterior (1/2,1/6,1/3) and its first-two-coordinate
swap, uniform three-component priors/fractions and T=(1,−1,0), have scores ±1.
Add (1/12,1/12,−1/6) to both posteriors: reliability is 1/24 but the scores do not
change. This is a valid one-parameter mixture submodel with V=1, not a zero-score trick.

Even a full binary chart does not make bad calibration certify retention distortion:
posteriors (3/4,1/4),(1/4,3/4), replaced by (5/8,3/8),(3/8,5/8), give reliability
1/32 and score error ε²=1/4, but both singleton-label retentions equal one.
In general any invertible score rescaling preserves the ratio for fixed labels.
Reliability estimation itself requires sampling/calibration-estimation assumptions.

### Linear reduction, comparator transport, and the three existing fixtures

Translations are false invariances: on s=(−1,0,1) with equal weights and labels
(0,0,1), retention changes from 3/4 to 9/10 after adding one. The smallest centered
nontrivial positive scalar example requires three atoms. Invertible **linear** maps
preserve retention; preserve labels by transporting q, not holding its coordinates fixed.

A*=E[shᵀ]Ṽ⁻¹ may be singular: s=±1, h=1 has V=Ṽ=1 but A*=0. Its least-squares
error identity remains valid; its transformed determinant ratio does not. Nonsingular
cross moment is necessary to apply the invariant budget at the minimizer. The minimum
error can otherwise only be approached by invertible maps; no finite limiting bracket
is implied. This scalar witness is minimal for nondegenerate mean-zero truth.

For O8.6 define −β⁻(q)≤log(η̃(q)/η(q∘h))≤β⁺(q). Then the deployed lower bound
subtracts β⁺ at the selected rule and β⁻ at the comparator. The former text reversed
these meanings. Its free replacement of q(s) by q(As) also changes the oracle target;
it is removed. Proxy-objective dominance remains an explicit premise.

All three existing negative constructions survive recomputation. The rho_min fixture
had a factor-four typo in its prose: ε²=κ²/(9+δ²), not 4κ²/(10V₂₂).
As δ→0 at fixed κ>0, I_Z,22=δ²/8, det Ĩ→9κ²/100>0 and det Ṽ→9/10+9κ²/100;
the reporting ratio diverges while ε≤κ/3. This refutes relative/log control from an
error upper bound, not absolute control (both retentions remain in [0,1]).
The boundary atom retains its mass-1/2 jump and exact 4/5→1/2 retention change.
The AUC example preserves ROC/rank cuts under **corresponding thresholds**, not every
fixed numerical threshold; it uses uncentred moments, not a proxy variance ratio.

## 9. Adversarial audit

| Attack | Outcome |
|---|---|
| Strictness and ties | Reporting does not require strict cells. Transfer has a fixed tie convention; boundary atoms explicitly obstruct continuity. |
| Singleton/empty cells | Singletons valid. Empty conditional centroids must be assigned a version (zero), including cross-label use. |
| Duplicate scores | Allowed for moment identities; deterministic score rules cannot split identical inputs without additional information. The rational search includes duplicates. |
| Singular information | Finite logs require both retained matrices positive; small-error domain suffices. No pseudodeterminant or ridge extension. |
| Nuisance singularity | D_s and profiled objectives excluded; no generalization inferred. |
| Atomic laws | All six new witnesses and the three existing fixtures are atomic; no atomless hypothesis is hidden. |
| Hidden compactness | None in the core identities; power rates need explicit moment, cell-mass and spectral uniformity. |
| First-order-to-finite jumps | Exact log brackets verified. Neither T₂ alone nor numerator sharpness controls the full finite ratio. |
| Empirical-to-population jumps | Algebra holds per law; finite-sample calculations provide no population generalization theorem. |
| Score-estimation error | Central target; reverse calibration and raw-error-to-retention necessity fail. |
| New-event extension | Only the frozen map and its explicit transport extend; no fitting or optimizer optimality is proved. |

## 10. Algorithmic consequence

No new solver or public estimator is introduced. Correct reporting bounds can use the
minimum of two alignment bounds and an explicitly nonsingular linear reduction.
The research selftest now checks admissibility before applying least-squares invariance.
Uniform perturbation control alone cannot upgrade an exchange-stable solution to a
global optimum; an optimization-gap certificate is a separate missing ingredient.

## 11. Deployability consequence

A truth/simulation score-error upper bound can certify a reporting interval under the
rank and conditioning conditions. AUC and calibration alone cannot. The swapped
reporting bracket requires error in the proxy metric, not silently the O8.4 true metric.
The engineering error-bar packet may proceed for independent held-out oracle-score
observations; this audit adds no public bias estimator or observable truth certificate.

## 12. Information-loss consequence

The corrected theorem controls relative/log reporting error and, with a margin budget,
true information transfer of the frozen rule. It does not guarantee retained information
from classifier calibration, the worst retention eigenvalue from AUC, or generalization
from fitted-sample diagnostics. O7 sampling uncertainty and O8 proxy reporting bias are
separate quantities; neither absorbs the other.

## 13. Updated status

| Target | Verdict |
|---|---|
| SCORE-ERROR-RETENTION-BUDGET | `project_proved`, audited after correcting scalar proof domain, bound ordering, positivity and invariance/composition readings. |
| SCORE-ERROR-RULE-TRANSFER | `project_proved`, audited with explicit metric, margin-range, empty-cell and rate assumptions; exponent/constants verified. |
| CLASSIFIER-CALIBRATION-SCORE-LIPSCHITZ | `bridge`, audited after restricting reverse score control; unconditional calibration/retention consequences refuted. |
| SCORE-ERROR-DOOR3-MEASURED | Remains `measured`; all three door3 rungs and 12 synthetic rows reproduced. |
| Three original CE fixtures | Verified, with rho_min prose formula and AUC threshold semantics corrected. |

Step 3's verdict-plus-audit gate is closed for these corrected statements. The broad
OP17/OP18 questions remain partly open; the research programme is not frozen or finished.

## 14. Registry patch

All four target nodes receive this audit pointer and explicit assumptions. Six new
boundary fixtures are attached to the affected nodes. Classifier ratio prerequisites
are hardened to full-observation posteriors. Brier/margin sources are registered;
the reporting theorem records a search gap. O8 prose, open-question summaries and
manuscript staleness agree with the corrected claims. No problem definition, public API,
`src/` behavior, or Lean evidence is changed.

## 15. Counterexample and regression artifacts

- `CE-SCORE-ERROR-LOG-LOWER-001`: positive-floor scalar log assertion.
- `CE-SCORE-ERROR-ALIGNMENT-ORDER-001`: false ordering of two valid bounds.
- `CE-SCORE-ERROR-TRANSLATION-001`: false affine-translation invariance.
- `CE-SCORE-ERROR-SINGULAR-LS-001`: inadmissible singular least-squares minimizer.
- `CE-CLASSIFIER-CALIBRATION-CHART-001`: positive reliability with zero score error.
- `CE-CLASSIFIER-CALIBRATION-RETENTION-001`: positive reliability/error with unchanged retention.

Every witness is serialized under `COUNTEREXAMPLES/` and independently recomputed in
`tests/test_research_claims.py`. Original fixtures remain pinned there. The original
selftest replay has 400 laws, 22,942 checks and no failures; its `B4_pairs=940` field
actually counts coordinate checks over 300 pairs, now stated accurately in prose.
`WORK/artifacts/AUDIT-SCORE-ERROR-BUDGET-001/` holds independent exact-search and
replication records. All 423 recorded door3 numeric fields and 305 synthetic numeric
fields agree exactly with the original artifacts on this machine (maximum absolute
difference zero). Numerical reproduction is evidence, not theorem authority.

The final handoff verification is recorded in this audit's verification appendix below.

## 16. Next dependency-blocking question

`OPEN-RETENTION-UNCERTAINTY`: can the audited frozen-rule O7 standard error be exposed
without silently certifying singular, weighted, refitted, or proxy-score settings?
This is closure step 1, packet `WORK/active/RETENTION-ERROR-BAR.md`. Its mathematical
prerequisite is already audited; the next work is engineering and contract enforcement.
The truth-free resolution-gap problem remains backlog, not the next active packet.

## Verification appendix — 8 September 2026

All commands ran in `.claude/worktrees/score-error-budget-audit-001`.

| Command | Result |
|---|---|
| `uv sync --all-extras --all-groups --locked` | Passed; locked environment installed, including example dependencies. |
| `uv run python agenticresearch/py/audit_score_error_budget.py exact` | Passed; counts in §6. |
| `uv run python agenticresearch/py/audit_score_error_budget.py fixtures` | Six fixtures serialized and pinned by independent tests. |
| `JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run python agenticresearch/py/audit_score_error_budget.py replicate` | Passed; 22,942 original checks, zero failures; 728 measured numeric fields reproduced exactly. |
| `uv run python agenticresearch/py/registry.py reindex` | Generated indexes current. |
| `uv run python agenticresearch/py/registry.py validate` | Registry clean. |
| `uv run python website/scripts/generate_atlas.py` | Atlas regenerated successfully. |
| `JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest tests/test_research_claims.py tests/test_research_registry.py tests/test_atlas_data.py` | 75 passed. |
| `uv run ruff check .` | Passed. |
| `uv run ruff format --check .` | Passed; 278 files already formatted. |
| `uv run ty check src` | Passed. |
| `JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto` | Final run: 590 passed in 161.27 s, including documentation execution. |
| `JAX_ENABLE_X64=0 MPLBACKEND=Agg uv run pytest tests/test_float32.py` | 4 passed. |
| `uv run mkdocs build --strict` | Passed. |
| `uv build` | Source distribution and wheel built successfully. |
| `git diff --check` | Passed. |

The first full test run had 589 passes and one Atlas error because a new literature
topic lacked a portal tradition. The annotations were moved into the established
score-compression topic, generated data rebuilt, targeted tests passed, and the entire
suite then passed as recorded above. The initial measurement replay lacked scikit-learn
before the locked all-extras sync; the completed replay includes it. These are resolved
validation failures, not waived checks. No Lean build was run because no formal statement,
proof, marker, or other formal evidence changed; parked obligations remain explicit.
