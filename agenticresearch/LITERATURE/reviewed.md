# Reviewed papers

Deep-reviewed papers with outcomes; every entry names the claim ids it informs. Populated by the LITERATURE-GRAPH work packets.

## Valassi (2020) — screened 29 August 2026

Bibliography key `Valassi-2020`; the anchoring annotation is in
`topics/05-hep-inference-aware.md`.

**Outcome: open attribution question, deliberately not settled.** The paper states, in the
single-parameter case, both the retained-information identity behind `FI-QUANT-IDENTITY`
(\(I_\theta=\sum_k s_k\phi_k^2\), cell-mean sensitivity \(\phi_k\)) and an efficiency ratio
FIP\(_3=I_\theta/I_\theta^{(\rm ideal)}\) that coincides with `INFO-D-EFFICIENCY` at \(s=1\). Both
claims now cite it. Neither claim's `status` or `publication_status` was changed and no
`literature_search_status` was set, because that determination belongs to a literature session
running the `protocols/literature.md` checklist, not to a wiring commit.

**The two questions that session must answer:**

1. Does FIP\(_3\) constitute prior art for `INFO-D-EFFICIENCY`, or only for its \(s=1\) restriction?
   The registry statement is the determinant ratio \((\det I_q/\det I_{\rm full})^{1/d}\); the two
   agree only at \(d=1\), where the determinant is not doing any work.
2. Is the argument "bin by \(\gamma_i\)" a *result* or a framing? Valassi asserts the optimal
   partitioning variable is the sensitivity and does not characterise the optimal cells. If it is a
   result, it is the scalar ancestor of the score-space reduction in `PROBLEM.md` and belongs in
   `KNOWN_RESULTS/01-universal.md` as a `[LIT]` attribution.

This is the same failure mode the DS11 audit caught (`AUDITS/AUDIT-DS-POPULATION-BRIDGE-001.md`):
a result believed novel that a neighbouring community had stated first in a special case. Resolve
it before any manuscript revision.

**Round-3 resolution (30 August 2026):** FIP\(_3\) is direct prior art for the
scalar \(d=1\) retained-information ratio and Valassi's sensitivity is the
scalar score-space reduction. It is not prior art for the multivariate
determinant normalization in `INFO-D-EFFICIENCY`, and the paper gives no
optimal-cell theorem. Keep the current claim statuses; describe the scalar
restriction explicitly in publication prose.

## CMS Collaboration (2025) — screened 29 August 2026

Bibliography key `CMS-2025`; the anchoring annotation is in
`topics/05-hep-inference-aware.md`.

**Outcome: motivation, not prior art. No claim cites it, deliberately.** SANNT minimises
\(\Delta r_s=\sqrt{(F^{-1})_{r_sr_s}}\) over one parameter of interest and up to 224 nuisance
parameters — the profiled objective of `DS-SCHUR` at \(s=1\), reached independently and run at
production scale. It never studies the geometry of the resulting partition, so it precedes no
registry theorem.

**What it does supply** is an unusually direct statement of the gap the \(D_s\) programme fills: the
paper records that a Fisher-based loss "introduces an ambiguous choice of binning" and that the
binned likelihood "is not differentiable at its bin edges", and reaches for INFERNO's softmax
histogram or a KDE surrogate. That is the softening `KNOWN_RESULTS/08-soft.md` exists to make
unnecessary. Worth citing in the motivation of any \(D_s\) manuscript.


## Pollard (1981) — read verbatim 30 August 2026 (DS15 audit)

Bibliography key `Pollard-1981`; annotation in `topics/04-vector-quantization.md`.

**Outcome: statement and hypotheses fully verified from the Yale scan.** The
consistency theorem assumes uniqueness of the optimal center set for *every*
\(j\le k\) (not just \(j=k\)) plus a \(\phi\)-moment; value convergence needs
no uniqueness. Informs `OPEN-DS-MARGINS-AT-OPTIMA` (Proposition 5 and
conclusions (2)–(3)) and `AUDIT-DS-MARGINS-AT-OPTIMA`. The every-\(j\)
hypothesis is discharged for log-concave efficient-score laws by the
uniqueness cluster (Kieffer/Graf–Luschgy 5.1/Mease–Nair).

## Fisher (1958) — read verbatim 30 August 2026 (DS15 audit)

Bibliography key `Fisher-1958`; annotation in `topics/04-vector-quantization.md`.

**Outcome: contiguity of optimal 1-D weighted SSE partitions confirmed with
its appendix proof** (pp. 789–792 read). Anchors `DS-SCALAR-EFFICIENT-DP` and
the interval-optimum leg of DS15's Proposition 4.

## Mease & Nair (2006) — read verbatim 30 August 2026 (DS15 audit)

Bibliography key `Mease-Nair-2006`; annotation in `topics/04-vector-quantization.md`.

**Outcome: (S)-boundary found.** Log-concavity of the density suffices for
scalar-quantizer uniqueness (likelihood-ratio ordering); Eubank (1988)'s
weaker condition is refuted by an explicit three-stationary-point
counterexample. DS15's assumption (S) must not be weakened toward
Eubank-type conditions. Informs `OPEN-DS-MARGINS-AT-OPTIMA`.

## Levrard (2015) — read verbatim 30 August 2026 (DS15 audit)

Bibliography key `Levrard-2015`; annotation in `topics/04-vector-quantization.md`.

**Outcome: the margin condition (Definition 2.1) is a hypothesis on the law
at population optima and is never proven to hold at empirical optima** —
confirming the structural contrast with DS15, which proves a margin *fails*
at optima. Registry arXiv id corrected (1405.6672; 1310.7138 is the 2013
precursor "Margin conditions for vector quantization").
## Snowballing round 3 — 30 August 2026 (bidirectional)

This was the first bidirectional citation round. All resolvable anchors in
`seeds.md` were traversed one hop backward and forward. OpenAlex supplied the
primary graph and metadata; Semantic Scholar was used only to repair missing
or rate-limited edges. Counts and source limitations are recorded in
`graph.json`.

### Fisher-information quantization and score access

- **Venkitasubramaniam, Tong & Swami (2007),** *Quantization for Maximin ARE
  in Distributed Estimation* (`Venkitasubramaniam-Tong-Swami-2007`), extends
  scalar score-function quantization to a worst-parameter ARE criterion and an
  iterative threshold design. It informs `TRACE-WHITENED-KMEANS`,
  `OPEN-D-HIGH-RATE`, and `OPEN-PARAMETER-MISMATCH`, but supplies no
  multivariate D or \(D_s\) result.
- **Barnes, Han & Özgür (2020),** *Lower Bounds for Learning Distributions
  under Communication Constraints via Fisher Information*
  (`Barnes-Han-Ozgur-2020`), carries quantized-Fisher geometry into minimax
  lower bounds, not a deterministic partition solver. It informs
  `FI-QUANT-IDENTITY` and `TRACE-WHITENED-KMEANS`.
- **Lam & Reibman (1993),** *Design of Quantizers for Decentralized
  Estimation Systems*, and **Gubner (1993),** *Distributed Estimation and
  Quantization*, are earlier scalar/distributed estimation ancestors. They
  optimize estimator performance under restricted communication models; they
  do not establish the hard multivariate conditional-score determinant
  objective. Both inform `FI-QUANT-IDENTITY` and `TRACE-WHITENED-KMEANS`.
- **Cranmer, Pavez & Louppe (2015),**
  *Approximating Likelihood Ratios with Calibrated Discriminative
  Classifiers* (`Cranmer-Pavez-Louppe-2015`), directly supports
  `CLASSIFIER-RATIO-ORACLE`. It does not propagate calibration error into
  Fisher or D/\(D_s\) loss.
- **Zhang, Blum, Kaplan & Lu (2018),** *A Fundamental Limitation on Maximum
  Parameter Dimension for Accurate Estimation With Quantized Data*
  (`Zhang-Blum-Kaplan-Lu-2018`), is direct distributed-estimation precedent
  for the alphabet/parameter-dimension rank obstruction in
  `FI-RANK-CEILING`.

### Optimal design and determinant exchange

- **Silvey & Titterington (1973),** *A Geometric Approach to Optimal Design
  Theory* (`Silvey-Titterington-1973`), proves D and \(D_s\) equivalence in
  approximate design and a convergent monotone D construction. It directly
  strengthens the attribution of `DS-CLASSICAL-DESIGN-THEORY`.
- **Silvey, Titterington & Torsney (1978),** *An Algorithm for Optimal Designs
  on a Design Space* (`Silvey-Titterington-Torsney-1978`), gives monotone
  finite-design-space algorithms. It is algorithmic precedent for
  `D-EXCHANGE-TERMINATES`, but its free design weights do not transfer to
  coupled hard-cell masses and conditional means.

### High-rate and vector quantization

- **Rakhlin & Caponnetto (2006),** *Stability of K-Means Clustering*
  (`Rakhlin-Caponnetto-2006`), is direct prior art for the
  compact-codebook/almost-minimizer rigidity used in DS16.1. Its bounded-law,
  nearest-center theorem does not cover arbitrary groupings, signed nuisance
  moments, or the profiled-information price.
- **Telgarsky & Vattani (2010),** *Hartigan's Method: k-means Clustering
  without Voronoi* (`Telgarsky-Vattani-2010`), supplies the nearest finite
  one-point-terminal geometry. It also reinforces that exchange stability is
  not a Voronoi certificate; it has no population or nuisance conclusion.
- **Zador (1982), Gersho (1979), and Bucklew & Wise (1982)** provide the
  classical high-rate scaling, cell-shape, and rigorous multidimensional
  additive-distortion templates now cited by `OPEN-D-HIGH-RATE`.
- **Gupta & Hero (2003),** *High-Rate Vector Quantization for Detection*, is
  the nearest task-aware bridge: an inference loss for binary detection admits
  high-rate point-density calculus. It still does not derive the logdet
  retained-Fisher or profiled-Schur expansion required by
  `OPEN-D-HIGH-RATE`.
- **Gray, Linder & Li (2002)** and **Lookabaugh & Gray (1989)** sharpen the
  entropy-constrained and geometric high-rate background, respectively. They
  remain additive-distortion results and are linked only to
  `OPEN-D-HIGH-RATE` in the discovery graph.

### Efficient-score compression and HEP categorization

- **Alsing & Wandelt (2019),** *Nuisance Hardened Data Compression for Fast
  Likelihood-Free Inference* (`Alsing-Wandelt-2019`), is the closest published
  representation-level antecedent to the efficient score used by
  `DS-EFFICIENT-SCORE-DOMINATION`. It preserves marginalized Fisher
  information locally/asymptotically for a continuous summary; it does not
  prove all-quantizer Loewner domination or a hard \(D_s\) upper bound.
- **Charnock, Lavaux & Wandelt (2018)** (IMNN) and **Alsing, Wandelt & Feeney
  (2018)** (DELFI+MOPED) learn or use continuous Fisher-aware summaries. They
  inform `REPRESENTATION-QUANTIZATION-LOSS` and
  `OPEN-REPRESENTATION-LOSS-ESTIMATION`, not hard-cell geometry.
- **Brehmer et al. (2020)** (MadMiner) is a practical learned-ratio/score
  pipeline informing `PROXY-TRUE-RETAINED-FI` and
  `OPEN-HEP-NUISANCE-SCALING`.
- **Wunsch et al. (2021)** and **Simpson & Heinrich (2023)** (neos) optimize
  differentiable binned/profiled inference objectives with nuisances. They
  are direct applied precedents for `DS-SCHUR` and
  `OPEN-HEP-NUISANCE-SCALING`, but rely on KDE/soft or end-to-end relaxations
  rather than exact hard score-space partitions.

## Proposed status changes — not applied

None. The round adds prior-art links and sharpens boundaries, but it does not
justify changing any claim's `status` or `literature_search_status`.
`OPEN-D-HIGH-RATE` remains open; `DS-EFFICIENT-SCORE-DOMINATION` remains an
internal project result; and Valassi (2020) precedes only the scalar
restriction of `INFO-D-EFFICIENCY`.

## Independent DS17 audit round — 31 August 2026

Full query log, counts, and the six-field comparison are in
`audits/AUDIT-DS-STABLE-BASINS-31-August-2026.md`. The researcher-authored
DS17 round was treated only as comparison.

### Primary-text conclusions

- **Flury (1990), `Flury-1990`:** defines globally MSE-optimal principal
  points, proves the two-point leading-eigenvector result for elliptical laws
  and the general span-dimension bound, but leaves the (k>2) leading
  principal-subspace statement as a conjecture. The former abstract-only note
  over-attributed a general existence direction.
- **Tarpey & Flury (1996), `Tarpey-Flury-1996`:** credits Hastie–Stuetzle for
  the terminology, defines random-vector self-consistency, and proves under
  LCM plus projection-measurability that a (q)-dimensional self-consistent
  support span is generated by (q) covariance eigenvectors. The result does
  not say they are the leading (q) for an arbitrary self-consistent summary.
- **Jakubowski (2021), `Jakubowski-2021`:** Theorem 1.2 and its proof are the
  exact classical source for DS17's monotone-covariance equality mechanism:
  zero covariance of two monotone transforms occurs iff one transform is a.s.
  constant. The independent-copy product identity is explicit.
- **Esary, Proschan & Walkup (1967),
  `Esary-Proschan-Walkup-1967`:** supplies the historical association
  definition and covariance-sign language, not Jakubowski's equality theorem.
- **Serinko & Babu (1992), `Serinko-Babu-1992`:** remains a valid scalar
  split-function/nonregular-asymptotics comparator. Its singular Hessian is
  not a nuisance-rank collapse or fixed-point nonexistence theorem.

### Classical and forward-source conclusions

- **Bickel, Klaassen, Ritov & Wellner (1993),
  `Bickel-Klaassen-Ritov-Wellner-1993`:** efficient-score projection and
  least-favourable-direction orthogonality are classical antecedents of the
  residual normal equation in DS17. They do not contain its binned endogenous
  tilt or partition obstruction.
- **Tarpey, Li & Flury (1995), `Tarpey-Li-Flury-1995`:** the leading
  (q)-eigenspace conclusion is pinned to principal points. Tarpey–Flury's
  broader arbitrary-self-consistent LCM result guarantees an eigenvector
  span, not necessarily the leading span.
- **Hastie & Stuetzle (1989), `Hastie-Stuetzle-1989`:** terminology
  provenance for self-consistent curves; vocabulary only for DS17.
- **Tarpey & Loperfido (2015), `Tarpey-Loperfido-2015`:** missed forward
  source generalizing principal-subspace results through self-consistency to
  non-elliptical mixture/skew-normal settings. It is mandatory structural
  comparison but still has no profiled-information or nuisance-rank theorem.

### Search verdict

The centered obstruction and endogenous root gate remain `search_gap` after
independent search. The LCM compound claim has genuine published prior art for
its structural conclusion (1), while its reduced-rule rank collapse, value
identity, sign-split family, and noncompilability conclusions remain unmatched.
The claim should be split or its literature status should explicitly record
that partial prior art.

## Round 7 — independent DS18 audit triangulation (31 August 2026)

Target: `DS-NONCENTERED-GLOBAL-BASIN-TRANSFER`; report
`AUDITS/AUDIT-DS-NONCENTERED-GLOBAL-BASIN-TRANSFER-001.md`; full six-field
table in
`LITERATURE/audits/AUDIT-DS-NONCENTERED-GLOBAL-BASIN-TRANSFER-31-August-2026.md`.

- **Kieffer (1983), `Kieffer-1983`:** the correct uniqueness citation for the
  named law. Log-concavity of the *density* suffices, and \(\operatorname{Unif}[-1,1]\)
  is log-concave; the strict-log-concavity routes (Fleischer 1964, Liu–Pagès
  Prop 11) do **not** cover it. Now linked to DS18 and to its audit.
- **Mease & Nair (2006), `Mease-Nair-2006`:** second independent uniqueness
  route under log-concavity of the density, plus the standing warning that
  Eubank-type weakenings are known-broken. Linked to DS18.
- **Pollard (1981), `Pollard-1981`:** supplies the empirical scalar
  consistency and, crucially for DS18's "**every** sequence of global
  optimizers" quantifier, a *selection-independent* almost-sure event. It does
  not connect the profiled maximiser to the distortion minimiser; DS18's own
  sandwich does that, and no source was found for it.
- **Rakhlin & Caponnetto (2006), `Rakhlin-Caponnetto-2006`:** the published
  template for DS18.1's strict isolation and DS18.2's empirical rigidity.
  Their almost-minimizer geometry lives in codebook space; DS18 needs it in
  the decision distance on partitions, starting from arbitrary
  \((X,Z)\)-measurable cells.
- **de Castro & Dorigo (2019), `deCastro-Dorigo-2019`:** newly key-registered.
  Closest applied statement of the DS18 objective (Asimov inverse Hessian, a
  profiled-Fisher surrogate) but over soft differentiable histograms, with no
  optimality theorem.

### Search verdict

The compound DS18 statement remains `search_gap` after an independent query
path. Two attribution defects in the frozen node were found and repaired: six
direct antecedents already in this project's bibliography were unlinked, and
the uniqueness citation was mis-scoped to a strict-log-concavity result that
the named law fails. The audit supplies the missing conditioning fact itself
(exact distortion Hessian with minimum eigenvalue \(1/6\)).

## Round 8 — independent DS19 audit triangulation (2 September 2026)

Target: `OPEN-DS-PRACTICAL-CERTIFIED-SOLVER` and the DS19 components; report
`AUDITS/AUDIT-DS-PRACTICAL-CERTIFIED-SOLVER-001.md`; full six-field table in
`LITERATURE/audits/AUDIT-DS-PRACTICAL-CERTIFIED-SOLVER-2-September-2026.md`.

- **Toledo (1993), `Toledo-1993`:** primary text read. Fixed-dimension
  concave maximisation with an evaluator whose comparisons are signs of
  bounded-degree polynomials, in time polynomial in the evaluator's arithmetic
  operations. The fixed-tilt interval DP qualifies with degree 2, so exact
  minimisation of the tilt dual is arithmetic-polynomial for fixed
  \(d_\lambda\) and *variable* \(K\) — wider than DS19.2's registered
  fixed-\((K,d_\lambda)\) scope. Bit complexity is not covered.
- **Megiddo (1983), `Megiddo-1983`:** the one-dimensional parametric-search
  template; at \(d_\lambda=1\) the audit's own root-separation bisection turns
  it into a polynomial-*bit* exact algorithm (report §7.5).
- **Grønlund et al. (2017), `Gronlund-etal-2017`** and **Wang & Song (2011),
  `Wang-Song-2011`:** the fixed-tilt DP is the classical 1-D \(k\)-means DP;
  \(O(KN)\) after sorting; weighted points covered; ties not treated.
- **Pukelsheim & Titterington (1983), `Pukelsheim-Titterington-1983`:**
  design-side subgradient/Lagrangian duality on the convex design-measure set;
  template only, the hard-partition interchange fails.
- **Carstensen (1983), `Carstensen-1983`:** secondary record; the original
  superpolynomial parametric shortest-path breakpoint construction.

### Search verdict

The compound DS19 statement remains `search_gap` after an independent query
path. One scope repair: the exact-computation clause of
`DS-TILT-DUAL-CERTIFICATE` is widened to fixed \(d_\lambda\), variable \(K\)
(arithmetic complexity, prior art) and to \(d_\lambda=1\) in polynomial bit
complexity (audit proof); `OPEN-DS-TILT-DUAL-EXACT-COMPLEXITY` is narrowed
accordingly. One complexity repair: fixed-tilt evaluation is \(O(KN)\) after
sorting.

## Round 10 — AUDIT-SCORE-ORACLE-ROBUSTNESS (5 September 2026)

Independent audit triangulation for `RETENTION-PLUGIN-CLT-FROZEN-SCALAR`;
table in `audits/AUDIT-SCORE-ORACLE-ROBUSTNESS-5-September-2026.md`.

- **Bhattacharya & Ghosh (1978), `Bhattacharya-Ghosh-1978`:** abstract read;
  Edgeworth validity for smooth functions of sample moments — backdrop of
  the O6.7 second-order readings.
- **Hall (1992), `Hall-1992`:** record only; studentized-statistic skewness
  in the smooth function model explains the measured sign flip.
- **Donner & Koval (1980), `Donner-Koval-1980`:** journal record; the
  nearest applied delta-method ratio variance (normal random effects).
- Verification notes corrected on `vanderVaart-1998` (Thm 3.1 primary
  text; Example 3.2; Ch. 20), `Cramer-1946` (Ch. 28 §28.4 primary text),
  `Serfling-1980` (theorem labels unresolved), `Hampel-Ronchetti-Rousseeuw-Stahel-1986`
  (p. 85, §2.1, secondary).

### Search verdict

Method prior art; the exact fixed-partition uncentred statement is a search
gap; no re-attribution.


## Round 11 — RETENTION-PLUGIN-VECTOR literature-first pass (5 September 2026)

Novelty search run **before** the derivation of the vector retention CLT
(future claim `RETENTION-PLUGIN-CLT-FROZEN-VECTOR`); table in
`audits/RETENTION-PLUGIN-CLT-FROZEN-VECTOR-5-September-2026.md`. Reframing:
\(\eta_D=(\det I_Z/\det V)^{1/d}\) is the geometric mean of the squared
uncentred canonical correlations between \(S\) and the cell indicator; at
the evaluation law it is a MANOVA between/total determinant ratio.

- **Radhakrishnan & Kshirsagar (1981), `Radhakrishnan-Kshirsagar-1981`:**
  abstract read; influence functions of the generalized variance, the
  MANOVA noncentrality matrix and its eigenvalues, canonical correlations
  and Wilks' \(\Lambda\) — the matrix influence function of O6.8 in MANOVA
  form.
- **Romanazzi (1992), `Romanazzi-1992`:** abstract read; influence function
  of each squared canonical correlation.
- **Muirhead & Waternaux (1980), `Muirhead-Waternaux-1980`:** abstract
  read; CCA asymptotics under finite fourth moments.
- **Fang & Krishnaiah (1982), `Fang-Krishnaiah-1982`:** record only;
  functions of eigenvalues under nonnormality.
- **Seo, Kanda & Fujikoshi (1994; 1995, `Seo-Kanda-Fujikoshi-1995`):**
  nonnormality in CCA functions and dimensionality tests (the singular
  \(I_Z\) endpoint).
- **Sugiura & Fujikoshi (1969), `Sugiura-Fujikoshi-1969`:** abstract read;
  fixed-alternative expansion of the LR determinant ratio (normal theory);
  Fujikoshi (2002) is the nonnormal companion (record).
- **Magnus & Neudecker, `Magnus-Neudecker-1999`; Anderson,
  `Anderson-2003`:** section numbers from contents scans held in
  `papers/local/`.
- **Cramer & Nicewander (1979), `Cramer-Nicewander-1979`;** van den Burg &
  Lewis (1988); Hotelling (1936); Wilks (1932): lineage of the measure.
- **Kendall & Stuart Vol. 2 (1961), `Kendall-Stuart-1961`:** primary text
  read (§26.21–26.24); the correlation ratio's definition, decomposition
  and normal-theory \(F\) tests, no large-sample variance. **Closes the O6
  gap.**
- Adjacent, not transferring: Ogasawara (2007), Cléroux & Ducharme
  (1989), Cai, Liang & Zhou (2015, PDF in `papers/`).

### Search verdict

Method prior art (delta method + determinant differential + the
influence functions above + fourth-moment CCA asymptotics); the exact
geometric-mean statement on a fixed partition with uncentred moments,
\(0/0:=0\), the plug-in variance and the endpoint treatment is a search
gap. The packet `WORK/active/RETENTION-PLUGIN-VECTOR.md` was rewritten to
cite the first and derive only the second.
