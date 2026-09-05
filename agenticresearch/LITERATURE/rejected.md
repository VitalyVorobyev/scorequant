# Rejected papers

Screened and found irrelevant, with one-line reasons, so nobody re-screens them.

## Snowballing round 2 — 30 August 2026

- **Tsitsiklis (1993), “Extremal properties of likelihood-ratio quantizers”**
  — hypothesis-testing/Ali–Silvey objective, not estimation Fisher D or
  \(D_s\); assessed against `FI-QUANT-IDENTITY`.
- **Barnes, Chen & Özgür (2020), “Fisher Information Under Local Differential
  Privacy”** — randomized privacy channels and information upper bounds, not
  deterministic hard-cell optimization; `FI-LOSS-DECOMPOSITION`,
  `TRACE-WHITENED-KMEANS`.
- **Lazebnik & Raginsky (2009), “Supervised Learning of Quantizer Codebooks by
  Information Loss Minimization”** — class-label mutual information rather
  than parameter-score Fisher information; `REPRESENTATION-QUANTIZATION-LOSS`.
- **Amari (2011), “On Optimal Data Compression in Multiterminal Statistical
  Inference”** — restricted multiterminal linear-threshold model, not a
  general score-space partition theorem; `FI-QUANT-IDENTITY`.
- **Zou et al. (2022), “Goal-Oriented Quantization”** — high-rate downstream
  decision cost, not Fisher determinant or Schur loss; `OPEN-D-HIGH-RATE`.
- **Bryant & Williamson (1978), “Asymptotic behaviour of classification
  maximum likelihood estimates”** — class-assignment likelihood with biased
  parameter limits, not consistency of the retained-information partition
  functional; `OPEN-D-UNRESTRICTED-CONSISTENCY`.
- **Bryant (1991), “Large-sample results for optimization-based clustering
  methods”** — potentially adjacent consistency theory, but the objective and
  assumptions could not be verified from primary text; it cannot close
  `OPEN-D-UNRESTRICTED-CONSISTENCY` or `OPEN-D-EXCHANGE-CONSISTENCY`.
- **Fukunaga & Koontz (1970), “A Criterion and an Algorithm for Grouping
  Data”** — learned-transform grouping, principally for two classes, not the
  retained-Fisher common-metric theorem; `D-POP-VORONOI`.
- **Maronna & Jacovkis (1974), “Multivariate Clustering Procedures with
  Variable Metrics”** — cluster-variable metrics without a conditional-score
  objective or self-consistent \(I_q^{-1}\) metric; `D-POP-VORONOI`.
- **Windham (1987), “Parameter modification for clustering criteria”** —
  general partition/mixture/fuzzy reformulations, not ScoreQuant's rank-two
  gain identity; `D-RANK2-MOVE`, `D-LOGDET-GAIN`.
- **Banerjee et al. (2005), Bregman clustering** — additive,
  source-independent Bregman distortion; no solution-dependent determinant or
  Schur objective; `D-POP-VORONOI`, `D-GUARDED-LLOYD`.
- **Kanungo et al. (2002), local-search approximation for k-means** —
  Euclidean additive approximation/exchange mechanics do not transfer to
  retained-Fisher logdet; `D-EXCHANGE-IMPLIES-VORONOI`,
  `D-EXCHANGE-TERMINATES`.
- **Wunsch et al. (2020), “Reducing the Dependence of the Neural Network
  Function to Systematic Uncertainties”** — suppresses nuisance dependence
  instead of optimizing profiled information; `OPEN-HEP-NUISANCE-SCALING`.
- **OASIS (2021)** — optimizes simulation-sample allocation, not observation
  partitioning; `FI-QUANT-IDENTITY`.
- **“Parametrized classifiers for optimal EFT sensitivity” (2021)** —
  unbinned classifier method without hard-category optimization or a Fisher
  theorem; `CLASSIFIER-RATIO-ORACLE`,
  `REPRESENTATION-QUANTIZATION-LOSS`.
- **Diekmann, Eich & Erdmann (2024/2026), classifier-output clustering** —
  ordinary Euclidean k-means in unwhitened classifier-output space, not D or
  \(D_s\) score-Fisher optimization; `TRACE-WHITENED-KMEANS`,
  `REPRESENTATION-QUANTIZATION-LOSS`.

## Independent DS17 audit — 31 August 2026

- **Generic FKG correlation papers** — require lattice/positive-association
  structure far stronger than DS17's conditional one-variable monotonicity;
  Chebyshev plus Jakubowski's equality theorem is the exact source pair;
  `DS-STABLE-BASINS-CENTERED-OBSTRUCTION`.
- **Choquet-integral covariance inequalities** — generalize the integration
  functional rather than the hard-quantizer or equality mechanism; no
  transfer to DS17.
- **Bali & Boente (2009), functional elliptical principal points** — extends
  the Tarpey–Li–Flury programme to Hilbert spaces but adds no endogenous tilt,
  profiled block, or fixed-point obstruction;
  `DS-STABLE-BASINS-LCM-CLASSIFICATION`.
- **Matsuura & Kurata (2010), location-mixture principal-subspace theorem** —
  adjacent extension for two principal points of spherical location mixtures;
  objective remains Euclidean MSE and the result is subsumed for this audit by
  the broader Tarpey–Loperfido forward source.
- **Application-specific semiparametric efficient-score papers** — repeatedly
  instantiate projection off a nuisance tangent space, but add no theorem
  beyond the Bickel–Klaassen–Ritov–Wellner classical source and no hard-cell
  geometry; `DS-STABLE-BASINS-FIXED-POINT-GATE`.

## RETENTION-PLUGIN-VECTOR literature-first pass — 5 September 2026

- **Liu, Bathke & Harrar (2011), “A nonparametric version of Wilks'
  lambda — asymptotic results and small sample approximations”** —
  rank-based statistic, not the moment plug-in; assessed against the
  future `RETENTION-PLUGIN-CLT-FROZEN-VECTOR`.
- **Jolliffe & Lukudu (1993), “The influence of a single observation on
  some standard test statistics”** — one-sample \(t\) and variance tests.
- **Kshirsagar & Gupta (1990), “Direction and collinearity factors of
  Wilks's \(\Lambda\) — a review”** — Bartlett factorization of \(\Lambda\)
  under normal theory; no asymptotics without normality.
- **Josse & Holmes (2016), “Measuring multivariate association and
  beyond”, Statistical Surveys 10:132–167** — survey of association
  coefficients (RV, dCov, kernel measures); no determinant-ratio
  asymptotics.
- **Anderson (1999), “Asymptotic theory for canonical correlation
  analysis”, JMVA 70:1–29** — normal sampling only; Muirhead & Waternaux
  (1980) is the fourth-moment source.
