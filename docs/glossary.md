# Glossary

### Bin

One category of a hard partition. After binning, events are represented only by integer
labels or aggregate counts.

### Calibration

The property of a probability estimate that its numerical values are quantitatively
meaningful, not merely correctly ordered. Score construction from a classifier requires
calibrated posteriors (or a ratio-estimation loss), because a ranking score or an arbitrary
monotone transform of a likelihood ratio does not determine the
[density ratios](#density-ratio) the score is built from.

### Compile bridge

The one theorem that turns a sample partitioning result into a reusable quantizer without
inventing a rule: when a D-optimal finite partition is [exchange-stable](#exchange-stability)
and its information matrix is nonsingular, it is provably identical to the nearest-cell
rule in its own \(I_q^{-1}\)-Mahalanobis metric, so `PartitionResult.compile_quantizer()`
returns exactly that rule and verifies label reproduction before returning it — at the
`gain_tolerance` the partition was optimized at, since that is the precision the solver
actually delivers. The bridge
exists for the log determinant only; a profiled-\(D_s\) partition has an analogous
population geometry but no exact finite implication, so it refuses to compile rather than
approximate one.

### D-efficiency

Geometric mean of retained-information eigenvalues, also called the geometric-mean
[retention](#retention). It summarizes balanced local information retention across
informative parameter directions and equals a single retained/unbinned ratio when there is
only one informative direction.

### Density ratio

A model density divided by another model density: \(p(x\mid\theta)/p(x\mid\theta_0)\), or a
component ratio \(\phi_k/\phi_{\rm ref}\). The score is the gradient of a log density
ratio, so ratios are the minimal statistical representation a score provider needs —
absolute normalization cancels, and any common event-wise factor is a free gauge. The factor
must be common to every column: ratio columns taken against different measures do not mix
(see `RatioProvenance.gauges`). Distinct from an [importance ratio](#importance-ratio).

### Efficient score

In a model with interest parameters \(\psi\) and nuisance parameters \(\lambda\), the part
of the interest score \(s_\psi\) that is left after regressing away the nuisance score
\(s_\lambda\): \(e(s) = s_\psi - BC^{-1}s_\lambda\), where \(B\) and \(C\) are blocks of the
unbinned information matrix. Profiled \(D_s\)-optimality's population stationarity
condition is a nearest-cell rule in this projection alone; the directions it annihilates
matter only through the regression coefficient, not directly.

### Exchange stability

A property of a finite labeling: no single row can move to another cell and improve the
objective by more than a stated `gain_tolerance`, evaluated by one exact scan over every
admissible relocation. `PartitionResult.exchange_stable` records whether the solver's own
output has this property at the tolerance it ran at, and `exchange_stability_report` checks
it for a labeling from any source, recording the tolerance on the report. Exchange
stability is necessary for the [compile bridge](#compile-bridge) but is checked
independently of it.

### Finite assignment

See [sample partitioning](#sample-partitioning).

### Fisher information

Expected outer product of the score. It measures local model sensitivity, not estimator
bias.

### Hard quantizer

A deterministic score-space mapping that assigns every score to exactly one bin. Contrast
with a randomized quantizer, which assigns a probability distribution over bins.

### Importance ratio

The factor \(p_{\theta_0}(x)/g(x)\) that reweights a sample drawn from a proposal
distribution \(g\) so that weighted averages estimate expectations under the reference law.
It is a property of the measure and enters ScoreQuant as source weights — never through a
score provider, which is where [density ratios](#density-ratio) live.

### Intensity

An unnormalized event-rate model. Unlike a probability density, its integral may encode
expected yield rather than one.

### Likelihood

The probability density or mass of observed data, viewed as a function of model
parameters.

### Likelihood ratio

Ratio of two likelihoods or component densities; see [density ratio](#density-ratio). A
classifier posterior divided by its class prior can estimate component ratios up to a
common event-wise factor, and direct estimators (KLIEP, uLSIF, neural ratio estimation)
target the same object without a classifier.

### Oracle

A calculation using the exact data-generating likelihood or exact score. A learned
classifier ratio is not an oracle merely because it is used without binning.

### Population design

Optimization of a measurable rule under a specified score law itself, rather than only its
finite realization. It is the inductive half of [space quantization](#space-quantization);
`fit_quantizer` performs it exactly when given an `IntegrationSource` and approximately, as
empirical inductive fitting, when given a finite sample.

### Ratio closure

The identity that exact [density ratios](#density-ratio) relative to a reference measure
integrate to one under it. `ratio_closure_report` measures the residual; a large value
flags estimator bias, a misdeclared training prior, or a measure mismatch. The check is
necessary but not sufficient, so closure never upgrades estimated provenance to exact.

### Reference point

Parameter value \(\theta_0\) at which scores and Fisher information are evaluated.

### Retention

The fraction of unbinned Fisher information a labeling keeps, reported per informative
direction as an eigenvalue ratio of \(I_{\text{full}}^{-1}I_q\) and summarized as
`geometric_mean_retention` (equivalently, [D-efficiency](#d-efficiency)) or
`arithmetic_mean_retention`. A retention of \(1\) loses nothing; a retention of \(0\) means
a direction carries no information after binning. \(1/\sqrt{\text{retention}}\) is the
resulting inflation of a Gaussian standard error in that direction.

### Sample partitioning

The transductive task: given one fixed weighted table of scores, choose the labels that
maximize the retained information *of those rows*. `optimize_partition` performs it and
returns a `PartitionResult`, which deliberately has no predict method — a labeling of one
table does not by itself determine what happens to a score that was not in it. Also called
finite assignment.

### Score

Gradient of log likelihood with respect to parameters. It describes an event's local
parameter sensitivity.

### Score law

The distribution or intensity measure induced on score space by a reference source and an
observation-to-score provider.

### Score provider

A map from observations to score vectors. It does not supply a reference measure; a source
must supply that separately.

### Simplex

Set of nonnegative fractions that sum to one. A \(K\)-component mixture has \(K-1\) free
directions.

### Source

An empirical table or integration rule that supplies the reference measure used by an
objective.

### Space quantization

The inductive task: given a score law — an empirical sample or a density over a bounded
box — choose a reusable rule that assigns any future score to a bin. `fit_quantizer`
performs it and returns a `QuantizerResult`, whose answer is a geometric object (a
transform, centers, sometimes a metric) with a well-defined `predict_scores` method,
because the rule is defined everywhere rather than only on the rows it was fit from.

### Surrogate information

Between-cell information computed from estimated scores. It is exact for those supplied
vectors but not automatically Fisher information of the original statistical model.

### Template

Conditional bin probabilities \(P(B_j\mid k)\) for a model component \(k\).

### Three doors

The three ways a weighted table of score rows can arise, differing in which statistical
representation you already possess rather than in what the optimizer does with the result:
door 1, precomputed `(event, score)` rows supplied directly as a `ScoreSample`; door 2,
component densities or an analytic score model, reached through an `ObservationSample` or
`IntegrationSource` together with `LinearComponentScore` or `ScoreFunction`; door 3,
[density ratios](#density-ratio) — analytic, classifier-derived, or from a direct ratio
estimator — reached through an `ObservationSample` with `scorequant.ratios.DensityRatioScore`
or `CentralLogRatioScore`. All three doors open onto the same object — a weighted score
table — and are validated together with a source, never supplied alone.

### Whitening

Linear scaling of informative score directions so that their unbinned Fisher matrix is the
identity. ScoreQuant does not mean-center scores while whitening.
