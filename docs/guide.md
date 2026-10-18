# User guide

ScoreQuant turns a weighted table of score vectors into a few hard bins that keep as much Fisher
information as possible. This page explains the pipeline once, then answers the four questions
that decide a call: which task, which input, which criterion, which solver. Every snippet runs, and
they share one namespace.

```python
import numpy as np

import scorequant as sq

rng = np.random.default_rng(21)
scores = rng.normal(size=(1_200, 2))  # N(mu, I2) at mu0 = 0 has s(x) = x
weights = np.ones(scores.shape[0])
```

## How it works

```text
score  ->  informative subspace  ->  criterion  ->  solver  ->  certificate
```

**Score.** The input is a weighted table of rows \((s_i, w_i)\), one coordinate per parameter,
evaluated at a reference point \(\theta_0\). The score is the gradient of a log density ratio,

$$
s(x)=\nabla_\theta\log\frac{p(x\mid\theta)}{p(x\mid\theta_0)}\bigg|_{\theta=\theta_0},
$$

so absolute densities are never needed. For labels \(b(i)\), with cell masses \(W_b\) and score
sums \(m_b\), the information kept by the bins and the information lost are

$$
I_B=\sum_b \frac{m_b m_b^\top}{W_b},
\qquad
I_\infty-I_B=\sum_b\sum_{i:b(i)=b} w_i\,(s_i-\mu_b)(s_i-\mu_b)^\top\;\succeq\;0 .
$$

Binning can only lose information, refining can only help, and the loss is exactly the scatter of
the score inside each cell.

**Informative subspace.** Directions of \(I_\infty\) below a relative eigenvalue threshold
(`rank_rtol`) are dropped rather than repaired with a ridge. Scores are projected and rescaled but
never translated: \(s=0\) means "no sensitivity", so centering would change the problem.

**Criterion.** `DOptimality` maximizes \(\log\det I_B\), balancing every parameter direction.
`ProfiledDOptimality(interest=...)` maximizes the log determinant of the Schur complement

$$
I_{\rm eff}=I_{\psi\psi}-I_{\psi\lambda}I_{\lambda\lambda}^{-1}I_{\lambda\psi},
$$

for parameters of interest \(\psi\) with nuisance parameters \(\lambda\) profiled from the same
labels. It answers a different question — how to spend the bins when only \(\psi\) is reported —
and may give up a lot of nuisance information to do it. `NormalizedTrace` is weighted k-means in
whitened score space: a well-understood baseline.

**Solver.** Configuration and criterion form a closed table, checked before any optimization runs:

<!-- generated: solver-matrix (do not edit by hand; run `pnpm generate:data`) -->
| Configuration | `optimize_partition` | `fit_quantizer` |
| --- | --- | --- |
| `DExchangeConfig` | `DOptimality`, `ProfiledDOptimality` | `DOptimality` |
| `MahalanobisLloydConfig` | `DOptimality`, `ProfiledDOptimality` | `DOptimality` |
| `KMeansConfig` | — | `NormalizedTrace` |
| `SoftVoronoiConfig` | — | `DOptimality`, `ProfiledDOptimality` |
| `ScalarDPConfig` | — | `DOptimality` |
<!-- /generated: solver-matrix -->

- **Exact exchange** moves one row at a time. A move changes \(I_B\) by a rank-two update, so its
  exact gain is a \(2\times 2\) determinant. Only strictly positive gains are accepted, so the
  objective rises monotonically and the run ends exchange-stable.
- **Guarded Mahalanobis-Lloyd** proposes a whole nearest-centroid relabelling in the current metric
  and keeps it only if the exactly recomputed objective improves. The guard is needed: an
  unguarded step can lower the determinant.
- **Soft Voronoi** optimizes centres with differentiable soft assignments, then hardens them. It
  reports the hardening gap, because the hard result is the one that counts.
- **Weighted k-means** is the trace baseline.
- **The scalar dynamic program** returns the global optimum when the score space is rank one.

**Certificate.** Nothing is certified implicitly; each check is an explicit call. Stability and
geometry are certified at the tolerance the solver stopped at (`gain_tolerance`), never at zero.

## 1. Which task?

> **Will you ever label an event that is not in this table?**

**No — the rows are the final object.** Use `optimize_partition`; the answer is a label vector.
Typical cases: a frozen Monte Carlo template set, a calibration sample, a study of how much a cell
budget can retain.

**Yes — future events must be labelled the same way.** Use `fit_quantizer`; the answer is a rule
that `predict_scores` applies anywhere. Typical cases: a trigger, a gate applied to new runs, a
categorization shipped with an analysis.

`PartitionResult` has no predict method on purpose: many rules reproduce the same labels on a
finite sample and disagree everywhere else. The one exception is a theorem, below.

## 2. Which input?

What you already have decides it:

- Scores already computed: `ScoreSample`, or pass the array straight to `optimize_partition`.
- A component or analytic model: `LinearComponentScore` or `ScoreFunction`, paired with
  `ObservationSample` or `IntegrationSource`.
- Density ratios — analytic, classifier-derived or from a ratio estimator: `DensityRatioScore` (or
  `CentralLogRatioScore` for paired central classifiers), paired with `ObservationSample`.

A source is the measure; a provider maps observations to scores. Estimated scores give a
supplied-score surrogate, never exact Fisher information.

## 3. Which criterion?

| Use | When |
| --- | --- |
| `DOptimality` | Every parameter matters. The default |
| `ProfiledDOptimality(interest=...)` | Some parameters are of interest and the rest are nuisance |
| `NormalizedTrace` | The baseline: after whitening it is exactly weighted k-means |

Run `NormalizedTrace` even when D is the goal: the gap between the two shows whether the
determinant geometry buys anything on your data.

## 4. Which solver?

| Configuration | Choose it when |
| --- | --- |
| `DExchangeConfig` | Default for both D criteria |
| `MahalanobisLloydConfig` | You want whole-sample relabelling in the criterion metric |
| `SoftVoronoiConfig` | You want a reusable rule fitted directly, including for profiled D |
| `KMeansConfig` | Pairs with `NormalizedTrace` |
| `ScalarDPConfig` | The score space is rank one, so the global optimum is computable |

## Fixed sample, D-optimal

```python
partition = sq.optimize_partition(
    scores,
    weights=weights,
    n_bins=5,
    criterion=sq.DOptimality(),
    config=sq.DExchangeConfig(seed=21),
)
diagnostics = {
    "stable": bool(partition.exchange_stable),
    "best_remaining_gain": float(partition.best_remaining_gain),
    "accepted_moves": partition.accepted_moves,
    "scans": partition.scans,
    "efficiency": float(partition.train_report.geometric_mean_retention),
}
```

`max_scans` bounds the work, `batch_moves=False` forces one relocation per scan, and
`solver_restarts` with `init` tries several seeded starts and keeps the best exact objective.

To check labels from anywhere else — another tool, a hand edit — run one exact scan over them:

```python
report = sq.exchange_stability_report(scores, partition.labels, weights=weights)
stability = (bool(report.stable), float(report.best_gain), report.best_move)
```

## The one crossing: compiling a partition

An exchange-stable D partition with nonsingular binned information is already a strict Voronoi
partition in the \(I_B^{-1}\) metric. That rule is canonical, so it can be handed back:

```python
if partition.exchange_stable:
    compiled = partition.compile_quantizer()
    future_bins = compiled.predict_scores(rng.normal(loc=0.2, size=(300, 2)))
```

`compile_quantizer()` checks that the rule reproduces every positive-weight training label and
refuses an unstable or degenerate result. It exists only for `DOptimality`: a globally optimal
profiled partition can violate its own nearest-cell geometry.

## Reusable rule from ready scores

```python
holdout = rng.normal(size=(400, 2))
quantizer = sq.fit_quantizer(
    sq.ScoreSample(scores, weights),
    validation=sq.ScoreSample(holdout),
    n_bins=5,
    criterion=sq.DOptimality(),
    config=sq.SoftVoronoiConfig(seed=21, initializer_restarts=4, max_steps=120, record_every=20),
)
soft_fit = {
    "train": float(quantizer.train_report.geometric_mean_retention),
    "validation": float(quantizer.validation_report.geometric_mean_retention),
    "hardening_gap": float(quantizer.hardening_gap),
    "objective_label": quantizer.trace.objective_label,
}
```

Validation is diagnostic: it never steers the fit. On a held-out sample of true scores, the frozen
rule's retention has a sampling standard error and a Wald interval (conditions in the
[API guide](api.md#held-out-retention-uncertainty)):

```python
uncertainty = sq.retention_uncertainty(holdout, quantizer.predict_scores(holdout), n_bins=5)
error_bar = (uncertainty.status, uncertainty.estimate, uncertainty.confidence_interval)
```

The k-means baseline for comparison:

```python
baseline = sq.fit_quantizer(
    sq.ScoreSample(scores, weights),
    n_bins=5,
    criterion=sq.NormalizedTrace(),
    config=sq.KMeansConfig(seed=21, solver_restarts=4),
)
baseline_efficiency = float(baseline.train_report.geometric_mean_retention)
```

## Parameters of interest with nuisance

For one parameter of interest, compute the certified ceiling first and start from its labels.
The remaining gap is then a real measure of what the solver left on the table:

```python
bound = sq.efficient_score_bound(scores, interest=(0,), weights=weights, n_bins=5)
profiled = sq.optimize_partition(
    scores,
    weights=weights,
    n_bins=5,
    criterion=sq.ProfiledDOptimality(interest=(0,)),
    config=sq.DExchangeConfig(seed=21),
    initial_labels=bound.labels,
)
profiled_gap = float(bound.gap_to(profiled))
```

More than one interest column raises `NotImplementedError` rather than returning an uncertified
number.

## Rank-one score spaces

When the informative score space is one-dimensional, the optimal cells are intervals and the
dynamic program finds the global optimum:

```python
scalar = sq.fit_quantizer(
    sq.ScoreSample(np.asarray(scores[:, :1])),
    n_bins=5,
    criterion=sq.DOptimality(),
    config=sq.ScalarDPConfig(),
)
scalar_efficiency = float(scalar.train_report.geometric_mean_retention)
```

## Proving a small instance optimal

For a small table, global optimality can be decided:

```python
small = rng.normal(size=(24, 2))
incumbent = sq.optimize_partition(small, n_bins=3, config=sq.DExchangeConfig(seed=1))
certificate = sq.certify_partition(small, n_bins=3, incumbent=incumbent.labels)
verdict = (certificate.status, float(certificate.gap), bool(certificate.incumbent_was_optimal))
```

The search is exponential, so `CertificationConfig` caps nodes and instance size. A spent budget
returns `status="budget_exhausted"` with the outstanding gap, never a claim of optimality.

## A classifier signal next to a background

In the HEP pattern the signal ratio comes from a trained classifier and the background from a Monte
Carlo or analytic density. Ratio columns may carry any per-event factor, but only one they all
share: a prior-corrected classifier ratio is taken over the training mixture, a Monte Carlo density
over a fixed base measure, and stacking the two leaves the training-mixture density inside every
score. Return every column in one gauge. For a two-class classifier with training priors
\(\pi\), \(\phi_\text{sig}/\phi_\text{bkg}=(\eta_\text{sig}/\pi_\text{sig})/(\eta_\text{bkg}/\pi_\text{bkg})\),
so in the background gauge the background column is 1 and no Monte Carlo density is needed:

```python
priors = np.array([0.5, 0.5])  # signal, background training priors


def classifier(x):  # stands in for a calibrated predict_proba
    signal = 1.0 / (1.0 + np.exp(-2.0 * x[:, 0]))
    return np.column_stack([signal, 1.0 - signal])


def background_gauge(x):
    eta = classifier(x)
    signal = (eta[:, 0] / priors[0]) / (eta[:, 1] / priors[1])
    return np.column_stack([signal, np.ones_like(signal)])


provider = sq.DensityRatioScore(
    background_gauge,
    sq.IntensityParameterization([0.1, 1.0]),  # signal, background reference yields
    provenance=sq.ScoreProvenance(
        kind="estimated_ratio",
        ratio=sq.RatioProvenance(gauges=("background", "background")),
    ),
)
events = sq.ObservationSample(np.random.default_rng(5).normal(size=(800, 1)))
hep_fit = sq.fit_quantizer(events, provider=provider, n_bins=4)
```

Declaring `gauges` is what lets the provider check them: a classifier column next to a Monte Carlo
density, declared `("training_mixture", "absolute")`, raises `ContractError` instead of returning
wrong scores. `DensityRatioScore.from_classifier` declares the training-mixture gauge for every
column; an undeclared ratio callback is not checked.

## Reading the numbers

Check the same handful every time: the retention spectrum and its geometric mean (the
D-efficiency), the effective rank against the number of parameters, the bin weights,
`exchange_stable` with `best_remaining_gain`, the geometry report's Voronoi violation, and the
provenance behind `information_kind`. Optimizer traces carry an `objective_label`; compare two
traces only under the same label. The [API guide](api.md) defines every field.
