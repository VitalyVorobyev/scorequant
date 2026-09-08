# Choosing your workflow

Four questions decide a ScoreQuant call: which task, which door, which criterion, which solver.
This page answers them in that order. Every snippet runs, and they share one namespace.

Install with `uv add scorequant`, or `pip install scorequant` outside a `uv` project.

```python
import numpy as np

import scorequant as sq

rng = np.random.default_rng(21)
scores = rng.normal(size=(1_200, 2))  # N(mu, I2) at mu0 = 0 has s(x) = x
weights = np.ones(scores.shape[0])
```

## 1. Which task?

> **Will you ever label an event that is not in this table?**

**No — the rows are the final object.** Use `optimize_partition`. You are solving a finite
assignment problem, and the answer is a label vector for those rows. Typical cases: a frozen Monte
Carlo template set, a fixed calibration sample, a study of how much information a given cell budget
can retain.

**Yes — future events must be labeled the same way.** Use `fit_quantizer`. You are choosing a
geometric rule on score space, and the answer is something `predict_scores` can apply anywhere.
Typical cases: a trigger, a gate applied to new runs, a categorization shipped with an analysis.

`PartitionResult` has no predict method, and that is deliberate: many different rules reproduce the
same labels on a finite sample and disagree everywhere else, so the sample optimum does not name
one of them. The exception is a theorem — see the crossing below.

## 2. Which door?

The door is fixed by what you already have, not by preference:

- Scores already computed \(\rightarrow\) `ScoreSample`, or pass the array straight to
  `optimize_partition`.
- A component or analytic model \(\rightarrow\) `LinearComponentScore` or `ScoreFunction`, paired
  with `ObservationSample` or `IntegrationSource`.
- Density ratios — analytic, classifier-derived, or from a direct ratio estimator
  \(\rightarrow\) `DensityRatioScore` (or `CentralLogRatioScore` for paired central classifiers),
  paired with `ObservationSample`.

[Three doors](three-doors.md) works each one through in full, including the source-versus-provider
contract and the shape rules.

## 3. Which criterion?

| Use | When |
| --- | --- |
| `DOptimality` | Every parameter matters. Maximizes \(\log\det I_B\), balancing all information directions; the default |
| `ProfiledDOptimality(interest=...)` | Some parameters are of interest and the rest are nuisance. Maximizes the Schur complement of the interest block under the same labels |
| `NormalizedTrace` | You want the well-understood baseline: after whitening it is exactly weighted k-means |

`NormalizedTrace` is worth running even when `DOptimality` is the goal, because the gap between the
two tells you whether the determinant geometry is buying anything on your data.

## 4. Which solver?

| Configuration | Choose it when |
| --- | --- |
| `DExchangeConfig` | Default for both D criteria. Exact positive-gain relocation, monotone, terminates exchange-stable |
| `MahalanobisLloydConfig` | You want whole-sample relabeling in the criterion metric rather than row-by-row relocation; each proposed batch is still verified against the exactly rebuilt objective before it is accepted |
| `SoftVoronoiConfig` | You want a reusable rule fitted directly in score space, including for profiled \(D_s\), and can accept a local optimum with a reported hardening gap |
| `KMeansConfig` | Pairs with `NormalizedTrace`, and makes a fast, deterministic baseline |
| `ScalarDPConfig` | The score space is rank one. Then the interval dynamic program returns the global optimum, not a local one |

Unsupported pairs are rejected before optimization starts, so a mistake here is an immediate error
rather than a silent substitution.

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

The exchange runs until no relocation improves the objective, accepting many verified relocations
per scan; `max_scans` bounds the work, `batch_moves=False` forces one relocation per scan, and
`solver_restarts` with `init` searches several seeded starting labelings and keeps the best exact final
objective.

To check labels this solver did not produce — an external tool, a hand edit, a batch run stopped by
`guard="reject"` — run one exact scan over them:

```python
report = sq.exchange_stability_report(scores, partition.labels, weights=weights)
stability = (bool(report.stable), float(report.best_gain), report.best_move)
```

## The one crossing: compiling a partition

If, and only if, an exchange-stable D partition has nonsingular between-cell information, its
labels are already the strict Voronoi partition of the \(I_B^{-1}\)-Mahalanobis metric. That rule
is canonical, so it can be handed back:

```python
if partition.exchange_stable:
    compiled = partition.compile_quantizer()
    future_bins = compiled.predict_scores(rng.normal(loc=0.2, size=(300, 2)))
```

`compile_quantizer()` verifies that the compiled rule reproduces every positive-weight training
label, and refuses an unstable or degenerate result. It exists only for `DOptimality`: exact
fixtures show that a globally optimal profiled partition can violate the corresponding nearest-cell
geometry, so a profiled result has no compilation method that could succeed by accident.

## Reusable rule from ready scores

Fit the rule directly when that is what you actually want, and let validation stay diagnostic:

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

The validation number is a point estimate. On a held-out sample of true scores the frozen rule's
retention carries a sampling standard error and a Wald interval — sampling variability only, for
true scores drawn from the reference law, never the bias an estimated score adds; the conditions and
the statuses that withhold the interval are in [the API guide](api.md#held-out-retention-uncertainty):

```python
uncertainty = sq.retention_uncertainty(holdout, quantizer.predict_scores(holdout), n_bins=5)
error_bar = (uncertainty.status, uncertainty.estimate, uncertainty.confidence_interval)
```

The hardening gap is the difference between the last soft objective and the retention of the final
hard labels. A large gap means the annealed surrogate did not commit; judge the run by the hard
number, never by the soft one.

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

For a profiled objective, compute the certified ceiling first and use its labels as the
initializer. The bound is exact for one interest parameter, so the remaining gap is a real
measurement of how much the solver left on the table:

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

`gap_to` is nonnegative up to floating-point error. More than one interest column raises
`NotImplementedError` rather than returning an uncertified number.

## Rank-one score spaces

When the informative score space is one-dimensional — one parameter, or a model whose scores
collapse to a single direction — the optimal cells are ordered intervals and the exact dynamic
program finds the global optimum:

```python
scalar = sq.fit_quantizer(
    sq.ScoreSample(np.asarray(scores[:, :1])),
    n_bins=5,
    criterion=sq.DOptimality(),
    config=sq.ScalarDPConfig(),
)
scalar_efficiency = float(scalar.train_report.geometric_mean_retention)
```

A higher-rank score space is rejected by name rather than approximated, and `max_rows` bounds the
exact quadratic recursion.

## Proving a small instance optimal

For a genuinely small table, global optimality can be decided rather than assumed:

```python
small = rng.normal(size=(24, 2))
incumbent = sq.optimize_partition(small, n_bins=3, config=sq.DExchangeConfig(seed=1))
certificate = sq.certify_partition(small, n_bins=3, incumbent=incumbent.labels)
verdict = (certificate.status, float(certificate.gap), bool(certificate.incumbent_was_optimal))
```

The search is exponential in the number of distinct score atoms, so `CertificationConfig` guards
both the node budget and the instance size and refuses an oversized problem by name. A spent budget
returns `status="budget_exhausted"` with the outstanding gap — never a claim of optimality.

## What to read afterwards

Whatever the path, check the same handful of numbers: the retention spectrum and its geometric mean
(the D-efficiency), the effective rank against the number of parameters, the bin weights and
effective sample sizes, `exchange_stable` with `best_remaining_gain`, the geometry report's
Voronoi violation, and the score provenance behind `information_kind`. The
[API guide](api.md) states what each one means; [Method overview](method.md) explains where they
come from.
