# ScoreQuant

ScoreQuant is a Python library for assigning observations to a limited number of hard bins
while retaining Fisher information about model parameters at a chosen reference point.
It optimizes bins from **scores** — gradients of the log likelihood — supplied directly or
computed through an explicit provider. It does not train the upstream classifier or fit the
final model parameters.

[![CI](https://github.com/VitalyVorobyev/scorequant/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/scorequant/actions/workflows/ci.yml)
[Documentation](https://vitalyvorobyev.github.io/scorequant/docs/) ·
[Walkthroughs](https://vitalyvorobyev.github.io/scorequant/walkthroughs/)

## Install

```bash
uv add scorequant
```

Python 3.12 or newer. JAX and Optax are installed dependencies; NumPy is also supported for
execution. Set `JAX_ENABLE_X64=1` before starting Python when using JAX in double precision.

## Quick start

For a Gaussian location model with unit covariance at reference mean zero, observations are
already score vectors. Fit a six-bin rule on one sample and apply it to new observations:

```python
import numpy as np
import scorequant as sq

rng = np.random.default_rng(7)
sample = sq.ScoreSample(
    rng.normal(size=(4_000, 2)),
    provenance=sq.ScoreProvenance(kind="exact", reference_point=(0.0, 0.0)),
)
fit = sq.fit_quantizer(
    sample, n_bins=6, criterion=sq.DOptimality(), config=sq.DExchangeConfig(seed=7)
)
rule = fit.quantizer
future_bins = rule.predict_scores(rng.normal(size=(1_000, 2)))
print(float(fit.train_report.geometric_mean_retention))
```

The printed value is empirical D-efficiency on the training sample, not a guarantee for new
data. It is the geometric mean of retained-information eigenvalues, relative to the unbinned
sample. Estimated scores instead yield **supplied-score surrogate** information; declaring
provenance does not validate the score estimator.

## Choose the task

| Need | Entry point | Output |
| --- | --- | --- |
| Labels for a fixed weighted sample | `optimize_partition(scores, ...)` | `PartitionResult.labels`; no predictor |
| A rule for future scores | `fit_quantizer(source, provider=..., ...)` | `QuantizerResult.quantizer`, with `predict_scores`, `save` and `load` |

A `ScoreSample` needs no provider. Observation samples and integration sources require one;
prediction keeps observation-to-score conversion explicit. Model density ratios belong to the
provider; importance weights belong to the source.

`DOptimality` preserves information jointly across parameters. `ProfiledDOptimality` targets
parameters of interest after profiling nuisance parameters. `NormalizedTrace` provides a
whitened k-means baseline.

<!-- generated: solver-matrix (do not edit by hand; run `pnpm generate:data`) -->
| Configuration | `optimize_partition` | `fit_quantizer` | Contract |
| --- | --- | --- | --- |
| `DExchangeConfig` | `DOptimality`, `ProfiledDOptimality` | `DOptimality` | Exact positive-gain relocations; monotone objective; terminates exchange-stable |
| `MahalanobisLloydConfig` | `DOptimality`, `ProfiledDOptimality` | `DOptimality` | A batch is adopted only if the exactly rebuilt objective improves; optional exact-exchange guard |
| `KMeansConfig` | — | `NormalizedTrace` | Weighted $k$-means in whitened score space |
| `SoftVoronoiConfig` | — | `DOptimality`, `ProfiledDOptimality` | Differentiable soft optimization then hardening, with the hardening gap reported |
| `ScalarDPConfig` | — | `DOptimality` | The exact global interval solution for rank-one score space |
<!-- /generated: solver-matrix -->

Exchange stability is a local certificate. Global certification is a separate, budgeted D-only
operation. Compiling a fixed partition into a rule requires verified D geometry; generic
profiled-Ds compilation is unsupported. See the [API guide](https://vitalyvorobyev.github.io/scorequant/docs/api/)
for solver limits, tolerances and refusals.

## Further reading

- [Get started](https://vitalyvorobyev.github.io/scorequant/get-started/): one complete runnable workflow.
- [Four walkthroughs](https://vitalyvorobyev.github.io/scorequant/walkthroughs/): interferometry, density ratios, cytometry and HEP.
- [Theory](https://vitalyvorobyev.github.io/scorequant/docs/book/) and [related work](https://vitalyvorobyev.github.io/scorequant/docs/related-work/).
- [Contributor guidance](AGENTS.md) and [MIT license](LICENSE).
