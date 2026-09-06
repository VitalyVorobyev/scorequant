# Examples

Every page here is a runnable story: problem, data, an API walkthrough with executed
snippets, and a discussion of which task, door, criterion, and solver were in play. The
first three enter through each of the three doors in turn; the Michelson-phase page covers
the analytic `ScoreFunction` route and the NumPy backend at once; the shootout puts every
solver and baseline on one problem; the next five are theory demonstrations — a
nuisance-parameter criterion, a soft relaxation, two counterexamples, and global
certification; the last two point into real data — a real classifier-to-score bridge on the
public FAIR Universe HiggsML dataset, and the complete FlowCyt population-fraction study.

Eleven of the twelve pages have a matching notebook under
[`examples/notebooks/`](https://github.com/VitalyVorobyev/scorequant/tree/main/examples/notebooks)
that runs the same story at a larger, more decisive scale. See [Three
doors](../book/ch04-scores-and-doors.md) for what a door is, and [Diagnostics and choosing a
method](../book/ch14-choosing-a-method.md) for a task-first decision guide.

| Page | Demonstrates | Task(s) | Door | Notebook |
| --- | --- | --- | --- | --- |
| [door1-score-events](door1-score-events.md) | Precomputed `(event, score)` rows, an exchange-stability certificate, and the compile bridge from a partition into a reusable rule | Both | 1 | [`door1_score_events.ipynb`](https://github.com/VitalyVorobyev/scorequant/blob/main/examples/notebooks/door1_score_events.ipynb) |
| [door2-mixture-densities](door2-mixture-densities.md) | Analytic component densities to a binned mixture-fraction measurement, with a two-parameter `IntegrationSource` fit | Space quantization | 2 | [`door2_mixture_densities.ipynb`](https://github.com/VitalyVorobyev/scorequant/blob/main/examples/notebooks/door2_mixture_densities.ipynb) |
| [door3-classifier](door3-classifier.md) | Classifier-derived density ratios turned into scores, retention against classifier quality, the ratio-closure check, and the surrogate-information caveat | Space quantization | 3 | [`door3_classifier.ipynb`](https://github.com/VitalyVorobyev/scorequant/blob/main/examples/notebooks/door3_classifier.ipynb) |
| [michelson-phase](michelson-phase.md) | An analytic `ScoreFunction` against a bounded `IntegrationSource` on the NumPy backend, a closed-form profiled ceiling, aliasing in a naive detector segmentation, the compiled D rule's comb, and the fragments of the profiled partition diagnosed | Both | 2 | [`michelson_phase.ipynb`](https://github.com/VitalyVorobyev/scorequant/blob/main/examples/notebooks/michelson_phase.ipynb) |
| [solver-shootout](solver-shootout.md) | Every solver the library dispatches, and the three canonical baselines, on one two-parameter problem | Both | 1 | [`solver_shootout.ipynb`](https://github.com/VitalyVorobyev/scorequant/blob/main/examples/notebooks/solver_shootout.ipynb) |
| [nuisance-profiled-ds](nuisance-profiled-ds.md) | Plain D against profiled \(D_s\) for a signal fraction with a floating background shape, an efficient-score ceiling, and a DP initializer | Sample partitioning | 1 | [`nuisance_profiled_ds.ipynb`](https://github.com/VitalyVorobyev/scorequant/blob/main/examples/notebooks/nuisance_profiled_ds.ipynb) |
| [soft-purification](soft-purification.md) | Responsibility-space relaxation, a temperature schedule, the hardening gap, and purification | Space quantization | 1 | [`soft_purification.ipynb`](https://github.com/VitalyVorobyev/scorequant/blob/main/examples/notebooks/soft_purification.ipynb) |
| [lloyd-nonmonotone](lloyd-nonmonotone.md) | The manuscript's non-monotone Lloyd step, and the guard's acceptance trace rescuing it | Sample partitioning | 1 | [`lloyd_nonmonotone.ipynb`](https://github.com/VitalyVorobyev/scorequant/blob/main/examples/notebooks/lloyd_nonmonotone.ipynb) |
| [ds-geometry-counterexample](ds-geometry-counterexample.md) | An exact rational fixture whose globally optimal profiled-\(D_s\) partition violates its own induced geometry, and why there is no compile bridge | Sample partitioning | 1 | [`ds_geometry_counterexample.ipynb`](https://github.com/VitalyVorobyev/scorequant/blob/main/examples/notebooks/ds_geometry_counterexample.ipynb) |
| [global-certification](global-certification.md) | Branch-and-bound optimality certificates and a multi-restart hit-rate study | Sample partitioning | 2 | [`global_certification.ipynb`](https://github.com/VitalyVorobyev/scorequant/blob/main/examples/notebooks/global_certification.ipynb) |
| [hep-classifier](../usecases/hep/index.md) | A signal-vs-background classifier and a `tes` minus/plus classifier feeding two different ratio doors on the public FAIR Universe HiggsML dataset, profiled \(D_s\) against a certified ceiling, and a delta convergence study | Both | 3 | [`hep_classifier.ipynb`](https://github.com/VitalyVorobyev/scorequant/blob/main/examples/notebooks/hep_classifier.ipynb) |
| [flowcyt-teaser](flowcyt-teaser.md) | A pointer into the complete real-data FlowCyt study: 600,000 real cells, a classifier-derived score, and a full population-fraction measurement | Space quantization | 3 | — see the [FlowCyt study](../usecases/flowcyt/index.md) |
