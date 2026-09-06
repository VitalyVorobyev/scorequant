---
title: How the API names each result
sidebar_label: API and theorems
sidebar_position: 7
---

import {ReferenceLink} from "@site/src/components/ReferenceLink";

# How the API names each result

The library is written so that the mathematics is visible in the object names rather than hidden
behind one opaque `fit`. This page is the map in both directions: from a message or a class name to
the result behind it, and from a result to the object that carries it. Field-level documentation
lives in the <ReferenceLink to="api/">API guide</ReferenceLink>; this page names the
*reason* each object exists.

## Two kinds of error, and why the difference matters

`ContractError` means the call itself is wrong — a bad shape, an unsupported pairing, a negative
weight. It is detectable from the arguments alone and the remedy is to change the call.

`RefusalError` means the call is valid but a **theorem-backed condition fails on your data**. It
carries a `counterexample` attribute holding the registry id of the exact example that forbids the
operation, and that id is printed in square brackets at the end of the message. A refusal is not a
numerical failure to be retried with a different seed; it is the library declining to assert
something that is known to be false in general.

Both derive from `ScoreQuantError`, so `except ScoreQuantError` catches everything the library
raises deliberately.

## Refusal messages and where they come from

Codes beginning `CE-` name counterexample fixtures rather than claims; the claim each one is
a boundary of is linked beside it.

| Code in the message | What triggered it | The result it comes from | What to do |
| --- | --- | --- | --- |
| `CE-DS-GLOBAL-GEOMETRY-001` | `compile_quantizer()` on a result fitted under `ProfiledDOptimality` | [DS-GLOBAL-NONGEOMETRIC](/research/claim-record#ds-global-nongeometric) — a global profiled optimum can violate its own first-order rule, so profiled labels have no canonical extension | Fit a reusable rule directly with `fit_quantizer` instead of converting a fixed-sample partition |
| `CE-D-VORONOI-CONVERSE-001` | `compile_quantizer()` on a partition whose stability certificate is `False` | [D-VORONOI-NOT-EXCHANGE](/research/claim-record#d-voronoi-not-exchange) — looking geometric is strictly weaker than being exchange-stable | Inspect `best_remaining_gain`; raise `max_scans`, or set `guard="exchange"` on the guarded batch solver |
| `CE-D-UNMERGED-DUPLICATES-001` | The terminal rule would relabel a training row by more than the gain tolerance | [D-UNMERGED-DUPLICATES-FAIL](/research/claim-record#d-unmerged-duplicates-fail) — split duplicate atoms can be vacuously stable while defeating label reproduction | Merge or consistently label coincident score rows; inspect `geometry.maximum_violation_gain` |
| `CE-DS-MARGINS-RANK-VACUITY-001` | A profiled fit or partition whose cell count cannot generate nonsingular binned information | The cardinality boundary of [OPEN-DS-MARGINS-AT-OPTIMA](/research/claim-record#open-ds-margins-at-optima), and behind it the rank ceiling [FI-RANK-CEILING](/research/claim-record#fi-rank-ceiling) | Raise `n_bins` above the score dimension. On an exactly centred sample no sample size helps — this is arithmetic, not a data problem |

An unsupported criterion-and-configuration pairing raises `ContractError` rather than a refusal,
because the library never implemented that combination in the first place; the message names the
configuration, the task, and the criteria that task does support.

## Criteria

| Object | The result it optimizes |
| --- | --- |
| `DOptimality` | The log determinant of the binned information ([FI-QUANT-IDENTITY](/research/claim-record#fi-quant-identity)). Chosen partly because it is invariant under invertible reparameterization ([D-REPARAM-INVARIANCE](/research/claim-record#d-reparam-invariance)) |
| `ProfiledDOptimality` | The log determinant of the Schur complement for the declared parameters of interest ([DS-SCHUR](/research/claim-record#ds-schur)). The `interest` argument is the parameter block; at least one nuisance column must remain |
| `NormalizedTrace` | Retained normalized trace, which after Fisher whitening is exactly weighted k-means ([TRACE-WHITENED-KMEANS](/research/claim-record#trace-whitened-kmeans)) |

There is no E-optimality criterion, and the reason is a counterexample rather than an omission:
[E-GLOBAL-GEOMETRY-FAILS](/research/claim-record#e-global-geometry-fails).

## Solver configurations

| Object | The result it implements |
| --- | --- |
| `DExchangeConfig` | Exact positive-gain one-point exchange: the closed-form relocation gain ([D-LOGDET-GAIN](/research/claim-record#d-logdet-gain)) on the rank-two update ([D-RANK2-MOVE](/research/claim-record#d-rank2-move)), accepted only on strictly positive exact gains so the search terminates ([D-EXCHANGE-TERMINATES](/research/claim-record#d-exchange-terminates)) |
| `MahalanobisLloydConfig` | Batch reassignment as a *proposal*, accepted on the exact objective ([D-GUARDED-LLOYD](/research/claim-record#d-guarded-lloyd)). The guard exists because the unguarded step is not monotone ([D-LLOYD-NONMONOTONE](/research/claim-record#d-lloyd-nonmonotone)); `guard="exchange"` hands the labels to the exact engine so the reported state stays compilable |
| `KMeansConfig` | Weighted k-means on whitened scores, which is the trace criterion exactly ([TRACE-WHITENED-KMEANS](/research/claim-record#trace-whitened-kmeans)) |
| `SoftVoronoiConfig` | Optimization of a randomized quantizer's exact information ([SOFT-RANDOMIZED-FIM](/research/claim-record#soft-randomized-fim)), which exists because the hard empirical objective is piecewise constant and has no usable gradient ([HARD-GEOMETRIC-EMPIRICAL-PIECEWISE-CONSTANT](/research/claim-record#hard-geometric-empirical-piecewise-constant)). Its guarantees are those of smooth nonconvex optimization and no more ([SOFT-FIXED-TEMP-STATIONARY](/research/claim-record#soft-fixed-temp-stationary)) |
| `ScalarDPConfig` | Exact interval dynamic programming on one score coordinate — the optimal cells are intervals, so the program returns a global optimum rather than a local one ([DS-SCALAR-EFFICIENT-DP](/research/claim-record#ds-scalar-efficient-dp)) |

## Results and certificates

| Object or field | The statement it carries |
| --- | --- |
| `PartitionResult` | A labelling of one fixed sample — the first of the [three questions](/research/the-problem). It deliberately has no `predict` method, because a finite labelling does not determine its own extension |
| `PartitionResult.compile_quantizer()` | The one exception: an exchange-stable, nonsingular `DOptimality` result compiles into a Mahalanobis rule ([D-FINITE-INDUCTIVE-CLOSURE](/research/claim-record#d-finite-inductive-closure)). It is bookkeeping, not a second fit |
| `GeometryReport` | Measures both sides of the exchange-implies-Voronoi theorem ([D-EXCHANGE-IMPLIES-VORONOI](/research/claim-record#d-exchange-implies-voronoi)) on the terminal state instead of assuming them. `guaranteed_violation_gain` is the theorem's lower bound; `maximum_violation_gain` is the exact gain the theorem bounds; `maximum_separation_residual` checks the leverage inequality ([D-LEVERAGE](/research/claim-record#d-leverage)), so a positive value indicates numerical trouble rather than a better partition |
| `StabilityReport` | One complete exact scan certifying that no single relocation improves a supplied labelling, at a stated `gain_tolerance`. A labelling certified at one tolerance is not certified at a smaller one |
| `PartitionCertificate` | The outcome of a bounded global search using the singleton-completion ceiling ([D-BB-SINGLETON-BOUND](/research/claim-record#d-bb-singleton-bound)). `status` is either `"optimal"` or `"budget_exhausted"`; the second proves nothing about the incumbent, and `incumbent_was_optimal` is `False` whenever it occurs |
| `EfficientScoreBound` | The full-data efficient-score ceiling on profiled information ([DS-EFFICIENT-SCORE-DOMINATION](/research/claim-record#ds-efficient-score-domination)), attained for one parameter of interest by the exact interval program. Its `labels` also serve as a strong initializer |
| `ProfiledGeometryReport` | Diagnoses the finite efficient-semimetric gap — the quantity that the profiled case has instead of a geometry theorem ([DS-FINITE-GEOMETRY-FAILS](/research/claim-record#ds-finite-geometry-fails)) |

## Information reports

| Field | The statement it reports |
| --- | --- |
| `InformationReport.retained_matrix`, `retained_eigenvalues` | The normalized retention spectrum, every eigenvalue in the unit interval ([INFO-RETENTION-SPECTRUM](/research/claim-record#info-retention-spectrum)) |
| `InformationReport.geometric_mean_retention` | Determinant efficiency against unbinned inference ([INFO-D-EFFICIENCY](/research/claim-record#info-d-efficiency)) |
| `InformationReport.arithmetic_mean_retention`, and the eigenvalue list | Direction-resolved diagnostics ([INFO-DIRECTIONAL-DIAGNOSTICS](/research/claim-record#info-directional-diagnostics)). They are reported separately because optimizing the determinant does not equalize retention across directions — a caution, not a theorem |
| `InformationReport.effective_rank` | The rank the binned information actually reached, against the ceiling `min(d, K-1)` ([FI-RANK-CEILING](/research/claim-record#fi-rank-ceiling)) |
| `ProfiledInformationReport.geometric_mean_retention` | Profiled efficiency ([INFO-DS-EFFICIENCY](/research/claim-record#info-ds-efficiency)), undefined where the Schur complement is singular |

## Scores, sources and providers

The library separates *where the reference measure comes from* (a sample, or a quadrature grid)
from *how observations become scores* (a provider). That split exists because the guarantees above
are statements about a weighted table of score rows and nothing else.

| Object | The statement behind it |
| --- | --- |
| `ScoreSample` | The scores are supplied directly; provenance is exact and every identity on these pages applies as written |
| `LinearComponentScore`, `scores_from_components` | Component densities give the local score of a linear mixture ([MIXTURE-RATIO-SCORE](/research/claim-record#mixture-ratio-score)) |
| `DensityRatioScore`, `CentralLogRatioScore` | The local score is the gradient of a log density *ratio*, so normalized densities are unnecessary ([RATIO-LOCAL-SCORE](/research/claim-record#ratio-local-score)) |
| `ratios_from_posteriors` | Calibrated classifier posterior odds recover component density ratios when the training priors are known ([CLASSIFIER-RATIO-ORACLE](/research/claim-record#classifier-ratio-oracle)) |
| `RatioProvenance`, `ScoreProvenance` | The record of which of the above applies. Estimated ratios never claim exact Fisher semantics, because the retained information is governed by the *true* score under the estimated rule ([PROXY-TRUE-RETAINED-FI](/research/claim-record#proxy-true-retained-fi)) and no theorem yet bounds the difference |
| `ratio_closure_report` | A necessary check that model ratios integrate to one under the reference measure. It is necessary and not sufficient: a small residual never upgrades estimated provenance to exact |

## Next

[The book, chapter by chapter](/research/book-contents) points at the derivations behind every row
of these tables.
