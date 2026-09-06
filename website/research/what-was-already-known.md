---
title: What was already known
sidebar_label: What was already known
sidebar_position: 3
---

import {ReferenceLink} from "@site/src/components/ReferenceLink";

# What was already known

Choosing a quantizer so that it preserves Fisher information is not a new idea. It sits at the
intersection of four traditions that developed largely independently, and most of the ingredients
used here are established results in one of them. This page is a plain-English tour of that
territory; the full survey, with citations and DOIs, is the
<ReferenceLink to="related-work/">related-work page in the reference</ReferenceLink>, and the sources behind the machinery used directly are in the
<ReferenceLink to="bibliography/">bibliography</ReferenceLink>.

## Four traditions

**Optimal experimental design** is where "maximize the determinant of an information matrix"
became a standard objective. Kiefer and Wolfowitz's equivalence between D- and G-optimality is
what turns a log-determinant objective into a local sensitivity condition, and it is the reason
the inverse information matrix keeps appearing as the natural metric. Whittle extended the
equivalence to general concave criteria. Näther and Reinsch developed the case where only some
parameters are of interest and the rest are nuisance — the *profiled* case, whose objective is a
Schur complement of the information matrix
([DS-SCHUR](/research/claim-record#ds-schur)). The optimization variable in this literature is a design
*measure*, not a hard assignment of observations to cells, so the language and the matrix criteria
transfer but the feasible set does not.

**Quantization for estimation** asks how to transmit a finite number of bits while losing as
little parameter information as possible. Venkitasubramaniam, Tong and Swami stated the problem
directly for distributed estimation and introduced score-function quantizers as the optimal or
benchmark structure — this is direct prior art for the idea of quantizing the score at all.
Farias and Brossier worked out the scalar high-resolution theory. Barnes, Han and Özgür
characterized quantized Fisher information geometrically in terms of conditional score means, and
solved the one-bit Gaussian location problem exactly; this is the closest theoretical predecessor
of the score-space formulation used here, and it is the source the retained-information identity
([FI-QUANT-IDENTITY](/research/claim-record#fi-quant-identity)) is attributed to. Dülek proved that for
exponential families a deterministic K-level quantizer depending only on sufficient statistics
exists, with a convex-polytope optimal partition for the trace criterion. **Polyhedral quantizer
geometry is therefore already known and cannot be claimed as new** — a point the ledger enforces
against every geometry statement on the following pages.

**Determinant clustering** has a long history in cluster analysis: Friedman and Rubin's invariant
grouping criteria, Marriott's practical study, Scott and Symons' likelihood-ratio clustering.
These usually minimize within-cluster scatter or maximize a likelihood ratio rather than
optimizing a between-cell Fisher matrix, so any novelty claim about "determinant clustering" has
to be narrow. The relocation-style solver family is Hartigan's method, analyzed against Lloyd's by
Telgarsky and Vattani; the centroidal-Voronoi machinery is Du, Faber and Gunzburger; consistency
of k-means is Pollard. All four are load-bearing attributions for results on the next page.

**Inference-aware categorization** is a recent line, mostly from particle physics, that optimizes
summaries or bins directly for the sensitivity of the downstream statistical analysis rather than
for a proxy loss. INFERNO trains a neural summary against a differentiable approximation of a
binned likelihood's uncertainty. ThickBrick optimizes event selection and categorization for
signal significance with an explicitly Lloyd-like iteration. GATO and BOBR optimize
multidimensional bin boundaries of classifier discriminants, by gradient descent and by Bayesian
optimization respectively. The neighbouring simulation-based-inference literature supplies the
scores themselves — the local score as a learned summary, built on calibrated classifier
likelihood ratios ([RATIO-LOCAL-SCORE](/research/claim-record#ratio-local-score),
[CLASSIFIER-RATIO-ORACLE](/research/claim-record#classifier-ratio-oracle)) or on direct density-ratio
estimators that skip the classification step
([MIXTURE-RATIO-SCORE](/research/claim-record#mixture-ratio-score)).

## What that literature already settles

The survey states the boundary as a table. In plain terms, all of the following were established
before this project and are cited here rather than claimed:

- A quantizer can be chosen to maximize Fisher information.
- The score, or a sufficient statistic, is the natural space in which to do it.
- A trace-optimal multivariate quantizer can have polyhedral geometry.
- Normalized trace after Fisher whitening is weighted k-means
  ([TRACE-WHITENED-KMEANS](/research/claim-record#trace-whitened-kmeans)) — a corollary of the loss
  identity, and explicitly not presented as a theorem of this project.
- Randomized rules reduce to deterministic ones when the score law has no atoms
  ([SOFT-HARD-ATOMLESS-EQUIVALENCE](/research/claim-record#soft-hard-atomless-equivalence)), by the
  Dvoretzky–Wald–Wolfowitz purification theorem.
- Optimal one-dimensional grouping to minimize information loss goes back to Cox and to Ogawa's
  work on spacings of order statistics.

## Where the literature stops

The survey identifies exactly one narrow place where a targeted search found no ready-made
treatment: the **full-matrix log-determinant objective for hard quantization of a multivariate
score, together with exact finite relocation algebra**. Everything on the next page lives inside
that gap or on its boundary.

First, the design literature optimizes over measures
and the clustering literature optimizes a different matrix functional, so the gap is a gap in the
*combination*, not in either ingredient. Second, the strongest specific claim in the gap — the
implication from exchange stability to a Voronoi structure — is recorded in the survey as still
awaiting a dedicated adversarial prior-art review. It is written as a finding, not as a priority
claim.

## Comparable software

No package matches this formulation end to end, and the useful comparison is by pipeline stage
rather than by feature list.

| Package | Stage | Objective | Relationship |
| --- | --- | --- | --- |
| MadMiner | Score estimation | Likelihood-ratio and score estimation | A supplier of scores, not a bin optimizer |
| INFERNO | Summary construction | Differentiable binned-likelihood uncertainty | The conceptual precedent for optimizing the downstream objective; a neural summary rather than a hard quantizer |
| ThickBrick | Categorization | Signal-discovery significance | The closest classical relative — Lloyd-like iteration on hard categories — with a different criterion |
| GATO / BOBR | Bin-boundary optimization | Binned-likelihood significance | The closest modern comparators for multidimensional bin-shape optimization |
| OptBinning | Supervised discretization | Mathematical programming against a target | Mature production binning; the objective is not Fisher information |
| scikit-learn `KBinsDiscretizer`, `KMeans` | Baseline | Uniform/quantile bins, Euclidean distortion | The natural baseline for the trace criterion after whitening, and a good initializer for the determinant one |

The practical difference is the level of abstraction: this library sits at the score-oracle
boundary rather than inside a domain-specific analysis workflow, so the same optimizer serves an
analytic likelihood, a linear component model, simulation-derived scores and classifier-derived
surrogates.

## Next

[What ScoreQuant adds](/research/what-scorequant-adds) states what was proved inside the gap this
page just drew.
