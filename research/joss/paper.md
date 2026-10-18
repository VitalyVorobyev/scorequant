---
title: 'ScoreQuant: hard binning in score space that keeps Fisher information'
tags:
  - Python
  - JAX
  - Fisher information
  - binning
  - D-optimality
  - statistical inference
authors:
  - name: Vitaly Vorobyev
    # orcid: 0000-0000-0000-0000  # placeholder: add the author's ORCID before submission
    affiliation: 1
affiliations:
  - name: Affiliation to be confirmed by the author  # placeholder
    index: 1
date: 18 October 2026
bibliography: paper.bib
---

# Summary

Many analyses reduce each observation to one of a few category labels before fitting a model:
histograms in particle physics, gates in flow cytometry, quantized readouts of a detector. The
labels keep only part of the Fisher information the observations carried, and the choice of
categories decides how much. ScoreQuant is a Python library that chooses hard categories to keep
as much of that information as possible. It works on scores, the gradients of the log likelihood
at a reference parameter, because the information a label keeps depends only on how it groups
scores. Given a weighted table of score vectors, it maximizes the log determinant of the retained
information (the D criterion), or of the profiled information for parameters of interest when
the others are nuisance parameters. The result is either a labelling of a fixed sample or a
reusable rule that assigns new observations to categories.

# Statement of need

Three groups of users bin data before inference. Particle physicists build event categories for
template fits with nuisance parameters, for example in pyhf [@pyhf]. Cytometry and other
count-based analyses report the numbers of cells or events in a few gates. Machine-learning
practitioners compress classifier outputs or learned scores into a few bins so that a downstream
likelihood stays simple.

The usual tools ignore the parameters being measured. Equal-width and quantile histograms, as in
NumPy [@numpy], and $k$-means clustering, as in scikit-learn [@sklearn], partition the observed
features or a classifier output by geometry alone. Slicing a classifier output works when one
signal-versus-background direction is all a fit needs, and loses information once a nuisance
parameter can imitate the signal. Inference-aware methods such as INFERNO and neos train a
neural summary end to end against a profiled likelihood
[@decastro2019; @wunsch2021; @simpson2023]; they are powerful but give no structural guarantee
about the bins they find. Machine-learned score estimates are widely used in particle physics
[@brehmer2018prd; @brehmer2020], with the final bins usually chosen by hand.

ScoreQuant adds the following.

- **Exact D-optimal exchange.** Single-point moves are accepted only when their exact
  determinant gain is positive, so the objective rises monotonically and the run ends at a
  labelling no single move can improve.
- **A reusable rule from a finite partition.** Every such stable labelling is a
  nearest-centroid (Voronoi) partition in the metric given by its own inverse information, so it
  compiles into a rule that reproduces the training labels exactly
  [@vorobyev2026]. When the check fails at the solver's tolerance, the library refuses rather
  than guessing.
- **Nuisance parameters.** A profiled $D_s$ criterion [@karlin1966] targets the parameters of
  interest. For one parameter of interest, an exact interval dynamic program on the efficient
  score gives a certified ceiling on what any rule with the same number of bins can keep.
- **Honest labelling of estimated scores.** Scores derived from classifiers or density-ratio
  estimators produce a "supplied-score surrogate" information, never a claim of exact Fisher
  information, and the provenance is recorded with the result.
- **Two backends and saved artifacts.** JAX [@jax] is the default and NumPy the portable
  runtime. A fitted rule saves to a versioned archive with no pickled objects that loads and
  predicts without JAX.

# Software design

The library separates three steps. A source holds the reference measure: a table of scores,
a weighted sample of observations, or quadrature points. A provider maps observations to scores
from exact component densities, model density ratios or a score function, so the optimizers
see only weighted score rows. Scores are never centred, because the origin of score space means
"no sensitivity".

Two entry points cover the two tasks. `optimize_partition` labels a fixed sample and returns no
predictor, because many rules reproduce the same labels and disagree elsewhere.
`fit_quantizer` returns a `Quantizer`, a frozen rule applied with `predict_scores`. Criteria
(D, profiled $D_s$ and a whitened $k$-means baseline) and solvers (exact exchange, guarded
Mahalanobis–Lloyd, soft Voronoi with a reported hardening gap, and an exact dynamic program for
one-dimensional score spaces) form a closed table that is checked before any optimization
starts. Certificates are explicit calls: exchange stability, Voronoi geometry, a budgeted
branch-and-bound certificate of global optimality for small instances, and a confidence interval
for held-out retention. When a guarantee does not hold, the library raises an error that names
an exact counterexample kept in the test suite.

Backend-specific code lives in one private adapter, and one conformance suite runs both
backends. The package never changes global JAX configuration.

# Research impact statement

ScoreQuant is new software and has no external users yet. Its evidence so far is of two kinds.
First, four worked applications on the project website exercise the library end to end: phase
estimation in a Michelson interferometer with the fringe frequency as a nuisance; a signal
fraction estimated through a calibrated classifier, where an exact oracle measures how far the
surrogate retention is from the truth; composition estimates from the FlowCyt bone-marrow
benchmark [@flowcyt]; and a Higgs boson signal-strength fit on a small subset of the FAIR
Universe HiggsML dataset [@fairuniverse], where the tau energy scale floats. Each is a
demonstration on public or simulated data, not a published analysis.

Second, the central theorem that justifies compiling a partition into a rule is proved in Lean 4
[@lean4] with Mathlib [@mathlib], and the proof is checked on every change in continuous
integration. A companion paper [@vorobyev2026] states the theorem, its limits and exact
counterexamples, which the library uses to refuse unsupported requests. The machine-checked
statements cover the finite combinatorial result, not the software that implements it.

# AI usage disclosure

<!-- Placeholder wording: the author must confirm or revise this section before submission. -->

Substantial parts of the code, tests, documentation and this paper were developed with AI
coding assistants, Anthropic's Claude used through Claude Code and Codex-style coding agents,
working under the author's direction. The author designed the library, reviewed the changes and
is responsible for the content. Numerical claims are guarded by the test suite, and the central
theorem is checked by the Lean kernel rather than by review alone.

# Acknowledgements

<!-- Placeholder: the author to add acknowledgements and funding, or state that there is none. -->

To be completed by the author.

# References
