# 9. Additional score-compression and ratio-estimation sources (v2 update)

> Curated theorem-level annotations. Machine records for the citation graph
> live in `graph.json`; `BIBLIOGRAPHY.md` (generated) maps every registry
> bibliography key to the heading that annotates it.

## Heavens, Jimenez & Lahav (2000) — MOPED

**Paper:** *Massive Lossless Data Compression and Multiple Parameter Estimation from Galaxy Spectra*  
**Result:** continuous linear compression to one summary per parameter preserving Fisher information under the paper's assumptions.  
**Use:** important ancestor for the “unbinned score/continuous compression is the information reference” viewpoint; not a finite hard quantizer.

- arXiv PDF: https://arxiv.org/pdf/astro-ph/9911102

## Alsing & Wandelt (2018) — generalized score compression

**Paper:** *Generalized Massive Optimal Data Compression*  
**Result:** likelihood-score compression gives locally Fisher-optimal continuous summaries under broad regularity conditions.  
**Use:** direct conceptual bridge from full observations to the score-space representation before finite quantization.

- arXiv PDF: https://arxiv.org/pdf/1712.00012

## Alsing & Wandelt (2019) — nuisance-hardened compression

**Key:** Alsing-Wandelt-2019

**Paper:** *Nuisance Hardened Data Compression for Fast Likelihood-Free Inference*
**Result:** projects the score into a locally/asymptotically
Fisher-preserving summary for parameters of interest using the nuisance
Schur geometry.
**Transfer:** closest published representation-level antecedent of the
full-data efficient score used by `DS-EFFICIENT-SCORE-DOMINATION`.
**Does not transfer:** the paper studies continuous summaries and does not
prove Loewner domination for every hard quantizer or the project’s global
\(D_s\) upper bound.

- DOI: https://doi.org/10.1093/mnras/stz1900
- arXiv: https://arxiv.org/abs/1903.01473

## Brehmer et al. — SALLY/SALLINO and MadMiner

**Use:** learned likelihood-score representations and practical score-space histograms in HEP. Establishes that learned scores and score histograms are prior art; ScoreQuant's question is how to optimize the hard cells under D/\(D_s\).

- Mining Gold PDF: https://arxiv.org/pdf/1805.12244
- MadMiner PDF: https://arxiv.org/pdf/1907.10621

## Wunsch et al. (2021)

**Paper:** *Optimal Statistical Inference in the Presence of Systematic Uncertainties Using Neural Network Optimization Based on Binned Poisson Likelihoods with Nuisance Parameters*  
**Use:** close HEP comparator for differentiable binned likelihood optimization with nuisances.

- arXiv PDF: https://arxiv.org/pdf/2003.07186

## Simpson & Heinrich (2022/23) — neos

**Paper:** *neos: End-to-End-Optimised Summary Statistics for High Energy Physics*  
**Use:** differentiable end-to-end expected-sensitivity optimization; adjacent software baseline, not score-Fisher hard partition theory.

- arXiv PDF: https://arxiv.org/pdf/2203.05570

## Density-ratio estimation

### Cranmer, Pavez & Louppe (2015)

**Key:** Cranmer-Pavez-Louppe-2015

*Approximating Likelihood Ratios with Calibrated Discriminative Classifiers*
establishes that a classifier trained to distinguish two samples learns a
monotone function of their likelihood ratio and that calibration can recover
the ratio. This directly supports `CLASSIFIER-RATIO-ORACLE`; it does not bound
the Fisher-information loss caused by calibration or approximation error.

- arXiv: https://arxiv.org/abs/1506.02169

Direct density-ratio estimation (KLIEP, uLSIF and related methods) is a mature alternative to separately estimating component densities. Classifier posterior odds are another route. For ScoreQuant these are **model-access backends**, not the quantizer itself.

Useful reference: Sugiyama, Suzuki & Kanamori, *Density Ratio Estimation in Machine Learning*.

### Research-agent instruction

When a theorem depends on a density-ratio/classifier assumption, search the ratio-estimation literature separately from the quantization literature. Do not infer exact Fisher preservation merely from classifier discrimination performance.

## Log-determinant concavity

**Key:** Boyd-Vandenberghe-2004

[Author-hosted text](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), §3.1.5,
printed p. 74. Supports the spectral log-det calculation in
`SCORE-ERROR-RETENTION-BUDGET`; it does not state that reporting theorem.

## Margin comparison inequalities

**Key:** Audibert-Tsybakov-2007

[Author manuscript](https://imagine.enpc.fr/~audibert/Mes%20articles/plugin_v3.pdf),
§5, Lemma 5.2/proof, manuscript pp. 19–20. Supports the split-and-optimize method
in `SCORE-ERROR-RULE-TRANSFER`; spatial margins and Fisher transfer require a reduction.

## Multiclass proper-score decomposition

**Key:** Brocker-2009

[Author manuscript](https://arxiv.org/pdf/0806.0813), equations (13) and (15).
Supports the Brier decomposition in `CLASSIFIER-CALIBRATION-SCORE-LIPSCHITZ`;
the chart constants and failure of an unconditional reverse implication are separate.
