# SCORE-ERROR-RETENTION-BUDGET — independent prior-art triangulation

**Date:** 8 September 2026. **Scope:** O8 and its classifier/rule-transfer companions.
**Outcome:** `search_gap` for the specific reporting expansion; `prior_art_found`
for the margin comparison method and proper-score decomposition. No novelty claim.
This was a targeted five-source pass, not a citation-saturation survey.

## Nearest sources and transfer boundaries

| Source and exact location | Problem, objective, feasible set | What transfers | What does not |
|---|---|---|---|
| [Boyd & Vandenberghe, *Convex Optimization* (2004), §3.1.5, printed p. 74](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) | Concavity of log determinant on positive-definite matrices; arbitrary matrix lines. | Eigenvalue reduction and the first/second derivatives of log-det. | The score-error Gram bound and retention-ratio cancellation are not stated there. Our integral remainder is checked independently, not attributed as their theorem. |
| [Audibert & Tsybakov, *Fast learning rates for plug-in classifiers*, §5, Lemma 5.2 and its proof, manuscript pp. 19–20](https://imagine.enpc.fr/~audibert/Mes%20articles/plugin_v3.pdf) | Binary Bayes classification, posterior margin assumption, Lp regression error; excess risk and disagreement. | Split at a margin threshold, use Markov, then minimize. The p=2 split gives our error-norm exponent after replacing the margin by spatial boundary distance. | Their classification margin and excess-risk loss are not our arbitrary quantizer boundary and retained information. The displayed (5.4) in this preprint appears to omit alpha in the norm exponent; use the preceding minimization algebra, not that isolated display. |
| [Bröcker, *Reliability, Sufficiency, and the Decomposition of Proper Scores*, equations (13), (15), manuscript pp. 6–7](https://arxiv.org/pdf/0806.0813) | Proper scores for finite-valued forecast targets; reliability, resolution and sufficiency. | Multiclass Brier reliability/resolution decomposition and conditioning identities. | Neither a lower score-error bound through a noninjective chart nor a lower retention-distortion bound follows. |
| [Barnes, Han & Özgür (2018), §II, Lemmas 1–2, PDF pp. 2–3](https://web.stanford.edu/~aozgur/FisherAllerton.pdf) | Fisher information of finite-bit encodings; trace-information bounds under regularity. | Conditional-score identity and its trace interpretation. | A proxy-score log-determinant reporting-error bound is not supplied. |
| [Cranmer, Pavez & Louppe (2015), §2.2, Theorem 1 and §2.3](https://arxiv.org/pdf/1506.02169) | Likelihood-ratio testing via a discriminative summary and its induced densities. | Distinguishes a monotone ranking statistic from the calibrated ratio and supports classifier ratio construction. | Ranking preservation does not preserve raw proxy score moments; forecast reliability alone does not recover the full-observation posterior. |

## Search record

Queries covered log-determinant perturbation, Fisher quantization retention under score
error, plug-in classifier Lp margin comparisons, and multiclass Brier decomposition.
The explicit score-error/retention queries mostly returned unrelated neural-network
weight quantization; none supplied the frozen-label expansion. This is only a search gap.
Primary texts above were opened and the named sections inspected. The author-hosted
Audibert–Tsybakov manuscript has different pagination from the journal version; no
journal page number is inferred. Boyd's `web.stanford.edu` URL failed to open; the
`stanford.edu` author-hosted copy succeeded.

## Cite versus derive

Cite the five sources for their imported machinery. Keep O8's conditional moment
expansion, trace-loss cancellation, specific chart constants and boundary fixtures as
project reductions, without claiming novelty. Ipsen–Rehman determinant perturbations,
Stewart–Sun, and a primary page citation for canonical-correlation least squares remain
unread; none is needed as a proof dependency of the independently checked elementary
argument. Any later novelty claim must resolve those gaps first.
