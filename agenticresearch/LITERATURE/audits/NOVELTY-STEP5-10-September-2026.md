# Novelty search, closure step 5 — 10 September 2026

Bounded adversarial novelty search per `protocols/literature.md` § "Novelty
search", run on three items: the two `search_gap` claims named for closure
step 5 (`SCORE-ERROR-RETENTION-BUDGET`, `RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE`)
and two unverified bibliography entries left over from the v9 audit
(Haynsworth 1968, Jakubowski 2021). Retrieval was WebSearch/WebFetch only; no
subscription access. Evidence labels follow the project convention:
**primary text** (statement actually read: PDF/HTML body, not just an
abstract), **primary record** (publisher/library metadata only), **abstract
only** (abstract or a search engine's synthesis of it, body not opened),
**secondary** (a citing/paraphrasing source, original not opened). Several
PDFs (Ipsen–Rehman, Boyd-adjacent binaries) would not parse through the
fetch tool available in this pass; where that happened the entry below is
marked **abstract only** or **not opened**, never claimed as primary text.

---

## 1. `SCORE-ERROR-RETENTION-BUDGET`

**Target statement.** A first-order expansion of
\(\log(\tilde\eta_D/\eta_D)\), the log-ratio of the reported to the true
geometric-mean retention of a *frozen* hard partition, under an additive
\(L^2\) score error \(e=\hat s-s\): an alignment term
\(T_1=(2/d)(E[e_Z^\top I_Z^{-1}c_Z]-E[e^\top V^{-1}s])\), a spurious-information
term \(T_2=(\varepsilon_R^2-\varepsilon^2)/d\), and an explicit curvature
remainder, built from a conditional-Jensen/Cauchy–Schwarz pair on the between-cell
Gram matrices and a Loewner sandwich on the log-det.

This continues the five-source pass already recorded in
`LITERATURE/audits/SCORE-ERROR-RETENTION-BUDGET-2026-09-08.md`, which named
three unread candidates (Ipsen–Rehman, Stewart–Sun, a canonical-correlation
least-squares primary page) as the open follow-up. This pass targets exactly
those three plus a fresh fan-out into errors-in-variables Fisher information
and quantized/compressed estimation with noisy sufficient statistics.

### Triangulation

| Source and exact location | Problem, objective, feasible set | What transfers | What does not |
|---|---|---|---|
| Ipsen & Rehman, *Perturbation Bounds for Determinants and Characteristic Polynomials*, SIAM J. Matrix Anal. Appl. 30(2):762–776 (2008), doi:10.1137/070704770 — **abstract only**; the author-hosted PDF (`ipsen.math.ncsu.edu/ps/simax070477.pdf`) would not parse through the fetch tool in this pass, so no theorem number or exact display was read. | Absolute and relative perturbation bounds for the coefficients of the characteristic polynomial of a general complex matrix \(A+E\), expressed as elementary symmetric functions of singular values; sharper bounds when \(A\) is normal or Hermitian positive-definite. Objective: worst-case (not mean-square/whitened) deviation of \(\det\) and the full spectrum. Feasible set: arbitrary square complex matrices. | The general machinery — determinant perturbation as a function of singular values of \(E\) — is the right family to *check* our Lemma 2–4 constants against. | Their bound is unconditional worst-case in the operator/singular-value norm, not conditional on a cell partition, not whitened by \(A^{-1/2}\), and carries no alignment/cross term split (no analogue of our \(T_1\)) — it bounds \(|{\det(A+E)}-\det A|\) directly, not a *ratio of two independently perturbed matrices* \(\log(\det\tilde I_Z/\det\tilde V)-\log(\det I_Z/\det V)\). Still unread beyond the abstract; no theorem can be cited as a proof dependency from this pass. |
| Stewart & Sun, *Matrix Perturbation Theory*, Academic Press, 1990 — **primary record only** (table of contents: Ch. 1 preliminaries/norms, Ch. 2 perturbation of linear systems and least squares, Ch. 3 the eigenvalue problem, Ch. 4 the generalized eigenvalue problem, Ch. 5 invariant subspaces). No chapter titled or indexed under "determinant" or "log-determinant" was found in any available table of contents. | Classical perturbation theory for eigenvalues, invariant subspaces, and the generalized eigenproblem, in unitarily-invariant norms. | The Weyl/Bauer–Fike-type eigenvalue perturbation bounds in Ch. 3–4 are the standard tool for bounding individual \(\rho_i\) shifts, which is adjacent to but not the same object as our log-det ratio. | No content on log-determinant ratios or partition-conditional (cell-averaged) perturbations was located; the book's determinant-relevant content, if any, was not identified even at the section level. This candidate stays effectively unread. |
| Björck & Golub, *Numerical methods for computing angles between linear subspaces*, Math. Comp. 27(123):579–594 (1973), doi:10.1090/S0025-5718-1973-0348991-3 — **abstract/record only**. | Principal angles/canonical correlations between two subspaces as a numerical linear-algebra object; for a single right-hand side reduces to ordinary least squares. | Establishes canonical correlations as principal-angle cosines, the correct lineage for the registry's uncentred canonical-correlation reading of \(\rho_i\). | Not the least-squares reconstruction identity the claim actually uses (\(\varepsilon_{\rm lin}^2=d-\sum_i r_i^2\) for the unconstrained least-squares proxy-to-truth map); no primary source for that specific identity was found in this pass — it remains an unread gap, same as in the prior audit. |
| Venkitasubramaniam, Tong & Swami, *Score-Function Quantization for Distributed Estimation* (CISS 2006), `lehigh.edu/~pav309/papers/VenkTongSwami_Quant_06CISS.pdf` — **abstract only**. | Scalar score-function quantizer design maximizing a Fisher-information lower bound for distributed estimation. | Conditional-score quantization framing (already the project's own terminology-island anchor, cf. the D-EXCHANGE-IMPLIES-VORONOI audit). | Scalar parameter, no additive score-error model, no reporting-bias/log-det expansion; already known to the project and not a new candidate. |
| *Information and Statistical Efficiency When Quantizing Noisy DC Values*, arXiv:1804.10402 — **abstract only**. | Fisher information of a scalar DC-value estimator under quantization of an already-noisy measurement. | Nearest existing hit for "quantization of a noisy statistic," i.e. the errors-in-variables-plus-quantization combination named in the task. | Scalar, additive-noise-then-quantize (not proxy-score-then-frozen-partition), no log-det/multivariate retention object, no alignment-term decomposition. |

### Search record

Eight queries: (1) `Ipsen Rehman "Perturbation bounds for determinants and characteristic polynomials"` — 8 candidates, 1 relevant (bibliographic confirmation only); (2) `first-order perturbation log determinant ratio rank-one matrix expansion trace` — 8 candidates, 1 tangentially relevant (a 2601.xxxxx log-det/trace-power estimator paper, unrelated objective); (3) `errors-in-variables Fisher information loss quantized sufficient statistic noisy score` — 7 candidates, 2 relevant (both scalar, listed above); (4) `"information loss" misspecified score estimation error Fisher information bound` — 7 candidates, 0 relevant (sandwich-covariance and score-matching hits, wrong problem); (5) `canonical correlation analysis measurement error attenuation Fisher information` — 10 candidates, 1 relevant (a 2512.22697 "canonical correlation regression with noisy data" preprint, screened by abstract only, not opened — Gaussian errors-in-variables CCA, closer in spirit than anything else found but not a log-det retention-expansion result); (6) `Stewart Sun "Matrix Perturbation Theory" determinant chapter log determinant bound` — 10 candidates, 1 relevant (table of contents only); (7) `least squares reconstruction error canonical correlations identity "d - sum" trace regression errors in variables` — 8 candidates, 0 relevant; (8) `Björck Golub 1973 "angles between linear subspaces" canonical correlations least squares distance` — 10 candidates, 1 relevant (lineage, not the identity). No query returned the specific two-term retention-ratio expansion, in this or any other formulation.

### Verdict

`search_gap`

No source located — in this pass or the 8 September pass it extends —
states a first-order expansion of a log-det retention ratio of a *fixed
hard partition* under score error, with an alignment term and a spurious-
information term. The three previously-named unread candidates remain
substantively unread (Ipsen–Rehman beyond its abstract, Stewart–Sun beyond
its table of contents, the canonical-correlation least-squares identity
with no primary source found at all); none of the new fan-out terms
(errors-in-variables Fisher information, quantized noisy sufficient
statistics, canonical correlation with measurement error) surfaced a closer
match. This reinforces, rather than resolves, the 8 September verdict.

### Cite versus derive

Nothing changes about what O8 should cite: Boyd–Vandenberghe for log-det
concavity, Audibert–Tsybakov for the margin/Markov method, Bröcker for the
proper-score decomposition, Barnes–Han–Özgür and Cranmer–Pavez–Louppe for
the information/classifier reductions — all already in the 8 September
audit. Ipsen–Rehman and Stewart–Sun stay named as background determinant-
perturbation literature but cannot be cited for any specific inequality
here, since neither was read past its abstract or table of contents in
either pass; if either is ever needed as a proof dependency, the primary
text must be obtained and read first (both are paywalled/binary-blocked
through the tools available in these two sessions). The core two-term
expansion, its Cauchy–Schwarz alignment bound and its Loewner-sandwich
curvature remainder stay project reductions, with the literature status
unchanged at `search_gap`.

---

## 2. `RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE`

**Target statement.** At the singular endpoint (\(I_Z\) of rank \(r<d\), so
\(\eta_D=0\)), the plug-in geometric-mean retention \(\hat\eta_D\) is
\(O_p(n^{-(d-r)/d})\), and under a nondegeneracy condition
\(n^{(d-r)/d}\hat\eta_D\) converges in law to an explicit positive random
variable built from a Schur complement of a Gaussian limit matrix (the
`W = G(I-Lambda)` construction). This is exactly the fixed-alternative,
fixed-dimension "dimensionality" question in canonical correlation
analysis: \(\hat\eta_D\) is the geometric mean of squared *uncentred* sample
canonical correlations between the true score \(S\) and the cell indicator
\(\mathbf 1_Z\), and \(r<d\) means \(d-r\) of the population canonical
correlations are exactly zero.

### Triangulation

| Source and exact location | Problem, objective, feasible set | What transfers | What does not |
|---|---|---|---|
| Hsu, *On the limiting distribution of the canonical correlations*, Biometrika 32(1):38–45 (1941) — **secondary** (not opened directly; Biometrika paywalled in this pass). Per a citing modern source: "the differences between sample and population canonical correlations are of order \(S^{-1/2}\) and their rescaled limit depends on ... whether they are equal to zero", and Hsu "derived the asymptotic joint distribution of the sample canonical correlations when the population canonical correlations are zero." | Fixed-dimension, fixed-population CCA (same regime as the claim: \(d\), \(K\) fixed, \(n\to\infty\)); the founding result for what happens to *individual* sample canonical correlations at a zero population value. | If accurately paraphrased, this is the classical antecedent of the *rate* half of the claim: each null-direction sample canonical correlation is \(O_p(n^{-1/2})\), which by squaring and multiplying over the \(d-r\) null directions forces \(\det\hat I_Z=O_p(n^{-(d-r)})\) and hence the claimed \(n^{-(d-r)/d}\) rate for the \(d\)-th-root geometric mean — the same algebra the registry proof already performs internally, now with a named classical source for the per-coordinate input rate. | Not read directly, so its exact joint-limit-law statement (form, moment structure) cannot be compared term-by-term with the registry's Schur-complement/\(\Lambda\)-projection construction; and Hsu's object is the *vector* of individual squared correlations, not the *determinant/geometric-mean* functional the registry claim needs — going from "each coordinate is \(O_p(n^{-1/2})\) with known joint law" to "the determinant's \(d\)-th root has this explicit limit" is exactly the algebraic step the registry proof supplies and that was not independently verified as already published. |
| Fujikoshi (1974), cited via secondary summaries (original not located/opened) — proposes a test statistic \(L_k\) for "the last \(p_1-k\), \(p_2-k\) population canonical correlations are zero" with asymptotic null distribution \(\chi^2_{(p_1-k)(p_2-k)}\). | Classical dimensionality/rank test in CCA — same null hypothesis structure as \(\operatorname{rank} I_Z=r<d\). | Confirms that the *sum*-type functional (a log-likelihood-ratio statistic, essentially \(-n\sum_i\log(1-r_i^2)\) over the null block) has a known \(n\)-scaled chi-squared limit at exactly this boundary — reinforcing that the boundary asymptotics of null canonical correlations is settled classical territory. | A sum/LRT-type statistic with a chi-squared limit is not the same object as the registry's *product* (determinant-ratio) statistic with its specific Schur-complement limit law; the two are related (\(-n\log(1-r^2)\approx nr^2\) for small \(r\)) but the registry needs the *joint* limit of the product, not just that each term or their sum has a limit. Not opened directly. |
| Glynn & Muirhead, *Inference in canonical correlation analysis*, J. Multivariate Anal. 8:468–478 (1978) — **secondary** (not opened; per a citing source, they give "a bias correction, so that the statistic has an asymptotic \(\chi^2\) distribution with \((p-k)(q-k)\) degrees of freedom"). | Bias-corrected LRT test for the same rank-deficiency null hypothesis. | Same family as Fujikoshi (1974); reinforces that this is a well-covered classical corner. | Same gap as above — a corrected chi-squared LRT statistic, not the product/geometric-mean determinant-ratio limit law with explicit Schur-complement structure that the registry states. |
| Anderson, *Asymptotic theory for canonical correlation analysis*, J. Multivariate Anal. 70(1):1–29 (1999) — **record only**; abstract could not be retrieved (ScienceDirect returned HTTP 403, and no open-access mirror was found in this pass). | Presumed general asymptotic theory (title only) for CCA, likely including the boundary/zero case as a special case, given the author and venue. | Unknown — could not confirm content. | Named as an open gap, not a found result; nothing here can be cited. |
| Seo, Kanda & Fujikoshi, *The effects of nonnormality on tests for dimensionality in canonical correlation and MANOVA models*, J. Multivariate Anal. 52(2):325–337 (1995) — already named in the registry's own assumptions as unread (`LITERATURE/gaps.md`); still unread after this pass (no accessible full text or detailed abstract found beyond the record-level description already on file). | Nonnormal-sampling correction to the classical dimensionality test. | Same as previously recorded: the closest classical statement of the endpoint problem. | Remains unread; this pass did not close that gap. |
| Muirhead, *Aspects of Multivariate Statistical Theory*, Wiley (1982/2005), Ch. 11 "Testing Independence Between \(k\) Sets of Variables and Canonical Correlation Analysis" — **record only** (table of contents/chapter title confirmed; chapter body not opened). | Textbook treatment of CCA asymptotics including, per the chapter title, the independence-testing/dimensionality question. | Presumed to contain the Hsu/Fujikoshi-type boundary results in a citable textbook form (standard for this literature), but not confirmed in this pass. | Content unread; cannot be used as a citation source without opening it. |
| *Nonparametric inference on non-negative dissimilarity measures at the boundary of the parameter space*, arXiv:2306.07492 — **not opened** (PDF would not parse through the fetch tool). | Title suggests exactly the right general shape (plug-in inference for a non-negative functional pinned at its zero boundary) but the specific estimand, rate, and whether it covers a log-det/geometric-mean statistic could not be confirmed. | Unknown — flagged as a promising lead for a future pass with better PDF access, not usable now. | Cannot be compared without reading it; not claimed as prior art. |

### Search record

Ten queries: (1) `degenerate delta method boundary asymptotics plug-in estimator rate n^{-1/2} slower` — 10 candidates, 2 relevant (the boundary-dissimilarity-measures preprint above; one tangential treatment-effect paper); (2) `Hsu 1941 canonical correlations zero population asymptotic distribution` — 9 candidates, 1 relevant (Hsu 1941, secondary paraphrase only); (3) `asymptotic distribution smallest eigenvalues sample covariance matrix when population eigenvalue is zero rank deficient Wishart` — 8 candidates, 0 relevant (all high-dimensional \(p/n\to\gamma\) random-matrix-theory results, wrong regime — our \(d,K\) are fixed); (4) `"sample generalized variance" singular population covariance matrix asymptotic distribution rate` — 10 candidates, 1 relevant (a reference to Anderson (1984) Thm 7.5.4 for the *nonsingular* generalized-variance limit, not the singular case); (5) `Hsu 1941 Biometrika "limiting distribution of the canonical correlations" theorem statement chi-squared` — 9 candidates, 1 relevant (bibliographic confirmation only, full text paywalled); (6) `Anderson dimensionality test canonical correlation "smallest" eigenvalues zero population asymptotic distribution rate n` — 10 candidates, 2 relevant (Anderson 1999 record; an elliptical-population chi-squared-approximation test); (7) `Fujikoshi dimensionality canonical correlation analysis test statistic asymptotic null distribution rank` — 10 candidates, 3 relevant (Fujikoshi 1974 \(L_k\); Fujikoshi 1975 \(O(n^{-2})\) expansion; Fujikoshi & Veitch 1979 "Estimation of Dimensionality in Canonical Correlation Analysis"); (8) `Glynn Muirhead 1978 "Inference in canonical correlation analysis" eigenvalues zero` — 9 candidates, 1 relevant; (9) `"product of squared canonical correlations" asymptotic distribution rank deficient zero eigenvalue determinant Wilks` — 8 candidates, 0 relevant (no source addresses the product/determinant form directly, only the sum/LRT form); (10) `Muirhead "Aspects of Multivariate Statistical Theory" canonical correlations zero population asymptotic Wishart limit smallest eigenvalues` — 7 candidates, 1 relevant (chapter title confirmed, content unread). Saturation was not reached — several leads (Anderson 1999, Seo–Kanda–Fujikoshi 1995, Fujikoshi & Veitch 1979, Muirhead Ch. 11, the boundary-dissimilarity preprint) remain unopened primary texts, all behind either a paywall or a PDF this pass's fetch tool could not parse.

### Verdict

`prior_art_found: Hsu (1941, Biometrika 32:38–45), and the dimensionality-test line it founded (Fujikoshi 1974; Glynn & Muirhead 1978, J. Multivariate Anal. 8:468–478), state — per secondary paraphrase, not primary text read in this pass — that sample canonical correlations at a zero population canonical correlation are O_p(n^{-1/2}) with a known joint/marginal limit law, which is the classical antecedent that forces the same n^{-(d-r)/d} rate this registry claim derives for the d-th-root geometric-mean determinant ratio, stated there in the sum/chi-squared-LRT formulation rather than the product/geometric-mean formulation used here.`

This is deliberately not softened to `search_gap`: per the adversarial
instruction, a rate-equivalent result "in another formulation" (the
classical dimensionality test) counts as prior art for the *rate* even
though none of it was read past a secondary paraphrase, and even though the
registry's specific *product*-statistic limit law (the explicit
Schur-complement/\(\Lambda\) random variable) was not located anywhere in
this form. The registry claim's own text already flags Seo–Kanda–Fujikoshi
(1995) as the closest classical setting and marks it unread; this pass adds
Hsu (1941) as the more fundamental, earlier antecedent of the rate itself,
and Fujikoshi (1974)/Glynn & Muirhead (1978) as the specific chi-squared
sum-statistic form — none of which were previously named in the claim's
`literature` field or in `LITERATURE/gaps.md`.

### Cite versus derive

Before any publication-facing novelty claim for this theorem, the project
should: (1) obtain and read Hsu (1941) primary text to confirm the exact
rate and limit-law statement quoted above only secondhand; (2) read
Fujikoshi (1974) and Glynn & Muirhead (1978) to confirm the sum/LRT
chi-squared claim and check whether either paper's proof, in passing,
already produces a determinant/product-type intermediate quantity that
would collapse the registry's `search_gap` entirely; (3) close the
already-flagged Seo–Kanda–Fujikoshi (1995) gap. Until then, the registry's
`literature_search_status: search_gap` should be *narrowed*, not removed:
the **rate** \(n^{-(d-r)/d}\) is very likely citable to Hsu/Fujikoshi/Glynn–
Muirhead once read (this pass's honest confidence is "probable prior art,
unread primary text"), while the **explicit limit-law construction** (the
Schur complement of \(G(I-\Lambda)\)) has no located antecedent in any
formulation and should keep standing as a project contribution. The
registry's `literature` field (`Robin-Smith-2000`, `Seo-Kanda-Fujikoshi-1995`,
`vanderVaart-1998`) should add `Hsu-1941`, `Fujikoshi-1974`, and
`Glynn-Muirhead-1978` as newly identified candidates once their primary
texts are confirmed, rather than being treated as settled by this pass's
secondary-only evidence.

---

## 3. Bibliography verification

| Item | Verified citation | Verified from |
|---|---|---|
| (a) | E. V. Haynsworth, "Determination of the inertia of a partitioned Hermitian matrix," *Linear Algebra and its Applications*, **1**(1), 73–81, **1968**. DOI: 10.1016/0024-3795(68)90050-5. | ScienceDirect journal listing for *Linear Algebra and its Applications* Vol. 1, Issue 1 (January 1968); independently cross-checked table of contents at `ftp.math.utah.edu/pub/tex/bib/toc/linala1960.html` (title, issue "Volume 1, Number 1", pages "73–81", year "January 1968"); DOI confirmed via web search resolving to `doi.org/10.1016/0024-3795(68)90050-5` and the ScienceDirect article page `sciencedirect.com/science/article/pii/0024379568900505`. |
| (b) | Adam Jakubowski, "A complement to the Chebyshev integral inequality," *Statistics & Probability Letters*, **168**, article 108934, **2021**. DOI: 10.1016/j.spl.2020.108934. Preprint: arXiv:2005.00788. | RePEc/IDEAS record `ideas.repec.org/a/eee/stapro/v168y2021ics0167715220302376.html` (author "Adam Jakubowski", volume 168, year 2021, DOI 10.1016/j.spl.2020.108934); ScienceDirect abstract page `sciencedirect.com/science/article/abs/pii/S0167715220302376`; arXiv abstract page `arxiv.org/abs/2005.00788` (author and exact title confirmed, submitted 2 May 2020) — arXiv itself does not display the journal reference, so the article number 108934 and volume/year are taken from the IDEAS/ScienceDirect records, which agree with each other and with the DOI suffix. |

Both entries in the v9 manuscript (ref. [55]/[56] "E. V. Haynsworth. Determination
of the inertia of a partitioned Hermitian matrix. Linear Algebra and its
Applications 1(1), 73–81, 1968.") and `LITERATURE/BIBLIOGRAPHY.md`
(`Jakubowski-2021`, "A complement to the Chebyshev integral inequality") are
confirmed correct as printed; no correction is needed to either citation's
author, title, journal, volume, pages/article number, or year. The
manuscript's Haynsworth citation does not currently carry a DOI; adding
`doi:10.1016/0024-3795(68)90050-5` would be a safe, verified addition if the
project wants DOIs on every reference, but is not required for correctness.

---

## Summary for the closing report

- `SCORE-ERROR-RETENTION-BUDGET`: **search_gap**, reinforced — no source
  found states the specific two-term retention-ratio expansion, in any
  formulation. Ipsen–Rehman and Stewart–Sun remain unread beyond
  abstract/table-of-contents; a canonical-correlation least-squares primary
  source for the \(\varepsilon_{\rm lin}^2=d-\sum r_i^2\) identity was still
  not found.
- `RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE`: **prior_art_found** (rate only,
  secondary-source confidence) — Hsu (1941) and the classical CCA
  dimensionality-test line (Fujikoshi 1974; Glynn & Muirhead 1978) give the
  same \(n^{-(d-r)/d}\)-type boundary rate in the sum/chi-squared
  formulation; none states the registry's specific product/geometric-mean
  determinant-ratio limit law (the Schur-complement/\(\Lambda\)
  construction), which stays a project contribution pending confirmation
  that Anderson (1999), Seo–Kanda–Fujikoshi (1995), or Muirhead's textbook
  chapter (all still unread) do not already contain it.
- Bibliography: both (a) Haynsworth 1968 (*Linear Algebra and its
  Applications* 1(1):73–81, DOI 10.1016/0024-3795(68)90050-5) and (b)
  Jakubowski 2021 (*Statistics & Probability Letters* 168:108934, DOI
  10.1016/j.spl.2020.108934) are verified correct as currently cited in the
  manuscript/bibliography.
- Candidate the manuscript should consider citing once read: none rises
  above "read the primary text first" for either target claim in this
  pass — the strongest lead (Hsu 1941 for the endpoint rate) is
  recommended above as the next read, not as a citation to insert now.
