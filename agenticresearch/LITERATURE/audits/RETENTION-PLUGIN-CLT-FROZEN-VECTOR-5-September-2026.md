# Prior-art pass before the vector retention CLT (RETENTION-PLUGIN-VECTOR)

**Key:** Radhakrishnan-Kshirsagar-1981, Romanazzi-1992, Muirhead-Waternaux-1980, Fang-Krishnaiah-1982, Seo-Kanda-Fujikoshi-1995, Sugiura-Fujikoshi-1969, Magnus-Neudecker-1999, Anderson-2003, Cramer-Nicewander-1979, Kendall-Stuart-1961

Run 5 September 2026, **before** the derivation, at the owner's request
("make a better literature search before spending time and compute in
proving something well known"). Mode: novelty search of
`protocols/literature.md` for the statement the packet
`WORK/active/RETENTION-PLUGIN-VECTOR.md` intends to prove (future claim id
`RETENTION-PLUGIN-CLT-FROZEN-VECTOR`), plus a one-hop forward-citation round
on the two nearest sources. Retrieval was done inline (OpenAlex, Crossref,
Google Books search-inside, archive.org OCR, arXiv); no subagent. Labels:
**primary text** (statement read: scan, OCR, abstract or PDF), **primary
record** (publisher/library table of contents or journal record),
**secondary** (citing source only). The two O6 items still open in
`gaps.md` were retried in the same pass; one is resolved below.

## The statement in classical terms

For a frozen rule \(Z=q(\hat s)\) and a \(d\)-dimensional oracle score \(S\)
with \(V=E[SS^\top]\), \(m_b=E[S\mathbf 1_{Z=b}]\), \(p_b=P(Z=b)\), the packet
targets

\[
\eta_D=\Big(\frac{\det I_Z}{\det V}\Big)^{1/d},\qquad
I_Z=\sum_b\frac{m_bm_b^\top}{p_b}=MP^{-1}M^\top,
\]

with \(M=E[S\,\mathbf 1_Z^\top]\in\mathbb R^{d\times K}\) and
\(P=\operatorname{diag}(p)=E[\mathbf 1_Z\mathbf 1_Z^\top]\). Hence
\(\det I_Z/\det V=\det(V^{-1}MP^{-1}M^\top)=\prod_{i=1}^d\rho_i^2\), where
\(\rho_i^2\) are the squared **uncentred canonical correlations** between
\(S\) and the one-hot vector \(\mathbf 1_Z\): \(\eta_D\) is their geometric
mean. At the evaluation law \(P_{\theta_0}\) (\(E[S]=0\), the retention
reading of O6.5) the uncentred and centred objects coincide, because
\(M\mathbf 1=E[S]=0\) makes \(MP^{-1}M^\top=M(P-pp^\top)^{+}M^\top\); then
\(I_Z\) is the between-group covariance of the cell means, \(V\) the total
covariance, and \(\Lambda=\det(V-I_Z)/\det V=\prod_i(1-\rho_i^2)\) is Wilks'
criterion for the grouping \(Z\). So, in classical language, O7 is **the
fixed-alternative (non-null) first-order asymptotics of a MANOVA
determinant ratio, under finite fourth moments and no normality, with the
grouping a fixed measurable map of a different variable**, reported as the
geometric mean of the squared canonical correlations rather than as
\(\Lambda\). The singular-\(I_Z\) endpoint (\(\eta_D=0\)) is the classical
"dimensionality" question (fewer than \(d\) non-zero canonical
correlations).

## What O7 will consume, and where it is already written down

1. **The vector delta method** for a smooth function of the sample mean of
   \(T=(\mathbf 1_{Z=b},S\mathbf 1_{Z=b},\operatorname{vech}SS^\top)_b\):
   van der Vaart Thm 3.1 (`vanderVaart-1998`, primary text, O6 audit).
2. **The differential of a determinant and of an inverse:**
   Magnus & Neudecker, Part Three, Chapter 8 "Some important
   differentials", §8.3 "The differential of a determinant" (p. 149:
   \(d|X|=|X|\operatorname{tr}X^{-1}dX\)), §8.4 "The differential of an
   inverse" (p. 151); Chapter 9 §9.10 "Scalar functions of a matrix, II:
   determinant" (p. 178) — *primary record* (contents scan,
   `papers/local/`). The conjectured
   \(\psi=(\eta_D/d)[2S^\top I_Z^{-1}c_Z-c_Z^\top I_Z^{-1}c_Z-S^\top V^{-1}S]\)
   is the chain rule through these two differentials; deriving it is a
   check, not a contribution.
3. **Influence functions of determinant-type multivariate parameters:**
   Radhakrishnan & Kshirsagar (1981) — *primary text* (abstract): influence
   functions "for the generalized variance, the matrix of regression
   coefficients, the noncentrality matrix \(\Sigma^{-1}\delta\) in
   multivariate analysis of variance and its eigenvalues, ... canonical
   correlations, principal components and parameters that correspond to
   Pillai's statistic, Hotelling's generalized \(T_0^2\) and Wilks' \(\Lambda\)",
   with \(r^2\), \(R^2\) and Mahalanobis \(D^2\) as special cases. This is
   the matrix influence function of O6.8 in its MANOVA form. Romanazzi
   (1992) — *primary text* (abstract): the influence function of each
   squared canonical correlation via perturbation of the generalized
   eigenproblem, with the squared multiple correlation as a special case.
4. **Asymptotic normality under finite fourth moments, no normality:**
   Muirhead & Waternaux (1980) — *primary text* (abstract): asymptotic
   distributions of sample canonical correlations and of the associated
   test statistics for populations with finite fourth moments, expressed
   through fourth-order cumulants, simple under ellipticity. Fang &
   Krishnaiah (1982) — *primary record* (JMVA 12(1):39–63): asymptotic
   distributions of functions of the eigenvalues of the standard random
   matrices for nonnormal populations. Seo, Kanda & Fujikoshi (1994,
   1995) — *primary text* (abstracts): expansions for functions of the
   canonical correlations under nonnormality, and the effect of
   nonnormality on the dimensionality tests in canonical-correlation and
   MANOVA models (the singular-\(I_Z\) endpoint).
5. **Non-null asymptotics of the likelihood-ratio (determinant) criterion:**
   Sugiura & Fujikoshi (1969) — *primary text* (abstract): asymptotic
   non-null expansion of the LR criterion for the multivariate linear
   hypothesis to order \(N^{-2}\), normal theory; Fujikoshi (2002) — *primary
   record* (JSPI 108:263–282): multivariate basic statistics and one-way
   MANOVA tests under nonnormality. Anderson, 2nd ed. contents — *primary
   record* (`papers/local/`): §7.5 "The generalized variance", §8.3–8.5
   (likelihood-ratio criterion, its distribution and asymptotic expansion),
   §8.9 "Multivariate analysis of variance", §12.2–12.4 (canonical
   correlations, estimation, inference), §13.5–13.6 (asymptotic
   distributions of roots, one and two Wishart matrices). The 3rd-edition
   theorem numbering for the non-normal asymptotics of the sample
   covariance is **unverified** (`gaps.md`).
6. **Lineage of the measure:** Hotelling (1936) *Relations between two sets
   of variates* (vector correlation and alienation coefficients as
   determinant ratios), Wilks (1932) *Certain generalizations in the
   analysis of variance* (\(\Lambda\)); Cramer & Nicewander (1979) — *primary
   text* (abstract): the symmetric, linearly invariant measures of
   multivariate association are all functions of the canonical correlations
   and are strictly ordered; van den Burg & Lewis (1988) compare a
   \(\Lambda\)-based symmetric index with a Pillai-trace one. Whether the
   geometric mean \((\prod\rho_i^2)^{1/d}\) itself is in Cramer &
   Nicewander's list could not be read (paywalled; `gaps.md`).

## Triangulation (six fields per source)

- **Radhakrishnan & Kshirsagar (1981), *Influence functions for certain
  parameters in multivariate analysis*, Comm. Statist. Theory Methods
  10(6):515–529 (`Radhakrishnan-Kshirsagar-1981`; primary text, abstract;
  62 citers).** **Exact problem:** Hampel influence functions of
  multivariate parametric functions for outlier detection. **Exact
  result:** closed-form influence functions for the generalized variance,
  the MANOVA noncentrality matrix and its eigenvalues, canonical
  correlations, and the parameters behind Pillai, Hotelling \(T_0^2\) and
  Wilks \(\Lambda\). **Objective:** none (diagnostics). **Feasible set:**
  population parameters of a multivariate model with a grouping.
  **Transfers:** the influence function of \(\log\det\) of a between/total
  ratio is here in the centred, several-populations form; at the
  evaluation law O7's \(I_Z\) *is* the between-group matrix, so the packet's
  conjectured \(\psi\) must reduce to their \(\Lambda\)/noncentrality-matrix
  influence function after the \(1/d\)-th power. **Does not:** no
  asymptotic-normality statement, no variance estimator, no fixed
  partition of a *different* variable, no uncentred form, no \(0/0:=0\)
  convention, no endpoint analysis. Full text not read (paywalled).
- **Muirhead & Waternaux (1980), *Asymptotic distributions in canonical
  correlation analysis and other multivariate procedures for nonnormal
  populations*, Biometrika 67(1):31–43 (`Muirhead-Waternaux-1980`; primary
  text, abstract; 165 citers).** **Exact problem:** limit laws of sample
  canonical correlations and their test statistics without normality.
  **Exact result:** asymptotic normality under finite fourth moments, with
  a covariance built from fourth-order cumulants; elliptical
  simplification; a corrected chi-squared test for zero coefficients.
  **Objective:** none. **Feasible set:** two continuous blocks with finite
  fourth moments and distinct population coefficients. **Transfers:** the
  fourth-moment hypothesis (A2) and the mechanism (delta method on the
  sample second-moment matrix) are exactly O7's; \(\eta_D\) is a smooth
  symmetric function of their \(\rho_i^2\), so their result *implies* the
  vector CLT off the endpoints by one more delta step. **Does not:** one
  block here is the bounded indicator \(\mathbf 1_Z\) (so the covariance
  simplifies to cell moments), the moments are uncentred, the function is
  the geometric mean (well-defined even with repeated \(\rho_i\), which
  their per-coefficient results exclude), and no plug-in variance or Wald
  coverage is stated.
- **Fang & Krishnaiah (1982), *Asymptotic distributions of functions of
  the eigenvalues of some random matrices for nonnormal populations*,
  J. Multivariate Anal. 12(1):39–63 (`Fang-Krishnaiah-1982`; primary
  record; 47 citers).** **Exact problem:** limit laws of smooth functions
  of the eigenvalues of covariance, MANOVA-type and canonical-correlation
  random matrices under nonnormal sampling. **Exact result:** (from record
  and citing sources) asymptotic normality with fourth-cumulant covariance
  for such functions; the determinant is the elementary symmetric function
  of the eigenvalues. **Objective/feasible set:** as above. **Transfers:**
  the closest general theorem to "\(\sqrt n(\hat\eta_D-\eta_D)\) is normal";
  a citation candidate once the text is read. **Does not:** text unread;
  fixed partition, uncentred moments and the endpoint treatment are not
  expected there.
- **Romanazzi (1992), *Influence in canonical correlation analysis*,
  Psychometrika 57(2):237–259 (`Romanazzi-1992`; primary text, abstract;
  45 citers).** **Exact problem:** influence functions of each \(\rho_i^2\)
  and canonical vector pair. **Exact result:** perturbation theory of the
  generalized eigenproblem gives them in closed form; three sample
  versions. **Transfers:** \(\log\eta_D=d^{-1}\sum_i\log\rho_i^2\), so the
  influence function of \(\eta_D\) is the average of theirs divided by
  \(\rho_i^2\); a second route to the same \(\psi\) as the determinant
  differential. **Does not:** requires simple \(\rho_i\) per coefficient (the
  determinant route does not), no CLT, no variance estimator.
- **Sugiura & Fujikoshi (1969), *Asymptotic expansions of the non-null
  distributions of the likelihood ratio criteria for multivariate linear
  hypothesis and independence*, Ann. Math. Statist. 40(3):942–952
  (`Sugiura-Fujikoshi-1969`; primary text, abstract).** **Exact problem:**
  non-null distribution of \(-\log\Lambda\)-type criteria. **Exact result:**
  expansion to order \(N^{-2}\) under a fixed alternative, without rank
  assumptions on the noncentrality matrix (normal theory). **Transfers:**
  the fixed-alternative asymptotic normality of a determinant ratio and
  the second-order terms that O6.7 measured for \(d=1\). **Does not:**
  Gaussian sampling; O7 needs the fourth-moment version (Fujikoshi 2002,
  `gaps.md`).
- **Seo, Kanda & Fujikoshi (1995), *The effects of nonnormality on tests
  for dimensionality in canonical correlation and MANOVA models*,
  J. Multivariate Anal. 52(2):325–337 (`Seo-Kanda-Fujikoshi-1995`; primary
  record; 38 citers).** **Transfers:** the singular-\(I_Z\) endpoint of O7
  is the dimensionality hypothesis; their nonnormal null theory says what
  the plug-in does when \(\eta_D=0\) exactly (rate \(1/n\), not
  \(1/\sqrt n\)). **Does not:** null theory only; no interval.
- **Kendall & Stuart, *The Advanced Theory of Statistics*, Vol. 2 (1961)
  (`Kendall-Stuart-1961`; primary text, archive.org OCR).** Resolves the
  O6 gap. Chapter 26 §26.21–26.22 define the population correlation
  ratio \(\eta_1^2=\operatorname{var}\{E(x\mid y)\}/\sigma_1^2\) (26.40),
  (26.47) and the sample version (26.48) with \(0\le r^2\le e^2\le1\)
  (26.49); §26.23–26.24 give the decomposition
  \(ns_1^2=nr^2s_1^2+ns_1^2(e^2-r^2)+ns_1^2(1-e^2)\) (26.50)–(26.51) — the
  centred analogue of O6.1's RSS identity — and the normal-theory \(F\)
  tests (26.52)–(26.55) attributed to Fisher, via Cochran's theorem with
  the \(y\)-arrays fixed. **No large-sample variance of the correlation
  ratio and no moment-based non-normal form appear in Chapter 26**; the
  chapter's exercises and Chapter 27's multiple-correlation sections
  (27.29, 27.31) treat the large-\(n\) agreement of fixed-array and
  bivariate-normal theory only. Vol. 1 Chapter 10 was not available in
  this OCR. **Transfers:** lineage and the decomposition identity.
  **Does not:** nothing on variance.
- **Screened, adjacent, not transferring:** Ogasawara (2007), JMVA
  98(9):1726–1750 (Edgeworth expansions of CCA estimators under
  nonnormality, including studentized forms and Rozeboom's between-set
  correlation — the second-order backdrop for O7's coverage table;
  secondary summary); Cléroux & Ducharme (1989), Comm. Statist. 18(4):
  1441–1454 (vector correlation under elliptical laws; asymptotic
  distribution; abstract); Cai, Liang & Zhou (2015), arXiv:1309.0482
  (Corollary 1: fixed-\(p\) CLT for \(\log\det\hat\Sigma\), Gaussian sampling;
  PDF held in `papers/`); van den Burg & Lewis (1988), Psychometrika
  53(1):109–122; Hotelling (1936); Wilks (1932). Rejected: Liu, Bathke &
  Harrar (2011) (rank-based nonparametric \(\Lambda\)); Jolliffe & Lukudu
  (1993) (one-sample tests); Kshirsagar & Gupta (1990) (factors of
  \(\Lambda\), normal theory); Josse & Holmes (2016) (survey of association
  measures, no asymptotics).

## Forward-citation round (one hop, title screen)

| Seed | Citers (OpenAlex) | Screened | Relevant |
|---|---|---|---|
| Radhakrishnan & Kshirsagar (1981) | 62 | 62 | Romanazzi 1992; Krzanowski-type "influence in canonical variates analysis" (1991); MANOVA influence diagnostics (1990, two parts); Kshirsagar & Gupta 1990 |
| Muirhead & Waternaux (1980) | 165 | 100 (most cited) | Fang & Krishnaiah 1982; Seo–Kanda–Fujikoshi 1994, 1995; Cléroux & Ducharme 1989; Ogasawara 2007; Yanagihara 2005 |
| Romanazzi (1992) | 45 | 45 | none with a determinant ratio (robust-CCA line) |

Saturation was not claimed: the second hop (citers of Fang & Krishnaiah
and of Seo–Kanda–Fujikoshi) is the next round if O7 is ever submitted as a
technique claim.

## Search verdict and consequence for the packet

- **Method: prior art found.** The vector CLT off the endpoints is the
  vector delta method (van der Vaart Thm 3.1) composed with the
  determinant and inverse differentials (Magnus & Neudecker §8.3–8.4); its
  influence function is a special case of Radhakrishnan & Kshirsagar's
  \(\Lambda\)/noncentrality-matrix influence functions and of Romanazzi's
  per-coefficient ones; its fourth-moment asymptotic normality is implied by
  Muirhead & Waternaux (1980) and, in the general eigenvalue-function form,
  by Fang & Krishnaiah (1982). None of this should be re-proved; it should
  be cited and *reduced* to the cell moments.
- **Exact statement: search gap** (never novelty): no located source states
  the geometric-mean form on a fixed partition of a different variable with
  uncentred moments and \(0/0:=0\), gives the plug-in variance
  \(\hat\sigma^2=n^{-1}\sum\hat\psi_i^2\) with its consistency, or treats the
  endpoints \(\eta_D\in\{0,1\}\) and the \(\sigma^2=0\) set. Those are the
  parts of `RETENTION-PLUGIN-VECTOR.md` worth a session; the packet was
  reordered accordingly (attack plan there).
- **O6 gaps:** Kendall & Stuart resolved (above; no variance formula
  exists there). Serfling §3.3 Theorem A/B labels remain unresolved
  (Google Books search-inside returns no snippet; the archive copy is
  lending-only; the e-bookshelf sample is HTML, not the book).
