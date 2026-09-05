# 10. Plug-in asymptotics for retention functionals

> Curated theorem-level annotations. Machine records for the citation graph
> live in `graph.json`; `BIBLIOGRAPHY.md` (generated) maps every registry
> bibliography key to the heading that annotates it. Opened 5 September 2026
> for `RETENTION-PLUGIN-CLT-FROZEN-SCALAR` (O6); every entry states its
> verification status — no primary text was read in the opening round.

## van der Vaart (1998) — Asymptotic Statistics

**Key:** vanderVaart-1998

**Paper:** *Asymptotic Statistics*, Cambridge University Press.
**Result:** Theorem 3.1 (delta method): if \(r_n(T_n-\theta)\Rightarrow T\) and
\(\phi\) is differentiable at \(\theta\), then \(r_n(\phi(T_n)-\phi(\theta))\Rightarrow\phi'_\theta(T)\).
Chapter 3 works the sample variance under \(E X^4<\infty\); Chapter 5
(Theorem 5.23) gives the M-estimator sandwich and the plug-in variance
principle.
**Use:** O6.2 is this theorem applied to the sample-mean vector of
\((\mathbf 1_{Z=b}, S\mathbf 1_{Z=b}, S^2)\); O6.3's consistency argument is the
plug-in principle made explicit for a polynomial functional.
**Verification:** Theorem 3.1 read in primary text (the author's lecture notes with identical numbering; book contents corroborate); the sample-variance example is Example 3.2 (primary text); Chapter 20 is the functional delta method (primary text, contents); Theorem 5.23 sits in §5.3 but its number is secondary (audit 5 Sep 2026).

## Cramér (1946) — Mathematical Methods of Statistics

**Key:** Cramer-1946

**Paper:** *Mathematical Methods of Statistics*, Princeton University Press.
**Result:** Chapter 28, functions of sample moments: a smooth function of
sample moments is asymptotically normal with variance from the first-order
expansion, under the moment conditions that make the underlying sample moments
asymptotically normal.
**Use:** the oldest form of O6.2's argument; the fourth-moment condition (A2)
is exactly the classical condition for the second sample moment.
**Verification:** Chapter 28 "Asymptotic properties of sampling distributions", §28.4 "Functions of moments" confirmed in primary text (archive.org contents scan); finite-sample companions §27.4, §27.7 (audit 5 Sep 2026).

## Serfling (1980) — Approximation Theorems of Mathematical Statistics

**Key:** Serfling-1980

**Paper:** *Approximation Theorems of Mathematical Statistics*, Wiley.
**Result:** Chapter 3 (transformations of given statistics), Section 3.3
Theorem B: the multivariate delta method for functions of asymptotically normal
vectors.
**Use:** alternative citation for O6.2 with the gradient-covariance form
\(\nabla g^\top\Sigma\nabla g\) stated explicitly.
**Verification:** Chapter 3 title and page (p. 117) primary record; the Theorem A / Theorem B labels of §3.3 remain unresolved after the audit's attempt (no readable text reachable, audit 5 Sep 2026) — cite as "Serfling 1980, §3.3" until read.

## Hampel, Ronchetti, Rousseeuw & Stahel (1986) — influence functions

**Key:** Hampel-Ronchetti-Rousseeuw-Stahel-1986

**Paper:** *Robust Statistics: The Approach Based on Influence Functions*, Wiley.
**Result:** for a statistical functional \(T\) at \(F\), the asymptotic variance
is \(V(T,F)=\int \mathrm{IF}(x;T,F)^2\,dF(x)\).
**Use:** O6.2's \(\sigma^2=E[\psi^2]\) is this formula; O6's selftest checks
the closed-form \(\psi\) against Gateaux finite differences, which is the
definition of the influence function.
**Verification:** Chapter 2 begins p. 78 (primary record); \(V(T,F)=\int\mathrm{IF}^2dF\) is cited to p. 85 within §2.1 (secondary); equation numbers unresolved (audit 5 Sep 2026).

## Wishart (1932) — distribution of the correlation ratio

**Key:** Wishart-1932

**Paper:** *A note on the distribution of the correlation ratio*, Biometrika
24(3/4):441–456.
**Result:** normal-theory exact distribution of the sample correlation ratio
\(\eta^2\) (between-group over total centred sum of squares).
**Use:** lineage only — O6's \(\hat\eta\) is the *uncentred* correlation ratio
of \(S\) on \(Z\) with a fixed partition; nothing normal-theory transfers.
**Verification:** existence and pages from the Biometrika contents index;
text not read.

## Kelley (2007) — confidence intervals for standardized effect sizes

**Key:** Kelley-2007

**Paper:** *Confidence intervals for standardized effect sizes: theory,
application, and implementation*, Journal of Statistical Software 20(8).
**Result:** noncentral-\(F\)/\(t\)/\(\chi^2\) confidence intervals for
\(\eta^2\), \(R^2\) and Cohen's \(d\) under normal-theory ANOVA (MBESS).
**Use:** the standard applied route for \(\eta^2\) intervals, assuming
Gaussian errors and a fixed design; O6 replaces that assumption by
\(E S^4<\infty\) and a first-order limit.
**Verification:** verified at jstatsoft.org.

## Bhattacharya & Ghosh (1978) — validity of the Edgeworth expansion

**Key:** Bhattacharya-Ghosh-1978

**Paper:** *On the validity of the formal Edgeworth expansion*, Annals of
Statistics 6(2):434–451.
**Result:** the formal Edgeworth expansion of \(\sqrt n[H(\bar Z)-H(\mu)]\)
is valid for smooth \(H\) of a sample-mean vector under moment and
Cramér-type conditions; the class covers all smooth functions of sample
moments.
**Use:** the theoretical backdrop of O6.7's second-order readings
(\(O(1/n)\) bias, skew); not used in the O6 proof.
**Verification:** abstract read (primary text), audit 5 Sep 2026.

## Hall (1992) — The Bootstrap and Edgeworth Expansion

**Key:** Hall-1992

**Paper:** *The Bootstrap and Edgeworth Expansion*, Springer.
**Result:** Edgeworth expansions for the smooth function model; the
studentized statistic's skewness term differs in sign and size from the
non-studentized one, the standard explanation of small-\(n\) under-coverage
of plug-in Wald intervals.
**Use:** explains the measured sign flip of the studentized skew in O6.7 and
the audit's replication; a bootstrap-\(t\) correction is the standard
remedy (out of the O6 packet's scope).
**Verification:** record only (secondary); the studentized skewness
discussion is cited to p. 76, section number unresolved.

## Donner & Koval (1980) — large-sample variance of an intraclass correlation

**Key:** Donner-Koval-1980

**Paper:** *The large sample variance of an intraclass correlation*,
Biometrika 67(3):719–722.
**Result:** delta-method variance of the ANOVA intraclass-correlation
estimator under the one-way random-effects model.
**Use:** the nearest applied delta-method treatment of a between-over-total
ratio; the grouping is random and the law normal, so nothing beyond the
algebraic shape transfers to O6's fixed partition of an oracle score.
**Verification:** journal record (primary record), audit 5 Sep 2026.

## Radhakrishnan & Kshirsagar (1981) — influence functions in multivariate analysis

**Key:** Radhakrishnan-Kshirsagar-1981

**Paper:** *Influence functions for certain parameters in multivariate
analysis*, Communications in Statistics — Theory and Methods 10(6):515–529.
doi:10.1080/03610928108828055.
**Result:** closed-form Hampel influence functions for the generalized
variance, the regression-coefficient matrix, the MANOVA noncentrality
matrix \(\Sigma^{-1}\delta\) and its eigenvalues, canonical correlations,
principal components and the parameters behind Pillai's statistic,
Hotelling's \(T_0^2\) and Wilks' \(\Lambda\); \(r^2\), \(R^2\) and Mahalanobis
\(D^2\) as special cases.
**Use:** the matrix influence function named in O6.8 exists here in the
centred several-populations form; at the evaluation law O7's \(I_Z\) is the
between-group matrix, so the packet's \(\psi\) must reduce to theirs.
**Verification:** abstract read (primary text, OpenAlex/T&F record);
full text paywalled (5 Sep 2026 vector-packet literature pass).

## Romanazzi (1992) — influence in canonical correlation analysis

**Key:** Romanazzi-1992

**Paper:** *Influence in canonical correlation analysis*, Psychometrika
57(2):237–259. doi:10.1007/BF02294507.
**Result:** perturbation theory of the generalized eigenproblem yields the
influence function of each squared canonical correlation and canonical
vector pair; three sample versions; the squared multiple correlation and
correspondence-analysis eigenvalues as special cases.
**Use:** second route to O7's influence function
(\(\log\eta_D=d^{-1}\sum_i\log\rho_i^2\)); needs simple \(\rho_i\), which the
determinant route does not.
**Verification:** abstract read (primary text, Crossref), 5 Sep 2026.

## Muirhead & Waternaux (1980) — canonical correlations without normality

**Key:** Muirhead-Waternaux-1980

**Paper:** *Asymptotic distributions in canonical correlation analysis and
other multivariate procedures for nonnormal populations*, Biometrika
67(1):31–43. doi:10.1093/biomet/67.1.31.
**Result:** for populations with finite fourth moments, the sample canonical
correlations and their test statistics are asymptotically normal with a
covariance involving fourth-order cumulants; elliptical simplification and
a corrected chi-squared test for zero coefficients.
**Use:** O7's fourth-moment hypothesis and mechanism; \(\eta_D\) is a smooth
symmetric function of their \(\rho_i^2\), so the vector CLT off the
endpoints follows by one delta step. One block of O7 is the bounded
indicator \(\mathbf 1_Z\), and the moments are uncentred.
**Verification:** abstract read (primary text, OUP page), 5 Sep 2026.

## Fang & Krishnaiah (1982) — functions of eigenvalues, nonnormal populations

**Key:** Fang-Krishnaiah-1982

**Paper:** *Asymptotic distributions of functions of the eigenvalues of
some random matrices for nonnormal populations*, Journal of Multivariate
Analysis 12(1):39–63. doi:10.1016/0047-259X(82)90081-1.
**Result:** limit laws of smooth functions of the eigenvalues of the
standard covariance/MANOVA/canonical-correlation random matrices under
nonnormal sampling (fourth-cumulant covariance).
**Use:** the closest general theorem to "\(\sqrt n(\hat\eta_D-\eta_D)\) is
normal"; citation candidate once read.
**Verification:** journal record only (primary record, OpenAlex); text
unread (5 Sep 2026).

## Seo, Kanda & Fujikoshi (1995) — dimensionality tests under nonnormality

**Key:** Seo-Kanda-Fujikoshi-1995

**Paper:** *The effects of nonnormality on tests for dimensionality in
canonical correlation and MANOVA models*, Journal of Multivariate Analysis
52(2):325–337. doi:10.1006/jmva.1995.1017. Companion: Seo, Kanda &
Fujikoshi (1994), Comm. Statist. Theory Methods 23(9):2615–2628
(expansions for functions of canonical correlations under nonnormality,
perturbation method; abstract read).
**Result:** null distributions of the dimensionality (rank) tests in CCA
and MANOVA under nonnormal populations.
**Use:** the singular-\(I_Z\) endpoint of O7 (\(\eta_D=0\)) is the
dimensionality hypothesis; the plug-in is then \(O_p(1/n)\), not
\(O_p(n^{-1/2})\).
**Verification:** journal record (primary record); the 1994 companion's
abstract read (5 Sep 2026).

## Sugiura & Fujikoshi (1969) — non-null expansions of the LR criterion

**Key:** Sugiura-Fujikoshi-1969

**Paper:** *Asymptotic expansions of the non-null distributions of the
likelihood ratio criteria for multivariate linear hypothesis and
independence*, Annals of Mathematical Statistics 40(3):942–952.
doi:10.1214/aoms/1177697599.
**Result:** the non-null distribution of the LR (determinant-ratio)
criterion for the multivariate linear hypothesis, expanded to order
\(N^{-2}\) under a fixed alternative with no rank assumption on the
noncentrality matrix; normal theory.
**Use:** fixed-alternative asymptotic normality of a determinant ratio and
the second-order terms behind O7's coverage table; the fourth-moment
version is Fujikoshi (2002), JSPI 108:263–282 (record only).
**Verification:** abstract read (primary text, OpenAlex), 5 Sep 2026.

## Magnus & Neudecker — matrix differential calculus

**Key:** Magnus-Neudecker-1999

**Paper:** *Matrix Differential Calculus with Applications in Statistics
and Econometrics*, Wiley (1988; revised 1999; 3rd ed. 2019).
**Result:** Part Three, Chapter 8 "Some important differentials": §8.3 "The
differential of a determinant" (p. 149, \(d|X|=|X|\operatorname{tr}X^{-1}dX\)),
§8.4 "The differential of an inverse" (p. 151); Chapter 9 §9.10 "Scalar
functions of a matrix, II: determinant" (p. 178).
**Use:** the two differentials behind O7's influence function.
**Verification:** contents scan (primary record, GBV catalogue PDF held in
`papers/local/`); the scan's edition is not identified on the page, and
the 2019 edition renumbers — verify page numbers before citing them
(5 Sep 2026).

## Anderson — An Introduction to Multivariate Statistical Analysis

**Key:** Anderson-2003

**Paper:** *An Introduction to Multivariate Statistical Analysis*, Wiley,
3rd ed. 2003 (2nd ed. 1984).
**Result:** §7.5 "The generalized variance"; §8.3–8.5 likelihood-ratio
criterion for linear hypotheses, its distribution and asymptotic expansion;
§8.9 "Multivariate analysis of variance"; §12.2–12.4 canonical
correlations, estimation and inference; §13.5–13.6 asymptotic
distributions of roots (one and two Wishart matrices). Normal theory
throughout these sections.
**Use:** background for O7's determinant ratio and the canonical-correlation
reading.
**Verification:** section titles from the 2nd-edition contents scan
(primary record, `papers/local/`); the 3rd-edition numbering and the
theorem giving the sample covariance's asymptotic law under finite fourth
moments (cited in the packet as "Anderson 2003 §3") are **unverified**
(5 Sep 2026).

## Cramer & Nicewander (1979) — measures of multivariate association

**Key:** Cramer-Nicewander-1979

**Paper:** *Some symmetric, invariant measures of multivariate
association*, Psychometrika 44(1):43–54. doi:10.1007/BF02293783.
**Result:** the symmetric, linearly invariant measures of association
between two sets of variables are all functions of the canonical
correlations, generalize \(R^2\), and are strictly ordered for any two sets.
**Use:** lineage of \(\eta_D=(\prod\rho_i^2)^{1/d}\) as a symmetric invariant
measure; van den Burg & Lewis (1988), Psychometrika 53(1):109–122, compare
the \(\Lambda\)-based and Pillai-based indices.
**Verification:** abstract read (primary text, Crossref); whether the
geometric mean itself is in their list is unread (paywalled, `gaps.md`),
5 Sep 2026.

## Kendall & Stuart (1961) — the correlation ratio, Vol. 2 Chapter 26

**Key:** Kendall-Stuart-1961

**Paper:** *The Advanced Theory of Statistics*, Vol. 2 *Inference and
Relationship*, Griffin, 1961 (2nd ed. 1963/Hafner).
**Result:** §26.21–26.22 define the population correlation ratio
(26.40), (26.47) and the sample version (26.48) with \(0\le r^2\le e^2\le1\)
(26.49); §26.23–26.24 give the decomposition (26.50)–(26.51) and the
normal-theory \(F\) tests (26.52)–(26.55) for \(\eta^2=0\), \(\rho=0\) and
linearity, attributed to Fisher, via Cochran's theorem with the arrays
fixed. No large-sample variance of the correlation ratio and no
moment-based non-normal form are given.
**Use:** lineage of O6's uncentred correlation ratio; (26.50) is the
centred analogue of O6.1's RSS identity. Closes the O6 gap "Kendall &
Stuart Vol. 2 Ch. 26 section".
**Verification:** primary text (archive.org OCR of the 1961 volume), 5 Sep
2026; Vol. 1 Chapter 10 (standard errors) not checked.
