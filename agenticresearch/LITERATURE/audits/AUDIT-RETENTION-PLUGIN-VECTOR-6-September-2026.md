# Prior-art triangulation for the O7 audit — 6 September 2026

**Claims:** `RETENTION-PLUGIN-CLT-FROZEN-VECTOR`, `RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE`
**Audit:** `AUDIT-RETENTION-PLUGIN-VECTOR`
**Search:** novelty search on the frozen statements, run by the auditor independently of
`RETENTION-PLUGIN-CLT-FROZEN-VECTOR-5-September-2026.md` (opened afterwards for comparison).
Retrieval by a delegated web pass (abstracts, publisher records, citing restatements); no
paywalled text was read. Labels: **VERIFIED** (statement seen in text), **ABSTRACT** (abstract
only), **RESTATED** (formula seen in a citing paper), **NOT FOUND**.

## Terminology fan-out used

Wilks' Lambda under fixed alternatives; canonical correlations with an indicator (dummy)
block; canonical discriminant analysis, nonnormal; between-over-total determinant ratio;
influence function of squared canonical correlations; asymptotics of eigenvalues of
\(S_1S_2^{-1}\)-type matrices with finite fourth moments; tests of rank / dimensionality;
smallest sample canonical correlations under rank deficiency; uncentred second-moment matrices.

## Triangulation table

| Source | Exact problem | Exact result | Objective | Feasible set | Transfers | Does not transfer |
|---|---|---|---|---|---|---|
| van der Vaart (1998) Thm 3.1 — VERIFIED (primary text, O6 audit) | delta method | \(\sqrt n(\phi(\bar T)-\phi(\theta))\Rightarrow\nabla\phi^\top N(0,\Sigma)\) for \(\phi\) differentiable at \(\theta\) | none | any map differentiable at the point; \(\bar T\) may take values anywhere in the domain | verbatim: O7.2 is this theorem applied to \(\phi(p,M,V)\), the estimator's own everywhere-defined functional | nothing about the functional's endpoints |
| Magnus & Neudecker (1999) §8.3–8.4 — VERIFIED (textbook) | matrix differentials | \(d\log\det A=\operatorname{tr}A^{-1}dA\), \(dA^{-1}=-A^{-1}(dA)A^{-1}\) | none | nonsingular \(A\) | verbatim: the gradient of \(\log\det I_Z-\log\det V\) | — |
| Romanazzi (1992) Psychometrika 57 — RESTATED via Taskinen et al. (2006) and arXiv:1705.04194 | influence in CCA | \(\operatorname{IF}(\rho_i^2)=2\rho_iu_iv_i-\rho_i^2(u_i^2+v_i^2)\) per squared canonical correlation, simple roots | none | two random blocks, centred covariance, distinct \(\rho_i\) | the per-coefficient form; O7.2's \(\psi\) is \((\eta_D/d)\sum_i\operatorname{IF}(\rho_i^2)/\rho_i^2\) with the second block the cell indicator | centring; the second block here is a fixed partition of a *different* variable; repeated roots not covered |
| Muirhead & Waternaux (1980) Biometrika 67 — ABSTRACT | CCA asymptotics under finite fourth moments | sample canonical correlations asymptotically normal; the covariance involves fourth cumulants; elliptical case explicit | none | finite fourth moments; distinct nonzero population correlations for the per-root statement | the moment condition (A2) is the classical one; nonnormality is the standard setting | per-root, centred, random blocks; no fixed partition, no determinant functional |
| Fang & Krishnaiah (1982) JMVA 12 — NOT FOUND (citation confirmed; abstract read by the researcher on 5 Sep, not by the auditor) | functions of eigenvalues, nonnormal | asymptotic normality of smooth symmetric functions of eigenvalues, multiplicities allowed | none | finite fourth moments | the determinant is a symmetric function, so repeated \(\rho_i\) are covered — O7's determinant route needs none of it | not read; cited for the method only |
| Seo, Kanda & Fujikoshi (1995) JMVA 52 — NOT FOUND (citation confirmed) | dimensionality tests in CCA/MANOVA under nonnormality | (not read) | test size | rank \(r<\min\) | the *problem* of O7.4(b) is theirs: smallest sample roots when the population rank is deficient | the O7.4(b) determinant-and-Schur derivation is not attributed to them |
| Robin & Smith (2000) Econometric Theory 16 — ABSTRACT (verified) | tests of rank from estimated characteristic roots | under population rank \(r\), the \(n\)-scaled smallest roots converge to a weighted sum of independent \(\chi^2\) | test | \(\sqrt n\)-consistent asymptotically normal matrix estimate | the shape of O7.4(b)'s limit: a quadratic form \(WP^{-1}W^\top\) of Gaussian limits of the null-direction moments; at \(d-r=1\) O7's limit is exactly such a weighted \(\chi^2\) (endpoint2: \((G_0+G_1)^2/\pi\)) | determinant vs trace; uncentred Gram matrix of a fixed partition; no test is built; the \(n^{-(d-r)/d}\) bias reading is O7's |
| Eaton & Tyler (1991) Ann. Statist. 19 — ABSTRACT | eigenvalues of a random symmetric matrix | Wielandt's inequality gives the asymptotic distribution of eigenvalues in a general setting, repeated roots included | none | \(\sqrt n\)-asymptotically normal matrix | general machinery behind both endpoints | not needed: the determinant route avoids eigenvalues |
| Anderson (2003) Ch. 8, 12 — VERIFIED (textbook) | Wilks' \(\Lambda\), CCA | normal-theory distribution of \(\Lambda\); canonical correlations | none | Gaussian | the classical reading in O7.0 (\(\prod(1-\rho_i^2)=\det(V-I_Z)/\det V\)) | normal theory transfers nothing to (A2)-only laws |
| Kendall & Stuart (1961) Vol. 2 §26 — VERIFIED (textbook) | correlation ratio decomposition | total = between + within, (26.50) | none | any grouping | O7.1 is the uncentred matrix form | sampling theory there is normal and centred |

## Outcome

- **Method: prior art, confirmed.** Delta method, matrix differentials, canonical-correlation
  influence functions (Romanazzi's per-coefficient form now RESTATED through Taskinen et al.
  2006 / Sinha et al. 2017), fourth-moment asymptotic normality of eigenvalue functionals
  (Muirhead & Waternaux ABSTRACT; Fang & Krishnaiah citation).
- **Exact statement: search gap, not novelty.** No located source states the uncentred
  fixed-partition determinant ratio on the library's cell moments with \(0/0:=0\), its explicit
  \(\psi\), the plug-in variance, the ellipsoid characterisation of \(\sigma^2=0\) with the
  atomless witness, or the \(n^{-(d-r)/d}\) rate with its determinant/Schur limit. The rank-test
  literature (Robin & Smith 2000) has the same *shape* of limit for the smallest roots; the
  determinant form and the bias reading are O7's.
- **No re-attribution.** Nothing in O7 is claimed as novelty; nothing found requires
  re-attributing a project-level statement to a published theorem.
- **Bibliographic note.** The researcher's `LITERATURE/gaps.md` entry stands: Radhakrishnan &
  Kshirsagar (1981) is still unread and unrestated (no accessible restatement found on 6 Sep
  either); Fang & Krishnaiah (1982) and Seo, Kanda & Fujikoshi (1995) remain citation-level.
  New keys: `Robin-Smith-2000`, `Taskinen-Croux-Kankainen-Ollila-Oja-2006`.
