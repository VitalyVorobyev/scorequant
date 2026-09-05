# Reference papers

Primary literature behind the ScoreQuant theory (see `research/score_space_quantization_article.html`
and `docs/bibliography.md`). Only openly redistributable versions are committed: arXiv preprints and
PMLR open-access proceedings. Publisher-only papers are listed with their DOI instead.

## Committed copies

| File | Reference |
| --- | --- |
| `1310.6945v1.pdf` | R. C. Farias, J.-M. Brossier. "Optimal Scalar Quantization for Parameter Estimation." arXiv:1310.6945 (2013). |
| `cranmer_pavez_louppe_2015_calibrated_ratios_arXiv1506.02169.pdf` | K. Cranmer, J. Pavez, G. Louppe. "Approximating Likelihood Ratios with Calibrated Discriminative Classifiers." arXiv:1506.02169 (2015). |
| `brehmer_louppe_pavez_cranmer_2018_mining_gold_arXiv1805.12244.pdf` | J. Brehmer, G. Louppe, J. Pavez, K. Cranmer. "Mining gold from implicit models to improve likelihood-free inference." PNAS 117(10), 5242–5249 (2020). arXiv:1805.12244. |
| `decastro_dorigo_2018_inferno_arXiv1806.04743.pdf` | P. de Castro, T. Dorigo. "INFERNO: Inference-Aware Neural Optimisation." Comput. Phys. Commun. 244, 170–179 (2019). arXiv:1806.04743. |
| `matchev_shyamsundar_2019_optimal_event_categorization_arXiv1911.12299.pdf` | K. T. Matchev, P. Shyamsundar. "Optimal event selection and categorization in high energy physics. Part I: Signal discovery." JHEP 03, 291 (2021). arXiv:1911.12299. |
| `erdmann_2026_learning_to_bin_arXiv2601.07756.pdf` | J. Erdmann, N. K. Kasaraguppe, F. Mausolf. "Learning to bin: differentiable and Bayesian optimization for multi-dimensional discriminants in high-energy physics." arXiv:2601.07756 (2026). |
| `valassi_2020_weight_derivative_regression_arXiv2003.12853.pdf` | A. Valassi. "Optimising HEP parameter fits via Monte Carlo weight derivative regression." CHEP2019 proceedings, EPJ Web of Conferences (2020). arXiv:2003.12853. Registry key `Valassi-2020`. |
| `cms_2025_systematic_aware_nn_training_arXiv2502.13047.pdf` | CMS Collaboration. "Development of systematic uncertainty-aware neural network trainings for binned-likelihood analyses at the LHC." Eur. Phys. J. C (2025). doi:10.1140/epjc/s10052-025-14713-w. arXiv:2502.13047, CMS-MLG-23-005. Registry key `CMS-2025`. |
| `telgarsky_vattani_2010_hartigan_kmeans_pmlr_v9.pdf` | M. Telgarsky, A. Vattani. "Hartigan's Method: k-means Clustering without Voronoi." AISTATS, PMLR 9, 820–827 (2010). |
| `barnes_han_ozgur_2019_fisher_communication_constraints_arXiv1902.02890.pdf` | L. P. Barnes, Y. Han, A. Özgür. "Lower Bounds for Learning Distributions under Communication Constraints via Fisher Information." JMLR 21(236) (2020). arXiv:1902.02890. Extends the Allerton 2018 quantized-Fisher-information geometry below. |
| `the-equivalence-of-two-extremum-problems.pdf` | J. Kiefer, J. Wolfowitz. "The Equivalence of Two Extremum Problems." Canadian J. Math. 12, 363–366 (1960). doi:10.4153/CJM-1960-030-4. Origin of the D-/G-optimality equivalence theorem. |
| `cai_liang_zhou_2015_log_determinant_sample_covariance_arXiv1309.0482.pdf` | T. T. Cai, T. Liang, H. H. Zhou. "Law of log determinant of sample covariance matrix and optimal estimation of differential entropy for high-dimensional Gaussian distributions." J. Multivariate Anal. 137, 161–172 (2015). arXiv:1309.0482. Corollary 1: fixed-dimension CLT for the log determinant (Gaussian). Screened for the vector retention packet, 5 Sep 2026. |

## Referenced, not redistributable here (use the DOI)

- P. Venkitasubramaniam, L. Tong, A. Swami. "Score-Function Quantization for Distributed Estimation." CISS 2006. doi:10.1109/CISS.2006.286494. (A copy may live locally in this folder; it is deliberately not committed.)
- L. P. Barnes, Y. Han, A. Özgür. "A Geometric Characterization of Fisher Information from Quantized Samples…" Allerton 2018. doi:10.1109/ALLERTON.2018.8635899.
- B. Dülek. "On the Optimality of Sufficient Statistics-Based Quantizers." IEEE TPAMI 45(3), 3567–3573 (2023). doi:10.1109/TPAMI.2022.3172282.
- H. P. Friedman, J. Rubin. "On Some Invariant Criteria for Grouping Data." JASA 62(320), 1159–1178 (1967). doi:10.1080/01621459.1967.10500923.
- A. J. Scott, M. J. Symons. "Clustering Methods Based on Likelihood Ratio Criteria." Biometrics 27(2), 387–397 (1971). doi:10.2307/2529003.
- J. A. Hartigan. *Clustering Algorithms.* Wiley (1975).
- M. Inaba, N. Katoh, H. Imai. "Applications of weighted Voronoi diagrams and randomization to variance-based k-clustering." SoCG 1994. doi:10.1145/177424.178042.
- A. Dvoretzky, A. Wald, J. Wolfowitz. "Elimination of Randomization in Certain Statistical Decision Procedures…" Ann. Math. Statist. 22(1), 1–21 (1951). doi:10.1214/aoms/1177729689.
- M. A. Khan, K. P. Rath, Y. Sun. "The Dvoretzky–Wald–Wolfowitz theorem and purification in atomless finite-action games." Int. J. Game Theory 34, 91–104 (2006). doi:10.1007/s00182-005-0004-3.
- D. Pollard. "Strong Consistency of K-Means Clustering." Ann. Statist. 9(1), 135–140 (1981). doi:10.1214/aos/1176345339.
- Q. Du, V. Faber, M. Gunzburger. "Centroidal Voronoi Tessellations: Applications and Algorithms." SIAM Review 41(4), 637–676 (1999). doi:10.1137/S0036144599352836.
- Q. Du, M. Emelianenko, L. Ju. "Convergence of the Lloyd Algorithm for Computing Centroidal Voronoi Tessellations." SIAM J. Numer. Anal. 44(1), 102–119 (2006). doi:10.1137/040617364.
- F. Pukelsheim. *Optimal Design of Experiments.* SIAM Classics. doi:10.1137/1.9780898719109.
- W. Näther, V. Reinsch. "D_s-optimality and Whittle's equivalence theorem." Series Statistics 12(3), 307–316 (1981). doi:10.1080/02331888108801591.
- P. Whittle. "Some General Points in the Theory of Optimal Experimental Design." JRSS B 35(1), 123–130 (1973). doi:10.1111/j.2517-6161.1973.tb00944.x.
- F. H. C. Marriott. "Practical Problems in a Method of Cluster Analysis." Biometrics 27(3), 501–514 (1971). doi:10.2307/2528592.
- D. R. Cox. "Note on Grouping." JASA 52(280), 543–547 (1957). doi:10.1080/01621459.1957.10501411. (Early direct prior art: 1D grouping chosen to minimize information loss.)
- J. Ogawa. "Contributions to the theory of systematic statistics, I." Osaka Math. J. 3(2), 175–213 (1951). (Optimal spacings of order statistics maximizing retained Fisher information — asymptotic 1D ancestor.)
- S. P. Lloyd. "Least Squares Quantization in PCM." IEEE Trans. Inf. Theory 28(2), 129–137 (1982; Bell Labs memo 1957). doi:10.1109/TIT.1982.1056489.
- J. Max. "Quantizing for minimum distortion." IRE Trans. Inf. Theory 6(1), 7–12 (1960). doi:10.1109/TIT.1960.1057548.
- J. N. Tsitsiklis. "Extremal properties of likelihood-ratio quantizers." IEEE Trans. Commun. 41(4), 550–558 (1993). doi:10.1109/26.223779. (Detection-side analogue: sufficiency of likelihood-ratio space for quantizer design.)
- R. M. Gray, D. L. Neuhoff. "Quantization." IEEE Trans. Inf. Theory 44(6), 2325–2383 (1998). doi:10.1109/18.720541. (Canonical survey of quantization theory.)

## Held locally, not committed (`papers/local/`, gitignored)

Publisher scans, catalogue contents pages and author-hosted notes that are
not ours to redistribute. Each entry says what it is and what was read from
it, so the next session need not re-fetch it. Re-download instructions are
in `agenticresearch/LITERATURE/audits/` files of the same date.

| File | What it is | What was used |
| --- | --- | --- |
| `anderson_1984_multivariate_2nd_ed_toc_gbv.pdf` | GBV catalogue scan of the contents of T. W. Anderson, *An Introduction to Multivariate Statistical Analysis*, 2nd ed. (1984). | Section titles §7.5, §8.3–8.5, §8.9, §12.2–12.4, §13.5–13.6 (registry key `Anderson-2003`). |
| `magnus_neudecker_matrix_differential_calculus_toc_gbv.pdf` | GBV catalogue scan of the contents of Magnus & Neudecker, *Matrix Differential Calculus* (1988/1999 edition, not identified on the page). | Ch. 8 §8.3 determinant, §8.4 inverse; Ch. 9 §9.10 (key `Magnus-Neudecker-1999`). |
| `vandervaart_1998_asymptotic_statistics_frontmatter_cup.pdf` | Cambridge University Press front matter and contents of van der Vaart, *Asymptotic Statistics* (1998). | Chapter 3 §3.1, Chapter 20 titles (key `vanderVaart-1998`, O6 audit). |
| `vandervaart_mathematische_statistiek_lecture_notes.pdf` | A. W. van der Vaart, *Mathematische Statistiek*, author-hosted lecture notes (Dutch) with the book's numbering. | Theorem 3.1 (delta method) statement read here (O6 audit). |

Also used but not stored: the archive.org OCR text of Kendall & Stuart
Vol. 2 (1961), read in full for Chapter 26 (key `Kendall-Stuart-1961`); it
is public on archive.org and 2 MB, so it is re-fetched on demand.
