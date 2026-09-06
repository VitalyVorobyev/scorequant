# Known results — index

> Canonical theorem/result ledger, one file per chapter. Read `PROBLEM.md`
> first. Resolve any claim id -- node, dependency closure, and proof prose --
> with `python py/registry.py show <ID> --deps --proof`.

**Version:** 3.1 · 3 September 2026  
**Purpose:** canonical theorem/result ledger. Read `PROBLEM.md` first.

Status vocabulary (mirrors `registry.json` `status_definitions`):

- **[LIT]** established literature.
- **[BRIDGE]** direct synthesis/translation of known results into ScoreQuant notation.
- **[PROJECT-PROVED]** derived in this project and currently treated as proved internally; still subject to publication-grade audit.
- **[COUNTEREXAMPLE]** explicit falsification in the project setting.
- **[MEASURED]** numerical evidence/regression test, not a theorem.
- **[CONJECTURE]** precise unproved conjecture.
- **[SEARCH-GAP]** no direct precedent located in targeted search; not a novelty proof.
- **[OPEN]** unresolved.

---

## Chapters

| Chapter | File | Results |
|---|---|---|
| 1. Universal information structure | `01-universal.md` | U1, U2, U3, U4, U5, U6, U7 |
| 2. Trace control case | `02-trace.md` | T1 |
| 3. Generic first-order and finite screening results | `03-screening.md` | G1, G2 |
| 4. Full D-optimality | `04-d-optimality.md` | D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12 |
| 5. \(D_s\)-optimality | `05a-ds-core.md` | DS0, DS1, DS2, DS3, DS4, DS5, DS6, DS7, DS8, DS9, DS10 |
| 5. \(D_s\)-optimality — finite-to-population bridge | `05b-ds-bridge.md` | DS11, DS12, DS13, DS14, DS15, DS16, DS17, DS18, DS19 |
| 6. E-optimality control theory | `06-e-optimality.md` | E1, E2, E3, E4, E5, E6 |
| 7. A-optimality control theory | `07-a-optimality.md` | A1, A2, A3, A4 |
| 8. Randomized/soft quantizers and empirical geometric optimization | `08-soft.md` | S1, S2, S3, S4, S5, S6 |
| 9. Empirical-to-population theory | `09-consistency.md` | C1, C2 |
| 10. Score/density-ratio/classifier access | `10-oracle.md` | O1, O2, O3, O4, O5, O6, O7 |
| 11. Information-efficiency outputs | `11-efficiency.md` | I1, I2, I3 |

## 12. Numerical evidence as regression tests — [MEASURED]

The measured ledger lives in `NUMERICAL_EVIDENCE.md` — one row per evidence
item (`N-*` id), with the claim node(s) it supports and the executable
source that produces it. It is not duplicated here; nothing in it is a proof.

Keep exact seeds/scripts or minimized fixtures beside any claim used in publication.

---

## 13. Guarantee hierarchy

Use this hierarchy precisely:

\[
\text{finite global optimum}
\subseteq
\text{one-point exchange stable}
\]

for any criterion where all admissible positive one-point moves are checked.

For full D, project theory strengthens this to

\[
\boxed{
\text{finite global}
\subseteq
\text{exchange stable}
\subsetneq
\text{strict self-consistent D-Voronoi}.
}
\]

For \(D_s\), A, and E the second inclusion fails in general.

Restricted-family local optima, population stationarity, and statistical consistency are separate notions and must not be placed in this finite inclusion chain.

---

## 14. Conservative novelty boundary

### Clearly known / inherited

- Fisher-optimal finite quantization and score-function quantization;
- scalar FI-loss/score-distortion theory;
- multivariate conditional-score representation of quantized FIM;
- trace-optimal polyhedral quantizers in sufficient-statistic space;
- D/\(D_s\)/A/E optimal-design theory and equivalence/sensitivity tools;
- determinant clustering and one-point exchange;
- vector quantization/CVT/Lloyd theory;
- differentiable inference-aware categorization;
- density-ratio estimation and classifier-ratio identities;
- DWW purification itself.

### Project synthesis/results that require publication-grade prior-art audit

- exact centroid-coupled rank-two relocation and closed D gain in this retained-between-score setting;
- D exchange-stability \(\Rightarrow\) strict self-consistent \(I_q^{-1}\)-Voronoi with quantitative finite gain;
- exact finite D inductive closure and geometric realizability of global optima;
- fixed-\((d,K)\) exact enumeration application and singleton-refinement B&B;
- \(D_s\) approximate finite geometry, efficient-score domination, and resulting upper certificates;
- criterion-separation counterexamples for \(D_s\), A, and E.

These are not to be labeled “first” solely because no direct precedent has yet been found.
