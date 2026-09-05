# LIT-RETENTION-PLUGIN-VECTOR — literature-first pass for the vector retention CLT

**Programme:** P4 (OP27) · **Opened and closed:** 5 September 2026 ·
**Status:** completed (ad hoc, owner's request; no packet was queued)

## Question

Before spending a session proving the vector geometric-mean retention CLT
(`WORK/active/RETENTION-PLUGIN-VECTOR.md`), is the statement, or its method,
already in the literature? Also: retry the two O6 primary-text gaps.

## Outcome

- **Method: prior art.** With \(M=E[S\mathbf 1_Z^\top]\), \(P=\operatorname{diag}p\),
  \(\eta_D^d=\det(V^{-1}MP^{-1}M^\top)=\prod\rho_i^2\): the geometric mean of the
  squared uncentred canonical correlations between the score and the cell
  indicator, a MANOVA between/total determinant ratio at the evaluation law.
  Delta method (van der Vaart 3.1) + determinant/inverse differentials
  (Magnus & Neudecker §8.3–8.4) + influence functions of \(\Lambda\)-type
  parameters (Radhakrishnan & Kshirsagar 1981; Romanazzi 1992) +
  fourth-moment CCA asymptotics (Muirhead & Waternaux 1980; Fang &
  Krishnaiah 1982) + non-null determinant-ratio expansions (Sugiura &
  Fujikoshi 1969; Fujikoshi 2002) + the dimensionality reading of singular
  \(I_Z\) (Seo, Kanda & Fujikoshi 1995).
- **Exact statement: search gap.** Geometric-mean form on a fixed partition of
  a different variable, uncentred moments, \(0/0:=0\), plug-in variance with
  consistency, endpoints and the \(\sigma^2=0\) set. Not novelty.
- **O6 gaps:** Kendall & Stuart Vol. 2 §26.21–26.24 read (primary text): no
  large-sample variance of the correlation ratio exists there. Serfling §3.3
  labels still open.

## Artifacts

- `LITERATURE/audits/RETENTION-PLUGIN-CLT-FROZEN-VECTOR-5-September-2026.md`
  (six-field table, forward round counts, verdict).
- `LITERATURE/topics/08-plug-in-asymptotics.md` (10 new keyed entries),
  `graph.json` round 11 (20 records), `reviewed.md`, `rejected.md`, `gaps.md`.
- `registry.json` bibliography +10; `OPEN-RETENTION-UNCERTAINTY` and
  `RETENTION-PLUGIN-CLT-FROZEN-SCALAR` link the keys.
- `papers/`: Cai–Liang–Zhou arXiv PDF committed; `papers/local/` (gitignored)
  holds the contents scans and notes, listed in `papers/README.md`.
- `WORK/active/RETENTION-PLUGIN-VECTOR.md` rewritten: cite-vs-derive table,
  endpoints first; `PLAYBOOK.md` prompt updated.

## Validation

`registry.py reindex`, `registry.py validate`, research-claims tests (see the
commit).

## Next question

Run `WORK/active/RETENTION-PLUGIN-VECTOR.md`. Before citing them for more than
the method, read the statements of Radhakrishnan & Kshirsagar (1981) and Fang
& Krishnaiah (1982) (paywalled; `gaps.md`).
