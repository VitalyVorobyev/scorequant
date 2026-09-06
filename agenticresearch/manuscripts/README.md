# Manuscripts — lagging integration target

**Role.** These are frozen publication-shaped snapshots of the research
program. The live scientific memory (`claims/`, `KNOWN_RESULTS/`,
`COUNTEREXAMPLES/`) is **ahead** of the manuscripts: results are audited,
assumptions hardened, and counterexamples added here first, and harvested into
a manuscript revision only when a publication decision is taken (see
`research-plan-proposal.md` — the paper is a by-product of the ledger, not a
driver). Never treat a manuscript statement as more current than the registry.

**Prefer the crosswalk below plus the registry over the article bodies.** The
v9 article is ~165 KB of Markdown source / ~195 KB rendered — greppable and
section-readable, but still an order of magnitude more than a claim lookup, so
open it only in a dedicated manuscript-revision task. Its Appendix H maps every
`NOVELTY_LEDGER.md` row to the section or appendix that carries it, and
Appendix G resolves every counterexample fixture id.

**Rendering.** The `.html` sibling is generated, never hand-edited:

```bash
uv run agenticresearch/py/render_manuscript.py agenticresearch/manuscripts/score_space_quantization_article_v9.md
```

The script (python-markdown declared inline, so `uv run` fetches it on demand)
carries the v8 stylesheet verbatim, loads MathJax 3 from its CDN, links `[n]`
citations to the `ref-n` anchors, renders `[novelty: …; ledger …]` tags as superscript provenance marks that stay
hidden until the sidebar's "Show provenance" button is pressed, and builds the
sidebar contents from the `##` headings (appendices under their own divider). Source conventions: `\(…\)` and
`\[…\]` math, section-local equation tags `(7.2)`, result boxes as
`<div class="theorem" markdown="1">`.

**Figures live in `figures/`, never inlined.** The six assets were base64
`data:` URIs until 29 Aug 2026, which made the article a 407 KB blob with
single lines over 100 KB: undiffable by git, unsearchable by grep, unloadable
by an agent. `py/registry.py validate` now fails on any `data:image/` payload
in the workspace, so this cannot regress. The `.html` needs its sibling
`figures/` directory to render — it is no longer a self-contained single file.

## Files

| File | What it is | State |
|---|---|---|
| `score_space_quantization_article_v9.md` / `.html` | Main research article, v9, 3 Sep 2026 (M12 session S5): "Information-optimal hard quantization of multivariate score space"; main text §1–§10 plus Appendices A–H | Current; drafted from `NOVELTY_LEDGER.md`, restructured the same day into main text + appendices after the owner rejected the flat first draft; Appendix H places every ledger row |
| `score_space_quantization_article_v8.md` / `.html` | Main research article, v8, 26 Aug 2026 | Superseded by v9; kept for the crosswalk and as the S2 reconciliation input |
| `NOVELTY_LEDGER.md` | Novelty ledger, 3 Sep 2026 (M12 session S2): one row per central v8 statement and per finding proved since v8, with novelty label, attribution and registry pointer | Current; the v9 draft (S5) is written from it |
| `../archive/doptimal_event_categorization_hep.html` | Shorter HEP-facing companion: D-optimal event categorization for multi-parameter inference | Archived 3 Sep 2026 (M12 consolidation); not a revision input |
| `../archive/scorequant_research_landscape_en.html` | Literature/landscape survey (history, key authors, software: MadMiner, INFERNO, ThickBrick, GATO, BOBR, OptBinning) | Archived 3 Sep 2026; superseded by `LITERATURE/` |

## Numbering crosswalk (manuscript v9 ↔ v8 ↔ registry)

Four numbering systems have circulated: manuscript v9 environments, manuscript
v8 environments, the older v5 research-notes numbering, and the registry claim
ids. The registry ids are canonical; the ledger row resolves each v9 result to
its claim ids and attribution. v9 has a main text (§1–§10, twelve numbered
results in one counter) and Appendices A–H (lettered results); the "first
draft" column is the rejected 3 September 2026 single-counter draft, kept only
because the S5 packet and the placement notes were written against it.

| Manuscript v9 | v8 | v9 first draft | v9 location | Novelty; ledger row |
|---|---|---|---|---|
| Proposition 1 (affine form of a common-metric stationary partition) | Proposition 1 | Proposition 1 | §3.2 | adaptation; V8-08 |
| Lemma B.1 (leverage bound) | Lemma 2 | Lemma 2 | Appendix B.2 | known; V8-10 |
| Theorem 2 (finite D exchange stability forces self-consistent Voronoi geometry) | Theorem 3 | Theorem 3 | §4.2 | apparently new; V8-11 |
| Proposition 3 (approximate finite efficient-Voronoi geometry) | Proposition 4 | Proposition 4 | §5.1 | unresolved; V8-24 |
| Lemma 4 (variational form of the generalized profiled information (classical)) | — | Lemma 5 | §5.2 | known; DS11-1 |
| Proposition C.1 (refinement monotonicity, neutral splits, and the exact domination gap) | — | Proposition 6 | Appendix C.3 | direct corollary; DS11-2 |
| Theorem 5 (population stationarity is efficient-Voronoi geometry) | — | Theorem 7 | §5.3 | adaptation; DS12-1 |
| Proposition 6 (exact profiled leverage bound at exchange-stable states) | — | Proposition 8 | §5.4 | apparently new; DS13-1 |
| Theorem 7 (conditional finite-to-population bridge under (M1)–(M5)) | — | Theorem 9 | §5.5 | adaptation; DS14-1 |
| Theorem 8 (margins dichotomy at global finite D_s optima (d_\psi=d_\lambda=1)) | — | Theorem 10 | §5.6 | apparently new; DS15-1 |
| Proposition C.2 (exact empirical sandwich and bracket limits) | — | Proposition 11 | Appendix C.7 | direct corollary; DS15-3 |
| Proposition C.3 (achievability by swap steering) | — | Proposition 12 | Appendix C.7 | unresolved; DS15-2 |
| Theorem 9 (margin price, value funnel, and floor) | — | Theorem 13 | §5.7 | apparently new; DS16-1 |
| Lemma C.4 (tilt-residual identity and the fixed-point gate) | — | Lemma 14 | Appendix C.9 | direct corollary; DS17-3 |
| Theorem 10 (conditional centering empties the margin-certified branch) | — | Theorem 15 | §5.8 | apparently new; DS17-1 |
| Proposition C.5 (merged branch on linear-conditional-mean laws) | — | Proposition 16 | Appendix C.9 | known; DS17-2 |
| Theorem 11 (exact off-class global basin and empirical transfer through global optima) | — | Theorem 17 | §5.9 | adaptation; DS18-1 |
| Theorem 12 (valid two-sided brackets and exact saddle closure) | — | Theorem 18 | §6.1 | adaptation; DS19-1 |
| Proposition D.1 (\Delta-consistency of the \beta=0 interval programme on the off-class law of Theorem 11) | — | Proposition 19 | Appendix D.3 | direct corollary; DS19-3 |
| Proposition D.2 (what is polynomial) | — | Proposition 20 | Appendix D.4 | direct corollary; DS19-5 |
| Proposition E.1 (exact A move oracle and finite termination) | — | Proposition 21 | Appendix E.3 | direct corollary; A1-1 |
| Proposition E.2 (tangent screening for A) | — | Proposition 22 | Appendix E.3 | direct corollary; A3-1 |
| Proposition F.1 (restricted-class empirical consistency) | Proposition 5 | Proposition 23 | Appendix F.5 | adaptation; V8-40 |

Equations are numbered per section or appendix in v9 and only when referenced;
appendices restate what they need under their own tag. The first-draft v8→v9
equation map is historical (git history).
Fixtures are cited in the main text as "fixture G\(n\)" and resolved by
Appendix G; Appendix H places every ledger row.

### v8 crosswalk (historical)

| Manuscript v8 | Registry claims | Notes |
|---|---|---|
| Proposition 1 (common-metric stationary partition is affine / Mahalanobis Voronoi) | GENERAL-FIRST-VARIATION, D-POP-VORONOI | population level |
| Lemma 2 (leverage bound) | D-LEVERAGE | |
| Theorem 3 (finite D exchange stability ⇒ self-consistent Voronoi) | D-EXCHANGE-IMPLIES-VORONOI, D-EXCHANGE-VIOLATION-LOWER-BOUND, D-FINITE-INDUCTIVE-CLOSURE | audited 26 Aug 2026 (`AUDITS/AUDIT-D-EXCHANGE-VORONOI-001.md`); registry statement adds merged-atom, exact-K, and zero-tolerance hypotheses plus boundary counterexample `CE-D-UNMERGED-DUPLICATES-001` — the manuscript does **not** yet carry these |
| Proposition 4 (approximate finite efficient-Voronoi bound for \(D_s\)) | DS-OKN-BOUND | called "Prop. 17" in the old v5 numbering |
| Proposition 5 (restricted-class empirical consistency) | CONSISTENCY-RESTRICTED-AFFINE | |
| *(absent)* | DS-PROFILED-VARIATIONAL (DS11), OPEN-DS-POP-COMMON-METRIC (DS12), DS-EXCHANGE-LEVERAGE-BOUND (DS13), OPEN-DS-FINITE-POP-BRIDGE (DS14) | audited 28 Aug 2026 (`AUDITS/AUDIT-DS-POPULATION-BRIDGE-001.md`); no manuscript counterpart yet |

## Current state

v9 (3 September 2026) is the latest snapshot and was audited independently against the novelty
ledger on 4 September 2026: 90 statements confirmed, 11 revised, 2 disputed, none claiming
novelty for a known result. Applied revisions and the remaining debt are recorded in
`WORK/completed/MANUSCRIPT-V9-AUDIT.md`. Two bibliographic items stay unverified (a Haynsworth
1968 attribution key; the Jakubowski 2021 volume details). Results proved after 3 September 2026
are in the registry only; the next revision is written when a publication decision requires it.
Registry-only since then: O6 (`RETENTION-PLUGIN-CLT-FROZEN-SCALAR`, 5 September 2026) — a
delta-method error bar for true scalar retention under a frozen rule; a v10 would cite it beside
the door3 surrogate-gap discussion, as a bridge, not as novelty. Independently audited 5 September
2026 (`AUDITS/AUDIT-SCORE-ORACLE-ROBUSTNESS-001.md`): verified; the \(\sigma^2=0\) characterisation
is corrected at \(\eta=0\) (no two-atom restriction there, atomless cells do not rescue (A4)) — any
v10 text must carry the hardened statement; no re-attribution to prior art.
Registry-only, 6 September 2026: O7 (`RETENTION-PLUGIN-CLT-FROZEN-VECTOR`, not yet audited) — the
vector companion of O6 for the geometric-mean D-retention the library reports, with the matrix
influence function, the ellipsoid characterisation of \(\sigma^2=0\)
(`CE-O7-ELLIPSOID-ZERO-VARIANCE-001`) and the singular-endpoint rate
(`RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE`). A v10 would cite it where the paper reports
`geometric_mean_retention` on held-out data, as a bridge from the delta method and the published
canonical-correlation influence functions, never as novelty; the measured heavy-tail
under-coverage (O7.7) belongs in any such text as a caveat.
The dated staleness log that preceded this section is `../archive/manuscript-staleness-log.md`.
