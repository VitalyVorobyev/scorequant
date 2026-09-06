# Manuscripts — lagging publication snapshot

The registry (`claims/`, `KNOWN_RESULTS/`, `COUNTEREXAMPLES/`) is always ahead of the
manuscript. A manuscript statement is never more current than a claim node. Open the article
body only in a manuscript-revision task (closure step 5 in `OPEN_PROBLEMS.md`).

| File | What it is |
|---|---|
| `score_space_quantization_article_v9.md` / `.html` | "Information-optimal hard quantization of multivariate score space", v9, 3 September 2026: main text §1–§10, Appendices A–H. Appendix G resolves every fixture id; Appendix H places every ledger row. |
| `NOVELTY_LEDGER.md` | One row per central statement with novelty label, attribution and claim ids. The v9 text was written from it. |
| `figures/` | The six figure assets; never inlined (`registry.py validate` rejects `data:` payloads). |

Render the HTML sibling with

```bash
uv run agenticresearch/py/render_manuscript.py agenticresearch/manuscripts/score_space_quantization_article_v9.md
```

(python-markdown declared inline; MathJax from its CDN; `\(…\)` and `\[…\]` math; result boxes
as `<div class="theorem" markdown="1">`; `[novelty: …; ledger …]` tags render as hidden
provenance marks).

## State

v9 was audited against the novelty ledger on 4 September 2026 (90 statements confirmed, 11
revised, 2 disputed, none claiming novelty for a known result; two bibliographic items
unverified: a Haynsworth 1968 key and the Jakubowski 2021 volume). Results proved after
3 September 2026 exist in the registry only and enter a v10 at closure step 5:

- O6 `RETENTION-PLUGIN-CLT-FROZEN-SCALAR` (audited 5 September 2026; the \(\sigma^2=0\)
  characterisation was hardened at \(\eta=0\) — carry the hardened statement).
- O7 `RETENTION-PLUGIN-CLT-FROZEN-VECTOR` with `RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE` and
  `CE-O7-ELLIPSOID-ZERO-VARIANCE-001` (audit pending). Cite as a bridge from the delta method
  and the published canonical-correlation influence functions, never as novelty; the measured
  heavy-tail under-coverage is a caveat in any such text.

Earlier snapshots (v8, the HEP companion, the landscape survey, the dated staleness log) are in
git history.
