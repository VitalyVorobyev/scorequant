# Manuscripts — lagging publication snapshot

The registry (`claims/`, `KNOWN_RESULTS/`, `COUNTEREXAMPLES/`) is always ahead of the
manuscript. A manuscript statement is never more current than a claim node. Open the article
body only in a manuscript-revision task.

| File | What it is |
|---|---|
| `score_space_quantization_article_v10.md` / `.html` | "Information-optimal hard quantization of multivariate score space", v10, 10 September 2026: main text §1–§11, Appendices A–I; editorial revision 11 September 2026. §8 and Appendix I carry the frozen-rule results O6–O8; §10.2 is the future-work section that absorbs the research backlog. Appendix G resolves every fixture id (G1–G32); Appendix H places every ledger row. |
| `NOVELTY_LEDGER.md` | One row per central statement with novelty label, attribution and claim ids (version 1.1: §1–§4 wrote v9, §5 wrote v10's §8). |
| `figures/` | The six figure assets; never inlined (`registry.py validate` rejects `data:` payloads). |

Render the HTML sibling with

```bash
uv run agenticresearch/py/render_manuscript.py agenticresearch/manuscripts/score_space_quantization_article_v10.md
```

(python-markdown declared inline; MathJax from its CDN; `\(…\)` and `\[…\]` math; result boxes
as `<div class="theorem" markdown="1">`; `[novelty: …; ledger …]` tags render as hidden
provenance marks).

## State

v10 matches the registry as of 10 September 2026 (closure step 5). Every result proved after
3 September 2026 — O6, O7 with `RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE` and its two boundary
fixtures, O8 with its six audit fixtures — is in §8 and Appendix I in its audited, hardened form;
the two bibliography items the v9 audit left unverified (Haynsworth 1968, Jakubowski 2021) are
confirmed with DOIs. Pending: the owner's review of v10; the two v9 statements the 4 September
audit recorded as disputed remain as v9 carried them; Hsu (1941) and Glynn & Muirhead (1978)
are cited for the endpoint rate from secondary sources and their primary texts are unread
(`LITERATURE/audits/NOVELTY-STEP5-10-September-2026.md`). A result proved after this date
enters a v11 only through a new closure-programme decision; until then it is listed here.

Earlier snapshots (v8, v9, the HEP companion, the landscape survey) are in git history.

Review disposition (11 September 2026, phase H): the historical packet at commit
`6a4c7b3`, `agenticresearch/WORK/completed/MANUSCRIPT-V9-AUDIT.md`, records two disputed
rows only as a count; it does not preserve their row-level identities or resolutions. Do not
interpret that absence as approval. Publication remains blocked on recovering those verdicts or
re-auditing the affected ledger scope. Explicit remaining attribution items in that packet are
V8-10 (regression leverage), V8-30 (minimum-eigenvalue superdifferential), and read/annotation
provenance for Hartigan/Haynsworth; the later DOI verification does not itself constitute the
missing source read. Retain the stated secondary-source qualifications for Hsu/Glynn–Muirhead.
The positive-tolerance library contract was independently narrowed on 11 September
(`AUDITS/AUDIT-D-COMPILE-TOLERANCE-001.md`): admissible individual moves only, with a separate
singleton-disagreement guard and no simultaneous-change bound. The positive-tolerance sentence following (4.6) now states these qualifications; exact-zero
theorems and existing formal markers are unchanged.

## Editorial revision — 11 September 2026

The v10 article has been rewritten for applied researchers. The main text explains the
information identity, assignment-to-rule result, nuisance limitations, and evaluation. Detailed
margin statements now accompany their proofs in Appendix C, the finite bracket in Appendix D,
and score-error and exceptional sampling results in Appendix I. Appendix A.6 retains detailed
prior-work comparisons; Appendix F.6 retains the full open-question list. Result identifiers,
equation tags, claim IDs, novelty labels, and bibliography numbering are preserved.

Two discrepancies found during the editorial fidelity pass require research disposition before
publication; this edit does not modify the registry or constitute an independent scientific audit:

- `A-FINITE-GEOMETRY-FAILS` and its A2 proof text infer a partition-level failure from
  `CE-A-DSTYLE-001`, which exhibits only a move-level failure. The manuscript now states only
  what that fixture establishes. Acceptance requires an independent audit of the stronger claim,
  followed by evidence or a scoped registry correction.
- `RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE` describes the rate at `d = 2, r = 1` as slower
  than the square-root scale, although its own formula gives equality. The manuscript distinguishes
  `d = 2` from `d > 2`; acceptance requires correcting the registry's explanatory wording through
  the research workflow. The rate formula and proposition are unchanged.

The existing unidentified v9 disputes and attribution/source-read qualifications remain pending.
The rewrite corrects stale closure and score-error summaries without declaring research frozen,
adding a capability, or satisfying owner/publication acceptance.
