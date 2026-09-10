# Protocol: literature search

Two distinct searches exist; never conflate them.

1. **Field coverage** — "have we found essentially all important papers around
   the problem?" Converges via citation saturation (below).
2. **Novelty search** — "has anybody proved this exact theorem, perhaps in
   another formulation?" Run only after the theorem statement is frozen, and
   repeat claim-by-claim near publication. An empty result is recorded as
   `literature_search_status: "search_gap"`; a search gap is not novelty.

## Per-theorem triangulation (the minimum, every investigation)

Find 3–5 nearest sources and state for each:

- exact problem;
- exact result;
- objective;
- feasible set;
- what transfers;
- what does not.

Record the table in `LITERATURE/audits/<CLAIM-ID>-<date>.md`, naming the claim
id (see the 26 Aug 2026 D-EXCHANGE-IMPLIES-VORONOI audit subsection as the
exemplar).

## Terminology fan-out

Our problem lives on a terminology island. For every search, generate
alternative mathematical descriptions; the standing vocabulary:

D-optimal quantization, \(D_s\) quantization, score-function quantization,
conditional-score Fisher information, determinant partition/exchange,
minimum-determinant clustering, Hartigan exchange, Mahalanobis/CVT/Bregman
quantization, communication-constrained estimation, inference-aware
categorization, HEP template binning, determinant criterion clustering,
between-cluster scatter determinant, finite-alphabet estimation,
information-preserving partitioning.

## Field coverage: bidirectional citation snowballing

- Seed from the anchor papers in `LITERATURE/seeds.md`.
- For each paper follow references (backward) and citing papers (forward);
  recursively inspect anything mathematically relevant.
- Maintain separate graphs per community (Fisher-information quantization,
  determinant clustering, optimal design, HEP inference-aware binning, …) and
  watch for them connecting.
- **Stopping rule — citation saturation:** stop when a full traversal round
  yields overwhelmingly duplicates, tangential works, or applications with no
  new relevant theorem. Record the per-round counts (candidates / relevant) so
  the coverage claim is auditable.
- Supplement with concept search (the fan-out above) and author/venue search:
  terminology islands, old monographs, independent rediscoveries, and very
  recent preprints all evade pure citation traversal.

## The literature artifact

Discovery state lives in `LITERATURE/`, not in agent context:

- `seeds.md` — anchor papers;
- `graph.json` — paper records: id, title, year, authors, source_of_discovery,
  cites_relevant, cited_by_relevant, research_area, **relevant_claims (claim
  ids)**, status (unread / screened / deeply_reviewed / irrelevant);
- `reviewed.md`, `rejected.md` — human-readable outcomes with reasons;
- `gaps.md` — communities or periods suspected under-covered.

Every paper must be linked to claim ids, never merely labelled "relevant to
ScoreQuant". Curated theorem-level annotations stay in `LITERATURE/topics/`, and
every bibliography key needs a `**Key:**` line under its annotating heading so
`registry.json` resolves to it (checked by `py/registry.py validate`).
