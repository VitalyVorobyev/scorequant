# ADR 0033: The research section is an atlas generated from the registry

**Status:** Accepted

**Extends:** [ADR 0031](0031-portal-reduced-to-four-surfaces.md) (Research stays one of the four
portal surfaces) and [ADR 0030](0030-formal-verification-track.md) (what a machine-checked mark
may claim). **Keeps:** [ADR 0027](0027-landing-page-at-the-root.md) (topology: the atlas lives
under `/portal/research/`).

## Context

The research surface was ten essay pages and one flat claim record: an allowlist of 65 of the
127 registry claims projected to five fields and listed by status. The essays were good prose,
but the section read as documentation with the registry as an appendix. The owner asked for a
public interface to an active research programme: a nonlinear, explorable map whose primary
objects are research entities rather than pages, whose headline is what ScoreQuant established,
and which lets a visitor answer research questions directly. What does a theorem rest on? What
does it enable? Which prior work is closest? Where does the result stop being true? Which
neighbouring questions are open? How do D, profiled `D_s`, trace, A and E differ, and which of
their regions are established, proved here, refuted, or empty?

The registry under `agenticresearch/` already is such a graph. Claims carry `dependencies`,
`implies`, `converse_failures`, `counterexamples`, `boundary_counterexamples`, `literature`,
`proof_location`, `audit` and `formal_proof`; the counterexample fixtures are exact rational data
small enough to draw; the literature chapters annotate every paper with what it establishes and
what it does not solve; the per-theorem audits name the nearest prior sources. What was missing
was a read seam and a set of projections.

## Decision

**The registry exports; the website projects.** `agenticresearch/py/registry.py export` writes
the whole research graph as one JSON document: claims with their proof prose, fixtures, the
bibliography and its curated annotations, the prior-art audits, the proof chapters, the audit
reports, the programme titles. It applies no publication policy and derives nothing.
`website/scripts/generate_atlas.py` owns everything the public atlas adds and writes
`website/src/generated/atlas.json`, which `tests/test_atlas_data.py` keeps equal to a fresh run.

**Every claim is published unless withheld.** `website/content/research-public.json` names the
withheld ids; a published claim whose edge points at a withheld id fails the build, so the public
graph never dangles. Presence in the registry still does not, by itself, publish a claim: the
withheld list is the gate, reviewed with every change to it.

**Provenance is derived, never assigned.** Seven public classes come mechanically from
`status` and `literature_search_status`: established (`literature`), derived from the
literature (`bridge`), proved here (`project_proved`), proved here with no direct precedent found
(`project_proved` with a recorded search gap), measured, counterexample, open; audit records are
verification records. A machine-checked mark is an overlay on a class, present exactly when the
claim carries `formal_proof`, and every place that explains it says that a Lean build certifies
the statement, never the implementation. The wording for the fourth class is always "no direct
precedent found", with the nearest prior work named beside it; the words "novel" and "first"
do not appear.

**Edges are typed for a reader.** `dependencies` is *rests on*; `implies` is *enables*, or
*raises* when the target is open, or *verified by* when the target is an audit record;
`converse_failures` is *converse fails*; the fixture fields are *refuted by* and *bounded by*;
`literature` is *cites*. This is the public side of roadmap phase F, done in the projection
without changing the registry schema.

**The proof prose is public.** Each claim page carries the chapter section behind its
`proof_location`, rendered at build time with KaTeX and with every claim, fixture and workspace
path linked. Internal work-tracking vocabulary is rewritten on the way out: open-problem numbers
become links to the open claim they name, programme numbers become the public theme name, and
the words for work packets become ordinary English; the test guards the output for the tokens.

**Editorial notes are keyed by entity.** The only hand-written atlas content is one short
Markdown note per entity under `website/content/atlas/notes/`, seeded from the retired essays'
"what it buys" paragraphs, plus three small configuration files: the public vocabulary and
headline choices (`config.json`), the library map (`implementation.json`), and the withheld
list. Every pointer in them must resolve to a published entity.

**The atlas is a set of views over one document.** A home page (the problem, the results
established here as cards with local maps, the *Known → New here → Boundary → Open* strip by
theme, the frontier, the ways in); an argument map with a deterministic layered layout computed
by the generator, a focus mode and five question buttons that re-highlight rather than re-lay
out; a landscape matrix of criteria against problem levels; a frontier by theme; a literature
landscape by tradition with a year strip; a machine-checked chain; the library relation; a
legend; and one page per claim, fixture, paper and author. All routes are static pages emitted
by a local Docusaurus plugin (`website/plugins/research-atlas/`), so search, link checking and
the accessibility scan cover every entity. The ten essay pages are retired with redirects into
the views that absorbed them.

## Consequences

- The `research` docs instance is gone; `tests/test_walkthrough_facts.py` classifies no research
  root. The atlas is guarded by `tests/test_atlas_data.py` (freshness, dangling edges, pointer
  resolution, provenance derivation, internal vocabulary) and by the website's unit and
  end-to-end tests.
- Seven registry statements and a few dozen proof sections use internal vocabulary; the projection
  rewrites it, and the registry should be edited upstream as those claims are next touched.
- Anything the atlas cannot show is missing from the registry, not from the website. New
  results reach the public surface by being recorded there; new prose reaches it as a note keyed
  by the entity it explains.
