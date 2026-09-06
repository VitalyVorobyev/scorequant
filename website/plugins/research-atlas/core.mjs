/**
 * The graph skeleton every atlas page shares (`core` prop), built from the
 * full atlas document. Kept apart from the plugin so that component tests can
 * build the same `core` from `src/generated/atlas.json` without Docusaurus.
 * The shape is typed by `src/atlas/core.ts`.
 */

export function coreClaim(claim) {
  return {
    id: claim.id,
    slug: claim.slug,
    kind: claim.kind,
    title: claim.title,
    statement: claim.statement,
    provenance: claim.provenance,
    criterion: claim.criterion,
    level: claim.level,
    theme: claim.theme,
    strip: claim.strip,
    parked: claim.parked,
    searchStatus: claim.searchStatus,
    machineChecked: claim.machineChecked !== null,
    audited: claim.audit !== null,
    chapter: claim.chapter,
    literature: claim.literature,
    implementedBy: claim.implementedBy,
    enforcedBy: claim.enforcedBy
  };
}

export function coreFixture(fixture) {
  return {
    id: fixture.id,
    slug: fixture.slug,
    criterion: fixture.criterion,
    level: fixture.level,
    dimension: fixture.dimension,
    K: fixture.K,
    falsifies: fixture.falsifies,
    citedBy: fixture.citedBy,
    refusal: fixture.refusal !== null
  };
}

export function corePaper(paper) {
  return {
    key: paper.key,
    slug: paper.slug,
    title: paper.title,
    year: paper.year,
    authors: paper.authors,
    tradition: paper.tradition,
    citedBy: paper.citedBy,
    relevantTo: paper.relevantTo,
    doi: paper.doi
  };
}

export function buildCore(atlas) {
  return {
    schemaVersion: atlas.schemaVersion,
    vocabulary: atlas.vocabulary,
    headline: atlas.headline,
    frontier: atlas.frontier,
    claims: Object.fromEntries(Object.values(atlas.claims).map((c) => [c.id, coreClaim(c)])),
    fixtures: Object.fromEntries(Object.values(atlas.fixtures).map((f) => [f.id, coreFixture(f)])),
    papers: Object.fromEntries(Object.values(atlas.papers).map((p) => [p.key, corePaper(p)])),
    authors: atlas.authors,
    traditions: atlas.traditions.map((t) => ({slug: t.slug, label: t.label, papers: t.papers})),
    chapters: atlas.chapters,
    themes: atlas.themes,
    strips: atlas.strips,
    edges: atlas.edges,
    layout: atlas.layout
  };
}
