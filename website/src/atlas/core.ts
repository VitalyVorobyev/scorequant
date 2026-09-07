/**
 * The graph skeleton every atlas page receives as its `core` prop.
 *
 * Built by `plugins/research-atlas/index.mjs` from `src/generated/atlas.json`
 * (typed in full by `src/data/atlas.ts`). It carries what a page needs to
 * draw glyphs, links and local graphs for any entity, and nothing rendered:
 * statements are plain text here, and proofs, notes and annotations travel
 * only in the route-specific `data` module.
 */
import type {
  Author,
  Chapter,
  CriterionEntry,
  Edge,
  EntityKind,
  Layout,
  LevelEntry,
  Provenance,
  Strip,
  Theme,
  VocabularyEntry
} from "../data/atlas";

export interface CoreClaim {
  audited: boolean;
  chapter: {file: string; label: string; section: string | null; slug: string} | null;
  criterion: string[];
  enforcedBy: string[];
  id: string;
  implementedBy: string[];
  kind: EntityKind;
  level: string;
  literature: string[];
  machineChecked: boolean;
  parked: boolean;
  provenance: Provenance;
  searchStatus: string | null;
  slug: string;
  statement: string;
  strip: string | null;
  theme: string | null;
  title: string;
}

export interface CoreFixture {
  K: number;
  citedBy: {claim: string; type: "refuted_by" | "bounded_by"}[];
  criterion: string[];
  dimension: number;
  falsifies: string;
  id: string;
  level: string;
  refusal: boolean;
  slug: string;
}

export interface CorePaper {
  authors: {name: string; slug: string}[];
  citedBy: string[];
  doi: string | null;
  key: string;
  relevantTo: string[];
  slug: string;
  title: string;
  tradition: string | null;
  year: number | null;
}

export interface Core {
  authors: Record<string, Author>;
  chapters: Chapter[];
  claims: Record<string, CoreClaim>;
  edges: Edge[];
  fixtures: Record<string, CoreFixture>;
  frontier: string[];
  headline: string[];
  layout: Layout;
  papers: Record<string, CorePaper>;
  schemaVersion: number;
  strips: Strip[];
  themes: Theme[];
  traditions: {label: string; papers: string[]; slug: string}[];
  vocabulary: {
    criteria: CriterionEntry[];
    edgeTypes: VocabularyEntry[];
    lanes: VocabularyEntry[];
    levels: LevelEntry[];
    provenance: VocabularyEntry[];
  };
}

/** Props every atlas page receives from the plugin's `modules`. */
export interface AtlasPageProps<Data> {
  core: Core;
  data: Data;
}

export function claimHref(slug: string): string {
  return `/research/claims/${slug}/`;
}

export function fixtureHref(slug: string): string {
  return `/research/counterexamples/${slug}/`;
}

export function paperHref(slug: string): string {
  return `/research/literature/${slug}/`;
}

export function authorHref(slug: string): string {
  return `/research/authors/${slug}/`;
}

/** The href of any entity id known to the core, or null. */
export function entityHref(core: Core, id: string): string | null {
  const claim = core.claims[id];
  if (claim) return claimHref(claim.slug);
  const fixture = core.fixtures[id];
  if (fixture) return fixtureHref(fixture.slug);
  const paper = core.papers[id];
  if (paper) return paperHref(paper.slug);
  return null;
}

export function levelLabel(core: Core, level: string): string {
  return core.vocabulary.levels.find((entry) => entry.id === level)?.label ?? level;
}

export function criterionLabel(core: Core, criterion: string): string {
  return core.vocabulary.criteria.find((entry) => entry.id === criterion)?.label ?? criterion;
}

export function provenanceLabel(core: Core, provenance: Provenance): string {
  return core.vocabulary.provenance.find((entry) => entry.id === provenance)?.label ?? provenance;
}
