import Link from "@docusaurus/Link";
import type {ReactNode} from "react";

import type {Core} from "./core";
import {claimHref, fixtureHref, paperHref, provenanceLabel} from "./core";
import {Glyph} from "./Glyph";

/** Rendered HTML from the build-time Markdown step. */
export function Html({html, className, as: Tag = "div"}: {html: string | null; className?: string; as?: "div" | "span" | "p"}): React.JSX.Element | null {
  if (html === null) return null;
  return <Tag className={className} dangerouslySetInnerHTML={{__html: html}} />;
}

/** A claim named by glyph, title and id, linking to its page. */
export function ClaimLink({core, id, showId = true}: {core: Core; id: string; showId?: boolean}): React.JSX.Element {
  const claim = core.claims[id];
  if (!claim) return <code>{id}</code>;
  return (
    <Link to={claimHref(claim.slug)} className="entity-link">
      <Glyph kind={claim.provenance} checked={claim.machineChecked} label={provenanceLabel(core, claim.provenance)} />
      <span className="entity-link__title">{claim.title}</span>
      {showId ? <span className="entity-link__id">{claim.id}</span> : null}
    </Link>
  );
}

/** A counterexample fixture named by glyph and id. */
export function FixtureLink({core, id}: {core: Core; id: string}): React.JSX.Element {
  const fixture = core.fixtures[id];
  if (!fixture) return <code>{id}</code>;
  return (
    <Link to={fixtureHref(fixture.slug)} className="entity-link">
      <Glyph kind="fixture" label="Counterexample fixture" />
      <span className="entity-link__id">{fixture.id}</span>
    </Link>
  );
}

/** A paper named by authors, year and title. */
export function PaperLink({core, id}: {core: Core; id: string}): React.JSX.Element {
  const paper = core.papers[id];
  if (!paper) return <code>{id}</code>;
  const who = paper.authors.map((a) => a.name).join(", ");
  return (
    <Link to={paperHref(paper.slug)} className="entity-link">
      <span className="entity-link__title">
        {who}
        {paper.year !== null ? ` (${paper.year})` : ""}
      </span>
      <span className="entity-link__id">{paper.title}</span>
    </Link>
  );
}

export interface EntityListProps {
  core: Core;
  ids: readonly string[];
  /** Rendered after the link on the same line (a depth, a type, a cell). */
  meta?: (id: string) => ReactNode;
  /** Rendered as a second line under the link. */
  note?: (id: string) => ReactNode;
  compact?: boolean;
  empty?: string;
}

/** A list of claims with their glyphs, or fixtures where the id is one. */
export function EntityList({core, ids, meta, note, compact = false, empty}: EntityListProps): React.JSX.Element | null {
  if (ids.length === 0) return empty ? <p className="atlas__muted">{empty}</p> : null;
  return (
    <ul className={compact ? "entity-list entity-list--compact" : "entity-list"}>
      {ids.map((id) => (
        <li key={id}>
          {id in core.claims ? <ClaimLink core={core} id={id} /> : id in core.fixtures ? <FixtureLink core={core} id={id} /> : <PaperLink core={core} id={id} />}
          {meta ? <span className="entity-list__meta">{meta(id)}</span> : null}
          {note ? <p className="entity-list__note">{note(id)}</p> : null}
        </li>
      ))}
    </ul>
  );
}

/** The legend naming every provenance class beside its glyph. */
export function Legend({core, fixtures = false}: {core: Core; fixtures?: boolean}): React.JSX.Element {
  return (
    <p className="atlas-legend" aria-label="Legend">
      {core.vocabulary.provenance
        .filter((entry) => entry.id !== "verification")
        .map((entry) => (
          <span key={entry.id}>
            <Glyph kind={entry.id as GlyphKind} /> {entry.label}
          </span>
        ))}
      <span>
        <Glyph kind="proved" checked /> machine-checked
      </span>
      {fixtures ? (
        <span>
          <Glyph kind="fixture" /> counterexample fixture
        </span>
      ) : null}
    </p>
  );
}

type GlyphKind = Parameters<typeof Glyph>[0]["kind"];
