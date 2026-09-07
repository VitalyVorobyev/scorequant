import Link from "@docusaurus/Link";

import type {Annotation, Paper} from "../../data/atlas";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps, Core} from "../core";
import {authorHref} from "../core";
import {EntityList, Html} from "../Entity";

/** An annotation whose Markdown fields and prose were rendered at build time. */
export interface RenderedAnnotation extends Annotation {
  fieldsHtml: Record<string, string>;
  proseHtml: string | null;
}

/** A bibliography entry with its annotation and note rendered. */
export interface RenderedPaper extends Omit<Paper, "annotation"> {
  annotation: RenderedAnnotation | null;
  noteHtml: string | null;
}

/**
 * The order an annotation is read in: what the source establishes, why that
 * matters here, what this work takes from it, and where it stops. Fields the
 * registry uses under other names fall through in their own order after these.
 */
const FIELD_ORDER = [
  "Result",
  "Result/use",
  "Why important",
  "Why it matters here",
  "Use",
  "Does not solve",
  "Does not transfer",
  "Caution"
];

/** The fields of an annotation, priority order first, then the rest. */
export function orderedFields(fields: Record<string, string>, drop: readonly string[] = []): [string, string][] {
  const entries = Object.entries(fields).filter(([name]) => !drop.includes(name));
  const rank = (name: string): number => {
    const index = FIELD_ORDER.indexOf(name);
    return index === -1 ? FIELD_ORDER.length : index;
  };
  return entries
    .map((entry, index) => ({entry, index}))
    .sort((a, b) => rank(a.entry[0]) - rank(b.entry[0]) || a.index - b.index)
    .map(({entry}) => entry);
}

/** An annotation's fields as a compact definition list. */
export function AnnotationFields({fields, drop}: {drop?: readonly string[]; fields: Record<string, string>}): React.JSX.Element | null {
  const entries = orderedFields(fields, drop);
  if (entries.length === 0) return null;
  return (
    <dl className="annotation-fields">
      {entries.map(([name, html]) => (
        <div key={name}>
          <dt>{name}</dt>
          <dd>
            <Html html={html} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** The literature page anchor of a tradition. */
export function traditionHref(slug: string): string {
  return `/research/literature/#tradition-${slug}`;
}

function traditionLabel(core: Core, slug: string): string {
  return core.traditions.find((tradition) => tradition.slug === slug)?.label ?? slug;
}

/**
 * One bibliography entry: what the source says, what this work took from it,
 * and which claims cite it.
 */
export default function PaperPage({core, data}: AtlasPageProps<RenderedPaper>): React.JSX.Element {
  const alsoRelevant = data.relevantTo.filter((id) => !data.citedBy.includes(id));
  return (
    <AtlasShell title={data.title} description={`${data.title}: annotation, and the claims that cite it.`}>
      <h1>{data.title}</h1>
      <p className="paper-meta">
        <span className="paper-meta__authors">
          {data.authors.map((author, index) => (
            <span key={author.slug}>
              {index > 0 ? ", " : ""}
              <Link to={authorHref(author.slug)}>{author.name}</Link>
            </span>
          ))}
        </span>
        {data.year === null ? null : <span>{data.year}</span>}
        {data.tradition === null ? null : (
          <span>
            <Link to={traditionHref(data.tradition)}>{traditionLabel(core, data.tradition)}</Link>
          </span>
        )}
      </p>
      {data.doi === null && data.arxiv === null && data.pdf === null ? null : (
        <p className="paper-links">
          {data.doi === null ? null : (
            <a href={`https://doi.org/${data.doi}`} rel="noopener noreferrer">
              doi:{data.doi}
            </a>
          )}
          {data.arxiv === null ? null : (
            <a href={`https://arxiv.org/abs/${data.arxiv}`} rel="noopener noreferrer">
              arXiv:{data.arxiv}
            </a>
          )}
          {data.pdf === null ? null : (
            <a href={data.pdf} rel="noopener noreferrer">
              PDF
            </a>
          )}
        </p>
      )}

      {data.annotation === null ? null : (
        <section className="paper-annotation">
          <h2>{data.annotation.heading}</h2>
          <AnnotationFields fields={data.annotation.fieldsHtml} drop={["Paper"]} />
          <Html html={data.annotation.proseHtml} />
        </section>
      )}

      {data.noteHtml === null ? null : <Html html={data.noteHtml} />}

      <section className="atlas__section">
        <h2>Cited by</h2>
        <EntityList core={core} ids={data.citedBy} empty="No claim cites this source yet." />
      </section>

      {alsoRelevant.length === 0 ? null : (
        <section className="atlas__section">
          <h2>Relevant to</h2>
          <p className="atlas__muted">Claims this source bears on without being cited in their proofs.</p>
          <EntityList core={core} ids={alsoRelevant} />
        </section>
      )}
    </AtlasShell>
  );
}
