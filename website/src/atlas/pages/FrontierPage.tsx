import Link from "@docusaurus/Link";
import {useMemo} from "react";

import type {Theme} from "../../data/atlas";
import {useRegisteredAnchors} from "../anchors";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps, Core, CoreClaim} from "../core";
import {claimHref} from "../core";
import {EntityList, Html} from "../Entity";
import type {Adjacency} from "../graph";
import {adjacency, boundary, raisedBy, restsOn} from "../graph";

export interface FrontierQuestion extends CoreClaim {
  noteHtml: string | null;
  statementHtml: string | null;
}

export interface FrontierData {
  themes: (Omit<Theme, "claims"> & {claims: FrontierQuestion[]})[];
}

/** What an answer would have to get past: exact fixtures and settled counterexamples. */
function excluded(core: Core, adj: Adjacency, id: string): string[] {
  const edge = boundary(adj, id);
  const counterexamples = restsOn(adj, id).filter((other) => core.claims[other]?.kind === "counterexample");
  return [...edge.bounded, ...edge.refuted, ...counterexamples];
}

/** The settled results the question stands next to, counterexamples aside. */
function settled(core: Core, adj: Adjacency, id: string): string[] {
  return restsOn(adj, id).filter((other) => core.claims[other]?.kind !== "counterexample");
}

function Question({adj, core, question}: {adj: Adjacency; core: Core; question: FrontierQuestion}): React.JSX.Element {
  const alreadyExcluded = excluded(core, adj, question.id);
  const nextTo = settled(core, adj, question.id);
  const wouldUnlock = raisedBy(adj, question.id);
  return (
    <section aria-labelledby={question.slug} className="frontier__question">
      <h3 id={question.slug}>
        <Link to={claimHref(question.slug)}>{question.editorial?.title ?? question.title}</Link>
      </h3>
      {question.parked ? (
        <p className="frontier__parked">
          <span className="provenance-band__mark">parked</span> set aside until an explicit decision reopens it.
        </p>
      ) : null}
      {question.editorial ? <p>{question.editorial.summary}</p> : null}
      <details className="research-disclosure">
        <summary>Statement and research context</summary>
        <Html className="statement statement--question" html={question.statementHtml} />
        <Html html={question.noteHtml} />
        <div className="frontier__lists">
          {alreadyExcluded.length > 0 ? (
            <div>
              <h4>Already excluded</h4>
              <EntityList compact core={core} ids={alreadyExcluded} />
            </div>
          ) : null}
          {nextTo.length > 0 ? (
            <div>
              <h4>Settled next to it</h4>
              <EntityList compact core={core} ids={nextTo} />
            </div>
          ) : null}
          {wouldUnlock.length > 0 ? (
            <div>
              <h4>Raised by</h4>
              <EntityList compact core={core} ids={wouldUnlock} />
            </div>
          ) : null}
        </div>
      </details>
    </section>
  );
}

/**
 * `/research/frontier/`: the open questions, by theme. Each one is stated in
 * full and surrounded by what already bounds it, what is settled beside it,
 * and the results an answer would extend.
 */
export default function FrontierPage({core, data}: AtlasPageProps<FrontierData>): React.JSX.Element {
  const adj = useMemo(() => adjacency(core.edges), [core.edges]);
  const rank = (id: string): number => {
    const i = core.frontier.indexOf(id);
    return i < 0 ? core.frontier.length : i;
  };
  const themes = data.themes
    .filter((theme) => theme.claims.length > 0)
    .map((theme) => ({
      ...theme,
      claims: [...theme.claims].sort((a, b) => rank(a.id) - rank(b.id) || (a.editorial?.title ?? a.title).localeCompare(b.editorial?.title ?? b.title)),
    }));
  useRegisteredAnchors(themes.flatMap((theme) => [theme.slug, ...theme.claims.map((question) => question.slug)]));

  return (
    <AtlasShell
      description="The open questions of the research record, grouped by theme, each with what bounds it, what is settled beside it and what an answer would extend."
      title="The frontier"
    >
      <h1>The frontier</h1>
      <p className="atlas__lead">
        Questions that remain unresolved in this research record. Open a question for its precise statement, settled neighbours, and known obstructions.
      </p>
      <nav className="research-theme-links" aria-label="Frontier themes">
        {themes.map((theme) => (
          <a key={theme.slug} href={`#${theme.slug}`}>
            {theme.label}
          </a>
        ))}
      </nav>

      {themes.map((theme) => (
        <section key={theme.slug} aria-labelledby={theme.slug} className="atlas__section">
          <h2 id={theme.slug}>{theme.label}</h2>
          <p>{theme.summary}</p>
          {theme.claims.map((question) => (
            <Question key={question.id} adj={adj} core={core} question={question} />
          ))}
        </section>
      ))}
    </AtlasShell>
  );
}
