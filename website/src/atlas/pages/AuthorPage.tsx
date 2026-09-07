import Link from "@docusaurus/Link";

import type {Author} from "../../data/atlas";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps} from "../core";
import {EntityList, PaperLink} from "../Entity";
import {traditionHref} from "./PaperPage";

/**
 * One author's sources in this bibliography, and the claims those sources
 * reach. An author page exists so that a reader who recognises a name can see
 * at once which parts of the record rest on that name's work.
 */
export default function AuthorPage({core, data}: AtlasPageProps<Author>): React.JSX.Element {
  const papers = data.papers
    .map((key) => core.papers[key])
    .filter((paper) => paper !== undefined)
    .sort((a, b) => (a.year ?? 0) - (b.year ?? 0));
  const claims = Array.from(new Set(papers.flatMap((paper) => paper.citedBy))).sort();
  const label = (slug: string): string => core.traditions.find((tradition) => tradition.slug === slug)?.label ?? slug;

  return (
    <AtlasShell title={data.name} description={`Sources by ${data.name} in the ScoreQuant bibliography, and the claims that cite them.`}>
      <h1>{data.name}</h1>
      <p className="atlas__lead">
        The sources this bibliography records under this name, in order of publication, and the
        claims they support.
      </p>

      <ul className="author-papers">
        {papers.map((paper) => (
          <li key={paper.key}>
            <PaperLink core={core} id={paper.key} />
            {paper.year === null ? null : <span className="literature-entry__year">{paper.year}</span>}
            {paper.tradition === null ? null : (
              <span className="entity-list__meta">
                <Link to={traditionHref(paper.tradition)}>{label(paper.tradition)}</Link>
              </span>
            )}
          </li>
        ))}
      </ul>

      <section className="atlas__section">
        <h2>Claims that cite this work</h2>
        <EntityList core={core} ids={claims} empty="No claim cites these sources yet." />
      </section>
    </AtlasShell>
  );
}
