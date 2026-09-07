import Link from "@docusaurus/Link";
import type {Tradition} from "../../data/atlas";
import {useRegisteredAnchors} from "../anchors";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps, Core} from "../core";
import {authorHref, paperHref} from "../core";
import {ClaimLink, Html} from "../Entity";
import {useAtlasQuery} from "../useAtlasQuery";
import {YearStrip} from "../YearStrip";
import type {RenderedAnnotation, RenderedPaper} from "./PaperPage";
import {AnnotationFields} from "./PaperPage";

export interface RenderedTradition extends Omit<Tradition, "notes"> {
  notes: RenderedAnnotation[];
}
export interface PriorArt {
  id: string;
  sources: {html: string | null; name: string}[];
}
export interface LiteratureData {
  papers: Record<string, RenderedPaper>;
  priorArt: PriorArt[];
  traditions: RenderedTradition[];
}

/** Bibliographic context first; detailed research records remain one step away. */
export default function LiteraturePage({core, data}: AtlasPageProps<LiteratureData>): React.JSX.Element {
  useRegisteredAnchors([...data.traditions.map((t) => `tradition-${t.slug}`), "tradition-other"]);
  const {params, update} = useAtlasQuery();
  const query = params.get("q") ?? "";
  const author = params.get("author") ?? "";
  const match = (paper: RenderedPaper): boolean =>
    (!author || paper.authors.some((a) => a.slug === author)) &&
    `${paper.title} ${paper.authors.map((a) => a.name).join(" ")} ${paper.year ?? ""} ${paper.note ?? ""} ${paper.annotation?.prose ?? ""}`
      .toLowerCase()
      .includes(query.toLowerCase());
  const placed = new Set(data.traditions.flatMap((t) => t.papers));
  const groups = [...data.traditions, {slug: "other", label: "Other sources", notes: [], papers: Object.keys(data.papers).filter((k) => !placed.has(k))}];
  const count = Object.values(data.papers).filter(match).length;
  return (
    <AtlasShell wide title="Literature" description="Publications, researchers, and the ideas behind information-preserving quantization.">
      <header className="research-opening">
        <h1>Literature</h1>
        <p className="atlas__lead">
          Find the publication behind a result. Each entry says what the paper establishes, how this project uses it, and what it leaves unsettled.
        </p>
      </header>
      <div className="research-filters">
        <label>
          Search publications
          <input type="search" value={query} onChange={(e) => update({q: e.target.value}, true)} placeholder="Title, author, year, or topic" />
        </label>
        <label>
          Browse by author
          <select value={author} onChange={(e) => update({author: e.target.value})}>
            <option value="">All authors</option>
            {Object.values(core.authors)
              .sort((a, b) => a.name.localeCompare(b.name))
              .map((a) => (
                <option key={a.slug} value={a.slug}>
                  {a.name}
                </option>
              ))}
          </select>
        </label>
      </div>
      {author && core.authors[author] ? (
        <p>
          <Link to={authorHref(author)}>View {core.authors[author].name}’s publications and related results →</Link>
        </p>
      ) : null}
      {query || author ? (
        <p role="status">
          {count ? `${count} matching publications.` : "No publications match this selection."}{" "}
          <button type="button" onClick={() => update({q: "", author: ""})}>
            Clear filters
          </button>
        </p>
      ) : null}
      <nav className="research-theme-links" aria-label="Literature traditions">
        {groups
          .filter((g) => g.papers.some((k) => data.papers[k] && match(data.papers[k])))
          .map((t) => (
            <a key={t.slug} href={`#tradition-${t.slug}`}>
              {t.label}
            </a>
          ))}
      </nav>
      <details className="research-disclosure">
        <summary>View the publication timeline</summary>
        <YearStrip core={core} />
      </details>
      {groups.map((tradition) => {
        const papers = tradition.papers.map((k) => data.papers[k]).filter((p): p is RenderedPaper => !!p && match(p));
        return (
          <section hidden={papers.length === 0} className="literature-tradition atlas__section" key={tradition.slug} id={`tradition-${tradition.slug}`}>
            <h2>{tradition.label}</h2>
            {papers.map((p) => (
              <Entry key={p.key} core={core} paper={p} />
            ))}
            {tradition.notes.length > 0 ? (
              <details className="research-disclosure">
                <summary>Context for this tradition</summary>
                {tradition.notes.map((n) => (
                  <div key={n.heading}>
                    <h3>{n.heading}</h3>
                    <AnnotationFields fields={n.fieldsHtml} />
                    <Html html={n.proseHtml} />
                  </div>
                ))}
              </details>
            ) : null}
          </section>
        );
      })}
      <details className="research-disclosure" id="closest-prior-work">
        <summary>Prior-art search records</summary>
        <p>These records cover the atlas independently of the home page’s selected results. A search gap records what was checked, not a priority claim.</p>
        {data.priorArt.map((entry) => (
          <section key={entry.id}>
            <h3>
              <ClaimLink core={core} id={entry.id} showId={false} />
            </h3>
            {entry.sources.map((source, i) => (
              <div key={`${source.name}-${i}`}>
                <strong>{source.name}</strong>
                <Html html={source.html} />
              </div>
            ))}
          </section>
        ))}
      </details>
    </AtlasShell>
  );
}

function Entry({paper}: {core: Core; paper: RenderedPaper}): React.JSX.Element {
  const fields = paper.annotation?.fieldsHtml;
  const summary = fields ? (fields["Why it matters here"] ?? fields["Why important"] ?? fields["Result/use"] ?? fields.Result ?? fields.Use) : null;
  return (
    <article className="research-paper-row">
      <h3>
        <Link to={paperHref(paper.slug)}>{paper.title}</Link>
      </h3>
      <p className="research-attribution">
        {paper.authors.map((a, i) => (
          <span key={a.slug}>
            {i ? ", " : ""}
            <Link to={authorHref(a.slug)}>{a.name}</Link>
          </span>
        ))}
        {paper.year ? ` · ${paper.year}` : ""}
      </p>
      {summary ? (
        <Html html={summary} />
      ) : (
        <p className="atlas__muted">
          {paper.citedBy.length ? "Cited in the research record; follow the publication for its related results." : "Background reading in this tradition."}
        </p>
      )}
    </article>
  );
}
