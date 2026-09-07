import type {Tradition} from "../../data/atlas";
import {useRegisteredAnchors} from "../anchors";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps, Core} from "../core";
import {ClaimLink, EntityList, Html, PaperLink} from "../Entity";
import {YearStrip} from "../YearStrip";
import type {RenderedAnnotation, RenderedPaper} from "./PaperPage";
import {AnnotationFields} from "./PaperPage";

/** A tradition whose editorial notes were rendered at build time. */
export interface RenderedTradition extends Omit<Tradition, "notes"> {
  notes: RenderedAnnotation[];
}

/** One headline claim and the nearest sources its prior-art audit named. */
export interface PriorArt {
  id: string;
  sources: {html: string | null; name: string}[];
}

export interface LiteratureData {
  papers: Record<string, RenderedPaper>;
  priorArt: PriorArt[];
  traditions: RenderedTradition[];
}

/**
 * The bibliography, arranged by the traditions it draws on.
 *
 * The page's job is attribution: for every source, what it establishes and
 * what this work took from it, so that a reader can tell an ingredient that
 * was already in the literature from one that was not. The year strip at the
 * top is the same set of sources placed on one axis, which is where the
 * independence of the traditions becomes visible.
 */
export default function LiteraturePage({core, data}: AtlasPageProps<LiteratureData>): React.JSX.Element {
  useRegisteredAnchors([...data.traditions.map((tradition) => `tradition-${tradition.slug}`), "tradition-other"]);
  const ordered = core.traditions
    .map((tradition) => data.traditions.find((entry) => entry.slug === tradition.slug))
    .filter((tradition): tradition is RenderedTradition => tradition !== undefined);
  const placed = new Set(ordered.flatMap((tradition) => tradition.papers));
  const other = Object.values(data.papers)
    .filter((paper) => paper.tradition === null && !placed.has(paper.key))
    .sort((a, b) => (a.year ?? 0) - (b.year ?? 0));

  return (
    <AtlasShell
      wide
      title="Literature"
      description="The traditions this work draws on, what each source establishes, and which claims cite it."
    >
      <h1>Literature</h1>
      <p className="atlas__lead">
        The problem sits where several traditions meet — optimal experimental design, quantization
        for estimation, determinant clustering, vector quantization, inference-aware summaries in
        particle physics, and plug-in asymptotics — each of which developed its half of the
        question without the others. Most of the ingredients used here are established in one of
        them, and are cited below rather than claimed.
      </p>

      <YearStrip core={core} />

      <div className="atlas__grid">
        <div>
          {ordered.map((tradition) => (
            <section className="literature-tradition atlas__section" key={tradition.slug} id={`tradition-${tradition.slug}`}>
              <h2>{tradition.label}</h2>
              {tradition.papers.map((key) => {
                const paper = data.papers[key];
                return paper ? <Entry core={core} key={key} paper={paper} /> : null;
              })}
              {tradition.notes.map((note) => (
                <div className="literature-note" key={note.heading}>
                  <h3>{note.heading}</h3>
                  <AnnotationFields fields={note.fieldsHtml} />
                  <Html html={note.proseHtml} />
                </div>
              ))}
            </section>
          ))}

          {other.length === 0 ? null : (
            <section className="literature-tradition atlas__section" id="tradition-other">
              <h2>Other sources</h2>
              <p className="atlas__muted">
                Sources cited for a single step, which belong to none of the traditions above.
              </p>
              {other.map((paper) => (
                <Entry core={core} key={paper.key} paper={paper} />
              ))}
            </section>
          )}
        </div>

        <section className="atlas__aside" aria-labelledby="closest-prior-work">
          <h2 id="closest-prior-work">Closest prior work for the headline results</h2>
          {data.priorArt.length === 0 ? (
            <p className="atlas__muted">No prior-art audit has been recorded yet.</p>
          ) : (
            <ul className="prior-art">
              {data.priorArt.map((entry) => (
                <li key={entry.id}>
                  <ClaimLink core={core} id={entry.id} showId={false} />
                  {entry.sources.length === 0 ? (
                    <p className="prior-art__sources">The audit found no nearer source than the traditions above.</p>
                  ) : (
                    <ul className="prior-art__sources">
                      {entry.sources.map((source) => (
                        <li key={source.name}>
                          <span className="prior-art__name">{source.name}</span>{" "}
                          <Html html={source.html} />
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </AtlasShell>
  );
}

/** One source: who and when, what the annotation says, and what cites it. */
function Entry({core, paper}: {core: Core; paper: RenderedPaper}): React.JSX.Element {
  return (
    <div className="literature-entry">
      <div className="literature-entry__head">
        <PaperLink core={core} id={paper.key} />
        {paper.year === null ? null : <span className="literature-entry__year">{paper.year}</span>}
      </div>
      {paper.annotation === null ? null : (
        <>
          <AnnotationFields fields={paper.annotation.fieldsHtml} drop={["Paper"]} />
          <Html html={paper.annotation.proseHtml} />
        </>
      )}
      {paper.citedBy.length === 0 ? null : (
        <>
          <p className="cited-by-label">Cited by</p>
          <EntityList compact core={core} ids={paper.citedBy} />
        </>
      )}
    </div>
  );
}
