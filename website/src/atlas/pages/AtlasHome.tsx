import Link from "@docusaurus/Link";
import type {HomeEntry} from "../../data/atlas";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps, Core} from "../core";
import {claimHref} from "../core";
import {Html} from "../Entity";

export interface HomeCard extends HomeEntry {
  summaryHtml: string | null;
  significanceHtml: string | null;
  qualificationHtml: string | null;
}
export interface HomeData {
  introHtml: string | null;
  central: HomeCard;
  results: HomeCard[];
  boundaries: HomeCard[];
  questions: HomeCard[];
}

function Finding({entry, core, central = false}: {entry: HomeCard; core: Core; central?: boolean}): React.JSX.Element {
  const claim = core.claims[entry.id];
  if (!claim) throw new Error(`Unknown featured claim: ${entry.id}`);
  const href = claimHref(claim.slug);
  return (
    <article className={central ? "research-finding research-finding--central" : "research-finding"}>
      {central ? <p className="research-eyebrow">Central result</p> : null}
      {central ? (
        <h2>{entry.title}</h2>
      ) : (
        <h3>
          <Link to={href}>{entry.title}</Link>
        </h3>
      )}
      <Html html={entry.summaryHtml} />
      <Html html={entry.significanceHtml} />
      <Html html={entry.qualificationHtml} className="research-qualification" />
      {central ? (
        <Link className="research-text-link" to={href}>
          Read the theorem and its prior art →
        </Link>
      ) : null}
    </article>
  );
}

/** A curated introduction. Exact statements and provenance live on linked entity pages. */
export default function AtlasHome({core, data}: AtlasPageProps<HomeData>): React.JSX.Element {
  return (
    <AtlasShell
      title="Research"
      description="The geometry of information-preserving hard bins, its limits under nuisance profiling, and the questions that remain."
      wide
    >
      <header className="research-opening">
        <p className="research-eyebrow">ScoreQuant · Research atlas</p>
        <h1>Research</h1>
        <Html className="atlas__lead" html={data.introHtml} />
      </header>
      <Finding entry={data.central} core={core} central />
      <section className="research-section" aria-labelledby="established">
        <h2 id="established">What we established</h2>
        <div className="research-columns">
          {data.results.map((entry) => (
            <Finding key={entry.id} entry={entry} core={core} />
          ))}
        </div>
      </section>
      <section className="research-section" aria-labelledby="boundaries">
        <h2 id="boundaries">Where it breaks</h2>
        <div className="research-columns research-columns--two">
          {data.boundaries.map((entry) => (
            <Finding key={entry.id} entry={entry} core={core} />
          ))}
        </div>
      </section>
      <section className="research-section" aria-labelledby="frontier">
        <h2 id="frontier">Open frontier</h2>
        <ol className="research-questions">
          {data.questions.map((entry) => (
            <li key={entry.id}>
              <Finding entry={entry} core={core} />
            </li>
          ))}
        </ol>
      </section>
      <section className="research-section" aria-labelledby="ways">
        <h2 id="ways">Explore further</h2>
        <div className="research-columns research-entrances">
          <div>
            <h3>
              <Link to="/research/landscape/">Explore the results →</Link>
            </h3>
            <p>Known results, contributions, and boundaries, organised by theme.</p>
          </div>
          <div>
            <h3>
              <Link to="/research/literature/">Follow the literature →</Link>
            </h3>
            <p>Publications, authors, and the ideas this work builds on.</p>
          </div>
          <div>
            <h3>
              <Link to="/research/frontier/">Visit the frontier →</Link>
            </h3>
            <p>Precise open questions and the evidence that constrains them.</p>
          </div>
        </div>
      </section>
    </AtlasShell>
  );
}
