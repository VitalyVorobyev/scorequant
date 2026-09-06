import Link from "@docusaurus/Link";
import {useMemo} from "react";

import {ReferenceLink} from "../../components/ReferenceLink";
import {Tex} from "../../components/Tex";
import {AtlasShell} from "../AtlasShell";
import {BandStrip} from "../BandStrip";
import type {AtlasPageProps, CoreClaim} from "../core";
import {claimHref, provenanceLabel} from "../core";
import {EntityList, Html, Legend, PaperLink} from "../Entity";
import {Glyph} from "../Glyph";
import {adjacency, boundary, cites, raisedBy} from "../graph";
import {LocalMap} from "../LocalMap";

export interface HomeCard extends CoreClaim {
  noteHtml: string | null;
  scopeHtml: string | null;
}

export interface HomeData {
  frontier: HomeCard[];
  headline: HomeCard[];
}

/**
 * `/research/`: the problem, the results established here, the shape of the
 * field, the current frontier, and the ways into the atlas.
 */
export default function AtlasHome({core, data}: AtlasPageProps<HomeData>): React.JSX.Element {
  const adj = useMemo(() => adjacency(core.edges), [core.edges]);
  return (
    <AtlasShell
      title="Research"
      description="A navigable map of what is known about information-preserving hard binning, what ScoreQuant established, where it fails, and what remains open."
      wide
    >
      <div className="atlas__measure">
        <h1>Research</h1>
        <p className="atlas__lead">
          A map of what is known about hard binning that preserves Fisher information, what this project established, where those results stop being true, and what remains
          open. Every page is generated from the project's research registry; nothing here is written twice.
        </p>

        <section aria-labelledby="problem">
          <h2 id="problem">The problem</h2>
          <p>
            An observation is reduced to one of <Tex>K</Tex> labels by a hard rule <Tex>{String.raw`q`}</Tex>. For a fixed rule in a regular model the label carries Fisher
            information <Tex>{String.raw`I_q`}</Tex> about the parameters, and the loss against the unbinned information <Tex>I</Tex> is an identity, not an estimate: it is
            the within-cell scatter of the <em>score</em>, the gradient of the log-likelihood at the reference point,
          </p>
          <Tex display>{String.raw`I - I_q \;=\; \sum_{b=1}^{K} \mathbb{E}\Big[\,\mathbb{1}\{q(X)=b\}\,\big(s(X)-\mu_b\big)\big(s(X)-\mu_b\big)^{\top}\Big] \;\succeq\; 0 .`}</Tex>
          <p>
            So the space in which to choose cells is score space, and a criterion turns the retained matrix into one number to maximise: the determinant (D), the determinant
            of the Schur complement after profiling out nuisance parameters (profiled <Tex>{String.raw`D_s`}</Tex>), the whitened trace, or one of the control criteria A and
            E. The question then splits into three levels that look like one. Which labels do <em>these</em> weighted scores get (finite assignment)? Which rule on score space
            labels the next observation (inductive rule)? And which rule is best under the population law itself, with no sample in the way (population design)? The{" "}
            <ReferenceLink to="book/ch05-information-after-binning/">reference derives the identity</ReferenceLink>; the atlas maps what is known at each level.
          </p>
        </section>

        <section aria-labelledby="established">
          <h2 id="established">Established here</h2>
          <p>
            The results below were proved in this project. Each card says what the result means, how it is classed, and what it rests on; the small map beside it is its
            neighbourhood in the argument. A result marked <Glyph kind="proved_new" /> passed a targeted search of the literature without a direct precedent being found,
            which records a search gap and nothing more; the nearest prior work is named on its page.
          </p>
        </section>
      </div>
      <div className="result-cards">
        {data.headline.map((card) => (
          <ResultCard key={card.id} core={core} card={card} adj={adj} />
        ))}
      </div>
      <Legend core={core} />

      <div className="atlas__measure">
        <section aria-labelledby="shape">
          <h2 id="shape">The shape of the field</h2>
          <p>
            One row per theme, read left to right: what the literature already settled, what this project added, the examples where a natural generalisation fails, and the
            questions still open. Each mark is one result; the <Link to="/research/landscape/">landscape</Link> arranges the same marks by criterion and problem level, and the{" "}
            <Link to="/research/map/">map</Link> draws the argument between them.
          </p>
        </section>
      </div>
      <BandStrip core={core} />

      <div className="atlas__measure">
        <section aria-labelledby="frontier">
          <h2 id="frontier">Where the frontier is now</h2>
          <p>Questions the settled results run up against, with what is already excluded and what an answer would unlock.</p>
          {data.frontier.map((card) => {
            const stop = boundary(adj, card.id);
            const excluded = [...stop.refuted, ...stop.bounded];
            const unlocks = raisedBy(adj, card.id);
            return (
              <div key={card.id} className="frontier-item">
                <h3>
                  <Link to={claimHref(card.slug)}>
                    <Glyph kind="open" /> {card.title}
                  </Link>
                </h3>
                {card.noteHtml !== null ? <Html html={card.noteHtml} /> : <p>{card.statement}</p>}
                {excluded.length > 0 ? (
                  <p className="frontier-item__line">
                    <span>Already excluded</span> <EntityList core={core} ids={excluded} compact />
                  </p>
                ) : null}
                {unlocks.length > 0 ? (
                  <p className="frontier-item__line">
                    <span>Raised by</span> <EntityList core={core} ids={unlocks} compact />
                  </p>
                ) : null}
              </div>
            );
          })}
          <p>
            <Link to="/research/frontier/">Every open question, by theme</Link>
          </p>
        </section>

        <section aria-labelledby="ways">
          <h2 id="ways">Ways in</h2>
          <dl className="ways-in">
            <div>
              <dt>
                <Link to="/research/map/">Map</Link>
              </dt>
              <dd>What does a result rest on, what does it enable, and where does it stop? The argument drawn as a graph, by problem level and criterion.</dd>
            </div>
            <div>
              <dt>
                <Link to="/research/landscape/">Landscape</Link>
              </dt>
              <dd>How do D, profiled Ds, trace, A and E differ, and which regions are established, proved here, refuted, or still empty?</dd>
            </div>
            <div>
              <dt>
                <Link to="/research/frontier/">Frontier</Link>
              </dt>
              <dd>What remains open, grouped by theme, with what is already excluded and what an answer would unlock.</dd>
            </div>
            <div>
              <dt>
                <Link to="/research/literature/">Literature</Link>
              </dt>
              <dd>Which papers matter, why, what each does not solve, and which results here rest on it.</dd>
            </div>
            <div>
              <dt>
                <Link to="/research/machine-checked/">Machine-checked</Link>
              </dt>
              <dd>Which statements are proved in Lean, against which frozen specification, and what is deliberately outside that track.</dd>
            </div>
            <div>
              <dt>
                <Link to="/research/library/">In the library</Link>
              </dt>
              <dd>Which objects and refusals of the Python library carry which results.</dd>
            </div>
            <div>
              <dt>
                <Link to="/research/how-to-read/">How to read</Link>
              </dt>
              <dd>The provenance classes, the edge types, and the rule that a search gap is not a novelty claim.</dd>
            </div>
          </dl>
        </section>
      </div>
    </AtlasShell>
  );
}

function ResultCard({core, card, adj}: {core: AtlasPageProps<HomeData>["core"]; card: HomeCard; adj: ReturnType<typeof adjacency>}): React.JSX.Element {
  const nearest = cites(adj, card.id).slice(0, 2);
  return (
    <article className="result-card" aria-labelledby={`card-${card.slug}`}>
      <h3 id={`card-${card.slug}`}>
        <Link to={claimHref(card.slug)}>{card.title}</Link>
      </h3>
      <div className="result-card__meaning">{card.noteHtml !== null ? <Html html={card.noteHtml} /> : <p>{card.statement}</p>}</div>
      <LocalMap core={core} id={card.id} thumbnail />
      <div className="result-card__foot">
        <span className="provenance-band">
          <span className="provenance-band__class">
            <Glyph kind={card.provenance} checked={card.machineChecked} /> {provenanceLabel(core, card.provenance)}
          </span>
          {card.machineChecked ? <span className="provenance-band__mark">machine-checked</span> : null}
        </span>
        {nearest.length > 0 ? (
          <span className="result-card__nearest">
            Nearest prior work:{" "}
            {nearest.map((key, i) => (
              <span key={key}>
                {i > 0 ? "; " : ""}
                <PaperLink core={core} id={key} />
              </span>
            ))}
          </span>
        ) : null}
      </div>
    </article>
  );
}
