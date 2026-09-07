import Link from "@docusaurus/Link";
import {useMemo} from "react";

import {ReferenceLink} from "../../components/ReferenceLink";
import type {Claim} from "../../data/atlas";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps, Core} from "../core";
import {criterionLabel, levelLabel, provenanceLabel} from "../core";
import {EntityList, FixtureLink, Html, PaperLink} from "../Entity";
import {adjacency, boundary, closure, enables, openNextDoor, raises, raisedBy, restsOn, verifiedBy} from "../graph";
import {LocalMap} from "../LocalMap";

export interface ClaimData extends Omit<Claim, "priorArt"> {
  assumptionsHtml: (string | null)[];
  noteHtml: string | null;
  priorArt: {date: string | null; sources: {html: string | null; name: string}[]; url: string}[];
  proofHtml: string | null;
  roleHtml: string | null;
  scopeHtml: string | null;
  statementHtml: string | null;
}

const KIND_WORD: Record<Claim["kind"], string> = {
  result: "Result",
  question: "Open question",
  counterexample: "Counterexample",
  evidence: "Measured evidence",
  audit: "Verification record",
};

/**
 * `/research/claims/<slug>/`: one research entity in the reading grammar —
 * meaning, statement, what it rests on, what it enables, where it stops,
 * what is open next door, the closest prior work, the machine check, the
 * library, the local map and the proof.
 */
export default function ClaimPage({core, data}: AtlasPageProps<ClaimData>): React.JSX.Element {
  const adj = useMemo(() => adjacency(core.edges), [core.edges]);
  const id = data.id;
  const rests = restsOn(adj, id);
  const chain = closure(adj, id, "rests_on", "out").filter((entry) => entry.depth > 1);
  const enabled = enables(adj, id).filter((other) => other !== id);
  const raised = raises(adj, id);
  const verifiers = verifiedBy(adj, id);
  const stop = boundary(adj, id);
  const nextDoor = openNextDoor(core, adj, id).filter((other) => other !== id);
  const motivatedBy = raisedBy(adj, id);
  // Papers the registry marks as relevant to this claim, beyond the ones it cites:
  // for a theorem with no citations these are its nearest prior work.
  const related = Object.values(core.papers)
    .filter((paper) => paper.relevantTo.includes(id) && !data.literature.includes(paper.key))
    .sort((a, b) => (a.year ?? 0) - (b.year ?? 0) || a.key.localeCompare(b.key))
    .map((paper) => paper.key);
  const isQuestion = data.kind === "question";
  const isCounter = data.kind === "counterexample";
  const sections = [
    {id: "meaning", label: "Meaning", show: data.noteHtml !== null || data.editorial !== null},
    {id: "statement", label: "Statement & assumptions", show: true},
    {id: "stops", label: "Limitations", show: stop.converse.length + stop.refuted.length + stop.bounded.length > 0 || data.scopeHtml !== null},
    {id: "prior-work", label: "Attribution", show: data.literature.length + related.length + data.priorArt.length > 0},
    {id: "proof", label: isQuestion ? "Full question" : "Proof & evidence", show: data.proofHtml !== null || data.audit !== null},
    {id: "relationships", label: "Relationships", show: true},
  ];
  const description = data.statement.length > 200 ? `${data.statement.slice(0, 197)}…` : data.statement;

  return (
    <AtlasShell title={data.title} description={description} wide>
      <div className="atlas__grid">
        <div className="atlas__measure">
          <p className="atlas__id">
            {KIND_WORD[data.kind]} · {data.id}
          </p>
          <h1>{data.editorial?.title ?? data.title}</h1>
          <ProvenanceBand core={core} data={data} />
          {data.proofHtml !== null ? (
            <p>
              <a href="#proof">{isQuestion ? "Read the full question" : "Read the proof and evidence"} →</a>
            </p>
          ) : null}

          {data.noteHtml !== null || data.editorial !== null ? (
            <section id="meaning" className="atlas__section">
              <h2>Meaning</h2>
              {data.editorial ? <p className="atlas__lead">{data.editorial.summary}</p> : null}
              {data.noteHtml ? (
                <details>
                  <summary>Scientific context</summary>
                  <Html html={data.noteHtml} />
                </details>
              ) : null}
            </section>
          ) : null}

          <section id="statement" className="atlas__section">
            <h2>Statement</h2>
            <Html html={data.statementHtml} className={`statement${isQuestion ? " statement--question" : isCounter ? " statement--counter" : ""}`} />
            {data.assumptions.length > 0 ? (
              <>
                <h3>Assumptions</h3>
                <ul className="assumptions">
                  {data.assumptionsHtml.map((html, i) => (
                    <li key={i}>
                      <Html html={html} as="span" />
                    </li>
                  ))}
                </ul>
              </>
            ) : null}
            {data.roleHtml !== null ? <Html html={data.roleHtml} className="atlas__muted" /> : null}
          </section>

          {stop.converse.length + stop.refuted.length + stop.bounded.length > 0 || data.scopeHtml !== null ? (
            <section id="stops" className="atlas__section">
              <h2>{isQuestion ? "Ruled out by" : "Where it stops"}</h2>
              {data.scopeHtml !== null ? <Html html={data.scopeHtml} className="scope-note" /> : null}
              {stop.converse.length > 0 ? (
                <>
                  <h3>The converse fails</h3>
                  <EntityList core={core} ids={stop.converse} />
                </>
              ) : null}
              {stop.refuted.length > 0 ? (
                <>
                  <h3>Refuted by</h3>
                  <EntityList core={core} ids={stop.refuted} note={(fid) => core.fixtures[fid]?.falsifies ?? ""} />
                </>
              ) : null}
              {stop.bounded.length > 0 ? (
                <>
                  <h3>Bounded by</h3>
                  <EntityList core={core} ids={stop.bounded} note={(fid) => core.fixtures[fid]?.falsifies ?? ""} />
                </>
              ) : null}
            </section>
          ) : null}

          {data.literature.length + related.length + data.priorArt.length > 0 ? (
            <section id="prior-work" className="atlas__section">
              <h2>Closest prior work</h2>
              {data.provenance === "proved_new" ? (
                <p>
                  A targeted search of the literature found no direct precedent for this statement. That records a search gap, not a novelty claim; the nearest
                  sources are named below.
                </p>
              ) : null}
              {data.literature.length > 0 ? (
                <ul className="entity-list">
                  {data.literature.map((key) => (
                    <li key={key}>
                      <PaperLink core={core} id={key} />
                    </li>
                  ))}
                </ul>
              ) : null}
              {related.length > 0 ? (
                <>
                  <h3>Nearest prior work</h3>
                  <ul className="entity-list">
                    {related.map((key) => (
                      <li key={key}>
                        <PaperLink core={core} id={key} />
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}
              {data.priorArt.length > 0 ? (
                <details className="research-disclosure">
                  <summary>Why each source is not this result: the search records</summary>
                  {data.priorArt.map((audit) => (
                    <div key={audit.url} className="prior-art">
                      <h3>Nearest sources{audit.date ? `, checked ${audit.date}` : ""}</h3>
                      {audit.sources.length > 0 ? (
                        <dl className="prior-art__sources">
                          {audit.sources.map((source) => (
                            <div key={source.name}>
                              <dt>{source.name}</dt>
                              <dd>
                                <Html html={source.html} as="span" />
                              </dd>
                            </div>
                          ))}
                        </dl>
                      ) : null}
                      <p className="atlas__muted">
                        <a href={audit.url} rel="noopener noreferrer">
                          The full search record
                        </a>
                      </p>
                    </div>
                  ))}
                </details>
              ) : null}
            </section>
          ) : null}

          <details className="research-disclosure" id="relationships">
            <summary>Relationships</summary>
            {rests.length > 0 ? (
              <section id="rests-on" className="atlas__section">
                <h2>{isQuestion ? "Related settled results" : "Rests on"}</h2>
                <EntityList core={core} ids={rests} />
                {chain.length > 0 ? (
                  <details className="chain">
                    <summary>The full chain beneath it</summary>
                    <EntityList
                      core={core}
                      ids={chain.map((entry) => entry.id)}
                      compact
                      meta={(other) => `depth ${chain.find((entry) => entry.id === other)?.depth ?? ""}`}
                    />
                  </details>
                ) : null}
              </section>
            ) : null}

            {enabled.length + raised.length + verifiers.length + (isQuestion ? motivatedBy.length : 0) > 0 ? (
              <section id="enables" className="atlas__section">
                <h2>{isQuestion ? "Where it comes from" : "Enables"}</h2>
                {isQuestion && motivatedBy.length > 0 ? (
                  <>
                    <h3>Raised by</h3>
                    <EntityList core={core} ids={motivatedBy} />
                  </>
                ) : null}
                {enabled.length > 0 ? <EntityList core={core} ids={enabled} /> : null}
                {raised.length > 0 ? (
                  <>
                    <h3>Raises</h3>
                    <EntityList core={core} ids={raised} />
                  </>
                ) : null}
                {verifiers.length > 0 ? (
                  <>
                    <h3>Verified by</h3>
                    <EntityList core={core} ids={verifiers} />
                  </>
                ) : null}
              </section>
            ) : null}

            {nextDoor.length > 0 && !isQuestion ? (
              <section id="open" className="atlas__section">
                <h2>Open next door</h2>
                <EntityList core={core} ids={nextDoor} />
              </section>
            ) : null}
          </details>
          {data.machineChecked !== null ? (
            <details id="machine-checked" className="atlas__section">
              <summary>Machine-checked statement</summary>
              <p>
                The statement is proved in {data.machineChecked.system} as <code>{data.machineChecked.declaration}</code>. Its hypotheses and conclusion are
                frozen in a specification file that was audited against this page's statement before the proof was written
                {data.machineChecked.statementAudit?.verdict ? ` (verdict: ${data.machineChecked.statementAudit.verdict})` : ""}. A Lean build certifies the
                theorem, not the Python and JAX code that implements it.
              </p>
              <ul>
                <li>
                  <a href={data.machineChecked.spec.url} rel="noopener noreferrer">
                    Frozen specification
                  </a>{" "}
                  <span className="atlas__id">{data.machineChecked.spec.path}</span>
                </li>
                <li>
                  <a href={data.machineChecked.proof.url} rel="noopener noreferrer">
                    Proof module
                  </a>{" "}
                  <span className="atlas__id">{data.machineChecked.proof.path}</span>
                </li>
                {data.machineChecked.statementAudit ? (
                  <li>
                    <a href={data.machineChecked.statementAudit.url} rel="noopener noreferrer">
                      Statement audit
                    </a>
                  </li>
                ) : null}
                <li>
                  <Link to="/research/machine-checked/">The whole machine-checked chain</Link>
                </li>
              </ul>
            </details>
          ) : null}

          {data.implementedBy.length + data.enforcedBy.length > 0 ? (
            <details id="library" className="atlas__section">
              <summary>In the library</summary>
              {data.implementedBy.length > 0 ? (
                <p>
                  Carried by{" "}
                  {data.implementedBy.map((name, i) => (
                    <span key={name}>
                      {i > 0 ? ", " : ""}
                      <code>{name}</code>
                    </span>
                  ))}
                  . <ReferenceLink to="symbols/">Reference</ReferenceLink>.
                </p>
              ) : null}
              {data.enforcedBy.length > 0 ? (
                <p>
                  Enforced by the refusal{data.enforcedBy.length > 1 ? "s" : ""}{" "}
                  {data.enforcedBy.map((code, i) => (
                    <span key={code}>
                      {i > 0 ? ", " : ""}
                      <FixtureLink core={core} id={code} />
                    </span>
                  ))}
                  : the library declines the operation and names the fixture.
                </p>
              ) : null}
              <p>
                <Link to="/research/library/">Every object and refusal</Link>
              </p>
            </details>
          ) : null}

          {data.kind !== "audit" ? (
            <details id="local-map" className="atlas__section">
              <summary>Local graph</summary>
              <figure className="atlas-figure">
                <LocalMap core={core} id={id} />
                <figcaption className="atlas-figure__caption">
                  Prerequisites to the left, what it enables or raises to the right, boundary cases below.{" "}
                  <Link to={`/research/map/?focus=${encodeURIComponent(id)}`}>Open in the map</Link>.
                </figcaption>
              </figure>
            </details>
          ) : null}

          {data.proofHtml !== null || data.audit !== null ? (
            <section id="proof" className="atlas__section">
              <h2>{isQuestion ? "The question in full" : isCounter ? "The example" : "Proof"}</h2>
              {data.proofHtml !== null && data.proof ? (
                <details className="proof" open={isQuestion}>
                  <summary>
                    {data.proof.label ? `${data.proof.label}. ` : ""}
                    {data.proof.title}
                  </summary>
                  <Html html={data.proofHtml} className="proof__body" />
                  <p className="proof__source">
                    From{" "}
                    <a href={data.proof.url} rel="noopener noreferrer">
                      {data.chapter ? data.chapter.label : data.proof.file}
                    </a>
                    {data.chapter ? (
                      <>
                        {" "}
                        · <Link to={`/research/claims/#group-${data.chapter.slug}`}>chapter contents</Link>
                      </>
                    ) : null}
                  </p>
                </details>
              ) : null}
              {data.audit ? (
                <p>
                  Independently audited{data.audit.verdict ? `: ${data.audit.verdict}` : ""}.{" "}
                  <a href={data.audit.url} rel="noopener noreferrer">
                    The audit report
                  </a>
                  .
                </p>
              ) : null}
              {data.artifact ? (
                <p className="atlas__muted">
                  Evidence:{" "}
                  <a href={data.artifact.url} rel="noopener noreferrer">
                    {data.artifact.path}
                  </a>
                  . Measured, never theorem authority.
                </p>
              ) : null}
            </section>
          ) : null}
        </div>

        <nav className="atlas__aside" aria-label="On this page">
          <h2>On this page</h2>
          <ul>
            {sections
              .filter((section) => section.show)
              .map((section) => (
                <li key={section.id}>
                  <a href={`#${section.id}`}>{section.label}</a>
                </li>
              ))}
          </ul>
          <h2>Where it sits</h2>
          <p className="atlas__aside-meta">
            {data.criterion.map((c) => criterionLabel(core, c)).join(", ")} · {levelLabel(core, data.level)}
          </p>
          <h2>Continue</h2>
          <ul>
            <li>
              <Link to={data.strip ? `/research/landscape/?theme=${encodeURIComponent(data.strip)}` : "/research/landscape/"}>
                {data.strip ? `Explore ${core.strips.find((strip) => strip.id === data.strip)?.label ?? "this theme"}` : "Explore the results"}
              </Link>
            </li>
            {data.kind !== "audit" ? (
              <li>
                <Link to={`/research/map/?focus=${encodeURIComponent(id)}`}>Open in the graph</Link>
              </li>
            ) : null}
            {data.theme ? (
              <li>
                <Link to={`/research/frontier/#${data.theme}`}>
                  Frontier: {core.themes.find((theme) => theme.slug === data.theme)?.label ?? data.theme}
                </Link>
              </li>
            ) : null}
            {data.chapter ? (
              <li>
                <Link to={`/research/claims/#group-${data.chapter.slug}`}>Chapter: {data.chapter.label}</Link>
              </li>
            ) : null}
          </ul>
        </nav>
      </div>
    </AtlasShell>
  );
}

function ProvenanceBand({core, data}: {core: Core; data: ClaimData}): React.JSX.Element {
  return (
    <p className="provenance-band">
      <span className="provenance-band__class">{provenanceLabel(core, data.provenance)}</span>
      {data.machineChecked !== null ? <span className="provenance-band__mark">machine-checked statement</span> : null}
      {data.audit !== null ? <span className="provenance-band__mark">independently audited</span> : null}
      <span className="provenance-band__mark">publication: {data.publicationStatus.replaceAll("_", " ")}</span>
      {data.searchStatus === "search_gap" && data.provenance !== "proved_new" ? <span className="provenance-band__mark">no direct precedent found</span> : null}
      {data.searchStatus === "prior_art_found" ? <span className="provenance-band__mark">prior art found</span> : null}
      {data.parked ? (
        <span className="provenance-band__mark" title="Set aside until an explicit decision reopens it">
          parked
        </span>
      ) : null}
    </p>
  );
}
