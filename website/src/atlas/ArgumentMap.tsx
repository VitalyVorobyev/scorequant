import Link from "@docusaurus/Link";
import {useMemo, useState} from "react";

import type {Edge, EdgeType, Provenance} from "../data/atlas";
import type {Core} from "./core";
import {claimHref, criterionLabel, entityHref, fixtureHref, levelLabel, provenanceLabel} from "./core";
import {ClaimLink, EntityList, FixtureLink, Legend} from "./Entity";
import {Glyph} from "./Glyph";
import {localLayout} from "./LocalMap";
import {useAtlasQuery} from "./useAtlasQuery";
import type {Question} from "./graph";
import {adjacency, askQuestion, boundary, egoGraph, enables, raises, restsOn, verifiedBy} from "./graph";
import type {Placed} from "./GraphFigure";
import {GraphFigure, nodeTitle} from "./GraphFigure";

const COL_WIDTH = 210;
const ROW_HEIGHT = 24;
const LEFT = 132;
const TOP = 48;
const QUESTIONS: {id: Question; label: string}[] = [
  {id: "rests-on", label: "What does it rest on?"},
  {id: "enables", label: "What does it enable?"},
  {id: "stops", label: "Where does it stop?"},
  {id: "prior-work", label: "Closest prior work"},
  {id: "open", label: "What is open next door?"},
];
const SCALES = [0.6, 0.8, 1, 1.25];

export interface ArgumentMapProps {
  core: Core;
}

/**
 * The full argument map: every claim and fixture at the position the
 * generator computed (columns are problem levels, bands are criteria), the
 * proof-prerequisite edges drawn faintly as the ground truth of the picture,
 * and a focus that brings one neighbourhood forward with its typed edges.
 * The five question buttons re-highlight the same drawing; they never move
 * anything. The focused id is mirrored in `?focus=` so a claim page can open
 * the map on itself.
 */
export function ArgumentMap({core}: ArgumentMapProps): React.JSX.Element {
  const {params, update} = useAtlasQuery();
  const requested = params.get("focus") ?? "D-EXCHANGE-IMPLIES-VORONOI";
  const valid = requested in core.claims || requested in core.fixtures;
  const focus = valid ? requested : "D-EXCHANGE-IMPLIES-VORONOI";
  const ask = params.get("ask");
  const question = QUESTIONS.find((q) => q.id === ask)?.id ?? null;
  const setQuestion = (q: Question | null): void => update({ask: q ?? ""});
  const full = params.get("view") === "full";
  const adj = useMemo(() => adjacency(core.edges), [core.edges]);
  const [scale, setScale] = useState(1);
  const [hiddenClasses, setHiddenClasses] = useState<Set<Provenance>>(new Set());
  const [hiddenLanes, setHiddenLanes] = useState<Set<string>>(new Set());
  const [showFixtures, setShowFixtures] = useState(true);

  const layout = core.layout;
  const neighbourhood = useMemo(() => localLayout(core, focus, {colGap: 230, rowGap: 38, maxPerColumn: Number.POSITIVE_INFINITY}), [core, focus]);
  const visible = useMemo(() => {
    if (!full) return neighbourhood.positions;
    const out: Record<string, Placed> = {};
    for (const [id, at] of Object.entries(layout.positions)) {
      const claim = core.claims[id];
      if (claim) {
        if (hiddenClasses.has(claim.provenance)) continue;
      } else if (!showFixtures) continue;
      if (hiddenLanes.has(at.lane)) continue;
      out[id] = {x: LEFT + at.x * COL_WIDTH + 18, y: TOP + at.y * ROW_HEIGHT + 12};
    }
    return out;
  }, [full, neighbourhood.positions, layout.positions, core.claims, hiddenClasses, hiddenLanes, showFixtures]);

  const width = full ? LEFT + layout.columns.length * COL_WIDTH : neighbourhood.width + 210;
  const height = full ? TOP + (layout.height + 1) * ROW_HEIGHT + 16 : neighbourhood.height;

  const {highlight, highlightEdges, edges} = useMemo(() => {
    const baseline = core.edges.filter((edge) => edge.type === "rests_on" && edge.source in visible && edge.target in visible);
    if (!focus || !(focus in visible)) {
      return {highlight: null, highlightEdges: null, edges: baseline};
    }
    let nodes: string[];
    let types: EdgeType[];
    if (question) {
      const asked = askQuestion(core, adj, focus, question);
      nodes = asked.nodes;
      types = asked.edgeTypes;
    } else {
      nodes = egoGraph(core, adj, focus, 1).nodes;
      types = ["rests_on", "enables", "raises", "verified_by", "converse_fails", "refuted_by", "bounded_by"];
    }
    const set = new Set(nodes.filter((id) => id in visible));
    const shown: Edge[] = core.edges.filter(
      (edge) =>
        types.includes(edge.type) && set.has(edge.source) && set.has(edge.target) && (question !== null || edge.source === focus || edge.target === focus),
    );
    const keys = new Set(shown.map((edge) => `${edge.source}|${edge.target}|${edge.type}`));
    for (const edge of baseline) {
      const key = `${edge.source}|${edge.target}|${edge.type}`;
      if (!keys.has(key)) shown.push(edge);
    }
    return {highlight: set, highlightEdges: new Set(types), edges: shown};
  }, [core, adj, focus, question, visible]);

  const select = (id: string): void => {
    update({focus: id, ask: ""});
  };
  const toggle = <T,>(set: Set<T>, value: T, update: (next: Set<T>) => void): void => {
    const next = new Set(set);
    if (next.has(value)) next.delete(value);
    else next.add(value);
    update(next);
  };

  const background = full ? (
    <g className="map-background">
      {layout.bands.map((band, i) => (
        <g key={band.id}>
          <rect
            className={i % 2 === 0 ? "band" : "band band--alt"}
            x={0}
            y={TOP + band.y0 * ROW_HEIGHT}
            width={width}
            height={(band.y1 - band.y0) * ROW_HEIGHT + 8}
          />
          <text className="label label--strong" x={12} y={TOP + band.y0 * ROW_HEIGHT + 16}>
            {core.vocabulary.lanes.find((lane) => lane.id === band.id)?.label ?? band.id}
          </text>
        </g>
      ))}
      {layout.columns.map((column) => (
        <g key={column.level}>
          <line className="grid-line" x1={LEFT + column.x * COL_WIDTH} y1={TOP - 8} x2={LEFT + column.x * COL_WIDTH} y2={height} />
          <text className="label label--strong" x={LEFT + column.x * COL_WIDTH + 8} y={TOP - 18}>
            {core.vocabulary.levels.find((level) => level.id === column.level)?.short ?? column.level}
          </text>
        </g>
      ))}
    </g>
  ) : undefined;

  const options = Object.keys(core.layout.positions)
    .map((id) => ({id, title: nodeTitle(core, id)}))
    .sort((a, b) => a.title.localeCompare(b.title));

  return (
    <div className="argument-map">
      {!valid ? (
        <p role="status">
          The requested entity is not in this atlas. Showing the central theorem.{" "}
          <button type="button" onClick={() => update({focus: "", ask: ""})}>
            Reset selection
          </button>
        </p>
      ) : null}
      <div className="research-view-switch">
        <button type="button" onClick={() => update({view: full ? "" : "full"})}>
          {full ? "Show neighbourhood" : "Show full graph"}
        </button>
      </div>
      <div className="argument-map__controls">
        <label className="argument-map__search">
          <span>Find a result</span>
          <input
            list="argument-map-nodes"
            placeholder="title or id"
            onChange={(event) => {
              const value = event.target.value.trim();
              const hit = options.find((o) => o.id === value || o.title === value || value.endsWith(`(${o.id})`));
              if (hit) {
                update({focus: hit.id, ask: ""});
              }
            }}
          />
          <datalist id="argument-map-nodes">
            {options.map((o) => (
              <option key={o.id} value={`${o.title} (${o.id})`} />
            ))}
          </datalist>
        </label>
        {full ? (
          <details className="research-disclosure">
            <summary>Advanced graph filters</summary>
            <fieldset className="argument-map__filter">
              <legend>Show</legend>
              {core.vocabulary.provenance
                .filter((entry) => entry.id !== "verification")
                .map((entry) => (
                  <label key={entry.id}>
                    <input
                      type="checkbox"
                      checked={!hiddenClasses.has(entry.id as Provenance)}
                      onChange={() => toggle(hiddenClasses, entry.id as Provenance, setHiddenClasses)}
                    />
                    <Glyph kind={entry.id as Provenance} /> {entry.label}
                  </label>
                ))}
              <label>
                <input type="checkbox" checked={showFixtures} onChange={() => setShowFixtures((v) => !v)} />
                <Glyph kind="fixture" /> fixtures
              </label>
            </fieldset>
            <fieldset className="argument-map__filter">
              <legend>Criteria</legend>
              {core.vocabulary.lanes.map((lane) => (
                <label key={lane.id}>
                  <input type="checkbox" checked={!hiddenLanes.has(lane.id)} onChange={() => toggle(hiddenLanes, lane.id, setHiddenLanes)} /> {lane.label}
                </label>
              ))}
            </fieldset>
          </details>
        ) : null}
        <div className="argument-map__zoom" role="group" aria-label="Zoom">
          {SCALES.map((value) => (
            <button key={value} type="button" aria-pressed={scale === value} onClick={() => setScale(value)}>
              {Math.round(value * 100)}%
            </button>
          ))}
        </div>
      </div>

      <div className="argument-map__body">
        <div className="argument-map__canvas" tabIndex={0}>
          <GraphFigure
            core={core}
            positions={visible}
            edges={edges}
            width={width}
            height={height}
            displayWidth={width * scale}
            focus={focus}
            highlight={highlight}
            highlightEdges={highlightEdges}
            labels
            labelChars={26}
            onSelect={select}
            background={background}
            title={full ? "Argument map of the research" : "Neighbourhood of the selected result"}
            description="Columns are problem levels from universal identities to application; bands are criteria. Lines are proof prerequisites; the focused result shows its typed neighbourhood."
          />
        </div>
        <div className="argument-map__panel" aria-live="polite">
          {focus ? <FocusPanel core={core} adj={adj} id={focus} question={question} onQuestion={setQuestion} /> : <IdlePanel core={core} />}
        </div>
      </div>

      {full ? (
        <details className="research-disclosure">
          <summary>Every result by problem level</summary>
          <div className="argument-map__list">
            <h2>Every result by problem level</h2>
            {layout.columns.map((column) => {
              const ids = Object.keys(layout.positions).filter((id) => layout.positions[id]?.x === column.x && id in core.claims);
              return (
                <section key={column.level}>
                  <h3>{levelLabel(core, column.level)}</h3>
                  <EntityList core={core} ids={ids.sort()} compact />
                </section>
              );
            })}
          </div>
        </details>
      ) : null}
      <details className="research-disclosure">
        <summary>Graph notation</summary>
        <Legend core={core} fixtures />
      </details>
    </div>
  );
}

function IdlePanel({core}: {core: Core}): React.JSX.Element {
  return (
    <div className="focus-panel">
      <p>
        Select a result to bring its neighbourhood forward. Columns run from what is true of any hard label, on the left, to results stated for one application,
        on the right; bands separate the criteria. Solid lines are proof prerequisites.
      </p>
      <p>Start with the headline results:</p>
      <EntityList core={core} ids={core.headline.slice(0, 4)} compact />
    </div>
  );
}

function FocusPanel({
  core,
  adj,
  id,
  question,
  onQuestion,
}: {
  core: Core;
  adj: ReturnType<typeof adjacency>;
  id: string;
  question: Question | null;
  onQuestion: (q: Question | null) => void;
}): React.JSX.Element {
  const claim = core.claims[id];
  if (!claim) {
    const fixture = core.fixtures[id];
    if (!fixture) return <div className="focus-panel" />;
    return (
      <div className="focus-panel">
        <p className="focus-panel__kind">
          <Glyph kind="fixture" /> Counterexample fixture
        </p>
        <h2 className="focus-panel__title">
          <Link to={fixtureHref(fixture.slug)}>{fixture.id}</Link>
        </h2>
        <p>{fixture.falsifies}</p>
        <h3>Cited by</h3>
        <EntityList
          core={core}
          ids={fixture.citedBy.map((c) => c.claim)}
          compact
          meta={(cid) => (fixture.citedBy.find((c) => c.claim === cid)?.type === "refuted_by" ? "refuted" : "bounded")}
        />
      </div>
    );
  }
  const stop = boundary(adj, id);
  const stops = [...stop.converse, ...stop.refuted, ...stop.bounded];
  return (
    <div className="focus-panel">
      <p className="focus-panel__kind">
        <Glyph kind={claim.provenance} checked={claim.machineChecked} /> {provenanceLabel(core, claim.provenance)}
        {claim.machineChecked ? " · machine-checked" : ""}
      </p>
      <h2 className="focus-panel__title">
        <Link to={claimHref(claim.slug)}>{claim.title}</Link>
      </h2>
      <p className="atlas__id">{claim.id}</p>
      <p className="focus-panel__meta">
        {claim.criterion.map((c) => criterionLabel(core, c)).join(", ")} · {levelLabel(core, claim.level)}
      </p>
      <p className="focus-panel__statement">{claim.editorial?.summary ?? claim.statement}</p>
      <div className="focus-panel__questions" role="group" aria-label="Ask of this result">
        {QUESTIONS.map((q) => (
          <button key={q.id} type="button" aria-pressed={question === q.id} onClick={() => onQuestion(question === q.id ? null : q.id)}>
            {q.label}
          </button>
        ))}
      </div>
      <Neighbours core={core} title="Rests on" ids={restsOn(adj, id)} />
      <Neighbours core={core} title="Enables" ids={enables(adj, id).filter((o) => o !== id)} />
      <Neighbours core={core} title="Raises" ids={raises(adj, id)} />
      <Neighbours core={core} title="Where it stops" ids={stops} />
      <Neighbours core={core} title="Verified by" ids={verifiedBy(adj, id)} />
      {claim.literature.length > 0 ? (
        <>
          <h3>Cites</h3>
          <ul className="focus-panel__cites">
            {claim.literature.map((key) => {
              const href = entityHref(core, key);
              const paper = core.papers[key];
              return (
                <li key={key}>
                  {href && paper ? <Link to={href}>{`${paper.authors.map((a) => a.name).join(", ")}${paper.year ? ` (${paper.year})` : ""}`}</Link> : key}
                </li>
              );
            })}
          </ul>
        </>
      ) : null}
      <p>
        <Link to={claimHref(claim.slug)}>Open the page</Link>
      </p>
    </div>
  );
}

function Neighbours({core, title, ids}: {core: Core; title: string; ids: string[]}): React.JSX.Element | null {
  if (ids.length === 0) return null;
  return (
    <>
      <h3>{title}</h3>
      <ul className="focus-panel__list">
        {ids.map((id) => (
          <li key={id}>{id in core.claims ? <ClaimLink core={core} id={id} showId={false} /> : <FixtureLink core={core} id={id} />}</li>
        ))}
      </ul>
    </>
  );
}
