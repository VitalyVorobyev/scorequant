import type {Edge} from "../data/atlas";
import type {Core} from "./core";
import {adjacency, boundary, enables, raises, restsOn, verifiedBy} from "./graph";
import type {Placed} from "./GraphFigure";
import {GraphFigure} from "./GraphFigure";

export interface LocalLayout {
  positions: Record<string, Placed>;
  edges: Edge[];
  width: number;
  height: number;
  columns: {upstream: string[]; downstream: string[]; below: string[]; above: string[]};
}

/**
 * The radius-one neighbourhood of a claim as a fixed reading layout: what it
 * rests on to the left, what it enables and raises to the right, where it
 * stops (converse failures, fixtures) below, and the audits that verified it
 * above. The same layout serves a card thumbnail and the claim page, at
 * different sizes.
 */
export function localLayout(core: Core, id: string, options: {colGap?: number; rowGap?: number; maxPerColumn?: number} = {}): LocalLayout {
  const colGap = options.colGap ?? 150;
  const rowGap = options.rowGap ?? 26;
  const cap = options.maxPerColumn ?? 8;
  const adj = adjacency(core.edges);
  // Each neighbour appears once, in the first role that claims it: verifiers
  // above, prerequisites left, boundary cases below, everything else right.
  const taken = new Set<string>([id]);
  const claimRole = (ids: string[], limit: number): string[] => {
    const out: string[] = [];
    for (const other of ids) {
      if (taken.has(other) || out.length >= limit) continue;
      taken.add(other);
      out.push(other);
    }
    return out;
  };
  const above = claimRole(verifiedBy(adj, id), 3);
  const upstream = claimRole(restsOn(adj, id), cap);
  const stop = boundary(adj, id);
  const below = claimRole([...stop.converse, ...stop.refuted, ...stop.bounded], cap);
  const downstream = claimRole([...enables(adj, id), ...raises(adj, id)], cap);
  const tall = Math.max(upstream.length, downstream.length, 1);
  const centreY = 16 + (above.length > 0 ? rowGap : 0) + ((tall - 1) * rowGap) / 2;
  const positions: Record<string, Placed> = {};
  const column = (ids: string[], x: number): void => {
    const top = centreY - ((ids.length - 1) * rowGap) / 2;
    ids.forEach((other, i) => {
      positions[other] = {x, y: top + i * rowGap};
    });
  };
  column(upstream, 16);
  column(downstream, 16 + 2 * colGap);
  positions[id] = {x: 16 + colGap, y: centreY};
  above.forEach((other, i) => {
    positions[other] = {x: 16 + colGap + (i - (above.length - 1) / 2) * colGap * 0.6, y: 12};
  });
  const belowY = centreY + ((tall - 1) * rowGap) / 2 + rowGap + 6;
  below.forEach((other, i) => {
    positions[other] = {x: 16 + colGap + (i - (below.length - 1) / 2) * colGap * 0.6, y: belowY};
  });
  const nodes = new Set(Object.keys(positions));
  const edges = core.edges.filter(
    (edge) => edge.type !== "cites" && nodes.has(edge.source) && nodes.has(edge.target) && (edge.source === id || edge.target === id)
  );
  const height = (below.length > 0 ? belowY : centreY + ((tall - 1) * rowGap) / 2) + 18;
  return {positions, edges, width: 32 + 2 * colGap, height, columns: {upstream, downstream, below, above}};
}

export interface LocalMapProps {
  core: Core;
  id: string;
  /** Thumbnail: no labels, tighter spacing. */
  thumbnail?: boolean;
  onSelect?: (id: string) => void;
}

export function LocalMap({core, id, thumbnail = false, onSelect}: LocalMapProps): React.JSX.Element {
  const layout = thumbnail ? localLayout(core, id, {colGap: 96, rowGap: 16, maxPerColumn: 5}) : localLayout(core, id, {colGap: 170, rowGap: 28});
  const claim = core.claims[id];
  const title = `Local map of ${claim?.title ?? id}`;
  const description = `${layout.columns.upstream.length} prerequisites to the left, ${layout.columns.downstream.length} results it enables or questions it raises to the right, ${layout.columns.below.length} boundary cases below.`;
  const labels = !thumbnail;
  // Labels extend to the right of a node; give the right column room.
  const width = layout.width + (labels ? 150 : 0);
  return (
    <GraphFigure
      core={core}
      positions={layout.positions}
      edges={layout.edges}
      width={width}
      height={layout.height}
      focus={id}
      labels={labels}
      labelChars={22}
      {...(onSelect ? {onSelect} : {})}
      title={title}
      description={description}
      className={thumbnail ? "local-map local-map--thumb" : "local-map"}
    />
  );
}
