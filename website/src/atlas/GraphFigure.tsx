import type {KeyboardEvent, ReactNode} from "react";

import type {Edge} from "../data/atlas";
import type {Core} from "./core";
import {GlyphShape} from "./Glyph";

export interface Placed {
  x: number;
  y: number;
}

export interface GraphFigureProps {
  core: Core;
  /** Pixel positions of every node to draw. */
  positions: Record<string, Placed>;
  edges: readonly Edge[];
  width: number;
  height: number;
  /** The node under focus, drawn with a halo. */
  focus?: string | null;
  /** Nodes drawn at full strength; every other node and edge is dimmed. Omit to dim nothing. */
  highlight?: ReadonlySet<string> | null;
  /** Edge types drawn at full strength when a highlight is active. */
  highlightEdges?: ReadonlySet<string> | null;
  /** Draw a truncated title beside each node. */
  labels?: boolean;
  labelChars?: number;
  onSelect?: (id: string) => void;
  /** Drawn beneath the nodes (bands, column rules, headings). */
  background?: ReactNode;
  title: string;
  description?: string;
  className?: string;
  /** CSS width in pixels; the drawing scales to it. Defaults to `width`. */
  displayWidth?: number;
}

const NODE_SCALE = 0.75;

export function nodeTitle(core: Core, id: string): string {
  const claim = core.claims[id];
  if (claim) return claim.title;
  const fixture = core.fixtures[id];
  if (fixture) return fixture.id;
  return id;
}

export function nodeKind(core: Core, id: string): Parameters<typeof GlyphShape>[0]["kind"] {
  const claim = core.claims[id];
  if (claim) return claim.provenance;
  return "fixture";
}

function truncate(text: string, chars: number): string {
  return text.length <= chars ? text : `${text.slice(0, chars - 1).trimEnd()}…`;
}

/**
 * The one SVG renderer behind every map in the atlas: a thumbnail on a
 * result card, the local map on a claim page, and the full argument map.
 * It draws typed edges beneath provenance glyphs, dims what is outside the
 * current highlight, and makes every node a keyboard-reachable control when
 * `onSelect` is given. Layout is the caller's business.
 */
export function GraphFigure({
  core,
  positions,
  edges,
  width,
  height,
  focus = null,
  highlight = null,
  highlightEdges = null,
  labels = false,
  labelChars = 24,
  onSelect,
  background,
  title,
  description,
  className,
  displayWidth
}: GraphFigureProps): React.JSX.Element {
  const dimmed = (id: string): boolean => highlight !== null && !highlight.has(id);
  const edgeDimmed = (edge: Edge): boolean =>
    highlight !== null && (!highlight.has(edge.source) || !highlight.has(edge.target) || (highlightEdges !== null && !highlightEdges.has(edge.type)));
  const interactive = onSelect !== undefined;
  const handleKey = (id: string) => (event: KeyboardEvent<SVGGElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect?.(id);
    }
  };
  return (
    <svg
      className={className}
      viewBox={`0 0 ${width} ${height}`}
      style={{width: displayWidth ?? width, height: "auto"}}
      // An image may not contain focusable content, so an interactive map is a
      // group of controls; a static one is an image with a title and description.
      role={interactive ? "group" : "img"}
      aria-label={title}
    >
      <title>{title}</title>
      {description ? <desc>{description}</desc> : null}
      <defs>
        <marker id="atlas-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L8 4 L0 8 Z" fill="var(--atlas-edge-strong)" />
        </marker>
      </defs>
      {background}
      <g className="edges">
        {edges.map((edge) => {
          const a = positions[edge.source];
          const b = positions[edge.target];
          if (!a || !b) return null;
          const dim = edgeDimmed(edge);
          return (
            <line
              key={`${edge.source}|${edge.target}|${edge.type}`}
              className={`edge edge--${edge.type}${dim ? " edge--dim" : ""}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              markerEnd={edge.type === "enables" && !dim ? "url(#atlas-arrow)" : undefined}
            />
          );
        })}
      </g>
      <g className="nodes">
        {Object.entries(positions).map(([id, at]) => {
          const kind = nodeKind(core, id);
          const claim = core.claims[id];
          const checked = claim?.machineChecked ?? false;
          const isFocus = focus === id;
          const label = nodeTitle(core, id);
          return (
            <g
              key={id}
              className={`node${dimmed(id) ? " node--dim" : ""}${isFocus ? " node--focus" : ""}`}
              transform={`translate(${at.x} ${at.y})`}
              tabIndex={interactive ? 0 : undefined}
              role={interactive ? "button" : undefined}
              aria-label={interactive ? label : undefined}
              onClick={interactive ? () => onSelect(id) : undefined}
              onKeyDown={interactive ? handleKey(id) : undefined}
              data-id={id}
            >
              <title>{label}</title>
              <circle className="node__halo" r={11} fill="transparent" />
              <g className={`glyph-${kind}`} transform={`translate(${-8 * NODE_SCALE} ${-8 * NODE_SCALE}) scale(${NODE_SCALE})`}>
                <GlyphShape kind={kind} />
                {checked ? (
                  <g className="glyph__check" transform="translate(8.5 8.5)">
                    <circle cx="4" cy="4" r="4" />
                    <path d="M2 4.2 L3.5 5.6 L6.2 2.6" />
                  </g>
                ) : null}
              </g>
              {labels ? (
                <text className="node__label" x={10} y={3.5}>
                  {truncate(label, labelChars)}
                </text>
              ) : null}
            </g>
          );
        })}
      </g>
    </svg>
  );
}
