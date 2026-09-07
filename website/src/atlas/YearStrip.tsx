import Link from "@docusaurus/Link";

import {linearScale, niceTicks} from "../components/charts/scale";
import type {Core, CorePaper} from "./core";
import {paperHref} from "./core";

/** One tradition's row of the strip: its label, and the papers that carry a year. */
interface Lane {
  label: string;
  papers: CorePaper[];
  slug: string;
}

const GUTTER = 300;
const WIDTH = 940;
const PAD_RIGHT = 18;
const TOP = 10;
const AXIS = 30;
const STACK = 9;
const MARK = 3.4;

/** The traditions in `core.traditions` order, then everything without one. */
export function lanes(core: Core): Lane[] {
  const dated = (key: string): CorePaper | null => {
    const paper = core.papers[key];
    return paper && paper.year !== null ? paper : null;
  };
  const out: Lane[] = [];
  for (const tradition of core.traditions) {
    const papers = tradition.papers.map(dated).filter((paper): paper is CorePaper => paper !== null);
    if (papers.length > 0) out.push({label: tradition.label, slug: tradition.slug, papers: sortByYear(papers)});
  }
  const loose = Object.values(core.papers).filter((paper) => paper.tradition === null && paper.year !== null);
  if (loose.length > 0) out.push({label: "Other sources", slug: "other", papers: sortByYear(loose)});
  return out;
}

function sortByYear(papers: CorePaper[]): CorePaper[] {
  return [...papers].sort((a, b) => (a.year ?? 0) - (b.year ?? 0) || a.key.localeCompare(b.key));
}

/** `Authors (year) Title`, the sentence a mark says when pointed at. */
function describe(paper: CorePaper): string {
  const who = paper.authors.map((author) => author.name).join(", ");
  return `${who} (${String(paper.year ?? "")}) ${paper.title}`;
}

/** Rows within a lane: papers of the same year are stacked, oldest lane first. */
function stacked(papers: CorePaper[]): {offset: number; paper: CorePaper}[] {
  const byYear = new Map<number, CorePaper[]>();
  for (const paper of papers) {
    const year = paper.year ?? 0;
    const group = byYear.get(year);
    if (group) group.push(paper);
    else byYear.set(year, [paper]);
  }
  const out: {offset: number; paper: CorePaper}[] = [];
  for (const group of byYear.values()) {
    for (const [index, paper] of group.entries()) {
      out.push({paper, offset: (index - (group.length - 1) / 2) * STACK});
    }
  }
  return out;
}

function laneHeight(papers: CorePaper[]): number {
  const counts = new Map<number, number>();
  for (const paper of papers) {
    const year = paper.year ?? 0;
    counts.set(year, (counts.get(year) ?? 0) + 1);
  }
  // `Array.from`, not a spread: the production bundle lowers a spread of a Map
  // iterator to `concat`, which hands Math.max the iterator object and yields NaN.
  const deepest = Math.max(1, ...Array.from(counts.values()));
  return Math.max(24, deepest * STACK + 10);
}

/**
 * When each tradition published, on one shared year axis.
 *
 * A lane per tradition, a mark per dated paper, and colour only as a second
 * reading of the lane the mark already sits in. The strip answers one
 * question — which of these literatures is old and which is still moving —
 * and every mark is a link into the paper it stands for. Below the strip's
 * breaking width the same papers are listed instead; exactly one of the two
 * is in the accessibility tree at any width.
 */
export function YearStrip({core}: {core: Core}): React.JSX.Element | null {
  const rows = lanes(core);
  if (rows.length === 0) return null;
  const years = rows.flatMap((lane) => lane.papers.map((paper) => paper.year ?? 0));
  const low = Math.min(...years);
  const high = Math.max(...years);
  const x = linearScale([low, high], [GUTTER + 14, WIDTH - PAD_RIGHT - 14]);
  const ticks = niceTicks(low, high, 6);

  let cursor = TOP;
  const placed = rows.map((lane) => {
    const height = laneHeight(lane.papers);
    const centre = cursor + height / 2;
    cursor += height;
    return {lane, centre, top: centre - height / 2, bottom: centre + height / 2};
  });
  const height = cursor + AXIS;

  return (
    <figure className="atlas-figure">
      <div className="year-strip">
        <svg viewBox={`0 0 ${String(WIDTH)} ${String(height)}`} width={WIDTH} height={height} role="group" aria-label="Publication years of the cited literature, one lane per tradition">
          <title>Publication years of the cited literature, one lane per tradition</title>
          {placed.map(({lane, centre, bottom}) => (
            <g key={lane.slug}>
              <line className="year-strip__lane-rule" x1={GUTTER} x2={WIDTH - PAD_RIGHT} y1={bottom} y2={bottom} />
              <text className="year-strip__lane-label" x={GUTTER - 12} y={centre + 4} textAnchor="end">
                {lane.label}
              </text>
              {stacked(lane.papers).map(({paper, offset}) => (
                <Link key={paper.key} to={paperHref(paper.slug)}>
                  <circle
                    className={`year-strip__mark tradition-${lane.slug}`}
                    cx={x(paper.year ?? low)}
                    cy={centre + offset}
                    r={MARK}
                  >
                    <title>{describe(paper)}</title>
                  </circle>
                </Link>
              ))}
            </g>
          ))}
          <line className="year-strip__axis" x1={GUTTER} x2={WIDTH - PAD_RIGHT} y1={cursor} y2={cursor} />
          {ticks.map((tick) => (
            <text key={tick} className="year-strip__tick" x={x(tick)} y={cursor + 16} textAnchor="middle">
              {tick}
            </text>
          ))}
        </svg>
      </div>
      <p className="visually-hidden year-strip__alt">
        {`The strip places every dated source on one year axis, ${String(low)} to ${String(high)}, in a lane per tradition: `}
        {rows
          .map((lane) => {
            const span = lane.papers.map((paper) => paper.year ?? 0);
            return `${lane.label}, ${String(Math.min(...span))} to ${String(Math.max(...span))}, ${String(lane.papers.length)} sources`;
          })
          .join("; ")}
        .
      </p>
      <dl className="year-strip__lists">
        {rows.map((lane) => (
          <div key={lane.slug}>
            <dt>{lane.label}</dt>
            <dd>
              <ul>
                {lane.papers.map((paper) => (
                  <li key={paper.key}>
                    <Link to={paperHref(paper.slug)}>
                      {paper.authors.map((author) => author.name).join(", ")} {paper.year}
                    </Link>
                  </li>
                ))}
              </ul>
            </dd>
          </div>
        ))}
      </dl>
      <figcaption className="atlas-figure__caption">
        Each mark is one source, placed at its year of publication and coloured by the tradition
        whose lane it sits in.
      </figcaption>
    </figure>
  );
}
