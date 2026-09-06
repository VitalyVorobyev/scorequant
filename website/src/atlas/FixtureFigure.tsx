import {extent, formatTick, linearScale, niceTicks} from "../components/charts/scale";

/**
 * A fixture coordinate, exactly as the registry stores it.
 *
 * Counterexamples are certified in rational arithmetic, so a score or a weight
 * arrives either as a JSON number or as a string holding an exact fraction
 * (`"-47/64"`). Nothing on this page rounds a stored value for display; the
 * parser exists only to place a mark on a picture.
 */
export function rational(value: string | number): number {
  if (typeof value === "number") return value;
  const text = value.trim();
  const slash = text.indexOf("/");
  if (slash === -1) return Number(text);
  const numerator = Number(text.slice(0, slash));
  const denominator = Number(text.slice(slash + 1));
  if (denominator === 0) return Number.NaN;
  return numerator / denominator;
}

export interface Point {
  x: number;
  y: number;
}

/**
 * The convex hull of a point set, counter-clockwise, by Andrew's monotone
 * chain. Duplicated points are dropped first, so a cell whose rows coincide
 * returns the single point rather than a degenerate polygon.
 */
export function convexHull(points: readonly Point[]): Point[] {
  const seen = new Set<string>();
  const unique: Point[] = [];
  for (const point of points) {
    const key = `${String(point.x)},${String(point.y)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(point);
  }
  if (unique.length < 3) return unique;
  const sorted = [...unique].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o: Point, a: Point, b: Point): number => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const half = (input: Point[]): Point[] => {
    const chain: Point[] = [];
    for (const point of input) {
      while (chain.length >= 2) {
        const last = chain[chain.length - 1];
        const before = chain[chain.length - 2];
        if (!last || !before || cross(before, last, point) > 0) break;
        chain.pop();
      }
      chain.push(point);
    }
    chain.pop();
    return chain;
  };
  const hull = [...half(sorted), ...half([...sorted].reverse())];
  return hull.length < 3 ? unique : hull;
}

export interface FixtureFigureProps {
  dimension: number;
  id: string;
  labelsAfter: number[] | null;
  labelsBefore: number[];
  /** Rows of exact coordinates; a one-dimensional fixture may be flat. */
  scores: (string | number)[][] | (string | number)[];
  weights: (string | number)[];
}

const PANEL_W = 320;
const PANEL_H = 250;
const STRIP_H = 130;
const MARGIN = {bottom: 34, left: 42, right: 14, top: 14};
const R_MAX = 9;
const R_MIN = 2.6;

/** `scores` as rows of numbers, whatever shape the fixture stored. */
export function coordinates(scores: FixtureFigureProps["scores"]): number[][] {
  return scores.map((row) => (Array.isArray(row) ? row.map(rational) : [rational(row)]));
}

function cellClass(label: number): string {
  return `cell-${String((((label % 8) + 8) % 8) + 1)}`;
}

interface Panel {
  caption: string;
  key: string;
  labels: number[];
}

function panelsOf(labelsBefore: number[], labelsAfter: number[] | null): Panel[] {
  const before: Panel = {key: "before", caption: "Labels before", labels: labelsBefore};
  if (labelsAfter === null) return [before];
  return [before, {key: "after", caption: "Labels after / optimum", labels: labelsAfter}];
}

/**
 * The atoms of a counterexample, drawn.
 *
 * One panel per labelling the fixture records — the assignment it starts from,
 * and the optimum it moves to — on a shared frame, so the two are read against
 * each other rather than each against itself. A mark's area is its weight and
 * its colour is its cell; each cell's convex hull is drawn behind its marks,
 * because what these fixtures usually show is two hulls that overlap, which is
 * exactly what a nearest-cell rule cannot produce. Above two score dimensions
 * there is nothing honest to draw, and the coordinates are tabulated instead.
 */
export function FixtureFigure({dimension, id, labelsAfter, labelsBefore, scores, weights}: FixtureFigureProps): React.JSX.Element {
  const rows = coordinates(scores);
  const mass = weights.map(rational);
  const panels = panelsOf(labelsBefore, labelsAfter);
  const cells = Array.from(new Set(panels.flatMap((panel) => panel.labels))).sort((a, b) => a - b);

  if (dimension > 2) {
    return (
      <>
        <CoordinateTable id={id} labelsAfter={labelsAfter} labelsBefore={labelsBefore} rows={rows} weights={weights} scores={scores} />
        <p className="atlas__muted">
          {`The fixture lives in ${String(dimension)} score dimensions, so its atoms are listed rather than drawn.`}
        </p>
      </>
    );
  }

  const flat = dimension < 2;
  const xs = rows.map((row) => row[0] ?? 0);
  const ys = flat ? [0] : rows.map((row) => row[1] ?? 0);
  const xDomain = extent(xs);
  const yDomain = extent(ys);
  const height = flat ? STRIP_H : PANEL_H;
  const area = {
    left: MARGIN.left,
    right: PANEL_W - MARGIN.right,
    top: MARGIN.top,
    bottom: height - MARGIN.bottom
  };
  const pad = (domain: [number, number]): [number, number] => {
    const slack = (domain[1] - domain[0]) * 0.12;
    return [domain[0] - slack, domain[1] + slack];
  };
  const x = linearScale(pad(xDomain), [area.left, area.right]);
  const y = flat
    ? linearScale([-1, 1], [area.bottom, area.top])
    : linearScale(pad(yDomain), [area.bottom, area.top]);
  const heaviest = Math.max(...mass, 0);
  const radius = (index: number): number => {
    const weight = mass[index] ?? 0;
    if (heaviest <= 0) return R_MIN;
    return Math.max(R_MIN, R_MAX * Math.sqrt(Math.max(weight, 0) / heaviest));
  };
  const place = (index: number): Point => ({
    x: x(rows[index]?.[0] ?? 0),
    y: flat ? y(0) : y(rows[index]?.[1] ?? 0)
  });

  return (
    <>
      <div className="fixture-figure">
        {panels.map((panel) => (
          <figure className="fixture-panel" key={panel.key}>
            <svg viewBox={`0 0 ${String(PANEL_W)} ${String(height)}`} role="img">
              <title>{`${id}: ${panel.caption.toLowerCase()}, ${String(rows.length)} atoms in ${String(new Set(panel.labels).size)} cells`}</title>
              <rect className="fixture-panel__frame" x={area.left} y={area.top} width={area.right - area.left} height={area.bottom - area.top} />
              {flat ? (
                <line className="fixture-panel__strip" x1={area.left} x2={area.right} y1={y(0)} y2={y(0)} />
              ) : (
                niceTicks(yDomain[0], yDomain[1], 3).map((tick) => (
                  <text key={`y${String(tick)}`} className="fixture-panel__tick" x={area.left - 6} y={y(tick) + 3} textAnchor="end">
                    {formatTick(tick)}
                  </text>
                ))
              )}
              {niceTicks(xDomain[0], xDomain[1], 3).map((tick) => (
                <text key={`x${String(tick)}`} className="fixture-panel__tick" x={x(tick)} y={area.bottom + 13} textAnchor="middle">
                  {formatTick(tick)}
                </text>
              ))}
              {cells.map((cell) => (
                <Hull key={cell} cell={cell} points={indicesOf(panel.labels, cell).map(place)} />
              ))}
              {rows.map((_, index) => (
                <circle
                  key={index}
                  className={`fixture-panel__point ${cellClass(panel.labels[index] ?? 0)}`}
                  cx={place(index).x}
                  cy={place(index).y}
                  r={radius(index)}
                />
              ))}
              <text className="fixture-panel__axis-label" x={(area.left + area.right) / 2} y={height - 4} textAnchor="middle">
                score 1
              </text>
              {flat ? null : (
                <text
                  className="fixture-panel__axis-label"
                  textAnchor="middle"
                  transform={`translate(11 ${String((area.top + area.bottom) / 2)}) rotate(-90)`}
                >
                  score 2
                </text>
              )}
            </svg>
            <figcaption>{panel.caption}</figcaption>
          </figure>
        ))}
      </div>
      <ul className="fixture-legend">
        {cells.map((cell) => (
          <li key={cell}>
            <svg viewBox="0 0 10 10" aria-hidden="true">
              <circle className={cellClass(cell)} cx="5" cy="5" r="5" />
            </svg>
            {`Cell ${String(cell)}`}
          </li>
        ))}
      </ul>
      <CoordinateTable hidden id={id} labelsAfter={labelsAfter} labelsBefore={labelsBefore} rows={rows} weights={weights} scores={scores} />
    </>
  );
}

function indicesOf(labels: readonly number[], cell: number): number[] {
  const out: number[] = [];
  for (const [index, label] of labels.entries()) if (label === cell) out.push(index);
  return out;
}

/** A cell's hull: a polygon, a segment for two atoms, nothing for one. */
function Hull({cell, points}: {cell: number; points: Point[]}): React.JSX.Element | null {
  const hull = convexHull(points);
  if (hull.length < 2) return null;
  const first = hull[0];
  const last = hull[hull.length - 1];
  if (hull.length === 2 && first && last) {
    return (
      <line
        className={`fixture-panel__hull fixture-panel__hull--line ${cellClass(cell)}`}
        x1={first.x}
        x2={last.x}
        y1={first.y}
        y2={last.y}
      />
    );
  }
  return (
    <polygon
      className={`fixture-panel__hull ${cellClass(cell)}`}
      points={hull.map((point) => `${String(point.x)},${String(point.y)}`).join(" ")}
    />
  );
}

interface CoordinateTableProps {
  hidden?: boolean;
  id: string;
  labelsAfter: number[] | null;
  labelsBefore: number[];
  rows: number[][];
  scores: FixtureFigureProps["scores"];
  weights: (string | number)[];
}

/** The atoms as exact text: the drawing's alternative, and the picture above two dimensions. */
function CoordinateTable({hidden = false, id, labelsAfter, labelsBefore, rows, scores, weights}: CoordinateTableProps): React.JSX.Element {
  const width = Math.max(...rows.map((row) => row.length), 1);
  const exact = (index: number, column: number): string => {
    const row = scores[index];
    const value = Array.isArray(row) ? row[column] : column === 0 ? row : undefined;
    return value === undefined ? "" : String(value);
  };
  const labels = labelsAfter;
  return (
    <table className={hidden ? "visually-hidden" : "fixture-coords"}>
      <caption className={hidden ? undefined : "visually-hidden"}>{`Atoms of ${id}`}</caption>
      <thead>
        <tr>
          <th scope="col">Atom</th>
          <th scope="col">Weight</th>
          {Array.from({length: width}, (_, column) => (
            <th key={column} scope="col">{`Score ${String(column + 1)}`}</th>
          ))}
          <th scope="col">Cell before</th>
          {labels ? <th scope="col">Cell after</th> : null}
        </tr>
      </thead>
      <tbody>
        {rows.map((_, index) => (
          <tr key={index}>
            <th scope="row">{index + 1}</th>
            <td>
              <code>{String(weights[index] ?? "")}</code>
            </td>
            {Array.from({length: width}, (_, column) => (
              <td key={column}>
                <code>{exact(index, column)}</code>
              </td>
            ))}
            <td>{labelsBefore[index]}</td>
            {labels ? <td>{labels[index]}</td> : null}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
