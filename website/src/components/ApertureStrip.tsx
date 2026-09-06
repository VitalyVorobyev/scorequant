import {useId, useMemo} from "react";

import {linearScale} from "./charts/scale";
import type {ApertureRun} from "../data/michelsonSweep";

/** One labelling's runs, drawn as one horizontal band with a left-hand label. */
export interface ApertureBand {
  label: string;
  runs: readonly ApertureRun[];
}

export interface ApertureStripProps {
  bands: readonly ApertureBand[];
  description: string;
  fringes: number;
  title: string;
  uMax: number;
}

const WIDTH = 560;
const BAND_HEIGHT = 40;
const LABEL_WIDTH = 152;
const MARGIN = {bottom: 34, right: 16, top: 10};

/**
 * A categorical colour per counter (bin label), ordered so the same counter
 * index always draws the same colour on every band and every strip on the
 * page.
 *
 * The first four entries are the site's own semantic tokens -- `--accent`,
 * `--warn`, `--bad`, `--ink-700` -- which already carry independent
 * light/dark values; re-measured here against both themes'
 * `--surface-raised` (white on light, `#0b1c38` on dark) they read
 * 4.82:1 / 7.72:1, 3.72:1 / 7.66:1, 5.46:1 / 8.65:1 and 10.58:1 / 9.14:1.
 * `tokens.css` has only those four distinct hues, though, and a sweep can
 * reach ten counters, so the remaining slots are explicit hex fallbacks --
 * fixed colours rather than tokens, since a fixed hue at a fixed lightness
 * has one contrast ratio in both themes by construction. Each was chosen at
 * a relative luminance of about 0.20-0.21 (WCAG's own formula, the same one
 * `tests/contrast.test.ts` runs), the band that clears 3:1 against a white
 * ground from above and a `#0b1c38` ground from below at once; every
 * fallback below measures at least 4.0:1 against each.
 */
const COUNTER_COLORS: readonly string[] = [
  "var(--accent)",
  "var(--warn)",
  "var(--bad)",
  "var(--ink-700)",
  "#178a8b",
  "#189136",
  "#7b6ce7",
  "#c937de",
  "#1f84c0",
  "#dd2db1",
  "#518c17",
  "#a05be4",
  "#5276e3",
  "#e03d7a",
  "#997919",
  "#c75f20"
] as const;

/** The colour for one counter index, wrapping rather than returning undefined. */
export function counterColor(label: number): string {
  return COUNTER_COLORS[label % COUNTER_COLORS.length] ?? "#8ea2ba";
}

/**
 * Convert one labelling of `n` equally spaced aperture nodes into aperture
 * runs, merging consecutive nodes that share a label.
 *
 * Node `i` of `n` spans `[i * uMax / n, (i + 1) * uMax / n]` -- the same
 * midpoint-quadrature tiling the fixed-table task and the committed sweep
 * both use, so a live re-fit's labels draw on exactly the geometry the
 * committed bands were drawn on.
 */
export function runsFromLabels(labels: readonly number[], uMax: number): ApertureRun[] {
  const n = labels.length;
  const first = labels[0];
  if (first === undefined) return [];
  const step = uMax / n;
  const runs: ApertureRun[] = [];
  let runStart = 0;
  let runLabel = first;
  for (let i = 1; i < n; i += 1) {
    const label = labels[i];
    if (label === undefined || label === runLabel) continue;
    runs.push({end: i * step, label: runLabel, start: runStart * step});
    runStart = i;
    runLabel = label;
  }
  runs.push({end: n * step, label: runLabel, start: runStart * step});
  return runs;
}

/**
 * One or more labellings of a bounded aperture, drawn as stacked horizontal
 * bands of coloured runs against a shared `u` axis marked in fringes.
 *
 * Used on the Michelson walkthrough to show what a labelling of the aperture
 * actually looks like -- a comb of intervals per counter rather than a single
 * contiguous segment -- for both the committed sweep and a reader's own
 * browser re-fit. Every run in `bands` is drawn as exactly one `<rect>`;
 * nothing is merged here; a caller that wants merged runs (a raw labels
 * array) uses `runsFromLabels` first.
 */
export function ApertureStrip({bands, description, fringes, title, uMax}: ApertureStripProps): React.JSX.Element {
  const titleId = useId();

  const {plotWidth, scale} = useMemo(() => {
    const width = WIDTH - LABEL_WIDTH - MARGIN.right;
    return {plotWidth: width, scale: linearScale([0, uMax], [0, width])};
  }, [uMax]);

  const height = bands.length * BAND_HEIGHT + MARGIN.top + MARGIN.bottom;
  const axisY = height - MARGIN.bottom;
  const fringeTicks = Array.from({length: fringes + 1}, (_, index) => index);

  return (
    <figure className="chart-figure chart-figure--wide aperture-strip">
      <svg viewBox={`0 0 ${String(WIDTH)} ${String(height)}`} role="img" aria-labelledby={titleId} width="100%">
        <title id={titleId}>{title}</title>
        <desc>{description}</desc>
        {fringeTicks.map((fringe) => {
          const x = LABEL_WIDTH + scale((fringe * uMax) / fringes);
          return <line key={`grid-${String(fringe)}`} className="chart-grid" x1={x} x2={x} y1={MARGIN.top} y2={axisY} />;
        })}
        {bands.map((band, bandIndex) => {
          const y = MARGIN.top + bandIndex * BAND_HEIGHT;
          return (
            <g key={band.label}>
              <text className="aperture-strip__label" x={LABEL_WIDTH - 10} y={y + BAND_HEIGHT / 2 + 4} textAnchor="end">
                {band.label}
              </text>
              {band.runs.map((run, runIndex) => (
                <rect
                  key={`${band.label}-${String(runIndex)}`}
                  className="aperture-strip__run"
                  x={LABEL_WIDTH + scale(run.start)}
                  y={y + 8}
                  width={Math.max(scale(run.end) - scale(run.start), 0)}
                  height={BAND_HEIGHT - 16}
                  fill={counterColor(run.label)}
                />
              ))}
            </g>
          );
        })}
        <line className="chart-axis" x1={LABEL_WIDTH} x2={WIDTH - MARGIN.right} y1={axisY} y2={axisY} />
        {fringeTicks.map((fringe) => (
          <text
            key={`tick-${String(fringe)}`}
            className="chart-tick"
            x={LABEL_WIDTH + scale((fringe * uMax) / fringes)}
            y={axisY + 16}
            textAnchor="middle"
          >
            {String(fringe)}
          </text>
        ))}
        <text className="chart-axis-label" x={LABEL_WIDTH + plotWidth / 2} y={height - 4} textAnchor="middle">
          Position along the aperture, in fringes
        </text>
      </svg>
      <table className="visually-hidden">
        <caption>{title}</caption>
        <thead>
          <tr>
            <th scope="col">Band</th>
            <th scope="col">Start (u)</th>
            <th scope="col">End (u)</th>
            <th scope="col">Counter</th>
          </tr>
        </thead>
        <tbody>
          {bands.flatMap((band) =>
            band.runs.map((run, runIndex) => (
              <tr key={`${band.label}-${String(runIndex)}`}>
                <th scope="row">{band.label}</th>
                <td>{run.start.toFixed(4)}</td>
                <td>{run.end.toFixed(4)}</td>
                <td>{String(run.label)}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </figure>
  );
}
