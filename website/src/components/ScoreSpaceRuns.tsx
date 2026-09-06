import {useId, useMemo} from "react";

import {counterColor} from "./ApertureStrip";
import {formatTick, linearScale} from "./charts/scale";
import type {ApertureRun} from "../data/michelsonSweep";

/** One labelling of the aperture, drawn as its own score-space panel. */
export interface ScoreSpaceBand {
  label: string;
  runs: readonly ApertureRun[];
}

export interface ScoreSpaceRunsProps {
  bands: readonly ScoreSpaceBand[];
  description: string;
  title: string;
  /** Fringe visibility `V` of the model, from the committed sweep. */
  visibility: number;
  uMax: number;
}

const PANEL = 250;
const GAP = 26;
const MARGIN = {bottom: 40, left: 56, right: 12, top: 26};
/** Samples per run. Enough that the tightest loop near `u = 0` still reads as a curve. */
const SAMPLES_PER_RUN = 24;

/**
 * The Michelson score map at the reference point, in closed form.
 *
 * `s_phi = -V sin u / (1 + V cos u)` and `s_eps = u * s_phi - V`, exactly the
 * two expressions `examples/michelson_phase.py`'s `michelson_score` evaluates
 * and the walkthrough derives. Because it is closed form, a browser can draw
 * this panel from aperture runs alone -- there is no second generated file to
 * keep in step with the first.
 */
function scoreAt(u: number, visibility: number): [number, number] {
  const sPhi = (-visibility * Math.sin(u)) / (1 + visibility * Math.cos(u));
  return [sPhi, u * sPhi - visibility];
}

/**
 * One labelling of the score curve, drawn as small multiples: one panel per
 * band, all on a shared pair of axes.
 *
 * The aperture strip beside this one answers "where on the detector is counter
 * k?"; this answers "which part of the inference does counter k own?", which is
 * the space the optimizer actually works in. Two criteria that both comb the
 * aperture can still spend their cells on completely different directions of
 * the score, and that difference is invisible in the aperture view.
 *
 * Every run is drawn as its own polyline, so a run boundary is a colour change
 * on a continuous curve rather than a gap. Colours come from `counterColor`,
 * shared with `ApertureStrip`, so counter `k` is the same colour in both views
 * and on every panel.
 */
export function ScoreSpaceRuns({
  bands,
  description,
  title,
  uMax,
  visibility
}: ScoreSpaceRunsProps): React.JSX.Element {
  const titleId = useId();

  const {height, panelHeight, panelWidth, polylines, sEpsMax, sPhiMax, width, x, y} = useMemo(() => {
    // The curve's own extent, measured rather than assumed: the envelope of
    // `s_eps` grows with `uMax`, so a hard-coded domain would silently clip a
    // model with more fringes.
    const probe = Array.from({length: 2001}, (_, index) => scoreAt((uMax * index) / 2000, visibility));
    const sPhiMax = Math.max(...probe.map(([sPhi]) => Math.abs(sPhi)));
    const sEpsMax = Math.max(...probe.map(([, sEps]) => Math.abs(sEps)));

    const innerWidth = PANEL - MARGIN.left - MARGIN.right;
    const innerHeight = PANEL - MARGIN.top - MARGIN.bottom;
    const scaleX = linearScale([-sPhiMax * 1.08, sPhiMax * 1.08], [0, innerWidth]);
    const scaleY = linearScale([-sEpsMax * 1.08, sEpsMax * 1.08], [innerHeight, 0]);

    const lines = bands.map((band, bandIndex) => {
      const originX = bandIndex * (PANEL + GAP) + MARGIN.left;
      return {
        label: band.label,
        originX,
        runs: band.runs.map((run) => {
          const points: string[] = [];
          for (let sample = 0; sample <= SAMPLES_PER_RUN; sample += 1) {
            const u = run.start + ((run.end - run.start) * sample) / SAMPLES_PER_RUN;
            const [sPhi, sEps] = scoreAt(u, visibility);
            points.push(`${String(originX + scaleX(sPhi))},${String(MARGIN.top + scaleY(sEps))}`);
          }
          return {color: counterColor(run.label), points: points.join(" ")};
        })
      };
    });

    return {
      height: PANEL,
      panelHeight: innerHeight,
      panelWidth: innerWidth,
      polylines: lines,
      sEpsMax,
      sPhiMax,
      width: bands.length * PANEL + Math.max(bands.length - 1, 0) * GAP,
      x: scaleX,
      y: scaleY
    };
  }, [bands, uMax, visibility]);

  return (
    <figure className="chart-figure chart-figure--wide score-space-runs">
      <svg viewBox={`0 0 ${String(width)} ${String(height)}`} role="img" aria-labelledby={titleId} width="100%">
        <title id={titleId}>{title}</title>
        <desc>{description}</desc>
        <text
          className="chart-axis-label"
          textAnchor="middle"
          transform={`rotate(-90 12 ${String(MARGIN.top + panelHeight / 2)})`}
          x={12}
          y={MARGIN.top + panelHeight / 2}
        >
          fringe-frequency score
        </text>
        {polylines.map((band) => (
          <g key={band.label}>
            <text
              className="score-space-runs__label"
              x={band.originX + panelWidth / 2}
              y={MARGIN.top - 10}
              textAnchor="middle"
            >
              {band.label}
            </text>
            <line
              className="chart-grid"
              x1={band.originX}
              x2={band.originX + panelWidth}
              y1={MARGIN.top + y(0)}
              y2={MARGIN.top + y(0)}
            />
            <line
              className="chart-grid"
              x1={band.originX + x(0)}
              x2={band.originX + x(0)}
              y1={MARGIN.top}
              y2={MARGIN.top + panelHeight}
            />
            {band.runs.map((run, runIndex) => (
              <polyline
                key={`${band.label}-${String(runIndex)}`}
                className="score-space-runs__run"
                fill="none"
                points={run.points}
                stroke={run.color}
              />
            ))}
            <text
              className="chart-axis-label"
              x={band.originX + panelWidth / 2}
              y={height - 6}
              textAnchor="middle"
            >
              phase score
            </text>
            {/* Only the extremes are labelled. The reader needs the span of each
                axis to see that the fringe-frequency score runs some twenty times
                wider than the phase score; a full tick ladder on a panel this
                size would be noise. */}
            {[-sPhiMax, sPhiMax].map((value) => (
              <text
                key={`x-${String(value)}`}
                className="chart-tick"
                x={band.originX + x(value)}
                y={MARGIN.top + panelHeight + 16}
                textAnchor="middle"
              >
                {formatTick(value)}
              </text>
            ))}
            {[-sEpsMax, 0, sEpsMax].map((value) => (
              <text
                key={`y-${String(value)}`}
                className="chart-tick"
                x={band.originX - 8}
                y={MARGIN.top + y(value) + 4}
                textAnchor="end"
              >
                {formatTick(value)}
              </text>
            ))}
          </g>
        ))}
      </svg>
      <figcaption className="visually-hidden">{description}</figcaption>
    </figure>
  );
}
