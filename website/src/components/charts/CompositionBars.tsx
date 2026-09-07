import {useMemo} from "react";

import {Legend} from "./Axes";
import {populationColor} from "./scale";
import type {PatientComposition} from "../../data/showcase";

interface CompositionBarsProps {
  patients: PatientComposition[];
  populations: string[];
  /**
   * Leave the last population out and draw the rest at their true fractions
   * on a shared axis. The dominant "other" compartment is nine tenths of every
   * patient, so a full-width stack shows the sampling design rather than the
   * variation the study measures.
   */
  targetsOnly?: boolean;
}

const WIDTH = 620;
const ROW_HEIGHT = 15;
const LABEL_WIDTH = 74;

/**
 * Per-patient population composition, one stacked bar per patient.
 *
 * This is the quantity the whole study estimates, so showing its spread across
 * patients up front is what makes the later error numbers interpretable: the
 * between-patient variation is much larger than any method's error.
 */
export function CompositionBars({patients, populations, targetsOnly = false}: CompositionBarsProps): React.JSX.Element {
  const rows = useMemo(() => {
    const shown = targetsOnly ? populations.length - 1 : populations.length;
    const span = targetsOnly
      ? Math.max(...patients.map((patient) => patient.fractions.slice(0, shown).reduce((sum, value) => sum + value, 0)), 1e-9)
      : 1;
    return patients.map((patient, index) => {
      let offset = 0;
      const segments = patient.fractions.slice(0, shown).map((fraction, population) => {
        const start = offset;
        offset += fraction;
        return {
          color: populationColor(population),
          label: populations[population] ?? String(population),
          value: fraction,
          width: (fraction / span) * (WIDTH - LABEL_WIDTH),
          x: LABEL_WIDTH + (start / span) * (WIDTH - LABEL_WIDTH),
        };
      });
      return {patient, segments, y: index * ROW_HEIGHT};
    });
  }, [patients, populations, targetsOnly]);
  const legendPopulations = targetsOnly ? populations.slice(0, -1) : populations;
  const height = rows.length * ROW_HEIGHT + 8;

  return (
    <figure className="chart-figure chart-figure--wide">
      <svg viewBox={`0 0 ${String(WIDTH)} ${String(height)}`} role="img">
        <title>Cell-population composition of every patient</title>
        <desc>
          {targetsOnly
            ? "One stacked bar per patient, split into the five target populations at their true fractions; the dominant remaining compartment is left out. Held-out patients are marked; reference patients are not."
            : "One stacked bar per patient, split into the six labelled populations. Held-out patients are marked; reference patients are not."}
        </desc>
        {rows.map((row) => (
          <g key={row.patient.patient}>
            <text className="chart-tick" x={LABEL_WIDTH - 6} y={row.y + 11} textAnchor="end">
              {`${row.patient.role === "held-out" ? "▸ " : ""}Case ${String(row.patient.patient)}`}
            </text>
            {row.segments.map((segment) => (
              <rect
                key={segment.label}
                x={segment.x}
                y={row.y + 2}
                width={Math.max(segment.width, 0)}
                height={ROW_HEIGHT - 4}
                fill={segment.color}
              >
                <title>{`Case ${String(row.patient.patient)} — ${segment.label}: ${(segment.value * 100).toFixed(1)}%`}</title>
              </rect>
            ))}
          </g>
        ))}
      </svg>
      <Legend entries={legendPopulations.map((name, index) => ({color: populationColor(index), label: name}))} />
      <figcaption>▸ marks the ten held-out patients, frozen before any fitting.</figcaption>
    </figure>
  );
}
