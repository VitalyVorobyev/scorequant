import {useMemo} from "react";

import {Axes, DEFAULT_FRAME, Legend, plotArea} from "./Axes";
import {linearScale, populationColor} from "./scale";
import type {PatientEstimate} from "../../data/showcase";

const FRAME = {...DEFAULT_FRAME, height: 340, marginLeft: 64, width: 520};

interface EstimateScatterProps {
  /** Populations to draw, by index; the dominant "other" compartment is usually left out. */
  populations: string[];
  patients: PatientEstimate[];
  visible: readonly number[];
}

/**
 * Held-out patients: the fitted fraction of each population against the
 * expert's composition, on logarithmic axes because the populations span
 * three decades of abundance.
 *
 * Filled marks are the estimates from the eight counts; hollow marks are the
 * unbinned classifier-ratio estimates of the same patients. The diagonal is
 * perfect agreement. The point of the chart is that both sets of marks sit
 * on the diagonal at every abundance, and that the distance between a
 * filled mark and its hollow twin is small compared with the spread of the
 * marks along the line.
 */
export function EstimateScatter({patients, populations, visible}: EstimateScatterProps): React.JSX.Element {
  const area = plotArea(FRAME);
  const {marks, x, y} = useMemo(() => {
    const values: number[] = [];
    for (const patient of patients) {
      for (const index of visible) {
        values.push(patient.expert[index] ?? 0, patient.binned[index] ?? 0, patient.unbinned[index] ?? 0);
      }
    }
    const floor = Math.max(1e-5, Math.min(...values.filter((value) => value > 0)));
    const ceiling = Math.max(...values);
    const domain: [number, number] = [Math.log10(floor) - 0.2, Math.log10(ceiling) + 0.2];
    const xScale = linearScale(domain, [area.left, area.right]);
    const yScale = linearScale(domain, [area.bottom, area.top]);
    const clamp = (value: number): number => Math.log10(Math.max(value, floor));
    const built = patients.flatMap((patient) =>
      visible.flatMap((index) => {
        const expert = clamp(patient.expert[index] ?? 0);
        return [
          {
            color: populationColor(index),
            cx: xScale(expert),
            cy: yScale(clamp(patient.binned[index] ?? 0)),
            filled: true,
            key: `${String(patient.patient)}-${String(index)}-binned`,
            title: `Case ${String(patient.patient)}, ${populations[index] ?? ""}: expert ${(patient.expert[index] ?? 0).toFixed(4)}, from counts ${(patient.binned[index] ?? 0).toFixed(4)}`,
          },
          {
            color: populationColor(index),
            cx: xScale(expert),
            cy: yScale(clamp(patient.unbinned[index] ?? 0)),
            filled: false,
            key: `${String(patient.patient)}-${String(index)}-unbinned`,
            title: `Case ${String(patient.patient)}, ${populations[index] ?? ""}: expert ${(patient.expert[index] ?? 0).toFixed(4)}, unbinned ${(patient.unbinned[index] ?? 0).toFixed(4)}`,
          },
        ];
      })
    );
    return {marks: built, x: xScale, y: yScale};
  }, [area.bottom, area.left, area.right, area.top, patients, populations, visible]);

  const [low, high] = x.domain;
  return (
    <figure className="chart-figure chart-figure--wide">
      <svg viewBox={`0 0 ${String(FRAME.width)} ${String(FRAME.height)}`} role="img">
        <title>Estimated against expert population fractions for the held-out patients</title>
        <desc>
          Log-log scatter of the fitted fraction of each population against the expert composition, one mark
          per held-out patient and population. Filled marks are the estimates from the bin counts; hollow marks
          are the unbinned classifier-ratio estimates. The diagonal is perfect agreement.
        </desc>
        <Axes
          frame={FRAME}
          x={x}
          xLabel="log₁₀ expert fraction"
          xTickCount={4}
          y={y}
          yLabel="log₁₀ estimated fraction"
          yTickCount={4}
        />
        <line className="chart-reference" x1={x(low)} x2={x(high)} y1={y(low)} y2={y(high)} />
        {marks.map((mark) => (
          <circle
            key={mark.key}
            cx={mark.cx}
            cy={mark.cy}
            r={mark.filled ? 4 : 4.5}
            fill={mark.filled ? mark.color : "none"}
            stroke={mark.color}
            strokeWidth={mark.filled ? 0 : 1.4}
          >
            <title>{mark.title}</title>
          </circle>
        ))}
      </svg>
      <Legend entries={visible.map((index) => ({color: populationColor(index), label: populations[index] ?? String(index)}))} />
      <figcaption>Filled: fitted from the eight counts. Hollow: fitted from the unbinned classifier ratios. An estimate of exactly zero is drawn on the lower edge.</figcaption>
    </figure>
  );
}
