import {useMemo} from "react";

import {extent, formatTick, linearScale} from "./charts/scale";

export interface BinningComparisonRow {
  /** Marks this row as the reference ceiling rather than a candidate binning. */
  isCeiling?: boolean;
  /** Marks this row as ScoreQuant's own result, so it reads as the one to notice. */
  isScoreQuant?: boolean;
  label: string;
  /** The generator's formatting of `value`. Rendered as-is; never reformatted here. */
  text: string;
  value: number;
}

export interface BinningComparisonProps {
  axisLabel: string;
  caption?: string;
  rows: readonly BinningComparisonRow[];
}

const WIDTH = 560;
const ROW_HEIGHT = 40;
const LABEL_WIDTH = 152;
const MARGIN = {bottom: 34, right: 16, top: 10};

/**
 * Beat 7's chart: what the reader's default binning retains against what
 * ScoreQuant's does, on the same data.
 *
 * Bars are drawn in row order rather than sorted, so a page can lead with the
 * baseline the reader would actually reach for. The ScoreQuant row gets the
 * accent fill; a `isCeiling` row (an unbinned or oracle reference) is drawn
 * as a vertical reference line instead of a competing bar, since it is not a
 * binning method the reader could choose.
 *
 * The SVG carries the whole story in its `<title>` (the chart's accessible
 * name); a visually-hidden table repeats every row as text for anyone who
 * cannot read the bars, including the axe scan's screen-reader check.
 */
export function BinningComparison({axisLabel, caption, rows}: BinningComparisonProps): React.JSX.Element {
  const barRows = rows.filter((row) => row.isCeiling !== true);
  const ceilingRow = rows.find((row) => row.isCeiling === true);
  const accessibleName = `${axisLabel}, compared across ${String(barRows.length)} binning methods`;

  const {plotWidth, scale} = useMemo(() => {
    const width = WIDTH - LABEL_WIDTH - MARGIN.right;
    // Leave room past the longest bar for its printed value, which otherwise
    // runs off the right edge when no ceiling row extends the axis.
    const [, high] = extent([0, ...rows.map((row) => row.value)]);
    const domain: [number, number] = [0, high * 1.14];
    return {plotWidth: width, scale: linearScale(domain, [0, width])};
  }, [rows]);

  const height = barRows.length * ROW_HEIGHT + MARGIN.top + MARGIN.bottom;
  const axisY = height - MARGIN.bottom;
  const originX = LABEL_WIDTH + scale(0);
  const ceilingX = ceilingRow === undefined ? undefined : LABEL_WIDTH + scale(ceilingRow.value);

  return (
    <figure className="chart-figure chart-figure--wide binning-comparison">
      <svg viewBox={`0 0 ${String(WIDTH)} ${String(height)}`} role="img">
        <title>{accessibleName}</title>
        <desc>
          {ceilingRow === undefined
            ? "Higher is better. The ScoreQuant row is drawn in the accent colour."
            : `Higher is better. The ScoreQuant row is drawn in the accent colour; the dashed line marks ${ceilingRow.label} at ${ceilingRow.text}.`}
        </desc>
        {barRows.map((row, index) => {
          const y = MARGIN.top + index * ROW_HEIGHT;
          const barWidth = Math.max(scale(row.value) - scale(0), 0);
          return (
            <g key={row.label}>
              <text
                className="binning-comparison__label"
                x={LABEL_WIDTH - 10}
                y={y + ROW_HEIGHT / 2 + 4}
                textAnchor="end"
              >
                {row.label}
              </text>
              <rect
                className={
                  row.isScoreQuant === true
                    ? "binning-comparison__bar binning-comparison__bar--scorequant"
                    : "binning-comparison__bar"
                }
                height={ROW_HEIGHT - 16}
                width={barWidth}
                x={originX}
                y={y + 8}
              />
              <text
                className="binning-comparison__value"
                x={originX + barWidth + 6}
                y={y + ROW_HEIGHT / 2 + 4}
              >
                {row.text}
              </text>
            </g>
          );
        })}
        {ceilingRow !== undefined && ceilingX !== undefined && (
          <>
            <line className="chart-reference" x1={ceilingX} x2={ceilingX} y1={MARGIN.top} y2={axisY} />
            <text className="chart-annotation" x={ceilingX - 6} y={MARGIN.top + 12} textAnchor="end">
              {`${ceilingRow.label}: ${ceilingRow.text}`}
            </text>
          </>
        )}
        <line className="chart-axis" x1={LABEL_WIDTH} x2={LABEL_WIDTH} y1={MARGIN.top} y2={axisY} />
        <line className="chart-axis" x1={LABEL_WIDTH} x2={WIDTH - MARGIN.right} y1={axisY} y2={axisY} />
        {scale.ticks(4).map((tick) => (
          <text
            key={`tick-${String(tick)}`}
            className="chart-tick"
            x={LABEL_WIDTH + scale(tick)}
            y={axisY + 16}
            textAnchor="middle"
          >
            {formatTick(tick)}
          </text>
        ))}
        <text
          className="chart-axis-label"
          x={LABEL_WIDTH + plotWidth / 2}
          y={height - 4}
          textAnchor="middle"
        >
          {axisLabel}
        </text>
      </svg>
      <table className="visually-hidden">
        <caption>{accessibleName}</caption>
        <thead>
          <tr>
            <th scope="col">Method</th>
            <th scope="col">{axisLabel}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <th scope="row">
                {row.label}
                {row.isScoreQuant === true ? " (ScoreQuant)" : ""}
                {row.isCeiling === true ? " (reference ceiling)" : ""}
              </th>
              <td>{row.text}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {caption !== undefined && <figcaption>{caption}</figcaption>}
    </figure>
  );
}
