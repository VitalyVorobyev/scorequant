import {useId} from "react";

export interface RatioWorkflowProps {
  caption: React.ReactNode;
}

const WIDTH = 1120;
const HEIGHT = 468;

const CARD_W = 190;
const CARD_H = 160;
const ROW_Y = 44;
const COLUMNS = [0, 232, 464, 696, 928] as const;

const ORACLE_Y = 300;
const ORACLE_H = 112;

// The crossing between the lanes is routed orthogonally -- down, across, down --
// so its label sits on the horizontal run instead of on top of a card.
const CROSSING_Y = 246;
const CORNER = 10;

/** One card of the top lane: an ordered step of the workflow a real analysis runs. */
interface Step {
  body: string[];
  kicker: string;
  title: string[];
  x: number;
}

/** One card of the bottom lane: a step available only because the model is synthetic. */
interface OracleStep {
  kicker: string;
  title: string[];
  width: number;
  x: number;
}

const STEPS: Step[] = [
  {kicker: "INPUT", title: ["Labelled", "simulations"], body: [], x: COLUMNS[0]},
  {kicker: "LEARN", title: ["Classifier"], body: ["calibrated", "posteriors"], x: COLUMNS[1]},
  {kicker: "CONVERT", title: ["Ratio to", "score"], body: [], x: COLUMNS[2]},
  {
    kicker: "COMPRESS",
    title: ["ScoreQuant"],
    body: ["K hard bins in", "the estimated score"],
    x: COLUMNS[3]
  },
  {kicker: "INFER", title: ["Fit θ"], body: ["from the bin", "counts alone"], x: COLUMNS[4]}
];

const ORACLE_STEPS: OracleStep[] = [
  {kicker: "ONLY IN THIS TOY MODEL", title: ["Exact densities"], width: 250, x: 232},
  {kicker: "REFERENCE", title: ["Exact score", "s(x; θ₀)"], width: 250, x: 520},
  {
    kicker: "EVALUATE",
    title: ["Achieved retention", "in the exact score"],
    width: 312,
    x: 808
  }
];

const COMPRESS_CENTRE = COLUMNS[3] + CARD_W / 2;
const ACHIEVED_CENTRE = 808 + 312 / 2;

/** A sampled normal bump, as an SVG polyline, for the labelled-simulations card. */
function bump(x: number, y: number, width: number, height: number, mu: number, sigma: number): string {
  return Array.from({length: 48}, (_, index) => {
    const t = index / 47;
    const value = Math.exp(-0.5 * ((t - mu) / sigma) ** 2);
    return `${(x + t * width).toFixed(1)},${(y + height - value * height).toFixed(1)}`;
  }).join(" ");
}

/** The y of a card's `index`-th title line, then of its `index`-th body line under them. */
function titleY(top: number, index: number): number {
  return top + 50 + index * 26;
}

function bodyY(top: number, titleLines: number, index: number): number {
  return top + 50 + titleLines * 26 + 6 + index * 20;
}

/**
 * The two lanes this walkthrough runs side by side, drawn without any data.
 *
 * Along the top, the route a real analysis can take: labelled simulations
 * train a classifier, its calibrated posteriors become a density ratio and
 * then a score, and ScoreQuant compresses that score into hard bins. Along
 * the bottom, the check only a synthetic model affords: the exact densities
 * give the exact score, and the bins fitted above are evaluated against it.
 * The single dashed link between the lanes is the whole point of the page --
 * the labels are the same, and only the score they are judged by differs.
 */
export function RatioWorkflow({caption}: RatioWorkflowProps): React.JSX.Element {
  const titleId = useId();
  const descId = useId();

  const rowMiddle = ROW_Y + CARD_H / 2;
  const oracleMiddle = ORACLE_Y + ORACLE_H / 2;
  const inputX = COLUMNS[0];
  const convertX = COLUMNS[2];
  const compressX = COLUMNS[3];
  const inferX = COLUMNS[4];

  return (
    <figure className="chart-figure ratio-workflow">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-labelledby={`${titleId} ${descId}`}
        width="100%"
      >
        <title id={titleId}>From a classifier to Fisher-preserving bins</title>
        <desc id={descId}>
          Two lanes. The upper lane runs left to right through five steps: labelled
          simulations, a classifier, the conversion of its posteriors into a density ratio
          and then a score, ScoreQuant compressing that score into K hard bins, and finally
          the fit of the signal fraction. The lower lane, marked as available only in this
          synthetic model, runs through the exact densities, the exact score, and the
          retention those same fitted bins achieve against it. A dashed arrow carries the
          fitted bins from the upper lane down into the lower one.
        </desc>

        <defs>
          <marker
            id={`${titleId}-arrow`}
            markerWidth="10"
            markerHeight="10"
            refX="8"
            refY="5"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path className="ratio-workflow__arrowhead" d="M0,0 L9,5 L0,10 z" />
          </marker>
          <marker
            id={`${titleId}-oracle-arrow`}
            markerWidth="10"
            markerHeight="10"
            refX="8"
            refY="5"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path className="ratio-workflow__arrowhead--oracle" d="M0,0 L9,5 L0,10 z" />
          </marker>
        </defs>

        <text className="ratio-workflow__lane" x={0} y={22}>
          ANALYSIS WORKFLOW
        </text>

        {STEPS.map((step, index) => (
          <g key={step.kicker} data-testid="workflow-card">
            <rect
              className={index === 0 ? "ratio-workflow__card--input" : "ratio-workflow__card"}
              x={step.x}
              y={ROW_Y}
              width={CARD_W}
              height={CARD_H}
              rx={10}
            />
            <text className="ratio-workflow__kicker" x={step.x + 14} y={ROW_Y + 24}>
              {step.kicker}
            </text>
            {step.title.map((line, lineIndex) => (
              <text
                key={line}
                className="ratio-workflow__title"
                x={step.x + 14}
                y={titleY(ROW_Y, lineIndex)}
              >
                {line}
              </text>
            ))}
            {step.body.map((line, lineIndex) => (
              <text
                key={line}
                className="ratio-workflow__body"
                x={step.x + 14}
                y={bodyY(ROW_Y, step.title.length, lineIndex)}
              >
                {line}
              </text>
            ))}
          </g>
        ))}

        {/* The two component densities, sketched inside the input card. */}
        <polyline
          className="ratio-workflow__signal"
          points={bump(inputX + 16, ROW_Y + 106, CARD_W - 32, 34, 0.66, 0.13)}
        />
        <polyline
          className="ratio-workflow__background"
          points={bump(inputX + 16, ROW_Y + 106, CARD_W - 32, 34, 0.36, 0.24)}
        />

        {/* The conversion the provider performs, written out inside its card. */}
        <text className="ratio-workflow__formula" x={convertX + 14} y={ROW_Y + 112}>
          r&#770;(x) &#8594; s&#770;(x; &#952;&#8320;)
        </text>
        <text className="ratio-workflow__body" x={convertX + 14} y={ROW_Y + 136}>
          posterior odds
        </text>
        <text className="ratio-workflow__body" x={convertX + 14} y={ROW_Y + 155}>
          over prior odds
        </text>

        {/* Four bin boundaries on a score axis, sketched inside the ScoreQuant card. */}
        <line
          className="ratio-workflow__axis"
          x1={compressX + 16}
          y1={ROW_Y + 140}
          x2={compressX + CARD_W - 16}
          y2={ROW_Y + 140}
        />
        {[0.25, 0.5, 0.75].map((fraction) => (
          <line
            key={fraction}
            className="ratio-workflow__cut"
            x1={compressX + 16 + fraction * (CARD_W - 32)}
            y1={ROW_Y + 124}
            x2={compressX + 16 + fraction * (CARD_W - 32)}
            y2={ROW_Y + 144}
          />
        ))}

        {/* A likelihood curve with its maximum marked, inside the inference card. */}
        <path
          className="ratio-workflow__signal"
          d={`M${inferX + 20},${ROW_Y + 144} Q${inferX + CARD_W / 2},${ROW_Y + 104} ${inferX + CARD_W - 20},${ROW_Y + 144}`}
        />
        <line
          className="ratio-workflow__cut"
          x1={inferX + CARD_W / 2}
          y1={ROW_Y + 116}
          x2={inferX + CARD_W / 2}
          y2={ROW_Y + 146}
        />

        {COLUMNS.slice(0, -1).map((column, index) => (
          <line
            key={column}
            className="ratio-workflow__flow"
            markerEnd={`url(#${titleId}-arrow)`}
            x1={column + CARD_W + 4}
            y1={rowMiddle}
            x2={(COLUMNS[index + 1] ?? 0) - 12}
            y2={rowMiddle}
          />
        ))}

        <text className="ratio-workflow__note" x={COLUMNS[1] + CARD_W + 21} y={228} textAnchor="middle">
          classifier quality enters here
        </text>

        <text className="ratio-workflow__lane" x={0} y={274}>
          CONTROLLED ORACLE CHECK &#183; SYNTHETIC MODEL ONLY
        </text>

        {ORACLE_STEPS.map((step) => (
          <g key={step.kicker} data-testid="oracle-card">
            <rect
              className="ratio-workflow__card--oracle"
              x={step.x}
              y={ORACLE_Y}
              width={step.width}
              height={ORACLE_H}
              rx={10}
            />
            <text className="ratio-workflow__kicker" x={step.x + 14} y={ORACLE_Y + 26}>
              {step.kicker}
            </text>
            {step.title.map((line, lineIndex) => (
              <text
                key={line}
                className="ratio-workflow__title"
                x={step.x + 14}
                y={ORACLE_Y + 56 + lineIndex * 26}
              >
                {line}
              </text>
            ))}
          </g>
        ))}

        {ORACLE_STEPS.slice(0, -1).map((step, index) => (
          <line
            key={step.kicker}
            className="ratio-workflow__flow--oracle"
            markerEnd={`url(#${titleId}-oracle-arrow)`}
            x1={step.x + step.width + 4}
            y1={oracleMiddle}
            x2={(ORACLE_STEPS[index + 1]?.x ?? 0) - 12}
            y2={oracleMiddle}
          />
        ))}

        {/* The one crossing between the lanes: the bins fitted above are the bins judged below. */}
        <path
          className="ratio-workflow__flow--oracle"
          markerEnd={`url(#${titleId}-oracle-arrow)`}
          d={
            `M${COMPRESS_CENTRE},${ROW_Y + CARD_H + 4}` +
            ` L${COMPRESS_CENTRE},${CROSSING_Y - CORNER}` +
            ` Q${COMPRESS_CENTRE},${CROSSING_Y} ${COMPRESS_CENTRE + CORNER},${CROSSING_Y}` +
            ` L${ACHIEVED_CENTRE - CORNER},${CROSSING_Y}` +
            ` Q${ACHIEVED_CENTRE},${CROSSING_Y} ${ACHIEVED_CENTRE},${CROSSING_Y + CORNER}` +
            ` L${ACHIEVED_CENTRE},${ORACLE_Y - 12}`
          }
        />
        <text
          className="ratio-workflow__note"
          x={(COMPRESS_CENTRE + ACHIEVED_CENTRE) / 2}
          y={CROSSING_Y - 14}
          textAnchor="middle"
        >
          the same fitted bins
        </text>

        <text className="ratio-workflow__note" x={0} y={446}>
          Top: the workflow a real analysis runs. Bottom: the oracle check, available here only
          because the synthetic densities are known.
        </text>
      </svg>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
