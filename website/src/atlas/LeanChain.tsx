import type {FormalModule} from "../data/atlas";

const BOX_WIDTH = 104;
const BOX_HEIGHT = 54;
const GAP = 26;
const TOP = 6;

/** `ScalarExchange` reads as two lines in a narrow box; `Config` as one. */
function lines(module: string): string[] {
  const parts = module.split(/(?=[A-Z])/).filter((part) => part.length > 0);
  if (parts.length < 2) return [module];
  const head = parts[0] ?? module;
  return [head, parts.slice(1).join("")];
}

/**
 * The Lean build as a chain: each module proved on the modules before it.
 *
 * The figure carries the order and nothing else. What each module certifies is
 * named in the list beside it, in HTML, where the claim names are links and the
 * text is selectable and searchable.
 */
export function LeanChain({modules}: {modules: readonly FormalModule[]}): React.JSX.Element {
  const width = modules.length * BOX_WIDTH + Math.max(modules.length - 1, 0) * GAP;
  const midline = TOP + BOX_HEIGHT / 2;
  return (
    <svg className="lean-chain__figure" role="img" viewBox={`0 0 ${width} ${BOX_HEIGHT + TOP * 2}`} width={width} height={BOX_HEIGHT + TOP * 2}>
      <title>
        {`The Lean modules in build order: ${modules.map((module) => module.module).join(", ")}. Each rests on the modules before it.`}
      </title>
      <defs>
        <marker id="lean-chain-arrow" markerHeight="6" markerWidth="6" orient="auto" refX="5" refY="3">
          <path className="lean-chain__arrow" d="M0 0 L6 3 L0 6 Z" />
        </marker>
      </defs>
      {modules.map((module, index) => {
        const x = index * (BOX_WIDTH + GAP);
        const label = lines(module.module);
        return (
          <g key={module.module}>
            {index > 0 ? (
              <line
                className="lean-chain__link"
                markerEnd="url(#lean-chain-arrow)"
                x1={x - GAP + 3}
                x2={x - 7}
                y1={midline}
                y2={midline}
              />
            ) : null}
            <rect
              className={`lean-chain__box${module.partial === true ? " lean-chain__box--partial" : ""}`}
              height={BOX_HEIGHT}
              rx="5"
              width={BOX_WIDTH}
              x={x}
              y={TOP}
            />
            <text className="lean-chain__label" textAnchor="middle" x={x + BOX_WIDTH / 2} y={label.length > 1 ? midline - 4 : midline + 4}>
              {label[0]}
            </text>
            {label.length > 1 ? (
              <text className="lean-chain__label" textAnchor="middle" x={x + BOX_WIDTH / 2} y={midline + 13}>
                {label[1]}
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}
