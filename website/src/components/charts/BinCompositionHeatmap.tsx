import {populationColor} from "./scale";
import type {BinComposition} from "../../data/showcase";

interface BinCompositionHeatmapProps {
  composition: BinComposition;
}

/**
 * What each learned category is, without projecting the partition.
 *
 * One row per bin, one cell per population, shaded by the reference
 * composition `P(population | bin)`: the probability that a cell landing in
 * that bin belongs to each population, at the reference mixture. A row that
 * is one dark cell is a category that acts as a gate for one population; a
 * row of pale cells is a mixed category, kept because it separates the
 * mixture in a direction no single gate covers. The numbers are the
 * generator's own text, never reformatted here.
 *
 * The table carries `tabIndex` for the same reason `theme/MDXComponents.tsx`
 * puts one on every Markdown table: the portal makes a wide table its own
 * horizontal scroll container, and a region that scrolls has to be reachable
 * by keyboard. The `<caption>` names it.
 */
export function BinCompositionHeatmap({composition}: BinCompositionHeatmapProps): React.JSX.Element {
  return (
    <figure className="chart-figure chart-figure--wide bin-composition">
      <table className="bin-composition__table" tabIndex={0}>
        <caption className="visually-hidden">
          Reference population composition of each learned bin, as probabilities that sum to one across a row.
        </caption>
        <thead>
          <tr>
            <th scope="col">Bin</th>
            {composition.populations.map((population, index) => (
              <th key={population} scope="col">
                <span aria-hidden="true" className="chart-swatch" style={{background: populationColor(index)}} />
                {population}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {composition.rows.map((row) => (
            <tr key={row.bin}>
              <th scope="row">{`bin ${String(row.bin)}`}</th>
              {row.composition.map((value, index) => (
                <td
                  key={composition.populations[index] ?? String(index)}
                  className={value >= 0.5 ? "bin-composition__cell bin-composition__cell--dominant" : "bin-composition__cell"}
                  style={{"--cell-strength": String(Math.min(1, value))} as React.CSSProperties}
                >
                  {value.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <figcaption>
        Each row is one of the {String(composition.bins)} learned categories; each entry is the probability
        that a cell in that category belongs to the column population, at the reference composition.
      </figcaption>
    </figure>
  );
}
