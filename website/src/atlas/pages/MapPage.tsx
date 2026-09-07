import {ArgumentMap} from "../ArgumentMap";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps} from "../core";

/**
 * `/research/map/`: the argument graph of the whole research programme.
 */
export default function MapPage({core}: AtlasPageProps<Record<string, never>>): React.JSX.Element {
  return (
    <AtlasShell
      title="Graph"
      description="Every result and counterexample of the research, placed by problem level and criterion, with proof prerequisites drawn and a focus that shows what each result rests on, enables, and where it stops."
      wide
    >
      <h1>Research graph</h1>
      <p className="atlas__lead atlas__measure">
        Begin with one result: what it rests on, what it enables, and where it stops. Select a neighbour to continue the argument, or open the full graph for a
        wider view.
      </p>
      <ArgumentMap core={core} />
    </AtlasShell>
  );
}
