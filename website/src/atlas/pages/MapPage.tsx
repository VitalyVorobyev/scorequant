import {ArgumentMap} from "../ArgumentMap";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps} from "../core";

/**
 * `/research/map/`: the argument graph of the whole research programme.
 */
export default function MapPage({core}: AtlasPageProps<Record<string, never>>): React.JSX.Element {
  return (
    <AtlasShell
      title="Argument map"
      description="Every result and counterexample of the research, placed by problem level and criterion, with proof prerequisites drawn and a focus that shows what each result rests on, enables, and where it stops."
      wide
    >
      <h1>Argument map</h1>
      <p className="atlas__lead atlas__measure">
        Each mark is one result, question or counterexample. A column is a problem level, from identities that hold for any hard label to statements made for one
        application; a band is a criterion. A line from one mark to another means the first rests on the second in its proof. Select a mark to see its neighbourhood, then
        ask it a question.
      </p>
      <ArgumentMap core={core} />
    </AtlasShell>
  );
}
