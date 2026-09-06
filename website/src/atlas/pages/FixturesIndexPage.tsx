import Link from "@docusaurus/Link";

import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps} from "../core";
import {criterionLabel, fixtureHref, levelLabel} from "../core";
import {ClaimLink} from "../Entity";
import {Glyph} from "../Glyph";

export interface FixturesIndexData {
  /** Rows of scores in each fixture, by fixture id. */
  sampleSize: Record<string, number>;
}

/**
 * `/research/counterexamples/`: every exact fixture in the record, with the
 * shape of its data and the statements it refutes or bounds.
 */
export default function FixturesIndexPage({core, data}: AtlasPageProps<FixturesIndexData>): React.JSX.Element {
  const order = core.vocabulary.criteria.map((entry) => entry.id);
  const rank = (criteria: readonly string[]): number => {
    const index = order.indexOf(criteria[0] ?? "");
    return index === -1 ? order.length : index;
  };
  const fixtures = Object.values(core.fixtures).sort((a, b) => rank(a.criterion) - rank(b.criterion) || a.id.localeCompare(b.id));

  return (
    <AtlasShell
      description="Every counterexample fixture in the research record: exact rational data, the statement it refutes and the hypotheses it sits on the edge of."
      title="Counterexamples"
    >
      <h1>Counterexamples</h1>
      <p className="atlas__lead">
        A fixture is a small table of exact rational scores and weights, together with the labels and objective values that follow from it,
        verified by enumerating every labelling or by an exact formula rather than by a run. Each one is cited by at least one statement: it
        either refutes that statement in the generality someone was tempted to give it, or sits on the edge of its hypotheses, showing that a
        condition cannot be dropped.
      </p>
      <p>
        A fixture marked <span className="provenance-band__mark">refusal</span> is one the library names out loud when it declines a call; the
        calls that trigger them are listed <Link to="/research/library/">in the library</Link>.
      </p>

      <div className="atlas-scroll">
        <table className="fixtures-index">
          <thead>
            <tr>
              <th scope="col">Fixture</th>
              <th scope="col">Criterion</th>
              <th scope="col">Level</th>
              <th scope="col">Rows</th>
              <th scope="col">d</th>
              <th scope="col">K</th>
              <th scope="col">What it settles</th>
            </tr>
          </thead>
          <tbody>
            {fixtures.map((fixture) => (
              <tr key={fixture.id}>
                <th scope="row">
                  <Link className="entity-link" to={fixtureHref(fixture.slug)}>
                    <Glyph kind="fixture" label="Counterexample fixture" />
                    <span className="entity-link__id">{fixture.id}</span>
                  </Link>
                  {fixture.refusal ? <span className="provenance-band__mark">refusal</span> : null}
                </th>
                <td>{fixture.criterion.map((criterion) => criterionLabel(core, criterion)).join(", ")}</td>
                <td>{levelLabel(core, fixture.level)}</td>
                <td className="fixtures-index__number">{data.sampleSize[fixture.id] ?? "—"}</td>
                <td className="fixtures-index__number">{fixture.dimension}</td>
                <td className="fixtures-index__number">{fixture.K}</td>
                <td>
                  <ul className="fixtures-index__cited">
                    {fixture.citedBy.map((citation) => (
                      <li key={`${citation.type}-${citation.claim}`}>
                        <span className="fixtures-index__verb">{citation.type === "refuted_by" ? "refutes" : "bounds"}</span>{" "}
                        <ClaimLink core={core} id={citation.claim} showId={false} />
                      </li>
                    ))}
                  </ul>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </AtlasShell>
  );
}
