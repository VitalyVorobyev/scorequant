import Link from "@docusaurus/Link";

import {ReferenceLink} from "../../components/ReferenceLink";
import type {LibraryObject, LibraryRefusal} from "../../data/atlas";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps, Core} from "../core";
import {fixtureHref} from "../core";
import {ClaimLink, EntityList, Html} from "../Entity";

export interface LibraryData {
  objects: (LibraryObject & {roleHtml: string | null})[];
  refusals: (LibraryRefusal & {reasonHtml: string | null; remedyHtml: string | null; triggerHtml: string | null})[];
}

/** The groups, in the order a reader meets them when fitting something. */
const GROUPS: readonly {id: string; label: string}[] = [
  {id: "criteria", label: "Criteria"},
  {id: "configurations", label: "Solver configurations"},
  {id: "results", label: "Results and certificates"},
  {id: "information", label: "Information reports"},
  {id: "providers", label: "Sources and score providers"}
];

function FixtureCode({code, core}: {code: string; core: Core}): React.JSX.Element {
  const fixture = core.fixtures[code];
  if (!fixture) return <code>{code}</code>;
  return (
    <Link to={fixtureHref(fixture.slug)}>
      <code>{code}</code>
    </Link>
  );
}

/**
 * `/research/library/`: where the research record touches the public API.
 * The refusals first, because they are the place a reader meets a research
 * result without asking for one, then every public object beside the results
 * it rests on.
 */
export default function LibraryPage({core, data}: AtlasPageProps<LibraryData>): React.JSX.Element {
  return (
    <AtlasShell
      description="The refusals the library raises, and every public object beside the results it rests on."
      title="In the library"
    >
      <h1>In the library</h1>
      <p className="atlas__lead">
        The library raises two different kinds of error. A contract error means the call was malformed: mismatched shapes, a negative weight, a
        criterion paired with a configuration that cannot serve it. A refusal is not that. The arguments are well formed and the computation
        would run; the library declines because what the call asks it to assert is false in general, and it names the exact counterexample that
        makes it false.
      </p>
      <p>
        Each refusal below therefore comes with a fixture: a small table of exact rational scores on which the asserted property demonstrably
        fails. A reader who doubts a refusal can read the fixture instead of taking the refusal on trust.
      </p>

      <h2>The refusals</h2>
      <table className="library-refusals">
        <thead>
          <tr>
            <th scope="col">Refusal</th>
            <th scope="col">Raised on</th>
            <th scope="col">Rests on</th>
            <th scope="col">Instead</th>
          </tr>
        </thead>
        <tbody>
          {data.refusals.map((refusal) => (
            <tr key={refusal.code}>
              <td>
                <FixtureCode code={refusal.code} core={core} />
                <Html className="library-refusals__reason" html={refusal.reasonHtml} />
              </td>
              <td>
                <Html html={refusal.triggerHtml} />
              </td>
              <td>
                <ul className="library-refusals__claims">
                  {refusal.claims.map((id) => (
                    <li key={id}>
                      <ClaimLink core={core} id={id} showId={false} />
                    </li>
                  ))}
                </ul>
              </td>
              <td>
                <Html html={refusal.remedyHtml} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>What each object rests on</h2>
      <p>
        Every public object below is listed with the results that justify it. Where the list is empty the object is bookkeeping: it reports or
        carries what another object established, and adds no claim of its own.
      </p>
      {GROUPS.map((group) => {
        const objects = data.objects.filter((object) => object.group === group.id);
        if (objects.length === 0) return null;
        return (
          <section key={group.id} aria-labelledby={`group-${group.id}`} className="atlas__section">
            <h3 id={`group-${group.id}`}>{group.label}</h3>
            <dl className="library-relation">
              {objects.map((object) => (
                <div key={`${object.group}-${object.name}`}>
                  <dt>
                    <ReferenceLink to="symbols/">
                      <code>{object.name}</code>
                    </ReferenceLink>
                  </dt>
                  <dd className="library-relation__role">
                    <Html html={object.roleHtml} />
                  </dd>
                  <dd className="library-relation__claims">
                    {object.claims.length > 0 ? (
                      <EntityList compact core={core} ids={object.claims} />
                    ) : (
                      <p className="atlas__muted">No result of its own.</p>
                    )}
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        );
      })}
    </AtlasShell>
  );
}
