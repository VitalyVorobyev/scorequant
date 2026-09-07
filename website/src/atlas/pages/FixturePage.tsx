import type {Fixture} from "../../data/atlas";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps} from "../core";
import {EntityList, Html} from "../Entity";
import {FixtureFigure} from "../FixtureFigure";

/** A refusal whose Markdown fields were rendered at build time. */
export interface RenderedRefusal {
  reasonHtml: string | null;
  remedyHtml: string | null;
  triggerHtml: string | null;
}

export interface FixtureData extends Fixture {
  consequencesHtml: (string | null)[];
  falsifiesHtml: string | null;
  noteHtml: string | null;
  refusalHtml: RenderedRefusal | null;
  verificationNotesHtml: string | null;
}

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/** `2026-08-26` as `26 August 2026`, without asking the runtime for a locale. */
export function longDate(iso: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (!match) return iso;
  const [, year, month, day] = match;
  const name = MONTHS[Number(month) - 1];
  if (year === undefined || day === undefined || name === undefined) return iso;
  return `${String(Number(day))} ${name} ${year}`;
}

const EDGE_WORDS: Record<"bounded_by" | "refuted_by", string> = {
  bounded_by: "bounded by this fixture",
  refuted_by: "refuted by this fixture",
};

/**
 * One counterexample, in full.
 *
 * A fixture is the only kind of entity in the atlas whose exact numbers are
 * the content, so this is the one page that prints them: the atoms and their
 * weights, the two labellings, and the objective values that separate them.
 * Everything else on the page says what those numbers rule out.
 */
export default function FixturePage({core, data}: AtlasPageProps<FixtureData>): React.JSX.Element {
  const objectives = [
    {label: "Labels before", value: data.objectiveBefore},
    {label: "Labels after", value: data.objectiveAfter},
  ].filter((row) => row.value !== null);
  const citing = data.citedBy.map((entry) => entry.claim);
  const edgeOf = new Map(data.citedBy.map((entry) => [entry.claim, entry.type]));
  const consequences = data.consequencesHtml.filter((html): html is string => html !== null);

  return (
    <AtlasShell
      title={`Counterexample ${data.id}`}
      description={`The counterexample ${data.id}: its atoms, the statement it falsifies, and the claims that cite it.`}
    >
      <h1>
        Counterexample <code className="fixture-id">{data.id}</code>
      </h1>

      <section className="atlas__section">
        <h2>What it falsifies</h2>
        <Html className="statement statement--counter" html={data.falsifiesHtml} />
      </section>

      {data.noteHtml === null ? null : <Html className="atlas__lead" html={data.noteHtml} />}

      <FixtureFigure
        dimension={data.dimension}
        id={data.id}
        labelsAfter={data.labelsAfter}
        labelsBefore={data.labelsBefore}
        scores={data.scores}
        weights={data.weights}
      />

      {objectives.length === 0 ? null : (
        <section className="atlas__section">
          <h2>Objective</h2>
          <p>The criterion value at each labelling, exactly as the fixture certifies it.</p>
          <table>
            <thead>
              <tr>
                <th scope="col">Labelling</th>
                <th scope="col">Objective</th>
              </tr>
            </thead>
            <tbody>
              {objectives.map((row) => (
                <tr key={row.label}>
                  <th scope="row">{row.label}</th>
                  <td>
                    <code>{String(row.value)}</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {consequences.length === 0 ? null : (
        <section className="atlas__section">
          <h2>Consequences</h2>
          <ul>
            {consequences.map((html) => (
              <li key={html}>
                <Html html={html} />
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="atlas__section">
        <h2>How it was verified</h2>
        {data.verification.method === null ? null : (
          <p>
            Method: <code>{data.verification.method}</code>.
          </p>
        )}
        <Html html={data.verificationNotesHtml} />
      </section>

      <section className="atlas__section">
        <h2>Claims that cite it</h2>
        <EntityList
          core={core}
          ids={citing}
          meta={(id) => {
            const type = edgeOf.get(id);
            return type ? EDGE_WORDS[type] : null;
          }}
          empty="No claim cites this fixture yet."
        />
      </section>

      {data.refusalHtml === null || data.refusal === null ? null : (
        <section className="atlas__section">
          <h2>What the library refuses</h2>
          <dl className="fixture-refusal">
            <div>
              <dt>Trigger</dt>
              <dd>
                <Html html={data.refusalHtml.triggerHtml} />
              </dd>
            </div>
            <div>
              <dt>Reason</dt>
              <dd>
                <Html html={data.refusalHtml.reasonHtml} />
              </dd>
            </div>
            <div>
              <dt>Remedy</dt>
              <dd>
                <Html html={data.refusalHtml.remedyHtml} />
              </dd>
            </div>
          </dl>
        </section>
      )}

      <p className="fixture-source">
        {data.date === null ? null : `Recorded ${longDate(data.date)}. `}
        <a href={data.url} rel="noopener noreferrer">
          The fixture as JSON
        </a>
        .
      </p>
    </AtlasShell>
  );
}
