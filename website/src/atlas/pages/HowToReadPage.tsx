import Link from "@docusaurus/Link";

import type {Provenance} from "../../data/atlas";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps} from "../core";
import {Glyph} from "../Glyph";

/**
 * `/research/how-to-read/`: the reading conventions in prose. Every glyph,
 * mark and edge type used anywhere in the atlas is named here once, with what
 * it does and does not assert.
 */
export default function HowToReadPage({core}: AtlasPageProps<Record<string, never>>): React.JSX.Element {
  const claims = Object.values(core.claims);
  const results = claims.filter((claim) => claim.kind !== "audit");
  const verifications = claims.length - results.length;
  const checked = claims.filter((claim) => claim.machineChecked).length;
  const fixtures = Object.keys(core.fixtures).length;
  const papers = Object.keys(core.papers).length;

  return (
    <AtlasShell
      description="The reading conventions of the research atlas: the provenance glyphs, the machine-checked mark, the edge types, the levels and the criteria."
      title="How to read this"
    >
      <h1>How to read this</h1>
      <p className="atlas__lead">
        Every entity in this record carries one provenance class, drawn as a glyph wherever it is named. The class says where the statement
        came from and how much authority it has, and shape carries it as well as colour, so the classes stay apart in print and for a reader
        who does not see the difference between two inks.
      </p>

      <h2>Where a statement came from</h2>
      <dl className="ways-in reading-key">
        {core.vocabulary.provenance.map((entry) => (
          <div key={entry.id}>
            <dt>
              <Glyph kind={entry.id as Provenance} label={entry.label} size="lg" /> {entry.label}
            </dt>
            <dd>{entry.description}</dd>
          </div>
        ))}
        <div>
          <dt>
            <Glyph kind="fixture" label="Counterexample fixture" size="lg" /> Counterexample fixture
          </dt>
          <dd>
            A small table of exact rational scores and weights, with the labels and the objective values it produces. It is the arithmetic
            behind a counterexample, checked by enumeration or by an exact formula rather than by a run.
          </dd>
        </div>
      </dl>

      <h2>The machine-checked mark</h2>
      <p>
        A glyph with a tick badge marks a statement written in Lean 4 against Mathlib and proved there. The mark is about the{" "}
        <em>statement</em> and never about the implementation: no proof assistant sees the Python or the JAX, and a passing Lean build says
        nothing about the code that computes with the result. The specification file is frozen and audited on its own before the proof is
        attempted, so what stands behind the mark is an independent reading of the Lean proposition against the registry statement. The whole
        track, module by module, is at <Link to="/research/machine-checked/">machine-checked statements</Link>.
      </p>

      <h2>What a search gap is</h2>
      <p>
        Some results are marked as proved here with no direct precedent found. That records the outcome of a targeted search of the
        literature, run against a written statement of the result: the search found no direct equivalent, and the nearest published work it did
        find is named on the result&apos;s own page. It is a statement about a search, not about priority, and it is not evidence that no
        equivalent exists. Where a search did find the result, the class says so instead and the source is cited.
      </p>

      <h2>How statements are connected</h2>
      <p>Every edge in the maps and every list on an entity page is one of these relations.</p>
      <dl className="ways-in reading-key">
        {core.vocabulary.edgeTypes.map((entry) => (
          <div key={entry.id}>
            <dt>{entry.label}</dt>
            <dd>{entry.description}</dd>
          </div>
        ))}
      </dl>

      <h2>The levels of the problem</h2>
      <p>
        A result is stated at one level. The level is what the result is about, and results at different levels do not substitute for one
        another: what is true of a labelling of one fixed table need not be true of a rule fitted on a sample, and neither settles the design
        under the population law.
      </p>
      <dl className="ways-in reading-key">
        {core.vocabulary.levels.map((entry) => (
          <div key={entry.id}>
            <dt>{entry.label}</dt>
            <dd>{entry.description}</dd>
          </div>
        ))}
      </dl>

      <h2>The criteria</h2>
      <p>A result is also stated for one or more criteria, which is what the binning is asked to maximise.</p>
      <dl className="ways-in reading-key">
        {core.vocabulary.criteria.map((entry) => (
          <div key={entry.id}>
            <dt>{entry.label}</dt>
            <dd>{entry.description}</dd>
          </div>
        ))}
      </dl>

      <h2>What these pages are</h2>
      <p>
        Every page in this section is generated from the project&apos;s research registry, the same file the library&apos;s own tests read, so
        a page and the code cannot drift apart without a failing build. The registry is a working record: an entry in it is a statement someone
        wrote down, together with what it rests on and how far it has been checked. Being in the registry does not by itself publish a claim,
        and it does not make a claim peer-reviewed; the provenance class on each entity is the whole of what is being asserted about it.
      </p>
      <p>
        As it stands the record holds {results.length} statements and {verifications} verification records, {fixtures} counterexample
        fixtures, {papers} papers of prior work, and {checked} statements checked in Lean.
      </p>
    </AtlasShell>
  );
}
