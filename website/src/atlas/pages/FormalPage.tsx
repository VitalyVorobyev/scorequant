import type {Claim, FormalModule} from "../../data/atlas";
import {AtlasShell} from "../AtlasShell";
import type {AtlasPageProps} from "../core";
import {ClaimLink, EntityList} from "../Entity";
import {LeanChain} from "../LeanChain";

type MachineChecked = NonNullable<Claim["machineChecked"]>;

export interface FormalData {
  axioms: string[];
  /** Every claim carrying a machine-checked mark, with its Lean record. */
  certified: {id: string; machineChecked: MachineChecked}[];
  chain: FormalModule[];
  outside: string[];
  readme: string;
}

function FileLink({label, path, url}: {label: string; path: string; url: string}): React.JSX.Element {
  return (
    <a href={url} rel="noopener" title={path}>
      {label}
    </a>
  );
}

/**
 * `/research/machine-checked/`: the Lean track. What the mark means, the
 * module chain in build order, the certified statements with their
 * declarations and audits, and what the track deliberately leaves out.
 */
export default function FormalPage({core, data}: AtlasPageProps<FormalData>): React.JSX.Element {
  const order = new Map(data.chain.flatMap((module) => module.claims).map((id, index) => [id, index]));
  const certified = [...data.certified].sort((a, b) => (order.get(a.id) ?? data.certified.length) - (order.get(b.id) ?? data.certified.length));
  const system = certified[0]?.machineChecked.system ?? null;

  return (
    <AtlasShell
      description="What a machine-checked mark means here, the Lean module chain, and the statements it certifies."
      title="Machine-checked statements"
    >
      <h1>Machine-checked statements</h1>
      <p className="atlas__lead">
        A machine-checked mark says that the <em>statement</em> of a result has been written in Lean 4 against Mathlib and proved there. It
        says nothing about the Python implementation: a Lean build certifies a proposition, never the code that computes with it.
      </p>
      <p>
        The specification file is frozen before the proof is attempted and audited on its own, so the question the audit answers is whether the
        Lean proposition is the registry statement, and only then does the proof have to go through. The workspace, its build instructions and
        its conventions are in the{" "}
        <a href={data.readme} rel="noopener">
          formal workspace README
        </a>
        .
      </p>

      <h2>The chain</h2>
      <p>
        Each module is proved on the modules before it, so the last statement in the chain rests on every definition that precedes it. A dashed
        box is a module that is only partly done, and the claims it touches carry no mark.
      </p>
      <figure className="atlas-figure lean-chain">
        <LeanChain modules={data.chain} />
        <figcaption className="atlas-figure__caption">The Lean modules in build order.</figcaption>
      </figure>
      <dl className="lean-chain__modules">
        {data.chain.map((module) => (
          <div key={module.module}>
            <dt>
              <code>{module.module}</code>
              {module.partial === true ? <span className="provenance-band__mark">in part</span> : null}
            </dt>
            <dd>
              <p>{module.result}.</p>
              {module.claims.length > 0 ? (
                <EntityList compact core={core} ids={module.claims} />
              ) : (
                <p className="atlas__muted">Supporting definitions; it certifies no statement of its own.</p>
              )}
            </dd>
          </div>
        ))}
      </dl>

      <h2>What is certified</h2>
      <table className="formal-table">
        <thead>
          <tr>
            <th scope="col">Statement</th>
            <th scope="col">Lean declaration</th>
            <th scope="col">Files</th>
            <th scope="col">Statement audit</th>
          </tr>
        </thead>
        <tbody>
          {certified.map((entry) => (
            <tr key={entry.id}>
              <td>
                <ClaimLink core={core} id={entry.id} />
              </td>
              <td>
                <code>{entry.machineChecked.declaration}</code>
              </td>
              <td className="formal-table__files">
                <FileLink label="spec" path={entry.machineChecked.spec.path} url={entry.machineChecked.spec.url} />{" "}
                <FileLink label="proof" path={entry.machineChecked.proof.path} url={entry.machineChecked.proof.url} />
              </td>
              <td>
                {entry.machineChecked.statementAudit ? (
                  <a href={entry.machineChecked.statementAudit.url} rel="noopener" title={entry.machineChecked.statementAudit.file}>
                    {entry.machineChecked.statementAudit.verdict ?? "audited, no verdict line"}
                  </a>
                ) : (
                  <span className="atlas__muted">none recorded</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {system === null ? null : <p>Every declaration above is checked by {system}.</p>}

      <h2>Outside the track</h2>
      <p>What the Lean build does not reach, and therefore does not certify:</p>
      <ul>
        {data.outside.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>

      <h2>The axioms it is allowed</h2>
      <p>
        Every certified declaration is checked to depend on no axiom beyond{" "}
        {data.axioms.map((axiom, index) => (
          <span key={axiom}>
            {index === 0 ? "" : index === data.axioms.length - 1 ? " and " : ", "}
            <code>{axiom}</code>
          </span>
        ))}
        , the ordinary classical foundation of Mathlib. Nothing in the chain is admitted by assumption.
      </p>
    </AtlasShell>
  );
}
