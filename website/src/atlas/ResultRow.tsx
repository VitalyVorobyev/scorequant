import Link from "@docusaurus/Link";
import type {Core, CoreClaim} from "./core";
import {claimHref, provenanceLabel} from "./core";

/** A readable index entry; exact statements stay on their permanent pages. */
export function ResultRow({core, claim}: {core: Core; claim: CoreClaim}): React.JSX.Element {
  return (
    <article className="research-result-row">
      <h3>
        <Link to={claimHref(claim.slug)}>{claim.editorial?.title ?? claim.title}</Link>
      </h3>
      {claim.editorial ? <p>{claim.editorial.summary}</p> : null}
      <p className="research-attribution">
        {[provenanceLabel(core, claim.provenance), claim.machineChecked ? "machine-checked" : null, claim.audited ? "audited" : null, claim.parked ? "parked" : null]
          .filter((mark) => mark !== null)
          .join(" · ")}
      </p>
    </article>
  );
}
