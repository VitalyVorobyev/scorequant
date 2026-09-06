import Link from "@docusaurus/Link";

import type {Core} from "./core";
import {claimHref, provenanceLabel} from "./core";
import {Glyph} from "./Glyph";
import {bands} from "./graph";

const COLUMNS: {key: keyof ReturnType<typeof bands>; label: string}[] = [
  {key: "known", label: "Known"},
  {key: "here", label: "New here"},
  {key: "boundary", label: "Boundary"},
  {key: "open", label: "Open"}
];

/**
 * The atlas in miniature: one row per theme of the research, four columns
 * for the reading grammar — what was established before, what this project
 * proved or measured, where it fails, what remains open — and one glyph per
 * claim, each a link. Reading a row left to right is reading the state of
 * that theme.
 */
export function BandStrip({core}: {core: Core}): React.JSX.Element {
  return (
    <div className="band-strip" role="table" aria-label="Known, new here, boundary and open results by theme">
      <div className="band-strip__row" role="row">
        <div className="band-strip__head" role="columnheader">
          Theme
        </div>
        {COLUMNS.map((column) => (
          <div key={column.key} className="band-strip__head" role="columnheader">
            {column.label}
          </div>
        ))}
      </div>
      {core.strips
        .filter((strip) => strip.claims.length > 0)
        .map((strip) => {
          const grouped = bands(core, strip.claims);
          return (
            <div key={strip.id} className="band-strip__row" role="row">
              <div className="band-strip__label" role="rowheader">
                {strip.label}
              </div>
              {COLUMNS.map((column) => (
                <div key={column.key} className="band-strip__cell" role="cell">
                  {grouped[column.key].length === 0 ? (
                    <span className="band-strip__empty" aria-label="none">
                      –
                    </span>
                  ) : (
                    grouped[column.key].map((id) => {
                      const claim = core.claims[id];
                      if (!claim) return null;
                      return (
                        <Link key={id} to={claimHref(claim.slug)} title={`${claim.title} (${provenanceLabel(core, claim.provenance)})`} aria-label={claim.title}>
                          <Glyph kind={claim.provenance} checked={claim.machineChecked} />
                        </Link>
                      );
                    })
                  )}
                </div>
              ))}
            </div>
          );
        })}
    </div>
  );
}
