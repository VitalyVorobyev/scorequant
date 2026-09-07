import Link from "@docusaurus/Link";
import {useMemo, useState} from "react";

import type {Provenance} from "../../data/atlas";
import type {AtlasPageProps, Core, CoreClaim} from "../core";
import {claimHref, provenanceLabel} from "../core";
import {EntityList, Legend} from "../Entity";
import {Glyph} from "../Glyph";
import {cellsOf} from "../graph";

/**
 * The three levels the library's own object lives at: a labelling of one
 * fixed table, a rule fitted on a sample, and the design under the score law.
 * They are drawn on a lighter ground so the reader sees which columns the
 * project is actually about.
 */
const PRIMARY_LEVELS: ReadonlySet<string> = new Set(["finite_assignment", "empirical_inductive_quantizer", "population_quantizer"]);

/** Reading order inside a cell: settled first, boundary and open last. */
const PROVENANCE_ORDER: readonly Provenance[] = ["established", "derived", "proved", "proved_new", "measured", "counterexample", "open", "verification"];

type Colouring = "provenance" | "verification";
type Mark = "checked" | "audited" | "none";

const MARK_WORDS: Record<Mark, string> = {
  checked: "statement machine-checked in Lean",
  audited: "re-derived by an independent audit",
  none: "no independent check recorded",
};

function cellKey(criterion: string, level: string): string {
  return `${criterion} ${level}`;
}

function markOf(claim: CoreClaim): Mark {
  if (claim.machineChecked) return "checked";
  return claim.audited ? "audited" : "none";
}

/** Every non-audit claim placed at each (criterion, level) pair it carries. */
function buildCells(core: Core): Map<string, string[]> {
  const cells = new Map<string, string[]>();
  for (const claim of Object.values(core.claims)) {
    if (claim.kind === "audit") continue;
    for (const cell of cellsOf(core, claim.id)) {
      const key = cellKey(cell.criterion, cell.level);
      const held = cells.get(key);
      if (held) held.push(claim.id);
      else cells.set(key, [claim.id]);
    }
  }
  const rank = (id: string): number => {
    const claim = core.claims[id];
    return claim ? PROVENANCE_ORDER.indexOf(claim.provenance) : PROVENANCE_ORDER.length;
  };
  for (const held of cells.values()) held.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  return cells;
}

function GlyphMark({colouring, core, id}: {colouring: Colouring; core: Core; id: string}): React.JSX.Element | null {
  const claim = core.claims[id];
  if (!claim) return null;
  const meaning = colouring === "provenance" ? provenanceLabel(core, claim.provenance) : MARK_WORDS[markOf(claim)];
  const tone = colouring === "provenance" ? "provenance" : markOf(claim);
  return (
    <Link className={`landscape__glyph landscape__glyph--${tone}`} title={`${claim.title}: ${meaning}`} to={claimHref(claim.slug)}>
      <Glyph checked={claim.machineChecked} kind={claim.provenance} />
      <span className="visually-hidden">{`${claim.title}: ${meaning}`}</span>
    </Link>
  );
}

/**
 * `/research/landscape/`: the whole record as one matrix, the criterion a
 * result is stated for against the level of the problem it settles. Selecting
 * a cell lists what stands in it; the second control repaints the same grid by
 * what has been checked rather than by where the result came from.
 */
export default function LandscapeMatrix({core}: AtlasPageProps<Record<string, never>>): React.JSX.Element {
  const cells = useMemo(() => buildCells(core), [core]);
  const [selected, setSelected] = useState<{criterion: string; level: string} | null>(null);
  const [colouring, setColouring] = useState<Colouring>("provenance");

  const criteria = core.vocabulary.criteria;
  const levels = core.vocabulary.levels;
  const selectedCriterion = selected ? criteria.find((entry) => entry.id === selected.criterion) : undefined;
  const selectedLevel = selected ? levels.find((entry) => entry.id === selected.level) : undefined;
  const selectedIds = selected ? (cells.get(cellKey(selected.criterion, selected.level)) ?? []) : [];

  return (
    <div>
      <div className="atlas__measure">
        <h2>Criterion and problem level</h2>
        <p className="atlas__lead">
          A result is stated for a criterion, which is what the binning is asked to maximise, and at a level of the problem: a labelling of one fixed table of
          scores, a rule fitted on a sample and applied to observations not yet seen, or the design under the score law itself. The grid places every result at
          that pair.
        </p>
        <p>
          A result stated for several criteria appears in each of their rows, so a glyph is a placement rather than a result. Verification records are left out,
          since they carry no result of their own. An empty cell means no results are recorded here. It is not evidence that the field has not studied that
          question.
        </p>
      </div>

      <div className="landscape__controls">
        <fieldset className="landscape__colouring">
          <legend>Colour by</legend>
          <label>
            <input
              checked={colouring === "provenance"}
              name="landscape-colouring"
              onChange={() => setColouring("provenance")}
              type="radio"
              value="provenance"
            />{" "}
            provenance
          </label>
          <label>
            <input
              checked={colouring === "verification"}
              name="landscape-colouring"
              onChange={() => setColouring("verification")}
              type="radio"
              value="verification"
            />{" "}
            verification
          </label>
        </fieldset>
        {colouring === "verification" ? (
          <p aria-label="Verification marks" className="atlas-legend landscape__marks">
            <span>
              <span className="landscape__glyph landscape__glyph--checked">
                <Glyph checked kind="proved" />
              </span>{" "}
              statement machine-checked in Lean, never the implementation
            </span>
            <span>
              <span className="landscape__glyph landscape__glyph--audited">
                <Glyph kind="proved" />
              </span>{" "}
              re-derived by an independent audit
            </span>
            <span>
              <span className="landscape__glyph landscape__glyph--none">
                <Glyph kind="proved" />
              </span>{" "}
              neither
            </span>
          </p>
        ) : (
          <Legend core={core} />
        )}
      </div>

      <div
        aria-label="Results by criterion and level"
        className="landscape"
        role="table"
        style={{gridTemplateColumns: `minmax(112px, 172px) repeat(${levels.length}, minmax(0, 1fr))`}}
      >
        <div className="landscape__row" role="row">
          <span className="landscape__corner" role="columnheader">
            Criterion
          </span>
          {levels.map((level) => (
            <span
              key={level.id}
              className={`landscape__head${PRIMARY_LEVELS.has(level.id) ? " landscape__head--primary" : ""}`}
              role="columnheader"
              title={level.description}
            >
              {level.short}
            </span>
          ))}
        </div>
        {criteria.map((criterion) => (
          <div key={criterion.id} className="landscape__row" role="row">
            <span className="landscape__criterion" role="rowheader" title={criterion.description}>
              {criterion.label}
            </span>
            {levels.map((level) => {
              const ids = cells.get(cellKey(criterion.id, level.id)) ?? [];
              const isSelected = selected?.criterion === criterion.id && selected.level === level.id;
              return (
                <div
                  key={level.id}
                  className={`landscape__cell${PRIMARY_LEVELS.has(level.id) ? " landscape__cell--primary" : ""}${isSelected ? " is-selected" : ""}`}
                  role="cell"
                >
                  <button
                    aria-pressed={isSelected}
                    className="landscape__pick"
                    onClick={() => setSelected(isSelected ? null : {criterion: criterion.id, level: level.id})}
                    type="button"
                  >
                    <span className="visually-hidden">{`${criterion.label} at ${level.label}: ${ids.length}`}</span>
                  </button>
                  {ids.length === 0 ? (
                    <span aria-hidden="true" className="landscape__empty">
                      &ndash;
                    </span>
                  ) : (
                    <span className="landscape__glyphs">
                      {ids.map((id) => (
                        <GlyphMark key={id} colouring={colouring} core={core} id={id} />
                      ))}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      <div aria-live="polite" className="atlas__measure landscape__detail">
        {selectedCriterion && selectedLevel ? (
          <>
            <h2>
              {selectedCriterion.label}, {selectedLevel.label.toLowerCase()}
            </h2>
            <p>{selectedCriterion.description}</p>
            <p>{selectedLevel.description}</p>
            <EntityList core={core} empty="No results recorded. Nothing stands here yet." ids={selectedIds} />
          </>
        ) : (
          <p className="atlas__muted">Select a cell to read what stands in it, and what its criterion and its level mean.</p>
        )}
      </div>

      <div className="atlas__measure">
        <h2>The criteria</h2>
        <dl className="ways-in">
          {criteria.map((entry) => (
            <div key={entry.id}>
              <dt>{entry.label}</dt>
              <dd>{entry.description}</dd>
            </div>
          ))}
        </dl>

        <h2>The levels</h2>
        <dl className="ways-in">
          {levels.map((entry) => (
            <div key={entry.id}>
              <dt>{entry.label}</dt>
              <dd>{entry.description}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}
