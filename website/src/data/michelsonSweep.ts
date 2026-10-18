import rawData from "../generated/michelson-sweep.json";

/**
 * One maximal constant-label run of the aperture, in fringe-phase `u`.
 *
 * `start` and `end` are the outer edges of the run's first and last node, so
 * consecutive runs of one labeling share an edge and together tile
 * `[0, uMax]` exactly; `label` is the bin index every node in the run shares.
 */
export interface ApertureRun {
  start: number;
  end: number;
  label: number;
}

/**
 * One swept bin budget's three labelings: their profiled phase-information
 * retention, the certified ceiling and bound gap at that budget, and each
 * labeling's aperture runs.
 *
 * `text` carries every numeric field already formatted by the same renderers
 * the walkthrough's fact table uses -- a page never formats a value itself,
 * so a static fallback table renders `text.*` verbatim.
 */
export interface MichelsonSweepRow {
  nBins: number;
  equalWidth: number;
  dOptimal: number;
  profiled: number;
  ceiling: number;
  boundGap: number;
  text: {
    equalWidth: string;
    dOptimal: string;
    profiled: string;
    ceiling: string;
    boundGap: string;
  };
  runs: {
    equalWidth: ApertureRun[];
    dOptimal: ApertureRun[];
    profiled: ApertureRun[];
  };
}

/**
 * The whole committed Michelson bin-budget sweep, aperture runs included.
 *
 * Written by `generate_walkthroughs.py`'s `write_michelson_sweep` to
 * `website/src/generated/michelson-sweep.json`, straight from
 * `examples/outputs/michelson-phase.json`; never edited by hand.
 */
export interface MichelsonSweep {
  schemaVersion: number;
  uMax: number;
  fringes: number;
  headlineBins: number;
  visibility: number;
  rows: MichelsonSweepRow[];
}

/** The raw JSON's aperture run, before conversion to `ApertureRun`. */
type RawRunTriple = [number, number, number];

interface RawMichelsonSweepRow {
  nBins: number;
  equalWidth: number;
  dOptimal: number;
  profiled: number;
  ceiling: number;
  boundGap: number;
  text: {
    equalWidth: string;
    dOptimal: string;
    profiled: string;
    ceiling: string;
    boundGap: string;
  };
  runs: {
    equalWidth: RawRunTriple[];
    dOptimal: RawRunTriple[];
    profiled: RawRunTriple[];
  };
}

interface RawMichelsonSweep {
  schemaVersion: number;
  uMax: number;
  fringes: number;
  headlineBins: number;
  visibility: number;
  rows: RawMichelsonSweepRow[];
}

/** Convert one labeling's `[start, end, label]` triples into `ApertureRun`s. */
function toRuns(triples: RawRunTriple[]): ApertureRun[] {
  return triples.map(([start, end, label]) => ({start, end, label}));
}

const raw = rawData as RawMichelsonSweep;

/**
 * The committed Michelson sweep, with every labeling's runs already converted
 * to `ApertureRun` objects -- done once here, at module load, rather than by
 * every consumer.
 */
export const michelsonSweep: MichelsonSweep = {
  schemaVersion: raw.schemaVersion,
  uMax: raw.uMax,
  fringes: raw.fringes,
  headlineBins: raw.headlineBins,
  visibility: raw.visibility,
  rows: raw.rows.map((row) => ({
    nBins: row.nBins,
    equalWidth: row.equalWidth,
    dOptimal: row.dOptimal,
    profiled: row.profiled,
    ceiling: row.ceiling,
    boundGap: row.boundGap,
    text: row.text,
    runs: {
      equalWidth: toRuns(row.runs.equalWidth),
      dOptimal: toRuns(row.runs.dOptimal),
      profiled: toRuns(row.runs.profiled),
    },
  })),
};
