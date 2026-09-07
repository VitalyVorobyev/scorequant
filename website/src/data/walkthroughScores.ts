import {siteUrl} from "../lib/site";

/**
 * One walkthrough's deterministic score table, in exactly the shape a
 * `LiveFit` problem (`website/src/components/liveFit/types.ts`) needs.
 *
 * Written by `generate_walkthroughs.py`'s `write_walkthrough_score_tables`
 * to `website/static/walkthrough-scores/<slug>.json`; never edited by hand.
 */
export interface WalkthroughScoreTable {
  detail: string;
  label: string;
  schema: string[];
  scores: number[][];
  weights: number[];
}

/**
 * Fetch one walkthrough's committed score table on demand.
 *
 * Kept out of the bundle: a `LiveFit` problem resolver is the only caller,
 * so a walkthrough that never activates its experiment fetches nothing
 * extra.
 */
export async function loadWalkthroughScoreTable(
  slug: string,
  signal?: AbortSignal
): Promise<WalkthroughScoreTable> {
  const url = siteUrl(`walkthrough-scores/${slug}.json`);
  const response = await fetch(url, signal === undefined ? {} : {signal});
  if (!response.ok) {
    throw new Error(`The "${slug}" walkthrough score table is unavailable (${String(response.status)}).`);
  }
  return (await response.json()) as WalkthroughScoreTable;
}

/**
 * The `/get-started` first-fit cell's committed score table, plus exactly
 * the run configuration the cell used: `n_bins=5`,
 * `DExchangeConfig(seed=21)`.
 *
 * Written by `website/scripts/generate_snippets.py`'s
 * `build_first_fit_score_table`, read straight out of the executed
 * `get_started_program.py` namespace rather than retyped; never edited by
 * hand.
 */
export interface FirstFitScoreTable extends WalkthroughScoreTable {
  nBins: number;
  seed: number;
  solver: "d_exchange";
}

/**
 * Fetch the `/get-started` first-fit score table on demand.
 *
 * Fetched only once the reader activates `GetStartedFirstFitLiveFit` -- the
 * committed retention it shows beforehand comes from the much smaller,
 * already-bundled `website/src/lib/snippets.ts` (`firstFitRetention`)
 * instead, so opening `/get-started` with no interaction fetches nothing
 * extra.
 */
export async function loadGetStartedScoreTable(signal?: AbortSignal): Promise<FirstFitScoreTable> {
  const url = siteUrl("walkthrough-scores/get-started.json");
  const response = await fetch(url, signal === undefined ? {} : {signal});
  if (!response.ok) {
    throw new Error(`The get-started score table is unavailable (${String(response.status)}).`);
  }
  return (await response.json()) as FirstFitScoreTable;
}
