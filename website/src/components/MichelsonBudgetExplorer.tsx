import {useState} from "react";

import {ApertureStrip, runsFromLabels} from "./ApertureStrip";
import {BinningComparison} from "./BinningComparison";
import {ScoreSpaceRuns} from "./ScoreSpaceRuns";
import {LiveFit} from "./liveFit/LiveFit";
import type {LiveFitProblem, LiveFitResult} from "./liveFit/types";
import {michelsonSweep} from "../data/michelsonSweep";
import {loadWalkthroughScoreTable} from "../data/walkthroughScores";

/**
 * The seed `examples/michelson_phase.py` used to produce the committed sweep
 * (see `walkthroughs/michelson.mdx`'s own `DExchangeConfig(seed=4)`). Shared
 * between the problem payload and the activation prose below so the number a
 * reader sees named is always the number the browser run actually uses,
 * never a second typed copy of it.
 */
const SWEEP_SEED = 4;

/**
 * The Michelson walkthrough's Explore step: a reader-chosen counter budget
 * `K`, drawn from the committed sweep `generate_walkthroughs.py` wrote to
 * `michelsonSweep`, plus a `LiveFit` that refits the profiled criterion at
 * that same budget in the reader's own browser.
 *
 * Keying `<LiveFit>` by `nBins` is what keeps a budget change from silently
 * reusing a stale run: changing the radio remounts the whole `LiveFit`
 * subtree, and `LiveFit`'s own unmount effect releases the site-wide
 * activation slot, so a fresh instance starts idle at the new budget's
 * committed result rather than carrying over an in-flight or completed fit
 * for the old one.
 */
export function MichelsonBudgetExplorer(): React.JSX.Element {
  const [selectedBins, setSelectedBins] = useState(michelsonSweep.headlineBins);
  const row = michelsonSweep.rows.find((candidate) => candidate.nBins === selectedBins);
  if (row === undefined) {
    throw new Error(`No generated Michelson sweep row exists for ${String(selectedBins)} bins.`);
  }
  const atHeadline = selectedBins === michelsonSweep.headlineBins;

  const problem = async (): Promise<LiveFitProblem> => {
    const table = await loadWalkthroughScoreTable("michelson");
    return {
      criterion: {interest: ["phase"], name: "profiled_d_optimality"},
      datasetId: "walkthroughs-michelson",
      initialization: "efficient_score_bound",
      nBins: row.nBins,
      schema: table.schema,
      scores: table.scores,
      seed: SWEEP_SEED,
      solver: "d_exchange",
      task: "optimize_partition",
      weights: table.weights
    };
  };

  const renderResult = (result: LiveFitResult): React.ReactNode => {
    const runs = runsFromLabels(result.labels, michelsonSweep.uMax);
    return (
      <>
        <ApertureStrip
          bands={[{label: "Your browser's fit", runs}]}
          description={`Your browser's own profiled fit on the same node table the study used, a comb of ${String(runs.length)} intervals at ${String(row.nBins)} counters.`}
          fringes={michelsonSweep.fringes}
          title={`Your browser's readout at ${String(row.nBins)} counters`}
          uMax={michelsonSweep.uMax}
        />
        <ScoreSpaceRuns
          bands={[{label: "Your browser's fit", runs}]}
          description={`Your browser's own profiled fit drawn in score space at ${String(row.nBins)} counters, each counter a colour along the score curve.`}
          title={`Your browser's fit in score space at ${String(row.nBins)} counters`}
          uMax={michelsonSweep.uMax}
          visibility={michelsonSweep.visibility}
        />
      </>
    );
  };

  return (
    <div className="budget-explorer">
      <fieldset className="budget-explorer__controls" role="radiogroup" aria-label="Counters K">
        <legend>Counters K</legend>
        {michelsonSweep.rows.map((candidate) => (
          <label
            key={candidate.nBins}
            className={
              selectedBins === candidate.nBins ? "budget-explorer__radio is-active" : "budget-explorer__radio"
            }
          >
            <input
              checked={selectedBins === candidate.nBins}
              name="michelson-bins"
              onChange={() => setSelectedBins(candidate.nBins)}
              type="radio"
            />
            {String(candidate.nBins)}
          </label>
        ))}
      </fieldset>
      <button
        className="budget-explorer__reset"
        disabled={atHeadline}
        onClick={() => setSelectedBins(michelsonSweep.headlineBins)}
        type="button"
      >
        Reset to the headline budget
      </button>

      <ApertureStrip
        bands={[
          {label: "Profiled Ds", runs: row.runs.profiled},
          {label: "Plain D", runs: row.runs.dOptimal},
          {label: "Equal segments", runs: row.runs.equalWidth}
        ]}
        description={`The profiled rule is a comb of ${String(row.runs.profiled.length)} intervals over the aperture and plain D a comb of ${String(row.runs.dOptimal.length)}; equal segments are ${String(row.nBins)} contiguous intervals on the same aperture.`}
        fringes={michelsonSweep.fringes}
        title={`Aperture readout at ${String(row.nBins)} counters`}
        uMax={michelsonSweep.uMax}
      />

      <ScoreSpaceRuns
        bands={[
          {label: "Profiled Ds", runs: row.runs.profiled},
          {label: "Plain D", runs: row.runs.dOptimal}
        ]}
        description={`The same two labellings drawn in the space they were optimized in: the score curve, four loops of growing fringe-frequency score, with each counter a colour. Profiled Ds cuts across the loops to separate the phase score; plain D spends cells resolving the loops from one another.`}
        title={`Score space at ${String(row.nBins)} counters, profiled Ds against plain D`}
        uMax={michelsonSweep.uMax}
        visibility={michelsonSweep.visibility}
      />

      <BinningComparison
        axisLabel="Phase information retained, after profiling"
        caption={`All three partitions use ${String(row.nBins)} counters on the same node table; the dashed line is the certified ceiling, not a partition you could choose. Higher is better.`}
        rows={[
          {label: "Equal segments", text: row.text.equalWidth, value: row.equalWidth},
          {label: "Plain D", text: row.text.dOptimal, value: row.dOptimal},
          {isScoreQuant: true, label: "Profiled Ds", text: row.text.profiled, value: row.profiled},
          {isCeiling: true, label: "Certified ceiling", text: row.text.ceiling, value: row.ceiling}
        ]}
      />

      <LiveFit
        key={row.nBins}
        id="walkthroughs-michelson"
        activationHint={
          "Fetches the same node table the study used and runs optimize_partition with the " +
          `profiled criterion for phase, the exact exchange solver, seed ${String(SWEEP_SEED)}, and ` +
          "the efficient-score-bound start -- exactly how the committed sweep row was produced. " +
          "The value should agree to the digits shown; summation order can differ in the last places."
        }
        activationLabel="Refit this budget in your browser"
        committedLabel={`Committed profiled retention at ${String(row.nBins)} counters`}
        committedRetention={row.profiled}
        formatRetention={(value) => value.toFixed(4)}
        liveLabel={`Your browser's profiled retention at ${String(row.nBins)} counters`}
        problem={problem}
        renderResult={renderResult}
      />
    </div>
  );
}
