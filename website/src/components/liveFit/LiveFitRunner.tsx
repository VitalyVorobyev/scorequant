import {useEffect, useRef, useState} from "react";

import {useLabRunner} from "../../lab/useLabRunner";
import type {LiveFitRunnerProps} from "./types";

/**
 * Talks to the browser runtime. Reached only through `LiveFit`'s
 * `await import("./LiveFitRunner")`, never imported at module scope by
 * anything in a route chunk -- that dynamic import is what keeps
 * `useLabRunner` (and, transitively, `runtimeClient.ts`'s `new Worker(...)`)
 * out of the ordinary page load.
 *
 * `useLabRunner` already owns the tab-wide runtime singleton, cold-start
 * bootstrapping, cancellation and unmount cleanup (`website/src/lab/useLabRunner.ts`);
 * this component's only job is to resolve the one problem it was asked to run,
 * start it once on mount, and render the result beside the committed one.
 */
export function LiveFitRunner({
  committedLabel,
  committedRetention,
  formatRetention,
  liveLabel,
  liveValue,
  onRelease,
  problem,
  renderResult
}: LiveFitRunnerProps): React.JSX.Element {
  const {cancel, error, progress, result, run, stage, state} = useLabRunner();
  const [loadError, setLoadError] = useState<string | null>(null);
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    problem()
      .then((resolved) => run(resolved))
      .catch((cause: unknown) => {
        setLoadError(cause instanceof Error ? cause.message : String(cause));
      });
    // Deliberately run-once: `problem` and `run` are stable for the lifetime
    // of one activation, and re-running on every render would restart the fit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const running = state === "loading" || state === "running";
  const failed = state === "error" || loadError !== null;

  return (
    <div className="live-fit live-fit--running">
      <div className="live-fit__status" aria-live="polite">
        <span className={`live-fit__state live-fit__state--${state}`}>{state}</span>
        <span className="live-fit__stage">{loadError !== null ? "Fetching the committed score table" : stage}</span>
        {running && <progress className="live-fit__progress" value={progress} max={1} />}
      </div>
      {failed && (
        <p className="live-fit__error" role="alert">
          {loadError ?? error}
        </p>
      )}
      {result !== null && (
        <>
          <div className="live-fit__results">
            <div className="live-fit__result">
              <span className="live-fit__badge">Committed result</span>
              <span className="live-fit__value">{formatRetention(committedRetention)}</span>
              <span className="live-fit__caption">{committedLabel}</span>
            </div>
            <div className="live-fit__result live-fit__result--live">
              <span className="live-fit__badge">Your browser&rsquo;s run</span>
              <span className="live-fit__value">{formatRetention(liveValue ? liveValue(result) : result.retention)}</span>
              <span className="live-fit__caption">{liveLabel}</span>
            </div>
          </div>
          {renderResult?.(result)}
        </>
      )}
      <div className="live-fit__actions">
        <button
          className="live-fit__button live-fit__button--secondary"
          type="button"
          onClick={() => {
            cancel();
            onRelease();
          }}
        >
          {state === "complete" || failed ? "Reset this demo" : "Cancel"}
        </button>
      </div>
    </div>
  );
}
