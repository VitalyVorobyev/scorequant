/**
 * Types shared by `LiveFit` (always in the route chunk) and `LiveFitRunner`
 * (reached only through a dynamic import, see `LiveFitRunner.tsx`).
 *
 * The problem and result are the wire protocol's own types. A type-only import
 * is erased at build time, so it never pulls the browser runtime into the
 * route chunk.
 */
import type {LabProblem, LabResult} from "../../lab/protocol";

/** A bounded problem a `LiveFit` demo can hand to the browser runtime. */
export type LiveFitProblem = LabProblem;

/** One placement's result. */
export type LiveFitResult = LabResult;

/**
 * Resolve the problem to run, lazily.
 *
 * A synchronous placement (the committed fixture is already in memory)
 * returns a resolved promise; an asynchronous one (the score table is a
 * static asset fetched only on demand, e.g. `/get-started`) fetches it here,
 * on activation, rather than before the reader ever asks for it.
 */
export type LiveFitProblemLoader = () => Promise<LiveFitProblem>;

/** Props `LiveFit` hands to the dynamically imported `LiveFitRunner`. */
export interface LiveFitRunnerProps {
  /** Human label for what the committed number measures. */
  committedLabel: string;
  /** The number already published on the page. */
  committedRetention: number;
  /** How to render a retention value as text, matching the page's own convention. */
  formatRetention: (value: number) => string;
  /**
   * Which number to show as "your browser's run", read from the result.
   * Defaults to `result.retention`; a placement that asks for
   * `report.profiledInterest` passes this to show `profiledRetention`
   * instead, since that is the number the committed value beside it measures.
   */
  liveValue?: (result: LiveFitResult) => number;
  /** Human label for what the reader's own run measures. */
  liveLabel: string;
  /** Called once the reader is done: resets `LiveFit` back to its idle view. */
  onRelease: () => void;
  /** Resolve the problem to actually run, once activated. */
  problem: LiveFitProblemLoader;
  /** Render extra content below the two result cards, once a result exists. */
  renderResult?: (result: LiveFitResult) => React.ReactNode;
}
