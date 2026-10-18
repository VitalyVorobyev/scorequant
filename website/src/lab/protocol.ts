/**
 * The request and event contract between the page and the browser worker.
 *
 * Both ends ship in the same bundle, so these types are the whole contract;
 * the Python adapter (`static/runtime/python/scorequant_browser_lab.py`) reads
 * the `LabProblem` fields by the same names.
 */

/** One score row: one column per parameter, at most `LAB_LIMITS.maxDimensions`. */
export type ScoreRow = number[];

/** The objective to optimize. Profiled Dₛ requires parameters of interest. */
export interface LabCriterion {
  name: "d_optimality" | "profiled_d_optimality" | "normalized_trace";
  /** Parameters of interest by name, resolved against `schema`. */
  interest?: string[];
}

export interface LabProblem {
  scores: ScoreRow[];
  weights: number[];
  nBins: number;
  solver: "d_exchange" | "mahalanobis_lloyd" | "kmeans" | "scalar_dp" | "soft_voronoi";
  seed: number;
  maxSteps?: number;
  maxScans?: number;
  criterion?: LabCriterion;
  /** Parameter names in score-column order, so a profiled criterion can name them. */
  schema?: string[];
  /** Identifier of the score table the request was built from, for provenance. */
  datasetId?: string;
  /** Which public task to run. Absent means `fit_quantizer`. */
  task?: "fit_quantizer" | "optimize_partition";
  /** Seed the exchange from `efficient_score_bound(...).labels`; profiled `optimize_partition` only. */
  initialization?: "efficient_score_bound";
  /** Also report the profiled retention for these parameters, whatever criterion was solved. */
  report?: {profiledInterest?: string[]};
}

export interface LabResult {
  labels: number[];
  centers: number[][];
  retention: number;
  objective: number;
  execution: string;
  /** The objective the reported retention is measured against. */
  criterionLabel?: string;
  /** Parameters of interest, echoed back by name when the criterion profiled. */
  interest?: string[];
  /** Profiled retention for `report.profiledInterest`. */
  profiledRetention?: number;
  /** Whether the result is exchange-stable; `optimize_partition` only. */
  exchangeStable?: boolean;
}

export interface LabRunRequest {
  protocolVersion: typeof PROTOCOL_VERSION;
  type: "run";
  runId: string;
  problem: LabProblem;
}

export interface LabEvent {
  protocolVersion: typeof PROTOCOL_VERSION;
  runId: string;
  type: "ready" | "progress" | "result" | "error" | "cancelled";
  stage?: string;
  progress?: number;
  result?: LabResult;
  message?: string;
}

/** Wire version, bumped with any change to the types above. */
export const PROTOCOL_VERSION = 4;

export const LAB_LIMITS = {
  maxBins: 16,
  // Six columns: the FlowCyt mixture score is five-dimensional once the
  // reference component is absorbed, and the exchange gain is quadratic in this
  // dimension, so the ceiling buys headroom without inviting arbitrary data.
  maxDimensions: 6,
  maxRows: 8_000,
  maxScans: 500,
  maxSteps: 500
} as const;

export function validateProblem(problem: LabProblem): string | null {
  if (problem.scores.length === 0) return "Add at least one score row.";
  if (problem.scores.length > LAB_LIMITS.maxRows) return `Browser runs are limited to ${LAB_LIMITS.maxRows.toLocaleString()} rows.`;
  if (problem.weights.length !== problem.scores.length) return "Scores and weights must contain the same number of rows.";
  const dimensions = problem.scores[0]?.length ?? 0;
  if (dimensions < 1 || dimensions > LAB_LIMITS.maxDimensions) return `Browser runs support one to ${LAB_LIMITS.maxDimensions} score dimensions.`;
  if (problem.scores.some((row) => row.length !== dimensions || row.some((value) => !Number.isFinite(value)))) return "Every score row must have the same finite dimension.";
  if (problem.weights.some((weight) => !Number.isFinite(weight) || weight < 0) || !problem.weights.some((weight) => weight > 0)) return "Weights must be finite and nonnegative, with at least one positive value.";
  if (!Number.isInteger(problem.nBins) || problem.nBins < 1 || problem.nBins > LAB_LIMITS.maxBins) return `Choose between one and ${LAB_LIMITS.maxBins} bins.`;
  if (problem.nBins > problem.scores.length) return "The number of bins cannot exceed the number of score rows.";
  if (problem.solver === "scalar_dp" && dimensions !== 1) return "Exact scalar DP requires one score dimension.";
  if ((problem.maxSteps ?? 1) > LAB_LIMITS.maxSteps || (problem.maxScans ?? 1) > LAB_LIMITS.maxScans) return "The requested solver budget exceeds the browser limit.";
  if (problem.schema !== undefined && problem.schema.length !== dimensions) return `A score schema must name all ${String(dimensions)} columns.`;
  return validateCriterion(problem, dimensions);
}

/**
 * Check parameters of interest against the declared schema and score space.
 *
 * Shared by `criterion.interest` (which parameters the objective profiles) and
 * `report.profiledInterest` (which parameters a post-hoc report profiles):
 * both name score columns the same way and fail the same way.
 */
function validateInterestNames(names: readonly string[], schema: readonly string[] | undefined, dimensions: number): string | null {
  if (names.length === 0) return "Choose at least one parameter of interest.";
  if (names.length >= dimensions) return "At least one nuisance parameter must remain unprofiled.";
  if (schema === undefined) return "Profiling by name needs a score schema.";
  const unknown = names.filter((name) => !schema.includes(name));
  if (unknown.length > 0) return `Unknown parameter${unknown.length > 1 ? "s" : ""}: ${unknown.join(", ")}.`;
  return null;
}

/**
 * Check the objective against the score space and task it will be applied to.
 *
 * The library refuses these pairings too, but it refuses them after the run has
 * been shipped to a worker and a 15 MB runtime has warmed up; saying so here
 * turns a slow failure into an immediate one.
 */
function validateCriterion(problem: LabProblem, dimensions: number): string | null {
  const task = problem.task ?? "fit_quantizer";
  const criterion = problem.criterion;
  const name = criterion?.name ?? "d_optimality";

  if (task === "optimize_partition") {
    // optimize_partition never accepts kmeans, scalar_dp, or soft_voronoi: it
    // owns the exact exchange and the guarded Lloyd solver, and nothing else.
    if (problem.solver !== "d_exchange" && problem.solver !== "mahalanobis_lloyd") {
      return "optimize_partition runs the exact exchange and guarded Lloyd solvers only.";
    }
    if (name === "normalized_trace") {
      return "Normalized trace is a fit_quantizer objective; optimize_partition has no k-means path.";
    }
  } else {
    // k-means implements the normalized trace and nothing else, so pairing it
    // with a determinant objective is refused rather than quietly approximated.
    if (problem.solver === "kmeans" && name !== "normalized_trace") {
      return "The k-means solver fits the normalized trace. Choose that criterion, or another solver.";
    }
    if (name === "normalized_trace") {
      return problem.solver === "kmeans" ? null : "Normalized trace is fitted by the k-means solver.";
    }
  }

  // Every D-optimal path needs at least as many bins as informative score
  // directions: with fewer, the between-cell information matrix is singular and
  // its log-determinant is undefined. The effective rank is only known once the
  // fit runs, but it is at most the score dimension.
  if (problem.nBins < dimensions) {
    return `D-optimality needs at least as many bins as informative score directions. This table has ${String(dimensions)}; raise the bin budget to ${String(dimensions)} or more.`;
  }

  const profiled = criterion !== undefined && name === "profiled_d_optimality";
  if (problem.initialization !== undefined && (task !== "optimize_partition" || !profiled)) {
    return "initialization needs task optimize_partition and criterion profiled_d_optimality.";
  }

  if (profiled) {
    // Profiled Dₛ needs the soft Voronoi solver as a fit_quantizer objective: an
    // exchange-stable profiled partition has no canonical reusable rule. As an
    // optimize_partition objective it runs on the exact exchange or guarded
    // Lloyd solver instead, already enforced above.
    if (task === "fit_quantizer" && problem.solver !== "soft_voronoi") {
      return "Profiled Dₛ needs the soft Voronoi solver: an exchange-stable profiled partition has no canonical reusable rule.";
    }
    const interestError = validateInterestNames(criterion.interest ?? [], problem.schema, dimensions);
    if (interestError !== null) return interestError;
  }

  if (problem.report?.profiledInterest !== undefined) {
    const reportError = validateInterestNames(problem.report.profiledInterest, problem.schema, dimensions);
    if (reportError !== null) return reportError;
  }

  return null;
}

export function isLabEvent(value: unknown): value is LabEvent {
  if (typeof value !== "object" || value === null) return false;
  const event = value as Partial<LabEvent>;
  return event.protocolVersion === PROTOCOL_VERSION && typeof event.runId === "string" && ["ready", "progress", "result", "error", "cancelled"].includes(event.type ?? "");
}

export function createRunRequest(problem: LabProblem): LabRunRequest {
  return {protocolVersion: PROTOCOL_VERSION, type: "run", runId: crypto.randomUUID(), problem};
}
