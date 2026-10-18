import {describe, expect, it} from "vitest";

import {createRunRequest, LAB_LIMITS, PROTOCOL_VERSION, validateProblem} from "../src/lab/protocol";
import type {LabProblem, ScoreRow} from "../src/lab/protocol";

const validProblem: LabProblem = {
  scores: [[-1, 0], [0, 1], [1, 0]],
  weights: [1, 1, 1],
  nBins: 2,
  solver: "d_exchange",
  seed: 7,
  maxScans: 20
};

describe("lab protocol", () => {
  it("accepts a bounded finite problem", () => {
    expect(validateProblem(validProblem)).toBeNull();
    const request = createRunRequest(validProblem);
    expect(request).toMatchObject({protocolVersion: PROTOCOL_VERSION, type: "run"});
  });

  it("rejects mismatched weights and scalar-only misuse", () => {
    expect(validateProblem({...validProblem, weights: [1]})).toContain("same number");
    expect(validateProblem({...validProblem, solver: "scalar_dp"})).toContain("one score dimension");
  });

  it("enforces the browser capacity envelope", () => {
    const scores = Array.from({length: LAB_LIMITS.maxRows + 1}, () => [0]);
    expect(validateProblem({...validProblem, scores, weights: scores.map(() => 1)})).toContain("8,000");
    expect(validateProblem({...validProblem, nBins: LAB_LIMITS.maxBins + 1})).toContain("16");
  });

  it("admits the five-dimensional mixture score the FlowCyt study produces", () => {
    // The v1 ceiling of four columns excluded the library's own flagship
    // dataset from its own browser lab.
    const scores: ScoreRow[] = Array.from({length: 8}, () => [0.1, 0.2, 0.3, 0.4, 0.5]);
    // Five bins for five directions: a determinant objective needs at least that.
    expect(validateProblem({...validProblem, scores, nBins: 5, weights: scores.map(() => 1)})).toBeNull();
    const tooWide = scores.map(() => [1, 2, 3, 4, 5, 6, 7]);
    expect(validateProblem({...validProblem, scores: tooWide, nBins: 7, weights: scores.map(() => 1)})).toContain("one to 6");
  });
});

describe("criterion selection", () => {
  const schema: [string, string, string] = ["T", "B", "HSPC"];
  const profiled: LabProblem = {
    ...validProblem,
    scores: Array.from({length: 8}, () => [0.1, 0.2, 0.3]),
    weights: Array.from({length: 8}, () => 1),
    nBins: 3,
    solver: "soft_voronoi",
    schema,
    criterion: {name: "profiled_d_optimality", interest: ["HSPC"]}
  };

  it("accepts a named profiled objective on the solver that supports it", () => {
    expect(validateProblem(profiled)).toBeNull();
  });

  it("refuses profiling on a solver with no reusable profiled rule", () => {
    // The library refuses this too, but only after a 15 MB runtime has warmed.
    expect(validateProblem({...profiled, solver: "d_exchange"})).toContain("soft Voronoi");
  });

  it("requires a nuisance parameter to remain", () => {
    expect(validateProblem({...profiled, criterion: {name: "profiled_d_optimality", interest: [...schema]}}))
      .toContain("nuisance");
  });

  it("names an interest parameter the schema does not declare", () => {
    expect(validateProblem({...profiled, criterion: {name: "profiled_d_optimality", interest: ["HSPCs"]}}))
      .toContain("Unknown parameter: HSPCs");
  });

  it("requires a schema before it can profile by name", () => {
    const withoutSchema: LabProblem = {...profiled};
    delete withoutSchema.schema;
    expect(validateProblem(withoutSchema)).toContain("score schema");
  });

  it("pairs normalized trace with its own solver, in both directions", () => {
    expect(validateProblem({...profiled, criterion: {name: "normalized_trace"}})).toContain("k-means");
    // The reverse pairing is the one the defaults can fall into: choosing the
    // k-means solver while the criterion is still a determinant objective.
    expect(validateProblem({...profiled, solver: "kmeans", criterion: {name: "d_optimality"}}))
      .toContain("normalized trace");
  });

  it("refuses a determinant objective with fewer bins than informative directions", () => {
    // Measured against the library: on the five-dimensional FlowCyt scores,
    // every D-optimal solver refuses four bins and accepts five.
    const scores: ScoreRow[] = Array.from({length: 40}, () => [0.1, 0.2, 0.3, 0.4, 0.5]);
    const base: LabProblem = {...validProblem, scores, weights: scores.map(() => 1), solver: "d_exchange"};
    expect(validateProblem({...base, nBins: 4})).toContain("at least as many bins");
    expect(validateProblem({...base, nBins: 5})).toBeNull();
  });
});

describe("optimize_partition task", () => {
  const schema: [string, string, string] = ["T", "B", "HSPC"];
  const partition: LabProblem = {
    ...validProblem,
    scores: Array.from({length: 8}, () => [0.1, 0.2, 0.3]),
    weights: Array.from({length: 8}, () => 1),
    nBins: 3,
    solver: "d_exchange",
    schema,
    task: "optimize_partition"
  };

  it("keeps v2 behavior when task is absent", () => {
    expect(validProblem.task).toBeUndefined();
    expect(validateProblem(validProblem)).toBeNull();
  });

  it("refuses solvers other than the exact exchange and guarded Lloyd solvers", () => {
    expect(validateProblem({...partition, solver: "soft_voronoi"}))
      .toContain("optimize_partition runs the exact exchange and guarded Lloyd solvers only.");
    expect(validateProblem({...partition, solver: "kmeans"}))
      .toContain("optimize_partition runs the exact exchange and guarded Lloyd solvers only.");
  });

  it("accepts a profiled objective on the exact exchange solver", () => {
    expect(validateProblem({...partition, criterion: {name: "profiled_d_optimality", interest: ["HSPC"]}}))
      .toBeNull();
    expect(validateProblem({
      ...partition,
      solver: "mahalanobis_lloyd",
      criterion: {name: "profiled_d_optimality", interest: ["HSPC"]}
    })).toBeNull();
  });

  it("refuses normalized trace: it has no k-means path", () => {
    expect(validateProblem({...partition, criterion: {name: "normalized_trace"}}))
      .toContain("Normalized trace is a fit_quantizer objective; optimize_partition has no k-means path.");
  });

  it("refuses initialization without task optimize_partition and a profiled criterion", () => {
    expect(validateProblem({...validProblem, initialization: "efficient_score_bound"}))
      .toContain("initialization needs task optimize_partition and criterion profiled_d_optimality.");
    expect(validateProblem({...partition, initialization: "efficient_score_bound"}))
      .toContain("initialization needs task optimize_partition and criterion profiled_d_optimality.");
    expect(validateProblem({
      ...partition,
      initialization: "efficient_score_bound",
      criterion: {name: "profiled_d_optimality", interest: ["HSPC"]}
    })).toBeNull();
  });

  it("refuses report.profiledInterest covering every column", () => {
    expect(validateProblem({...partition, report: {profiledInterest: [...schema]}})).toContain("nuisance");
  });

  it("refuses an unknown report.profiledInterest name", () => {
    expect(validateProblem({...partition, report: {profiledInterest: ["HSPCs"]}}))
      .toContain("Unknown parameter: HSPCs");
  });
});
