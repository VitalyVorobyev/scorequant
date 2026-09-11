import {describe, expect, it} from "vitest";

import {adjacency, askQuestion, bands, boundary, cellsOf, closure, egoGraph, enables, openNextDoor, raises, restsOn, sameCell, verifiedBy} from "../../src/atlas/graph";
import {core} from "../atlasFixtures";

const adj = adjacency(core.edges);
const D5 = "D-EXCHANGE-IMPLIES-VORONOI";

describe("graph queries over the real atlas", () => {
  it("reads the registry's dependency and implication fields as typed edges", () => {
    expect(restsOn(adj, D5)).toEqual(["D-LOGDET-GAIN", "D-LEVERAGE"]);
    expect(enables(adj, D5)).toContain("D-FINITE-INDUCTIVE-CLOSURE");
    expect(raises(adj, D5)).toContain("OPEN-CRITERION-CHARACTERIZATION");
    expect(verifiedBy(adj, D5)).toEqual(["AUDIT-D-EXCHANGE-VORONOI"]);
  });

  it("keeps audit inputs outside the proof closure but reachable as evidence", () => {
    const audit = "AUDIT-DS-PRACTICAL-CERTIFIED-SOLVER";
    const claim = "OPEN-DS-PRACTICAL-CERTIFIED-SOLVER";
    expect(restsOn(adj, audit)).toEqual([]);
    expect(verifiedBy(adj, claim)).toContain(audit);
    expect(closure(adj, claim, "rests_on", "out").map((entry) => entry.id)).not.toContain(audit);
    expect(closure(adj, audit, "references", "out").map((entry) => entry.id)).toContain(claim);
    expect(egoGraph(core, adj, claim).nodes).toContain(audit);
  });

  it("names where a result stops", () => {
    const stop = boundary(adj, D5);
    expect(stop.converse).toEqual(["D-VORONOI-NOT-EXCHANGE"]);
    expect(stop.bounded).toEqual(["CE-D-UNMERGED-DUPLICATES-001"]);
    expect(stop.refuted).toEqual([]);
    expect(boundary(adj, "D-VORONOI-NOT-EXCHANGE").refuted).toEqual(["CE-D-VORONOI-CONVERSE-001"]);
  });

  it("walks the transitive chain nearest first and never revisits a node", () => {
    const chain = closure(adj, D5, "rests_on", "out");
    const ids = chain.map((entry) => entry.id);
    expect(ids.slice(0, 2)).toEqual(["D-LOGDET-GAIN", "D-LEVERAGE"]);
    expect(ids).toContain("FI-QUANT-IDENTITY");
    expect(new Set(ids).size).toBe(ids.length);
    expect(chain.every((entry, i) => i === 0 || entry.depth >= (chain[i - 1]?.depth ?? 0))).toBe(true);
  });

  it("builds an ego graph over claims and fixtures but not papers", () => {
    const ego = egoGraph(core, adj, D5, 1);
    expect(ego.nodes).toContain("CE-D-UNMERGED-DUPLICATES-001");
    expect(ego.nodes.every((id) => id in core.claims || id in core.fixtures)).toBe(true);
    expect(ego.edges.every((edge) => edge.type !== "cites")).toBe(true);
  });

  it("answers the five map questions with non-empty highlight sets", () => {
    for (const question of ["rests-on", "enables", "stops", "prior-work", "open"] as const) {
      const asked = askQuestion(core, adj, D5, question);
      expect(asked.nodes[0]).toBe(D5);
      expect(asked.edgeTypes.length).toBeGreaterThan(0);
    }
    expect(askQuestion(core, adj, D5, "stops").nodes).toContain("D-VORONOI-NOT-EXCHANGE");
  });

  it("places a claim in one landscape cell per criterion and finds its neighbours", () => {
    expect(cellsOf(core, D5)).toEqual([{criterion: "D", level: "finite_assignment"}]);
    expect(sameCell(core, D5)).toContain("D-LOGDET-GAIN");
    expect(sameCell(core, D5)).not.toContain(D5);
    expect(openNextDoor(core, adj, D5).every((id) => core.claims[id]?.kind === "question")).toBe(true);
  });

  it("sorts claims into the four reading bands and drops audits", () => {
    const grouped = bands(core, Object.keys(core.claims));
    expect(grouped.known).toContain("FI-QUANT-IDENTITY");
    expect(grouped.here).toContain(D5);
    expect(grouped.boundary).toContain("D-VORONOI-NOT-EXCHANGE");
    expect(grouped.open).toContain("OPEN-CRITERION-CHARACTERIZATION");
    const all = [...grouped.known, ...grouped.here, ...grouped.boundary, ...grouped.open];
    expect(all.some((id) => id.startsWith("AUDIT-"))).toBe(false);
  });
});
