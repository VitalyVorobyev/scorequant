/**
 * Pure queries over the atlas graph.
 *
 * Every view asks the same few questions of the same edge list — what a claim
 * rests on, what it enables, where it stops, what is open next to it — so the
 * answers live here as functions of the `core` document and are tested in
 * isolation (`tests/atlasGraph.test.ts`). Nothing in this module touches React
 * or the DOM.
 */
import type {Edge, EdgeType} from "../data/atlas";
import type {Core} from "./core";

export interface Adjacency {
  incoming: Map<string, Edge[]>;
  outgoing: Map<string, Edge[]>;
}

export function adjacency(edges: readonly Edge[]): Adjacency {
  const incoming = new Map<string, Edge[]>();
  const outgoing = new Map<string, Edge[]>();
  for (const edge of edges) {
    const out = outgoing.get(edge.source);
    if (out) out.push(edge);
    else outgoing.set(edge.source, [edge]);
    const inc = incoming.get(edge.target);
    if (inc) inc.push(edge);
    else incoming.set(edge.target, [edge]);
  }
  return {incoming, outgoing};
}

function targets(edges: Edge[] | undefined, types: readonly EdgeType[]): string[] {
  return (edges ?? []).filter((edge) => types.includes(edge.type)).map((edge) => edge.target);
}

function sources(edges: Edge[] | undefined, types: readonly EdgeType[]): string[] {
  return (edges ?? []).filter((edge) => types.includes(edge.type)).map((edge) => edge.source);
}

/** Direct prerequisites: the claims `id` rests on. */
export function restsOn(adj: Adjacency, id: string): string[] {
  return targets(adj.outgoing.get(id), ["rests_on"]);
}

/** Claims that rest on `id`, plus those it directly enables. */
export function enables(adj: Adjacency, id: string): string[] {
  const direct = targets(adj.outgoing.get(id), ["enables"]);
  const dependants = sources(adj.incoming.get(id), ["rests_on"]);
  return unique([...direct, ...dependants]);
}

/** Open questions `id` raises. */
export function raises(adj: Adjacency, id: string): string[] {
  return targets(adj.outgoing.get(id), ["raises"]);
}

/** Audits that re-derived `id`. */
export function verifiedBy(adj: Adjacency, id: string): string[] {
  return targets(adj.outgoing.get(id), ["verified_by"]);
}

/** Where `id` stops being true: converse failures and the fixtures on its edge. */
export function boundary(adj: Adjacency, id: string): {converse: string[]; refuted: string[]; bounded: string[]} {
  const out = adj.outgoing.get(id);
  return {
    converse: targets(out, ["converse_fails"]),
    refuted: targets(out, ["refuted_by"]),
    bounded: targets(out, ["bounded_by"])
  };
}

/** Papers `id` cites. */
export function cites(adj: Adjacency, id: string): string[] {
  return targets(adj.outgoing.get(id), ["cites"]);
}

/** The claims that raise, enable or rest on `id` (incoming side). */
export function raisedBy(adj: Adjacency, id: string): string[] {
  return sources(adj.incoming.get(id), ["raises"]);
}

/**
 * Transitive closure along one edge type, breadth first, nearest first.
 * `direction` "out" follows edges from the node; "in" follows them into it.
 */
export function closure(adj: Adjacency, id: string, type: EdgeType, direction: "out" | "in"): {id: string; depth: number}[] {
  const seen = new Set<string>([id]);
  const order: {id: string; depth: number}[] = [];
  let frontier = [id];
  let depth = 0;
  while (frontier.length > 0) {
    depth += 1;
    const next: string[] = [];
    for (const current of frontier) {
      const step = direction === "out" ? targets(adj.outgoing.get(current), [type]) : sources(adj.incoming.get(current), [type]);
      for (const other of step) {
        if (seen.has(other)) continue;
        seen.add(other);
        order.push({id: other, depth});
        next.push(other);
      }
    }
    frontier = next;
  }
  return order;
}

/** Every node within `radius` edges of `id`, ignoring direction and edge type, with the edges among them. */
export function egoGraph(core: Core, adj: Adjacency, id: string, radius = 1, types?: readonly EdgeType[]): {nodes: string[]; edges: Edge[]} {
  const keep = (edge: Edge): boolean => (types ? types.includes(edge.type) : edge.type !== "cites");
  const nodes = new Set<string>([id]);
  let frontier = [id];
  for (let step = 0; step < radius; step += 1) {
    const next: string[] = [];
    for (const current of frontier) {
      for (const edge of [...(adj.outgoing.get(current) ?? []), ...(adj.incoming.get(current) ?? [])]) {
        if (!keep(edge)) continue;
        const other = edge.source === current ? edge.target : edge.source;
        if (!(other in core.claims) && !(other in core.fixtures)) continue;
        if (!nodes.has(other)) {
          nodes.add(other);
          next.push(other);
        }
      }
    }
    frontier = next;
  }
  const edges = core.edges.filter((edge) => keep(edge) && nodes.has(edge.source) && nodes.has(edge.target));
  return {nodes: Array.from(nodes), edges};
}

/** The landscape cell a claim occupies: one entry per criterion it carries. */
export function cellsOf(core: Core, id: string): {criterion: string; level: string}[] {
  const claim = core.claims[id];
  if (!claim) return [];
  return claim.criterion.map((criterion) => ({criterion, level: claim.level}));
}

/** Claims in the same landscape cell as `id` (any shared criterion, same level), excluding audits. */
export function sameCell(core: Core, id: string): string[] {
  const claim = core.claims[id];
  if (!claim) return [];
  return Object.values(core.claims)
    .filter((other) => other.id !== id && other.kind !== "audit" && other.level === claim.level && other.criterion.some((c) => claim.criterion.includes(c)))
    .map((other) => other.id)
    .sort();
}

/** Open questions next door: raised by `id`, or open in the same cell. */
export function openNextDoor(core: Core, adj: Adjacency, id: string): string[] {
  const nearby = sameCell(core, id).filter((other) => core.claims[other]?.kind === "question");
  return unique([...raises(adj, id), ...nearby]);
}

export type Question = "rests-on" | "enables" | "stops" | "prior-work" | "open";

/** The highlight set a map "question" button produces for a focused node. */
export function askQuestion(core: Core, adj: Adjacency, id: string, question: Question): {nodes: string[]; edgeTypes: EdgeType[]} {
  switch (question) {
    case "rests-on":
      return {nodes: [id, ...closure(adj, id, "rests_on", "out").map((entry) => entry.id)], edgeTypes: ["rests_on"]};
    case "enables":
      return {
        nodes: unique([id, ...closure(adj, id, "rests_on", "in").map((entry) => entry.id), ...targets(adj.outgoing.get(id), ["enables"])]),
        edgeTypes: ["rests_on", "enables"]
      };
    case "stops": {
      const edge = boundary(adj, id);
      return {nodes: [id, ...edge.converse, ...edge.refuted, ...edge.bounded], edgeTypes: ["converse_fails", "refuted_by", "bounded_by"]};
    }
    case "prior-work":
      return {nodes: [id, ...cites(adj, id)], edgeTypes: ["cites"]};
    case "open":
      return {nodes: [id, ...openNextDoor(core, adj, id)], edgeTypes: ["raises"]};
  }
}

/** Claims grouped by the four bands of the reading grammar. */
export function bands(core: Core, ids: readonly string[]): {known: string[]; here: string[]; boundary: string[]; open: string[]} {
  const out = {known: [] as string[], here: [] as string[], boundary: [] as string[], open: [] as string[]};
  for (const id of ids) {
    const claim = core.claims[id];
    if (!claim || claim.kind === "audit") continue;
    switch (claim.provenance) {
      case "established":
      case "derived":
        out.known.push(id);
        break;
      case "proved":
      case "proved_new":
      case "measured":
        out.here.push(id);
        break;
      case "counterexample":
        out.boundary.push(id);
        break;
      case "open":
        out.open.push(id);
        break;
      default:
        break;
    }
  }
  return out;
}

export function unique<T>(values: readonly T[]): T[] {
  return Array.from(new Set(values));
}
