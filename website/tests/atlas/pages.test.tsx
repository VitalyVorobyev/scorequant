import {render, screen, within} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import {BandStrip} from "../../src/atlas/BandStrip";
import {localLayout} from "../../src/atlas/LocalMap";
import AtlasHome from "../../src/atlas/pages/AtlasHome";
import type {HomeCard} from "../../src/atlas/pages/AtlasHome";
import ClaimPage from "../../src/atlas/pages/ClaimPage";
import type {ClaimData} from "../../src/atlas/pages/ClaimPage";
import MapPage from "../../src/atlas/pages/MapPage";
import {atlas, core} from "../atlasFixtures";

const D5 = "D-EXCHANGE-IMPLIES-VORONOI";

function claimData(id: string): ClaimData {
  const claim = atlas.claims[id];
  if (!claim) throw new Error(`no claim ${id}`);
  return {
    ...claim,
    statementHtml: `<p>${claim.statement}</p>`,
    assumptionsHtml: claim.assumptions.map((a) => `<span>${a}</span>`),
    noteHtml: claim.note === null ? null : `<p>${claim.note}</p>`,
    scopeHtml: claim.scope === null ? null : `<p>${claim.scope}</p>`,
    roleHtml: claim.role === null ? null : `<p>${claim.role}</p>`,
    proofHtml: claim.proof === null ? null : `<p>${claim.proof.markdown.slice(0, 80)}</p>`,
    priorArt: claim.priorArt.map((audit) => ({...audit, sources: audit.sources.map((s) => ({name: s.name, html: `<span>${s.text}</span>`}))}))
  };
}

function card(id: string): HomeCard {
  const claim = core.claims[id];
  const full = atlas.claims[id];
  if (!claim || !full) throw new Error(`no claim ${id}`);
  return {...claim, noteHtml: full.note === null ? null : `<p>${full.note}</p>`, scopeHtml: null};
}

describe("ClaimPage", () => {
  it("renders the reading grammar for the central theorem", () => {
    render(<ClaimPage core={core} data={claimData(D5)} />);
    expect(screen.getByRole("heading", {level: 1}).textContent).toBe(core.claims[D5]?.title);
    const headings = screen.getAllByRole("heading", {level: 2}).map((h) => h.textContent);
    expect(headings).toEqual(expect.arrayContaining(["Meaning", "Statement", "Rests on", "Enables", "Where it stops", "Machine-checked", "In the library", "Local map", "Proof"]));
    expect(screen.getByText("machine-checked statement")).toBeInTheDocument();
    expect(screen.getByRole("link", {name: /Frozen specification/})).toHaveAttribute("href", expect.stringContaining("ExchangeVoronoiSpec.lean"));
    expect(screen.getByRole("img", {name: /Local map of/})).toBeInTheDocument();
    expect(screen.getByRole("link", {name: "Open in the map"})).toHaveAttribute("href", `/research/map/?focus=${D5}`);
  });

  it("reorders the sections for an open question", () => {
    render(<ClaimPage core={core} data={claimData("OPEN-DS-MARGINS-NONCENTERED")} />);
    const headings = screen.getAllByRole("heading", {level: 2}).map((h) => h.textContent);
    expect(headings).toContain("Already excluded");
    expect(headings).toContain("The question in full");
    expect(headings).not.toContain("Proof");
    expect(screen.getByText("Open question · OPEN-DS-MARGINS-NONCENTERED")).toBeInTheDocument();
  });

  it("never prints internal work-tracking vocabulary", () => {
    for (const id of ["OPEN-DS-MARGINS-NONCENTERED", "DS-STABLE-BASINS-CENTERED-OBSTRUCTION", D5]) {
      const {container, unmount} = render(<ClaimPage core={core} data={claimData(id)} />);
      expect(container.textContent).not.toMatch(/\bP[1-8]\b|\bOP\d+\b|packet|WORK\//);
      unmount();
    }
  });
});

describe("AtlasHome", () => {
  it("opens with the problem, then the headline results, the strip, the frontier and the ways in", () => {
    render(<AtlasHome core={core} data={{headline: core.headline.map(card), frontier: core.frontier.map(card)}} />);
    const headings = screen.getAllByRole("heading", {level: 2}).map((h) => h.textContent);
    expect(headings).toEqual(["The problem", "Established here", "The shape of the field", "Where the frontier is now", "Ways in"]);
    expect(document.querySelectorAll(".result-card")).toHaveLength(core.headline.length);
    const strip = screen.getByRole("table", {name: /Known, new here/});
    expect(within(strip).getAllByRole("row").length).toBeGreaterThan(5);
    expect(within(strip).getAllByRole("columnheader").map((h) => h.textContent)).toEqual(["Theme", "Known", "New here", "Boundary", "Open"]);
    expect(document.querySelector(".katex-display")).not.toBeNull();
    expect(screen.queryByText(/who this is for/i)).toBeNull();
  });
});

describe("BandStrip", () => {
  it("links every glyph to a claim page and leaves empty cells marked", () => {
    render(<BandStrip core={core} />);
    const links = screen.getAllByRole("link");
    expect(links.length).toBeGreaterThan(100);
    expect(links.every((link) => link.getAttribute("href")?.startsWith("/research/claims/"))).toBe(true);
    expect(screen.getAllByLabelText("none").length).toBeGreaterThan(0);
  });
});

describe("MapPage", () => {
  it("draws every non-audit claim and fixture as a node and exposes the mobile list", () => {
    render(<MapPage core={core} data={{}} />);
    const figure = screen.getByRole("group", {name: "Argument map of the research"});
    const nodes = figure.querySelectorAll(".node");
    const expected = Object.values(core.claims).filter((c) => c.kind !== "audit").length + Object.keys(core.fixtures).length;
    expect(nodes).toHaveLength(expected);
    expect(figure.querySelectorAll(".edge--rests_on").length).toBeGreaterThan(100);
    expect(screen.getByRole("heading", {name: "Every result by problem level"})).toBeInTheDocument();
    expect(screen.getByRole("group", {name: "Zoom"})).toBeInTheDocument();
  });
});

describe("localLayout", () => {
  it("places prerequisites left, consequences right and boundary cases below the focus", () => {
    const layout = localLayout(core, D5);
    const focus = layout.positions[D5];
    if (!focus) throw new Error("focus not placed");
    for (const id of layout.columns.upstream) expect(layout.positions[id]?.x).toBeLessThan(focus.x);
    for (const id of layout.columns.downstream) expect(layout.positions[id]?.x).toBeGreaterThan(focus.x);
    for (const id of layout.columns.below) expect(layout.positions[id]?.y).toBeGreaterThan(focus.y);
    expect(layout.edges.every((edge) => edge.source === D5 || edge.target === D5)).toBe(true);
  });
});
