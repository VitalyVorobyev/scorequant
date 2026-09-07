import {fireEvent, render, screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import {localLayout} from "../../src/atlas/LocalMap";
import AtlasHome from "../../src/atlas/pages/AtlasHome";
import {homeData} from "./renderData";
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
    priorArt: claim.priorArt.map((audit) => ({...audit, sources: audit.sources.map((s) => ({name: s.name, html: `<span>${s.text}</span>`}))})),
  };
}

describe("ClaimPage", () => {
  it("renders the reading grammar for the central theorem", () => {
    render(<ClaimPage core={core} data={claimData(D5)} />);
    expect(screen.getByRole("heading", {level: 1}).textContent).toBe(core.claims[D5]?.editorial?.title);
    const headings = screen.getAllByRole("heading", {level: 2}).map((h) => h.textContent);
    expect(headings).toEqual(expect.arrayContaining(["Meaning", "Statement", "Where it stops", "Proof"]));
    expect(screen.getByText("machine-checked statement")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Machine-checked statement", {selector: "summary"}));
    expect(screen.getByRole("link", {name: /Frozen specification/})).toHaveAttribute("href", expect.stringContaining("ExchangeVoronoiSpec.lean"));
    fireEvent.click(screen.getByText("Local graph", {selector: "summary"}));
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
  it("tells a bounded story without registry projections", () => {
    const {container} = render(<AtlasHome core={core} data={homeData()} />);
    expect(screen.getAllByRole("heading", {level: 2}).map((h) => h.textContent)).toEqual([
      atlas.home.central.title,
      "What we established",
      "Where it breaks",
      "Open frontier",
      "Explore further",
    ]);
    expect(container.querySelectorAll(".research-finding")).toHaveLength(9);
    expect(container.querySelectorAll("svg, .glyph, .atlas-legend, table, input")).toHaveLength(0);
    const article = container.querySelector("article.atlas__wide");
    expect(article?.textContent).not.toMatch(/D-EXCHANGE|DS-TILT|OPEN-/);
    expect(article?.textContent).toContain("merged duplicate atoms");
    // The trust line under each finding is registry status, not editorial copy,
    // so it sits outside the word budget.
    const trust = container.querySelector(".research-finding--central .research-trust");
    expect(trust?.textContent).toMatch(/^Proved here · machine-checked in Lean · independently audited$/);
    container.querySelectorAll(".research-trust").forEach((line) => line.remove());
    const words = article?.textContent.split(/\s+/).length ?? 0;
    expect(words).toBeGreaterThanOrEqual(450);
    expect(words).toBeLessThanOrEqual(550);
  });
});

describe("MapPage", () => {
  it("starts with a neighbourhood and draws the full graph only on demand", () => {
    render(<MapPage core={core} data={{}} />);
    const neighbourhood = screen.getByRole("group", {name: "Neighbourhood of the selected result"});
    expect(neighbourhood.querySelectorAll(".node").length).toBeLessThan(30);
    expect(screen.queryByRole("group", {name: "Argument map of the research"})).toBeNull();
    fireEvent.click(screen.getByRole("button", {name: "Show full graph"}));
    const figure = screen.getByRole("group", {name: "Argument map of the research"});
    expect(figure.querySelectorAll(".node")).toHaveLength(
      Object.values(core.claims).filter((c) => c.kind !== "audit").length + Object.keys(core.fixtures).length,
    );
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
