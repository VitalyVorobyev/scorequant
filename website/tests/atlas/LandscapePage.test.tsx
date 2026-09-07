import {fireEvent, render, screen, within} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import LandscapePage from "../../src/atlas/pages/LandscapeMatrix";
import {atlas, core} from "../atlasFixtures";

const EMPTY: Record<string, never> = {};

/** The button that selects one (criterion, level) cell. */
function cell(criterion: string, level: string): HTMLElement {
  const button = screen.getByRole("button", {name: new RegExp(`^${criterion} at ${level}:`)});
  const parent = button.parentElement;
  if (!parent) throw new Error(`cell ${criterion}/${level} has no container`);
  return parent;
}

describe("LandscapePage", () => {
  it("places every non-audit claim at each criterion and level it carries", () => {
    const {container} = render(<LandscapePage core={core} data={EMPTY} />);

    const placements = Object.values(atlas.claims)
      .filter((claim) => claim.kind !== "audit")
      .reduce((sum, claim) => sum + claim.criterion.length, 0);
    expect(container.querySelectorAll(".landscape__glyph")).toHaveLength(placements);

    // D-optimality at the finite level is the densest cell in the record.
    const finiteD = within(cell("D-optimality", "Finite assignment")).getAllByRole("link");
    expect(finiteD.length).toBeGreaterThan(1);
    expect(finiteD.some((link) => link.getAttribute("title")?.startsWith("Finite D exchange stability implies strict self-consistent Voronoi"))).toBe(true);
    expect(finiteD.every((link) => link.getAttribute("href")?.startsWith("/research/claims/"))).toBe(true);
  });

  it("reads an empty cell as empty territory rather than as a number", () => {
    render(<LandscapePage core={core} data={EMPTY} />);
    const empty = cell("Normalised trace", "Universal identities");
    expect(within(empty).queryAllByRole("link")).toHaveLength(0);
    expect(empty.querySelector(".landscape__glyphs")).toBeNull();
    const dash = empty.querySelector(".landscape__empty");
    expect(dash?.textContent).toBe("\u2013");
    expect(dash).toHaveAttribute("aria-hidden", "true");
  });

  it("emphasises the three primary levels and names every level and criterion", () => {
    const {container} = render(<LandscapePage core={core} data={EMPTY} />);
    expect(container.querySelectorAll(".landscape__head--primary")).toHaveLength(3);
    for (const level of atlas.vocabulary.levels) {
      expect(screen.getAllByText(level.description).length).toBeGreaterThan(0);
    }
    for (const criterion of atlas.vocabulary.criteria) {
      expect(screen.getAllByText(criterion.description).length).toBeGreaterThan(0);
    }
  });

  it("lists the claims of a cell, with both descriptions, once it is selected", () => {
    render(<LandscapePage core={core} data={EMPTY} />);
    expect(screen.getByText(/Select a cell/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", {name: /^E-optimality at Finite assignment:/}));
    const expected = Object.values(atlas.claims).filter(
      (claim) => claim.kind !== "audit" && claim.level === "finite_assignment" && claim.criterion.includes("E"),
    );
    const heading = screen.getByRole("heading", {name: /E-optimality, finite assignment/i});
    expect(heading).toBeInTheDocument();
    for (const claim of expected) expect(screen.getAllByText(claim.title).length).toBeGreaterThan(0);
    expect(screen.queryByText(/Select a cell/)).not.toBeInTheDocument();
  });

  it("repaints the same grid by verification when asked, and only then", () => {
    const {container} = render(<LandscapePage core={core} data={EMPTY} />);
    expect(container.querySelectorAll(".landscape__cell .landscape__glyph--checked")).toHaveLength(0);
    expect(container.querySelectorAll(".landscape__glyph--provenance").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("radio", {name: /verification/}));
    const checked = Object.values(atlas.claims)
      .filter((claim) => claim.kind !== "audit" && claim.machineChecked !== null)
      .reduce((sum, claim) => sum + claim.criterion.length, 0);
    expect(container.querySelectorAll(".landscape__cell .landscape__glyph--checked")).toHaveLength(checked);
    expect(container.querySelectorAll(".landscape__glyph--provenance")).toHaveLength(0);
    expect(container.querySelectorAll(".landscape__cell .landscape__glyph--audited").length).toBeGreaterThan(0);
    expect(screen.getByText(/never the implementation/)).toBeInTheDocument();
  });
});
