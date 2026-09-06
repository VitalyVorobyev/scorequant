import {render, screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import HowToReadPage from "../../src/atlas/pages/HowToReadPage";
import {atlas, core} from "../atlasFixtures";

const EMPTY: Record<string, never> = {};

describe("HowToReadPage", () => {
  it("names every provenance class beside its glyph, with the registry's own description", () => {
    const {container} = render(<HowToReadPage core={core} data={EMPTY} />);
    for (const entry of atlas.vocabulary.provenance) {
      expect(screen.getByText(entry.description), entry.id).toBeInTheDocument();
      expect(screen.getAllByTitle(entry.label).length, entry.id).toBeGreaterThan(0);
    }
    expect(container.querySelectorAll(".reading-key .glyph")).toHaveLength(atlas.vocabulary.provenance.length + 1);
  });

  it("says a machine-checked mark is on the statement and never on the implementation", () => {
    render(<HowToReadPage core={core} data={EMPTY} />);
    const paragraph = screen.getByText(/never about the implementation/);
    expect(paragraph.textContent).toContain("Lean 4");
    expect(paragraph.textContent).toContain("Mathlib");
    expect(screen.getByRole("link", {name: /machine-checked statements/})).toHaveAttribute("href", "/research/machine-checked/");
  });

  it("states the search-gap rule as a fact about a search rather than about priority", () => {
    render(<HowToReadPage core={core} data={EMPTY} />);
    const paragraph = screen.getByText(/records the outcome of a targeted search/);
    expect(paragraph.textContent).toContain("no direct precedent found");
    expect(paragraph.textContent).toContain("not about priority");
    expect(paragraph.textContent).toContain("not evidence that no equivalent exists");
  });

  it("describes every edge type, level and criterion", () => {
    render(<HowToReadPage core={core} data={EMPTY} />);
    for (const entry of [...atlas.vocabulary.edgeTypes, ...atlas.vocabulary.levels, ...atlas.vocabulary.criteria]) {
      expect(screen.getByText(entry.label), entry.id).toBeInTheDocument();
      expect(screen.getByText(entry.description), entry.id).toBeInTheDocument();
    }
  });

  it("says what the registry is and does not claim publication for being in it", () => {
    render(<HowToReadPage core={core} data={EMPTY} />);
    const paragraph = screen.getByText(/Being in the registry does not by itself publish a claim/);
    expect(paragraph.textContent).toContain("generated from the project");
  });

  it("closes with the totals, counted from the record itself", () => {
    const {container} = render(<HowToReadPage core={core} data={EMPTY} />);
    const claims = Object.values(atlas.claims);
    const statements = claims.filter((claim) => claim.kind !== "audit").length;
    const audits = claims.length - statements;
    const checked = claims.filter((claim) => claim.machineChecked !== null).length;
    const sentence = container.querySelector("p:last-of-type")?.textContent ?? "";
    expect(sentence).toContain(`${statements} statements`);
    expect(sentence).toContain(`${audits} verification records`);
    expect(sentence).toContain(`${Object.keys(atlas.fixtures).length} counterexample`);
    expect(sentence).toContain(`${Object.keys(atlas.papers).length} papers`);
    expect(sentence).toContain(`${checked} statements checked in Lean`);
  });
});
