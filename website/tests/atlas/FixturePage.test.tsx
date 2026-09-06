import {render, screen, within} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import FixturePage, {longDate} from "../../src/atlas/pages/FixturePage";
import {atlas, core, fixtureData} from "./renderData";

function page(id: string): HTMLElement {
  const {container} = render(<FixturePage core={core} data={fixtureData(id)} />);
  return container;
}

function section(container: HTMLElement, heading: string): HTMLElement | undefined {
  return Array.from(container.querySelectorAll("section.atlas__section")).find(
    (node) => node.querySelector("h2")?.textContent === heading
  ) as HTMLElement | undefined;
}

describe("longDate", () => {
  it("spells the month out without asking the runtime for a locale", () => {
    expect(longDate("2026-08-26")).toBe("26 August 2026");
    expect(longDate("2026-01-01")).toBe("1 January 2026");
  });

  it("returns anything it cannot parse unchanged", () => {
    expect(longDate("August 2026")).toBe("August 2026");
  });
});

describe("FixturePage", () => {
  it("titles the page with the fixture id in mono under a plain heading", () => {
    page("CE-DS-GLOBAL-GEOMETRY-001");
    const title = screen.getByRole("heading", {level: 1});
    expect(title.textContent).toBe("Counterexample CE-DS-GLOBAL-GEOMETRY-001");
    expect(title.querySelector("code")?.textContent).toBe("CE-DS-GLOBAL-GEOMETRY-001");
  });

  it("puts the note before the figure", () => {
    const container = page("CE-DS-GLOBAL-GEOMETRY-001");
    const text = container.textContent;
    expect(text).toContain("an exact eight-row, two-score-dimension, three-cell example");
    expect(text.indexOf("This is the fixture behind")).toBeLessThan(text.indexOf("Labels before"));
  });

  it("draws the atoms and states what the fixture falsifies", () => {
    const container = page("CE-DS-GLOBAL-GEOMETRY-001");
    expect(container.querySelectorAll(".fixture-panel")).toHaveLength(2);
    const statement = container.querySelector(".statement--counter");
    expect(statement?.textContent).toContain("self-consistent under its own first-order");
  });

  it("prints the exact objective values in a table, and omits the table when there are none", () => {
    const withValues = page("CE-DS-TILT-DUAL-GAP-001");
    const objective = section(withValues, "Objective");
    const rows = objective?.querySelectorAll("tbody tr");
    expect(rows).toHaveLength(2);
    expect(objective?.textContent).toContain("2925/296");
    expect(objective?.textContent).toContain("116805/11816");

    const without = page("CE-DS-GLOBAL-GEOMETRY-001");
    expect(section(without, "Objective")).toBeUndefined();
  });

  it("names the verification method and its notes", () => {
    const container = page("CE-DS-GLOBAL-GEOMETRY-001");
    const how = section(container, "How it was verified");
    expect(how?.querySelector("code")?.textContent).toBe("exact rational exhaustive enumeration");
    expect(how?.textContent).toContain("Canonical fixture selected from the newest");
  });

  it("says in words how each citing claim is bound to the fixture", () => {
    const container = page("CE-DS-GLOBAL-GEOMETRY-001");
    const citing = section(container, "Claims that cite it");
    expect(citing).toBeDefined();
    const items = citing ? Array.from(citing.querySelectorAll(".entity-list > li")) : [];
    expect(items).toHaveLength(atlas.fixtures["CE-DS-GLOBAL-GEOMETRY-001"]?.citedBy.length ?? 0);
    expect(citing?.textContent).toContain("refuted by this fixture");
    expect(citing?.textContent).toContain("bounded by this fixture");
  });

  it("lists the consequences when the fixture records any", () => {
    const container = page("CE-DS-POP-WASTED-CELLS-001");
    const consequences = section(container, "Consequences");
    expect(consequences?.querySelectorAll("li")).toHaveLength(
      atlas.fixtures["CE-DS-POP-WASTED-CELLS-001"]?.consequences.length ?? 0
    );
    expect(section(page("CE-DS-GLOBAL-GEOMETRY-002"), "Consequences")).toBeUndefined();
  });

  it("gives the library refusal its trigger, reason and remedy, with the symbols in mono", () => {
    const container = page("CE-DS-GLOBAL-GEOMETRY-001");
    const refusal = section(container, "What the library refuses");
    if (!refusal) throw new Error("the fixture records a refusal, so the page must show it");
    const terms = Array.from(refusal.querySelectorAll("dt")).map((node) => node.textContent);
    expect(terms).toEqual(["Trigger", "Reason", "Remedy"]);
    expect(within(refusal).getByText("compile_quantizer()").tagName).toBe("CODE");
    expect(refusal.textContent).toContain("no canonical extension");
  });

  it("omits the refusal block for a fixture the library does not refuse on", () => {
    expect(section(page("CE-DS-TILT-DUAL-GAP-001"), "What the library refuses")).toBeUndefined();
  });

  it("dates the record and links the JSON on GitHub", () => {
    const container = page("CE-DS-GLOBAL-GEOMETRY-001");
    const source = container.querySelector(".fixture-source");
    expect(source?.textContent).toContain("Recorded 26 August 2026");
    const link = source?.querySelector("a");
    expect(link).toHaveAttribute("href", atlas.fixtures["CE-DS-GLOBAL-GEOMETRY-001"]?.url ?? "");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("renders every fixture in the registry without throwing", () => {
    for (const fixture of Object.values(atlas.fixtures)) {
      expect(() => page(fixture.id), fixture.id).not.toThrow();
    }
  });
});
