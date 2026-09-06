import {render, screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import {convexHull, coordinates, FixtureFigure, rational} from "../../src/atlas/FixtureFigure";
import {atlas} from "../atlasFixtures";

function figureOf(id: string): HTMLElement {
  const fixture = atlas.fixtures[id];
  if (!fixture) throw new Error(`no fixture ${id}`);
  const {container} = render(
    <FixtureFigure
      dimension={fixture.dimension}
      id={fixture.id}
      labelsAfter={fixture.labelsAfter}
      labelsBefore={fixture.labelsBefore}
      scores={fixture.scores}
      weights={fixture.weights}
    />
  );
  return container;
}

describe("rational", () => {
  it("reads the exact fractions the registry stores", () => {
    expect(rational("-47/64")).toBeCloseTo(-0.734375, 12);
    expect(rational("1/8")).toBe(0.125);
    expect(rational("0")).toBe(0);
    expect(rational(2)).toBe(2);
    expect(rational("3416482747129/14376887844864")).toBeCloseTo(0.2376372, 6);
  });

  it("reads a plain decimal and a negative integer string", () => {
    expect(rational("-3")).toBe(-3);
    expect(rational("2.5")).toBe(2.5);
    expect(rational(-8.1)).toBe(-8.1);
  });

  it("refuses a zero denominator rather than returning an infinity", () => {
    expect(rational("1/0")).toBeNaN();
  });

  it("keeps every stored weight and score of a real fixture finite", () => {
    for (const fixture of Object.values(atlas.fixtures)) {
      for (const weight of fixture.weights) expect(Number.isFinite(rational(weight)), fixture.id).toBe(true);
      for (const row of coordinates(fixture.scores)) {
        for (const value of row) expect(Number.isFinite(value), fixture.id).toBe(true);
      }
    }
  });
});

describe("convexHull", () => {
  it("returns the corners of a square and drops the interior point", () => {
    const hull = convexHull([
      {x: 0, y: 0},
      {x: 2, y: 0},
      {x: 2, y: 2},
      {x: 0, y: 2},
      {x: 1, y: 1}
    ]);
    expect(hull).toHaveLength(4);
    expect(hull).not.toContainEqual({x: 1, y: 1});
  });

  it("collapses duplicates and returns the points themselves below three", () => {
    expect(convexHull([{x: 1, y: 1}, {x: 1, y: 1}])).toEqual([{x: 1, y: 1}]);
    expect(convexHull([{x: 0, y: 0}, {x: 1, y: 1}])).toHaveLength(2);
  });

  it("returns the endpoints of a collinear set rather than a zero-area polygon", () => {
    const hull = convexHull([
      {x: 0, y: 0},
      {x: 1, y: 1},
      {x: 2, y: 2}
    ]);
    expect(hull.length).toBeGreaterThanOrEqual(2);
  });
});

describe("FixtureFigure", () => {
  it("draws the eight atoms of CE-DS-GLOBAL-GEOMETRY-001 in two panels", () => {
    const container = figureOf("CE-DS-GLOBAL-GEOMETRY-001");
    const panels = container.querySelectorAll(".fixture-panel");
    expect(panels).toHaveLength(2);
    for (const panel of Array.from(panels)) {
      expect(panel.querySelectorAll("circle.fixture-panel__point")).toHaveLength(8);
    }
    expect(screen.getByText("Labels before")).toBeInTheDocument();
    expect(screen.getByText("Labels after / optimum")).toBeInTheDocument();
  });

  it("colours each atom by its cell and draws one hull per cell", () => {
    const container = figureOf("CE-DS-GLOBAL-GEOMETRY-001");
    const first = container.querySelector(".fixture-panel");
    expect(first).not.toBeNull();
    // Labels are 0,0,1,0,0,1,2,2: four atoms in cell 0, two in cell 1, two in cell 2.
    expect(first?.querySelectorAll("circle.cell-1")).toHaveLength(4);
    expect(first?.querySelectorAll("circle.cell-2")).toHaveLength(2);
    expect(first?.querySelectorAll("circle.cell-3")).toHaveLength(2);
    // Cells of four, two and two atoms: one polygon and two segments.
    expect(first?.querySelectorAll("polygon.fixture-panel__hull")).toHaveLength(1);
    expect(first?.querySelectorAll("line.fixture-panel__hull--line")).toHaveLength(2);
  });

  it("gives every panel an image role, a title and a text alternative", () => {
    const container = figureOf("CE-DS-GLOBAL-GEOMETRY-001");
    const svgs = container.querySelectorAll("svg[role='img']");
    expect(svgs).toHaveLength(2);
    for (const svg of Array.from(svgs)) expect(svg.querySelector("title")?.textContent).toContain("CE-DS-GLOBAL-GEOMETRY-001");
    const table = container.querySelector("table.visually-hidden");
    expect(table).not.toBeNull();
    expect(table?.querySelectorAll("tbody tr")).toHaveLength(8);
    expect(table?.textContent).toContain("-65/8");
  });

  it("draws a one-dimensional fixture as a single strip with one panel when there is no optimum", () => {
    const container = figureOf("CE-O6-ETA-ZERO-MULTIATOM-VARIANCE-001");
    expect(container.querySelectorAll(".fixture-panel")).toHaveLength(1);
    expect(container.querySelectorAll("circle.fixture-panel__point")).toHaveLength(6);
    expect(container.querySelectorAll("line.fixture-panel__strip")).toHaveLength(1);
    const ys = Array.from(container.querySelectorAll("circle.fixture-panel__point")).map((node) => node.getAttribute("cy"));
    expect(new Set(ys).size).toBe(1);
    expect(screen.queryByText("score 2")).toBeNull();
  });

  it("sizes a mark by the square root of its weight", () => {
    const container = figureOf("CE-O6-ETA-ZERO-MULTIATOM-VARIANCE-001");
    const radii = Array.from(container.querySelectorAll("circle.fixture-panel__point")).map((node) =>
      Number(node.getAttribute("r"))
    );
    // Weights are 1/8 four times then 1/4 twice: area doubles, radius scales by sqrt(2).
    const light = radii[0];
    const heavy = radii[4];
    expect(light).toBeDefined();
    expect(heavy).toBeDefined();
    expect((heavy ?? 0) / (light ?? 1)).toBeCloseTo(Math.SQRT2, 6);
  });

  it("tabulates a fixture above two score dimensions instead of drawing it", () => {
    const container = figureOf("CE-DS-MARGINS-RANK-VACUITY-001");
    expect(container.querySelector("svg")).toBeNull();
    const table = container.querySelector("table.fixture-coords");
    expect(table).not.toBeNull();
    expect(table?.querySelectorAll("tbody tr")).toHaveLength(4);
    expect(screen.getByText(/3 score dimensions/)).toBeInTheDocument();
  });

  it("renders every real fixture without throwing", () => {
    for (const fixture of Object.values(atlas.fixtures)) {
      expect(() => figureOf(fixture.id), fixture.id).not.toThrow();
    }
  });
});
