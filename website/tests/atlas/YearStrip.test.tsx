import {render, screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import {lanes, YearStrip} from "../../src/atlas/YearStrip";
import {core} from "../atlasFixtures";

const dated = Object.values(core.papers).filter((paper) => paper.year !== null);

describe("YearStrip", () => {
  it("draws one mark per dated source", () => {
    const {container} = render(<YearStrip core={core} />);
    expect(container.querySelectorAll("circle.year-strip__mark")).toHaveLength(dated.length);
  });

  it("places every source in a lane, and gives a lane to every tradition that has one", () => {
    const rows = lanes(core);
    expect(rows.flatMap((lane) => lane.papers)).toHaveLength(dated.length);
    for (const tradition of core.traditions) {
      const hasDated = tradition.papers.some((key) => typeof core.papers[key]?.year === "number");
      expect(rows.some((lane) => lane.slug === tradition.slug), tradition.slug).toBe(hasDated);
    }
    expect(rows.at(-1)?.slug).toBe("other");
  });

  it("labels each lane and colours its marks by the tradition", () => {
    const {container} = render(<YearStrip core={core} />);
    for (const lane of lanes(core)) {
      expect(screen.getAllByText(lane.label).length).toBeGreaterThan(0);
      expect(container.querySelectorAll(`circle.tradition-${lane.slug}`).length).toBe(lane.papers.length);
    }
  });

  it("links every mark to its paper page and names the source in a title", () => {
    const {container} = render(<YearStrip core={core} />);
    const marks = Array.from(container.querySelectorAll("circle.year-strip__mark"));
    const alsing = marks.find((mark) => mark.querySelector("title")?.textContent.includes("Alsing"));
    expect(alsing?.querySelector("title")?.textContent).toBe(
      "Alsing, Wandelt (2019) Nuisance Hardened Data Compression for Fast Likelihood-Free Inference"
    );
    expect(alsing?.parentElement?.getAttribute("href")).toBe("/research/literature/alsing-wandelt-2019/");
  });

  it("carries a group role, a title and a text alternative naming every tradition", () => {
    const {container} = render(<YearStrip core={core} />);
    const svg = container.querySelector("svg[role='group']");
    expect(svg?.querySelector("title")?.textContent).toContain("Publication years");
    const alternative = container.querySelector(".year-strip__alt");
    expect(alternative).not.toBeNull();
    for (const lane of lanes(core)) expect(alternative?.textContent).toContain(lane.label);
  });

  it("repeats the same sources as lists for the narrow layout", () => {
    const {container} = render(<YearStrip core={core} />);
    const lists = container.querySelector(".year-strip__lists");
    expect(lists?.querySelectorAll("dt")).toHaveLength(lanes(core).length);
    expect(lists?.querySelectorAll("li")).toHaveLength(dated.length);
  });

  it("spans the full published range on one axis", () => {
    const years = dated.map((paper) => paper.year ?? 0);
    const {container} = render(<YearStrip core={core} />);
    const ticks = Array.from(container.querySelectorAll(".year-strip__tick")).map((node) => Number(node.textContent));
    expect(Math.min(...ticks)).toBeGreaterThanOrEqual(Math.min(...years));
    expect(Math.max(...ticks)).toBeLessThanOrEqual(Math.max(...years));
    expect(ticks.length).toBeGreaterThan(2);
  });
});
