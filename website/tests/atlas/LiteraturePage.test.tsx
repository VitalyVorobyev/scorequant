import {render, screen, within} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import LiteraturePage from "../../src/atlas/pages/LiteraturePage";
import {atlas, core, literatureData} from "./renderData";

const data = literatureData();

function page(): HTMLElement {
  const {container} = render(<LiteraturePage core={core} data={data} />);
  return container;
}

describe("LiteraturePage", () => {
  it("opens on the traditions rather than on the reader", () => {
    page();
    const lead = screen.getByText(/The problem sits where several traditions meet/);
    expect(lead.textContent).toContain("cited below rather than claimed");
    expect(lead.textContent).not.toMatch(/\b(you|we)\b/i);
  });

  it("heads one section per tradition, in the core's order, each with its anchor", () => {
    const container = page();
    const sections = Array.from(container.querySelectorAll("section.literature-tradition"));
    const slugs = sections.map((section) => section.id);
    expect(slugs).toEqual([...core.traditions.map((tradition) => `tradition-${tradition.slug}`), "tradition-other"]);
    for (const tradition of core.traditions) {
      const section = container.querySelector(`#tradition-${tradition.slug}`);
      expect(within(section as HTMLElement).getByRole("heading", {level: 2}).textContent).toBe(tradition.label);
    }
  });

  it("lists every source of a tradition with its year and a link to its page", () => {
    const container = page();
    const section = container.querySelector("#tradition-score-compression");
    expect(section).not.toBeNull();
    const entries = section?.querySelectorAll(".literature-entry");
    expect(entries).toHaveLength(atlas.traditions.find((t) => t.slug === "score-compression")?.papers.length ?? 0);
    const link = within(section as HTMLElement).getByRole("link", {name: /Alsing, Wandelt/});
    expect(link).toHaveAttribute("href", "/research/literature/alsing-wandelt-2019/");
  });

  it("puts the annotation fields in reading order and drops the redundant paper title", () => {
    const container = page();
    const entry = Array.from(container.querySelectorAll(".literature-entry")).find((node) =>
      node.textContent.includes("Nuisance Hardened Data Compression")
    );
    const terms = entry ? Array.from(entry.querySelectorAll(".annotation-fields dt")).map((node) => node.textContent) : [];
    expect(terms[0]).toBe("Result");
    expect(terms).toContain("Does not transfer");
    expect(terms.indexOf("Does not transfer")).toBeGreaterThan(terms.indexOf("Result"));
    expect(terms).not.toContain("Paper");
  });

  it("names the claims each source is cited by", () => {
    const container = page();
    const entry = Array.from(container.querySelectorAll(".literature-entry")).find((node) =>
      node.textContent.includes("Nuisance Hardened Data Compression")
    );
    expect(entry?.textContent).toContain("Cited by");
    for (const id of atlas.papers["Alsing-Wandelt-2019"]?.citedBy ?? []) {
      expect(entry?.textContent).toContain(id);
    }
  });

  it("renders a tradition's own notes as prose blocks", () => {
    const container = page();
    const section = container.querySelector("#tradition-software");
    const notes = section?.querySelectorAll(".literature-note");
    expect(notes).toHaveLength(atlas.traditions.find((t) => t.slug === "software")?.notes.length ?? 0);
    expect(section?.textContent).toContain("Current software gap");
  });

  it("groups the sources that belong to no tradition", () => {
    const container = page();
    const section = container.querySelector("#tradition-other");
    const expected = Object.values(atlas.papers).filter((paper) => paper.tradition === null);
    expect(section?.querySelectorAll(".literature-entry")).toHaveLength(expected.length);
    expect(section?.textContent).toContain("Principal Curves");
  });

  it("lists the nearest prior work beside each headline result that has an audit", () => {
    const container = page();
    const aside = container.querySelector(".atlas__aside");
    expect(aside?.textContent).toContain("Closest prior work for the headline results");
    const items = aside?.querySelectorAll(".prior-art > li");
    expect(items).toHaveLength(data.priorArt.length);
    expect(items?.length).toBeGreaterThan(0);
    expect(aside?.textContent).toContain("Kiefer\u2013Wolfowitz");
    // The whole bullet survives, wrapped lines included, not just its first line.
    expect(aside?.textContent).toContain("approximate-design determinant maximization");
    expect(aside?.textContent).toContain("no partition-induced centroids or label-removal update.");
    for (const entry of data.priorArt) {
      const claim = core.claims[entry.id];
      expect(claim, entry.id).toBeDefined();
      expect(aside?.textContent).toContain(claim?.title);
    }
  });

  it("shows the year strip and one mark per dated source", () => {
    const container = page();
    const dated = Object.values(core.papers).filter((paper) => paper.year !== null);
    expect(container.querySelectorAll("circle.year-strip__mark")).toHaveLength(dated.length);
  });
});
