import {fireEvent, render, screen, within} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import type {LibraryData} from "../../src/atlas/pages/LibraryPage";
import LibraryPage from "../../src/atlas/pages/LibraryPage";
import {atlas, core} from "../atlasFixtures";

const data: LibraryData = {
  objects: atlas.library.objects.map((object) => ({...object, roleHtml: `<p>${object.role}.</p>`})),
  refusals: atlas.library.refusals.map((refusal) => ({
    ...refusal,
    reasonHtml: `<p>${refusal.reason}</p>`,
    remedyHtml: `<p>${refusal.remedy}</p>`,
    triggerHtml: `<p>${refusal.trigger}</p>`
  }))
};

describe("LibraryPage", () => {
  it("separates a refusal from a contract error and says a refusal names its fixture", () => {
    render(<LibraryPage core={core} data={data} />);
    expect(screen.getByText(/A contract error means the call was malformed/)).toBeInTheDocument();
    expect(screen.getByText(/declines because what the call asks it to assert is false in general/)).toBeInTheDocument();
  });

  it("tables every refusal with its fixture, trigger, claims and remedy", () => {
    render(<LibraryPage core={core} data={data} />);
    for (const refusal of atlas.library.refusals) {
      const cell = screen.getByText(refusal.code);
      const line = cell.closest("tr");
      if (!line) throw new Error(`no row for ${refusal.code}`);

      const fixture = atlas.fixtures[refusal.code];
      if (!fixture) throw new Error(`refusal ${refusal.code} names no fixture in the data`);
      expect(within(line).getByRole("link", {name: refusal.code})).toHaveAttribute("href", `/research/counterexamples/${fixture.slug}/`);

      expect(within(line).getByText(refusal.trigger)).toBeInTheDocument();
      expect(within(line).getByText(refusal.remedy)).toBeInTheDocument();
      for (const id of refusal.claims) {
        const claim = atlas.claims[id];
        if (!claim) throw new Error(`no claim ${id}`);
        expect(within(line).getByRole("link", {name: (name: string) => name.includes(claim.title)})).toHaveAttribute("href", `/research/claims/${claim.slug}/`);
      }
    }
  });

  it("relates every public object to the results it rests on, group by group", () => {
    const {container} = render(<LibraryPage core={core} data={data} />);
    const groups = ["Criteria", "Solver configurations", "Results and certificates", "Information reports", "Sources and score providers"];
    const article = container.querySelector("article.atlas__wide, article.atlas__measure");
    if (!article) throw new Error("no page article");
    const summaries = Array.from(article.querySelectorAll("details > summary"));
    expect(summaries.map((heading) => heading.textContent)).toEqual(groups);
    for (const summary of summaries) fireEvent.click(summary);

    for (const object of atlas.library.objects) {
      const name = screen.getAllByText(object.name);
      expect(name.length, object.name).toBeGreaterThan(0);
      const relation = name[0]?.closest("div");
      if (!relation) throw new Error(`no relation row for ${object.name}`);
      for (const id of object.claims) {
        const claim = atlas.claims[id];
        if (!claim) throw new Error(`no claim ${id}`);
        expect(within(relation).getByRole("link", {name: (name: string) => name.includes(claim.title)})).toBeInTheDocument();
      }
      if (object.claims.length === 0) expect(within(relation).getByText("No result of its own.")).toBeInTheDocument();
    }
  });

  it("sends every object name into the generated reference", () => {
    const {container} = render(<LibraryPage core={core} data={data} />);
    const links = Array.from(container.querySelectorAll("a.reference-link"));
    expect(links).toHaveLength(atlas.library.objects.length);
    for (const link of links) {
      expect(link.getAttribute("href")).toBe("/scorequant/docs/symbols/");
      expect(link.getAttribute("target")).toBe("_blank");
    }
  });
});
