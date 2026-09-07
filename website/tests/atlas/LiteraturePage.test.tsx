import {fireEvent, render, screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";
import LiteraturePage from "../../src/atlas/pages/LiteraturePage";
import {atlas, core, literatureData} from "./renderData";

describe("LiteraturePage", () => {
  it("lists every paper with author links while disclosing the timeline and search records", () => {
    const {container} = render(<LiteraturePage core={core} data={literatureData()} />);
    expect(container.querySelectorAll(".research-paper-row")).toHaveLength(Object.keys(atlas.papers).length);
    const timeline = container.querySelector(".year-strip")?.closest("details");
    expect(timeline).not.toHaveAttribute("open");
    expect(container.querySelector("#closest-prior-work")).not.toHaveAttribute("open");
    expect(container.querySelectorAll('a[href^="/research/authors/"]').length).toBeGreaterThan(20);
  });
  it("searches publications and filters by author with URL state", () => {
    const {container} = render(<LiteraturePage core={core} data={literatureData()} />);
    fireEvent.change(screen.getByRole("searchbox"), {target: {value: "Nuisance Hardened"}});
    expect(container.querySelectorAll(".research-paper-row")).toHaveLength(1);
    expect(window.location.search).toContain("q=Nuisance+Hardened");
    fireEvent.click(screen.getByRole("button", {name: "Clear filters"}));
    const author = Object.values(core.authors).find((a) => a.name === "Kiefer");
    if (!author) throw new Error("Kiefer missing");
    fireEvent.change(screen.getByRole("combobox"), {target: {value: author.slug}});
    expect(container.querySelectorAll(".research-paper-row")).toHaveLength(author.papers.length);
    expect(screen.getByRole("link", {name: /View Kiefer/})).toHaveAttribute("href", `/research/authors/${author.slug}/`);
  });
  it("keeps prior-art coverage independent of home selection", () => {
    const data = literatureData();
    expect(data.priorArt.map((p) => p.id)).toEqual(
      Object.values(atlas.claims)
        .filter((c) => c.priorArt.length)
        .map((c) => c.id),
    );
    expect(data.priorArt.some((p) => !atlas.headline.includes(p.id))).toBe(true);
  });
});
