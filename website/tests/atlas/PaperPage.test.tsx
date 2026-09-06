import {render, screen, within} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import PaperPage, {orderedFields} from "../../src/atlas/pages/PaperPage";
import {atlas, core, renderedPaper} from "./renderData";

function page(key: string): HTMLElement {
  const {container} = render(<PaperPage core={core} data={renderedPaper(key)} />);
  return container;
}

describe("orderedFields", () => {
  it("puts the result, then why it matters, then the use, then the boundary", () => {
    const names = orderedFields({
      Use: "u",
      "Does not transfer": "d",
      Result: "r",
      "Why it matters here": "w"
    }).map(([name]) => name);
    expect(names).toEqual(["Result", "Why it matters here", "Use", "Does not transfer"]);
  });

  it("keeps an unlisted field, in its own order, after the four", () => {
    const names = orderedFields({Verification: "v", Sources: "s", Use: "u"}).map(([name]) => name);
    expect(names).toEqual(["Use", "Verification", "Sources"]);
  });

  it("drops what the caller asks it to", () => {
    expect(orderedFields({Paper: "p", Use: "u"}, ["Paper"]).map(([name]) => name)).toEqual(["Use"]);
  });
});

describe("PaperPage", () => {
  it("titles the page with the paper and links each author to their page", () => {
    page("Alsing-Wandelt-2019");
    expect(screen.getByRole("heading", {level: 1}).textContent).toBe(
      "Nuisance Hardened Data Compression for Fast Likelihood-Free Inference"
    );
    expect(screen.getByRole("link", {name: "Alsing"})).toHaveAttribute("href", "/research/authors/alsing/");
    expect(screen.getByRole("link", {name: "Wandelt"})).toHaveAttribute("href", "/research/authors/wandelt/");
    expect(screen.getByText("2019")).toBeInTheDocument();
  });

  it("links the tradition back to its anchor on the literature page", () => {
    page("Alsing-Wandelt-2019");
    expect(screen.getByRole("link", {name: "Score compression and ratio estimation"})).toHaveAttribute(
      "href",
      "/research/literature/#tradition-score-compression"
    );
  });

  it("offers the external identifiers as plain outbound links", () => {
    const container = page("Alsing-Wandelt-2019");
    const doi = screen.getByRole("link", {name: /doi:/});
    expect(doi).toHaveAttribute("href", "https://doi.org/10.1093/mnras/stz1900");
    expect(doi).toHaveAttribute("rel", "noopener noreferrer");
    expect(screen.getByRole("link", {name: /arXiv:/})).toHaveAttribute("href", "https://arxiv.org/abs/1903.01473");
    expect(container.querySelectorAll(".paper-links a")).toHaveLength(2);
  });

  it("shows the annotation under its own heading, without repeating the title", () => {
    const container = page("Alsing-Wandelt-2019");
    const annotation = container.querySelector(".paper-annotation");
    expect(within(annotation as HTMLElement).getByRole("heading", {level: 2}).textContent).toBe(
      "Alsing & Wandelt (2019) — nuisance-hardened compression"
    );
    expect(annotation?.textContent).toContain("nuisance Schur geometry");
    const terms = annotation ? Array.from(annotation.querySelectorAll("dt")).map((node) => node.textContent) : [];
    expect(terms).not.toContain("Paper");
  });

  it("lists the claims that cite it, and the ones it is merely relevant to separately", () => {
    const container = page("Alsing-Wandelt-2019");
    const paper = atlas.papers["Alsing-Wandelt-2019"];
    const sections = Array.from(container.querySelectorAll("section.atlas__section"));
    const citedBy = sections.find((node) => node.textContent.startsWith("Cited by"));
    const relevant = sections.find((node) => node.textContent.startsWith("Relevant to"));
    for (const id of paper?.citedBy ?? []) expect(citedBy?.textContent).toContain(id);
    expect(relevant?.textContent).toContain("DS-EFFICIENT-SCORE-DOMINATION");
    // An id that is already cited must not be repeated in the second list.
    expect(relevant?.textContent).not.toContain("DS-SCHUR ");
  });

  it("omits the relevance section when it would repeat the citations", () => {
    const container = page("Barnes-Han-Ozgur-2018");
    const headings = Array.from(container.querySelectorAll("h2")).map((node) => node.textContent);
    expect(headings).not.toContain("Relevant to");
  });

  it("renders every paper in the bibliography without throwing", () => {
    for (const key of Object.keys(atlas.papers)) {
      expect(() => render(<PaperPage core={core} data={renderedPaper(key)} />), key).not.toThrow();
    }
  });
});
