import {render, screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import AuthorPage from "../../src/atlas/pages/AuthorPage";
import {atlas, core} from "./renderData";

function page(slug: string): HTMLElement {
  const author = atlas.authors[slug];
  if (!author) throw new Error(`no author ${slug}`);
  const {container} = render(<AuthorPage core={core} data={author} />);
  return container;
}

describe("AuthorPage", () => {
  it("names the author and lists their sources oldest first", () => {
    const container = page("tarpey");
    expect(screen.getByRole("heading", {level: 1}).textContent).toBe("Tarpey");
    const items = Array.from(container.querySelectorAll(".author-papers li"));
    expect(items).toHaveLength(3);
    const years = items.map((item) => Number(item.querySelector(".literature-entry__year")?.textContent));
    expect(years).toEqual([...years].sort((a, b) => a - b));
  });

  it("links each source to its page and names the tradition it belongs to", () => {
    const container = page("tarpey");
    const first = container.querySelector(".author-papers li");
    expect(first?.querySelector("a")).toHaveAttribute("href", "/research/literature/tarpey-li-flury-1995/");
    expect(first?.textContent).toContain("Vector quantization and Voronoi theory");
  });

  it("gathers the claims reached through those sources, without repetition", () => {
    const container = page("tarpey");
    const expected = new Set(
      (atlas.authors.tarpey?.papers ?? []).flatMap((key) => atlas.papers[key]?.citedBy ?? [])
    );
    const list = container.querySelector("section.atlas__section .entity-list");
    expect(list?.querySelectorAll(":scope > li")).toHaveLength(expected.size);
    for (const id of expected) expect(list?.textContent).toContain(id);
  });

  it("says so plainly when nothing cites the author's work", () => {
    const orphan = Object.values(atlas.authors).find((author) =>
      author.papers.every((key) => (atlas.papers[key]?.citedBy.length ?? 0) === 0)
    );
    expect(orphan).toBeDefined();
    if (!orphan) return;
    render(<AuthorPage core={core} data={orphan} />);
    expect(screen.getByText("No claim cites these sources yet.")).toBeInTheDocument();
  });

  it("renders every author in the bibliography without throwing", () => {
    for (const author of Object.values(atlas.authors)) {
      expect(() => render(<AuthorPage core={core} data={author} />), author.slug).not.toThrow();
    }
  });
});
