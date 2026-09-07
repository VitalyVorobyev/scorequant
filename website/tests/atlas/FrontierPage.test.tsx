import {fireEvent, render, screen, within} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import type {CoreClaim} from "../../src/atlas/core";
import type {FrontierData} from "../../src/atlas/pages/FrontierPage";
import FrontierPage from "../../src/atlas/pages/FrontierPage";
import {atlas, core} from "../atlasFixtures";

function claimOf(id: string): CoreClaim {
  const claim = core.claims[id];
  if (!claim) throw new Error(`no claim ${id}`);
  return claim;
}

const data: FrontierData = {
  themes: atlas.themes.map((theme) => ({
    ...theme,
    claims: theme.claims.map((id) => ({
      ...claimOf(id),
      statementHtml: `<p>Statement of ${id}.</p>`,
      noteHtml: `<p>What ${id} would settle.</p>`,
    })),
  })),
};

function section(name: string | RegExp): HTMLElement {
  const heading = screen.getByRole("heading", {name});
  const parent = heading.closest("section");
  if (!parent) throw new Error(`no section for ${String(name)}`);
  return parent;
}

describe("FrontierPage", () => {
  it("gives every theme that has questions a heading, and skips the ones that do not", () => {
    render(<FrontierPage core={core} data={data} />);
    for (const theme of atlas.themes) {
      const heading = screen.queryByRole("heading", {name: theme.label, level: 2});
      if (theme.claims.length > 0) expect(heading, theme.slug).not.toBeNull();
      else expect(heading, theme.slug).toBeNull();
    }
    expect(atlas.themes.some((theme) => theme.claims.length === 0)).toBe(true);
  });

  it("states each question in full, in the theme's own order, and links it to its page", () => {
    render(<FrontierPage core={core} data={data} />);
    const foundations = atlas.themes.find((theme) => theme.slug === "foundations");
    if (!foundations) throw new Error("no foundations theme");
    const block = section("Foundations");
    const titles = within(block)
      .getAllByRole("heading", {level: 3})
      .map((heading) => heading.textContent);
    const rank = (id: string): number => (core.frontier.includes(id) ? core.frontier.indexOf(id) : core.frontier.length);
    const ordered = [...foundations.claims].sort(
      (a, b) => rank(a) - rank(b) || (claimOf(a).editorial?.title ?? "").localeCompare(claimOf(b).editorial?.title ?? ""),
    );
    expect(titles).toEqual(ordered.map((id) => claimOf(id).editorial?.title));
    for (const id of foundations.claims) {
      expect(within(block).getByText(`Statement of ${id}.`)).toBeInTheDocument();
      expect(within(block).getByRole("link", {name: claimOf(id).editorial?.title ?? claimOf(id).title})).toHaveAttribute(
        "href",
        `/research/claims/${claimOf(id).slug}/`,
      );
    }
  });

  it("surrounds a question with what bounds it, what is settled beside it and what it would unlock", () => {
    render(<FrontierPage core={core} data={data} />);
    const question = screen.getByRole("heading", {name: claimOf("OPEN-DS-MARGINS-NONCENTERED").editorial?.title ?? ""}).closest("section");
    if (!question) throw new Error("no question block");

    fireEvent.click(within(question).getByText("Statement and research context", {selector: "summary"}));
    expect(within(question).getByRole("heading", {name: "Already excluded"})).toBeInTheDocument();
    expect(within(question).getByText("CE-DS-MARGINS-RANK-VACUITY-001")).toBeInTheDocument();
    expect(within(question).getByRole("heading", {name: "Settled next to it"})).toBeInTheDocument();
    expect(within(question).getByRole("heading", {name: "Raised by"})).toBeInTheDocument();

    const settled = atlas.edges.filter((edge) => edge.source === "OPEN-DS-MARGINS-NONCENTERED" && edge.type === "rests_on");
    for (const edge of settled) expect(within(question).getAllByText(edge.target).length).toBeGreaterThan(0);
  });

  it("marks a parked question and says what parked means", () => {
    const {container} = render(<FrontierPage core={core} data={data} />);
    const parked = Object.values(atlas.claims).filter((claim) => claim.parked && claim.theme !== null);
    expect(parked).toHaveLength(1);
    const marks = container.querySelectorAll(".provenance-band__mark");
    expect(marks).toHaveLength(parked.length);
    expect(screen.getByText(/set aside until an explicit decision reopens it/)).toBeInTheDocument();

    const block = parked[0] ? screen.getByRole("heading", {name: claimOf(parked[0].id).editorial?.title ?? ""}).closest("section") : null;
    expect(block?.querySelector(".provenance-band__mark")).not.toBeNull();
  });
});
