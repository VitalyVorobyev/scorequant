import {fireEvent, render, screen, within} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import ClaimsIndexPage from "../../src/atlas/pages/ClaimsIndexPage";
import {atlas, core} from "../atlasFixtures";

const EMPTY: Record<string, never> = {};

function renderedIds(container: HTMLElement): string[] {
  return Array.from(container.querySelectorAll(".entity-link__id, .claims-index__section > div > .atlas__id")).map((node) => node.textContent);
}

describe("ClaimsIndexPage", () => {
  it("lists every claim in the record exactly once, before any filtering", () => {
    const {container} = render(<ClaimsIndexPage core={core} data={EMPTY} />);
    const shown = renderedIds(container);
    expect(new Set(shown).size).toBe(shown.length);
    expect([...shown].sort()).toEqual(Object.keys(atlas.claims).sort());
  });

  it("keeps the chapters in registry order, with their sections and section labels", () => {
    const {container} = render(<ClaimsIndexPage core={core} data={EMPTY} />);
    const headings = Array.from(container.querySelectorAll("h2")).map((heading) => heading.textContent);
    const chapters = atlas.chapters.map((chapter) => chapter.label);
    expect(headings.slice(0, chapters.length).map((text) => text.replace(/source$/, ""))).toEqual(chapters);
    expect(headings.at(-1)).toBe("Verification records");

    const universal = atlas.chapters[0];
    if (!universal) throw new Error("no chapters in the data");
    const block = screen.getByRole("heading", {name: new RegExp(`^${universal.label}`), level: 2}).closest("section");
    if (!block) throw new Error("no chapter section");
    for (const section of universal.sections) {
      if (section.claims.length === 0) continue;
      expect(within(block).getByText(section.label ?? "")).toBeInTheDocument();
      for (const id of section.claims) expect(within(block).getByText(id)).toBeInTheDocument();
    }
  });

  it("groups the open questions by theme and keeps a counterexample without a chapter", () => {
    render(<ClaimsIndexPage core={core} data={EMPTY} />);
    const open = screen.getByRole("heading", {name: "Open questions", level: 2}).closest("section");
    if (!open) throw new Error("no open-questions group");
    for (const theme of atlas.themes) {
      const questions = theme.claims.filter((id) => atlas.claims[id]?.chapter === null);
      if (questions.length === 0) continue;
      expect(within(open).getByRole("heading", {name: theme.label, level: 3})).toBeInTheDocument();
      for (const id of questions) expect(within(open).getByText(id)).toBeInTheDocument();
    }
    expect(screen.getByText("D-UNMERGED-DUPLICATES-FAIL")).toBeInTheDocument();
  });

  it("puts the verification records in their own group, and says what one is", () => {
    render(<ClaimsIndexPage core={core} data={EMPTY} />);
    const audits = screen.getByRole("heading", {name: "Verification records", level: 2}).closest("section");
    if (!audits) throw new Error("no verification group");
    const expected = Object.values(atlas.claims).filter((claim) => claim.kind === "audit");
    expect(expected.length).toBeGreaterThan(0);
    for (const claim of expected) expect(within(audits).getByText(claim.id)).toBeInTheDocument();
    expect(within(audits).getByText(/carries no result of its own/)).toBeInTheDocument();
  });

  it("narrows the lists in the browser without hiding anything on the server", () => {
    const {container} = render(<ClaimsIndexPage core={core} data={EMPTY} />);
    fireEvent.change(screen.getByLabelText(/Filter by title or identifier/), {target: {value: "leverage"}});
    const shown = renderedIds(container);
    const expected = Object.values(atlas.claims)
      .filter((claim) => claim.id.toLowerCase().includes("leverage") || claim.title.toLowerCase().includes("leverage"))
      .map((claim) => claim.id);
    expect(shown.sort()).toEqual(expected.sort());
    expect(screen.getByText(`${expected.length} matching.`)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/Filter by title or identifier/), {target: {value: "no such claim"}});
    expect(screen.getByText("Nothing matches.")).toBeInTheDocument();
    expect(renderedIds(container)).toHaveLength(0);
  });
});
