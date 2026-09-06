import {render, screen, within} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import type {FixturesIndexData} from "../../src/atlas/pages/FixturesIndexPage";
import FixturesIndexPage from "../../src/atlas/pages/FixturesIndexPage";
import {atlas, core} from "../atlasFixtures";

const data: FixturesIndexData = {
  sampleSize: Object.fromEntries(Object.values(atlas.fixtures).map((fixture) => [fixture.id, fixture.weights.length]))
};

function row(id: string): HTMLElement {
  const line = screen.getByText(id).closest("tr");
  if (!line) throw new Error(`no row for ${id}`);
  return line;
}

describe("FixturesIndexPage", () => {
  it("says what a fixture is before tabling any of them", () => {
    render(<FixturesIndexPage core={core} data={data} />);
    expect(screen.getByText(/exact rational scores and weights/)).toBeInTheDocument();
    expect(screen.getByText(/verified by enumerating every labelling or by an exact formula/)).toBeInTheDocument();
  });

  it("gives every fixture a row, sorted by criterion and then by identifier", () => {
    const {container} = render(<FixturesIndexPage core={core} data={data} />);
    const ids = Array.from(container.querySelectorAll("tbody .entity-link__id")).map((node) => node.textContent);
    const order = atlas.vocabulary.criteria.map((entry) => entry.id);
    const expected = Object.values(atlas.fixtures)
      .slice()
      .sort((a, b) => order.indexOf(a.criterion[0] ?? "") - order.indexOf(b.criterion[0] ?? "") || a.id.localeCompare(b.id))
      .map((fixture) => fixture.id);
    expect(ids).toEqual(expected);
    expect(ids.length).toBe(Object.keys(atlas.fixtures).length);
  });

  it("reports the shape of the data and what each fixture settles", () => {
    render(<FixturesIndexPage core={core} data={data} />);
    const fixture = atlas.fixtures["CE-DS-MATRIX-TILT-NONQUASICONVEX-001"];
    if (!fixture) throw new Error("no CE-DS-MATRIX-TILT-NONQUASICONVEX-001 in the data");
    const line = row(fixture.id);
    expect(within(line).getByRole("link", {name: new RegExp(fixture.id)})).toHaveAttribute("href", `/research/counterexamples/${fixture.slug}/`);
    expect(within(line).getByText("Profiled Ds-optimality")).toBeInTheDocument();
    expect(within(line).getByText("Finite assignment")).toBeInTheDocument();
    const numbers = Array.from(line.querySelectorAll(".fixtures-index__number")).map((node) => node.textContent);
    expect(numbers).toEqual([String(fixture.weights.length), String(fixture.dimension), String(fixture.K)]);

    for (const citation of fixture.citedBy) {
      const claim = atlas.claims[citation.claim];
      if (!claim) throw new Error(`no claim ${citation.claim}`);
      const link = within(line).getByRole("link", {name: (name: string) => name.includes(claim.title)});
      expect(link).toHaveAttribute("href", `/research/claims/${claim.slug}/`);
      const verb = link.parentElement?.textContent ?? "";
      expect(verb.startsWith(citation.type === "refuted_by" ? "refutes" : "bounds")).toBe(true);
    }
  });

  it("marks the fixtures the library refuses on, and only those", () => {
    const {container} = render(<FixturesIndexPage core={core} data={data} />);
    const refusals = Object.values(atlas.fixtures).filter((fixture) => fixture.refusal !== null);
    expect(container.querySelectorAll(".provenance-band__mark")).toHaveLength(refusals.length + 1);
    for (const fixture of refusals) {
      expect(within(row(fixture.id)).getByText("refusal")).toBeInTheDocument();
    }
  });
});
