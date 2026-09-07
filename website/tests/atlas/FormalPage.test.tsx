import {render, screen, within} from "@testing-library/react";
import {describe, expect, it} from "vitest";

import type {Claim} from "../../src/data/atlas";
import type {FormalData} from "../../src/atlas/pages/FormalPage";
import FormalPage from "../../src/atlas/pages/FormalPage";
import {atlas, core} from "../atlasFixtures";

type MachineChecked = NonNullable<Claim["machineChecked"]>;

function certified(): {id: string; machineChecked: MachineChecked}[] {
  const rows: {id: string; machineChecked: MachineChecked}[] = [];
  for (const claim of Object.values(atlas.claims)) {
    if (claim.machineChecked !== null) rows.push({id: claim.id, machineChecked: claim.machineChecked});
  }
  return rows;
}

const data: FormalData = {...atlas.formal, certified: certified()};

function row(name: (accessibleName: string) => boolean): HTMLElement {
  const cell = screen.getByRole("cell", {name});
  const parent = cell.closest("tr");
  if (!parent) throw new Error("no row for that statement");
  return parent;
}

describe("FormalPage", () => {
  it("says the mark is on the statement and not on the implementation", () => {
    render(<FormalPage core={core} data={data} />);
    expect(screen.getByText(/says nothing about the Python implementation/)).toBeInTheDocument();
    expect(screen.getByText(/frozen before the proof is attempted and audited on its own/)).toBeInTheDocument();
    expect(screen.getByRole("link", {name: /formal workspace README/})).toHaveAttribute("href", atlas.formal.readme);
  });

  it("draws the module chain in build order, dashing the module that is only partly done", () => {
    const {container} = render(<FormalPage core={core} data={data} />);
    const figure = screen.getByRole("img", {name: /Lean modules in build order/});
    expect(figure.querySelector("title")?.textContent).toContain(atlas.formal.chain.map((module) => module.module).join(", "));
    expect(container.querySelectorAll(".lean-chain__box")).toHaveLength(atlas.formal.chain.length);
    const partial = atlas.formal.chain.filter((module) => module.partial === true);
    expect(partial.length).toBeGreaterThan(0);
    expect(container.querySelectorAll(".lean-chain__box--partial")).toHaveLength(partial.length);
  });

  it("names each module's claims outside the figure, as links", () => {
    const {container} = render(<FormalPage core={core} data={data} />);
    const modules = container.querySelector(".lean-chain__modules");
    if (!modules) throw new Error("no module list");
    for (const module of atlas.formal.chain) {
      expect(within(modules as HTMLElement).getByText(module.module)).toBeInTheDocument();
      for (const id of module.claims) {
        const claim = atlas.claims[id];
        if (!claim) throw new Error(`no claim ${id}`);
        expect(within(modules as HTMLElement).getByRole("link", {name: (name: string) => name.includes(claim.title)})).toHaveAttribute("href", `/research/claims/${claim.slug}/`);
      }
    }
  });

  it("tables every certified statement with its declaration, files and statement audit", () => {
    const {container} = render(<FormalPage core={core} data={data} />);
    expect(screen.getAllByRole("row")).toHaveLength(data.certified.length + 1);

    const voronoi = atlas.claims["D-EXCHANGE-IMPLIES-VORONOI"];
    if (!voronoi?.machineChecked) throw new Error("D-EXCHANGE-IMPLIES-VORONOI is not machine-checked in the data");
    const line = row((name: string) => name.includes(voronoi.title));
    expect(within(line).getByText(voronoi.machineChecked.declaration)).toBeInTheDocument();
    expect(within(line).getByRole("link", {name: "spec"})).toHaveAttribute("href", voronoi.machineChecked.spec.url);
    expect(within(line).getByRole("link", {name: "proof"})).toHaveAttribute("href", voronoi.machineChecked.proof.url);
    const audit = voronoi.machineChecked.statementAudit;
    if (!audit) throw new Error("no statement audit recorded");
    expect(within(line).getByRole("link", {name: audit.verdict ?? "audited, no verdict line"})).toHaveAttribute("href", audit.url);
    expect(container.textContent).toContain(voronoi.machineChecked.system);
  });

  it("lists what the track leaves out and the axioms it is allowed", () => {
    render(<FormalPage core={core} data={data} />);
    for (const item of atlas.formal.outside) expect(screen.getByText(item)).toBeInTheDocument();
    for (const axiom of atlas.formal.axioms) expect(screen.getByText(axiom)).toBeInTheDocument();
    expect(atlas.formal.axioms).toHaveLength(3);
  });
});
