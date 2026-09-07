import {act, fireEvent, render, screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";
import LandscapePage from "../../src/atlas/pages/LandscapePage";
import {core} from "../atlasFixtures";

describe("Explore", () => {
  it("opens with themes and reveals readable results only after selection", () => {
    const {container} = render(<LandscapePage core={core} data={{}} />);
    expect(container.querySelector(".research-result-row")).toBeNull();
    expect(container.querySelector(".landscape")).toBeNull();
    const buttons = screen.getAllByRole("button", {name: /results/});
    expect(buttons[0]).toHaveTextContent(/^D-optimalityStart here/);
    expect(buttons[0]).toHaveTextContent(/\d+ results · \d+ known · \d+ established here · \d+ boundary · \d+ open/);
    fireEvent.click(screen.getByRole("button", {name: /^D-optimality/}));
    expect(window.location.search).toBe("?theme=d");
    expect(screen.getByRole("link", {name: "Stable D partitions have a geometric rule"})).toBeInTheDocument();
    expect(screen.getByRole("heading", {name: "Known"})).toBeInTheDocument();
    expect(screen.getByRole("heading", {name: "Boundary"})).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", {name: "Clear filters"}));
    expect(container.querySelector(".research-result-row")).toBeNull();
  });
  it("searches all records, reports empty selections, and follows restored URL state", () => {
    render(<LandscapePage core={core} data={{}} />);
    fireEvent.change(screen.getByRole("searchbox"), {target: {value: "D-EXCHANGE-IMPLIES-VORONOI"}});
    expect(screen.getByRole("status")).toHaveTextContent("1 results");
    fireEvent.change(screen.getByRole("searchbox"), {target: {value: "no such theorem"}});
    expect(screen.getByRole("status")).toHaveTextContent("No results recorded");
    act(() => {
      window.history.replaceState(null, "", "?theme=ds");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(screen.getByRole("searchbox")).toHaveValue("");
    fireEvent.click(screen.getByText("Change research theme", {selector:"summary"}));
    expect(screen.getByRole("button", {name: /^Profiled Ds-optimality/})).toHaveAttribute("aria-pressed", "true");
  });
  it("retains the advanced matrix on demand", () => {
    const {container} = render(<LandscapePage core={core} data={{}} />);
    fireEvent.click(screen.getByRole("button", {name: "Matrix (advanced)"}));
    expect(container.querySelector(".landscape")).not.toBeNull();
    expect(window.location.search).toContain("view=matrix");
  });
});
