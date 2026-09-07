import {render, screen} from "@testing-library/react";
import {createElement} from "react";
import {describe, expect, it} from "vitest";

import {RatioWorkflow} from "../src/components/RatioWorkflow";

describe("RatioWorkflow", () => {
  it("gives the schematic an accessible name built from its title and description", () => {
    render(createElement(RatioWorkflow, {caption: "The two lanes."}));
    expect(
      screen.getByRole("img", {name: /^From a classifier to Fisher-preserving bins/})
    ).toBeInTheDocument();
  });

  it("renders the caption it was given", () => {
    render(createElement(RatioWorkflow, {caption: "The two lanes."}));
    expect(screen.getByText("The two lanes.")).toBeInTheDocument();
  });

  it("draws the five workflow steps and the three oracle steps", () => {
    const {container} = render(createElement(RatioWorkflow, {caption: "The two lanes."}));
    expect(container.querySelectorAll("[data-testid='workflow-card']")).toHaveLength(5);
    expect(container.querySelectorAll("[data-testid='oracle-card']")).toHaveLength(3);
  });

  // The dashed link from the ScoreQuant card down into the oracle lane is the
  // page's whole argument: the labels are the same, only the score judging them
  // differs. A refactor that loses it loses the reason the figure exists.
  it("keeps one dashed crossing between the two lanes", () => {
    const {container} = render(createElement(RatioWorkflow, {caption: "The two lanes."}));
    expect(container.querySelectorAll("path.ratio-workflow__flow--oracle")).toHaveLength(1);
  });
});
