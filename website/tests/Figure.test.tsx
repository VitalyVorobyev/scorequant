import {render, screen} from "@testing-library/react";
import {createElement} from "react";
import {describe, expect, it} from "vitest";

import {Figure} from "../src/components/Figure";

const base = {
  alt: "Phase score along the detector: a periodic curve over four fringe periods.",
  caption: "The phase score along the detector.",
  src: "/scorequant/figures/michelson-phase-score.svg"
};

describe("Figure", () => {
  it("names the image by its alt text and renders the caption", () => {
    render(createElement(Figure, base));
    const image = screen.getByRole("img", {name: base.alt});
    expect(image).toHaveAttribute("src", base.src);
    expect(screen.getByText(base.caption)).toBeInTheDocument();
  });

  it("carries only the base frame class by default", () => {
    const {container} = render(createElement(Figure, base));
    expect(container.querySelector("figure")).toHaveClass("chart-figure");
    expect(container.querySelector("figure")).not.toHaveClass("chart-figure--wide");
    expect(container.querySelector("figure")).not.toHaveClass("chart-figure--paper");
  });

  it("adds the wide and paper modifiers only when asked", () => {
    const {container} = render(createElement(Figure, {...base, paper: true, wide: true}));
    expect(container.querySelector("figure")).toHaveClass(
      "chart-figure",
      "chart-figure--wide",
      "chart-figure--paper"
    );
  });
});
