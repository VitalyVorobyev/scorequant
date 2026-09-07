import {render, screen} from "@testing-library/react";
import {createElement} from "react";
import {describe, expect, it} from "vitest";

import {BinCompositionHeatmap} from "../src/components/charts/BinCompositionHeatmap";
import {CompositionBars} from "../src/components/charts/CompositionBars";
import {EstimateScatter} from "../src/components/charts/EstimateScatter";
import {MethodComparison} from "../src/components/charts/MethodComparison";
import type {BinComposition, MethodSeries, PatientComposition, PatientEstimate} from "../src/data/showcase";

const populations = ["T cells", "B cells", "other"];

const composition: BinComposition = {
  bins: 2,
  populations,
  rows: [
    {bin: 1, composition: [0.9, 0.05, 0.05], dominant: "T cells"},
    {bin: 0, composition: [0.2, 0.3, 0.5], dominant: "other"}
  ]
};

const estimates: PatientEstimate[] = [
  {patient: 5, expert: [0.05, 0.01, 0.94], binned: [0.048, 0.009, 0.943], unbinned: [0.051, 0.0095, 0.9395]},
  {patient: 6, expert: [0.03, 0.02, 0.95], binned: [0.031, 0.019, 0.95], unbinned: [0.03, 0.021, 0.949]}
];

const methods: MethodSeries[] = [
  {
    isScoreQuant: true,
    key: "soft_voronoi",
    label: "Soft Voronoi",
    points: [
      {bins: 5, heldOutEfficiency: 0.39, macroRmse: 0.03, minimumBinCount: 39},
      {bins: 8, heldOutEfficiency: 0.98, macroRmse: 0.0019, minimumBinCount: 6}
    ]
  },
  {
    isScoreQuant: false,
    key: "marker_kmeans",
    label: "k-means on raw markers",
    points: [
      {bins: 5, heldOutEfficiency: 0.01, macroRmse: 0.04, minimumBinCount: 1000},
      {bins: 8, heldOutEfficiency: null, macroRmse: 0.03, minimumBinCount: 900}
    ]
  }
];

const patients: PatientComposition[] = [
  {patient: 1, role: "reference", fractions: [0.1, 0.05, 0.85]},
  {patient: 5, role: "held-out", fractions: [0.05, 0.01, 0.94]}
];

describe("BinCompositionHeatmap", () => {
  it("renders one row per bin with every population's probability", () => {
    render(createElement(BinCompositionHeatmap, {composition}));
    expect(screen.getByText("bin 1")).toBeInTheDocument();
    expect(screen.getByText("bin 0")).toBeInTheDocument();
    expect(screen.getByText("0.90")).toBeInTheDocument();
    expect(screen.getByText("0.50")).toBeInTheDocument();
  });

  it("marks the dominant cells and no others", () => {
    const {container} = render(createElement(BinCompositionHeatmap, {composition}));
    // 0.9 in the first row and 0.5 in the second are the only entries at or above one half.
    expect(container.querySelectorAll(".bin-composition__cell--dominant")).toHaveLength(2);
  });
});

describe("EstimateScatter", () => {
  it("draws a filled and a hollow mark per patient and visible population", () => {
    const {container} = render(
      createElement(EstimateScatter, {patients: estimates, populations, visible: [0, 1]})
    );
    // 2 patients x 2 populations x (binned + unbinned).
    expect(container.querySelectorAll("circle")).toHaveLength(8);
    expect(screen.getByRole("img", {name: /Estimated against expert population fractions/})).toBeInTheDocument();
  });
});

describe("MethodComparison", () => {
  it("plots the error metric with the unbinned reference line by default", () => {
    render(
      createElement(MethodComparison, {
        baseline: {label: "Unbinned classifier ratio", macroRmse: 0.0017},
        methods
      })
    );
    expect(screen.getByRole("img", {name: /macro RMSE against bin budget/})).toBeInTheDocument();
    expect(screen.getAllByText("Unbinned classifier ratio").length).toBeGreaterThan(0);
  });

  it("plots the retained information without a reference line and skips missing points", () => {
    const {container} = render(createElement(MethodComparison, {methods, metric: "heldOutEfficiency"}));
    expect(screen.getByRole("img", {name: /D-efficiency against bin budget/})).toBeInTheDocument();
    expect(screen.queryByText("Unbinned classifier ratio")).not.toBeInTheDocument();
    // The baseline's 8-bin point has no efficiency and is left out: 2 + 1 marks.
    expect(container.querySelectorAll("circle")).toHaveLength(3);
  });
});

describe("CompositionBars", () => {
  it("draws every population by default and drops the last one when asked", () => {
    const full = render(createElement(CompositionBars, {patients, populations}));
    expect(full.container.querySelectorAll("rect")).toHaveLength(6);
    full.unmount();
    const targets = render(createElement(CompositionBars, {patients, populations, targetsOnly: true}));
    expect(targets.container.querySelectorAll("rect")).toHaveLength(4);
    expect(screen.queryByText("other")).not.toBeInTheDocument();
  });
});
