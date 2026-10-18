import {describe, expect, it, vi} from "vitest";

// Mocked before the module under test is imported (vitest hoists `vi.mock`
// above the imports below) so `facts.ts`'s own import of the generated JSON
// resolves to this fixture instead of the committed `{"pages": {}}`
// placeholder.
vi.mock("../src/generated/walkthrough-data.json", () => ({
  default: {
    pages: {
      hep: {
        headlineGap: {
          source: "examples/outputs/hep-classifier.json#/scorequant_vs_classifier_binning/profiled_retention_gap",
          text: "0.5008",
          value: 0.5007568385278702
        }
      }
    },
    schemaVersion: 1
  }
}));

const {factsFor, factValue, WalkthroughFactError} = await import("../src/lib/facts");

describe("factsFor", () => {
  it("returns the generator's formatted text for a present key", () => {
    expect(factsFor("hep")("headlineGap")).toBe("0.5008");
  });

  it("throws a named error naming the page when the page is missing", () => {
    expect(() => factsFor("michelson")("headlineGap")).toThrow(WalkthroughFactError);
    expect(() => factsFor("michelson")("headlineGap")).toThrow(/"michelson"/);
  });

  it("throws a named error naming the key when the page exists but the key does not", () => {
    expect(() => factsFor("hep")("nope")).toThrow(WalkthroughFactError);
    expect(() => factsFor("hep")("nope")).toThrow(/"nope"/);
  });
});

describe("factValue", () => {
  it("returns the raw number behind a fact", () => {
    expect(factValue("hep", "headlineGap")).toBeCloseTo(0.5007568385278702);
  });

  it("throws for a missing key, same as factsFor", () => {
    expect(() => factValue("hep", "nope")).toThrow(WalkthroughFactError);
  });
});
