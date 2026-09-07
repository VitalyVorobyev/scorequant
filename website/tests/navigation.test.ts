import {describe, expect, it} from "vitest";

import {isActiveNavEntry} from "../src/lib/navigation";

// The deployed baseUrl (the portal owns the site root, with the documentation
// beneath it at /docs/, ADR 0033). The predicate may not hard-code it — it is
// exercised here only as an input fixture.
const base = "/scorequant";

describe("primary navigation highlighting", () => {
  it("marks /walkthroughs active on its own route and on nested routes", () => {
    expect(isActiveNavEntry(`${base}/walkthroughs/`, "/walkthroughs")).toBe(true);
    expect(isActiveNavEntry(`${base}/walkthroughs/michelson/`, "/walkthroughs")).toBe(true);
  });

  it("does not mark /walkthroughs active on the research route", () => {
    expect(isActiveNavEntry(`${base}/research/`, "/walkthroughs")).toBe(false);
  });

  it("marks /research active on its own route and on nested routes", () => {
    expect(isActiveNavEntry(`${base}/research/`, "/research")).toBe(true);
    expect(isActiveNavEntry(`${base}/research/some-page/`, "/research")).toBe(true);
  });

  it("does not mark /get-started active on the home route", () => {
    expect(isActiveNavEntry(`${base}/`, "/get-started")).toBe(false);
  });
});
