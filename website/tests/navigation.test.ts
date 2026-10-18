import {describe, expect, it} from "vitest";

import {isActiveNavEntry} from "../src/lib/navigation";

// The deployed baseUrl (the portal owns the site root, with the documentation
// beneath it at /docs/, ADR 0035). The predicate may not hard-code it — it is
// exercised here only as an input fixture.
const base = "/scorequant";

describe("primary navigation highlighting", () => {
  it("marks /walkthroughs active on its own route and on nested routes", () => {
    expect(isActiveNavEntry(`${base}/walkthroughs/`, "/walkthroughs")).toBe(true);
    expect(isActiveNavEntry(`${base}/walkthroughs/michelson/`, "/walkthroughs")).toBe(true);
  });

  it("does not mark /walkthroughs active on the get-started route", () => {
    expect(isActiveNavEntry(`${base}/get-started/`, "/walkthroughs")).toBe(false);
  });

  it("does not mark /get-started active on the home route", () => {
    expect(isActiveNavEntry(`${base}/`, "/get-started")).toBe(false);
  });
});
