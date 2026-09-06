import AxeBuilder from "@axe-core/playwright";
import type {Page} from "@playwright/test";
import {expect, test} from "@playwright/test";

/**
 * Drives the real header control (`.theme-toggle`, wired to Docusaurus's
 * `useColorMode`) rather than `page.emulateMedia`, so the scan below
 * exercises what a reader actually does. Docusaurus persists the chosen
 * theme to `localStorage` and restores it on every navigation, so a route
 * that was left in dark mode by an earlier iteration would otherwise leak
 * into the next route's "light" scan; this only clicks the toggle when the
 * page is not already in the requested theme, which both avoids that leak
 * and gives every scan a known starting state.
 */
async function setTheme(page: Page, theme: "light" | "dark"): Promise<void> {
  const current = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
  if (current === theme) return;
  const label = theme === "dark" ? "Switch to dark mode" : "Switch to light mode";
  await page.getByRole("button", {name: label}).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
  // The header is `position: sticky` with `backdrop-filter: blur(18px)`, which
  // needs its own compositor pass; sampling it (or nearby recolored text) in
  // the same tick the attribute flips is a real, reproducible source of
  // false-positive `color-contrast` violations at transiently-blended colors
  // that never render for a reader — confirmed by scanning this page 15 times
  // back-to-back and watching the reported foreground hex (and the flagged
  // element) change from run to run while the steady-state token pairs stay
  // comfortably compliant. A short settle avoids scanning mid-repaint.
  await page.waitForTimeout(200);
}

/**
 * Every route the shell renders. The scan below covers a subset; this list is
 * what the landmark check walks, and it is deliberately the wider of the two.
 */
const ROUTES = [
  "./",
  "./get-started/",
  "./walkthroughs/",
  "./walkthroughs/flowcyt/",
  "./walkthroughs/hep/",
  "./walkthroughs/michelson/",
  "./walkthroughs/ratios/",
  "./research/"
];

/**
 * The routes scanned by axe. The two docs-plugin routes are scanned as well as
 * the home page: they render through swizzled theme components
 * (src/theme/DocRoot/Layout/Main and src/theme/DocItem/Layout) rather than the
 * stock ones, so nothing upstream vouches for their markup.
 */
const SCANNED = [
  "./",
  "./get-started/",
  "./research/",
  "./walkthroughs/",
  "./walkthroughs/flowcyt/",
  "./walkthroughs/hep/",
  "./walkthroughs/michelson/",
  "./walkthroughs/ratios/"
];

test("home defines the task and runs nothing", async ({page}) => {
  await page.goto("./");
  // Definitions and references, in ordinary type: no slogan, no demo, no
  // measured comparison. The equations render through KaTeX at build time.
  await expect(page.getByRole("heading", {name: "ScoreQuant", level: 1})).toBeVisible();
  const sections = ["The setting", "Hard binning, and what it costs", "The task, stated twice", "The criteria", "Where the scores come from", "Where each of these is derived"];
  const headings = (await page.getByRole("heading", {level: 2}).allInnerTexts()).map((text) => text.replace(/[\u200B\s]+$/g, ""));
  expect(headings).toEqual(sections);
  expect(await page.locator(".katex-display").count()).toBeGreaterThan(1);
  await expect(page.getByRole("link", {name: "Why bin at all"})).toHaveAttribute("href", /\/scorequant\/docs\/book\/ch01-why-bin\//);
  await expect(page.getByRole("button", {name: /browser/i})).toHaveCount(0);
});

test("every route renders one main landmark and none of them loads a runtime", async ({page}) => {
  const heavyRequests: string[] = [];
  page.on("request", (request) => {
    if (/pyodide|marimo|scorequant-.*\.whl/.test(request.url())) heavyRequests.push(request.url());
  });
  for (const route of ROUTES) {
    await page.goto(route);
    // #main-content is AppShell's own landmark, and it must be the only one.
    // The stock layout of the docs plugin renders a second <main> of its own
    // inside it; src/theme/DocRoot/Layout/Main exists to strip that. Nested
    // landmarks are invalid HTML and an accessibility failure, and the axe
    // scan below only covers SCANNED, so this count is what catches a
    // regression on the routes it does not reach.
    await expect(page.locator("#main-content")).toBeVisible();
    expect(await page.locator("main").count(), `nested landmark on ${route}`).toBe(1);
  }
  // Every route must cost an ordinary page load; only pressing a run button
  // on a walkthrough may reach the runtime.
  expect(heavyRequests).toEqual([]);
});

/**
 * One test per route rather than one loop inside a single test. Sixteen axe
 * analyses plus their navigations and theme settles do not fit in one 30s
 * budget on a CI runner — the loop form failed there while passing locally,
 * which is the least useful way for a suite to fail. Split, each route gets
 * its own budget and the projects' workers run them in parallel; the set of
 * assertions is unchanged.
 */
for (const route of SCANNED) {
  test(`${route} has no accessibility violations in either theme`, async ({page}) => {
    await page.goto(route);
    for (const theme of ["light", "dark"] as const) {
      await setTheme(page, theme);
      const accessibility = await new AxeBuilder({page}).analyze();
      expect(accessibility.violations, `accessibility violations on ${route} (${theme} mode)`).toEqual([]);
    }
  });
}

test("core learning routes render and search opens from the keyboard", async ({page}) => {
  await page.goto("./walkthroughs/");
  await expect(page.getByRole("heading", {name: "Walkthroughs", level: 1})).toBeVisible();
  await expect(page.locator(".walkthrough-card")).toHaveCount(4);
  await expect(page.getByRole("link", {name: "A classifier instead of a likelihood"})).toBeVisible();
  await page.keyboard.press("Control+k");
  await expect(page.getByRole("dialog", {name: "Search ScoreQuant"})).toBeVisible();
  await page.getByPlaceholder("Search concepts, tasks, and symbols").fill("ExecutionConfig");
  await expect(page.getByRole("link", {name: /ExecutionConfig/})).toBeVisible();
});

test("the get-started LiveFit demo stays behind its click, then actually reaches the runtime", async ({page}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "One activation-gated runtime pass is sufficient.");
  test.setTimeout(120_000);
  const heavyRequests: string[] = [];
  page.on("request", (request) => {
    if (/pyodide|marimo|scorequant-.*\.whl/.test(request.url())) heavyRequests.push(request.url());
  });

  await page.goto("./get-started/");
  await expect(page.getByRole("heading", {name: "Get started", level: 1})).toBeVisible();
  await expect(page.getByText("D-efficiency printed by get_started_program.py")).toBeVisible();
  const committedValue = await page.locator(".live-fit__committed .live-fit__value").innerText();

  // The invariant's usual, negative direction: navigating to the page and
  // letting it settle -- including the committed panel above, which needs no
  // runtime at all -- must not have reached Pyodide or the wheel.
  expect(heavyRequests).toEqual([]);

  await page.getByRole("button", {name: "Refit this table in your browser"}).click();
  await expect(page.locator(".live-fit__state")).toHaveText(/complete|error/, {timeout: 110_000});
  await expect(page.locator(".live-fit__state")).toHaveText("complete");

  // The other direction, which a demo that silently never worked would also
  // pass if only the negative half were asserted: activation actually did
  // reach the heavy runtime, and the run that came back agrees with the
  // number this page already published.
  expect(heavyRequests.length).toBeGreaterThan(0);
  const liveValue = await page.locator(".live-fit__result--live .live-fit__value").innerText();
  expect(liveValue).toBe(committedValue);
});

test("the flowcyt walkthrough tells the study end to end without loading a runtime", async ({page}) => {
  // The narrative route must stay a narrative route: the moment it pulls the
  // wheel it stops meeting the ordinary-route budget, and the reader pays 15 MB
  // for a page they may only be reading.
  const heavyRequests: string[] = [];
  page.on("request", (request) => {
    if (/pyodide|marimo|scorequant-.*\.whl|flowcyt-scores/.test(request.url())) {
      heavyRequests.push(request.url());
    }
  });
  await page.goto("./walkthroughs/flowcyt/");

  await expect(page.getByRole("heading", {name: /Bone-marrow cell populations/})).toBeVisible();
  for (const section of [
    "The cells, their labels, and a patient's fractions",
    "The data and its licence",
    "The numbers"
  ]) {
    await expect(page.getByRole("heading", {name: section})).toBeVisible();
  }

  // The licence is not decoration: the data is CC BY-NC-SA and the attribution
  // travels with anything derived from it.
  await expect(page.getByText("CC-BY-NC-SA-4.0")).toBeVisible();
  await expect(page.getByText(/Marchand-Maillet/)).toBeVisible();

  await expect(page.getByRole("img", {name: /composition of every patient/})).toBeVisible();
  await expect(page.getByRole("img", {name: /macro RMSE against bin budget/})).toBeVisible();
  await expect(page.getByRole("img", {name: /FS INT intensity distribution/})).toBeVisible();

  expect(heavyRequests).toEqual([]);

  const accessibility = await new AxeBuilder({page}).analyze();
  expect(accessibility.violations).toEqual([]);
});

test("the michelson article runs from the instrument to the experiment without loading a runtime", async ({page}) => {
  const heavyRequests: string[] = [];
  page.on("request", (request) => {
    if (/pyodide|marimo|scorequant-.*\.whl|walkthrough-scores/.test(request.url())) heavyRequests.push(request.url());
  });
  await page.goto("./walkthroughs/michelson/");
  await expect(page.getByRole("heading", {name: /Phase Estimation in a Michelson Interferometer/, level: 1})).toBeVisible();

  // The article order: the subject before the library, the experiment last.
  const sections = ["1. The Michelson interferometer", "2. Why bin the measurement?", "3. Measurement model", "4. The analytic score", "What a photon tells you", "The objective", "The result", "Try it: the counter budget", "What it means"];
  // Docusaurus appends a zero-width-space anchor to every heading; strip it.
  const headings = (await page.getByRole("heading", {level: 2}).allInnerTexts()).map((text) => text.replace(/[\u200B\s]+$/g, ""));
  expect(headings).toEqual(sections);

  // The bench diagram, the fringe law and the three analytic-score panels come
  // before any code; the criterion that defines the objective is stated before
  // the result. The score panels are served from `static/figures/`, so a
  // reference into the gitignored `walkthrough-figures/` fails here too, not
  // only in tests/figures.test.ts.
  await expect(page.getByRole("img", {name: /Michelson interferometer bench/})).toBeVisible();
  await expect(page.getByRole("img", {name: "Fringe intensity along the aperture"})).toBeVisible();
  await expect(page.getByRole("img", {name: /^Phase score along the detector/})).toBeVisible();
  await expect(page.getByRole("img", {name: /^Fringe-frequency score along the detector/})).toBeVisible();
  await expect(page.getByRole("img", {name: /^The Michelson model in score space/})).toBeVisible();
  await expect(page.getByText(/would optimise the information\s+about the pair/)).toBeVisible();
  expect(await page.locator(".katex-display").count()).toBeGreaterThan(1);

  // The experiment: one control, keyboard-operable, with a reset and a static table.
  const radios = page.getByRole("radiogroup", {name: "Counters K"}).getByRole("radio");
  await expect(radios).toHaveCount(4);
  await expect(page.getByRole("radio", {name: "6"})).toBeChecked();
  await expect(page.getByRole("img", {name: /Aperture readout at 6 counters/})).toBeVisible();
  const reset = page.getByRole("button", {name: "Reset to the headline budget"});
  await expect(reset).toBeDisabled();
  await page.getByRole("radio", {name: "6"}).focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("radio", {name: "8"})).toBeChecked();
  await expect(page.getByRole("img", {name: /Aperture readout at 8 counters/})).toBeVisible();
  await expect(reset).toBeEnabled();
  await reset.click();
  await expect(page.getByRole("radio", {name: "6"})).toBeChecked();
  await expect(page.getByRole("img", {name: /Aperture readout at 6 counters/})).toBeVisible();
  await expect(page.getByRole("table", {name: /committed sweep/i})).toBeVisible();
  await expect(page.getByRole("button", {name: "Refit this budget in your browser"})).toBeVisible();

  expect(heavyRequests).toEqual([]);
});

test("the michelson refit reproduces the committed profiled retention at the headline budget", async ({page}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "One activation-gated runtime pass is sufficient.");
  test.setTimeout(240_000);
  await page.goto("./walkthroughs/michelson/");
  const committedValue = await page.locator(".budget-explorer .live-fit__committed .live-fit__value").innerText();
  await page.getByRole("button", {name: "Refit this budget in your browser"}).click();
  await expect(page.locator(".live-fit__state")).toHaveText(/complete|error/, {timeout: 230_000});
  await expect(page.locator(".live-fit__state")).toHaveText("complete");
  const liveValue = await page.locator(".live-fit__result--live .live-fit__value").innerText();
  expect(liveValue).toBe(committedValue);
  await expect(page.getByRole("img", {name: /Your browser's readout at 6 counters/})).toBeVisible();
});
