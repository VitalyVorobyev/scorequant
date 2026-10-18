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
  "./research/",
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
  "./walkthroughs/",
  "./walkthroughs/flowcyt/",
  "./walkthroughs/hep/",
  "./walkthroughs/michelson/",
  "./walkthroughs/ratios/",
  "./research/",
];

test("home states the task and runs nothing", async ({page}) => {
  await page.goto("./");
  // The site root: one sentence, one picture of score-space cells, then the
  // reason, the four results and where to start.
  await expect(page.getByRole("heading", {name: "ScoreQuant", level: 1})).toBeVisible();
  await expect(page.getByRole("img", {name: /^Three panels\. Top left: the raw score plane tinted by six convex cells/})).toBeVisible();
  const sections = ["Why score space", "Walkthroughs", "Get started"];
  const headings = (await page.getByRole("heading", {level: 2}).allInnerTexts()).map((text) => text.replace(/[\u200B\s]+$/g, ""));
  expect(headings).toEqual(sections);
  await expect(page.locator(".home-card")).toHaveCount(4);
  await expect(page.getByRole("link", {name: "Phase Estimation in a Michelson Interferometer"})).toHaveAttribute("href", /\/scorequant\/walkthroughs\/michelson\/?$/);
  // The score is written once, inline, through KaTeX at build time.
  expect(await page.locator(".katex").count()).toBeGreaterThan(0);
  // The documentation stays one click from the root through the shell: the
  // primary navigation's Reference entry -- asserted
  // in the DOM rather than by role, because at mobile widths it sits behind the
  // menu button -- and the footer's Documentation link, which is visible at
  // every width.
  await expect(page.locator('nav[aria-label="Primary"] a').filter({hasText: "Reference"})).toHaveAttribute("href", "/scorequant/docs/");
  await expect(page.getByRole("contentinfo").getByRole("link", {name: "Documentation"})).toHaveAttribute("href", "/scorequant/docs/");
  await expect(page.getByRole("button", {name: /browser/i})).toHaveCount(0);
});

test("the research page states the theorem, its limits and the record, and is in the navigation", async ({page}) => {
  await page.goto("./research/");
  await expect(page.getByRole("heading", {name: "Research", level: 1})).toBeVisible();
  const sections = ["The theorem", "Where it stops", "What is new", "Proofs and counterexamples"];
  const headings = (await page.getByRole("heading", {level: 2}).allInnerTexts()).map((text) => text.replace(/[\u200B\s]+$/g, ""));
  expect(headings).toEqual(sections);
  await expect(page.locator('nav[aria-label="Primary"] a').filter({hasText: "Research"})).toHaveAttribute("href", /\/scorequant\/research\/?$/);
  await expect(page.getByRole("link", {name: "research/formal"})).toHaveAttribute("href", "https://github.com/VitalyVorobyev/scorequant/tree/main/research/formal");
  await expect(page.getByRole("link", {name: "tests/fixtures/counterexamples"})).toHaveAttribute("href", "https://github.com/VitalyVorobyev/scorequant/tree/main/tests/fixtures/counterexamples");
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
 *
 * That budget is set here rather than left at the default 30s: an axe scan
 * walks every node twice, once per theme, and a long walkthrough on a
 * contended CI runner can approach the default.
 */
for (const route of SCANNED) {
  test(`${route} has no accessibility violations in either theme`, async ({page}) => {
    test.setTimeout(120_000);
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
  await expect(page.getByRole("link", {name: "From a classifier to Fisher-preserving bins"})).toBeVisible();
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

  await expect(page.getByRole("heading", {name: /Bone-marrow cell populations/, level: 1})).toBeVisible();

  // The result before the method: the held-out estimate, then why marker
  // gates fall short, then how the categories are built and applied.
  const sections = [
    "1. A patient's composition",
    "2. The counts recover the composition",
    "3. Why gates and clusters fall short",
    "4. Categories drawn in score space",
    "5. What the categories are",
    "6. Applying the frozen rule",
    "7. How many categories",
    "8. What the result means",
  ];
  // Docusaurus appends a zero-width-space anchor to every heading; strip it.
  const headings = (await page.getByRole("heading", {level: 2}).allInnerTexts()).map((text) => text.replace(/[\u200B\s]+$/g, ""));
  expect(headings).toEqual(sections);

  // The licence is not decoration: the data is CC BY-NC-SA and the attribution
  // travels with anything derived from it.
  await expect(page.getByText("CC-BY-NC-SA-4.0")).toBeVisible();
  await expect(page.getByText(/Marchand-Maillet/)).toBeVisible();

  await expect(page.getByRole("img", {name: /composition of every patient/})).toBeVisible();
  await expect(page.getByRole("img", {name: /macro RMSE against bin budget/})).toBeVisible();
  await expect(page.getByRole("img", {name: /D-efficiency against bin budget/})).toBeVisible();
  await expect(page.getByRole("img", {name: /Estimated against expert population fractions/})).toBeVisible();

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
  const sections = [
    "1. The interferometer",
    "2. The model and its score",
    "3. Why equal segments fail",
    "4. Binning in score space",
    "5. The D-optimal partition",
    "6. Reporting the phase alone",
    "7. A reusable rule for the phase",
    "8. What the result means",
  ];
  // Docusaurus appends a zero-width-space anchor to every heading; strip it.
  const headings = (await page.getByRole("heading", {level: 2}).allInnerTexts()).map((text) => text.replace(/[\u200B\s]+$/g, ""));
  expect(headings).toEqual(sections);

  // The bench diagram, the phase score, the fringe law cut into one segment
  // per fringe and the score-space curve carry the argument; the two study
  // figures -- the D geometry, then the profiled partition -- follow in that
  // order. The score panels are served
  // from `static/figures/` and the study figures from the generated
  // `walkthrough-figures/`, so a wrong lane fails here too, not only in
  // tests/figures.test.ts.
  await expect(page.getByRole("img", {name: /Michelson interferometer bench/})).toBeVisible();
  await expect(page.getByRole("img", {name: "Fringe intensity along the aperture"})).toBeVisible();
  await expect(page.getByRole("img", {name: /^Phase score along the detector/})).toBeVisible();
  await expect(page.getByRole("img", {name: /^The Michelson model in score space/})).toBeVisible();
  await expect(page.getByRole("img", {name: /^Three panels\. Top left: the raw score plane tinted by six convex cells/})).toBeVisible();
  await expect(page.getByRole("img", {name: /^Three panels\. Top: the score trajectory's four loops coloured by the six profiled cells/})).toBeVisible();
  // The question is stated before any result is quoted. Asserted on prose that
  // carries no math: KaTeX splits an expression across spans, so a regex over a
  // sentence containing $D_s$ would be matching the renderer, not the article.
  await expect(page.getByText("The choice of bins then decides how much information about the phase survives.")).toBeVisible();
  expect(await page.locator(".katex-display").count()).toBeGreaterThan(1);

  // The experiment: one control, keyboard-operable, with a reset and a static table.
  const radios = page.getByRole("radiogroup", {name: "Counters K"}).getByRole("radio");
  await expect(radios).toHaveCount(4);
  await expect(page.getByRole("radio", {name: "6"})).toBeChecked();
  await expect(page.getByRole("img", {name: /Aperture readout at 6 counters/})).toBeVisible();
  // Both spaces, not only the aperture: the score space is where the criteria
  // actually differ, and it follows the budget with the strip.
  await expect(page.getByRole("img", {name: /Score space at 6 counters, profiled Ds against plain D/})).toBeVisible();
  const reset = page.getByRole("button", {name: "Reset to the headline budget"});
  await expect(reset).toBeDisabled();
  await page.getByRole("radio", {name: "6"}).focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("radio", {name: "8"})).toBeChecked();
  await expect(page.getByRole("img", {name: /Aperture readout at 8 counters/})).toBeVisible();
  await expect(page.getByRole("img", {name: /Score space at 8 counters, profiled Ds against plain D/})).toBeVisible();
  await expect(reset).toBeEnabled();
  await reset.click();
  await expect(page.getByRole("radio", {name: "6"})).toBeChecked();
  await expect(page.getByRole("img", {name: /Aperture readout at 6 counters/})).toBeVisible();
  // The static text alternative to the explorer's bars, which is what a reader
  // who cannot see the chart actually gets. The explorer carries the page's
  // only comparison chart.
  await expect(page.getByRole("table", {name: /Phase information retained, after profiling, compared across 3 binning methods/})).toHaveCount(1);
  await expect(page.getByRole("button", {name: "Refit this budget in your browser"})).toBeVisible();

  expect(heavyRequests).toEqual([]);
});

/**
 * The two reading affordances of `src/theme/DocItem/Layout`, split by viewport.
 *
 * Both depend on layout, which jsdom does not perform, so they cannot be
 * asserted from a component test; the arithmetic behind them is covered by
 * `tests/readingProgress.test.ts` instead. `scaleX(0)` and `scaleX(1)` compute
 * to the matrices below.
 */
const EMPTY_BAR = "matrix(0, 0, 0, 1, 0, 0)";
const FULL_BAR = "matrix(1, 0, 0, 1, 0, 0)";

test("the progress bar reports how much of the article is left, at every width", async ({page}) => {
  await page.goto("./walkthroughs/michelson/");
  const bar = page.locator(".reading-progress__fill");
  await expect(bar).toHaveCSS("transform", EMPTY_BAR);
  await page.evaluate(() => {
    window.scrollTo(0, document.documentElement.scrollHeight);
  });
  // Full, not merely non-empty: the article ends above the footer, so reaching
  // the end of the document means there is nothing of it left to scroll in.
  await expect(bar).toHaveCSS("transform", FULL_BAR);
});

test("the contents panel marks the section the reader is in", async ({page}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop", "The contents panel is hidden below 1080px.");
  await page.goto("./walkthroughs/michelson/");
  const contents = page.getByRole("navigation", {name: "On this page"});
  const target = contents.getByRole("link", {name: "6. Reporting the phase alone"});
  // Nothing is current above the first heading: the title belongs to no section.
  await expect(contents.getByRole("link")).toHaveCount(8);
  await expect(contents.locator("a[aria-current]")).toHaveCount(0);
  await target.click();
  await expect(target).toHaveAttribute("aria-current", "location");
  await expect(contents.locator("a[aria-current]")).toHaveCount(1);
  // The heading it jumped to clears the sticky header rather than parking under
  // it, which is what `scroll-margin-top` on the headings buys. Polled because
  // the scroll is smooth and `boundingBox` does not retry on its own.
  await expect
    .poll(async () => {
      const heading = await page.getByRole("heading", {name: /6\. Reporting the phase alone/}).boundingBox();
      const header = await page.locator(".site-header").boundingBox();
      if (heading === null || header === null) return -1;
      return heading.y - (header.y + header.height);
    })
    .toBeGreaterThanOrEqual(0);
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

