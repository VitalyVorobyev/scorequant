/**
 * The geometry of the reading progress bar and the "On this page" highlight.
 *
 * `src/theme/DocItem/Layout` measures the DOM and delegates the arithmetic to
 * `src/lib/readingProgress`, which is what makes this testable: jsdom performs
 * no layout, so a test that rendered the component would read zero for every
 * rectangle and prove nothing about where the bar actually sits. The numbers
 * below are therefore fed in directly, in the same CSS pixels the component
 * reads from `getBoundingClientRect` and `--header-height`.
 */

import {describe, expect, it} from "vitest";

import {activeHeadingId, articleProgress} from "../src/lib/readingProgress";

const HEADER = 72;

describe("articleProgress", () => {
  // A 4000px article read through a 800px window under a 72px header: the
  // readable span is 4000 - (800 - 72) = 3272px.
  const article = {top: 200, height: 4000};
  const viewport = 800;
  const progressAt = (scrollY: number): number =>
    articleProgress(article.top, article.height, scrollY, viewport, HEADER);

  it("is zero while the article's first line is still under the header's edge", () => {
    expect(progressAt(0)).toBe(0);
    expect(progressAt(article.top - HEADER)).toBe(0);
  });

  it("is one once the article's last line reaches the bottom of the viewport", () => {
    expect(progressAt(article.top - HEADER + 3272)).toBe(1);
  });

  it("reports the fraction of the readable span in between", () => {
    expect(progressAt(article.top - HEADER + 1636)).toBeCloseTo(0.5);
    expect(progressAt(article.top - HEADER + 818)).toBeCloseTo(0.25);
  });

  it("clamps rather than running past either end", () => {
    // Scrolled above the article (a page whose header banner is taller than the
    // scroll position) and scrolled past its end, e.g. into a long footer.
    expect(progressAt(-500)).toBe(0);
    expect(progressAt(9999)).toBe(1);
  });

  it("reports an article shorter than its own window as fully read", () => {
    // 400px of article in a 800px viewport: there is nothing left to scroll
    // into view, so a bar that stayed empty would be reporting a lie.
    expect(articleProgress(200, 400, 0, viewport, HEADER)).toBe(1);
    // And exactly at the boundary, where the readable span is zero.
    expect(articleProgress(200, viewport - HEADER, 0, viewport, HEADER)).toBe(1);
  });
});

describe("activeHeadingId", () => {
  // The line an anchor click parks a heading on: the `scroll-margin-top` the
  // stylesheet gives it, which clears the 72px header by 24px.
  const PARK = HEADER + 24;

  // Three sections, as they sit in the viewport at one scroll position.
  const headings = [
    {id: "one", top: -400},
    {id: "two", top: 40},
    {id: "three", top: 900}
  ];

  it("has no active section while the reader is above the first heading", () => {
    expect(activeHeadingId([{id: "one", top: 300}], PARK, false)).toBeNull();
  });

  it("takes the last heading that has passed the parking line", () => {
    // "two" is at 40px, above the 96px line; "three" is far below it.
    expect(activeHeadingId(headings, PARK, false)).toBe("two");
  });

  it("counts a heading left exactly on the parking line as passed", () => {
    // Where an anchor click leaves it, allowing for the sub-pixel slack. This
    // is the case that matters: testing against the header's own edge instead
    // highlights the section above the one the reader just jumped to.
    expect(activeHeadingId([{id: "one", top: PARK}], PARK, false)).toBe("one");
    expect(activeHeadingId([{id: "one", top: PARK + 8}], PARK, false)).toBe("one");
    expect(activeHeadingId([{id: "one", top: PARK + 9}], PARK, false)).toBeNull();
    // And one that has climbed all the way to the header's own edge, well
    // above the parking line, is passed too.
    expect(activeHeadingId([{id: "one", top: HEADER}], PARK, false)).toBe("one");
  });

  it("takes the last heading once the document is scrolled to its end", () => {
    // A closing section shorter than the viewport never brings its own heading
    // up to the parking line, so it could never otherwise become active.
    expect(activeHeadingId(headings, PARK, true)).toBe("three");
  });

  it("has no active section when the page has no headings", () => {
    expect(activeHeadingId([], PARK, false)).toBeNull();
    expect(activeHeadingId([], PARK, true)).toBeNull();
  });
});
