/**
 * The arithmetic behind the reading progress bar and the "On this page"
 * highlight, kept apart from the component that renders them.
 *
 * `src/theme/DocItem/Layout` is the only caller. It measures the DOM and hands
 * the numbers here, because jsdom performs no layout: every rectangle it
 * reports is zero, so a test that drove the component directly would assert
 * nothing about the geometry. These two functions are pure, so the geometry is
 * what `tests/readingProgress.test.ts` actually checks.
 *
 * Every vertical coordinate below is in CSS pixels, and each parameter says
 * whether it is measured from the top of the document or from the top of the
 * viewport. `headerOffset` is the portal's sticky header height -- the
 * `--header-height` token -- which is the line the reader's eye actually starts
 * at, since everything above it is covered.
 */

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

/**
 * The fraction of the article already scrolled past, in `[0, 1]`.
 *
 * Zero while the article's first line is still at or below the header's bottom
 * edge, and one once its last line has reached the bottom of the viewport --
 * that is, when there is nothing left to scroll into view rather than when the
 * article's top has travelled its own height. The readable span is therefore
 * the article's height less the window it is read through.
 *
 * An article shorter than that window has no readable span at all; it is
 * entirely visible, so it reports fully read rather than dividing by zero.
 *
 * @param articleTop - The article's top edge, from the top of the document.
 * @param articleHeight - The article's full height, visible or not.
 * @param scrollY - How far the document is scrolled.
 * @param viewportHeight - The window's inner height.
 * @param headerOffset - The height of the sticky header covering the top.
 */
export function articleProgress(
  articleTop: number,
  articleHeight: number,
  scrollY: number,
  viewportHeight: number,
  headerOffset: number
): number {
  const readable = articleHeight - (viewportHeight - headerOffset);
  if (readable <= 0) return 1;
  return clamp01((scrollY + headerOffset - articleTop) / readable);
}

/** One heading of the contents list, with its current position on screen. */
export interface HeadingPosition {
  /** The heading's `id`, which is what the contents list links to. */
  id: string;
  /** The heading's top edge, from the top of the viewport. */
  top: number;
}

/**
 * How far past the parking line a heading may still sit and count as passed.
 *
 * Without it, a heading left exactly on that line by an anchor click flickers
 * between passed and not passed on sub-pixel scroll positions, and the
 * highlight lands on the section above the one the reader just jumped to.
 */
const PASSED_SLACK = 8;

/**
 * The id of the heading whose section the reader is currently in.
 *
 * The last heading to have travelled up past `parkOffset` wins: sections are
 * delimited by their own headings, so the reader is inside the most recent one
 * they scrolled through. `null` means the reader is still above the first
 * heading -- an article's title and opening paragraph belong to no section, and
 * highlighting the first entry there would be a claim the page does not
 * support.
 *
 * @param headings - The contents entries, in document order.
 * @param parkOffset - The line an anchor click leaves a heading on, which is
 *   the heading's `scroll-margin-top` and sits below the header rather than at
 *   its edge. Testing against the header alone would leave the section the
 *   reader just jumped to unhighlighted.
 * @param atBottom - Whether the document is scrolled to its end. The last
 *   section can be shorter than the viewport, in which case its heading never
 *   reaches that line and it could never otherwise be the active one.
 */
export function activeHeadingId(
  headings: readonly HeadingPosition[],
  parkOffset: number,
  atBottom: boolean
): string | null {
  const last = headings.at(-1);
  if (last === undefined) return null;
  if (atBottom) return last.id;
  let active: string | null = null;
  for (const heading of headings) {
    if (heading.top > parkOffset + PASSED_SLACK) break;
    active = heading.id;
  }
  return active;
}
