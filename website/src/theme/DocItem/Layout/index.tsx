import {useDoc} from "@docusaurus/plugin-content-docs/client";
import ContentVisibility from "@theme/ContentVisibility";
import DocItemContent from "@theme/DocItem/Content";
import DocItemPaginator from "@theme/DocItem/Paginator";
import type {Props} from "@theme/DocItem/Layout";
import type {ReactNode, RefObject} from "react";
import {useEffect, useRef, useState} from "react";

import type {HeadingPosition} from "../../../lib/readingProgress";
import {activeHeadingId, articleProgress} from "../../../lib/readingProgress";

/** How close to the document's end still counts as having reached it. */
const BOTTOM_SLACK = 2;

/** The sticky header's height, read from the token that declares it. */
function headerOffset(): number {
  const declared = getComputedStyle(document.documentElement).getPropertyValue("--header-height");
  return Number.parseFloat(declared) || 0;
}

interface ReadingPosition {
  progress: number;
  activeId: string | null;
}

/**
 * Where the reader is in `article`, and which of `ids` names the section
 * they are in.
 *
 * The measurement is a scroll scan rather than an `IntersectionObserver`. With
 * a dozen headings the per-frame cost is nil, and a scan can take its two
 * thresholds from the stylesheet itself -- the `--header-height` token for the
 * bar, each heading's own `scroll-margin-top` for the highlight -- so neither
 * can drift away from where the page actually puts things, which is the failure
 * an observer's root margins invite.
 *
 * Both values start at their pre-scroll state so that the server render and
 * React's first client render agree; the effect below does not run until after
 * hydration, which is when a scroll position exists to measure.
 */
function useReadingPosition(
  article: RefObject<HTMLElement | null>,
  ids: readonly string[]
): ReadingPosition {
  const [position, setPosition] = useState<ReadingPosition>({progress: 0, activeId: null});
  // A fresh array every render would re-run the effect on every render, so it
  // keys off the ids' joined form instead. Heading ids are Docusaurus slugs,
  // which never contain a space.
  const key = ids.join(" ");

  useEffect(() => {
    let frame = 0;

    const measure = (): void => {
      frame = 0;
      const element = article.current;
      if (element === null) return;
      const offset = headerOffset();
      const rect = element.getBoundingClientRect();
      const headings: HeadingPosition[] = [];
      // The line a heading comes to rest on when the reader jumps to it: its
      // own `scroll-margin-top`, read back from the stylesheet rather than
      // recomputed here. A heading parked by an anchor click sits *below* the
      // header, not at its edge, so a spy that tested against the header alone
      // would highlight the section above the one just jumped to. All the
      // headings share one rule, so the first one that resolves speaks for the
      // rest; without any, there is nothing to highlight and the value is moot.
      let parkOffset = offset;
      for (const id of key.split(" ").filter((candidate) => candidate !== "")) {
        const heading = document.getElementById(id);
        if (heading === null) continue;
        if (headings.length === 0) {
          parkOffset = Number.parseFloat(getComputedStyle(heading).scrollMarginTop) || offset;
        }
        headings.push({id, top: heading.getBoundingClientRect().top});
      }
      const atBottom =
        window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - BOTTOM_SLACK;
      setPosition({
        progress: articleProgress(
          rect.top + window.scrollY,
          element.offsetHeight,
          window.scrollY,
          window.innerHeight,
          offset
        ),
        activeId: activeHeadingId(headings, parkOffset, atBottom)
      });
    };

    // Coalesced to one measurement per painted frame: a scroll fires far more
    // often than that, and every extra call is a layout read for a bar that
    // cannot move until the next paint anyway.
    const schedule = (): void => {
      if (frame === 0) frame = window.requestAnimationFrame(measure);
    };

    measure();
    window.addEventListener("scroll", schedule, {passive: true});
    window.addEventListener("resize", schedule, {passive: true});
    return () => {
      if (frame !== 0) window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, [article, key]);

  return position;
}

/**
 * The article frame for the docs-plugin routes: `/walkthroughs`, `/research`
 * and `/get-started`.
 *
 * The stock `DocItem/Layout` cannot run inside the ScoreQuant shell. Its table
 * of contents calls `useTOCHighlight`, which reads
 * `document.querySelector(".navbar").clientHeight` to offset anchors against
 * the Docusaurus navbar. This portal renders its own header from `AppShell` and
 * has no `.navbar` element, so that query returns `null` and every docs page
 * with headings throws `Cannot read properties of null (reading 'clientHeight')`
 * during hydration — the page renders correctly on the server and then crashes
 * to Docusaurus's error boundary in the browser.
 *
 * So the contents list below is built directly from `useDoc().toc`, and the
 * scroll-spy that highlights the reader's section is `useReadingPosition`
 * above, which reads the offsets it needs out of the stylesheet rather than
 * querying a navbar that does not exist. The same measurement drives the
 * progress bar pinned under the header — the contents panel is `display: none`
 * below 1080px, so the bar is what tells a reader on a narrow screen how much
 * of the article is left. The
 * stock Infima `row`/`col` grid, breadcrumbs, version banners and the edit/tags
 * footer are dropped with the stock layout — neither instance is versioned and
 * neither sets `editUrl`.
 *
 * `ContentVisibility` is kept: it is what warns a reader that a page is a draft
 * or unlisted, which is a correctness signal rather than theming.
 */
export default function DocItemLayout({children}: Props): ReactNode {
  const {metadata, frontMatter, toc} = useDoc();
  const showToc = frontMatter.hide_table_of_contents !== true && toc.length > 1;
  const article = useRef<HTMLElement>(null);
  const {progress, activeId} = useReadingPosition(
    article,
    toc.map((heading) => heading.id)
  );
  return (
    <div className={showToc ? "doc-article doc-article--with-toc" : "doc-article"}>
      {/*
       * Decorative, and deliberately so: a `role="progressbar"` whose
       * `aria-valuenow` changed on every frame of a scroll would be
       * announcement spam. The position a screen reader needs is the
       * `aria-current` below, on the contents entry itself.
       */}
      <div className="reading-progress" aria-hidden="true">
        <div className="reading-progress__fill" style={{transform: `scaleX(${progress})`}} />
      </div>
      <article className="doc-article__body" ref={article}>
        <ContentVisibility metadata={metadata} />
        <DocItemContent>{children}</DocItemContent>
        <DocItemPaginator />
      </article>
      {showToc && (
        <nav className="doc-article__toc" aria-label="On this page">
          <span className="doc-article__toc-title">On this page</span>
          <ul>
            {toc.map((heading) => (
              <li key={heading.id} data-level={heading.level}>
                <a
                  href={`#${heading.id}`}
                  aria-current={heading.id === activeId ? "location" : undefined}
                >
                  {heading.value}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </div>
  );
}
