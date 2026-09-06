import type {ReactNode} from "react";

import {REFERENCE_BASE} from "../lib/site";

export interface ReferenceLinkProps {
  /** Path inside the MkDocs reference, without a leading slash: `examples/michelson-phase/`. */
  to: string;
  children: ReactNode;
}

/**
 * A link from a portal page into the MkDocs reference.
 *
 * The reference is a separately built site mounted beside the portal
 * (`REFERENCE_BASE`, ADR 0027), so this is a plain full-page anchor rather
 * than a router `Link` — and, because it is a different site with a different
 * shell, it opens in a new tab. Following it in place drops the reader into an
 * unrelated UI with no way back to the article they were reading; a new tab
 * leaves the article where they left it. `rel="noopener noreferrer"` is the
 * standard companion to `target="_blank"`.
 *
 * The arrow glyph is decorative (`aria-hidden`) and the destination is
 * announced instead by a visually hidden suffix, so the accessible name says
 * the link opens a new tab rather than leaving that a visual-only cue.
 *
 * Pages must never write this URL by hand. A raw MDX anchor
 * (`href="pathname:///reference/…"`) is not routed through Docusaurus's link
 * handling, so that string reaches the browser verbatim and is not a URL at
 * all; a markdown `pathname://` link is rewritten but drops the site's
 * `baseUrl`. `website/tests/reference-links.test.ts` fails on either.
 */
export function ReferenceLink({to, children}: ReferenceLinkProps): React.JSX.Element {
  return (
    <a
      className="reference-link"
      href={`${REFERENCE_BASE}${to.replace(/^\//, "")}`}
      rel="noopener noreferrer"
      target="_blank"
    >
      {children}
      <svg
        aria-hidden="true"
        className="reference-link__icon"
        focusable="false"
        height="1em"
        viewBox="0 0 16 16"
        width="1em"
      >
        <path
          d="M6 3h7v7M13 3 6.5 9.5M11 10.5V13H3V5h2.5"
          fill="none"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.5"
        />
      </svg>
      <span className="visually-hidden"> (opens in a new tab)</span>
    </a>
  );
}
