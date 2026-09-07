import useBrokenLinks from "@docusaurus/useBrokenLinks";

/**
 * Register in-page anchors with Docusaurus's broken-link checker.
 *
 * The checker knows an anchor only when a component reports it during the
 * server render; the theme's `Heading` does that for MDX, but the atlas pages
 * write their own `<h2 id>` and `<section id>` elements, so without this call
 * every `#chapter` or `#theme` link from another page fails the build as a
 * broken anchor even though the id is in the HTML.
 */
export function useRegisteredAnchors(ids: readonly string[]): void {
  const links = useBrokenLinks();
  for (const id of ids) links.collectAnchor(id);
}
