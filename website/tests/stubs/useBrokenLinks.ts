/** `@docusaurus/useBrokenLinks` for vitest: collection is a build-time concern. */
export default function useBrokenLinks(): {collectAnchor: (id: string) => void; collectLink: (href: string) => void} {
  return {collectAnchor: () => undefined, collectLink: () => undefined};
}
