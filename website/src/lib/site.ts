/**
 * The one base path constant for the whole portal.
 *
 * Nine source files used to hard-code the portal path directly: `useBaseUrl` cannot serve all of them
 * because `lab.worker.ts` is a Web Worker with no React context. So every
 * site-rooted literal collapses into this one module instead, including the
 * ones that do have React context — one source of truth beats two
 * mechanisms that happen to agree today.
 *
 * `SITE_BASE` matches `docusaurus.config.ts`'s `baseUrl` exactly and must be
 * changed in lockstep with it. The portal owns the site root as of
 * ADR 0033, which retired the hand-written landing page that used to.
 */
export const SITE_BASE = "/scorequant/";

/**
 * Where the assembled MkDocs documentation is mounted. It is a child of the
 * site root the portal now serves: the portal is at `/` and the documentation
 * beneath it at `/docs/` (ADR 0033).
 */
export const REFERENCE_BASE = "/scorequant/docs/";

/**
 * Join `SITE_BASE` with `path` without doubling the slash between them.
 *
 * `path` may or may not carry a leading slash; either way the result has
 * exactly one slash between the site base and the path.
 */
export function siteUrl(path: string): string {
  const base = SITE_BASE.endsWith("/") ? SITE_BASE.slice(0, -1) : SITE_BASE;
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${base}${suffix}`;
}
