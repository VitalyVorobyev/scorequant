import {readFileSync} from "node:fs";
import {dirname, resolve} from "node:path";
import {fileURLToPath} from "node:url";

import {describe, expect, it} from "vitest";

/**
 * The parity evidence ADR 0019 requires (spec T2/T3, done criteria).
 *
 * This checks only the manifest's own shape — that it is internally
 * consistent and covers the pre-cut sitemap count. It cannot check that
 * every stub actually exists in the assembled tree, or that every stub's
 * target resolves to a real page there, because Vitest never builds that
 * tree. `website/scripts/assemble-site.mjs` runs that half of the parity
 * check itself, right after it writes the tree, and fails the build loudly
 * if a stub is missing or a target does not resolve.
 */

interface RedirectEntry {
  from: string;
  to: string;
}

interface UnstubbedEntry {
  path: string;
  // Optional in the type, required in the file: this is parsed from disk, so the
  // type is an assertion about JSON that nothing has checked yet. The last test in
  // this file is what actually enforces that the field is present and substantive.
  reason?: string;
}

interface RedirectsManifest {
  redirects: RedirectEntry[];
  schemaVersion: number;
  sourceSitemapCount: number;
  unstubbed: UnstubbedEntry[];
}

const websiteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifest = JSON.parse(readFileSync(resolve(websiteRoot, "redirects.json"), "utf8")) as RedirectsManifest;

describe("website/redirects.json", () => {
  it("covers exactly the pre-cut sitemap count", () => {
    expect(manifest.sourceSitemapCount).toBe(manifest.unstubbed.length + manifest.redirects.length);
  });

  it("has no duplicate `from`, and no `from` collides with an unstubbed path", () => {
    const froms = manifest.redirects.map((entry) => entry.from);
    expect(new Set(froms).size).toBe(froms.length);

    const unstubbedPaths = new Set(manifest.unstubbed.map((entry) => entry.path));
    const collisions = froms.filter((from) => unstubbedPaths.has(from));
    expect(collisions).toEqual([]);
  });

  it("every `from` ends with a trailing slash", () => {
    const offenders = manifest.redirects.filter((entry) => !entry.from.endsWith("/"));
    expect(offenders).toEqual([]);
  });

  it("every `to` ends with a trailing slash", () => {
    const offenders = manifest.redirects.filter((entry) => !entry.to.endsWith("/"));
    expect(offenders).toEqual([]);
  });

  it("every unstubbed path is either the empty string or ends with a trailing slash", () => {
    const offenders = manifest.unstubbed.filter((entry) => entry.path !== "" && !entry.path.endsWith("/"));
    expect(offenders).toEqual([]);
  });

  // Every pre-cut stub points into the MkDocs documentation at docs/, because
  // that is where every pre-cut page now lives (ADR 0027 restored the three
  // narrative pages S8 and S10 had retired into portal routes). The one other
  // family is the retired research essays (ADR 0033), whose stubs point at the
  // atlas view under research/ that absorbed each of them -- the portal owns the
  // site root since ADR 0035, so that family lost its portal/ prefix with the
  // move. A `to` outside those two roots is a mistake rather than an exception.
  it("every `to` starts with docs/ or, for a retired research essay, research/", () => {
    const offenders = manifest.redirects.filter((entry) => {
      if (entry.from.startsWith("research/")) {
        return !entry.to.startsWith("research/") && !entry.to.startsWith("docs/");
      }
      return !entry.to.startsWith("docs/");
    });
    expect(offenders).toEqual([]);
  });

  // An unstubbed entry is a URL that silently stops redirecting, so the reason it
  // is safe has to travel with it. One is excluded today: the site root, which
  // the portal's home page occupies (ADR 0035). A one-word reason would defeat
  // the point.
  it("every unstubbed path carries a substantive reason", () => {
    const offenders = manifest.unstubbed.filter((entry) => (entry.reason ?? "").trim().length < 40);
    expect(offenders).toEqual([]);
  });
});
