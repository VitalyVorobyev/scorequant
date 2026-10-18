import {readdirSync, readFileSync, statSync} from "node:fs";
import {dirname, join, resolve} from "node:path";
import {fileURLToPath} from "node:url";

import {describe, expect, it} from "vitest";

// SITE_BASE in ../src/lib/site is the one place the portal's base path is
// allowed to live. This walks the whole `src` tree (never `node_modules` or
// `build`) and fails the moment the deployed prefix is written anywhere else.
// The prefix `/scorequant/` also occurs inside GitHub source URLs
// (`.../blob/main/src/scorequant/config.py`), so the search is for a *quoted
// literal* beginning with it, which is how a base path is actually written.
const websiteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const srcRoot = join(websiteRoot, "src");
const generatedRoot = join(srcRoot, "generated");

/** A string, template or JSX-attribute literal that opens with the deployed prefix. */
const QUOTED_BASE = /["'`]\/scorequant\b/;

function listFiles(dir: string): string[] {
  const entries = readdirSync(dir);
  const files: string[] = [];
  for (const entry of entries) {
    const path = join(dir, entry);
    // Generated data is written by the Python generators, not authored here,
    // and carries repository URLs rather than site paths.
    if (path === generatedRoot) continue;
    const stat = statSync(path);
    if (stat.isDirectory()) {
      files.push(...listFiles(path));
    } else if (stat.isFile()) {
      files.push(path);
    }
  }
  return files;
}

describe("SITE_BASE is the only place the portal base path lives", () => {
  it("finds the deployed prefix in src/lib/site.ts and nowhere else under website/src", () => {
    const offenders = listFiles(srcRoot)
      .filter((path) => QUOTED_BASE.test(readFileSync(path, "utf8")))
      .map((path) => path.slice(websiteRoot.length + 1));
    expect(offenders).toEqual(["src/lib/site.ts"]);
  });
});
