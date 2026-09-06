import {existsSync, readFileSync, readdirSync} from "node:fs";
import {dirname, join, resolve} from "node:path";
import {fileURLToPath} from "node:url";

import {describe, expect, it} from "vitest";

// Every image an MDX page shows is addressed by a hand-written string inside
// `siteUrl(...)`. Nothing else checks those strings: `onBrokenLinks: "throw"`
// only follows route links, not `<img src>`, and the MDX directories sit
// outside the tsconfig `include`, so a wrong path builds green and 404s in
// production. Worse, `static/walkthrough-figures/` is gitignored and refilled
// from `docs/examples/assets/` by `scripts/generate_walkthroughs.py`, so a file
// merely *placed* there resolves on the machine that placed it and nowhere
// else. This resolves each reference against the lane that actually owns it.
const websiteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const contentDirs = ["walkthroughs", "get-started", "research"];
const staticRoot = join(websiteRoot, "static");
const generatorPath = join(websiteRoot, "scripts/generate_walkthroughs.py");

/** The `walkthrough-figures/` names `generate_walkthroughs.py` writes, read from its `FIGURES` map. */
function generatedFigureNames(): string[] {
  const source = readFileSync(generatorPath, "utf8");
  const block = /FIGURES\s*=\s*\{([\s\S]*?)\}/.exec(source)?.[1];
  if (block === undefined) throw new Error("no FIGURES map in scripts/generate_walkthroughs.py");
  const names: string[] = [];
  for (const match of block.matchAll(/"([^"]+)":/g)) {
    const [, name] = match;
    if (name !== undefined) names.push(name);
  }
  return names;
}

/** Every `siteUrl("...")` argument in the MDX under `contentDirs`, tagged with its file. */
function siteUrlReferences(): {file: string; path: string}[] {
  const found: {file: string; path: string}[] = [];
  for (const dir of contentDirs) {
    const root = join(websiteRoot, dir);
    if (!existsSync(root)) continue;
    for (const entry of readdirSync(root)) {
      if (!entry.endsWith(".mdx") && !entry.endsWith(".md")) continue;
      const source = readFileSync(join(root, entry), "utf8");
      for (const match of source.matchAll(/siteUrl\("([^"]+)"\)/g)) {
        const [, path] = match;
        if (path !== undefined) found.push({file: `${dir}/${entry}`, path});
      }
    }
  }
  return found;
}

describe("every figure an MDX page references resolves to a real file", () => {
  const references = siteUrlReferences();

  it("finds references to check", () => {
    expect(references.length).toBeGreaterThan(0);
  });

  it("resolves committed assets on disk under website/static", () => {
    const missing = references
      .filter(({path}) => !path.startsWith("walkthrough-figures/"))
      .filter(({path}) => !existsSync(join(staticRoot, path)))
      .map(({file, path}) => `${file} -> static/${path}`);
    expect(missing).toEqual([]);
  });

  it("resolves generated assets through the FIGURES map, not through the gitignored directory", () => {
    const generated = generatedFigureNames();
    const missing = references
      .filter(({path}) => path.startsWith("walkthrough-figures/"))
      .filter(({path}) => !generated.includes(path.slice("walkthrough-figures/".length)))
      .map(({file, path}) => `${file} -> ${path}`);
    expect(missing).toEqual([]);
  });
});

describe("the two figure directories keep their separate roles", () => {
  it("keeps static/figures committed and static/walkthrough-figures gitignored", () => {
    const gitignore = readFileSync(resolve(websiteRoot, "../.gitignore"), "utf8");
    expect(gitignore).toContain("website/static/walkthrough-figures/");
    expect(gitignore).not.toContain("website/static/figures/");
  });

  it("keeps hand-added files out of the generated directory", () => {
    const generated = new Set(generatedFigureNames());
    const strays = readdirSync(join(staticRoot, "walkthrough-figures"))
      .filter((name) => !generated.has(name))
      .filter((name) => name !== ".DS_Store");
    expect(strays).toEqual([]);
  });
});
