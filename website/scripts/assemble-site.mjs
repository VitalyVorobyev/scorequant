import {cp, mkdir, readdir, readFile, rm, stat} from "node:fs/promises";
import {dirname, resolve} from "node:path";
import {fileURLToPath} from "node:url";

const website = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const project = resolve(website, "..");
const portalBuild = resolve(website, "build");
const referenceSite = resolve(project, "site");
const assembled = resolve(project, ".pages-preview");
const referenceMount = "docs";

// Two surfaces, one tree: the Docusaurus portal owns the site root
// and the MkDocs documentation is mounted beneath it at /docs/. Both are build
// outputs -- `docusaurus build` (whose `baseUrl` must match the site root) and
// `mkdocs build --strict`. The portal's own home page is the front door.
await rm(assembled, {recursive: true, force: true});
await mkdir(assembled, {recursive: true});
await cp(portalBuild, assembled, {recursive: true});

/**
 * The portal must not itself emit a `docs/` route.
 *
 * The MkDocs tree is copied *into* the portal's tree now rather than beside
 * it, so a portal route named `docs` would be silently overwritten by the
 * reference and the portal page would vanish from the published site with
 * every gate still green. Nothing emits one today -- the portal's routes are
 * `get-started` and `walkthroughs` -- and this is what keeps that true.
 */
if (await directoryExists(resolve(portalBuild, referenceMount))) {
  process.stderr.write(
    `assemble:site: the portal build emits a "${referenceMount}/" route, which is where the\n` +
      `  MkDocs documentation is mounted. Rename that route, or move the\n` +
      `  reference mount, before the copy silently replaces one with the other.\n`,
  );
  process.exit(1);
}

await cp(referenceSite, resolve(assembled, referenceMount), {recursive: true});

const failures = [];

/**
 * Every portal link into the reference must resolve.
 *
 * This is the one class of link nothing else can check. Docusaurus's
 * `onBrokenLinks: "throw"` follows route links inside the portal, and
 * `website/tests/reference-links.test.ts` requires a link into the reference
 * to go through the `ReferenceLink` component -- but neither knows whether the
 * MkDocs page on the other end exists, because neither has ever seen the
 * assembled tree. Here it exists, so every `/scorequant/docs/...` href the
 * built portal carries is resolved against it.
 */
const referencePrefix = `/scorequant/${referenceMount}/`;
const portalPages = await portalHtmlFiles(assembled);
const referenceHrefs = new Map();
for (const page of portalPages) {
  const html = await readFile(page, "utf8");
  for (const match of html.matchAll(/href="([^"#?]+)"/g)) {
    const href = match[1];
    if (!href.startsWith(referencePrefix)) continue;
    if (!referenceHrefs.has(href)) referenceHrefs.set(href, page);
  }
}
for (const [href, page] of referenceHrefs) {
  const relative = href.slice("/scorequant/".length);
  const target = href.endsWith("/")
    ? resolve(assembled, relative, "index.html")
    : resolve(assembled, relative);
  if (!(await fileExists(target))) {
    failures.push(
      `reference link "${href}" in ${page.slice(assembled.length + 1)} does not resolve: expected ${target}`,
    );
  }
}

if (failures.length > 0) {
  process.stderr.write("assemble:site: reference-link check failed:\n");
  for (const failure of failures) process.stderr.write(`  - ${failure}\n`);
  process.exit(1);
}

process.stdout.write(
  `Assembled site at ${assembled} (portal at the root, MkDocs under ${referenceMount}/, ` +
    `${referenceHrefs.size} reference links verified).\n`,
);

/**
 * Every `index.html` the portal build emitted, skipping the reference mount.
 *
 * The MkDocs tree lives under the same root now, and it links within itself
 * with its own relative URLs; only the portal's links into `/docs/` are this
 * check's business.
 */
async function portalHtmlFiles(root) {
  const found = [];
  for (const entry of await readdir(root, {withFileTypes: true})) {
    if (entry.name === referenceMount) continue;
    const path = resolve(root, entry.name);
    if (entry.isDirectory()) {
      found.push(...(await portalHtmlFiles(path)));
    } else if (entry.isFile() && entry.name.endsWith(".html")) {
      found.push(path);
    }
  }
  return found;
}

async function fileExists(path) {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
}

async function directoryExists(path) {
  try {
    return (await stat(path)).isDirectory();
  } catch {
    return false;
  }
}
