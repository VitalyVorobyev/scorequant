import {cp, mkdir, readdir, readFile, rm, stat, writeFile} from "node:fs/promises";
import {dirname, resolve} from "node:path";
import {fileURLToPath} from "node:url";

const website = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const project = resolve(website, "..");
const portalBuild = resolve(website, "build");
const referenceSite = resolve(project, "site");
const assembled = resolve(project, ".pages-preview");
const redirectsManifestPath = resolve(website, "redirects.json");
const referenceMount = "docs";

// Two surfaces, one tree (ADR 0035): the Docusaurus portal owns the site root
// and the MkDocs documentation is mounted beneath it at /docs/. Both are build
// outputs -- `docusaurus build` (whose `baseUrl` must match the site root) and
// `mkdocs build --strict`. The hand-written landing page ADR 0027 put at the
// root is gone; the portal's own home page is the front door.
await rm(assembled, {recursive: true, force: true});
await mkdir(assembled, {recursive: true});
await cp(portalBuild, assembled, {recursive: true});

/**
 * The portal must not itself emit a `docs/` route (ADR 0035).
 *
 * The MkDocs tree is copied *into* the portal's tree now rather than beside
 * it, so a portal route named `docs` would be silently overwritten by the
 * reference and the portal page would vanish from the published site with
 * every gate still green. Nothing emits one today -- the portal's routes are
 * `get-started`, `walkthroughs` and `research` -- and this is what keeps that
 * true. Same discipline as the redirect-stub collision check below.
 */
if (await directoryExists(resolve(portalBuild, referenceMount))) {
  process.stderr.write(
    `assemble:site: the portal build emits a "${referenceMount}/" route, which is where the\n` +
      `  MkDocs documentation is mounted (ADR 0035). Rename that route, or move the\n` +
      `  reference mount, before the copy silently replaces one with the other.\n`,
  );
  process.exit(1);
}

await cp(referenceSite, resolve(assembled, referenceMount), {recursive: true});

/**
 * The redirect stub template (spec T3).
 *
 * `noindex` keeps the stub out of search results so the canonical target is
 * what gets indexed. The visible link is for a reader whose browser blocks
 * the refresh.
 */
function stubHtml(toAbsolute, toFull) {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Moved — ScoreQuant</title>
    <link rel="canonical" href="${toFull}" />
    <meta http-equiv="refresh" content="0; url=${toAbsolute}" />
    <meta name="robots" content="noindex" />
  </head>
  <body>
    <p>This page moved to <a href="${toAbsolute}">${toAbsolute}</a>.</p>
  </body>
</html>
`;
}

async function fileExists(path) {
  try {
    const info = await stat(path);
    return info.isFile();
  } catch {
    return false;
  }
}

async function directoryExists(path) {
  try {
    const info = await stat(path);
    return info.isDirectory();
  } catch {
    return false;
  }
}

const manifest = JSON.parse(await readFile(redirectsManifestPath, "utf8"));

/**
 * A stub must never overwrite real content (ADR 0025, kept by ADR 0035).
 *
 * The portal now owns the root, so a collision means a portal route, the
 * reference mount, or a manifest entry that names one of them. No portal route
 * shares a name with a stub today, and this check is what turns a future one
 * into a loud build failure instead of a silently replaced page.
 */
for (const {from, to} of manifest.redirects) {
  const toAbsolute = `/scorequant/${to}`;
  const toFull = `https://vitalyvorobyev.github.io/scorequant/${to}`;
  const stubPath = resolve(assembled, from, "index.html");
  if (await fileExists(stubPath)) {
    process.stderr.write(
      `assemble:site: refusing to write a redirect stub over real content.\n` +
        `  "${from}" already exists in the assembled tree at ${stubPath}.\n` +
        `  Either drop "${from}" from website/redirects.json (recording it under\n` +
        `  "unstubbed" with the reason), or rename the route it collides with.\n`,
    );
    process.exit(1);
  }
  await mkdir(dirname(stubPath), {recursive: true});
  await writeFile(stubPath, stubHtml(toAbsolute, toFull));
}

/**
 * Self-verification (spec item 4).
 *
 * `website/tests/redirects.test.ts` cannot check that every stub exists and
 * every stub's target resolves to a real page, because Vitest never builds
 * the assembled tree. This is that same parity check, run here instead,
 * right after the tree it needs exists. CI gets the parity evidence from
 * `pnpm assemble:site` failing loudly; the Vitest suite only checks the
 * manifest's own shape.
 */
const failures = [];

for (const {from} of manifest.redirects) {
  const stubPath = resolve(assembled, from, "index.html");
  if (!(await fileExists(stubPath))) {
    failures.push(`missing stub for "${from}": expected ${stubPath}`);
  }
}

for (const {from, to} of manifest.redirects) {
  const targetPath = resolve(assembled, to, "index.html");
  if (!(await fileExists(targetPath))) {
    failures.push(`redirect "${from}" -> "${to}" does not resolve: expected ${targetPath}`);
  }
}

/**
 * Every portal link into the reference must resolve (ADR 0035).
 *
 * This is the one class of link nothing else can check. Docusaurus's
 * `onBrokenLinks: "throw"` follows route links inside the portal, and
 * `website/tests/reference-links.test.ts` requires a link into the reference
 * to go through the `ReferenceLink` component -- but neither knows whether the
 * MkDocs page on the other end exists, because neither has ever seen the
 * assembled tree. Here it exists, so every `/scorequant/docs/...` href the
 * built portal carries is resolved against it. It replaces the landing-link
 * check ADR 0027 introduced, which had the same purpose and lost its subject
 * when the landing page did.
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
  process.stderr.write("assemble:site: redirect and reference-link parity check failed:\n");
  for (const failure of failures) process.stderr.write(`  - ${failure}\n`);
  process.exit(1);
}

process.stdout.write(
  `Assembled site at ${assembled} (portal at the root, MkDocs under ${referenceMount}/, ` +
    `${manifest.redirects.length} redirect stubs and ${referenceHrefs.size} reference links verified).\n`,
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
