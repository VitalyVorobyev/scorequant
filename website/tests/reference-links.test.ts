import {readFileSync, readdirSync} from "node:fs";
import {join} from "node:path";

import {describe, expect, it} from "vitest";

/**
 * Every page that reaches the MkDocs reference must do it through
 * `ReferenceLink`, and that component must open a new tab.
 *
 * The rule exists because the two failures it catches are both silent. A raw
 * MDX anchor written `href="pathname:///reference/..."` is not routed through
 * Docusaurus's link handling, so the literal string reaches the browser and is
 * not a URL at all -- `onBrokenLinks` never sees it, because it resolves route
 * links rather than raw hrefs. A markdown `[x](pathname://...)` link *is*
 * rewritten, but drops the site's `baseUrl`, which is the other half of the
 * same hole. Both shipped on the research pages until September 2026 (the
 * research section is generated from the registry since ADR 0033 and has no
 * MDX of its own).
 *
 * The second half of the rule is a product decision (ADR 0027): the reference
 * is a separate site with its own shell, so following a link into it in place
 * strands the reader in an unfamiliar UI with no way back.
 */

const ROOT = join(__dirname, "..");
const CONTENT_DIRS = ["walkthroughs", "get-started"];
const COMPONENT = join(ROOT, "src/components/ReferenceLink.tsx");

/** Every `.md`/`.mdx` file under the portal's authored content directories. */
function contentFiles(): string[] {
  return CONTENT_DIRS.flatMap((dir) =>
    readdirSync(join(ROOT, dir), {recursive: true, withFileTypes: true})
      .filter((entry) => entry.isFile() && /\.mdx?$/.test(entry.name))
      .map((entry) => join(entry.parentPath, entry.name))
  );
}

describe("cross-site reference links", () => {
  it("never reaches the reference by a hand-written pathname:// URL", () => {
    const offenders = contentFiles().filter((file) => readFileSync(file, "utf8").includes("pathname://"));
    expect(offenders.map((file) => file.slice(ROOT.length + 1))).toEqual([]);
  });

  it("opens the reference in a new tab, safely, and says so", () => {
    const source = readFileSync(COMPONENT, "utf8");
    expect(source).toContain('target="_blank"');
    expect(source).toContain("noopener");
    expect(source).toContain("opens in a new tab");
  });
});
