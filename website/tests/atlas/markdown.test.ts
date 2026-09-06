import {describe, expect, it} from "vitest";

import {createRenderer} from "../../plugins/research-atlas/markdown.mjs";
import {atlas} from "../atlasFixtures";

const hrefs = new Map<string, string>();
for (const claim of Object.values(atlas.claims)) hrefs.set(claim.id, `/research/claims/${claim.slug}/`);
for (const fixture of Object.values(atlas.fixtures)) hrefs.set(fixture.id, `/research/counterexamples/${fixture.slug}/`);
const render = createRenderer(hrefs);

describe("build-time markdown rendering", () => {
  it("renders the central theorem's proof with KaTeX and links every id and path", () => {
    const proof = atlas.claims["D-EXCHANGE-IMPLIES-VORONOI"]?.proof;
    if (!proof) throw new Error("no proof");
    const html = render(proof.markdown);
    expect(html).not.toBeNull();
    expect(html).toContain('class="katex-display"');
    expect(html).toContain("katex");
    expect(html).toContain('href="/research/counterexamples/ce-d-unmerged-duplicates-001/"');
    expect(html).toMatch(/href="https:\/\/github\.com\/VitalyVorobyev\/scorequant\/blob\/main\/agenticresearch\/AUDITS\/AUDIT-D-EXCHANGE-VORONOI-001\.md"/);
    expect(html).not.toContain("\\(");
  });

  it("links bare ids in plain text and leaves math and unknown tokens alone", () => {
    const html = render("See D-LOGDET-GAIN and `CE-D-VORONOI-CONVERSE-001`; not NOT-A-CLAIM. $D-LOGDET-GAIN$ stays math.");
    expect(html).toContain('href="/research/claims/d-logdet-gain/"');
    expect(html).toContain('href="/research/counterexamples/ce-d-voronoi-converse-001/"');
    expect(html).toContain("NOT-A-CLAIM");
    expect(html?.match(/d-logdet-gain/g)?.length).toBe(1);
  });

  it("returns null for empty input", () => {
    expect(render(null)).toBeNull();
    expect(render("")).toBeNull();
  });
});
