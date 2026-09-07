/**
 * The Research Atlas as Docusaurus routes.
 *
 * Reads `src/generated/atlas.json` once and registers one static route per
 * view and per entity: claims, counterexample fixtures, papers and authors.
 * Every page therefore has its own `index.html`, is indexed by Pagefind, and
 * is checked by `onBrokenLinks`. The pages under `src/atlas/pages/` receive
 * two data modules: `core`, the graph skeleton shared by every route, and a
 * route-specific module carrying the rendered HTML the page needs.
 *
 * Markdown is rendered here (see `markdown.mjs`) so the browser bundle carries
 * no parser and KaTeX runs at build time.
 */
import {readFile} from "node:fs/promises";
import {dirname, resolve} from "node:path";
import {fileURLToPath} from "node:url";

import {buildCore, coreClaim} from "./core.mjs";
import {createRenderer} from "./markdown.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const website = resolve(here, "..", "..");
const atlasPath = resolve(website, "src", "generated", "atlas.json");
const pages = (name) => `@site/src/atlas/pages/${name}.tsx`;

/** Trailing-slash route under the site base. */
function route(baseUrl, ...segments) {
  const path = [baseUrl.replace(/\/$/, ""), "research", ...segments].filter(Boolean).join("/");
  return `${path}/`;
}

export default function researchAtlasPlugin(context) {
  const {baseUrl} = context.siteConfig;
  return {
    name: "research-atlas",

    getPathsToWatch() {
      return [atlasPath];
    },

    async loadContent() {
      const atlas = JSON.parse(await readFile(atlasPath, "utf8"));
      const hrefs = new Map();
      for (const claim of Object.values(atlas.claims)) hrefs.set(claim.id, route(baseUrl, "claims", claim.slug));
      for (const fixture of Object.values(atlas.fixtures)) hrefs.set(fixture.id, route(baseUrl, "counterexamples", fixture.slug));
      const render = createRenderer(hrefs);
      return {atlas, render, hrefs};
    },

    async contentLoaded({content, actions}) {
      const {atlas, render} = content;
      const {createData, addRoute} = actions;

      const core = buildCore(atlas);
      const corePath = await createData("atlas-core.json", JSON.stringify(core));

      const add = async (segments, component, name, data) => {
        const dataPath = await createData(`${name}.json`, JSON.stringify(data));
        addRoute({
          path: route(baseUrl, ...segments),
          component: pages(component),
          exact: true,
          modules: {core: corePath, data: dataPath}
        });
      };

      const renderedClaim = (claim) => ({
        ...claim,
        statementHtml: render(claim.statement),
        assumptionsHtml: claim.assumptions.map((a) => render(a)),
        noteHtml: render(claim.note),
        scopeHtml: render(claim.scope),
        roleHtml: render(claim.role),
        proofHtml: claim.proof ? render(claim.proof.markdown) : null,
        priorArt: claim.priorArt.map((audit) => ({
          ...audit,
          sources: audit.sources.map((s) => ({name: s.name, html: render(s.text)}))
        }))
      });
      // A fixture's refusal trigger, reason and remedy are Markdown with
      // backticked symbol names, exactly like the library's own refusal table,
      // so they are rendered here rather than printed with their backticks.
      const renderedFixture = (fixture) => ({
        ...fixture,
        falsifiesHtml: render(fixture.falsifies),
        noteHtml: render(fixture.note),
        consequencesHtml: fixture.consequences.map((c) => render(c)),
        verificationNotesHtml: render(fixture.verification.notes),
        refusalHtml: fixture.refusal
          ? {
              triggerHtml: render(fixture.refusal.trigger),
              reasonHtml: render(fixture.refusal.reason),
              remedyHtml: render(fixture.refusal.remedy)
            }
          : null
      });
      const renderedPaper = (paper) => ({
        ...paper,
        noteHtml: render(paper.note),
        annotation: paper.annotation
          ? {
              ...paper.annotation,
              fieldsHtml: Object.fromEntries(Object.entries(paper.annotation.fields).map(([k, v]) => [k, render(v)])),
              proseHtml: render(paper.annotation.prose)
            }
          : null
      });

      // Views.
      const card = (id) => {
        const claim = atlas.claims[id];
        return {...coreClaim(claim), noteHtml: render(claim.note), scopeHtml: render(claim.scope)};
      };
      await add([], "AtlasHome", "atlas-home", {
        headline: atlas.headline.map(card),
        frontier: atlas.frontier.map(card)
      });
      await add(["map"], "MapPage", "atlas-map", {});
      await add(["landscape"], "LandscapePage", "atlas-landscape", {});
      await add(["frontier"], "FrontierPage", "atlas-frontier", {
        themes: atlas.themes.map((theme) => ({
          ...theme,
          claims: theme.claims.map((id) => ({
            ...coreClaim(atlas.claims[id]),
            statementHtml: render(atlas.claims[id].statement),
            noteHtml: render(atlas.claims[id].note)
          }))
        }))
      });
      await add(["literature"], "LiteraturePage", "atlas-literature", {
        traditions: atlas.traditions.map((t) => ({
          ...t,
          notes: t.notes.map((n) => ({
            ...n,
            fieldsHtml: Object.fromEntries(Object.entries(n.fields).map(([k, v]) => [k, render(v)])),
            proseHtml: render(n.prose)
          }))
        })),
        papers: Object.fromEntries(Object.values(atlas.papers).map((p) => [p.key, renderedPaper(p)])),
        // The nearest published sources a headline result's prior-art audit
        // named. A claim whose audit has not been run contributes nothing
        // rather than an empty entry.
        //
        // The registry stores each source as a name plus the run of the audit
        // sentence that followed it, so a source's text keeps the conjunction
        // that led to the next name; a list is not a sentence, and the
        // conjunction is dropped. Names carry the LaTeX double hyphen.
        priorArt: atlas.headline
          .filter((id) => atlas.claims[id].priorArt.length > 0)
          .map((id) => ({
            id,
            sources: atlas.claims[id].priorArt.flatMap((audit) =>
              audit.sources.map((s) => ({name: s.name, html: render(s.text)}))
            )
          }))
      });
      // The Lean chain, plus the full machine-checked record of every claim it
      // certifies: the declaration, the two files and the statement audit live
      // on the claim rather than on the module.
      await add(["machine-checked"], "FormalPage", "atlas-formal", {
        ...atlas.formal,
        certified: Object.values(atlas.claims)
          .filter((claim) => claim.machineChecked !== null)
          .map((claim) => ({id: claim.id, machineChecked: claim.machineChecked}))
      });
      // Library roles, refusal triggers, reasons and remedies are Markdown with
      // backticked symbol names, so they are rendered here like every other
      // prose field rather than printed with their backticks showing.
      await add(["library"], "LibraryPage", "atlas-library", {
        objects: atlas.library.objects.map((object) => ({...object, roleHtml: render(object.role)})),
        refusals: atlas.library.refusals.map((refusal) => ({
          ...refusal,
          reasonHtml: render(refusal.reason),
          remedyHtml: render(refusal.remedy),
          triggerHtml: render(refusal.trigger)
        }))
      });
      await add(["how-to-read"], "HowToReadPage", "atlas-how-to-read", {});
      await add(["claims"], "ClaimsIndexPage", "atlas-claims-index", {});
      // Row counts are the one fixture fact the shared core does not carry.
      await add(["counterexamples"], "FixturesIndexPage", "atlas-fixtures-index", {
        sampleSize: Object.fromEntries(Object.values(atlas.fixtures).map((fixture) => [fixture.id, fixture.weights.length]))
      });

      // Entities.
      for (const claim of Object.values(atlas.claims)) {
        await add(["claims", claim.slug], "ClaimPage", `claim-${claim.slug}`, renderedClaim(claim));
      }
      for (const fixture of Object.values(atlas.fixtures)) {
        await add(["counterexamples", fixture.slug], "FixturePage", `fixture-${fixture.slug}`, renderedFixture(fixture));
      }
      for (const paper of Object.values(atlas.papers)) {
        await add(["literature", paper.slug], "PaperPage", `paper-${paper.slug}`, renderedPaper(paper));
      }
      for (const author of Object.values(atlas.authors)) {
        await add(["authors", author.slug], "AuthorPage", `author-${author.slug}`, author);
      }
    }
  };
}
