/**
 * Route `data` props built by hand for the component tests.
 *
 * The plugin renders every Markdown field through `markdown.mjs` at build
 * time; a test only needs *some* HTML in those fields, so `html` wraps the
 * source text in a paragraph and escapes nothing else. Everything that is not
 * rendered — ids, coordinates, labels, years — comes straight from the real
 * generated atlas, so a test that asserts a number is asserting the shipped
 * number.
 */
import {atlas, core} from "../atlasFixtures";
import type {LiteratureData, RenderedTradition} from "../../src/atlas/pages/LiteraturePage";
import type {FixtureData} from "../../src/atlas/pages/FixturePage";
import type {RenderedPaper} from "../../src/atlas/pages/PaperPage";

export {atlas, core};

export function html(text: string | null): string | null {
  if (text === null || text === "") return null;
  // Backticked symbol names become `<code>` in the real renderer, and several
  // fields the pages show are nothing but a symbol name, so the stub keeps at
  // least that much of Markdown.
  return `<p>${text.replace(/`([^`]+)`/g, "<code>$1</code>")}</p>`;
}

export function renderedPaper(key: string): RenderedPaper {
  const paper = atlas.papers[key];
  if (!paper) throw new Error(`no paper ${key}`);
  return {
    ...paper,
    noteHtml: html(paper.note),
    annotation: paper.annotation
      ? {
          ...paper.annotation,
          fieldsHtml: Object.fromEntries(
            Object.entries(paper.annotation.fields).map(([name, value]) => [name, html(value) ?? ""])
          ),
          proseHtml: html(paper.annotation.prose)
        }
      : null
  };
}

export function renderedTradition(slug: string): RenderedTradition {
  const tradition = atlas.traditions.find((entry) => entry.slug === slug);
  if (!tradition) throw new Error(`no tradition ${slug}`);
  return {
    ...tradition,
    notes: tradition.notes.map((note) => ({
      ...note,
      fieldsHtml: Object.fromEntries(Object.entries(note.fields).map(([name, value]) => [name, html(value) ?? ""])),
      proseHtml: html(note.prose)
    }))
  };
}

export function literatureData(): LiteratureData {
  return {
    traditions: atlas.traditions.map((tradition) => renderedTradition(tradition.slug)),
    papers: Object.fromEntries(Object.keys(atlas.papers).map((key) => [key, renderedPaper(key)])),
    priorArt: atlas.headline
      .filter((id) => (atlas.claims[id]?.priorArt.length ?? 0) > 0)
      .map((id) => ({
        id,
        sources: (atlas.claims[id]?.priorArt ?? []).flatMap((audit) =>
          audit.sources.map((source) => ({
            name: source.name.replace(/--/g, "–"),
            html: html(source.text.replace(/[\s,]+(and|plus|with)\s*$/i, ""))
          }))
        )
      }))
  };
}

export function fixtureData(id: string): FixtureData {
  const fixture = atlas.fixtures[id];
  if (!fixture) throw new Error(`no fixture ${id}`);
  return {
    ...fixture,
    falsifiesHtml: html(fixture.falsifies),
    noteHtml: html(fixture.note),
    consequencesHtml: fixture.consequences.map((text) => html(text)),
    verificationNotesHtml: html(fixture.verification.notes),
    refusalHtml: fixture.refusal
      ? {
          triggerHtml: html(fixture.refusal.trigger),
          reasonHtml: html(fixture.refusal.reason),
          remedyHtml: html(fixture.refusal.remedy)
        }
      : null
  };
}
