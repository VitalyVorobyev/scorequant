/**
 * Markdown from the research registry, rendered to HTML at build time.
 *
 * Proof prose, statements and editorial notes are Markdown with TeX math
 * (`$…$`, `$$…$$` after the generator normalised the registry's `\(…\)`
 * delimiters). They are rendered once here, in the plugin, so that no page
 * ships a Markdown parser to the browser and KaTeX runs at build time exactly
 * as it does for the walkthroughs.
 *
 * Two rewrites make the prose navigable: every claim or fixture id, whether
 * bare or in backticks, becomes a link to its atlas page; every workspace path
 * in backticks (`AUDITS/…`, `formal/…`, `py/…`) becomes a link to the file on
 * GitHub. Math and fenced code are left untouched because the walk only visits
 * text and inline-code nodes.
 */
import rehypeKatex from "rehype-katex";
import rehypeStringify from "rehype-stringify";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import {unified} from "unified";
import {visit} from "unist-util-visit";

const GITHUB = "https://github.com/VitalyVorobyev/scorequant/blob/main/agenticresearch/";
const ID_PATTERN = /\b([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)\b/g;
const PATH_PATTERN = /^(?:AUDITS|COUNTEREXAMPLES|KNOWN_RESULTS|LITERATURE|formal|py|artifacts|manuscripts|tests)\/[\w./-]+$/;

/**
 * @param {Map<string, string>} hrefs id -> absolute href
 */
function linkIds(hrefs) {
  return () => (tree) => {
    visit(tree, ["text", "inlineCode"], (node, index, parent) => {
      if (!parent || index === undefined || parent.type === "link") return;
      const value = node.value;
      if (node.type === "inlineCode") {
        const href = hrefs.get(value);
        if (href) {
          parent.children[index] = {type: "link", url: href, children: [{type: "inlineCode", value}]};
          return;
        }
        if (PATH_PATTERN.test(value)) {
          // A fixture or claim named by its file resolves to its page, not to GitHub.
          const stem = value.split("/").pop()?.replace(/\.(json|md)$/, "") ?? "";
          const entity = hrefs.get(stem);
          if (entity) {
            parent.children[index] = {type: "link", url: entity, children: [{type: "inlineCode", value}]};
            return;
          }
          const path = value.startsWith("artifacts/") ? `WORK/${value}` : value;
          parent.children[index] = {
            type: "link",
            url: GITHUB + path,
            children: [{type: "inlineCode", value}]
          };
        }
        return;
      }
      const pieces = [];
      let last = 0;
      for (const match of value.matchAll(ID_PATTERN)) {
        const href = hrefs.get(match[1]);
        if (!href) continue;
        if (match.index > last) pieces.push({type: "text", value: value.slice(last, match.index)});
        pieces.push({type: "link", url: href, children: [{type: "inlineCode", value: match[1]}]});
        last = match.index + match[1].length;
      }
      if (pieces.length === 0) return;
      if (last < value.length) pieces.push({type: "text", value: value.slice(last)});
      parent.children.splice(index, 1, ...pieces);
      return index + pieces.length;
    });
  };
}

/**
 * Build a renderer bound to the atlas's id -> href map.
 *
 * @param {Map<string, string>} hrefs
 * @returns {(markdown: string | null | undefined) => string | null}
 */
export function createRenderer(hrefs) {
  const processor = unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(linkIds(hrefs))
    .use(remarkRehype)
    .use(rehypeKatex, {throwOnError: false, strict: false})
    .use(rehypeStringify);
  return (markdown) => {
    if (markdown === null || markdown === undefined || markdown === "") return null;
    return String(processor.processSync(markdown));
  };
}
