# ADR 0032: Two directories for portal figures, one committed and one derived

**Status:** Accepted

**Extends:** [ADR 0019](0019-react-learning-portal.md)

## Context

A walkthrough shows two kinds of image: a plot a study produced, and an illustration written for
the article. Until now the portal had one directory for both, `website/static/walkthrough-figures/`,
and that directory is derived: `website/scripts/generate_walkthroughs.py` refills it on every build
by copying each entry of its `FIGURES` map out of `docs/examples/assets/`, and `.gitignore` excludes
it so a figure is not committed twice.

Nothing said so, and nothing checked it. On 5 September 2026 three hand-made SVGs were placed in
that directory and referenced from `walkthroughs/michelson.mdx`. They rendered on the machine that
made them and would have 404'd everywhere else. Every gate stayed green:

- `onBrokenLinks: "throw"` resolves route links, not `<img src>` against the static tree;
- `walkthroughs/` is outside the `tsconfig.json` `include`, so MDX is never type-checked;
- no test read an MDX file looking for asset references.

The failure is silent in the worst way — correct locally, broken in production, with a passing
build in between. The general problem is that one directory carried two contradictory rules and the
contradiction was invisible at the point of use.

## Decision

**Two directories, opposite rules, and a test that knows which is which.**

| Directory | Committed? | Filled by | Holds |
| --- | --- | --- | --- |
| `website/static/figures/` | yes | a person, or a script writing there deliberately | images an article references directly |
| `website/static/walkthrough-figures/` | no, gitignored | `generate_walkthroughs.py` from `docs/examples/assets/` | a study's own committed figure, copied to where Docusaurus can serve it |

Both are referenced identically, through `siteUrl()` and the `Figure` component; the split is about
provenance and lifetime, not about how a page reaches an image.

**A figure that can be derived is still derived.** The three Michelson score panels are written by
`make_score_figures()` in `examples/michelson_phase.py` from the same closed-form `s_phi` and
`s_eps` the study already computes. They go straight to `website/static/figures/` and are committed
there. Regenerable and committed is not a contradiction: the same is already true of every PNG under
`docs/examples/assets/`.

**`website/tests/figures.test.ts` is the gate.** It reads every `siteUrl("...")` literal out of the
MDX under `walkthroughs/`, `get-started/` and `research/`, and resolves each against the lane that
owns it — a `figures/` path against the committed tree on disk, a `walkthrough-figures/` path
against the `FIGURES` map parsed out of the generator, so the check works on a clean checkout
without having run `pnpm generate`. It also fails on any stray file sitting in the derived directory.

**`Figure` gained a `paper` prop.** A rendered plot carries its own light ground baked into the
file and cannot follow the theme the way the hand-drawn SVG charts do. `paper` mats it on a card
painted `--figure-paper`, the one token deliberately absent from the dark block in `tokens.css`, so
under the dark theme the plot reads as a deliberate print rather than a bare white slab.

## Consequences

A hand-added image now has an obvious home, and putting it in the wrong one is a red test rather
than a production 404. The cost is a second directory to explain, which `website/static/figures/README.md`
and the portal README carry.

Rejected: importing SVGs as modules (`import x from "./x.svg"`), which `@docusaurus/plugin-svgr` makes
a build error on a missing file and would have closed the same hole. It inlines each 35 KB file into
the page bundle, against the LCP budget the portal's hand-drawn charts already exist to protect, and
it would have made the walkthroughs the only place in the tree that reaches an asset by import.
