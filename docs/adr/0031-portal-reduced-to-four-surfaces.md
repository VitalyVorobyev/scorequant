# ADR 0031: The portal is four surfaces, and its pages are articles

**Status:** Accepted

**Supersedes in part:** [ADR 0029](0029-lessons-replace-the-free-form-lab.md) (the lesson
index at `/portal/lab/`), [ADR 0019](0019-react-learning-portal.md) and
[ADR 0020](0020-development-blog.md) (the API catalogue, Benchmarks page and development
blog). **Keeps:** [ADR 0027](0027-landing-page-at-the-root.md) (topology) and
[ADR 0028](0028-focused-research-and-teaching.md) (teaching content is executable and states
its contract).

## Context

The owner reviewed the published portal on 5 September 2026, after PRs #50 and #51, and
rejected its shape. The home page was a sales page: a slogan in display type, a marketing
subhead over a comparison table, a live demo, and sections titled as invitations, none of which
defined the task the library solves, which is not common knowledge. The Lessons page was a list
of links to the walkthroughs with nothing of its own. The API catalogue and the Benchmarks page
duplicated or fell short of the MkDocs documentation. The blog was one post. The walkthroughs
index was an essay about the walkthroughs. The Michelson page, though it followed the seven-step
pattern of ADR 0028, read as a procedure: numbered steps, a contract table and a card grid of
numbers, with no account of what the instrument is, why the phase matters or what the data
looks like. Every teaching page opened with an italic *"Who this is for"* note, an internal
authoring remark leaked to the public surface.

## Decision

**The home page is definitions and references.** It defines, in order, the reference point and
the score, the Fisher information, hard binning and the identity that prices it, the two tasks
and what each returns, the criteria (the solver pairs stay in the method overview), and exact
versus estimated score provenance; then it lists where each definition is derived, in the book, the API guide,
the examples and the portal's own pages. It quotes no measurement and runs nothing. Ordinary
type sizes.

**The portal has four surfaces:** Get started, Walkthroughs, Research, and the Reference link
into the MkDocs documentation. The Lessons index, the API catalogue, the Benchmarks page and
the blog are removed rather than demoted. The search dialog keeps the public symbols as hits and
sends them to the generated reference.

**The walkthroughs index is a card per problem:** a title, one paragraph stating what is
measured and what is to be estimated, one line naming the data and what kind of evidence it is,
and then tags naming the task, input route, criterion and solver. The problem and the data come
before any API name; the task tag is always present.

**A walkthrough is an article.** The seven-step order of ADR 0028 remains the author's
checklist: a page must still state the problem, the model and its score, the admissible labels
and the criterion, the run, the evaluation, one experiment and the interpretation, in that
order. It does not number them or head them by their step names. It opens with the subject,
what the instrument or measurement is, why the question matters and what the data is, in the
register of a popular-science article, and states the contract inside that prose rather than in
a table. Numbers appear in sentences that give them meaning, not in card grids. Opening
"who this is for" notes are removed everywhere.

**Browser computation stays behind a page's explicit action**, as ADR 0028 and ADR 0029
require: the get-started refit and the Michelson budget explorer are unchanged.

## Consequences

- `website/src/pages/` holds the home page only. The blog preset is off. The score-space live
  figure and its portal data are deleted; the M13-A correspondence tests for it live in git
  history, and a compiled-rule drawing can return on a walkthrough that needs one.
- No redirect stubs for the removed portal routes: `website/redirects.json` records the pre-cut
  MkDocs sitemap, and the portal's own retired routes have never been stubbed.
- The root landing page's portal card lists Get started, the walkthroughs and the research
  record.
- Roadmap M13-C applies the article form to the ratios, FlowCyt and HEP pages.
- ADR 0030 remains reserved for the formal-verification pilot (roadmap M13-E).

## Alternatives considered

Keeping the removed pages one click further away keeps what the owner rejected. Keeping the
numbered step headings keeps the procedure register; the checklist is for the author, and the
reader is owed an article.

## Amendment (6 September 2026)

Walkthrough sections may carry ordinal numbers in their headings. The owner's revision of the
Michelson article on 6 September 2026 numbers its sections because the page is read as a
two-stage argument whose later steps refer back to earlier ones, and a number is the shortest
such reference. The seven-step checklist of ADR 0028 still governs the order; the step *names*
still do not become headings, and the register stays that of an article.
