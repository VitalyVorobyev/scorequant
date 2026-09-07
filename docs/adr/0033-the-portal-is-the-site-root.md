# ADR 0033: The portal is the site root, and the landing page is retired

**Status:** Accepted

**Supersedes:** [ADR 0027](0027-landing-page-at-the-root.md); keeps
[ADR 0026](0026-one-workflow-publishes-the-site.md) unchanged; amends
[ADR 0031](0031-portal-reduced-to-four-surfaces.md) on what the home page contains

## Context

ADR 0027 gave the site root to a hand-written landing page because neither larger surface could
be trusted with it: the portal's home page read as assertion rather than instruction, and the
MkDocs site had been demoted without a link back. The landing page was the smallest thing that
could hold the root honestly — one file, no build step, only links.

ADR 0031 then rebuilt the portal around that judgement. The home page became definitions rather
than a sales page, the Lessons index, the API catalogue, the Benchmarks page and the blog were
removed, and every walkthrough became an article whose numbers trace to a committed run. On
7 September 2026 the owner judged the result good enough to be the front door and asked for the
landing page to be removed.

Two pages at the front is then the cost rather than the safeguard. A reader arriving at the root
reads a page whose only content is a list of the two places to go, and then reads the portal's
home page, which introduces the same subject again. The landing page also carries its own copy of
the identity, the install line and the quickstart, kept in step with the README and the MkDocs
source by the four guards in `tests/test_landing.py`. Retiring it removes a surface, not a
capability.

## Decision

**Two surfaces, one assembled tree, one workflow:**

| URL under `https://vitalyvorobyev.github.io/scorequant/` | Serves | Source |
| --- | --- | --- |
| `/` | the Docusaurus learning portal | `website/` |
| `/docs/…` | the MkDocs documentation and book | `docs/`, `mkdocs.yml` |

`landing/` and `tests/test_landing.py` are deleted. `baseUrl` and `SITE_BASE` move to
`/scorequant/` in lockstep, and `website/scripts/assemble-site.mjs` copies the portal build to the
root and the MkDocs build into `docs/` beneath it.

**The home page keeps the register ADR 0031 set and loses two of its parts.** It defines the
problem, the score, why score space is the place to choose cells, the two tasks, where scores come
from and what is optimised, and it still quotes no measurement and runs nothing. It no longer
carries the displayed binning-cost identity, nor the list of where each definition is derived.
Both belonged to a page that was competing with a landing page for the role of first thing read;
the front door does not need to prove the theorem on the doorstep. The consequence is that the
home page has no link of its own into the MkDocs reference — the portal's primary navigation
carries a *Reference* entry on every page, including this one, so the documentation stays one
click away from the root rather than zero.

**The portal's `404.html` now serves the whole domain.** GitHub Pages serves the root
`404.html` for any unmatched path, so an unknown URL under `/docs/` now lands on the portal's
not-found shell rather than the MkDocs one. That is the better of the two: it carries the
site navigation.

**The two days the portal spent at `/portal/` are deliberately not stubbed.** This is the same
judgement ADR 0027 made about the one day the portal spent at the root: the only links to those
URLs are in `mkdocs.yml`, `README.md` and `docs/`, and this decision re-points all of them.
`website/redirects.json` is otherwise unchanged — its fifty-two stubs still target `docs/…`, its
one unstubbed path is still the root, and its `sourceSitemapCount` invariant still accounts for
exactly the fifty-three pre-cut MkDocs URLs. Only that root entry's stated reason changes: the URL
now serves the portal rather than the landing page.

**The cross-surface link check survives its subject.** ADR 0027 had the assemble script resolve
every link the landing page carried against the assembled tree, because a hand-written file has no
link checker. The landing page is gone but the gap it covered is not: Docusaurus's
`onBrokenLinks: "throw"` follows route links inside the portal and knows nothing about
`/scorequant/docs/…`, and `website/tests/reference-links.test.ts` checks that such a link goes
through the `ReferenceLink` component, not that its target exists. The assemble script now
resolves every `/scorequant/docs/…` href in the built portal against the assembled tree instead,
and fails the build on a miss.

## Consequences

- Two places state the topology and must move together: `website/src/lib/site.ts` and
  `mkdocs.yml`. The assemble script's reference-link check is what catches a mismatch, as its
  landing-link check did before.
- The base-path guard (`website/tests/site-base.test.ts`) can no longer search for the substring
  `scorequant/portal`, because `/scorequant/` also occurs inside the GitHub source URLs in the
  generated portal data. It searches for a *quoted literal* opening with the deployed prefix
  instead, which is both narrower and stricter: it now also catches a hand-written
  `"/scorequant/docs/"`.
- `pyproject.toml` needs no change. It advertises the root as Homepage — now the portal — and
  `/docs/` as Documentation, both still correct.
- The portal must never emit a `docs/` route. The MkDocs tree is copied into the portal's tree
  rather than beside it, so such a route would be replaced silently; the assemble script refuses
  the build instead.
- No release is cut for this. It changes published URLs and no packaged code.
