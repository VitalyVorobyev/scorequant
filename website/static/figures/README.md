# Committed figure files

Hand-added images the portal serves at `/figures/...`, referenced from MDX as
`<Figure src={siteUrl("figures/name.svg")} ... />`.

Files here are **committed**. Its neighbour `../walkthrough-figures/` is not: that
directory is gitignored and regenerated on every build by
`website/scripts/generate_walkthroughs.py`, which copies each entry of its `FIGURES`
map out of `docs/examples/assets/`. A file dropped into `walkthrough-figures/` exists
only on the machine that put it there and 404s everywhere else, so put hand-added
images here instead.

`website/tests/figures.test.ts` enforces both halves of that rule.
