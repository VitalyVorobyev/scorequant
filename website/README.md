# ScoreQuant learning portal

A Docusaurus site with an in-browser ScoreQuant runtime (Pyodide + the NumPy backend). It is
deliberately isolated from Python packaging: Node and pnpm are pinned here, `uv` owns everything
Python.

The portal is served at `/scorequant/portal/`, beside the MkDocs documentation at
`/scorequant/docs/`; a hand-written landing page in `landing/` owns the site root and links to
both (ADR 0027). `website/src/lib/site.ts` is the one place those bases are written down, and
`baseUrl` in `docusaurus.config.ts` must match `SITE_BASE` there.

```bash
corepack pnpm install --frozen-lockfile
corepack pnpm start        # http://localhost:3000/scorequant/portal/
```

Two things that are not obvious and will stop a first run:

- **Node >= 20** (`engines` in `package.json`); `.node-version` pins `24.19.0`, which is what CI
  builds on. A mismatch is not a hard failure — pnpm prints `[WARN] Unsupported engine` and runs
  the command anyway — so match `.node-version` when a build difference would matter.
- **`corepack` is not bundled with Node 25 and later.** Every command below invokes `corepack pnpm`
  so that pnpm's version comes from `packageManager` rather than from your machine. On a current
  Node, install it once with `npm install -g corepack@latest`, or the commands fail with
  `command not found: corepack`.
- **`start` shells back into `uv`.** It runs `generate` first, which executes
  `uv run python website/scripts/generate_data.py`, `generate_showcase.py`,
  `generate_walkthroughs.py` and `generate_snippets.py` from the repository root, and needs the
  `portal` dependency group. Run `uv sync --all-extras --all-groups --locked` at the root first.

`corepack pnpm build` additionally downloads the pinned Pyodide release and builds the ScoreQuant
wheel, so it needs network access.

## Where a figure file goes

Two directories under `static/`, with opposite rules (ADR 0032):

- **`static/figures/` is committed.** Hand-added images live here and are referenced as
  `<Figure src={siteUrl("figures/name.svg")} ... />`. The Michelson score panels are written here
  by `examples/michelson_phase.py`, so they are regenerable *and* committed.
- **`static/walkthrough-figures/` is generated and gitignored.** `scripts/generate_walkthroughs.py`
  refills it on every build by copying each entry of its `FIGURES` map out of `docs/examples/assets/`.
  A file merely placed there exists only on the machine that placed it and 404s everywhere else.

`tests/figures.test.ts` resolves every `siteUrl("...")` in the MDX against the lane that owns it, so
putting a file in the wrong one fails the suite instead of shipping a broken image.

Full instructions, including the generated files you must not hand-edit and the checks CI runs, are
in [`docs/playbook.md`](../docs/playbook.md). The design contract is
[ADR 0019](../docs/adr/0019-react-learning-portal.md); the backend contract is
[ADR 0018](../docs/adr/0018-explicit-multi-backend-execution.md).
