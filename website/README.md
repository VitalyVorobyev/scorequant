# ScoreQuant learning portal

A Docusaurus site: a landing page, get-started, four walkthroughs, the Ideas series and a
research page. Two pages refit a result in the reader's browser, on Pyodide and ScoreQuant's
NumPy backend. Node and pnpm are pinned here; `uv` owns everything Python.

The portal owns the site root, `/scorequant/`, and the MkDocs reference is mounted beneath it at
`/scorequant/docs/`. `src/lib/site.ts` is the one place those bases are written, and `baseUrl` in
`docusaurus.config.ts` must match `SITE_BASE` there.

## Running it

```bash
uv sync --all-extras --all-groups --locked   # at the repository root, once
corepack pnpm install --frozen-lockfile
corepack pnpm start                          # http://localhost:3000/scorequant/
corepack pnpm validate                       # generate, typecheck, lint, unit tests, build
```

- **Node >= 20.** `.node-version` pins what CI builds on; pnpm only warns on a mismatch.
- **`corepack` is not bundled with Node 25 and later.** Install it once with
  `npm install -g corepack@latest`; every command goes through it so pnpm's version comes from
  `packageManager`.
- **`start` and `validate` shell back into `uv`.** Both run `generate` first (below), from the
  repository root.
- **`build` needs the network.** It downloads the pinned Pyodide release and builds the
  ScoreQuant wheel (`prepare:runtime`) before building the site.

## Layout

| Path | What it holds |
| --- | --- |
| `get-started/`, `walkthroughs/`, `ideas/` | The MDX pages, each a docs-plugin instance. |
| `src/pages/` | The landing and research pages. |
| `src/components/` | Figures, charts and the `LiveFit` demo, used from MDX and pages. |
| `src/lab/` | The browser runtime: protocol types, worker, client, hook. |
| `src/data/`, `src/lib/` | Typed readers of the generated data; navigation and helpers. |
| `src/theme/` | Small swizzles: the shell layout, the table of contents, keyboard-reachable tables. |
| `src/css/` | Five stylesheets, listed in cascade order in `docusaurus.config.ts`. |
| `src/generated/` | JSON written by the generators. Committed; never edit by hand. |
| `scripts/` | The generators, the runtime preparation and the site assembly. |
| `static/` | Figures, score tables for the live demos, the browser runtime. |
| `tests/` | Vitest unit tests, and `tests/e2e/` for Playwright. |

The stylesheets are `tokens.css` (the palette, its semantic aliases and the dark-mode override),
`shell.css` (element defaults, header, footer, breakpoints), `pages.css` (landing, research and
walkthroughs index), `prose.css` (the docs frame, tables, code) and `figures.css` (charts,
diagrams, `LiveFit`). Colours are always tokens; `tests/contrast.test.ts` checks the pairs.

## Generated data

`pnpm generate` runs four Python scripts through `uv`:

- `generate_data.py` writes the criterion/solver matrix into the docs and the root README.
- `generate_showcase.py` writes the FlowCyt chart data.
- `generate_walkthroughs.py` writes every number a page prints, each with a pointer into a
  committed evidence file, the live demos' score tables and the walkthrough figures.
- `generate_snippets.py` runs `get_started_program.py` cell by cell and captures the output that
  get-started shows.

`tests/test_walkthrough_facts.py` and its neighbours re-resolve these against the evidence, so a
stale file fails the Python suite.

## The live demos

`LiveFit` shows the committed result and one button. Only the click loads `LiveFitRunner`, through
a dynamic import, so an ordinary page load never fetches the runtime; the e2e suite checks that.
The runner calls `useLabRunner`, which validates the problem (`src/lab/protocol.ts`) and hands it
to `runtimeClient.ts`, the one place a `Worker` is created: one worker per tab, kept warm between
runs and terminated on cancel. `lab.worker.ts` loads Pyodide, installs the wheel and runs
`static/runtime/python/scorequant_browser_lab.py`, which `tests/test_browser_lab.py` also runs
under CPython. The request and event types in `protocol.ts` are the whole contract between the
page and the worker; the Python adapter reads the same field names.

Two rules keep the production build working: spread a `Set` with `Array.from`, never `[...set]`,
and keep focusable nodes out of an `svg[role=img]`.

## Where a figure file goes

Two directories under `static/`, with opposite rules:

- **`static/figures/` is committed.** Hand-added images live here and are referenced as
  `<Figure src={siteUrl("figures/name.svg")} ... />`.
- **`static/walkthrough-figures/` is generated and gitignored.** `generate_walkthroughs.py` refills
  it on every build from `examples/outputs/`. A file merely placed there 404s everywhere else.

`tests/figures.test.ts` resolves every `siteUrl("...")` in the MDX against the lane that owns it.

## Building the published site

`corepack pnpm build` writes `build/`. `corepack pnpm assemble:site` then copies it and the MkDocs
`site/` into `.pages-preview/`, with the reference under `docs/`. CI (`.github/workflows/site.yml`)
runs the reference build, typecheck, lint, unit tests, the build, Playwright and the assembly.

The full handoff gate is in [`docs/development.md`](../docs/development.md), and the design
principles in [`docs/system-design.md`](../docs/system-design.md).
