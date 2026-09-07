# Developer playbook

How to run each part of this repository, in the order you would meet them, with the failure each
step produces when a prerequisite is missing. [Development](development.md) is the reference for
*what* the commands check; this page is the reference for *getting them to run at all*.

This page is not published to the documentation site.

## Toolchain

| Tool | Version | Pinned by | Used for |
| --- | --- | --- | --- |
| `uv` | any recent | — | every Python environment, command, build and lock |
| Node | **>=20**, CI builds 24.19.0 | `website/package.json` `engines`, `website/.node-version` | the React portal only |
| pnpm | **11.0.9** | `website/package.json` `packageManager` | the React portal only |

Python is managed entirely by `uv`; you never create a virtualenv by hand and never run `pip`.

**Node has two numbers and they mean different things.** `engines` in `package.json` is the real
requirement, `>=20` — inherited from Docusaurus 3, which declares `>=20.0`. `.node-version` is
`24.19.0`, the single version CI builds on and therefore the one to develop against when a build
difference would matter.

Do not read a `[WARN] Unsupported engine` line as a failure. pnpm prints it and then runs the
command anyway, so an engine mismatch is a warning you will scroll past rather than a stop. That
is the argument for keeping `engines` honest: a bound nothing enforces and nothing needs only
trains you to ignore the warning that would matter.

```bash
fnm install 24.19.0 && fnm use 24.19.0     # or: nvm install 24.19.0 && nvm use
node --version                             # v24.19.0 matches CI; >=20 will build
```

pnpm comes from `corepack`, which reads the `packageManager` field and fetches exactly that pnpm,
so the version is a property of the repository rather than of your machine. **Corepack is no longer
bundled with Node.** It shipped with Node through 24 and was unbundled in 25, so on a current Node
`corepack pnpm` fails with `command not found` until you install it once:

```bash
npm install -g corepack@latest    # only needed on Node >= 25
```

Always invoke it as `corepack pnpm`, and always from inside `website/`: corepack resolves the
`packageManager` field from its own working directory, so `pnpm --dir website ...` picks the wrong
version or none at all.

## The library

```bash
uv sync --all-extras --all-groups --locked
```

That single command installs the library, its extras, and every dependency group — including
`portal`, which holds `griffe` and `marimo`. The portal build shells back into this environment, so
skipping it makes the *website* fail later with a confusing Python error.

Run things:

```bash
uv run python -c "import scorequant as sq; print(sq.__all__)"
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto -m "not docs_execution"   # library tier
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto -m docs_execution         # prose tier
JAX_ENABLE_X64=0 MPLBACKEND=Agg uv run pytest tests/test_float32.py             # float32 leg
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest tests/test_fit.py -k name         # one test, no -n
```

`-n auto` for a full run, omitted for a targeted one. X64 is an application choice made by the
caller; the package never touches global JAX configuration at import.

## The reference documentation site (MkDocs)

```bash
uv run mkdocs serve            # http://127.0.0.1:8000
uv run mkdocs build --strict   # what CI runs; a broken link or nav entry fails it
```

The API reference is collected from docstrings by mkdocstrings. `development.md`, `roadmap.md`,
`system-design.md`, this page, `decisions.md` and `programme/**` are excluded from the published site
(`mkdocs.yml` `exclude_docs`), and `tests/test_readme.py` asserts that the exclusion list and the
prose guard agree.

## The React portal

A Docusaurus site with a real Pyodide runtime, isolated from Python packaging.

### First run

```bash
cd website
corepack pnpm install --frozen-lockfile
corepack pnpm start                    # http://localhost:3000/scorequant/
```

`start` runs `generate` first, and `generate` shells back to
`uv run python website/scripts/generate_data.py` from the repository root. **A bare
`pnpm install && pnpm start` fails** if the Python environment is not synced with the `portal`
group, because that script imports `griffe` to read the API surface.

In dev mode the Lab falls back to its fixture and search falls back to an in-memory route list;
both need the production build below.

### Production build

```bash
cd website
corepack pnpm build
```

`build` is `prepare:runtime` followed by `build:content`. `prepare:runtime` **downloads the pinned
Pyodide release (~15 MB) from GitHub and runs `uv build --wheel` from the repository root**, so it
needs network access and a working Python environment. It prunes Pyodide to the seven files the Lab
actually loads and drops the ScoreQuant wheel into `website/static/runtime/wheels/`.

Note `website/static/runtime/manifest.json` pins the wheel by filename, including its version — a
version bump requires re-running `prepare:runtime` rather than editing the manifest.

### Checks

```bash
corepack pnpm typecheck     # tsc --noEmit, strict
corepack pnpm lint          # eslint, type-aware
corepack pnpm test          # vitest
corepack pnpm validate      # generate + lessons + typecheck + lint + test + build:site
corepack pnpm test:e2e      # playwright; needs a prior build, it serves build/
```

### Previewing both sites together

```bash
uv run mkdocs build --strict
cd website && corepack pnpm assemble:site
```

This writes `.pages-preview/` — the portal at the root, the MkDocs documentation at `/docs/`, and
the pre-cut redirect stubs — and prints the redirect and reference-link parity check
([ADR 0035](decisions.md)).
Running it locally does **not** deploy.

Deployment is `site.yml`, and it is the same tree: a pull request builds and uploads it for
inspection, and a push to `main` publishes it to
[the site](https://vitalyvorobyev.github.io/scorequant/)
([ADR 0026](decisions.md)). That workflow is the only place the
strict MkDocs build and the Playwright suites run in CI.

### Things that will cost you an hour if nobody says them

- `website/src/lab/protocol.generated.ts` is committed **and** generated. Edit
  `website/schema/lab-protocol.schema.json` and run `corepack pnpm generate:protocol`.
- `website/src/generated/portal-data.json` is likewise committed and generated, by
  `website/scripts/generate_data.py`.
- The stylesheet is ten files under `website/src/css/`, listed **in cascade order** in
  `docusaurus.config.ts` under `theme.customCss`. There are no CSS modules, so order is the only
  thing resolving equal-specificity collisions: `tokens.css` first, `live-fit.css` last. Adding a
  file means adding it to that array, in the right place.
- There is no backend and no HTTP call to one. The `remote-jax` and `remote-pytorch` runners exist
  in the protocol schema and are explicitly unimplemented.
- In CI, `corepack enable` must run both before and after `setup-node`, because `cache: pnpm`
  shells out to `pnpm store path` in between.

The browser Lab has its own contract worth knowing: the wire protocol is
`website/schema/lab-protocol.schema.json`, the browser-side Python is
`website/static/runtime/python/scorequant_browser_lab.py`, and `tests/test_browser_lab.py` runs
that module under CPython and pins it against the committed fixtures. A change to one side that
the other does not follow fails there rather than in a browser — which is how the stale wheel was
caught when the config fields were renamed.

## Examples and the FlowCyt study

Every example honors one environment variable:

```bash
SCOREQUANT_EXAMPLE_FAST=1 uv run python -m examples.gaussian_location
```

Any non-empty value shrinks dataset sizes and optimizer budgets. Unset keeps the full
research-scale sizes that produced the committed gallery figures — so leave it unset only when you
intend to regenerate and inspect them.

The FlowCyt study runs from the committed 1.3 MB fixture by default:

```bash
uv run python -m examples.cell_population --fixture examples/data/flowcyt_fixture.npz --quick
uv run python -m examples.cell_population --data-dir flowcyt-results/data_original   # full study
```

The full 3.7 GB dataset is gitignored and never committed. The FlowCyt data is licensed
CC BY-NC-SA 4.0, separately from ScoreQuant's MIT license; anything derived from it inherits
attribution and share-alike.

## Benchmarks

```bash
JAX_ENABLE_X64=1 uv run python benchmarks/bench.py --rows 20000 --bins 8
JAX_ENABLE_X64=1 uv run python benchmarks/bench.py --check benchmarks/baselines.json \
  --time-tolerance 10 --quality-rtol 1e-6
```

Run these unpinned and single-process — no `-n`, no thread pinning — because that is how
`benchmarks/baselines.json` was measured. Quality is the real regression signal; wall-clock is
deliberately given a loose tolerance because CI machines differ.

## Publishing a release

Nothing publishes automatically. `release.yml` triggers on a `v*` tag, so publication is a
deliberate push.

The trusted publisher is configured on pypi.org, which is what removes the need for an API token
anywhere. All four fields have to keep matching the workflow or the OIDC exchange is refused:

| Field | Value |
| --- | --- |
| PyPI project | `scorequant` |
| Owner | `VitalyVorobyev` |
| Repository | `scorequant` |
| Workflow | `release.yml` |
| Environment | `pypi` |

There is no TestPyPI rehearsal. A version number on PyPI is spent the moment it is published and
cannot be reused, so the rehearsal that matters is local — build the artifact, install it into a
scratch environment away from the source tree, and check it imports and runs:

```bash
uv build
uv venv /tmp/sq && uv pip install --python /tmp/sq/bin/python dist/scorequant-*.whl
cd /tmp && /tmp/sq/bin/python -c "import scorequant as sq; print(sq.__version__)"
```

`cd /tmp` matters: run from the repository root and the import resolves to `src/` rather than to
the wheel, and the check proves nothing. Then publish:

```bash
git tag -a v0.2.0 -m "ScoreQuant 0.2.0"
git push origin v0.2.0
```

The workflow re-runs the full handoff gate, checks the tag matches the packaged version, runs
`twine check` so the README cannot render as raw text on the project page, and only then publishes.
Bumping a version means editing `pyproject.toml` alone — `scorequant.__version__` is read from
installed metadata — plus a `CHANGELOG.md` entry and re-running the portal's `prepare:runtime`,
since `website/static/runtime/manifest.json` pins the wheel by filename.

## Before handing work off

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check src
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto
JAX_ENABLE_X64=0 MPLBACKEND=Agg uv run pytest tests/test_float32.py
uv build
uv run mkdocs build --strict
```

Plus `corepack pnpm validate` in `website/` when the portal changed, and
`(cd agenticresearch/formal && lake build --wfail)` when the Lean formal evidence changed.
