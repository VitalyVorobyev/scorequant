# Development

How to run every part of the repository. The contract is [`AGENTS.md`](../AGENTS.md); the design
is [`system-design.md`](system-design.md). This page is not published.

## Toolchain

| Tool | Version | Used for |
| --- | --- | --- |
| `uv` | any recent | every Python environment, command, build and lock |
| Node | >=20; CI builds `website/.node-version` (24.19.0) | the portal only |
| pnpm | `packageManager` in `website/package.json`, via `corepack pnpm` | the portal only |

Corepack is not bundled with Node 25+; install it once with `npm install -g corepack@latest`.
Always run `corepack pnpm` from inside `website/`. A `[WARN] Unsupported engine` line is a
warning, not a failure.

## The library

```bash
uv sync --all-extras --all-groups --locked
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto -m "not docs_execution"   # library tier
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto -m docs_execution         # prose tier
JAX_ENABLE_X64=0 MPLBACKEND=Agg uv run pytest tests/test_float32.py             # float32 leg
```

The prose tier executes the README, documentation and portal code fences. A bare `uv run pytest`
runs everything. Use `-n auto` for full runs only; `tests/conftest.py` pins each xdist worker to
one compute thread so JAX workers do not fight over cores. X64 is an application choice: the
package never changes global JAX configuration.

Ruff enforces annotations and bans `typing.Any`; `ty check src` type-checks the package.

## Reference documentation (MkDocs)

```bash
uv run mkdocs serve            # http://127.0.0.1:8000
uv run mkdocs build --strict   # what CI runs
```

The API reference is collected from docstrings by mkdocstrings; `docs/api.md` is the hand-written
guide and the inventory of the stable API. `development.md` and `system-design.md` are excluded
from the published site, and `tests/test_readme.py` checks the exclusion list.

## The portal (Docusaurus)

```bash
cd website
corepack pnpm install --frozen-lockfile
corepack pnpm start          # http://localhost:3000/scorequant/
corepack pnpm validate       # generate + typecheck + lint + unit tests + build
corepack pnpm test:e2e       # Playwright against the production build
```

`start` and `validate` run `generate` first, which calls Python generators through `uv run` from
the repository root (`website/scripts/*.py`); they need the synced environment above.
`corepack pnpm build` also downloads the pinned Pyodide release and builds the ScoreQuant wheel
for the browser runtime, so it needs network access.

Generated and committed — never hand-edit:
- `website/src/generated/*.json`, written by the generators from `examples/` outputs and `src/`.

The browser-side Python is `website/static/runtime/python/scorequant_browser_lab.py`;
`tests/test_browser_lab.py` runs it under CPython against committed fixtures.

The stylesheets in `website/src/css/` are listed in cascade order in `docusaurus.config.ts`.

If Playwright cannot start its web server (some sandboxes kill it), start
`corepack pnpm exec docusaurus serve --host 127.0.0.1 --port 4173 --no-open` yourself; the suite
reuses a running server.

### Both sites together

```bash
uv run mkdocs build --strict
cd website && corepack pnpm assemble:site
```

This writes `.pages-preview/` (portal at the root, MkDocs at `/docs/`) and checks that every
portal link into the reference resolves. It does not deploy: `site.yml` deploys on a push to
`main`.

## Examples

Every example honours `SCOREQUANT_EXAMPLE_FAST`: any non-empty value shrinks sizes and budgets.
Use `examples/_env.py`'s `is_fast_mode()` and `example_scale(full, fast)` in example code.

The FlowCyt study runs from the committed fixture:

```bash
uv run python -m examples.cell_population --fixture examples/data/flowcyt_fixture.npz --quick
```

The full dataset is gitignored. It is licensed CC BY-NC-SA 4.0, separately from ScoreQuant's
MIT license, and anything derived from it inherits attribution and share-alike.

## Lean proofs

```bash
cd research/formal && lake exe cache get && lake build --wfail
```

Needed only when `research/formal` changes; CI runs it with `leanchecker` and an axiom audit.
See `research/formal/README.md`.

## Benchmarks

```bash
JAX_ENABLE_X64=1 uv run python benchmarks/bench.py --check benchmarks/baselines.json \
  --time-tolerance 10 --quality-rtol 1e-6
```

Run single-process, as the baselines were measured. Quality is the regression signal, not time.
Regenerate after an intentional change with `--regenerate benchmarks/baselines.json --repeats 1`.

## Releasing

Nothing publishes automatically: `release.yml` runs on a `v*` tag pushed by the owner, re-runs
the gate, checks the tag against the packaged version, runs `twine check` and publishes through
PyPI Trusted Publishing (project `scorequant`, workflow `release.yml`, environment `pypi`).

A version bump edits `pyproject.toml` (the package reads its version from metadata), adds a
`CHANGELOG.md` entry, edits the wheel filename in `website/static/runtime/manifest.json` (it pins
the wheel by name, and `prepare:runtime` reads the name from it), and re-runs
`corepack pnpm prepare:runtime`. Rehearse locally, away from the
source tree:

```bash
uv build
uv venv /tmp/sq && uv pip install --python /tmp/sq/bin/python dist/scorequant-*.whl
cd /tmp && /tmp/sq/bin/python -c "import scorequant as sq; print(sq.__version__)"
```

## Handoff gate

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check src
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto
JAX_ENABLE_X64=0 MPLBACKEND=Agg uv run pytest tests/test_float32.py
uv build
uv run mkdocs build --strict
```

Add `corepack pnpm validate` and `test:e2e` in `website/` when the portal changed, and the Lean
build when `research/formal` changed.
