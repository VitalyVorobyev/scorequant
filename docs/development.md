# Development

ScoreQuant uses [uv](https://docs.astral.sh/uv/) for environments, dependencies, command execution, locking, and package builds. Do not maintain a parallel pip or Conda workflow.

## Set up the repository

```bash
uv sync --all-extras --all-groups --locked
```

Normal CPython installs include JAX and Optax; Emscripten/Pyodide installs use the NumPy backend
without them. Matplotlib, notebook tooling, and both documentation stacks remain optional
development concerns.

## Validate changes

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check src
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto
JAX_ENABLE_X64=0 MPLBACKEND=Agg uv run pytest tests/test_float32.py
uv build
uv run mkdocs build --strict
```

## Validate formal research claims

Selected registry claims carry a machine-checked proof in an isolated Lean 4 workspace. It is a
separate toolchain, needed only when the formal evidence changes:

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh   # once
cd agenticresearch/formal && lake exe cache get && lake build --wfail
```

The build is a required pull-request and release gate, alongside `leanchecker` and an axiom
audit allowing exactly `propext`, `Classical.choice` and `Quot.sound`. It certifies the stated
theorems, not that ScoreQuant's Python/JAX code implements them. Upgrading `lean-toolchain`, the
Mathlib revision and `lake-manifest.json` is a dedicated reviewed change; they move together.
See [ADR 0030](decisions.md).

X64 is an explicit application and CI choice. The package never changes global JAX configuration during import.

## Test tiers and parallelism

The suite is tiered by what a test is for rather than by how long it takes, so a
bare `uv run pytest` still runs everything. `tests/conftest.py` marks the modules
that execute published prose -- documentation snippets, README fences, and the
notebooks -- with `docs_execution`; they re-run the same `examples/` generators
the library tier already asserts against, so what they catch is presentation
drift rather than a numerical regression. CI runs the two tiers as parallel jobs:

```bash
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto -m "not docs_execution"
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto -m docs_execution
```

Use `-n auto` for a full run and omit it when running one test. Under
`pytest-xdist`, `tests/conftest.py` pins each worker to a single compute thread
before anything imports JAX: XLA sizes its thread pool from the host core count,
so unpinned workers each try to claim the whole machine and spend their time
descheduling one another. The benchmark harness deliberately runs unpinned and
single-process, because that is how `benchmarks/baselines.json` was measured.

Ruff enforces complete function annotations and bans importing `typing.Any`. `ty` then checks those annotations across `src/`; public conversion boundaries use `numpy.typing.ArrayLike` rather than leaking JAX-specific types.

## Preview documentation

```bash
uv run mkdocs serve
```

The generated API reference is collected from NumPy-style docstrings with mkdocstrings. Public API changes must update docstrings, the handwritten [API guide](api.md), examples, and an ADR when the decision is durable.

## Develop the learning portal

The React workspace is intentionally isolated from uv and Python packaging. Use the Node and pnpm
versions pinned by `website/package.json`:

```bash
cd website
corepack pnpm install --frozen-lockfile
corepack pnpm start
corepack pnpm validate
corepack pnpm build
corepack pnpm assemble:site
```

Portal data generators invoke Python through `uv run` from the repository root; do not install a
second Python environment under `website/`. MkDocs remains the engineering reference while the
portal owns curated learning and the browser lab. The production `build` downloads and prunes the
pinned Pyodide release to its 15 MB core/NumPy/micropip runtime, builds the local Emscripten-safe
ScoreQuant wheel, exports the locked marimo lesson, and generates Pagefind. `assemble:site`
combines the built portal at the site root with an existing strict MkDocs build at `/docs/`
(ADR 0035); it does not deploy.

## Example fast mode

Every example script and notebook honors one environment variable, `SCOREQUANT_EXAMPLE_FAST`: any non-empty value shrinks dataset sizes and optimizer budgets for CI and quick local checks, while an unset variable keeps the full research-scale sizes used for the committed gallery figures. `examples/_env.py` provides the two helpers example code should use directly, `is_fast_mode()` and `example_scale(full, fast)`; `tests/test_notebooks.py` sets it before executing every notebook.

```bash
SCOREQUANT_EXAMPLE_FAST=1 uv run python -m examples.gaussian_location
```

## Benchmark harness

`benchmarks/bench.py` is a deterministic, seeded timing-and-quality harness, not a runtime
promise. It covers every public solver path — `d_exchange` and `d_exchange_nobatch`, `lloyd`,
`kmeans`, `soft`, `scalar_dp`, `profiled_exchange`, `certify`, and the two reusable rules,
`predict` (`QuantizerResult.predict_scores`) and `compile` (`PartitionResult.compile_quantizer`) —
over a `--rows` × `--dims` × `--bins` × `--scenarios` matrix, reporting wall-clock seconds (minimum
over `--repeats`), process-lifetime peak RSS, and a solver-appropriate quality metric (a
log-determinant objective or a geometric-mean retention) alongside solver diagnostics such as
`accepted_moves`, `scans`, and `exchange_stable`. `--max-scans` caps the exchange scan budget,
which turns a cell into a fixed-work steady-state probe for row counts whose full convergence
exceeds one measurement window:

```bash
JAX_ENABLE_X64=1 uv run python benchmarks/bench.py --rows 20000,100000 --bins 8,64
JAX_ENABLE_X64=1 uv run python benchmarks/bench.py --rows 200000 --bins 8 --scenarios d_exchange --json out.json
```

`benchmarks/baselines.json` pins a small CI-suitable matrix (recorded machine, Python, and JAX
versions live in its `environment` field; see that file rather than dated prose here for absolute
numbers). Check a fresh run against it with:

```bash
JAX_ENABLE_X64=1 uv run python benchmarks/bench.py --check benchmarks/baselines.json \
  --time-tolerance 10 --quality-rtol 1e-6
```

`--check` re-runs exactly the scenarios recorded in the baseline file (its own `--rows`/`--bins`
flags are ignored in this mode) and prints a comparison table. It fails (exit 1) if any scenario
runs slower than `--time-tolerance` times its baseline — deliberately loose, since CI machines
differ — or if a quality metric drifts beyond `--quality-rtol`; deterministic seeds make quality
the real regression signal, since it is exact for a given seed and code path. The `benchmarks` CI
job runs this check after the test suite passes. To refresh the file after an intentional
performance change, replay its recorded cell list and review the diff:

```bash
JAX_ENABLE_X64=1 uv run python benchmarks/bench.py --regenerate benchmarks/baselines.json --repeats 1
```

Use `--repeats 1`, matching how `--check` runs: the harness reports the minimum over repeats, so
a baseline recorded warm and checked cold reports spurious slowdowns on compilation-dominated
cells.

## Profiling and the Rust question

`benchmarks/profile.py` is a sampling profiler that drives the same scenario runners, separating
JIT warm-up from steady state and writing folded stacks plus JSON summaries under
`benchmarks/profiles/`. `benchmarks/README.md` holds the full measured campaign to one million
rows: a per-solver bottleneck table, a phase decomposition of one exchange scan, a measured
machine roofline, and the applied optimizations with before/after numbers.

Its conclusion, recorded so the question does not get reopened on intuition: **the numerical core
does not justify a Rust port.** At a million rows with 64 bins and 8 score dimensions, one
D-exchange scan spends 86.7% inside a single compiled XLA kernel, 9.9% inside NumPy's compiled
reductions, and 1.6% in JAX's Python dispatch; across a whole run, Python-interpreter
orchestration is at most about 4%, so a port that made it free would return about 1.04x. The
kernel does sit at only 7% of this machine's measured float64 matmul roofline, but that headroom
is a formulation problem — reshaping the batched contraction into a GEMM measures 3.5x while
staying in JAX — not a language problem.

The one path with a real case is the branch-and-bound certifier in `certify.py`, which runs no
JAX kernel and measures a flat ~34–40k nodes per second across a 260x range of tree sizes, with
roughly 38% of that time in NumPy allocation and dispatch on 2x2 matrices. A port of `_Search`
alone would plausibly return 40x or more. It stays deferred: it would put a compiled extension in
the wheel for one bounded diagnostic, and `AGENTS.md` requires an approved roadmap change before a
second numerical backend.

## Repository guidance

The root `AGENTS.md` is the durable engineering contract for coding sessions. It records numerical invariants, module boundaries, `uv` commands, validation expectations, and deferred scope.
