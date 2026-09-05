# ScoreQuant contributor guidance

## Project contract

ScoreQuant compresses continuous or high-dimensional events into hard bins while preserving Fisher information. Keep task and representation semantics explicit:

```text
Source + (densities | density ratios | scores) -> ScoreProvider -> score law -> partition or quantizer
```

- `optimize_partition(scores, ...)` owns fixed-sample assignment and returns no predictor.
- `fit_quantizer(source, provider=...)` owns reusable score-space rules.
- `scores_from_components(Phi, coefficients)` and the ratio algebra in `ratios.py` are explicit
  adapters, not fitting tasks.
- Prediction is always `predict_scores`; observation-to-score conversion remains visible.
- Model density ratios enter through providers; importance ratios are source weights. The two
  never share an argument, and estimated ratios never claim exact Fisher semantics.

JAX remains the default runtime and Optax owns its gradient updates. NumPy is the approved portable
runtime from roadmap M9 and must implement the same shared mathematics, never a copied solver tree.
Do not add PyTorch or mutate global JAX configuration. Backend primitives stay private; there is no
public registry or provisional backend class.

## Numerical invariants

- Scores and weights are finite; weights are nonnegative and at least one is positive.
- Zero-weight rows remain predictable but contribute no information or optimization measure.
- Never center scores: the score-space origin has statistical meaning.
- Project numerically singular Fisher directions out; do not add information with a ridge.
- Validation samples are diagnostic only and never influence gradients, stopping, or checkpoint selection.
- Use deterministic seeds and judge optimizers by the final hardened partition, not only a soft objective.
- Preserve parameter-reparameterization, bin-relabeling, event-ordering, uniform-weight-scaling, and split-weight-duplication invariants.

## Code placement and API discipline

- Keep backend-free domain contracts separate from private execution adapters, shared mathematical
  kernels, solver orchestration, and the public task API. Fisher statistics remain in
  `information.py`, transforms in `transforms.py`, linear-model adapters in `components.py`, and
  density-ratio algebra in `ratios.py`; split oversized solver modules by stable responsibility.
- Keep backend-name branches inside execution resolution and adapters. Maintain one capability
  table and one conformance suite instead of per-backend task logic or duplicated tests.
- Keep dataset generators, comparisons, tuning, custom figure layouts, and exploratory logic in `examples/`, tests, or benchmarks.
- Add public concepts only when they are reusable, stable, documented, and non-duplicative. Prefer private helpers over provisional public APIs.
- Avoid aliases and overlapping entry points. Document intentional compatibility breaks and update the API guide, examples, and an ADR when the decision is durable.
- Keep optional visualization dependencies lazy and outside numerical hot paths.
- Avoid `O(N^2)` work in general-purpose solvers. The exact rank-one interval DP is an
  intentional capacity-limited exception guarded by `ScalarDPConfig.max_rows`. Optimization
  histories store aggregate metrics and center snapshots, never per-observation responsibilities.

## Research workflow

- `agenticresearch/` is the mathematical scientific memory (claim registry, counterexample bank, open-problem queue). It governs itself through its own `agenticresearch/AGENT.md` and is excluded from the Ruff gate.
- The library crosses into it at exactly two points: `tests/test_research_claims.py` reads counterexample fixtures from `agenticresearch/COUNTEREXAMPLES/`, and `tests/test_research_registry.py` runs `agenticresearch/py/registry.py validate` plus the index-freshness check. Research results become library behavior only by being copied into deterministic regression tests or theorem-cited code paths.
- Selected claims carry `formal_proof`: a machine-checked Lean proof of a *separately frozen and
  independently audited* statement, under `agenticresearch/formal/` and governed by
  `protocols/formalization.md` and ADR 0030. Partial or unfrozen Lean coverage is recorded in
  `KNOWN_RESULTS/` prose instead, never as a registry marker, and no Lean result certifies the
  Python/JAX implementation. Lean and Lake are the only language workspace beside Python/uv and
  Node/pnpm; run `(cd agenticresearch/formal && lake build --wfail)` when formal evidence changes.
- The registry is one file per claim under `agenticresearch/claims/`, with vocabularies in `registry.json`. Every index is generated — never hand-edit `claims/INDEX.md`, `COUNTEREXAMPLES/INDEX.md`, or `LITERATURE/BIBLIOGRAPHY.md`; run `python agenticresearch/py/registry.py reindex`.
- Code in `src/` that relies on a theorem names it; code that refuses a capability names the counterexample forcing the refusal. Keep both in sync with the registry.

## Tooling: use uv

`uv` is the only supported Python environment, dependency, build, and command runner. Do not
introduce pip, Conda, Poetry, or manually edit `uv.lock`. The isolated `website/` workspace uses
its pinned Node and pnpm versions.

```bash
uv sync --all-extras --all-groups --locked
uv run ruff check .
uv run ruff format --check .
uv run ty check src
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto
JAX_ENABLE_X64=0 MPLBACKEND=Agg uv run pytest tests/test_float32.py
uv build
uv run mkdocs build --strict
```

`pytest` is tiered by what a test is for, not by how long it takes, so a bare
`uv run pytest` still runs everything. `tests/conftest.py` marks the modules
that execute published prose -- documentation snippets, README fences,
notebooks -- as `docs_execution`, and CI runs the two tiers as parallel jobs:

```bash
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto -m "not docs_execution"  # library
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest -n auto -m docs_execution        # prose
```

Add `-n auto` for a full run and leave it off for a targeted one. Under xdist,
`tests/conftest.py` pins each worker to a single compute thread: XLA sizes its
pool from the host core count, so unpinned workers oversubscribe the machine
and parallelism becomes a net loss. Benchmarks deliberately run unpinned and
single-process, because that is how `benchmarks/baselines.json` was measured.

Use `uv add`, `uv remove`, and `uv lock` for dependency changes. Run commands through `uv run` so local and CI environments stay aligned.

## Engineering and documentation practices

- Make small, cohesive changes and reuse existing abstractions before adding new ones.
- Preserve unrelated work. Avoid destructive Git commands and hidden behavior changes.
- Use type annotations, meaningful names, and NumPy-style docstrings for every public object. Comments should explain why, especially for numerical choices, rather than narrate code.
- Use explicit array-like and recursive JSON contracts at conversion boundaries. `typing.Any` is prohibited in `src/`; Ruff `ANN401`, a banned-import rule, and `ty` enforce this.
- Add deterministic tests for changed behavior and numerical edge cases. Use fixed seeds and measurable assertions; avoid brittle pixel snapshots.
- Validate in proportion to risk: targeted tests while iterating, then the full commands above before handoff.
- Update user guides when workflows change. Run MkDocs in strict mode so broken navigation, links, or reference collection fail CI.
- Keep architecture decisions in `docs/adr/` and executable phase gates in `docs/roadmap.md`; do not create parallel planning files.
- Completed programmes and session packets are deleted or archived when they close; git history is
  their record. Only the current milestone, the ADRs and dated reviews under `docs/programme/`
  are standing text.
- Do not commit caches, local environments, build output, or `site/`. Commit gallery images only when intentionally regenerated and visually inspected.
- Do not push, merge, tag, publish, or deploy unless the user authorizes that action.

Persistence, services, remote execution, PyTorch, signed weights, and advanced statistical
objectives remain deferred to the roadmap. The static React portal and NumPy backend are approved
by M9/M10 and ADRs 0018/0019.

## Completion checklist

Before finishing, run the relevant tests plus Ruff, the strict documentation build, and the package build. Report the exact validation performed and explain any check that could not be run.
