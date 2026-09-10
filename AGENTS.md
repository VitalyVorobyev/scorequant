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

- `agenticresearch/` governs itself through its own `agenticresearch/README.md` and `OPEN_PROBLEMS.md`, and is excluded from the Ruff gate.
- The library crosses into it at exactly two points: `tests/test_research_claims.py` reads counterexample fixtures from `agenticresearch/COUNTEREXAMPLES/`, and `tests/test_research_registry.py` runs `agenticresearch/py/registry.py validate` plus the index-freshness check.
- Research results become library behaviour only by being copied into deterministic regression tests or theorem-cited code paths.
- Code in `src/` that relies on a theorem names it; code that refuses a capability names the counterexample forcing the refusal.
- Selected claims carry a machine-checked Lean `formal_proof` under `agenticresearch/formal/` (ADR 0030/0037); no Lean result certifies the Python/JAX implementation.

## Tooling: use uv

- `uv` is the only supported Python environment, dependency, build, and command runner: no pip, Conda, or Poetry, and never hand-edit `uv.lock` — use `uv add`, `uv remove`, and `uv lock` instead.
- The isolated `website/` workspace uses its own pinned Node and pnpm versions.
- Every command, test tier, and the handoff gate live in [`docs/development.md`](docs/development.md); run commands through `uv run` so local and CI environments stay aligned.

## Engineering and documentation practices

- Make small, cohesive changes and reuse existing abstractions before adding new ones.
- Preserve unrelated work. Avoid destructive Git commands and hidden behavior changes.
- Use type annotations, meaningful names, and NumPy-style docstrings for every public object. Comments should explain why, especially for numerical choices, rather than narrate code.
- Use explicit array-like and recursive JSON contracts at conversion boundaries. `typing.Any` is prohibited in `src/`; Ruff `ANN401`, a banned-import rule, and `ty` enforce this.
- Add deterministic tests for changed behavior and numerical edge cases. Use fixed seeds and measurable assertions; avoid brittle pixel snapshots.
- Validate in proportion to risk: targeted tests while iterating, then the handoff gate in `docs/development.md` before handoff.
- Update user guides when workflows change. Run MkDocs in strict mode so broken navigation, links, or reference collection fail CI.
- Keep durable decisions in `docs/decisions.md` (one short entry each, never a narrative record) and executable phase gates in `docs/roadmap.md`; do not create parallel planning files.
- Completed programmes, session packets and dated reviews are deleted when they close; git history is their record. Standing planning text is the current milestone in `docs/roadmap.md` and `docs/decisions.md`.
- Do not commit caches, local environments, build output, or `site/`. Commit gallery images only when intentionally regenerated and visually inspected.
- Do not push, merge, tag, publish, or deploy unless the user authorizes that action.

Persistence, services, remote execution, PyTorch, signed weights, and advanced statistical
objectives remain deferred to the roadmap. The static React portal and NumPy backend are approved
by M9/M10 and ADRs 0018/0019.

## Completion checklist

Before finishing, run the handoff gate in `docs/development.md` (or the targeted subset the change warrants) and report exactly which checks ran and any that could not.
