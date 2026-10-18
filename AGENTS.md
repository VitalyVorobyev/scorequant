# ScoreQuant contributor guidance

ScoreQuant bins continuous or high-dimensional events into a few hard categories while
preserving Fisher information about model parameters. The library is the product; the portal
explains and demonstrates it; `research/` holds the paper and the Lean proofs behind it.

```text
Source + (densities | density ratios | scores) -> ScoreProvider -> score law -> partition or quantizer
```

- `optimize_partition(scores, ...)` labels a fixed sample and returns no predictor;
  `fit_quantizer(source, provider=...)` returns a reusable rule. Prediction is always
  `predict_scores`, so observation-to-score conversion stays visible.
- Model density ratios enter through providers; importance ratios are source weights. Estimated
  ratios never claim exact Fisher semantics.

Read `docs/system-design.md` for the design principles and module map, `docs/development.md`
for every command and the handoff gate, and `ROADMAP.md` for the plan. There are no other
planning files.

## Numerical invariants

- Scores and weights are finite; weights are nonnegative and at least one is positive.
- Zero-weight rows remain predictable but contribute no information or optimization measure.
- Never center scores. Project numerically singular Fisher directions out; never add a ridge.
- Validation samples are diagnostic only: they never influence gradients, stopping or selection.
- Use deterministic seeds; judge optimizers by the final hard partition, not a soft objective.
- Preserve reparameterization, bin-relabeling, event-ordering, uniform-weight-scaling and
  split-weight-duplication invariants.
- Avoid `O(N^2)` work; the exact scalar interval DP is the one capped exception
  (`ScalarDPConfig.max_rows`). Histories store aggregates, never per-row responsibilities.

## Code and API discipline

- Keep backend-free contracts, the private execution adapters (`_execution.py`), shared kernels,
  solver orchestration and the public task API separate. Backend-name branches live only in the
  adapters; one conformance suite covers both backends. JAX is the default, NumPy the portable
  runtime. No PyTorch, no global JAX configuration.
- Add a public name only when it is reusable, stable, documented and not a second route to
  something that exists. No aliases. Prefer private helpers.
- Code that relies on a theorem names it. A refusal raises `RefusalError` with the id of the exact
  counterexample kept in `tests/fixtures/counterexamples/`.
- Dataset generators, comparisons, tuning and figures belong in `examples/`, never in `src/`.
- Type annotations everywhere; `typing.Any` is banned in `src/`. NumPy-style docstrings on every
  public object. Comments explain why, especially for numerical choices.
- A public API change updates its docstring, `docs/api.md`, any affected example and
  `CHANGELOG.md` in the same change.

## Writing

- Plain sentences; say the reason simply. One idea per paragraph, few inline numbers.
- State a caveat once, where it matters, and link to the reference rather than retelling it.
- No process vocabulary in published text: no milestones, session names or internal ids.

## Working rules

- `uv` is the only Python runner; never hand-edit `uv.lock`. The portal uses its pinned Node and
  pnpm through `corepack pnpm`.
- Small, cohesive changes. Preserve unrelated work; no destructive Git commands.
- Deterministic tests with fixed seeds and measurable assertions for every behaviour change.
- Do not keep dated reviews, session notes or decision journals in the repository; git history
  is the record.
- Do not commit caches, environments, build output or `site/`.
- Do not push, merge, tag, publish or deploy without the owner's authorization.
- Before handoff, run the handoff gate in `docs/development.md` (or the subset the change
  warrants) and report exactly which checks ran.
