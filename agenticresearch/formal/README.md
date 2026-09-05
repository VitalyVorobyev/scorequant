# ScoreQuant formal proofs

This pinned Lean 4 + Mathlib workspace contains machine-checked proofs for
selected load-bearing claims in the ScoreQuant research registry. It verifies
the stated mathematics; it does not verify that the Python/JAX implementation
implements the same formulas.

## Build

Install [Elan](https://lean-lang.org/install/) once, then run:

```bash
cd agenticresearch/formal
lake exe cache get
lake build --wfail
```

`lean-toolchain`, the Mathlib revision in `lakefile.lean`, and
`lake-manifest.json` are committed together. Upgrade them only in a dedicated
reviewed change, keeping Lean and Mathlib on the same stable release.

## What is covered

The finite D chain of `KNOWN_RESULTS/04-d-optimality.md`, in arbitrary
dimension `d`:

| Module | Result |
| --- | --- |
| `Config.lean` | weighted sample, cells, `I(z) = Σ_c W_c μ_c μ_cᵀ` |
| `Relocation.lean` | D2, the exact rank-two relocation identity |
| `DetGain.lean` | D3, the determinant ratio, via Weinstein–Aronszajn |
| `Leverage.lean` | D4, both leverage inequalities |
| `ScalarExchangeSpec.lean` / `ScalarExchange.lean` | the frozen scalar core inside D5 |
| `ExchangeVoronoiSpec.lean` / `ExchangeVoronoi.lean` | D5, exchange stability ⇒ strict `I⁻¹`-Voronoi |
| `Corollaries.lean` | D7 realizability, D8 termination |
| `Counterexamples.lean` | two exact boundary witnesses for D5 |
| `AxiomAudit.lean` | a guarded `#print axioms` per exported theorem |

Not covered: profiled `D_s`, population or atomless statements, anything
asymptotic, positive gain tolerances, singular objectives, capacity or balance
constraints, and the compiled predictor `D-FINITE-INDUCTIVE-CLOSURE`. The
converse `D-VORONOI-NOT-EXCHANGE` is covered only as the explicit witness in
`Counterexamples.lean`, not as a general statement. Each frozen spec lists its
own non-coverage; ADR 0030 governs the scope.

## Trust policy

- A reviewed `*Spec.lean` file is the informal-to-formal boundary, and freezes
  hypotheses, conclusion **and the implication between them**. A prover may edit
  the corresponding proof module but may not change the specification without a
  new statement audit.
- Exported theorems have a `#guard_msgs` axiom audit. The current allowlist is
  Lean's standard `propext`, `Classical.choice`, and `Quot.sound`; `sorryAx` and
  project-defined axioms are forbidden.
- `lake build --wfail` rejects warnings, including unfinished proofs, and CI
  additionally checks the full namespace with `axiom-audit` and the bundled
  `leanchecker`.
- `AxiomAudit.lean` is the root module's only import, so every module reachable
  from it is in the default build target. A new module that nothing guards is a
  module CI does not build.
- Formal proof metadata belongs to the canonical claim node under `claims/`,
  and only where the statement is separately frozen and audited. The Lean tree
  is evidence, not a second claim registry.

Nanoda is intentionally not a gate: its current parser rejects modern Lean
export streams (`leanprover/lean-action#169`). It may be added after upstream
compatibility is restored; the project does not downgrade Lean to obtain it.
