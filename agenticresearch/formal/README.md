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

## Trust policy

- A reviewed `*Spec.lean` file is the informal-to-formal boundary. A prover may
  edit the corresponding proof module but may not change the specification
  without a new statement audit.
- Exported theorems have a `#guard_msgs` axiom audit. The current allowlist is
  Lean's standard `propext`, `Classical.choice`, and `Quot.sound`; `sorryAx` and
  project-defined axioms are forbidden.
- `lake build --wfail` rejects warnings, including unfinished proofs, and CI
  additionally checks the full namespace with `axiom-audit` and the bundled
  `leanchecker`.
- Formal proof metadata belongs to the canonical claim node under `claims/`.
  The Lean tree is evidence, not a second claim registry.

The pilot covers only the real-arithmetic core of the finite D-exchange lower
bound. It does not yet cover matrices, determinants, logarithms, strict
Voronoi geometry, singleton handling, or duplicate atoms.

Nanoda is intentionally not a gate yet: its current parser rejects modern Lean
export streams (`leanprover/lean-action#169`). It may be added after upstream
compatibility is restored; the project does not downgrade Lean to obtain it.
