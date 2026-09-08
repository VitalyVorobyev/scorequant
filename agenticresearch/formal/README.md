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

Frozen specification files are listed first in each pair; the proof module
follows the slash.

| Module | Result |
| --- | --- |
| `ConfigSpec.lean` / `Config.lean` | weighted sample, cells, `I(z) = Σ_c W_c μ_c μ_cᵀ`; the rank-one update identities |
| `RelocationSpec.lean` / `Relocation.lean` | D2, the exact rank-two relocation identity |
| `DetGainSpec.lean` / `DetGain.lean` | D3, the determinant ratio, via Weinstein–Aronszajn |
| `LeverageSpec.lean` / `Leverage.lean` | D4, both leverage inequalities |
| `ScalarExchangeSpec.lean` / `ScalarExchange.lean` | the frozen scalar core inside D5 |
| `ExchangeVoronoiSpec.lean` / `ExchangeVoronoi.lean` | D5, exchange stability ⇒ strict `I⁻¹`-Voronoi |
| `MergeSpec.lean` / `Merge.lean` | merge invariance: merging duplicate rows preserves masses, sums, centroids and `I` |
| `ClosureSpec.lean` / `Closure.lean` | D6, the compiled nearest-centroid rule and duplicate inheritance |
| `TerminationSpec.lean` / `Termination.lean` | D8, strict ascent, no cycles, termination at a stable state |
| `Corollaries.lean` | D7 realizability, and the earlier weaker D8 fragments |
| `Counterexamples.lean` | two exact boundary witnesses for D5 |
| `AxiomAudit.lean` | a guarded `#print axioms` per exported theorem |

Not covered: profiled `D_s`, population or atomless statements, anything
asymptotic, positive gain tolerances (`D-COMPILE-TOLERANCE-GUARANTEE`), singular
objectives, capacity or balance constraints, and D5's second duplicate branch —
`ClosureDuplicateConclusion` assumes the inherited labeling rather than deriving
that labels are constant on a duplicate class. The converse
`D-VORONOI-NOT-EXCHANGE` is covered only as the explicit witness in
`Counterexamples.lean`, not as a general statement. The termination argument is
generic in the objective, so it also covers §DS3's and §A1's termination
sentences — but neither `F_s` nor `F_A` exists as a Lean object here, so those
claims are recorded in `KNOWN_RESULTS/` prose and carry no marker. Each frozen
spec lists its own non-coverage; ADR 0037 governs the scope, and ADR 0030 the
marking rules.

## Trust policy

- A reviewed `*Spec.lean` file is the informal-to-formal boundary, and freezes
  hypotheses, conclusion **and the implication between them**. A prover may edit
  the corresponding proof module but may not change the specification without a
  new statement audit.
- **A frozen file imports only frozen files.** Definitional closure is not
  enough on its own: a proof module anywhere in a frozen file's import graph can
  introduce an instance, notation or `macro_rules` that changes how an audited
  statement elaborates, without touching an audited file. `ExchangeVoronoiSpec.lean`
  still imports `Leverage.lean` and is the one remaining exception, owed a fix
  under its own audit.
- The freeze is closed under definitional dependency: every definition a frozen
  statement is written in lives in a frozen file too, which is what
  `ConfigSpec.lean` exists for. A frozen conclusion mentioning `fisher` or
  `relocate` guarantees nothing if those are editable — redefining `relocate` as
  the identity would make `ExchangeStable` vacuous without touching an audited
  file.
- Each claim's `formal_proof.declaration` names a theorem whose *type is the
  frozen conclusion*, not a restatement of it, so the mark cannot drift from the
  audited statement.
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
