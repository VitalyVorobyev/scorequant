# Machine-checked proofs

A pinned Lean 4 + Mathlib project that checks the finite D-optimality results behind
ScoreQuant's exchange solver and its `compile_quantizer` guarantee. It verifies the mathematics
as stated; it does not verify that the Python implementation computes the same formulas.

## Build

Install [Elan](https://lean-lang.org/install/) once, then:

```bash
cd research/formal
lake exe cache get
lake build --wfail
```

`lean-toolchain`, the Mathlib revision in `lakefile.lean` and `lake-manifest.json` move together,
in a dedicated change.

## What is proved

Everything is over a finite weighted sample in arbitrary dimension `d`, with
`I(z) = Σ_c W_c μ_c μ_cᵀ` the binned Fisher information of a labelling `z`. Each pair is a
frozen statement (`*Spec.lean`) and its proof. Module docstrings name results by the ids they
had in the research registry (for example `D-EXCHANGE-IMPLIES-VORONOI`); that registry is kept
in git at the `research-archive-2026-10` tag.

| Module | Result |
| --- | --- |
| `ConfigSpec` / `Config` | weighted sample, cells, binned information; the rank-one update identities |
| `RelocationSpec` / `Relocation` | moving one point changes `I` by an exact rank-two update |
| `DetGainSpec` / `DetGain` | the exact determinant ratio of a move (Weinstein–Aronszajn) |
| `LeverageSpec` / `Leverage` | the leverage inequalities used in the main proof |
| `ScalarExchangeSpec` / `ScalarExchange` | the scalar core of the main theorem |
| `ExchangeVoronoiSpec` / `ExchangeVoronoi` | **main theorem:** if no single move raises `log det I`, every point is strictly nearest its own centroid in the `I⁻¹` metric |
| `MergeSpec` / `Merge` | merging duplicate rows preserves masses, centroids and `I` |
| `ClosureSpec` / `Closure` | the compiled nearest-centroid rule reproduces every training label |
| `TerminationSpec` / `Termination` | exact positive-gain exchange strictly ascends and terminates |
| `Corollaries` | every global optimum is such a nearest-centroid rule |
| `Counterexamples` | two exact witnesses showing the main theorem's hypotheses are needed |
| `AxiomAudit` | a guarded `#print axioms` for every exported theorem |

Not covered: nuisance-profiled `D_s`, population or asymptotic statements, positive gain
tolerances, singular objectives, and the duplicate-class branch of the main theorem
(`ClosureDuplicateConclusion` assumes inherited labels rather than deriving them).

## Trust

- A `*Spec.lean` file freezes hypotheses, conclusion and the implication between them; a proof
  module may change, a specification may not. Frozen files import only frozen files, with one
  known exception: `ExchangeVoronoiSpec.lean` imports `Leverage.lean`.
- `lake build --wfail` rejects warnings and unfinished proofs. CI also runs `leanchecker` and an
  axiom audit that allows only `propext`, `Classical.choice` and `Quot.sound`.
- `AxiomAudit.lean` is the root module's only import, so every module it reaches is built.
