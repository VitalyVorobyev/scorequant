# FORMAL-D-EXCHANGE-GENERAL-D — machine-check the finite D chain in general dimension

**Programme:** P7 (FOUNDATIONS) · **Opened:** 5 September 2026 · **Status:** completed

Bounded verification lane, run alongside the active question rather than in place of it, as
`OPEN_PROBLEMS.md` permits for a formal verification of a frozen claim.

## Goal

Lift the PR #28 pilot from the scalar inequality inside D5 to the whole finite D chain in
arbitrary dimension `d`, and decide honestly which registry claims that lets us mark as
machine-checked.

## Why it matters

`D-EXCHANGE-IMPLIES-VORONOI` is manuscript v9 Theorem 2 and the only contribution the novelty
ledger labels `apparently new` that is pure finite linear algebra. It is load-bearing for the
compiled predictor and for the D7/D8 corollaries, and it rested on human algebra plus one
adversarial audit. Everything else labelled `apparently new` — DS15, DS16, DS17 — needs
almost-sure convergence, uniform laws over VC classes, or atomless first variation, so this is
the one flagship result a proof assistant can reach today.

## Relevant claims

`D-RANK2-MOVE`, `D-LOGDET-GAIN`, `D-LEVERAGE`, `D-EXCHANGE-IMPLIES-VORONOI`,
`D-EXCHANGE-VIOLATION-LOWER-BOUND`, `D-EXCHANGE-SCALAR-CORE`,
`D-GLOBAL-GEOMETRIC-REALIZABILITY`, `D-EXCHANGE-TERMINATES`.

## Outcome

**Proved, in Lean 4.33.1 + Mathlib, for arbitrary `d`.** The chain D2 → D3 → D4 → D5, plus the
D7 and D8 corollaries. The pilot's frozen scalar core plugs in unmodified: its
`ScalarExchangeAssumptions` turned out to be exactly what the matrix layer discharges, with
`q_δ ≤ 1/W_a + 1/W_b` supplied by D4 and the positivity facts by `PosDef`.

Structural points the prose is careful about, and which the formalization reproduces rather than
assumes: distinct centroids are *derived* from stability, and a singleton source needs no move
because its own distance is zero.

**Independent statement audit** (`AUDITS/FORMALIZATION-D-EXCHANGE-IMPLIES-VORONOI-001.md`):
verdict `match after hardening`, no blocking finding. It caught two things worth recording:

1. the frozen file contained the conclusion but neither the hypotheses nor the implication, so a
   prover could have weakened the theorem without touching an audited file — fixed by freezing
   `VoronoiAssumptions` and `ExchangeVoronoiConclusion`;
2. the docstring's claim that the determinant and log-determinant objectives "agree" was false,
   with an explicit witness. The direction chosen was the sound one, but a reader who believed
   the justification might have "simplified" the spec into an unsound weakening.

**Registry marks are deliberately narrower than the Lean tree.** Only
`D-EXCHANGE-IMPLIES-VORONOI`, `D-EXCHANGE-VIOLATION-LOWER-BOUND` and `D-EXCHANGE-SCALAR-CORE`
carry `formal_proof`, because only their statements are separately frozen and audited. D2, D3
and D4 are proved but colocated with their proofs; D7 and D8 are proved only in part — D7's
equal-optimum-value half and D8's "terminates at a stable state" phrasing are not formalized.
`KNOWN_RESULTS/04-d-optimality.md` records each of those, with the declaration name and what is
missing.

## Deliverables

`agenticresearch/formal/ScoreQuantFormal/{Config,Relocation,DetGain,Leverage,ExchangeVoronoiSpec,ExchangeVoronoi,Corollaries,AxiomAudit}.lean`;
the statement audit; `formal_proof` on three claims; the `_check_formal_proofs` validator and its
tests; `protocols/formalization.md`; ADR 0030; the `formal` CI job.

## Stop conditions met

Proved, with the non-coverage stated in the spec docstring, in the audit report and in ADR 0030.

## Next dependency-blocking question

Freeze and audit the D2/D3/D4 statements so `D-RANK2-MOVE`, `D-LOGDET-GAIN` and `D-LEVERAGE` can
carry `formal_proof` — then, under a new ADR, `DS-EXCHANGE-LEVERAGE-BOUND` (DS13), the second
`apparently new` result that is finite algebra.
