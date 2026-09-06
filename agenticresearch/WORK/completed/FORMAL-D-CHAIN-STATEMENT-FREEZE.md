# FORMAL-D-CHAIN-STATEMENT-FREEZE — freeze and audit D2, D3 and D4

**Programme:** P7 (FOUNDATIONS) · **Opened:** 6 September 2026 · **Status:** completed

The `Next dependency-blocking question` left by `FORMAL-D-EXCHANGE-GENERAL-D`, taken up
directly: the Lean tree proves more than the registry is allowed to say it proves.

## Goal

Make the `formal_proof` marking *complete* — every claim whose statement the Lean tree
actually establishes carries the mark, and every claim that carries it has a separately
frozen, independently audited statement. Concretely: freeze and audit D2, D3 and D4 so
`D-RANK2-MOVE`, `D-LOGDET-GAIN` and `D-LEVERAGE` can carry `formal_proof`.

## Why it matters

Three claims are machine-checked in general dimension but recorded only as prose, because
their statements were colocated with their proofs. That asymmetry is not a safety margin —
it is a bookkeeping gap that makes the ledger understate what has been verified, and it
leaves the D chain's load-bearing algebra resting on prose in the one place a proof
assistant has already reached.

## Known blockers

The freeze is not closed under definitional dependency. `ExchangeVoronoiSpec.lean` is
audited, but the definitions its statement is written in — `relocate` (`Relocation.lean`),
`qform` (`DetGain.lean`), `fisher`, `centroid`, `cellMass`, `mahalanobis` (`Config.lean`) —
all live in modules the prover may edit. Redefining `relocate` as the identity would make
`ExchangeStable` vacuous without touching an audited file. Freezing D2/D3/D4 forces this
open, because the same definitions appear in their conclusions.

## Relevant claims

`D-RANK2-MOVE`, `D-LOGDET-GAIN`, `D-LEVERAGE`. Read-only context: `D-EXCHANGE-SCALAR-CORE`,
`D-EXCHANGE-IMPLIES-VORONOI`, `D-EXCHANGE-VIOLATION-LOWER-BOUND`, `FI-QUANT-IDENTITY`.

## Recommended starting points

`protocols/formalization.md` steps B, C and E. `ScalarExchangeSpec.lean` is the cleanest
precedent: definitions and target propositions only, no proofs, hypotheses and conclusion
and the arrow between them all frozen.

## Required deliverables

A `*Spec.lean` per claim; a frozen definitional layer the specs can be written in; one
exported theorem per claim whose *type is the frozen conclusion*, not a restatement of it;
an independent statement audit per claim in `AUDITS/FORMALIZATION-<CLAIM>-001.md`;
`#print axioms` gates; `formal_proof` on the three claims; the prose in
`KNOWN_RESULTS/04-d-optimality.md` corrected; `registry.py reindex` and `validate` clean.

## Outcome

**Done.** `D-RANK2-MOVE`, `D-LOGDET-GAIN` and `D-LEVERAGE` carry `formal_proof`, against
`RelocationSpec.lean`, `DetGainSpec.lean` and `LeverageSpec.lean` and the declarations
`rank_two_relocation`, `det_relocation_gain` and `leverage_inequality` — each typed as the
frozen conclusion rather than restating it. Four independent audits, all `match after
hardening`: `FORMALIZATION-D-RANK2-MOVE-001`, `-D-LOGDET-GAIN-001` and its second round
`-002`, `-D-LEVERAGE-001`.

The blocker was real and the freeze now closes it. `ConfigSpec.lean` holds the vocabulary the
statements are written in, so `relocate`, `qform`, `fisher`, `centroid`, `cellMass` and
`mahalanobis` are no longer editable by a prover. That retroactively closes the same hole under
the already-audited `ExchangeVoronoiSpec.lean`.

Three things the audits changed rather than confirmed:

1. **D3 was named for a logarithm it did not contain.** The frozen statement was the determinant
   identity alone, while the claim's title and §D3's boxed result are `ΔF_D = log[·]`. Marking it
   would have marked a partly covered claim. The `F_D` form was added to the frozen statement and
   proved, as a second conjunct carrying its own positivity — *added*, not substituted, so the
   identity's hypotheses stayed weak. The node's `assumptions` had asked for a positive
   definiteness the identity never uses, and now record symmetry and nonsingularity.
2. **D2 was narrower than its claim node.** The formalization needs a distinct, nonempty
   destination cell; the node listed only positive weights and a nonempty source. Node patched.
3. **D4's frozen inequality is true with no hypotheses at all** — 10,034 exact-rational instances,
   including 1,832 singular-`fisher` and 2,064 empty-cell cases, zero violations. Lean's
   `0⁻¹ = 0` collapses both sides together, so the hypotheses are fidelity conventions rather than
   guards against falsity. Its `a ≠ b` was dropped so the frozen statement is literally the
   registry statement, and the node gained the `assumptions` field it had never had.

Also recorded, because it is the kind of thing a later reader mistakes for an oversight: nothing
formal ties `fisher` to a statistical Fisher information. It is *defined* as `∑_c m_c m_cᵀ / W_c`;
`Var(E[S∣Z])` equals that only under `E[S] = 0` and normalized weights, neither assumed and
neither statable in this vocabulary. `D-LEVERAGE` is a `bridge` node inheriting it from
`FI-QUANT-IDENTITY`.

## Deliverables

`formal/ScoreQuantFormal/{ConfigSpec,RelocationSpec,DetGainSpec,LeverageSpec}.lean`; the three
frozen-arrow theorems and their `#print axioms` gates; four audit reports; `formal_proof` on three
claims; `assumptions` corrected on all three nodes and `statement` on `D-LOGDET-GAIN`; §D2, §D3
and §D4 of `KNOWN_RESULTS/04-d-optimality.md`; ADR 0030 amended with the definitional-closure and
declaration-type rules.

## Stop conditions met

Four verdicts, no `mismatch`, three marks attached, `registry.py validate` clean.

## Next dependency-blocking question

`DS-EXCHANGE-LEVERAGE-BOUND` (DS13) — the profiled leverage bound, the second `apparently new`
result that is finite algebra. ADR 0030 keeps profiled `D_s` outside the Lean track, so this needs
its own scope decision before any Lean is written, not a prover's discretion.
