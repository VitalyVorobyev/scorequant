# Protocol: publication-grade adversarial audit

Required only when a result becomes load-bearing: promotion toward a novelty
or publication claim, or a guarantee the ScoreQuant library ships. Ordinary
investigations use `protocols/theorem.md` alone.

**Independence rule.** The audit must run in a session that did not produce
the proof (a fresh agent context, whatever the harness). An auditor that
inherits the prover's context inherits the prover's blind spots.

**Exemplar.** `AUDITS/AUDIT-D-EXCHANGE-VORONOI-001.md` is the reference for
size and rigor.

## Output contract (all 16 items, in order)

1. Target statement
2. Criterion and problem level
3. Status before attempt
4. Dependencies (rechecked, not just listed)
5. Nearest literature (per `protocols/literature.md`, recorded in `LITERATURE/`)
6. Counterexample search (per `protocols/numerical.md`)
7. Algebraic reduction
8. Proof / counterexample / conditional result
9. Adversarial audit (each attack from `protocols/theorem.md` §G, with outcome)
10. Algorithmic consequence
11. Deployability consequence
12. Information-loss consequence
13. Updated status
14. Registry patch under `claims/` (including `assumptions` made fully explicit)
15. Counterexample/regression artifact (exact fixture + test) if applicable
16. Closing status line for `OPEN_PROBLEMS.md` (a new question is a backlog
    edit, never a new packet)

## Result of an audit

- The audit report is a permanent artifact in `AUDITS/`.
- The audited claim gains an `audit:` pointer, hardened `assumptions`, and —
  if the prior-art search came back empty — `literature_search_status:
  "search_gap"` (never a novelty claim).
- Boundary failures discovered during the audit become
  `boundary_counterexamples` with fixtures.
- Any statement the manuscript now contradicts is listed under § State in
  `manuscripts/README.md`; a re-attribution to prior art especially must not
  reach a submission unnoticed.
