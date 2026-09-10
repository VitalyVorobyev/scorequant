# Protocol: theorem investigation

Use this for any derivation, proof attempt, or open-problem attack. It is a
working method, not an output format; the heavyweight output contract applies
only to publication-critical audits (`protocols/audit.md`).

## A. Normalize the target

Before any mathematics, write down:

- exact statement;
- criterion (D / in-bin \(D_s\) / projected efficient-score D / E / A / general);
- problem level (`finite_assignment`, `empirical_inductive_quantizer`, `population_quantizer`, …);
- decision variable;
- assumptions;
- desired conclusion;
- deployability implication;
- information-loss implication.

## B. Query the claim graph

Query the claim graph with `uv run python agenticresearch/py/registry.py show
<ID> --deps --proof` (see `README.md`). List established
prerequisites, unresolved dependencies, and known counterexamples nearby.
Do not prove something already recorded `project_proved` unless the task is
explicitly an audit.

## C. Prior art

Run the triangulation step of `protocols/literature.md` (3–5 nearest sources,
what transfers / what does not) before investing in a proof.

## D. Falsify before proving

Run the default exact search of `protocols/numerical.md`. If a counterexample
is found, minimize it, serialize it to `COUNTEREXAMPLES/`, and stop the proof
attempt.

## E. Use the strongest known algebra first

For any finite relocation begin with the exact rank-two update

\[
\Delta I=\alpha u_au_a^\top-\beta u_bu_b^\top .
\]

Before expensive exact evaluation of a concave criterion, apply supergradient
screening:

\[
\Delta F\le \alpha u_a^\top Gu_a-\beta u_b^\top Gu_b .
\]

For D use the exact 2×2 determinant ratio. For \(D_s\) track full and nuisance
blocks exactly. For E use exact/safe eigenvalue evaluation only after
screening.

## F. Proof attempt

List lemmas before proving. For every imported theorem verify the feasible-set
assumptions explicitly — the recurring failure mode is transferring a result
from a superficially similar feasible set.

## G. Self-adversarial pass

Attack your own result on:

- strictness and ties;
- singleton/empty cells;
- duplicate scores;
- singular information;
- nuisance singularity;
- atomic laws;
- hidden compactness;
- first-order-to-finite jumps;
- empirical-to-population jumps;
- score-estimation error;
- new-event extension.

## H. Information-loss implication

State whether the result bounds \(\eta_D\) or \(\eta_{D_s}\), the worst
normalized retention eigenvalue, or train-only versus held-out/population
performance.

## Completion

After every investigation:

1. patch the claim node under `claims/` and run `py/registry.py reindex`;
2. add/minimize any counterexample in `COUNTEREXAMPLES/`;
3. update `LITERATURE/` with exact theorem/page metadata for new prior art;
4. state whether `PROBLEM.md` assumptions change;
5. add numerical regression tests if the claim is computational;
6. write the packet's Outcome section and the one status line for
   `OPEN_PROBLEMS.md`; a new question is a backlog edit, never a new packet;
7. note the manuscript impact under § State in `manuscripts/README.md`; the
   paper is harvested from the registry at closure step 5.
