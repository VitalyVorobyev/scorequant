# ADR 0030: A bounded formal-verification track for the finite D chain

**Status:** Accepted

**Extends:** [ADR 0028](0028-focused-research-and-teaching.md) (formal proof work is a bounded
verification lane). **Supersedes:** the unmerged ADR on branch
`codex/formal-verification-pilot`, which carried the number 0024 that main has since given to
[ADR 0024](0024-error-hierarchy-and-fit-pipeline.md). This ADR takes the number reserved for it
by [ADR 0031](0031-portal-reduced-to-four-surfaces.md).

## Context

`D-EXCHANGE-IMPLIES-VORONOI` — manuscript v9 Theorem 2, and the one contribution the novelty
ledger labels `apparently new` that is pure finite linear algebra — was supported by human
algebra, one independent adversarial audit, and exact-rational regression tests. That is the
project's strongest evidence short of a proof assistant, and it is what the theorem's own audit
already delivered.

PR #28 opened a Lean 4.33.1 + Mathlib workspace and machine-checked the *scalar* inequality
inside that theorem: a real-arithmetic bound on six numbers, deliberately stopping before
matrices, determinants, logarithms and the geometric argument. The branch then went 89 commits
stale and collided with main on three points — ADR 0024, the M12 milestone name, and
`registry.py`'s `render_index` — while its prose additions were orphaned by main's rewrites of
`AGENT.md`, `README.md` and `PLAYBOOK.md`.

Two things were worth deciding at once: how to land that work, and how far the Lean track should
reach. The novelty ledger answers the second. Of its eight `apparently new` rows, only two are
finite algebra — Theorem 2 and the profiled leverage bound DS13. The rest (DS15 margins
dichotomy, DS16 price and funnel, DS17 basins) need almost-sure convergence of global optima,
uniform laws over VC classes, and atomless first variation; O6 needs the delta method. Those are
research-grade formalization projects in their own right.

## Decision

**Land the pilot's Lean tree, re-author its integration.** The four Lean source files are taken
verbatim from PR #28. Everything else — the registry contract, the protocol, the CI gate, the
documentation — is written against current main rather than merged. PR #28 is closed as
superseded and credited here and in the pull request that replaces it.

**The approved track is the finite D chain in general dimension `d`,** and nothing beyond it.
Machine-checked in this round: D2 the rank-two relocation identity, D3 the determinant gain, D4
both leverage inequalities, D5 the flagship theorem with its quantitative bound, and the D7 and
D8 corollaries. This widens the pilot's scope, which roadmap M13-E had capped at "stop at
existing pilot"; the widening is what this ADR authorizes.

**Profiled `D_s`, population measure theory and asymptotics stay out** until a further ADR. The
profiled leverage bound DS13 is finite algebra and `apparently new`, so it is the obvious next
round — but `D_s` entering the Lean track is a scope decision, not a prover's discretion.

**A claim carries `formal_proof` only when its statement is separately frozen and independently
audited.** The field names a `*Spec.lean` boundary that a prover may not edit after the audit,
the proof module, the exported declaration, and the audit report. D2, D3 and D4 were initially
unmarked on this rule, their statements living beside their proofs; they have since been frozen
and audited, and now carry the field. D7 and D8 remain unmarked, and not for a bookkeeping
reason — they are only partly covered, D7's equal-optimum-value half and D8's "terminates at a
stable state" phrasing being unformalized. `KNOWN_RESULTS` records what Lean proves in each
case; the registry marker is reserved for audited coverage.

**The freeze is closed under definitional dependency, and the marked declaration's *type* is the
frozen conclusion.** Two rules that the first round left implicit and freezing D2/D3/D4 forced
open. First, every definition a frozen statement is written in must itself live in a frozen file,
which is what `ConfigSpec.lean` exists for: `ExchangeVoronoiSpec.lean` was audited while
`relocate`, `qform` and `fisher` still sat in editable proof modules, so a prover could have
redefined `relocate` as the identity and made `ExchangeStable` vacuous without touching an
audited file. Second, `formal_proof.declaration` names a theorem whose type *is* the frozen
`…Conclusion`, not a restatement of it, so the mark cannot drift from the audited statement while
still compiling.

**Counterexample claims do not carry `formal_proof`.** The field asserts that the claim's own
`statement` is machine-checked, and a counterexample claim's statement is the proposition being
*refuted*. Machine-checked witnesses, when written, are recorded in prose and in the Lean tree.

**The trust gate is a build, an axiom allowlist and an external checker.** CI runs
`lake build --wfail` over a pinned toolchain and manifest, `leanchecker`, and a namespace-wide
axiom audit allowing exactly `propext`, `Classical.choice` and `Quot.sound`. `AxiomAudit.lean`
additionally pins each exported theorem's axiom list with `#guard_msgs`, so drift fails the
build rather than passing silently. Nanoda stays disabled: its parser rejects Lean 4.33.1's
export stream, and the project will not downgrade its kernel to gain a second checker.

## Consequences

The claim graph can now say exactly which statement is machine-checked, under which frozen
specification, audited by whom — and, just as importantly, which ones are not. A Lean build
proves a theorem; it does not prove that the theorem is the one we meant, which is what the
statement audit is for, and it does not prove that `src/` implements it, which nothing here
attempts. The first statement audit demonstrated the value: it found that the frozen file
contained the conclusion but neither the hypotheses nor the implication, so a prover could have
weakened the theorem without touching an audited file, and it found a false justification in the
docstring about the determinant and log-determinant objectives being interchangeable.

The D2/D3/D4 round kept demonstrating it. D3's audit refused to mark a claim named for a
log-determinant gain whose frozen statement contained no logarithm — the fix was to prove the
`F_D` form, not to retitle the claim — and it corrected the node's `assumptions`, which had
demanded a positive definiteness the identity never uses. D2's audit found the formalization
narrower than its claim node, which now records that the destination cell must be nonempty.
D4's audit established that its frozen inequality is true with *no* hypotheses, because Lean's
`0⁻¹ = 0` collapses both sides together at an empty cell and at a singular `I`; the hypotheses
are fidelity conventions, not guards against falsity, and that is worth knowing before someone
reads their presence as evidence of content. The same audit made explicit that nothing formal
ties `fisher` to a statistical Fisher information: it is *defined* as `∑_c m_c m_cᵀ / W_c`, and
the identification is inherited from `FI-QUANT-IDENTITY` rather than proved.

Toolchain upgrades are a dedicated reviewed change: `lean-toolchain`, the Mathlib revision and
`lake-manifest.json` move together or not at all. CI pays for one cached Lean job. Elan and Lake
are needed only to edit or run the formal gate; Python stays `uv`-only.

A proof that fails is not a crisis and not a silent downgrade — it routes back to the ordinary
theorem or audit workflow, exactly as a failed re-derivation would.

## Validation

`lake build --wfail` over the whole tree; `#print axioms` guards on every exported theorem;
`registry.py validate` and `reindex --check`; `tests/test_research_registry.py`, which asserts
that partly-formalized claims stay unmarked and that a dangling declaration is a violation.
