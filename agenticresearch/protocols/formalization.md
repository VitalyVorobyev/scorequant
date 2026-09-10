# Protocol: formalization

Use this to machine-check a claim that is already stable. Formalization is a
verification lane, not a way to settle open mathematics: a claim whose status is
`open`, `conjecture`, `measured` or `counterexample` is not a target.

Scope is set by [ADR 0037](../../docs/decisions.md): the finite theory — any
statement over finitely many weighted rows, finitely many labels and real
matrices. Population measure theory and asymptotics are permanently out of
scope; ADR 0030's marking rules and trust gate stand unchanged.

## A. Select and normalize

Pick one atomic statement. If the claim node bundles several — a realizability
half and a value half, an identity and its corollary — either formalize all of
it or split the node. Write down what the formalization will *not* cover before
writing any Lean.

## B. Freeze the statement

Create `formal/ScoreQuantFormal/<Name>Spec.lean` containing definitions and the
target proposition only, with no proofs. Freeze **hypotheses, conclusion, and
the arrow between them** — as `ScalarExchangeSpec.lean` and
`ExchangeVoronoiSpec.lean` do. A spec that freezes only the conclusion lets a
prover add a hypothesis and weaken the theorem without touching an audited file.

Map the registry `statement` clause by clause in the file's docstring, and give
the non-coverage list a section of its own.

## C. Audit the statement independently

A fresh session with no shared derivation context — given the claim nodes, the
prose proof, the prior audit, the boundary counterexamples and the Lean files,
never the formalizing session's transcript. Verdict is one of `exact match`,
`match after hardening`, `mismatch`. A mismatch blocks proof work.

The audit belongs in `AUDITS/FORMALIZATION-<CLAIM>-NNN.md` and is what
`formal_proof.statement_audit` points at. If the statement changes, patch the
canonical claim node; never let the prover change it silently.

## D. Prove

The prover may edit the proof module and add private lemmas. It may not edit the
frozen spec, use `sorry` or `admit`, introduce a project axiom, or claim the
formal result says anything about the Python/JAX implementation. Iterate with
`lake build --wfail`.

## E. Gate and record

Add a `#guard_msgs`-pinned `#print axioms` for every exported theorem. The
allowlist is exactly `propext`, `Classical.choice`, `Quot.sound`.

Attach `formal_proof` — `system`, `spec`, `file`, `declaration`,
`statement_audit` — **only** where step C actually happened. Lean results whose
statements are not separately frozen and audited, or that cover a claim only in
part, are recorded in `KNOWN_RESULTS/` prose instead, naming the declaration and
saying what is missing. Then `registry.py reindex` and `validate`.

## Escalation

A missing hypothesis, a false statement or an unresolved dependency stops
formalization and opens an ordinary theorem or audit task. Repeated proof-search
stalls are a scoping signal, not a reason to reach for a specialized prover:
AxProver, LeanDojo-v2 and SafeVerify each need their own reviewed decision and
none belongs in the baseline workspace.
