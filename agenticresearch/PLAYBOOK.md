# Running a research session

Choose work from `OPEN_PROBLEMS.md`. A file in `WORK/active/` may be parked; its location is
not authorization to resume it. One session executes one selected packet.

## Next session

```text
Independent audit of RETENTION-PLUGIN-CLT-FROZEN-VECTOR (O7) and its companions
RETENTION-PLUGIN-SINGULAR-ENDPOINT-RATE and CE-O7-ELLIPSOID-ZERO-VARIANCE-001, in a fresh
context that has not seen the RETENTION-PLUGIN-VECTOR derivation. Inputs: the claim nodes
(registry.py show <ID> --deps --proof), KNOWN_RESULTS/10-oracle.md section O7, the audited
O6 and its audit AUDITS/AUDIT-SCORE-ORACLE-ROBUSTNESS-001.md, the instrument
py/retention_plugin_vector.py and its artifacts under WORK/artifacts/RETENTION-PLUGIN-VECTOR/.
Follow protocols/audit.md. Attack: the phi-route delta method with the everywhere-defined
functional; the three trace identities behind E[psi] = 0 and sum psi_hat = 0; the ellipsoid
characterisation of sigma^2 = 0 (is the "iff" tight, and is the absolutely-continuous
sufficiency correctly stated?); the singular-endpoint rate proposition (the independence of
the null-direction Gaussian blocks, the rank-r projection Lambda, the K >= d condition, and
whether the lower bound's nondegeneracy hypothesis is necessary); the library-agreement
claim including the rank_rtol caveat; and the reading of the heavy-tail under-coverage as
second order (replicate with fresh seeds; probe a bounded law of your own). Verdict per
protocols/audit.md; harden statements rather than re-derive. No src/ change.
```

After the audit, the next research packet is the genuine OP27 remainder: rules refitted on
the evaluation sample (empirical-process argument under a margin condition, or an exact
counterexample to \(\sqrt n\)-normality), targeting `OPEN-RETENTION-UNCERTAINTY`.

## Other session types

- **Independent audit:** a fresh context receives a claim ID and frozen proof/artifacts,
  then follows `protocols/audit.md`. Do not give it the derivation transcript.
- **Literature:** identify one claim or question and follow `protocols/literature.md`.
  A search gap does not prove novelty.
- **Formal statement audit:** a fresh context receives the claim nodes, the prose proof and
  the frozen `*Spec.lean`, and returns `exact match` / `match after hardening` / `mismatch`
  per `protocols/formalization.md`. Never give it the formalizing session's transcript.
- **Formal prover:** may edit the proof module only. No `sorry`, no project axiom, no edit
  to the frozen spec, no claim about `src/`.
- **Bookkeeping:** name the exact registry/document change; do no new mathematics.

Derivation remains with its owner; wide reading can be delegated under `AGENT.md`.
Do not prescribe a model hierarchy or start additional packets automatically.

## Handoff

Record the verdict, changed claim IDs, evidence, limitations and one proposed next action.
After claim edits, regenerate indexes. Validate with:

```bash
uv run python agenticresearch/py/registry.py reindex
uv run python agenticresearch/py/registry.py validate
JAX_ENABLE_X64=1 MPLBACKEND=Agg uv run pytest tests/test_research_claims.py tests/test_research_registry.py
(cd agenticresearch/formal && lake build --wfail)   # only when formal evidence changed
```

Run the contributor checks relevant to other changed files. Promotion to a shipped guarantee
or publication claim requires independent audit. Pushes and merges require owner authorization.
