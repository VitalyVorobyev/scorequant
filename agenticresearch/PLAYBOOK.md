# Running a research session

Choose work from `OPEN_PROBLEMS.md`. A file in `WORK/active/` may be parked; its location is
not authorization to resume it. One session executes one selected packet.

## Next session

```text
Execute agenticresearch/WORK/active/RETENTION-PLUGIN-VECTOR.md in a fresh context.
Check out branch score-oracle-robustness at its tip (the AUDIT-SCORE-ORACLE-ROBUSTNESS
commit). Follow agenticresearch/README.md and protocols/theorem.md: falsify in exact
arithmetic before proving, derive the matrix influence function of the geometric-mean
retention yourself, and stop at proved, reduced or refuted with the O7 section, the
claim nodes and the measured table. No src/ change, no public API. Close with a
plain-English report of what was proved and what remains.
```

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
