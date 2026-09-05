# ScoreQuant research workspace

Mathematical memory for D- and Ds-optimal hard quantization: claims, proofs, counterexamples
and measured evidence. Claims are atomic; a session addresses one scientific question.

## Canonical read order

1. `PROBLEM.md` — scientific definitions and scope.
2. `AGENT.md` — invariants and claim lookup protocol.
3. The selected `WORK/active/` packet; `OPEN_PROBLEMS.md` alone selects current work.
4. Relevant claims through `uv run python agenticresearch/py/registry.py show <ID> --deps --proof`,
   then only the cited proofs, counterexamples and evidence needed by the packet.
5. The applicable `protocols/{theorem,audit,literature,numerical,algorithm,formalization}.md`.

Read manuscript metadata only for a paper task; do not load manuscript bodies or historical
plans for an ordinary derivation session. `archive/` is history, not current instructions.

## Map

| Path | Authority |
| --- | --- |
| `OPEN_PROBLEMS.md` | Work selection and parked questions |
| `WORK/` | One bounded question per packet |
| `PLAYBOOK.md` | Session launch prompts |
| `claims/`, `registry.json` | Claim status, assumptions and dependencies |
| `KNOWN_RESULTS/` | Proof prose, reached from claim links |
| `COUNTEREXAMPLES/`, `AUDITS/` | Permanent falsification fixtures and independent audits |
| `LITERATURE/`, `../papers/` | Prior art and search record |
| `NUMERICAL_EVIDENCE.md` | Measurements, never theorem authority |
| `manuscripts/README.md` | Paper snapshot metadata |

Indexes are generated; run `uv run python agenticresearch/py/registry.py reindex` after claim
edits and `validate` before handoff. Library tests enforce registry integrity and counterexample
fixtures. Results enter production only through theorem-cited code or deterministic tests.
