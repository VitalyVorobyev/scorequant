# CLAUDE.md

Read `AGENTS.md` first: it is the engineering contract (task boundary, numerical invariants,
module ownership, API discipline) and wins on any conflict with this file. Commands, test
tiers, the portal, benchmarks, releases and the handoff gate are in `docs/development.md`;
architecture is in `docs/system-design.md`; the executable plan is `docs/roadmap.md`;
durable decisions are `docs/decisions.md`. The current milestone is M14 (software,
demonstrations, documentation); research is frozen (ADR 0043) and `agenticresearch/README.md`
applies only after an explicit owner reopen.

`uv` is the only Python runner. Run the handoff gate from `docs/development.md` before handoff.

## Subagents

- Never spawn `fable` subagents: the owner's plan budget cannot absorb them. Delegate
  mechanical work to `sonnet` or `haiku` and review work to `opus`, and keep the count low.
  Work that needs the strongest model is done inline by the orchestrating session, one section
  at a time.
