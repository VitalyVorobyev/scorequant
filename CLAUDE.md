# CLAUDE.md

Read `AGENTS.md` first: it is the engineering contract (task boundary, numerical invariants,
module ownership, API discipline) and wins on any conflict with this file. Commands, test
tiers, the portal, benchmarks, releases and the handoff gate are in `docs/development.md`;
architecture is in `docs/system-design.md`; durable decisions are `docs/decisions.md`. The plan is `ROADMAP.md`; research is finished.

`uv` is the only Python runner. Run the handoff gate from `docs/development.md` before handoff.

## Subagents

- Never spawn `fable` subagents: the owner's plan budget cannot absorb them. Delegate
  mechanical work to `sonnet` or `haiku` and review work to `opus`, and keep the count low.
  Work that needs the strongest model is done inline by the orchestrating session, one section
  at a time.
