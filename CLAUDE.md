# CLAUDE.md

Read `AGENTS.md` first: it is the engineering contract (task boundary, numerical invariants,
module ownership, API discipline) and wins on any conflict with this file. Commands, test
tiers, the portal, benchmarks, releases and the handoff gate are in `docs/development.md`;
architecture is in `docs/system-design.md`; the executable plan is `docs/roadmap.md`;
durable decisions are `docs/decisions.md`. Research sessions follow `agenticresearch/README.md`.

`uv` is the only Python runner. Run the handoff gate from `docs/development.md` before handoff.

## Subagents

- Never spawn `fable` subagents: the owner's plan budget cannot absorb them (session limit hit
  on 3 September 2026 with two parallel fable writers). Delegate to `opus`, `sonnet` or `haiku`
  and keep the count low. Work that needs the strongest model is done inline by the
  orchestrating session, one section at a time.
