# CLAUDE.md

Read `AGENTS.md` first; it is the contract and wins on any conflict. Commands and the handoff
gate are in `docs/development.md`, design principles in `docs/system-design.md`, the plan in
`ROADMAP.md`.

## Subagents

Never spawn `fable` subagents: the owner's plan budget cannot absorb them. Delegate mechanical
work to `sonnet` or `haiku` and reviews to `opus`, and keep the count low. Work that needs the
strongest model is done inline, one section at a time.
