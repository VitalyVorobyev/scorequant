# Contributing to ScoreQuant

Thank you for your interest. Bug reports, documentation fixes and new examples are the most
useful contributions. The rules every change follows are in [`AGENTS.md`](AGENTS.md); the
design principles are in [`docs/system-design.md`](docs/system-design.md).

## Set up

ScoreQuant uses [`uv`](https://docs.astral.sh/uv/) for every Python command. From a clone of the
repository:

```bash
uv sync --all-extras --all-groups --locked
```

Python 3.12 or newer is required. Set `JAX_ENABLE_X64=1` when running tests or examples.

## Run the checks

[`docs/development.md`](docs/development.md) lists every command: the library tests, the prose
tier, the float32 leg, the documentation site, the portal and the handoff gate a change must pass
before review. Run the subset your change touches, and say in the pull request which checks ran.

## Report an issue

Open an issue at <https://github.com/VitalyVorobyev/scorequant/issues>. Please include the
ScoreQuant, Python and JAX versions, a minimal script that reproduces the problem with a fixed
seed, and what you expected instead. For a wrong number, say which quantity and why you think it
is wrong.

## What is welcome

- Bug fixes, each with a deterministic test that fails before the fix.
- Corrections and clarifications to the documentation and the walkthroughs.
- New examples that apply the library to a real inference problem.
- Performance improvements that keep results unchanged and come with a benchmark.

Small, focused pull requests are easiest to review. A change to the public API also updates its
docstring, `docs/api.md` and `CHANGELOG.md`.

## What is out of scope

The following are not planned: samplers, streaming aggregation, signed weights, generic
profiled compilation, new criteria or backends, a compiled extension, PyTorch support, and
services. Please open an issue to discuss before starting work in these areas.

## Code of conduct

Participants are expected to follow the
[Contributor Covenant, version 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Report unacceptable behaviour to the maintainer at vit.vorobiev@gmail.com.
