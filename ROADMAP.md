# Roadmap

ScoreQuant is a software product: a library that bins events into a few hard categories while
keeping as much Fisher information as possible. The research behind it is finished and lives
in `research/` as a short paper and machine-checked proofs. What remains is to make the
library, its explanations and its demonstrations good enough to be used and cited.

This file is the only plan. Shipped work is recorded in `CHANGELOG.md`, and earlier plans are
in git history.

## Milestones

| | Milestone | Done when | Status |
| --- | --- | --- | --- |
| M0 | **Lean repository.** Research apparatus, Research Atlas, ADR journal and duplicated narrative removed; MkDocs reduced to reference; examples and tests cut to what the product and its demos use. | The size budget below holds and the handoff gate is green. | Done, except the budget (see below). |
| M1 | **Release 0.3.0.** Version bump, Beta classifier, changelog, `prepare:runtime`, local wheel rehearsal; the owner tags. | 0.3.0 on PyPI. | Done: released 18 Oct 2026. |
| M2 | **Library quality for 1.0.** Dtype-aware tolerances, one singularity rule, chunked memory, `partition.py` split, mixed-gauge guard, API cut to a core plus five submodules. | Every item done with tests; the API inventory in `docs/api.md` is final. | Done; released as 0.4.0, the candidate for 1.0. |
| M3 | **Portal.** Landing, get-started on `fit_quantizer`, four walkthroughs at about half length, the Ideas series, a research page, simpler machinery. | A newcomer can say what the library does, why score space, and what a walkthrough's result means, without help. | Done; needs the owner's read. |
| M4 | **Science in three formats.** The arXiv paper (`research/paper`), the JOSS paper (`research/joss`), the Ideas series. | The paper and the JOSS paper are submitted. | Drafts done and building in CI. The arXiv paper waits until a mathematician on the team has reviewed it. |
| M5 | **Recognition.** PyPI 1.0 and conda-forge; JOSS and arXiv; a Zenodo DOI; a pyhf template-fit demo; a benchmark page against equal-width, quantile and k-means binning; posts to the HEP, ML and cytometry communities. | External users, citations or a talk. | `CITATION.cff`, the pyhf demo and the benchmark page done; outreach is the owner's. |

## What remains

**Owner actions:** read the portal, the benchmark page and the two papers; bring a mathematician
onto the team to review the paper before arXiv; fill the JOSS placeholders (ORCID, affiliation,
AI-use wording, acknowledgements); connect Zenodo to the repository; outreach.

**Engineering:**
- 1.0.0 after the owner's review of the 0.4.0 API and the JOSS review.
- A conda-forge recipe after 1.0.
- Bring `tests/` and `examples/` back inside their budgets: the walkthrough evidence tests and the
  HEP and FlowCyt study packages carry the most weight.

## Size budget

Lines of code per area (Python, TypeScript, CSS, Markdown, Lean), excluding committed data
(JSON evidence, fixtures, generated portal data) and lockfiles. Re-measure at every milestone:
`git ls-files <area> | grep -E '\.(py|ts|tsx|mjs|css|mdx|md|lean|yml)$' | xargs wc -l | tail -1`.

| Area | After M0 | 18 Oct 2026, after M2–M4 | Budget | Holds |
| --- | ---: | ---: | ---: | --- |
| `src/` | 9.6k | 9.8k | 9.5k | the library |
| `tests/` | 9.8k | 10.4k | 8k | library behaviour, counterexample tests, walkthrough evidence, one prose tier |
| `examples/` | 11.5k | 11.7k | 8k | the four walkthrough studies, the Lloyd counterexample, the Ideas toys |
| `docs/` | 1.6k | 1.7k | 2k | user guide, API guide, symbol reference, glossary, development, system design |
| `website/` | 12.6k | 10.8k | 10k | landing, get-started, walkthroughs, Ideas, research page |
| `research/` | 2.9k | 3.3k | 5k | Lean proofs, the arXiv paper, the JOSS paper |
| `benchmarks/` | 1.1k | 1.1k | 1.1k | timing and quality harness, profiler |

Before M0 the repository tracked 935 files and about 160k lines; after it, 322 files and 86k lines
including data. Tests, examples and the website are still over budget; see What remains.

## Not planned

Samplers, streaming aggregation, signed weights, generic profiled compilation, new criteria or
backends, a compiled extension, PyTorch, services. New research needs an explicit owner decision.
