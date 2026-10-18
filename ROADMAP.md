# Roadmap

ScoreQuant is a software product: a library that bins events into a few hard categories while
keeping as much Fisher information as possible. The research behind it is finished and lives
in `research/` as a short paper and machine-checked proofs. What remains is to make the
library, its explanations and its demonstrations good enough to be used and cited.

This file is the only plan. Shipped work is recorded in `CHANGELOG.md`, and earlier plans are
in git history.

## Milestones

| | Milestone | Done when |
| --- | --- | --- |
| M0 | **Lean repository.** Research apparatus, Research Atlas, ADR journal and duplicated narrative removed; MkDocs reduced to reference; examples and tests cut to what the product and its demos use. | The size budget below holds and the handoff gate is green. |
| M1 | **Release 0.3.0.** Version bump, Beta classifier, changelog, `prepare:runtime`, local wheel rehearsal; the owner tags. | 0.3.0 on PyPI, with `retention_uncertainty`, backend-free artifacts and the float32 warning. |
| M2 | **Library quality for 1.0.** See the list below. | Every item done with tests; the API inventory in `docs/api.md` is final. |
| M3 | **Portal: readable, illustrative, instructive.** A landing page with a picture and a live demo above the fold; get-started on `fit_quantizer`; the four walkthroughs rewritten at about half their length; the Ideas series; one research page; simpler machinery. | A newcomer can say what the library does, why score space, and what a walkthrough's result means, without help. |
| M4 | **Science in three formats.** A ~10-page arXiv paper on the D theorem and its limits; a JOSS software paper; the Ideas series as the accessible version. | The paper and the JOSS paper are submitted. |
| M5 | **Recognition.** PyPI 1.0 and conda-forge; JOSS and arXiv; `CITATION.cff` and a Zenodo DOI; a pyhf template-fit demo for HEP; posts to the HEP, ML and cytometry communities; a benchmark page against equal-width, quantile and k-means binning. | External users, citations or a talk. |

### M2 items

- **Precision.** Make `gain_tolerance` and the rank thresholds scale with dtype, so float32 runs
  mean something; today the fallback only warns.
- **One singularity rule.** Replace the `slogdet`-sign and NumPy-default rank tests in
  `information.py` and `solvers/soft.py` with the shared threshold. Rebuild exact cell
  statistics before reporting `objective`.
- **Memory at scale.**
  - Chunk the `[N, B, P]` profiled geometry in `partition.py` and the soft-solver tensors.
  - Compile the soft gradient.
  - Raise when the soft solver stalls on a singular Fisher matrix.
- **Split `partition.py`.**
  - Exchange and Lloyd move to `solvers/`, the exchange algebra to a private module, and the
    geometry reports to diagnostics.
  - Merge the three nearest-centre assignment routines and the repeated nuisance validation.
- **Mixed-gauge ratios.** The HEP pattern (a classifier signal ratio plus a Monte Carlo or
  analytic background density) is correct only when every ratio column shares one gauge.
  Add a guard at the provider boundary and a worked example.
- **API before the freeze.** Cut the 62 exported names to a core plus submodules:
  - one route each for linear models and ratios;
  - the ratio and provider stack moved to a submodule;
  - `plot_*` marked non-stable;
  - one naming scheme for iteration caps and tolerances;
  - `ScalarDPConfig.seed` removed.

### M3 Ideas series

One idea and one live demo per post:
1. Bin by score, not by feature.
2. Why k-means is the wrong clustering for D.
3. When Lloyd goes backwards.
4. Nuisance parameters break the rule.
5. How much did my binning keep?

## Size budget

Lines of code per area (Python, TypeScript, CSS, Markdown, Lean), excluding committed data
(JSON evidence, fixtures, generated portal data) and lockfiles. Re-measure at every milestone:
`git ls-files <area> | grep -E '\.(py|ts|tsx|mjs|css|mdx|md|lean|yml)$' | xargs wc -l | tail -1`.

| Area | 18 Oct 2026, after M0 | Budget | Holds |
| --- | ---: | ---: | --- |
| `src/` | 9.9k | 9.5k | the library |
| `tests/` | 9.8k | 8k | library behaviour, counterexample tests, walkthrough evidence, one prose tier |
| `examples/` | 11.5k | 8k | the four walkthrough studies and the Lloyd counterexample |
| `docs/` | 1.6k | 2k | user guide, API guide, symbol reference, glossary, development, system design |
| `website/` | 10.8k in M3 | 10k | landing, get-started, walkthroughs, Ideas, research page |
| `research/` | 2.9k | 5k | Lean proofs now; the paper and the JOSS paper in M4 |
| `benchmarks/` | 1.1k | 1.1k | timing and quality harness, profiler |

Before M0 the repository tracked 935 files and about 160k lines; after it, 322 files and 86k lines
including data. The examples and website budgets are met by the M3 walkthrough rewrite.

## Not planned

Samplers, streaming aggregation, signed weights, generic profiled compilation, new criteria or
backends, a compiled extension, PyTorch, services. New research needs an explicit owner decision.
