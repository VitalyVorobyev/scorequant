# Benchmarks and profiling

Developer tools; nothing here is a runtime promise or published.

- `bench.py`: deterministic seeded timing and quality harness behind `baselines.json` and the
  `benchmarks` CI job.
- `profile.py`: sampling profiler that drives `bench.py`'s own scenario runners, so a profile
  measures exactly the code path the harness measures.

## Running the harness

```bash
# one matrix sweep
JAX_ENABLE_X64=1 uv run python benchmarks/bench.py --rows 20000,100000 --dims 3 --bins 8,64

# a single heavy cell, recorded to JSON
JAX_ENABLE_X64=1 uv run python benchmarks/bench.py \
  --rows 200000 --dims 8 --bins 64 --scenarios d_exchange --repeats 3 --json out.json

# regression check, exactly as CI runs it
JAX_ENABLE_X64=1 uv run python benchmarks/bench.py --check benchmarks/baselines.json \
  --time-tolerance 10 --quality-rtol 1e-6
```

Scenarios: `d_exchange`, `d_exchange_nobatch`, `lloyd`, `kmeans`, `soft`, `scalar_dp`,
`profiled_exchange`, `predict`, `compile`, `certify`. `--max-scans` caps the exchange budget, so a
capped cell is a fixed-work probe rather than a converged fit.

## Refreshing `baselines.json`

The cell list is a curated contract that covers every solver cheaply. Regeneration replays it:

```bash
JAX_ENABLE_X64=1 uv run python benchmarks/bench.py --regenerate benchmarks/baselines.json --repeats 1
```

Use `--repeats 1`: the first call in a process pays JIT compilation, and the harness reports the
minimum over repeats, so a baseline and a check with different repeat counts disagree. Quality
digits are exact for a given seed and code path; a moved quality digit is a numerical change that
needs an explanation, not a new baseline.

## Running the profiler

```bash
JAX_ENABLE_X64=1 uv run python benchmarks/profile.py \
  --scenarios d_exchange --rows 200000 --dims 8 --bins 64 \
  --warmups 1 --rate 200 --json benchmarks/profiles/sampled.json --folded-dir benchmarks/profiles
```

`--warmups N` separates JIT compilation from steady state. The JSON checkpoints after every cell,
so an interrupted run resumes. `*.folded` files render with `flamegraph.pl`, `inferno-flamegraph`
or speedscope.

## What the measurements showed

- At one million rows, eight parameters and 64 bins, one exact exchange scan takes about 0.4 s;
  about 87% of it runs inside one compiled XLA kernel and under 2% is Python dispatch. Porting the
  numerical core to Rust would not pay.
- `certify_partition` is the one hot path without a compiled inner loop (about 40,000 nodes per
  second of Python recursion). It is the only Rust candidate, and it stays deferred.
- Stability and geometry are certified at the solver's `gain_tolerance`, not at zero; at about a
  million rows a few boundary rows sit inside that tolerance, which is why the tolerance is part
  of the contract.
