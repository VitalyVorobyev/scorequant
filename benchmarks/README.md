# ScoreQuant benchmarks and profiling

Developer notes. Nothing here is a runtime promise, and none of it is published to the
documentation site. `docs/development.md` carries the short summary; this file carries the
measurements behind it.

Two tools live here:

- `bench.py` — deterministic seeded timing and quality harness. It backs `baselines.json` and
  the `benchmarks` CI job.
- `profile.py` — sampling profiler that drives `bench.py`'s own scenario runners, so a profile
  always measures exactly the code path the harness measures.

---

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
`profiled_exchange`, `predict`, `compile`, `certify`.

`--max-scans` caps the exchange scan budget. A capped cell is a fixed-work steady-state probe,
not a converged fit; it is how the per-scan costs below were measured at row counts whose full
convergence exceeds a single measurement window.

### Refreshing `baselines.json`

The cell list in `baselines.json` is a curated contract, not a matrix product: it covers every
solver while staying cheap enough for a shared CI runner. Regeneration replays that recorded
list rather than rebuilding a matrix from flags, so a timing refresh can never silently change
which scenarios CI checks:

```bash
JAX_ENABLE_X64=1 uv run python benchmarks/bench.py --regenerate benchmarks/baselines.json --repeats 1
```

**Use `--repeats 1`.** The baseline and the check must agree on repeat count, because the first
call in a process pays JIT compilation and `_run_timed` reports the minimum over repeats. A
baseline recorded with `--repeats 3` and checked with `--repeats 1` reports spurious 5–50x
slowdowns on the cells whose compilation dominates.

Review the diff before committing. Quality metrics are exact for a given seed and code path, so
any quality digit that moves is a real numerical change and needs an explanation, not a new
baseline.

## Running the profiler

```bash
JAX_ENABLE_X64=1 uv run python benchmarks/profile.py \
  --scenarios d_exchange --rows 200000 --dims 8 --bins 64 \
  --warmups 1 --rate 200 --json benchmarks/profiles/sampled.json --folded-dir benchmarks/profiles
```

`--warmups N` runs the cell N times untimed and unsampled first, which separates JIT compilation
from steady state. Results checkpoint to `--json` after every cell, so an interrupted campaign
resumes instead of repeating finished work.

Artifacts under `profiles/`:

- `*.folded` — aggregated folded stacks. Render with `flamegraph.pl`, `inferno-flamegraph`, or
  by dropping the file into speedscope. Runs of one repeated frame are collapsed to a single
  frame, because recursion depth is not what these profiles are read for and keeping it makes
  nearly every sample of a recursive search a unique stack.
- `sampled.json` — elapsed seconds plus top self-time and cumulative-time frames per cell.
- `matrix.json` — the full timing/quality campaign behind the tables below.

### Methodology and its limits

- Machine: Apple M4 Pro, 12 cores, 24 GB, Darwin 25.5.0. Python 3.13.12, JAX 0.11.1, CPU
  backend, `JAX_ENABLE_X64=1`. Every number below is from this machine.
- Fixed seed 2026 throughout. `--repeats 3` at 20 000 and 100 000 rows, `--repeats 2` for the
  capped 10^6-row probes, `--repeats 1` for the 200 000-row cells, whose single-run noise is
  small relative to their runtime. `_run_timed` reports the minimum.
- Both profilers are statistical. A frame's share is a sample count, not a duration, and time
  inside an XLA kernel is attributed to the Python frame that dispatched it — which is exactly
  what makes `_scan`'s self time readable as kernel time, since that is where `np.asarray`
  blocks on the device result.
- The sampler thread must take the GIL to read the main thread's frames, so it sets the
  interpreter switch interval to 1 ms. Frames that release the GIL are still mildly
  over-sampled relative to pure-Python frames; the phase decomposition below is the
  cross-check, and it agrees with the sampled attribution to within a few percent.
- **`py-spy` is pinned in the `dev` group and is the better tool on Linux**, where
  `uv run py-spy record --format speedscope -o out.json -- python benchmarks/bench.py ...`
  works directly. On macOS it requires root (`This program requires root on OSX`), which is why
  the in-repo sampler exists; it needs no elevated permissions and attributes to the same
  frames.
- `jax.profiler` traces were not used. They need TensorBoard or Perfetto tooling that is not in
  this dev environment, and the question they would answer — is the kernel near what the
  hardware can do — is answered directly and more usefully by the roofline measurement below.

---

## Measured machine roofline

Same JAX, same XLA:CPU backend, float64, measured by running each kernel in a loop and
comparing `ru_utime + ru_stime` against wall time:

| workload | time | rate | cpu/wall parallelism |
| --- | ---: | ---: | ---: |
| f64 matmul 1024x1024 | 4.75 ms | 452 GFLOP/s | 9.06x |
| f64 matmul 2048x2048 | 39.52 ms | 435 GFLOP/s | 9.79x |
| f64 elementwise stream, 32M | 2.21 ms | 231 GB/s | 3.92x |

This is the reference the exchange kernel is judged against. XLA:CPU reaches 452 GFLOP/s on
well-shaped float64 work on this machine, and saturates memory at 231 GB/s.

---

## Bottleneck table

### Where a whole solver run spends its time

Sampled self time, final code, `--warmups 0` (so the JIT-compile row is visible). Frames below
1% are omitted; full data in `profiles/sampled.json`.

| solver | cell | wall | dominant frames (self %) |
| --- | --- | ---: | --- |
| `d_exchange` | N=2e5, R=8, B=64 | 95.6 s | `partition:_scan` 69.1, `quantizers:_single_kmeans` 11.0, `jax:apply_primitive` 7.7, `numpy:_wrapfunc` (argmax) 6.8, `jax:backend_compile_and_load` 2.4 |
| `lloyd` | N=2e5, R=8, B=64 | 78.1 s | `partition:_scan` 60.7, `quantizers:_single_kmeans` 14.2, `numpy:_wrapfunc` 7.8, `partition:_assign_nearest` 7.5, `jax:apply_primitive` 6.6 |
| `d_exchange` | N=1e6, R=8, B=64, 63 scans | 85.6 s | `quantizers:_single_kmeans` 59.1, `partition:_scan` 26.8, `numpy:_wrapfunc` 3.8, `jax:apply_primitive` 3.2, `jax:backend_compile_and_load` 2.7 |
| `kmeans` | N=1e6, R=3, B=8 | 6.0 s | `quantizers:_single_kmeans` 90.6, `quantizers:weighted_kmeans` 3.2, `_weighted_kmeans_plus_plus` 1.8, `jax:apply_primitive` 1.7 |
| `soft` | N=1e6, R=3, B=8 | 9.9 s | `quantizers:_single_kmeans` 59.4, `jax:apply_primitive` 14.0, `jax:ufunc_api.__call__` 6.9, `jax:reductions.sum` 5.5, `jax:_operator_matmul` 2.8, `_SoftHistory.append` 1.5 |
| `profiled_exchange` | N=2e4, R=3, B=8 | 3.8 s | `partition:_scan` 32.7, `jax:apply_primitive` 15.0, `numpy:_wrapfunc` 7.8, `jax:_operator_lt` 6.0, `jax:ufunc_api.__call__` 5.3, `jax:squeeze` 5.3, `_single_kmeans` 4.3, `partition:_rank_two_block` 2.4, `partition:init_state` 2.2 |
| `scalar_dp` | N=2e4, R=1, B=8 | 9.1 s | `quantizers:scalar_interval_dp` 87.9, `numpy:_wrapfunc` 11.8 |
| `certify` | 40 atoms, R=2, B=3 | 18.5 s | `numpy:outer` 23.9, `certify:_partial_information` 18.6, `numpy:slogdet` 14.7, `certify:_bound` 11.6, `certify:_explore` 6.9, `numpy:_outer_dispatcher` 6.0, `certify:_assign` 4.6, `numpy:zeros_like` 4.5 |

Cumulative attribution for the reference D cell (N=2e5, R=8, B=64) splits the run cleanly:

| stage | cumulative % |
| --- | ---: |
| candidate scans (`_scan`) | 83.5 |
| k-means++ initialization (`initializer_restarts=8`) | 12.4 |
| batch acceptance, relocation, rank-two updates | 2.5 |
| result assembly, reports, and validation | 1.6 |

### Where one scan spends its time

Directly instrumented, not sampled: each phase is timed separately with
`jax.block_until_ready`, and JAX dispatch is isolated by timing the same compiled kernel on a
one-row input where the arithmetic is negligible.

| phase | N=1e6, R=8, B=64 | N=1e6, R=3, B=8 |
| --- | ---: | ---: |
| chunk rows / chunks per scan | 10 485 / 96 | 123 361 / 9 |
| jitted candidate-gain kernel | 3.286 ms (88.3%) | 2.809 ms (71.4%) |
| — XLA compute | 3.226 ms (**86.7%**) | 2.761 ms (**70.2%**) |
| — JAX Python dispatch | 0.060 ms (**1.6%**) | 0.047 ms (**1.2%**) |
| host row slice | 0.066 ms (1.8%) | 0.132 ms (3.4%) |
| `np.asarray` materialization | 0.000 ms (0.0%) | 0.000 ms (0.0%) |
| host NumPy argmax + gather | 0.368 ms (9.9%) | 0.995 ms (25.3%) |
| **modelled scan** | **357 ms** | **35.4 ms** |

`np.asarray` is free because a CPU-backend JAX array is already host memory; there is no
transfer to recover.

Measured against the model, by capping the scan budget and differencing (the cap also subtracts
initialization exactly):

| cell | 3 scans | 63 scans | marginal | modelled |
| --- | ---: | ---: | ---: | ---: |
| N=1e6, R=8, B=64 | 57.03 s | 79.78 s | **379 ms/scan** | 357 ms |
| N=1e6, R=3, B=8 | 5.51 s | 8.03 s | **42.0 ms/scan** | 35.4 ms |

An independent earlier repeat of the heavy cell gave 363 ms/scan, so read that figure as
370 ms ±5%. The model accounts for 94% of it; the remainder is batch acceptance, whose exact
rebuild runs once or more per scan.

### Scaling

| solver | N=2e4 | N=1e5 | N=2e5 | N=1e6 |
| --- | ---: | ---: | ---: | ---: |
| `d_exchange` (R=8, B=64) | 2.83 s / 159 scans | 17.60 s / 300 | 92.57 s / 1008 | 379 ms per scan (not converged; see below) |
| `lloyd` (R=8, B=64) | 2.80 s | 28.60 s | 73.05 s | — |
| `kmeans` (R=8, B=64) | 1.51 s | 5.93 s | 11.90 s | — |
| `soft` (R=8, B=64) | 2.30 s | 9.62 s | 19.99 s | — |
| `predict` (R=8, B=64) | 2.9 ms | 12.9 ms | 24.2 ms | — |
| `compile` (R=8, B=64) | 6.7 ms | 30.4 ms | 117.6 ms | — |
| `kmeans` (R=3, B=8) | 0.23 s | 0.70 s | — | 5.76 s |
| `soft` (R=3, B=8) | 0.40 s | 1.07 s | — | 8.79 s |
| `predict` (R=3, B=8) | 0.4 ms | 0.9 ms | — | 8.7 ms |

The exchange grows superlinearly and irregularly, because per-scan cost is linear in N *and* the
scan count climbs with it: 159 → 300 → 1008 scans over 20k → 100k → 200k rows, giving 32.7x more
wall time for 10x more rows and 5.3x more for the last doubling alone. Do not fit an exponent to
three points; read it as "the scan count is the term that hurts".

The reusable paths do not behave that way. `predict` is linear and stays in the tens of
milliseconds — 8.7 ms at 10^6 rows with R=3, B=8. `compile` is the same shape of work plus one
verification pass over the training rows: 117.6 ms at 2e5 rows with R=8, B=64. Neither was
measurable at 10^6 rows with R=8, B=64, since both need a converged exchange first.

Guarded batch acceptance is the reason the exchange is usable at all:

| cell | batch | single move | ratio |
| --- | ---: | ---: | ---: |
| N=2e4, R=3, B=8 | 0.31 s / 70 scans | 3.17 s / 1757 scans | **10.3x** |
| N=2e4, R=8, B=64 | 2.83 s / 159 scans | 24.91 s / 2741 scans | **8.8x** |

Batching also lands a marginally better objective in both cells (-1.305884 vs -1.305882;
-4.471085 vs -4.472833), so this is not a speed-for-quality trade.

### Memory

Peak RSS is process-lifetime, so treat it as a ceiling. The one path worth flagging: `soft` at
N=2e5, R=8, B=64 pushed peak RSS to 5.06 GB, against 0.79 GB for every other solver at that
shape. The soft objective differentiates through a dense `[N, n_bins]` responsibility matrix
(102 MB per copy at that shape) and autodiff retains several. `soft` is the memory-limiting
path at large N x large `n_bins`, not the exchange.

---

## Rust go/no-go

**Verdict: no-go for the numerical core. Go — but deferred — for `certify.py` only.**

### The numerical core: no

At N=10^6, B=64, R=8, one D-exchange scan is **379 ms** steady state, of which:

- **86.7% inside a single jitted XLA kernel**,
- 9.9% inside NumPy's compiled `argmax` and fancy-index gather,
- 1.8% host array slicing,
- **1.6% JAX Python dispatch**.

Across a whole converged run (N=2e5, R=8, B=64), scans are 83.5% of the time and the k-means++
initialization — itself a jitted Lloyd scan — is another 12.4%. That leaves 2.5% in batch
acceptance and rank-two updates and 1.6% in result assembly, both of which are eager JAX calls on
rank-sized matrices and so are mostly dispatch. Counting all of those against Python and adding
the 1.3% of dispatch inside the scans, **Python-interpreter orchestration is at most about 4%** of
the run.

Amdahl, stated plainly: a Rust port that made Python orchestration *entirely free* would recover
4%, a 1.04x speedup. That is the whole prize on offer for the usual reason a port is proposed.

The kernel is nevertheless far from the hardware:

| | achieved | measured roofline | fraction |
| --- | ---: | ---: | ---: |
| float64 throughput | 32.7 GFLOP/s | 452 GFLOP/s | **7.2%** |
| residual-tensor traffic | 38.7 GB/s | 231 GB/s | **17%** |
| cpu/wall parallelism | 6.36x | 9.1–9.8x | 65–70% |

So there is real headroom — roughly 3–5x — but it is a **formulation** problem, not a language
problem. The gain kernel evaluates `q_destination` as a batched `[chunk, B, R] x [R, R]`
contraction, which is `chunk*B*R^2` multiply-adds of low arithmetic intensity that XLA emits as
a fused loop rather than a BLAS-3 call. Expanding the same quadratic form as
`x^T I x - 2 x^T I m + m^T I m` turns the bin axis into one GEMM and cuts the flop count about
sevenfold. Measured on the same chunk: **0.953 ms against 3.31 ms, a 3.5x speedup, still in
JAX**. A Rust port of the badly shaped contraction would land where XLA already is; beating a
452 GFLOP/s BLAS-3 path with hand-written Rust is not a small effort, and the reformulation is
available today without leaving the language.

Explicit gates for reopening this:

1. **Throughput gate.** If a converged D-exchange at N=10^6, B=64, R<=8 must finish under 60 s,
   the levers in order are (a) the algebraic reformulation above, worth a measured 3.5x on the
   kernel, and (b) the k-means++ initialization, which is 12% of the converged 200 000-row run
   and 59% of a scan-bounded 10^6-row run at `initializer_restarts=8`. A port becomes the next question only
   if both are done and the target is still missed.
2. **Orchestration gate.** A port of a numerical path is justified when Python orchestration
   exceeds roughly 25% of steady state. Measured today: 1.6% of a scan and <=4% of a run — a
   factor of six below the threshold.
3. `AGENTS.md` forbids a second numerical backend without an approved roadmap change and a
   concrete second backend. Neither gate above supplies one.

### `certify.py`: the one genuine candidate

`certify_partition`'s branch-and-bound search is the only hot path in the library with no JAX
kernel in its inner loop. It is pure Python recursion over tiny NumPy calls, and it measures
exactly that way:

| atoms | nodes | wall | nodes/s | status |
| ---: | ---: | ---: | ---: | --- |
| 24 | 7 631 | 0.21 s | 37 000 | optimal |
| 32 | 26 809 | 0.69 s | 39 000 | optimal |
| 40 | 683 241 | 17.53 s | 39 000 | optimal |
| 48 | 2 000 001 | 50.41 s | 39 700 | budget exhausted |

The node rate is **flat at ~34–40k nodes/s across a 260x range of tree sizes**: it is set by the
interpreter, not by the data. 26 microseconds per node, for a node whose actual arithmetic is a
few hundred flops. The attribution says the same thing:

- `np.outer` on a 2-vector — **23.9%**
- `np.linalg.slogdet` on a 2x2 matrix — **14.7%**
- NumPy's own `__array_function__` dispatchers (`_outer_dispatcher`, `_unary_dispatcher`,
  `_assert_stacked_square`) — **9.5%**
- `np.zeros_like` allocating one rank-sized zero matrix per node — **4.5%**

**Scope of a Rust module**, precisely: the `_Search` class (`__init__`'s suffix moments,
`_explore`, `_bound`, `_partial_information`, `_assign`, `_restore`) and `_labels_objective` —
about 250 lines, fixed-size R x R linear algebra with `R <= 8` and `n_bins` small, no JAX, no
autodiff, no arrays larger than `max_rows <= 512`. Nothing else would move.

**Expected bound**: a node reduces to one R x R symmetric rank-one update, a running-sum matrix
add over at most `n_bins` cells, and one Cholesky log-determinant — 200–600 ns on this machine
against the measured 26 microseconds. That is **40–130x**, and the conservative claim is
**>=40x**. In user terms, the same 2 000 000-node budget that stalls at 48 atoms today would
reach roughly 80M nodes, buying perhaps six to eight more atoms of exact certification depth
(the tree grows like Stirling numbers of the second kind, so the payoff in atoms is
logarithmic in the rate).

**Cost**: a compiled extension in the wheel, a build matrix across macOS/Linux/Windows and
Python versions, a PyO3/maturin toolchain, and a pure-Python fallback so a source install still
works. That is a large change to the distribution story for one bounded diagnostic.

**Recommendation: defer.** Reconsider when certifying more than ~48 atoms becomes a user-facing
requirement. Before then there is a cheaper step with no new toolchain: the profile indicates
that replacing `np.outer`, `np.zeros_like`, and `np.linalg.slogdet` with preallocated buffers
and a hand-rolled R x R determinant would remove a large part of the ~38% currently spent in
NumPy allocation and dispatch. That figure is read off the attribution, not measured on a
prototype, so treat it as an indication rather than a promise.

---

## Optimizations applied

Three changes, all verified bit-identical: every quality metric, objective, `accepted_moves`,
`scans`, and `best_remaining_gain` in `baselines.json` is unchanged to the last recorded digit,
and `tests/test_golden_engine.py` passes with its frozen label lists intact.

1. **Compile the candidate-gain kernel** (`partition._d_chunk_gains`,
   `partition._profiled_chunk_gains`). The per-chunk gain evaluation was a chain of ~25 eagerly
   dispatched JAX primitives that materialized a `[chunk, n_bins, rank]` residual tensor —
   43 MB per chunk at N=2e5, R=8, B=64. Compiled, XLA fuses the residual into the contractions
   that consume it. Measured on one chunk: **5.57 ms -> 3.31 ms, 1.68x, output bit-identical**
   (`np.array_equal` true, zero `argmax` changes).
2. **Stop building a discarded Schur complement.** `_ProfiledDObjective.init_state` called
   `_profiled_blocks`, which computes the Schur complement via a `solve` and a matmul, and then
   threw it away — the profiled exchange's gains telescope with the difference of two log
   determinants and never need it. It also factored the nuisance block twice. Split out
   `information._nuisance_information`, which keeps the nonsingularity guard and its error
   message and returns the log determinant the caller was recomputing.
3. **Compile the exact state rebuild** (`partition._cell_information_matrices`,
   `information._nuisance_block_slogdet`). Guarded batch acceptance rebuilds the criterion state
   from scratch on every candidate batch, so this runs more often than a move does; on a
   rank-sized matrix each primitive costs more to dispatch than to execute.

Before and after, same machine, same seeds, `--repeats 1` on both sides:

| cell | before | after | speedup |
| --- | ---: | ---: | ---: |
| `profiled_exchange` N=2e4, R=3, B=8 | 8.230 s | 4.294 s | **1.92x** |
| `d_exchange` N=2e4, R=3, B=8 | 2.773 s | 1.665 s | **1.67x** |
| `d_exchange` N=1e5, R=3, B=8 | 2.888 s | 2.066 s | **1.40x** |
| `d_exchange` N=2e5, R=8, B=64 | 129.06 s | 92.57 s | **1.39x** |
| `lloyd` N=2e5, R=8, B=64 | 99.16 s | 73.05 s | **1.36x** |
| `lloyd` N=2e4, R=3, B=8 | 0.525 s | 0.402 s | 1.31x |
| `soft` N=2e4, R=3, B=8 | 0.804 s | 0.746 s | 1.08x |
| `kmeans` N=2e4, R=3, B=8 (control) | 0.525 s | 0.507 s | 1.04x |
| `kmeans` N=2e5, R=8, B=64 (control) | 11.68 s | 11.90 s | 0.98x |
| `predict` N=2e4, R=3, B=8 (control) | 0.379 ms | 0.396 ms | 0.96x |

The controls touch none of the changed code, and their 0.96–1.04x spread is the noise floor of a
single-run comparison. The exchange speedups are well outside it.

## Optimizations deliberately not taken

Recorded so they are not re-attempted.

- **Expanded quadratic forms** — 0.953 ms vs 3.31 ms per chunk, **3.5x**, largest deviation
  6.7e-16 and zero `argmax` changes on the tested chunk. Not taken: it is not bit-identical, it
  replaces the residual-first formulation with a difference of large quadratic forms (trading
  cancellation safety for flops), and `baselines.json` is recorded on arm64/macOS and checked on
  Linux at `--quality-rtol 1e-6`, so widening the cross-platform numerical envelope is a real
  risk. This wants its own ADR and a written error analysis, not a drive-by edit. It is
  nonetheless the single largest lever in the library.
- **On-device argmax reduction** — measured *slower*: 4.02 ms vs 3.31 ms for the kernel with the
  reduction fused in. The host reduction it would replace costs 0.368 ms, and `np.asarray` on a
  CPU-backend array is zero-copy, so there is no transfer to save.
- **Compiling the soft-schedule history checkpoints** — measured 23.9 ms per checkpoint at
  N=1e6, R=3, B=8 (5.3% of the run) and 31.1 ms at N=1e5, R=8, B=64 (6.4%). Too small to justify
  perturbing a recorded diagnostic whose values reach users through `hardening_gap`.
- **Porting or rewriting `scalar_interval_dp`** — 87.9% of its runtime is already inside
  vectorized NumPy stripes. Its cost is the `O(n_bins * N^2)` recursion, not the implementation.
  The lever, if one is ever needed, is a monotone/SMAWK divide-and-conquer DP.
- **Compiling `init_state` end to end** — `profiled_exchange` still shows roughly 40% eager JAX
  dispatch on rank-sized matrices after optimization 3, concentrated in the per-rebuild gathers
  and the per-move `_rank_two_block`. Collapsing the rest requires moving a documented
  data-dependent guard (`_rank_two_inverse_update`'s exact-refresh-on-drift branch) into
  compiled control flow, which changes when the exact refresh fires. That is a behavioural
  change, not a cheap win.

---

## Closed: the terminal geometry guard at N = 10^6

Not a performance issue, but it is what stopped a converged 10^6-row D-exchange measurement,
so it belongs with the numbers. Fixed in
[ADR 0016](../docs/decisions.md); the diagnosis
below is kept because it is the measurement that motivated the change.

`optimize_partition` at N=1 000 000, R=3, B=8, seed 2026 converges normally — 280 scans, 99 001
accepted moves, `exchange_stable=True` — and then raises:

```
ValueError: terminal D state is geometrically degenerate;
duplicate/tied score atoms must be merged or assigned consistently
```

Diagnosed exactly: **13 rows out of 1 000 000** disagree between the labels the exchange holds
and `_assign_nearest`'s Mahalanobis-Voronoi assignment. Their two quadratic forms differ by a
relative 1.8e-6 to 7.4e-5 on values of order 1.5e-6 — an absolute gap of about **1e-11, inside
the exchange's own `gain_tolerance = 1e-10`**. Each of those rows is genuinely exchange-stable
at the tolerance the solver declares; the terminal self-check in `optimize_d_partition` is an
exact-equality test, and `PartitionResult.compile_quantizer` carries the same exact check.

So this is a tolerance-consistency limit, not a degeneracy: the certificate is issued at
tolerance tau and then verified at tolerance zero. It first appears around 10^6 rows simply
because that is where the expected number of rows within tau of a cell boundary reaches one.
`lloyd` at the same shape trips it identically, since it shares the terminal check.

The fix is exactly that: a terminal state is accepted when every mismatching row's relocation
gain is within `gain_tolerance`, matching the exchange's own certificate, and both
`GeometryReport` and `StabilityReport` now record the tolerance they hold at. The same cell
now converges and compiles — 280 scans, 99 001 accepted moves, `best_remaining_gain`
8.65e-11, `geometry.maximum_violation_gain` 8.65e-11 over 13 violating moves, all under the
default 1e-10 — in about 20 s on the reference machine. `compile_quantizer`'s contract moved
with it: the compiled rule is self-consistent at the tolerance, not at zero, and its
`predict_scores` differs from `PartitionResult.labels` on exactly those 13 boundary rows.
See [ADR 0016](../docs/decisions.md).
