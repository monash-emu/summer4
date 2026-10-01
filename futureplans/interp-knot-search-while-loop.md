# Interp knot search lowers to a `while` loop in every vector-field call

## Seen on

`perf/gradient-performance`, profiling the Kiribati TB port's calibration gradient
(`monash-emu/kiribati-tb-summer4`, `docs/gradient-performance.md` on
`feat/gradient-performance`).

## What is wrong

`_eval_interp` (`src/summer4/flows/compiled.py`) finds the knot with `jnp.searchsorted`
(sigmoidal and step tables) and `jnp.interp` (linear tables, which calls `searchsorted`).
Their default `method="scan"` is a `fori_loop` binary search: a `scan` in the jaxpr and an
XLA `while` in the compiled program. The Kiribati vector field has three (death-rate table,
births, treatment success), so a calibration solve runs three small loops (~8 iterations of
~5 kernels each) in each of its ~4,300 vector-field calls, and again in every recomputation
of the reverse pass (16 `while` loops in the compiled gradient against 10 without them).

## Prototype on this branch

`_knot_index(xs, x, side)` calls `jnp.searchsorted(..., method="compare_all")` for tables of
up to `_COMPARE_ALL_MAX_KNOTS = 1024` knots (one fused comparison against every knot) and
`method="scan_unrolled"` above that (straight-line `log2(n)` steps). `_linear_interp` is
`jnp.interp`'s formula on `_knot_index`. Answers are identical (same indices; log density and
gradient bit-identical on the Kiribati model). Tests: `tests/test_interp_knot_search.py`
(indices against NumPy on both sides of the threshold, values and gradients against
`jnp.interp` including a repeated knot, and no `while`/`scan` in the vector field or its
gradient for linear, sigmoidal and step tables; that last test fails on `c9548d5`).

## Measured gain (small)

In-process interleaved A/B on the Kiribati calibration model (Dopri5 at 1.4e-4, MAP point,
15 rounds on a loaded 10-core M4; `scripts/bench/searchsorted_ab.py` in the port):

| | wall speedup | CPU-time speedup |
|---|---|---|
| one vector-field call | 1.01x | 1.06x |
| log density | 1.01x | 1.04x |
| gradient | 1.09x | 1.02x |

XLA:CPU runs a small `while` cheaply; the loops were not the half of the vector field their
kernel count suggested. Worth landing because it is free and removes loops from every
gradient, not because it is a large speedup.

## Done when

- `_eval_interp` uses a loop-free knot search (this prototype, or `searchsorted` with an
  explicit method) and the no-loop test is in `tests/`.
- `benchmarks/test_bench_tb_scale.py` records the before/after vector-field time.
