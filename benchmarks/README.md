# Benchmarks

```bash
pixi run bench        # taxonomy table (pytest-benchmark)
pixi run bench-json   # taxonomy JSON keyed by JAX version
pixi run test         # includes the slow TB-scale suite
```

`benchmarks/results/` is gitignored: machine-specific pytest-benchmark dumps are
compared within a run. The **TB-scale baseline table below is committed** — it
is the regression reference for a Kiribati-shaped model (step 11 / WP16 §16a).

## Taxonomy (`test_bench_taxonomy.py`)

| Benchmark | Sizes | Measures |
| --- | --- | --- |
| `test_bench_build` | 10, 1 000, 100 000 compartments | Cartesian map construction |
| `test_bench_select_cold` | 10, 1 000, 100 000 | Query resolution with an empty cache |

See `docs/dev/benchmarking.md` and `docs/dev/performance.ipynb`.

## TB-scale baseline (`test_bench_tb_scale.py`)

Synthetic Kiribati *shape*, no real data:

- **160** compartments — 10 states × 8 age bands (`0 3 5 10 15 18 40 65`) × 2 reachability
- **~35** named flows — generalised FOI with `Lookup` mixing, per-age `Data.table` deaths, recycled births, `TraitChain.from_breakpoints` ageing, four reinfection paths
- **~180** `OutputSet` names — yearly saves **1850–2035**
- Solvers: fixed-step `euler` (`dt=1`) and adaptive `dopri5` (`max_steps=100000`;
  the span-derived default is still short of this stiff horizon — pass an explicit
  ceiling, and check `SolverInfo.ok`)

Wall times are best-of-2 after one warm-up on the machine that wrote the table.
`vmap×64` varies `contact_rate` over 64 draws. `OutputSet evaluate` is the host/JIT
cost of the named DAG after one solve.

| Solver | Build+compile (s) | VF eqns | Run jaxpr eqns | Wall / run (s) | vmap×64 (s) | OutputSet evaluate (s) | Machine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `euler` | 0.012 | 651 | 1273 | 0.697 | 1.210 | 0.012 | macOS arm64 (2026-09-23) |
| `dopri5` | 0.012 | 651 | 104 | 0.015 | 0.207 | 0.011 | macOS arm64 (2026-09-23) |

`compile(fuse_compartment_updates=True)` (default) fuses per-flow **scatters**
into one `scatter-add` (source gathers stay per-flow). A/B uses **median of
interleaved** fused/looped samples with `jax.block_until_ready`.

**SIR** (4 flows, `t=0..40`) — noise floor; jaxpr 79→50 eqns, scatter-add 6→1.

**TB-scale** (160 comps, 35 flows, 1850–2035), interleaved median of 8, macOS
arm64 2026-09-23:

| fuse_compartment_updates | VF eqns | scatter-add | gather | solve (s) | grad (s) | value_and_grad (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `True` | 651 | 10 | 39 | 0.0057 | 0.0290 | 0.0290 |
| `False` | 1041 | 76 | 39 | 0.0075 | 0.0343 | 0.0328 |

Speedups (looped/fused): solve **1.32×**, grad **1.18×**, `value_and_grad`
**1.13×**.

Src-gather fusion was tried and reverted (AD slice→pad tax): see
`futureplans/vf-gather-and-mul-followups.md`.

Warm euler stays elevated because XLA's algebraic simplifier can still hit a
stuck loop on this shape (`futureplans/tb-scale-euler-xla-simplifier.md`), though
the fused jaxpr is smaller. Prefer dopri5 columns for runtime regressions until
that note is closed.

Re-run the A/B:

```bash
pixi run pytest benchmarks/test_bench_tb_scale.py::test_fuse_compartment_updates_timing_ab -v -s
```

Re-measure and replace baseline rows by running:

```bash
pixi run pytest benchmarks/test_bench_tb_scale.py -v -s
```

The test prints a markdown row block. Paste it into the table above (keep the
Machine column). Numbers will differ across hardware; what matters for
regressions is the same machine (or CI) before/after a change.

## Weighted `Reduce` sums and the XLA:CPU runtime (`weighted_reduce.py`)

A 41-compartment chain whose `active` stage is a 6 x 6 grid; the infectious pool weights each
cell, written three ways: 36 scalar `Reduce` terms (`terms`), a `ForceOfInfection` with 12
weight pairs (`foi`), and one unweighted `Reduce` for scale (`plain`). 100 years: Euler
`dt=0.05` (2000 steps) and the default adaptive Tsit5. Best of 20 after a warm call,
`devq bench --strict`, macOS arm64 12-core mini, jax/jaxlib 0.6.2, float64, 2026-11-20.
`pixi run -e default sh benchmarks/weighted_reduce_suite.sh` (add `--no-fuse` for the
term-by-term form); `benchmarks/weighted_reduce_numerics.py` prints the fused/unfused
differences.

`terms` form (vector-field jaxpr equations: 263 top level / 407 with nested `_take` bodies
unfused, 84 / 84 fused):

| Configuration | Euler solve (ms) | Euler grad (ms) | Tsit5 solve (ms) | Tsit5 grad (ms) |
| --- | ---: | ---: | ---: | ---: |
| before: thunk runtime, term by term (`main`) | 2.57 | 9.07 | 0.67 | 3.26 |
| thunk runtime, fused | 3.47 | 3.92 | 1.14 | 2.23 |
| legacy runtime, term by term | 1.00 | 1.70 | 0.34 | 0.94 |
| **after: legacy runtime, fused (defaults)** | **0.95** | **1.59** | **0.31** | **0.89** |

- The legacy runtime (`--xla_cpu_use_thunk_runtime=false`, summer4's default on jaxlib
  0.4.32-0.6.x) is the large win: 2.7x on the Euler solve, 5.7x on its gradient, 2.2x and 3.5x
  for Tsit5. Fusion adds 5-10% on top.
- Under the thunk runtime, fusion makes the Euler gradient 2.3x faster but compact graphs
  (fused `terms`, `foi`, `plain`) solve forward about 1.4-1.7x *slower* than the 36-term graph.
  The cause is XLA:CPU's concurrency-optimised thunk scheduler: with
  `--xla_cpu_enable_concurrency_optimized_scheduler=false` (jaxlib 0.11.1) the fused form wins
  everywhere (Euler 1.52 vs 1.67 ms, grad 3.16 vs 9.28 ms; Tsit5 0.62 vs 0.65 ms, grad 2.46 vs
  2.85 ms) and `foi` drops from 3.58 to 1.61 ms. summer4 sets that flag on jaxlib 0.7.0-0.12.x,
  where the legacy runtime no longer exists (checked to be accepted on 0.7.0, 0.8.0, 0.11.1 and
  0.12.1; XLA aborts on unknown flags, so later versions are left alone until checked).

jaxlib 0.11.1, `terms` form (thunk runtime only):

| Configuration | Euler solve (ms) | Euler grad (ms) | Tsit5 solve (ms) | Tsit5 grad (ms) |
| --- | ---: | ---: | ---: | ---: |
| before: default scheduler, term by term | 3.56 | 12.01 | 0.64 | 3.11 |
| default scheduler, fused | 2.74 | 6.21 | 1.14 | 4.23 |
| scheduler off, term by term | 1.67 | 9.28 | 0.65 | 2.85 |
| **after: scheduler off, fused (defaults)** | **1.52** | **3.16** | **0.62** | **2.46** |

- Fused vs term by term agree to 2.9e-16 (vector field), 2.2e-15 (final prevalence) and
  1.3e-14 (gradient), relative, on both runtimes and jaxlib 0.6.2 / 0.11.1.
