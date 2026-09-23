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
