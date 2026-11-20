<!--
Accepted 2026-11-20 as proposed in conversation. Historical record once committed: do not edit it.
-->

# Fuse weighted `Reduce` sums; faster XLA:CPU settings on import

Branch: `perf/fuse-weighted-reduce`, cut from `main` at `5dab1b5`.

## Context

A rate written as `sum_k c_k * Reduce(where=sel_k)` (an infectious pool with a weight per
cell) lowers to one gather and sum per term. On a 36-cell grid the vector field had 263 jaxpr
equations (407 counting nested `_take` bodies), and reverse-mode gradients through `lax.scan`
transpose every gather into a scatter-add per step. Separately, XLA:CPU's thunk runtime
(default from jaxlib 0.4.32) is much slower for the small operations inside a solver loop
(jax-ml/jax#26145, #29772, #40150).

## What to do

1. **Fusion** (`src/summer4/flows/fuse_reduce.py`): at compile time, rewrite the largest linear
   part of a rate or `Multiply`/`Overwrite` value that sums 2 or more scalar `Reduce` terms
   with constant or parameter-only coefficients into `WeightedReduce`: one gather over the
   union of the selectors and a dot product with a per-compartment weight vector.
   - Constant weights fold to an array at compile time; parameter weights (`ReduceWeights`) are
     hoisted to run stage.
   - Nested sums, subtraction, unary minus, `c * R`, `R * c` and `R / c` are covered. Grouped
     `Reduce`, products of `Reduce`s, time- or state-dependent coefficients, and arguments of
     `ForceOfInfection`/`Defer` are left as written.
   - `compile(fuse_reductions=True)`, mirroring `fuse_compartment_updates`.
2. **XLA:CPU runtime** (`src/summer4/_cpu_runtime.py`, imported by `summer4/__init__.py`):
   - jaxlib 0.4.32 <= v < 0.7.0: set `--xla_cpu_use_thunk_runtime=false` (legacy runtime);
   - jaxlib 0.7.0 <= v < 0.13.0: set `--xla_cpu_enable_concurrency_optimized_scheduler=false`
     (checked accepted on 0.7.0, 0.8.0, 0.11.1, 0.12.1; XLA aborts on unknown flags);
   - opt out with `SUMMER4_CPU_RUNTIME=thunk` or a user `XLA_FLAGS` naming either flag; warn if
     JAX's backend already started; report through `runtime_info()`.
3. Benchmark (`benchmarks/weighted_reduce.py`), guide page 10 section 5, changelog.

## Acceptance

- Fused and term-by-term results agree at rtol 1e-12 (vector field, Euler and Tsit5 solves and
  gradients) on overlapping, disjoint and empty selectors; empty selectors still give 0.
- Equation-count regression: fused vector field no larger than a single plain `Reduce` + 8.
- Runtime tests pass on jax 0.6.2 and the `latest` environment.
- Acceptance notebook `examples/notebooks/42-fused-reductions.ipynb`.
- Public API and semantics unchanged apart from the additive `fuse_reductions` keyword and
  `runtime_info()` fields; numerics change only by summation order (quantified in
  `benchmarks/README.md`).
