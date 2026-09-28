# Every `sample_until` chunk recompiles numpyro's sampling loop

**Where:** `MCMCRun.extend` / `MCMCRun._sample` in
`src/summer4/epi/calibration/workflow/mcmc.py`, which call
`numpyro.infer.MCMC.run` once per chunk.

**What is wrong.** `MCMC.run` → `numpyro.util.fori_collect` wraps its loop in a
fresh closure (`loop_fn`) on every call and `jit`s it, so each call is a cache
miss and a full XLA compile. The inner step (`_body_fn`) is cached; the outer
loop is not. Measured on the notebook-22 SIR model (step 27, numpyro 0.16.1,
CPU, 2 chains, vectorized):

| chunk size (draws per chain) | seconds per `extend` chunk |
|---|---|
| 5 | 0.47 |
| 50 | 0.47 |
| 500 | 0.49 |

The cost is all compilation (`backend_compile_and_load` ≈ 0.40 s of 0.60 s in a
profile); sampling itself is nearly free here. On a TB-scale model, compile
time is seconds to minutes, so a run of *k* chunks pays *k* compiles. This is
why chunk size should not be small, and why notebook 22's seeding sweep runs
replicates as extra chains in one `MCMC.run` and uses `wf.replay` instead of
many `sample_until` calls.

`warmup_until` also compiles once per round, but that is inherent: each round
has a different `num_warmup`, which changes numpyro's adaptation schedule.

**Options.**

1. Drive the kernel directly for continuation chunks: one jitted `lax.scan`
   over `jax.vmap(mcmc.sampler.sample)` from `mcmc.last_state`, compiled once
   per `MCMCRun`, with the kernel's public `postprocess_fn` for constrained
   draws. Keeps the caller's `MCMC` for warmup and the first chunk. Needs care
   for ensemble kernels (not vmapped per chain) and for keeping
   `mcmc.post_warmup_state` in step so the caller can still resume with plain
   numpyro.
2. Document JAX's persistent compilation cache
   (`jax.config.update("jax_compilation_cache_dir", ...)` with a lower
   `jax_persistent_cache_min_compile_time_secs`): every chunk lowers to the
   same HLO, so later chunks become disk-cache hits. A global setting, so it is
   the user's choice, not something summer4 should switch on.
3. Upstream: a numpyro change that caches `loop_fn` by its static arguments.

**Done when** a `sample_until` run of *k* chunks compiles its sampling loop a
bounded number of times independent of *k* (check with a counter on
`jax.jit` cache misses, or a timing test like the table above).
