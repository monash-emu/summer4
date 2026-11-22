<!--
Accepted 2026-11-23 as proposed in conversation. Historical record once committed: do not edit it.
-->

# `sample_until` compiles once, not once per chunk

Branch: `perf/mcmc-chunk-cache`, cut from `main` at `5dab1b5`. Resolves
`futureplans/mcmc-chunk-recompile.md`, prompted by a user report ("numpyro's parallel chains
recompile on every chunk; a cached pmap belongs in `sample_until`").

## Context

`MCMCRun._sample` calls numpyro's `MCMC.run` once per chunk. numpyro builds a fresh per-chain
`functools.partial` on each call and maps it with `jax.pmap` (`"parallel"`) or runs it batched
through a freshly jitted `fori_collect` loop (`"vectorized"`), so every chunk misses JAX's compile
cache. Measured (numpyro 0.18.0, jax 0.6.2, toy NUTS model, 4 chains): one compile per chunk,
about 1.0 s (vectorized) and 0.46 s (parallel), almost all of it compile.

## What to do

1. `_ChainMapCache` in `workflow/mcmc.py`: a callable `chain_method` keeping one `jax.pmap`
   (parallel) or `jax.jit(jax.vmap(...))` (vectorized) per sampling setup, keyed on the
   per-chain partial's method, collected fields, removed sites, `_collection_params`,
   `num_warmup`, `num_samples` and `thinning`; uncached when the key cannot be formed (model
   arguments, unexpected numpyro internals).
2. `MCMCRun._run_cached`: swap in the callable, a private copy of numpyro's function cache and a
   private shallow copy of the kernel (its `_sample_fn` is set per chain under the callable but
   batched under `"vectorized"`) for each `run`, and restore all three afterwards.
3. Skip the cache for one chain, `"sequential"`, ensemble kernels and `progress_bar=True`.
4. Docs: `sample_until` docstring, `wf.pilot` docstring, notebook 29 text, futureplan
   marked resolved, changelog.

## Acceptance

- Compile count: after the second chunk (after the first, following `warmup_until`) a chunk
  causes no XLA compile, for both methods (`tests/test_calib_mcmc_chunk_cache.py`).
- Draws identical to numpyro's own mapping driven by plain `MCMC.run` calls, including after
  `warmup_until`.
- `mcmc.chain_method` restored; plain numpyro can continue from the run's last state.
- Existing calibration workflow tests pass.
