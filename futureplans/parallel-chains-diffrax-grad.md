# Parallel MCMC chains fail with gradients through the adaptive diffrax solve

Composability finding `CX45`.

## What is wrong today

`chain_method="parallel"` (numpyro maps one chain per device with pmap/shard_map; summer4
exposes one host device per CPU core) fails in CI's `latest` pixi environment (JAX 0.11.1,
numpyro 0.21, equinox 0.13.8, diffrax 0.7.2) when NUTS gradients go through the default
adaptive Tsit5 solve:

```
ValueError: Closure-converted function called with different dynamic arguments to the
example arguments provided
```

raised inside equinox's checkpointed while loop backward pass
(`equinox/internal/_loop/checkpointed.py`, around lines 1018 and 1049). The `default` env
(JAX 0.6.2) passes everything.

Three-way reproduction on `examples/notebooks/31-calibration-output-targets.ipynb`:

- parallel + Tsit5: fails
- vectorized + Tsit5: passes
- parallel + Euler: passes

So `chain_method="auto"` is pinned to `"vectorized"` (`resolve_chain_method` in
`src/summer4/epi/calibration/workflow/mcmc.py`). Parallel stays covered by
`tests/test_chain_method.py::test_explicit_parallel_with_euler_recovers_parameter`.

## What "done" looks like

Parallel + Tsit5 + NUTS passes in the `latest` env (notebook 31 and a parallel
auto-style test with the default solver). Then `"auto"` goes back to `"parallel"` when
`jax.local_device_count() >= num_chains > 1`, the guide (page 10), release notes, notebook 36
section D and `CX45` are updated, and the Euler-only test becomes a default-solver test.
Likely routes: an equinox/diffrax fix upstream, or a workaround that avoids the
closure-conversion mismatch (for example `adjoint=diffrax.DirectAdjoint()` or
`RecursiveCheckpointAdjoint` settings under shard_map).
