# `wf.optimize` maps starts only by `vmap` (Kiribati W7, `CX60`)

## What is wrong today

`OptimizeRun` advances every start together in one compiled program: `_run_chunk_vmap`
(`src/summer4/epi/calibration/workflow/optimize.py`) is `jax.vmap` over the starts of a
`lax.scan` of steps. Under `vmap`, an adaptive solve runs every lane to the slowest lane's step
count, and a line search (L-BFGS) to the slowest lane's line search.

The Kiribati first port measured it: an `optax.lbfgs` `OptimizeMethod` did not finish one 10-step
chunk for 24 starts in 18 minutes, while scipy L-BFGS-B with one host loop per start, run
concurrently in 8 threads over a jitted value-and-gradient, did 24 starts in 285 s
(`kiribati-tb-summer4/src/kiribati_tb/pipeline.py`, `lbfgs`; now in `tb-core`).

The other half of W7, `wf.Optax` not passing `value_fn` (so `optax.lbfgs` could not run at all),
was fixed on `feat/kiribati-gaps` (`CX58`).

## What done looks like

- `OptimizeRun(..., map=)` (or a `StartMap` policy object, one policy per piece): `"vmap"` (today's
  default), `jax.lax.map` (sequential, one compiled step program, no lane coupling), and a
  host-parallel map over devices or threads (`jax.pmap` / `shard_map` over the host CPU devices
  summer4 already exposes).
- The chunk record and `Candidates` history are the same whichever map is used, and a test checks
  that `lax.map` and `vmap` give the same fits on a fixed-step model.
- The docstring of `optimize` says which map to pick for an adaptive solver.
