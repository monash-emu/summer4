<!--
Accepted 2026-10-30. Historical record once merged: do not edit. Follow-up work goes in a
new plan.
-->

# float64 by default

## Context

JAX computes in float32 unless its process-wide `jax_enable_x64` flag is on, and summer4
never turned it on, so every solve, output and likelihood ran in float32. summer2 ran in
float64 (`computegraph/jaxify.py` turns the flag on when imported). float32 keeps about
seven significant figures: a compartment of 100 million moves in steps of 8 people, so
small flows into large compartments are lost and populations drift.

We want float64 by default, with float32 still available.

## Shape

Option (a) of two considered: turn the flag on when summer4 is imported.

- `src/summer4/_precision.py`, imported first by `summer4/__init__.py`, calls
  `jax.config.update("jax_enable_x64", True)` unless `JAX_ENABLE_X64` is in the
  environment (then JAX has already applied the user's choice).
- Opting out: `JAX_ENABLE_X64=0` before starting Python, or
  `jax.config.update("jax_enable_x64", False)` after importing summer4 and before
  compiling.
- The flag is process-wide, so it also applies to other JAX code in the process.
  That is a composability limit, recorded as `CX39` (deferred).

Option (b), a per-model `compile(dtype=)` with a separate likelihood width, is deferred:
`futureplans/per-model-dtype.md`.

## Files

- `src/summer4/_precision.py` (new), `src/summer4/__init__.py`
- `tests/test_float64_default.py` (new)
- `examples/notebooks/33-float64-default.ipynb` (acceptance)
- `docs/user/09-running-and-results.ipynb` (a "Number precision" section),
  `docs/user/07-from-summer2.md` (summary-table row)
- `docs/evaluation/composability.md` (`CX39`), `futureplans/per-model-dtype.md`
- `CHANGELOG.md`

## Tests

- In fresh subprocesses: `import summer4` turns the flag on and runs are float64; so
  does importing summer4 after JAX has created arrays; `JAX_ENABLE_X64=0` / `1` wins;
  turning the flag off after import gives float32.
- In the test process: the flag is on and a run is float64.
- A 100-million-person SIR seeded with one case: float64 conserves the population to
  1e-6 and records the first infections in S; float32 drifts by more than a person and
  S stays at exactly 1e8.
- The whole suite passes in float64.

## Acceptance notebook

`33-float64-default.ipynb`, the same 100-million-person SIR with `tsit5`:

1. The flag and dtypes after `import summer4`; the spacing of representable numbers at
   1e8 for each width.
2. `S(0) − S(t)` at both widths against infections integrated from the infection flow.
3. Population drift over a year, both widths.
4. How to choose float32, and the run time at each width.
