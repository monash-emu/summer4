# Diffrax `run()` rebuilds break the JIT cache

## Problem

`CompiledModel.run` goes through `diffrax_solve`, and Diffrax's
`diffeqsolve` is `@eqx.filter_jit`. That cache does not hit across
repeated `run()` calls with the same save plan and solver settings.

On every call, `diffrax_solve`
(`src/summer4/solvers/diffrax_backend.py`) rebuilds:

1. `ODETerm(vf)` with a new Python `vf` closure
2. `SubSaveAt(..., fn=_group_fn(...))` with a new `fn` per save group

Those callables are static leaves for equinox and are keyed by **identity**.
Each `run()` therefore looks like a different program, so equinox misses and
the whole solve is traced again.

`_group_fn` also **closes over `params`** and `del`s `args`. Even with
stable function objects, baking params into the static tree would recompile
on every parameter draw. The comment in `_group_fn` already points at the
right design: take prepared params from `args`.

## Evidence (performance suite)

On `performance-review-s2`, a bare `compiled.run` loop on `sir` / Diffrax
Euler / 200 steps stayed at ~0.3 s every call. The same solve with
`ODETerm` / `SubSaveAt` built once and wrapped in `eqx.filter_jit` dropped
to a warm median of ~0.0003 s (about 1000×). The committed
`benchmarks/recorded-summer4.json` “warm” column is therefore mostly
**recompile-per-call**, not a warm Diffrax solve. summer2's graph runner
does stay warm (~0.00017 s for the same cell).

## Done when

1. Save / observe path uses `args` (prepared params), not a closed-over
   `params`.
2. `ODETerm` / `SubSaveAt` callables are cache-stable across `run()` —
   equinox `Module`s with structural equality, or built once and cached on
   `CompiledModel` keyed by save-plan + solver config.
3. Ideally summer4 owns a `filter_jit` boundary that only takes array
   pytrees on the hot path, so users do not need an outer `jax.jit`.

After that, repeated `compiled.run(params_i, ...)` with the same save plan
and solver settings pays one compile, then solve times in the same ballpark
as the outer-jit probe above.
