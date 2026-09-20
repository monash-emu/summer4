# Box float params in `prepare()`

## Problem

a2 fixed Diffrax `run()` closure-identity misses so equinox Modules share a
`filter_jit` key across calls. A second miss remained: equinox treats **Python**
`float` leaves as static. Plain `{str: float}` parameter dicts therefore
recompile on every draw even when the Modules and save plan are stable.

`np.float64` and JAX arrays already pass `eqx.is_array`; callers can box by
hand, and an outer `@jax.jit` also works — but every host loop that passes
Python floats silently pays compile cost again.

## Approach

1. Add `box_float_leaves` in `summer4.flows.stages`: `jax.tree_util.tree_map`
   that promotes plain `float` leaves to `jnp.asarray(..., dtype=jnp.float64)`.
   Leave `int` / `bool` alone (indices and flags).
2. Call it from `CompiledModel.prepare` after optional `prepare_fn` and before
   hoist evaluation; also re-box when the input is already a `Prepared`.
3. Tests: unit check on prepared dtypes; warm Diffrax timing across distinct
   Python-float parameter draws.
4. User-gate notebook (`05-solvers`) and run-stages / changelog notes.

## Exit checks

- `pixi run lint && pixi run format-check && pixi run check-notebooks`
- `pixi run test && pixi run check-branch && pixi run coverage && pixi run roadmap`
