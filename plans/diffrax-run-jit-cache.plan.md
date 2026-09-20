# Diffrax `run()` JIT cache

## Problem

`CompiledModel.run` goes through `diffrax_solve`, and Diffrax's `diffeqsolve`
is `@eqx.filter_jit`. That cache did not hit across repeated `run()` calls
with the same save plan and solver settings.

On every call, `diffrax_solve` rebuilt:

1. `ODETerm(vf)` with a new Python `vf` closure
2. `SubSaveAt(..., fn=_group_fn(...))` with a new `fn` per save group

Those callables are static leaves for equinox and are keyed by **identity**.
Each `run()` therefore looked like a different program, so equinox missed and
the whole solve was traced again.

`_group_fn` also closed over `params` and `del`d `args`. Even with stable
function objects, baking params into the static tree would recompile on every
parameter draw.

## Approach

1. Save / observe path takes prepared params from Diffrax `args`, not a closure.
2. Vector field and save callbacks are process-stable **equinox Modules**
   (`DiffraxVectorField`, `DiffraxSaveFn`) created once via
   `_ensure_eqx_modules`. Structural equality follows `CompiledModel`'s digest
   and the request/`keep` tree, so new Module instances per `run()` still hit
   Diffrax's `filter_jit` cache.
3. Nested class definitions inside the solve path are avoided: a fresh class
   object per call breaks Module equality and reintroduces the miss.

## Exit checks

- Repeated `compiled.run(...)` with the same save plan / solver: one cold
  compile, then warm solves ~1000× faster on the SIR Euler 200-step probe.
- Changing params between calls still changes the trajectory (args path).
- Existing solver tests and `examples/notebooks/05-solvers.ipynb` still pass.

Promoted from `futureplans/diffrax-run-jit-cache.md`.
