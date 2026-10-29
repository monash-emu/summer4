# Per-model number width (`dtype=`)

## Today

Importing summer4 turns on JAX's process-wide `jax_enable_x64` flag unless
`JAX_ENABLE_X64` is set (`src/summer4/_precision/__init__.py`, `plans/float64-default.plan.md`).
Every model, output and likelihood in the process then runs in float64; float32 means
turning the flag off for the whole process, including other JAX libraries
(composability finding `CX39`).

That is enough for one model per process. It cannot express:

- a float32 model and a float64 model in the same process (a fast float32 scenario
  sweep beside a float64 calibration);
- mixed precision: solve in float32 for speed, but sum the log-likelihood in float64,
  which is where NUTS is most sensitive;
- float32 on a backend without float64 (Apple Metal) while keeping float64 elsewhere.

## Shape

Keep `jax_enable_x64` on, so both widths can exist, and make the width a property of
the compiled model:

```python
cm = model.compile(dtype=jnp.float32)      # default: jnp.float64
```

- Take a dtype object, not a string (composability principle P1).
- One place converts to it: `y0`, params (`_as_jax_params` in
  `src/summer4/flows/stages.py` takes the dtype instead of reading the global flag),
  hoisted rate values, time grids, interpolation tables (`src/summer4/data.py` and
  `src/summer4/time.py` build float64 NumPy arrays), and solver tolerances. diffrax follows
  `y0`'s dtype; outputs follow the solve.
- The dtype is part of the compiled model's jit cache key.
- Calibration: `BayesianModel` casts numpyro's float64 draws to the model's dtype at
  the model boundary. The likelihood width is a separate, public setting (one policy
  per piece), defaulting to the model's.

## The trap

With x64 on, any float64 constant that meets a float32 state silently promotes the
whole computation back to float64, so the model runs in float64 while claiming
float32. A test must catch this: `jax.make_jaxpr` of the vector field, a full run and
a likelihood with `dtype=jnp.float32` contains no `f64` values (checking every
equation's `outvars` avals), for a model that uses interpolation, a mixing matrix and
an `OutputSet`.

## Not this

A `with summer4.precision(...)` context manager. The flag must hold both when a
function is traced and every time it is called, so a scoped switch is easy to get
wrong.

## Done when

`compile(dtype=)` exists, the no-`f64` jaxpr test passes for float32, float64 stays
the default, and `CX39` is `done`.
