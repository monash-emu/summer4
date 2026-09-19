# Performance suite handoff

Living note for `performance-review-s2`. The contract is
`plans/performance-review-s2.plan.md` and is not updated from here.

## After step 1

- Locked `summerepi2` **1.3.6**, JAX **0.4.38** (`jax_enable_x64` true before
  `import summer2`). Probe command, from the repo root:

  ```bash
  pixi run --manifest-path summer2bench/pixi.toml probe
  ```

  Branch cut from `origin/main` at `573d056` (`Merge branch 'feat/table-interp'`).
  Local `main` was two docs commits ahead of that; they are not on this branch.

- `get_runner` call in `summer2bench/probe.py`. `solver` is a `**backend_args`
  keyword forwarded to `build_run_model`, not a positional. `parameters` is `{}`
  because the contact rate `0.35` and recovery `0.1` are plain floats, not graph
  parameters. Outputs are requested first:

  ```python
  model.request_output_for_flow("infection", "infection")
  model.request_output_for_flow("recovery", "recovery")
  return model.get_runner(
      PARAMETERS,
      include_full_outputs=True,
      solver=solver,  # "euler" or "rk4"
  )
  ```

- After `runner.run(PARAMETERS)`, both solvers: `model.outputs` is `float64`
  shape `(21, 3)` (initial time plus 20 steps of `dt=0.1`). Each requested flow
  in `model.derived_outputs` is `float64` shape `(21,)` — keys `infection` and
  `recovery` only. `raw_results` was left at its default `False`. The jitted
  `runner.function` return has the same dtypes and shapes. Both reduced sums
  (compartments plus both flow series) were finite.

- Template copied in full except `.git`: `pixi.toml`, `pixi.lock`, `LICENSE`,
  `.gitignore`, `.gitattributes`, `notebooks/EstivalPyMC.ipynb`. Nothing else
  dropped. `pixi install` on osx-arm64 succeeded. `pymc` 5.25.1 and `numpyro`
  0.19.0 import; `estival` imports (no `__version__`). ArviZ prints a future
  warning on `import pymc`. The notebook was not run; its cells already have
  `execution_count: null` and empty outputs. `win-64` and `linux-64` were not
  installed. Pixi warns that `pixi.lock` is format v6 and suggests `pixi lock`
  to upgrade to v7; that was not run, so the template lock is unchanged.

- Flow outputs were requested before `get_runner`, which finalises the graph.
  `summer2bench/time_run.py` is the timer later steps should extend: `build`
  is construction plus `get_runner` and excludes the first call; the first
  `call` is compile time; later calls are warm. Import it only after
  `jax_enable_x64` is set, because the module imports `jax`.

## After step 2

- Spec is `summer2bench/spec.json`. Top-level keys: `t0`, `dt`, `step_counts`,
  `smoke_steps`, `compartments`, `infectious_compartments`, `population`,
  `initial_infectious`, `initial_population`, `contact_rate`, `recovery`,
  `adjustments`, `contact_knots`, `age_bands`, `static_mixing_diagonal`,
  `static_mixing`, `tv_mixing_diagonal_weights`, `tv_mixing_bin_width`,
  `tv_mixing`, `locations`, `strains`, `flows`. Model code reads that file.
  `contact_knots.values[2]` is the exact product `0.35 * 0.95`, stored as
  `0.33249999999999996`. Do not recompute the knots in model code.
- Smoke command, from the repo root:

  ```bash
  pixi run --manifest-path summer2bench/pixi.toml smoke
  ```

  The reduced sum is `time_run.reduce_runner_outputs` (compartments plus each
  named flow). Extend that function; do not write a second timer.

  `summer2bench/models.py` enables `jax_enable_x64` before importing summer2.
  Import that module before any other summer2 import. Builders are
  `prepare_model` (requests `infection` and `recovery`, does not call
  `get_runner`) then `runner_for`. The call is:

  ```python
  prepared.model.get_runner(
      prepared.parameters,
      include_full_outputs=True,
      solver=solver,  # "euler" or "rk4"
  )
  ```

- The 24-multiply chain is `contact_with_adjustments`: a Python loop of
  `Parameter.__mul__`, not `adjust.Multiply`. One-line shape, repeated over
  `spec["adjustments"]`:

  ```python
  rate = rate * Parameter(f"adjustment_{index}")  # starts as Parameter("contact_rate")
  ```

  That builds `computegraph.types.Function` nodes whose `func` is
  `jax.numpy.multiply` (24 of them). The innermost breakpoint is
  `Function: 'multiply', args=(Parameter contact_rate, Parameter adjustment_0)`.
  `adjust.Multiply.get_new_value` is the host path and is not used.
  `jax.make_jaxpr` of the euler runner contains no `pure_callback`,
  `io_callback`, `debug_callback`, `ffi_call`, or `host_callback`.
- The interpolation is `time_varying_contact`. One-line call, `x_axis`
  left as the default `Time`:

  ```python
  get_linear_interpolation_function(times, values)  # both jnp.float64 arrays from contact_knots
  ```

  `get_linear_interpolation_function` lives in `summer2.functions`. Clamp is
  `interpolate_linear`'s out-of-bounds branches. Checked on the same Function
  the model receives: outside the knots the value is the nearest endpoint.
  The euler `sir_tv` jaxpr also has no host callback.
- Keys `get_runner` consumed (`model.get_input_parameters()`), both solvers:
  - `sir`: `contact_rate`, `recovery`
  - `sir_adjust`: `adjustment_0` .. `adjustment_23`, `contact_rate`, `recovery`
  - `sir_tv`: `recovery` only. The knots are not parameters. They are frozen
    `Data` (`InterpolatorScaleData`, float64) in `static_cg` (`_var1`, `_var3`).
    `timestep_cg` evaluates `interpolate_linear(Time, _var1, _var3)` and
    assigns that to `infection_rate`.
- At 200 steps, both solvers, all three models: `outputs` float64 `(201, 3)`;
  each of `infection` and `recovery` float64 `(201,)`; no other derived keys;
  compartment names `S`, `I`, `R`; reduced sums finite. With every adjustment
  equal to 1, `sir_adjust` matches `sir` (euler reduced `201025205.72790095`,
  rk4 `202931966.75927642`). `sir_tv` is slightly different, as it should be.
- For step 3, do not rediscover these:
  - Constant `infection_rate` is a `static_cg` Function. The same key in
    `timestep_cg` is only `static_inputs[infection_rate]`, not a second
    evaluation and not a host callback.
  - `BaseTransitionFlow.stratify` copies `param=self.param`, so the multiply
    chain is one shared object, not one chain per stratum. The conservation
    `Multiply(1/n)` is added only when the destination is stratified and the
    source is not, and not for strains. Age on both ends of infection should
    not add it.
  - A traced scalar `Function` of `Time` is not evidence that `mixing_matrix`
    accepts one. If the age-mixing API only takes a host callable, stop and
    record that. Do not time the callable path. The pattern that did trace
    is frozen `Data` plus `interpolate_linear` inside the step.
