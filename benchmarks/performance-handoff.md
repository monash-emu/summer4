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
