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

## After step 3

- Smoke command, from the repo root:

  ```bash
  pixi run --manifest-path summer2bench/pixi.toml smoke-stratified
  ```

  Nothing failed to trace. No blank cells. Do not switch these models to
  `AgeStratification`: spec bands are `a0`..`a15`, and that class requires
  integer strata starting at 0 and inserts ageing flows. Age is a plain
  `Stratification("age", spec["age_bands"], spec["compartments"])`. Population
  split is that class's default, `1/n` per stratum.
- `mixing_matrix` arguments `get_runner` accepted. Only the age stratification
  has one. Location and strain leave `mixing_matrix` as `None`.
  - Static (`age_mix`): `set_mixing_matrix(jnp.asarray(spec["static_mixing"], dtype=jnp.float64))`.
    A float64 array, shape `(16, 16)`, not a callable. The traced matrix at
    `t0` equals that spec array.
  - Time-varying (`age_mix_tv`, `stress`): `set_mixing_matrix(time_varying_mixing(spec))`.
    That is a `computegraph` `Function`, not a host callable:

    ```python
    Function(_select_mixing_matrix, (Time, Data(stack), Data(bin_width), Data(last_index)))
    ```

    `_select_mixing_matrix` is `floor(t / bin_width)` then `clip` to
    `0 .. len(stack) - 1`, then `stack[index]`. `bin_width` is
    `spec["tv_mixing_bin_width"]`. Euler jaxpr of both models has no host
    callback. Checked against the spec slices at `t0` (index 0), one bin
    later (index 1), just below `t0` (clamped to 0), and past the last bin
    (clamped to the last slice).
- Within-strain infection is `StrainStratification("strain", spec["strains"], spec["compartments"])`
  applied after age and location, with no mixing matrix. summer2 sets
  `_disease_strains` to `s0`..`s3` (confirmed on the runner) and the jax
  runner computes force of infection per strain. Both ends of infection are
  stratified, so each flow stays inside one strain. Location is
  `Stratification("location", spec["locations"], spec["compartments"])` and
  does not get a mixing matrix.
- `stress` contact rate is `stress_contact`: the step-2 interpolator, then
  the same 24 `Parameter.__mul__` adjustments. Not `contact_rate` times the
  interpolator.
- `stress` build time is construction plus `get_runner`, not the solve.
  Euler `1.370s`, RK4 `0.569s`. Not minutes. Euler is the cold build; RK4
  reuses the process. Compile was euler `0.593s`, rk4 `1.259s`.
- Counts from `len(model.compartments)`, both solvers: `age_mix` 48,
  `age_mix_tv` 48, `stress` 3840. Flows float64 shape `(201,)`, compartments
  `(201, 48)` or `(201, 3840)`, sums finite. At 200 steps (`t` in `[0, 20]`)
  `age_mix` and `age_mix_tv` euler sums match (`203482887.8567661`) because
  every step is still in bin 0 and `tv_mixing[0]` is the static matrix.
  That is not a failed time index. `stress` euler reduced
  `201000912.24408036`, rk4 `201712150.98083156`.

## After step 4

- Recorded matrix is `summer2bench/recorded.json`: a JSON list of 36 records.
  **36 ok, 0 failed.** No blank cells, no checkpoint stop. Command, from the
  repo root:

  ```bash
  pixi run --manifest-path summer2bench/pixi.toml record
  ```

  Each cell is its own process, so a kill would have been `status: failed`
  instead of a lost file. Do not compare `build_s` with step 3. There the RK4
  build reused the process; here every build is cold. Summer2 import sits
  outside the timer. `build_s` is still construction plus `get_runner`.
  `compile_s` is the first call. `warm_median_s` is the median of the next
  five, each reduced and `block_until_ready`. The first call is not one of
  the five. `warm_s` keeps those five samples.
- Sanity pair, both float64, Apple M4. `sir` / euler / 200 warm median
  `0.000170s` (compile `0.157s`). `sir` / rk4 / 200 warm median `0.000427s`
  (compile `0.263s`). The spot check passed: the euler warm median is smaller
  than compile. One longer cell does not: `stress` / rk4 / 2000 warm median
  `4.800s` against compile `4.478s`. That is the solve. The five samples are
  all multi-second and the slowest is not the first, so the timer was not
  rerun.
- `stress` at 8_000 steps did not run out of memory. Peak RSS of that child
  (`ru_maxrss`, bytes on macOS, includes the JAX process) was euler
  `1353695232` (1.26 GiB) and rk4 `1388462080` (1.29 GiB). Warm medians were
  euler `2.800s` and rk4 `8.401s`.
- Machine and date, copied from every record: `macOS-15.7.9-arm64-arm-64bit;
  arm64; Apple M4; 24 GiB`, date `2026-09-20`. JAX `0.4.38`, `summerepi2`
  `1.3.6`, dtype `float64`. Compartment counts: `sir` / `sir_adjust` /
  `sir_tv` 3, `age_mix` / `age_mix_tv` 48, `stress` 3840.

## After step 5

- Classical RK4 is `ClassicalRK4` in `benchmarks/diffrax_rk4.py`. Import:

  ```python
  from benchmarks.diffrax_rk4 import ClassicalRK4  # or same-dir: from diffrax_rk4 import ClassicalRK4
  ```

  The `CompiledModel.run` kwargs that hit Diffrax `ConstantStepSize` are built by
  `fixed_step_run_kwargs` in `benchmarks/models.py`:

  ```python
  built.compiled.run(
      built.parameters,
      built.y0,
      solver=diffrax.Euler(),  # or ClassicalRK4()
      t0=0.0,
      dt=0.1,
      steps=200,
      max_steps=200,
      save=built.plan,
  )
  ```

  No `rtol`, no `atol`, and never the string `solver="euler"` (that string is
  the hand-rolled backend). Smoke command, from the repo root:

  ```bash
  pixi run python benchmarks/smoke_unstratified.py
  ```

- `FlowMass(flow=..., sum_over=(pop, "source"))` reduces the unstratified
  infection/recovery edges onto the single `pop` trait `"all"`. The saved
  array is shape `(201, 1)`; `flow_series` takes column 0 to match summer2's
  length-201 series. Compartments are float64 shape `(201, 3)`.
- Contact rate: `sir` is `Param("contact_rate")`. The 24 adjustments are
  `contact_with_adjustments` — a left-fold of `Param("adjustment_i")`
  multiplies on that rate. The 8-knot interpolation is
  `time_varying_contact` → `linear(Time(), times, values)` from
  `summer4.timevarying`, with knot floats read from the spec (not NumPy
  `interp` inside the step).
- `jax_enable_x64` is set at import of `benchmarks/models.py`, before any
  `jit`. Smoke confirmed compartment and flow dtypes `float64` under both
  Diffrax solvers (JAX 0.6.2). `benchmarks/harness.py` is the three-timer
  split for step 7. No `src/summer4` change. Stratified ports are step 6.

## After step 6

- Builders are all `build_model(name)` in `benchmarks/models.py`. Compile-time
  compartment counts from `compiled.pmap.size`: `age_mix` 48, `age_mix_tv` 48,
  `stress` 3840. Smoke command, from the repo root:

  ```bash
  pixi run python benchmarks/smoke_stratified.py
  ```

- Time-varying mixing is a `Lookup` inside the MixingMatrix rate tree, so it
  evaluates in the jitted step:

  ```python
  Lookup(Param("tv_mixing"), floor(Time() / bin_width))  # clamp=True
  ```

  The table passed as the `tv_mixing` parameter is float64 shape `(8, 16, 16)`.
  Static `age_mix` uses `jnp.asarray(spec["static_mixing"])` baked into
  `MixingMatrix` with `normalize="none"`. `ForceOfInfection.group_by` is age
  only. Location and strain are further `stratify` calls with no second mixing
  matrix. Within-strain infection is `ForceOfInfection.per_trait`, which adds
  flows `infection_s0`..`infection_s3`; a `SaveFn` sums them into the saved
  key `infection`.
- 200-step smoke status (all run, float64, finite sums):

  | model | diffrax-euler | diffrax-rk4 |
  | --- | --- | --- |
  | age_mix | ok | ok |
  | age_mix_tv | ok | ok |
  | stress | ok | ok |

  At 200 steps (`t` in `[0, 20]`), `age_mix` and `age_mix_tv` euler reduced
  sums match (`203532235.19050047`) because every step is still in mixing bin
  0. No `futureplans/` note. No `src/summer4` change.

## After step 7

- JSON paths: `summer2bench/recorded.json` (summer2) and
  `benchmarks/recorded-summer4.json` (summer4). Comparison table:
  `benchmarks/README.md`. Protocol write-up: `docs/dev/benchmarking.md`.
  Recorder: `pixi run bench-models` → `benchmarks/record_models.py`.
  Regenerate the README table with `python benchmarks/write_readme_table.py`
  after either JSON changes.
- Failed cells: summer2 **0 / 36**, summer4 **0 / 36**.
- `sir` / 200 warm medians (float64, Apple M4, 2026-09-20), **pre** Diffrax
  JIT-cache fix (warm column was recompile-per-call):

  | library | solver | warm median |
  | --- | --- | --- |
  | summer2 | euler | 0.000170 s |
  | summer2 | rk4 | 0.000427 s |
  | summer4 | diffrax-euler | 2.025 s |
  | summer4 | diffrax-rk4 | 2.288 s |

  Diffrax was confirmed to take 8000 steps on an 8000-step call
  (`SolverInfo.num_steps == 8000`, save shape `(8001, …)`). For small models
  the warm time barely grew with step count because Diffrax was recompiling
  every call. That reading is superseded by the re-record below.
- **Diffrax JIT cache** is fixed on `main` (`plans/diffrax-run-jit-cache.plan.md`,
  PR #13): equinox Modules for VF / `SubSaveAt`, params via `args`.
  Roadmap still reports step 6 as next; this branch did not touch it.
  There is no step 8.

## After JIT-cache re-record

- Re-ran `pixi run bench-models` after deleting
  `benchmarks/recorded-summer4.json` (the recorder skips already-ok cells).
  summer2 JSON was left unchanged. Regenerated `benchmarks/README.md` via
  `python benchmarks/write_readme_table.py`. Cleared the "re-record if warm
  looks like recompile" notes in that script and in
  `docs/dev/benchmarking.md`.
- Failed cells: summer4 **0 / 36**. Machine string on the new records:
  `macOS-15.7.9-arm64-arm-64bit-Mach-O; arm64; Apple M4; 24 GiB` (same host;
  `platform.platform()` now includes `-Mach-O`).
- `sir` / 200 warm medians (float64, Apple M4, 2026-09-20), **post** fix:

  | library | solver | warm median |
  | --- | --- | --- |
  | summer2 | euler | 0.000170 s |
  | summer2 | rk4 | 0.000427 s |
  | summer4 | diffrax-euler | 0.001464 s |
  | summer4 | diffrax-rk4 | 0.001327 s |

  Warm time now scales with step count on small models (`sir` / euler:
  200 → 1.46 ms, 2_000 → 3.13 ms, 8_000 → 11.1 ms). `stress` also drops
  (euler 8_000: 2.25 s → 1.30 s; rk4 8_000: 4.36 s → 3.28 s) and still
  dominates compile on the longer cells.
- Plan steps 1–7 are closed. No further plan step; follow-ups belong in
  `futureplans/` or a new plan if someone starts optimizing Diffrax overhead
  relative to summer2.

