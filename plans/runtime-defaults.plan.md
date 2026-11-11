<!--
Accepted 2026-11-11 as proposed. Historical record once committed: do not edit it.
Corrections go in the roadmap's "Corrections to committed plans" table; position lives in
docs/dev/roadmap.md (step 49).
-->

# Runtime defaults: an accurate solver and every CPU core, with nothing to configure

Branch: `feat/runtime-defaults`, cut from `main` at `6f1101f` (release 0.3.0a1).
Roadmap: **step 49** (Phase N), inserted before step 28.

## Context

Two defaults in summer4 0.3.0a1 surprise modellers. Both were found while writing the guide
(`futureplans/guide-on-ramp-sharp-edges.md`, and guide page 09).

1. **The default solver is fixed-step Euler.** `CompiledModel.run(..., solver="euler")`
   (`src/summer4/flows/compiled.py:1183`) gives wrong answers on oscillatory models unless `dt`
   is tiny. The Lotka–Volterra page needs `solver="dopri5", rtol=1e-8, atol=1e-8` to keep its
   cycles closed. Diffrax itself has no default solver: `diffeqsolve` requires one and defaults to
   `ConstantStepSize`. Diffrax's documentation recommends `Tsit5` as the general-purpose explicit
   solver, used with an adaptive `PIDController`.
2. **JAX sees one CPU device.** Nothing in `src/` sets `--xla_force_host_platform_device_count`.
   Calibration's MCMC defaults to `chain_method="vectorized"`
   (`src/summer4/epi/calibration/model.py:378`, `workflow/mcmc.py:406`), so its chains share one
   device, and numpyro's `"parallel"` method cannot spread chains across cores.

Checked on 2026-11-11: importing `summer4`, `summer4.epi` and `summer4.epi.calibration` does
**not** initialise JAX's backend (`jax._src.xla_bridge.backends_are_initialized()` is `False`).
So an XLA flag set during `import summer4` still takes effect, which is how the float64 switch
already works (`src/summer4/_precision/__init__.py`).

## User decisions (2026-11-11)

- Users should not notice or have to do anything: the new defaults apply on import.
- There must be ways to change or turn off each default, for testing, GPU use, and so on.
- Use the diffrax default solver. Diffrax has none, so we use its documented recommendation,
  Tsit5. This plan checks that before relying on it.

## Design

### D1. Host devices on import

`src/summer4/_devices/__init__.py` is a new module. Like `_precision`, it imports only `os` and
`warnings` at module level, plus whatever it needs to detect an initialised backend. It is
imported from `src/summer4/__init__.py` straight after `_precision`, before anything else that
could touch JAX's backend.

| Situation | Behaviour |
|---|---|
| Nothing set | Append `--xla_force_host_platform_device_count=<os.cpu_count()>` to `XLA_FLAGS` |
| `XLA_FLAGS` already contains `--xla_force_host_platform_device_count` | Leave it alone: the user chose |
| `SUMMER4_HOST_DEVICES=<n>`, with n ≥ 1 | Use n |
| `SUMMER4_HOST_DEVICES=0` (or `off`, `false`) | Do nothing |
| JAX's backend is already initialised (JAX was used before summer4 was imported) | Do nothing, and `warnings.warn` once, naming the remedy (import summer4 first, or set `XLA_FLAGS`) |

- The flag only adds **CPU** devices. On a GPU machine JAX still uses the GPU by default.
  `JAX_PLATFORMS` stays the user's way to choose a platform, and the docs say so.
- Inspection: `summer4.runtime_info()` returns a small frozen dataclass with these fields:
  - `x64`;
  - `host_devices_requested`;
  - `devices`, which calls `jax.devices()` and is lazy, so it does not initialise the backend
    at import;
  - `default_solver`;
  - `source`, which says where each value came from: default, environment, or user flag.

  It is public, so tests and users can check what happened.

### D2. Default solver

- `run(solver=None, ...)` means "the default solver". `resolve_solver(None, rtol=, atol=)` returns
  `default_solver(rtol=, atol=)`.
- `default_solver(*, rtol=1e-6, atol=1e-9)` is public, in `summer4.solvers`. It returns
  `Diffrax(diffrax.Tsit5(), stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol))`,
  and its docstring states that one line. That is the composability rule: conveniences are built
  from the pieces, and the docstring says which pieces.
- `rtol` / `atol` passed to `run()` with the default solver override its tolerances. With
  `"euler"` they are still refused.
- `dt` stays required. With an adaptive solver it is the first step size (`dt0`), and it sets the
  save grid when no `SavePlan.ts` is given, exactly as with `"tsit5"` today. Step R2 confirms
  this from `SolveSpec`.
- Process-wide override for tests and benchmarks:
  - `SUMMER4_SOLVER` set to any `KNOWN_SOLVER_NAMES` entry is read once, on first resolution;
  - `summer4.solvers.set_default_solver(backend_or_name | None)` sets the default, and `None`
    restores it;
  - a solver passed to `run()` always wins.
- `max_steps`: `default_max_steps` (`solvers/base.py:19`) already gives at least 4096 steps,
  with headroom. R2 checks that a stiff-ish tb-scale run does not hit it with Tsit5.

### D3. Chains use the devices

- `chain_method` defaults to `"auto"` in `BayesianModel.sample` (`model.py:378`) and
  `wf.run_mcmc` (`workflow/mcmc.py:406`). `resolve_chain_method("auto", num_chains)` is public,
  in `summer4.epi.calibration`:
  - it gives `"parallel"` when `jax.local_device_count() >= num_chains > 1`;
  - otherwise it gives `"vectorized"`.
- An explicit `chain_method` is passed through unchanged.
- Users who build their own numpyro `MCMC` (the composable path) choose `chain_method`
  themselves. The docs show `resolve_chain_method` as the one line to call.

### What changes for existing users

- **Numbers change.** Any model run without `solver=` now uses Tsit5 instead of Euler. Results
  move towards the exact solution. Code that needs Euler's exact numbers passes
  `solver="euler"`.
- **Devices.** `jax.device_count()` on CPU becomes the core count after `import summer4`. A
  single-device computation is unchanged in value. Its speed must be checked: step R0 measures it.
- **MCMC chains** run in parallel when there are enough devices. Draws for a given seed may
  differ from `"vectorized"`, so tests compare distributions, not draws.

## Steps

Workers are Claude subagents (`docs/dev/roadmap.md` § *Running a step with tiered workers*):

- haiku implements where this plan names the files and spellings, and sonnet otherwise;
- the checker is one tier up, read-only, in fresh context;
- each step is one commit on `feat/runtime-defaults`.

Heavy runs go on the mini (`mini build` / `mini bench`; pixi is `/opt/homebrew/bin/pixi` there).

### R0 — benchmark gate · sonnet, checker opus

Decide with numbers before changing any default. Write `benchmarks/runtime_defaults.py` and run
it on the mini with `mini bench --strict`. Build first, as `~/.devq/CLAUDE-devq.md` says.

1. **Solver.** For three models, take wall time per `run()` after compile, and the error against
   a reference (`dopri5`, rtol and atol `1e-12`):
   - the guide-01 SIR;
   - the guide-09 Lotka–Volterra;
   - the tb-scale bench model (`benchmarks/tb_scale.py`).

   Compare three solvers:
   - Euler at the `dt` the docs use;
   - the proposed default;
   - Tsit5 at rtol `1e-4` and atol `1e-7`.

   Also time one `jax.grad` of a target log-likelihood for the SIR.
2. **Devices.** Time a single SIR `run()` and the tb-scale `run()` with the device count at 1 and
   at the core count. Each count runs in its own process, using `XLA_FLAGS`.
3. **Chains.** For the guide-08 calibration, time 4 chains of 500 warmup and 500 samples,
   `"vectorized"` against `"parallel"`, with the device count at the core count.

Write the results to `benchmarks/results/runtime_defaults.json` and a short table in
`docs/dev/benchmarking.md`.

**Gate:** stop and report to the user before R1 if any of these hold:

- the default solver is more than 3× slower than Euler per run on any model, or its gradient
  costs more than 3×;
- forcing the device count slows a single-device run by more than 10%;
- parallel chains are not faster than vectorized ones.

The user then chooses tolerances or drops a piece.

### R1 — host devices on import · haiku, checker sonnet

**Files:**

- the new `src/summer4/_devices/__init__.py`;
- `src/summer4/__init__.py`, which imports it after `_precision` and exports `runtime_info`;
- the new `src/summer4/runtime.py`, which holds the `RuntimeInfo` dataclass and `runtime_info()`;
- the new `tests/test_runtime_devices.py`.

Implement D1. Every test runs in a fresh interpreter (`subprocess.run([sys.executable, "-c", ...])`)
with a controlled environment. The cases are:

- defaults give `jax.device_count() == os.cpu_count()`;
- `SUMMER4_HOST_DEVICES=2` gives 2, and `=0` gives 1;
- a user `XLA_FLAGS` setting of 3 is kept;
- importing jax and calling `jax.devices()` before summer4 gives a warning and 1 device;
- `runtime_info()` reports each `source` correctly.

**Sanity check:**

- `python -c "import summer4; from jax._src import xla_bridge as xb; assert not xb.backends_are_initialized()"`
  passes;
- `pixi run lint` passes;
- the new tests pass.

### R2 — default solver · sonnet, checker opus

**Files:**

- `src/summer4/solvers/resolve.py` (`resolve_solver(None, ...)`, `default_solver`,
  `set_default_solver`, `SUMMER4_SOLVER`);
- `src/summer4/solvers/__init__.py`;
- `src/summer4/flows/compiled.py` (the `run` default, plus a docstring that names the default
  and its one-line expansion);
- `src/summer4/runtime.py` (the `default_solver` field);
- the new `tests/test_default_solver.py`;
- callers.

1. Implement D2. Tests:
   - `run()` with no solver equals `run(solver=default_solver())` exactly;
   - the default reproduces an analytic exponential decay to `1e-6` relative;
   - the guide-09 orbit stays closed, with peak drift under `1e-4`;
   - `rtol=` / `atol=` override the default's tolerances;
   - `SUMMER4_SOLVER=euler` (in a subprocess) and `set_default_solver("euler")` restore Euler;
   - an explicit `solver=` wins over both;
   - it works under `jit` and `grad`;
   - `jax.make_jaxpr` of one `run` does not grow with the number of saves (50 against 500).
2. Run the full suite on the mini. For every failure, decide per test, writing the decision into
   the step's commit message:
   - if the test's expected values were hand-derived from Euler arithmetic, pin
     `solver="euler"`;
   - if it checks a model property (conservation, equilibrium, agreement with summer2), keep the
     new default and tighten or loosen the tolerance with a one-line reason.
   - Never weaken a test only to make it pass.
3. Do the same for `examples/notebooks/` and every page under `docs/`. Remove the solver
   workaround from guide page 09, and say in one sentence that the default handles it.

### R3 — chains use the devices · sonnet, checker opus

**Files:**

- `src/summer4/epi/calibration/model.py`;
- `src/summer4/epi/calibration/workflow/mcmc.py`;
- `src/summer4/epi/calibration/__init__.py` (exports `resolve_chain_method`);
- the new `tests/test_chain_method.py`.

Implement D3. Tests:

- `resolve_chain_method` gives the right method for device counts and chain counts, with
  `jax.local_device_count` monkeypatched;
- an explicit method is passed through;
- in a subprocess with 4 host devices, a 4-chain guide-08 fit with `"auto"` runs with
  `"parallel"` and recovers the parameter within the same interval as `"vectorized"`.

### R4 — docs, gate notebook, records · sonnet, checker opus

- **Docs page:** the new `docs/guide/10-runtime.md` ("What summer4 sets up for you"). It covers:
  - float64;
  - host devices;
  - the default solver;
  - chain method;

  and for each one, the environment variable and the function that change it. It ends with a
  GPU section (`JAX_PLATFORMS`, and what host devices do and don't do there).
- **Guide pages:** guide page 01 gains one sentence pointing at the new page, and page 08 notes
  that chains use every core.
- **Gate notebook:** `examples/notebooks/36-runtime-defaults.ipynb`, in the acceptance style
  (a *What to check* per section, no asserts). Its figures are:
  - device count before and after import;
  - Euler against the default against the reference, on Lotka–Volterra (phase plane);
  - R0's timing bars;
  - vectorized against parallel chain time.
- **Composability survey:** add two findings with the next free CX IDs, and set both `done`:
  - the default solver is silently inaccurate;
  - host devices are unused.

  Then run `pixi run composability-write`.
- **Coverage ledger:** update any row whose implementation names the old Euler default, then run
  `pixi run coverage-write`.
- **Release notes:** CHANGELOG `[Unreleased]` and `docs/releases.md` get a "Changed" section
  with the migration line `solver="euler"`, plus the environment variables.
- **futureplans:** in `guide-on-ramp-sharp-edges.md`, mark the solver item done. Then check
  whether `nuts-gate-test-flaky.md` and the MCMC notes change.

### R5 — checks, PR, handoff · orchestrator

- **Checks:**
  - `AGENTS.md` *Required checks*, with `test` and `docs-strict` run on the mini;
  - GitHub CI runs `latest`.
- **PR:** open it into `main`, citing step 49, the R0 numbers and the CX IDs.
- **Roadmap handoff:** step 49 `done`, step 28 `next`.
- **User gate:** notebook 36 and guide page 10.

## Decisions (user, 2026-11-11: accepted as proposed)

1. **Tolerances:** rtol `1e-6` and atol `1e-9`, unless R0's gate shows they are too slow. If it
   does, the user chooses.
2. **Calibration:** the new default solver applies inside calibration too, unless R0's gradient
   timings trip the gate.
3. **Release:** ship as `0.3.0a2` straight after merge.
