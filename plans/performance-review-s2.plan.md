---
name: summer2 performance suite
overview: "Cut `performance-review-s2` from `main` and add a float64, warm-JIT benchmark ladder: the same models in a summer2pixi environment (Euler and RK4) and in summer4 through fixed-step Diffrax, across step counts from 200 into the thousands, saving compartments and raw flow outputs."
todos:
  - id: step-1
    content: "Step 1: branch, summer2pixi shell, float64 probe, timing harness"
    status: pending
  - id: step-2
    content: "Step 2: shared spec and unstratified summer2 models"
    status: pending
  - id: step-3
    content: "Step 3: stratified summer2 models including stress"
    status: pending
  - id: step-4
    content: "Step 4: record the full summer2 timing matrix"
    status: pending
  - id: step-5
    content: "Step 5: summer4 Diffrax RK4 and unstratified ports"
    status: pending
  - id: step-6
    content: "Step 6: summer4 stratified ports"
    status: pending
  - id: step-7
    content: "Step 7: summer4 timings, comparison table, protocol docs"
    status: pending
isProject: false
---

# Summer2 vs summer4 performance suite

This is not roadmap step 6 (`feat/ageing-sugar`). That step stays `next`. This branch does not move the roadmap or the coverage ledger, and it does not change `src/summer4`.

Float64 is the dtype for ODE work. Both processes enable it before anything is compiled (`jax.config.update("jax_enable_x64", True)`). There is no float32 arm. This branch does not flip summer4's import-time default — that would move every existing tolerance — but the benchmark must not inherit float32 by accident.

## What "fair" means

summer2's reference is `CompartmentalModel.get_runner(..., solver=...)`, which jits the graph runner. Two fixed-step solvers, both already in [`summer2/runner/jax/solvers.py`](https://github.com/monash-emu/summer2/blob/main/summer2/runner/jax/solvers.py):

- `solver="euler"` — `solvers.euler` (`lax.scan`, one rate evaluation per step)
- `solver="rk4"` — `solvers.rk4` (classical four-stage Runge–Kutta)

`model.run()` without an explicit solver is not the reference: `build_run_model` remaps the default `solve_ivp` onto adaptive Dormand–Prince. The SciPy loops in `summer2/solver.py` are out.

summer4 must go through Diffrax. The string `solver="euler"` is the hand-rolled backend in [`euler_backend.py`](src/summer4/solvers/euler_backend.py), and [`resolve_diffrax_solver`](src/summer4/solvers/diffrax_backend.py) rejects that string. Passing `rtol` or `atol` selects `PIDController` (adaptive). The timed call therefore passes a Diffrax solver instance, a fixed `dt`, `steps`, and `max_steps`, and no tolerances, so [`diffrax_solve`](src/summer4/solvers/diffrax_backend.py) uses `ConstantStepSize`.

- Euler: `diffrax.Euler()`.
- RK4: Diffrax does not ship classical RK4. A small `AbstractERK` with the classical Butcher tableau (weights `1/6, 1/3, 1/3, 1/6`, nodes `0, 1/2, 1/2, 1`), defined in the benchmark package and passed as a solver instance. `diffeqsolve` still runs it. Do not substitute `Heun`, `Bosh3`, or `Tsit5` — those are different stage counts, so they would not match summer2's four evaluations per step.

The Diffrax call overhead (term, save, step-size controller) is part of what the summer4 number is supposed to show. Do not route either arm through `euler_solve`.

Same `dt` for both solvers, so a given step count is the same wall of time. RK4 does four rate evaluations per step; that extra work stays in the comparison.

Protocol, both sides:

- Enable float64 before the first compile. Record both JAX versions (summer2 stays on its 0.4.x pin; this repo's default env is JAX 0.6). Do not put them in one process.
- Step ladder, fixed `dt=0.1`, `t0=0`, for every model and both solvers: **200, 2_000, and 8_000** steps (`t1` of 20, 200, and 800). The Diffrax backend defaults `max_steps` to 4096, so every summer4 call passes `max_steps` at least as large as the step count. A run that hits that ceiling is not a result.
- Save the full compartment trajectory and one raw series per named flow (infection, recovery, and any death or importation the model declares), at every step. summer2: `request_output_for_flow` for each flow, `include_full_outputs=True`. summer4: a `SavePlan` with `Compartments` plus a `FlowMass` per flow. No cumulative sums, no per-capita rates, no age standardisation, no `OutputSet`. Those transforms are later work.
- The jitted function's inputs, the vector field, and the saves are JAX arrays (`jax` / `jnp` only). No NumPy inside the step, no host Python callbacks, no host interpolation. Host code may build constant index tables before compile; it does not run again inside the timed call. summer2 time-varying pieces are graph nodes the runner freezes (`summer2.functions`), not closures. If a shape cannot be traced, stop and record that — do not fall back to the slow path and call it the reference.
- Timed region is the warm call only. Build the model and the runner outside the timer. One untimed call compiles. The timed call ends in `jax.block_until_ready` on a reduced output (a sum of compartments and of each flow series), so XLA cannot delete the solve and async dispatch cannot report a short time.
- Also record, separately, host build time and first-call compile time. They are not mixed into the median.
- No adaptive solvers.

summer4 cannot put a mixing matrix on more than one property ([`futureplans/foi-multi-property-mixing.md`](futureplans/foi-multi-property-mixing.md)). Every stratified model mixes on **age only**; extra strata are real compartments but homogeneous on those axes. summer2 is built the same way, so the reference is not doing Kronecker mixing summer4 cannot express.

## Model ladder

One shared numeric spec (rates, strata names, mixing arrays, adjustment factors, time knots) so the two builders cannot drift. All adjustment factors are `1.0` and the time-varying contact rate is a shallow linear ramp, so the runs stay finite without becoming a calibration study. Each model is run at all three step counts and both solvers.

- `sir` — S/I/R, frequency infection, constant recovery. 3 compartments. Flows: infection, recovery.
- `sir_adjust` — same, plus a chain of 24 multiplicative adjustments on the contact rate.
- `sir_tv` — contact rate is linear in time (8 knots), as a JAX/graph interpolation, not a Python callable.
- `age_mix` — 16 age groups, static 16×16 mixing held as a `jnp` array. 48 compartments.
- `age_mix_tv` — same, mixing is a lookup of 8 matrices over time, indexed inside the jitted step.
- `stress` — S/I/R × 16 age × 20 location × 4 strain = **3840** compartments, age mixing only, plus the 24-adjustment chain, the time-varying contact rate, and the time-varying age mixing.

## Frozen numbers

Later steps must not renegotiate these. They live in one JSON file, written in step 2, and both codebases read that file.

- `t0 = 0`, `dt = 0.1`. Step counts are 200, 2_000, 8_000 (`t1` of 20, 200, 800). Smoke tests in steps 1–3 and 5–6 use 200 steps only, except step 1's probe which uses 20.
- Compartments `S`, `I`, `R`. Population `1_000_000`. Initial infectious `10`, the rest susceptible. Stratified models split both evenly across strata.
- `contact_rate = 0.35`, `recovery = 0.1`.
- Adjustments: 24 factors, every one `1.0`, multiplied onto the contact rate in order.
- Time-varying contact rate: 8 knots at times `0, 100, 200, 300, 400, 500, 600, 700` and values `contact_rate * [1, 1.05, 0.95, 1.1, 0.9, 1, 1.05, 1]`. Clamp outside the knots. This is a graph or JAX interpolation, not a Python callable.
- Age bands `a0`..`a15` (16). Static mixing is symmetric: `0.05 * ones((16, 16)) + 0.95 * eye(16)`. Equal population per band, so a reciprocity check can pass. Do not draw a random matrix.
- Time-varying mixing: 8 such matrices, the diagonal weight stepping `0.95, 0.9, 0.85, 0.8, 0.85, 0.9, 0.95, 0.9`, indexed inside the step by `floor(t / 100)` clamped to `0..7`.
- Locations `l0`..`l19` (20). Strains `s0`..`s3` (4). Infection stays inside a strain. Mixing is on age only.
- Named flows are `infection` and `recovery` only. No death, no importation, no ageing chain.

## How a session runs

Do one step, then stop. Do not start the next step in the same session.

1. Read the contract above, then that step's *Read first*, then the latest `## After step …` section of [`benchmarks/performance-handoff.md`](benchmarks/performance-handoff.md). The handoff is the living note. This plan is the contract and is not rewritten to record what you found — `plans/*.plan.md` is immutable once step 1 has copied it.
2. Do the work under *Do*. Where this file and an earlier guess disagree, this file wins.
3. Run the step's *Exit checks*. All of them must pass before you call the step done.
4. Append one `## After step N` section to the handoff file, covering every bullet under *Leave for the next worker*. Commit that with the step's code. A step with no handoff section has not finished.

Branch is `performance-review-s2`, cut from latest `origin/main`. No `src/summer4` edits, so the feature bar (notebook, ledger, roadmap) does not apply. `pixi run bench` stays the taxonomy suite. Model timings get their own task in step 7. [`benchmarks/results/`](benchmarks/results/) stays gitignored; committed numbers go in the JSON files named below.

If a requested summer2 shape cannot be traced by `get_runner`, stop and write that in the handoff. Do not substitute a host Python callback and call it the reference. If a summer4 jaxpr is obviously pathological, add a `futureplans/` note rather than fixing it in that step.

---

## Step 1 — summer2pixi shell

### Summary

Cut the branch, stand up a separate summer2 environment from [monash-emu/summer2pixi](https://github.com/monash-emu/summer2pixi), and prove the measurement path on one tiny SIR: float64, both fixed-step solvers, compartment and flow saves, one warm jitted call. No model ladder and no timings worth keeping.

### Read first

1. The contract and *Frozen numbers* above.
2. [summer2pixi `pixi.toml`](https://github.com/monash-emu/summer2pixi/blob/main/pixi.toml) — Python 3.10, `summerepi2>=1.3.6,<2`, plus estival, numpyro, and pymc. Keep that file. Do not redesign it.
3. summer2 `CompartmentalModel.get_runner` and `request_output_for_flow`. Request outputs before `get_runner`; `get_runner` finalises the graph.

### Do

- `git fetch origin` and `git switch -c performance-review-s2 origin/main`. Copy this plan to `plans/performance-review-s2.plan.md` on that branch. Do not edit the copy afterwards.
- Clone `monash-emu/summer2pixi` into a temp directory and copy its skeleton into `summer2bench/` without its `.git`. If the clone fails, stop.
- Add a pixi task that runs the probe. Invoke it as `pixi run --manifest-path summer2bench/pixi.toml …` so the parent summer4 env is never the one that imports summer2. No `test_*.py` under `summer2bench/` (parent `pixi run bench` collects those) and do not add `summer2bench/` to the ruff paths in the parent [`pixi.toml`](pixi.toml).
- The probe enables `jax_enable_x64` before importing summer2. It builds S/I/R with frequency infection and recovery, requests both flow outputs, then `get_runner(..., solver="euler")` and `get_runner(..., solver="rk4")`. Twenty steps, `dt=0.1`. One untimed call, then one call whose result is passed to `jax.block_until_ready`.
- Put the three-timer split in `summer2bench/time_run.py` even though this step only uses it for the probe: build time is model construction plus `get_runner` and excludes the first call; compile time is that first call; warm time is the later call. Later steps import this module rather than reinventing it.
- After `pixi install`, record the locked `summerepi2` version. Do not replace the template's PyPI requirement with a git URL unless `get_runner` is missing from that lock — in that case, stop.

### Exit checks

- The probe prints `float64` for the compartment output and for both flow series, under both solvers.
- Both reduced sums are finite.
- `pixi run lint` and `pixi run bench` from the repo root still do not import summer2.

### Leave for the next worker

Append `## After step 1` to `benchmarks/performance-handoff.md`:

- Locked `summerepi2` version, JAX version, and the command that runs the probe.
- The exact `get_runner` call, including how `solver` is passed (it is a backend kwarg, not a positional).
- Shapes and dtypes of `model.outputs` and of each requested flow after `runner.run`.
- Anything dropped or broken in the template (pymc install, platforms, notebooks).
- Confirmation that outputs were requested before `get_runner`.

---

## Step 2 — shared spec and unstratified summer2 models

### Summary

Write the JSON spec and the three models that do not stratify: `sir`, `sir_adjust`, `sir_tv`. Smoke them at 200 steps on both solvers. Do not time the ladder and do not build age or stress models.

### Read first

1. The contract, *Frozen numbers*, and `## After step 1`.
2. `summer2bench/time_run.py` from step 1. Extend it; do not write a second timer.
3. summer2 parameter-graph helpers (`summer2.parameters`, `summer2.functions`). The adjustment chain and the time-varying rate have to be nodes that `get_runner` freezes.

### Do

- Write `summer2bench/spec.json` with every frozen number: rates, the 24 adjustment factors, the 8 knots, age labels, the static mixing matrix, the 8 time-varying matrices, location labels, strain labels. Both later codebases read this file. Do not duplicate those numbers as literals in model code.
- `sir` — frequency infection, constant recovery, parameters from the spec.
- `sir_adjust` — the same, with the 24 factors multiplied onto the contact rate in order, as graph nodes. A Python loop may *build* that chain. It must not run inside the step.
- `sir_tv` — contact rate is the 8-knot interpolation from the spec, clamped outside the knots, as a summer2 function the runner traces.
- Each model requests `infection` and `recovery` before `get_runner`. Smoke: 200 steps, both solvers, warm call, finite sum of compartments and of each flow. Assert compartment count 3 and that each flow series has length 201 (the initial time plus 200 steps).

### Exit checks

- Spec file is the only place the frozen numbers are written.
- All three smokes pass in the summer2 env, float64.
- A breakpoint in the adjustment chain is not a host callback: the runner builds without a Python function being invoked per step. If you cannot show that, say so in the handoff and stop.

### Leave for the next worker

Append `## After step 2`:

- Path and top-level keys of `spec.json`.
- The class or function that built the 24-multiply chain, and the one that built the interpolation, with a one-line call shape.
- Parameter dict keys `get_runner` actually consumed.
- Flow-output shapes confirmed at 200 steps.
- Any graph feature that looked like it would not survive stratification (so step 3 does not rediscover it on the large model).

---

## Step 3 — stratified summer2 models

### Summary

Add `age_mix`, `age_mix_tv`, and `stress`, still summer2 only, still smoke-only. This session is the stratification and mixing API. It does not run 2_000 or 8_000 steps.

### Read first

1. `## After step 2` and `summer2bench/spec.json`.
2. summer2 `Stratification`: `mixing_matrix` on age only. Location and strain get no mixing matrix.
3. The contract note on age-only mixing. Do not attach a second mixing matrix to match a Kronecker product summer4 cannot express.

### Do

- `age_mix` — stratify age into the 16 spec bands, static mixing matrix from the spec, equal split of the initial population. Expect 48 compartments.
- `age_mix_tv` — the same, but the mixing matrix is the spec's stack of 8, indexed by `floor(t / 100)` inside the traced step. If the runner only accepts a host callable for `mixing_matrix`, stop and record that. Do not time the callable path.
- `stress` — age × location × strain on top of S/I/R, infection within strain, age mixing is the time-varying stack, contact rate carries both the 24 adjustments and the time-varying interpolation. Expect **3840** compartments. Assert that count from `model.compartments` rather than from arithmetic you hope is right.
- Smoke all three at 200 steps, both solvers, through `time_run.py`. Record build time in the handoff even though this step does not keep a results table. A stress build that takes many minutes is a fact step 4 needs.

### Exit checks

- Compartment counts are 48, 48, and 3840.
- Flow series length 201, float64, finite sums, both solvers.
- Mixing matrices in the traced model are the spec arrays, not a freshly randomised matrix.

### Leave for the next worker

Append `## After step 3`:

- The `mixing_matrix` argument that `get_runner` accepted for the static and the time-varying cases.
- How within-strain infection was expressed.
- Measured build time of `stress` (construction plus `get_runner`, not the solve) for each solver.
- Anything that failed to trace. A failed model is a blank cell, not a silently simpler model.

---

## Step 4 — summer2 timing matrix

### Summary

Run the full summer2 matrix and write the numbers down. No new model structure. No summer4 code.

### Read first

1. `## After step 3`, especially stress build time and any blank cells.
2. `summer2bench/time_run.py`.
3. The step ladder in *Frozen numbers*: 6 models × 2 solvers × 3 step counts = 36 cells, minus whatever step 3 marked untraceable.

### Do

- For each cell, record build time, compile time (first call), and the median of 5 warm calls. Each warm call ends in `block_until_ready` on the sum of compartments and of both flow series. The first call is not one of the five.
- Write `summer2bench/recorded.json` (committed, not under `benchmarks/results/`). Each record has model, solver, steps, compartments, build_s, compile_s, warm_median_s, jax version, `summerepi2` version, dtype `float64`, machine, date. A cell that runs out of memory or does not finish is an explicit `status: failed` with the reason. Do not shrink the model or the step count to force a number.
- Do not start summer4 work if a cell fails. Record it and stop the matrix at a sensible checkpoint (finish the model you are on, then hand off). The JSON is the result even if some cells are failures.

### Exit checks

- JSON contains every cell, including failed ones. No missing keys.
- A spot check: the `sir` / euler / 200 warm median is smaller than its compile time, and the output dtype is float64. If the warm call is slower than compile, the timer is including compilation and the matrix must be rerun.

### Leave for the next worker

Append `## After step 4`:

- Path to `recorded.json` and the count of ok versus failed cells.
- The `sir` / euler / 200 and `sir` / rk4 / 200 warm medians, as a sanity pair the summer4 session can compare against.
- Peak memory or an OOM note for `stress` at 8_000 steps, if you saw one.
- Machine description and the date, copied from the JSON so the next session does not have to parse it to know whether the numbers are usable.

---

## Step 5 — summer4 Diffrax path and unstratified ports

### Summary

Open the summer4 side. Add a classical RK4 that Diffrax will run with a fixed step, and port `sir`, `sir_adjust`, and `sir_tv`. Smoke at 200 steps. Do not port the stratified models and do not run the long ladder.

### Read first

1. `## After step 4` and `summer2bench/spec.json`. Read the spec; do not retype the numbers.
2. [`src/summer4/solvers/diffrax_backend.py`](src/summer4/solvers/diffrax_backend.py). `solver="euler"` never reaches Diffrax. `rtol` or `atol` selects `PIDController`. Fixed step is a solver instance, `dt`, `steps`, `max_steps`, and no tolerances, which selects `ConstantStepSize`.
3. [`FlowMass`](src/summer4/results/plan.py) and a `CompiledModel.run` call in [`tests/test_solvers.py`](tests/test_solvers.py). FlowMass without a reduction stores every edge. summer2's flow output is one series per flow. Match that one series. Reducing to the flow total inside the save request is the output, not a further transform. Cumulative sums, per-capita rates, and `OutputSet` are still out.

### Do

- Enable `jax_enable_x64` at import of the benchmark modules, before the first `jit`. Assert a saved compartment array is float64.
- `benchmarks/diffrax_rk4.py` — `AbstractERK` with the classical tableau (weights `1/6, 1/3, 1/3, 1/6`, nodes `0, 1/2, 1/2, 1`). Dummy error weights are acceptable only because the call uses `ConstantStepSize`. Do not pass this solver with `rtol`. Not `Heun`, not `Tsit5`.
- `benchmarks/models.py` — builders for the three unstratified models, reading `summer2bench/spec.json`. Contact rate, adjustments, and the interpolation are `jnp` / rate nodes. `linear` from [`src/summer4/timevarying.py`](src/summer4/timevarying.py) is the interpolation; do not call NumPy `interp` inside the step.
- Save plan: compartments plus one `FlowMass` series for `infection` and one for `recovery`.
- Run kwargs: `solver=diffrax.Euler()` or the RK4 instance, `t0=0`, `dt=0.1`, `steps=200`, `max_steps` at least 200. No `rtol`, no `atol`, no `solver="euler"`.
- Smoke both solvers. Finite sum, compartment count 3, each flow series length 201, float64.
- A short `benchmarks/harness.py` with the same three-timer split as `summer2bench/time_run.py`. Step 7 uses it. This step only smokes.

### Exit checks

- Grep the new benchmark modules for `solver="euler"` and for `rtol=`. Neither may appear in the timed or smoke calls.
- Smokes pass under `pixi run` in the default summer4 env (JAX 0.6), with x64 on.
- `pixi run lint` on the new files. No `src/summer4` diff.

### Leave for the next worker

Append `## After step 5`:

- Import path of the RK4 class and the exact `run(...)` keyword arguments that hit `ConstantStepSize`.
- How `FlowMass` was reduced to one series per flow, and the resulting shape.
- Which rate node implemented the 24 adjustments and which implemented the 8-knot interpolation.
- Confirmed float64 dtype under `jit`.

---

## Step 6 — stratified summer4 ports

### Summary

Port `age_mix`, `age_mix_tv`, and `stress`. Smoke at 200 steps, both Diffrax solvers. No full timing matrix.

### Read first

1. `## After step 5` and `## After step 3` (the summer2 mixing and strain mechanism you are matching).
2. [`src/summer4/epi/mixing.py`](src/summer4/epi/mixing.py) and [`Lookup`](src/summer4/flows/rates.py). A `Lookup` of a `(8, 16, 16)` table is the time-varying mixing matrix. The index is `floor(Time() / 100)` clamped to `0..7`, evaluated inside the jitted step.
3. [`futureplans/foi-multi-property-mixing.md`](futureplans/foi-multi-property-mixing.md). `ForceOfInfection.group_by` is age only. Location and strain are further `stratify` calls with no second mixing matrix. Within-strain infection is `ForceOfInfection.per_trait` or an equivalent selector, matching what step 3 actually used.

### Do

- Extend `benchmarks/models.py`. Same spec file. Static mixing is a `jnp` array from the spec, not a NumPy array rebuilt every call.
- Assert compartment counts 48, 48, and 3840 after `compile()`.
- Smoke 200 steps, both solvers, `max_steps>=200`, flow series length 201, float64, finite sums.
- If `stress` compile is so large that a 200-step smoke cannot finish in this session, stop, leave the builder in place, and record the failure. Do not delete strata to make it compile.

### Exit checks

- Counts 48 / 48 / 3840.
- Smokes that you claim passed were actually run. The handoff lists any that were not.
- No `src/summer4` diff. A pathological jaxpr gets a `futureplans/` note, not a solver rewrite.

### Leave for the next worker

Append `## After step 6`:

- Builder names and the compile-time compartment counts.
- Whether the time-varying mixing `Lookup` is inside the jitted step (it must be) and the table shape you passed.
- 200-step smoke status per model per solver.
- Any `futureplans/` file added, by path.

---

## Step 7 — summer4 timings and the comparison write-up

### Summary

Run the summer4 matrix with the same timer as summer2, then write the comparison table and the protocol doc. This session does not change models.

### Read first

1. `## After step 4` (`summer2bench/recorded.json`) and `## After step 6`.
2. `benchmarks/harness.py` and `summer2bench/time_run.py`. The warm median is the median of 5 calls after one discarded compile, with `block_until_ready`. Use that definition for the committed numbers. pytest-benchmark may drive the calls, but the table does not quote a different statistic than the summer2 JSON.
3. [`docs/dev/benchmarking.md`](docs/dev/benchmarking.md) and [`pixi.toml`](pixi.toml) tasks `bench` and `bench-json`.

### Do

- Add `pixi run bench-models`, pointing at the model file only. Do not fold these cases into `pixi run bench` or into [`docs/dev/performance.ipynb`](docs/dev/performance.ipynb).
- 6 × 2 × 3 cells, same failure rule as step 4: `status: failed` rather than a smaller model. `max_steps` at least the step count on every call (the backend default of 4096 would truncate 8_000).
- Write `benchmarks/recorded-summer4.json`, committed, same fields as the summer2 file plus `solver` values `diffrax-euler` and `diffrax-rk4`.
- [`benchmarks/README.md`](benchmarks/README.md): one table, both libraries, columns model, solver, steps, compartments, build, compile, warm median, dtype, JAX version. State the machine and the date. Do not claim a winner in prose beyond what the table shows. Note the two JAX versions and that summer4's number includes Diffrax.
- Update [`docs/dev/benchmarking.md`](docs/dev/benchmarking.md) with the protocol: float64, warm JIT, fixed-step Diffrax versus summer2 Euler and RK4, the step ladder, compartment plus one raw flow series, age-only mixing, and where the JSON files live.

### Exit checks

- `pixi run bench` does not run the model matrix.
- `pixi run lint` and `pixi run format-check` pass. `pixi run check-branch` passes because `src/summer4` is untouched.
- The README table has a row for every cell in both JSON files, failures included.
- `pixi run roadmap` is unchanged and still reports step 6 as next. Do not edit [`docs/dev/roadmap.md`](docs/dev/roadmap.md).

### Leave for the next worker

Append `## After step 7`. There is no step 8. The note is the close-out:

- Paths of both JSON files and of the README table.
- Count of failed cells on each side.
- The `sir` / 200 warm medians for summer2 Euler, summer2 RK4, Diffrax Euler, and Diffrax RK4, so a later reader can see the overhead without opening the JSON.
- Any `futureplans/` note from step 6, restated in one line.
