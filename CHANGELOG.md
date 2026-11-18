# Changelog

All notable changes to summer4 are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers
follow [PEP 440](https://peps.python.org/pep-0440/). summer4 is an alpha: the
API is not stable.

Install from the matching git tag (see [docs/index.md](docs/index.md#install)).
Narrative release notes for the documentation site live in
[docs/releases.md](docs/releases.md).

## [Unreleased]

### Added

- **summer2-style output requests on `OutputSet`** (`feat/output-requests`, roadmap step 51,
  `CX49`, `CX50`). `compartments(name, where)`, `flow(name, flow, source=, dest=, values=)`,
  `aggregate(name, sources)`, `cumulative_integral(name, source, start=)`, `cumsum(...)` and
  `function(name, fn, *args)` are each one assignment; the docstring gives the expression, and
  each returns it so it chains like any other. `by=[age, reach, (age, reach)]` adds one output per
  stratum (`prevXage_15Xreach_far`; `naming=` changes the names), and `fan_out` does the same for
  any expression. `OutputSet(flow_values="midpoint")` gives summer2's flow-output convention.
  Raw assignments, `ref`, ufuncs and `defer` mix freely with requests.
- **Intermediates:** `OutputSet.set(name, expr, keep=False)` and `keep=` on every request
  (summer2's `save_results=False`). `kept()` lists the rest, `subset(names)` keeps only `names`,
  and `evaluate(result, params, names=)` returns `names` (default `kept()`), computing only what
  they read. `posterior_runs` records `kept()`; `BayesianModel` evaluates every name, so a target
  may score an intermediate.
- **`CompiledModel.run(..., outputs=out)`** solves for what `out` needs and returns its evaluated
  outputs: `out.evaluate(model.run(..., save=out.plan(save)), params)`.

- **Save any expression** (`feat/save-expressions`, roadmap step 40, `CX32`).
  `outputs[name] = expr` saves the value of any rate expression that reads time, state or flows,
  during the solve: a seasonal contact rate, `ForceOfInfection(...)` by age (one series per trait;
  `over` is read from the expression), `Reduce(...)`, `FlowRef(...).sum_by()`. In a mixed
  expression such as `Param("k") * Time() * outputs.ref("inc")`, the step-stage part is saved and
  the rest is applied afterwards. The quantity is `SavedExpr(expr, over=None)`; it shares the
  vector field's per-call values, so an expression a flow also uses is not recomputed.
- **`field_refs(expr, Schema)`, `Params()`, `StateArray()` and `Item`** (step 41, with the part of
  step 39 it needs, `CX31`). `field_refs(defer(f)(Params(), y=StateArray(), t=Time()), Schema)`
  computes several values from parameters, time and state in one function; `f` runs once per
  step however many rates read it (a per-call memo of shared nodes), and parameter-only parts of
  the model still hoist to run start. `field_refs(Params(), Schema)` is typed parameter access.

### Removed

- **`FlowModel.compile(derived_fn=)`, `derived_refs`, `DerivedFn`, `CompiledModel.derived_fn`,
  `computed_paths`, rule R3 and the `params_are_static` keyword** of `rate_stage` and
  `build_hoist_table` (step 41). Use `field_refs(defer(f)(Params(), y=StateArray(), t=Time()),
  Schema)`, and `field_refs(Params(), Schema)` in place of `derived_refs(Schema)`. A custom
  node's `__rate_stage__` takes no arguments.
- **`SaveFn` and `ComputedValue`** (step 40). Assign the expression (`outputs[name] = expr`) or
  save `SavedExpr(expr)`. The `EVERYTHING` save plan no longer saves every parameter path.
- Downstream code using any of these (for example the Kiribati port's `SaveFn` outputs) must move
  to the replacements when it upgrades.

### Changed

- `defer(f)(outputs.ref(...), Time())` no longer raises: the `Time()` argument is saved and
  reaches `f` as an `Output`.
- `OutputSet.evaluate` no longer returns names assigned with `keep=False`; pass
  `names=outputs.keys()` for every name. Sets without `keep=False` are unaffected.

## [0.3.0a5] — 2026-11-17

Tag: [`v0.3.0a5`](https://github.com/monash-emu/summer4/releases/tag/v0.3.0a5)

Native Windows support (`fix/windows-support`, PR #81).

### Added

- **Windows (x86_64, CPU) is a supported pixi platform.** `pixi.toml` lists `win-64`, and every
  environment in `pixi.lock` solves for it (jaxlib publishes a `cp313` `win_amd64` wheel). Before,
  `pixi install` refused to run on Windows. JAX labels its Windows wheels experimental, and native
  Windows GPU is unsupported (WSL2 is the route).
- **CI:** a `Test (Windows)` job runs the quick suite on `windows-latest` on every push.

### Fixed

- Tests read docs and notebooks as UTF-8 rather than the locale default (cp1252 on Windows, which
  fails on non-ASCII text). Ruff now requires an explicit encoding on text IO (`PLW1514`).
- `.gitattributes` keeps LF line endings on git hooks and shell scripts, which `sh` cannot run
  with CRLF from a Windows checkout.

## [0.3.0a4] — 2026-11-17

Tag: [`v0.3.0a4`](https://github.com/monash-emu/summer4/releases/tag/v0.3.0a4)

Multi-model calibration helpers (`feat/multi-model`, PR #79, roadmap step 50) and a `Property`
fix (PR #78).

### Added

- **Multi-model calibration helpers** on `BayesianModel` (`feat/multi-model`, plan
  `plans/multi-model-helpers.plan.md`, package `CP9`, closes `CX46`, `CX47`, `CX48`):
  `check_params` and `log_likelihood(strict=True)` (no silent fallback to `fixed_params`),
  `sample_params(prefix, given=, record=)` and `factor(name, params)` (prefixed sites, given
  hierarchical or shared values), and `local_params(draws, prefix)` with
  `posterior_runs(params_fn=)` to replay any member model from a joint posterior. Acceptance
  notebook `examples/notebooks/38-calibration-multi-model.ipynb`.
- **Guide page `docs/guide/11-several-models.ipynb`**, "Calibrating several models together":
  shared parameter names, strict name checks, a joint model with `sample_params` / `given=` /
  `factor`, choosing centred or non-centred per member, and replaying each model.

### Changed

- **CI:** the `latest` (current JAX) test environment no longer runs on pull requests. It runs
  nightly on `main` (`.github/workflows/nightly.yml`) and on demand.

### Fixed

- **`Property` is hashable when built from a list** (`fix/property-hashable`, plan
  `plans/property-hashable.plan.md`): `traits` is stored as a tuple whatever sequence is
  passed, so `Property("focus", ["low", "high"])` works as a `split=` key (it raised
  `unhashable type: 'list'`). A bare string for `traits` now raises instead of becoming one
  trait per character, and a `split=` that names one property twice (a `Property` and its
  name) now raises instead of keeping the last entry.

## [0.3.0a3] — 2026-11-16

Tag: [`v0.3.0a3`](https://github.com/monash-emu/summer4/releases/tag/v0.3.0a3)

Parameterised split proportions (`feat/param-split`, plan `plans/param-split.plan.md`).

### Added

- **`split=` proportions can be rate expressions** on `TransitionFlow` and `EntryFlow`: a `Param`,
  arithmetic such as `1 - Param("p")`, or a function of `Time()`. One flow now replaces the
  one-flow-per-destination workaround with complementary rates. Numeric proportions behave as
  before (they must be `>= 0` and sum to 1). Expression proportions are evaluated at run time and
  renormalised within each source compartment's destinations, so they act as relative weights;
  changing the parameter needs no recompile, and gradients flow through it.
- Guide page *Stratification* gains "Splitting a flow by a parameter" and "What an adjustment can
  be" (the `Multiply` / `Overwrite` / `Transform` list, `where=` with `Dest(...)` / `Source(...)`,
  and ordering). Acceptance notebook `examples/notebooks/37-param-split.ipynb`.

## [0.3.0a2] — 2026-11-11

Tag: [`v0.3.0a2`](https://github.com/monash-emu/summer4/releases/tag/v0.3.0a2)

Runtime defaults (roadmap step 49, `feat/runtime-defaults`, composability findings `CX43` and
`CX44`): an accurate solver and every CPU core, with nothing to configure. Guide page 10,
"What summer4 sets up for you", explains each default and how to change it.

### Changed

- **The default solver is now adaptive Tsit5 with a `PIDController`** (rtol `1e-6`, atol `1e-9`),
  not fixed-step Euler. Results move towards the exact solution, so numbers change for any model
  run without `solver=`. Migration: pass `solver="euler"` for the old numbers, or set
  `SUMMER4_SOLVER=euler` (or call `set_default_solver("euler")`) for a whole process. `dt` is still
  required: with the default it is the first step and sets the save grid.
- Importing summer4 now exposes **one CPU device per core** to JAX (it sets
  `--xla_force_host_platform_device_count`). `SUMMER4_HOST_DEVICES=0` turns it off; an `XLA_FLAGS`
  setting of your own is kept. Import summer4 before using JAX, or the setting comes too late
  (summer4 warns).
- MCMC `chain_method` accepts `"auto"` (the default) in `BayesianModel.sample` and `wf.run_mcmc`;
  `"auto"` is currently `"vectorized"` and `resolve_chain_method` is public. Parallel chains are
  opt-in and not yet compatible with gradients through the adaptive solver on recent JAX (they
  work with `solver="euler"`); see `futureplans/parallel-chains-diffrax-grad.md`.

### Added

- `summer4.runtime_info()`: whether 64-bit numbers are on, the host devices requested and why, and
  the default solver.
- `summer4.solvers.default_solver(rtol=, atol=)` and `summer4.solvers.set_default_solver(...)`.
- `summer4.epi.calibration.resolve_chain_method`.
- Environment variables `SUMMER4_HOST_DEVICES` and `SUMMER4_SOLVER`.
- Guide page 10 and acceptance notebook `examples/notebooks/36-runtime-defaults.ipynb`.

### Note

- A legacy `derived_fn` that turns parameters into Python floats (for example
  `float(params["beta"])`) fails under the adaptive default, because the values are traced by JAX.
  Keep them as JAX arrays, or pass `solver="euler"`.

## [0.3.0a1] — 2026-10-30

Tag: [`v0.3.0a1`](https://github.com/monash-emu/summer4/releases/tag/v0.3.0a1)

The clean surface (roadmap step 48): renamed spellings removed with no deprecation period,
documentation rebuilt in the unified idiom and published at
<https://monash-emu.github.io/summer4/>.

### Added

- Scalar `Reduce`: `Reduce(where=sel)` is the number of individuals in the matching
  compartments (one scalar, so `Param("a") * Reduce(where=sel)` is mass action), and
  `Reduce()` is the whole population. `Reduce(group_by, where=...)` is unchanged. Guide
  page 09 uses it instead of `ForceOfInfection` (`feat/clean-surface`, step 48).
- A GitHub Pages workflow builds the documentation on every push to `main` and
  publishes it to <https://monash-emu.github.io/summer4/>. A `docs` job on pull
  requests builds the same site as a downloadable artifact, so reviewers can read
  it before merge (`feat/clean-surface`, step 48).
- `tests/test_docs_idiom.py` fails if a page under `docs/` uses a removed or legacy
  spelling (`derived_fn`, `derived_refs`, `SaveFn`, `ComputedValue`, `sum_over`,
  `.total(`, `.cumulative(`, or a callable passed as `kind=`), so the documentation
  stays on the unified idiom as Phase M lands.
- Time-free outputs broadcast over time. An output reduced over time
  (`integrate()`, `at(t)`) has no `time` dim, and now combines with a series:
  `rate / rate.integrate()` is the share of the whole-period total at each time,
  `prev - prev.at(t)` the change since `t`, and
  `by_age / by_age.integrate().sum_by()` each age's share of the all-age total.
  It used to raise *Output time axes differ*, eagerly and in an `OutputSet`
  (`feat/time-free-outputs`; `CX39`).
- `at(when)` is a lazy post-op: `outputs.ref("prev").at(10.0)` and
  `Compartments(...).sum_by().at(10.0)` work in an `OutputSet`.
- `array()` and `scalar()` give an output's values as a plain array, eagerly
  (`Output.array()`) and lazily (`outputs.ref(name).array()`). Array nodes combine
  with numbers, `Param` expressions, NumPy ufuncs and `defer`; an `OutputSet` name
  that evaluates to an array is in `Result.arrays`, and `result[name]` finds it.
  `with_values(arr)` goes back: the output's times, dims and map with new values of
  the same shape, so the result can be plotted or scored by a `Target` (`CX40`).
- `Output.cumulative_integral(start=, end=)`: the running trapezoid integral of a
  rate over time, on the same save times and 0 at `start`. On a `FlowMass` it
  counts people whatever the save spacing. It equals
  `integrate_intervals().cumsum()` with a leading 0
  (`feat/cumulative-integral`).
- `Output.cumsum(start=, end=)`: the running sum `cumulative()` always was, for
  amounts per interval (`integrate_intervals()` output, `midpoint()` series,
  counts). Both are also on output expressions (`outputs.ref(...)`) and save
  specs (`FlowMass(...).sum_by().cumulative_integral()`).

### Changed

- Importing summer4 turns on JAX's 64-bit mode (`jax_enable_x64`), so models, outputs
  and likelihoods run in float64, as in summer2. Before, they ran in JAX's default
  float32, which in a compartment of 100 million moves in steps of 8 people. The flag is
  process-wide. For float32, set `JAX_ENABLE_X64=0` before starting Python, or call
  `jax.config.update("jax_enable_x64", False)` after importing summer4 and before
  compiling (`feat/float64-default`; `CX41`, deferred).
- The NUTS tuning gate (notebook 29) claims the Laplace start cuts dense warm-up
  trajectories by at least a third, not by half. In float64 the dense warm-up from the
  identity sometimes adapts quickly, leaving the Laplace start about 2x ahead; the
  diagonal claim (more than half) is unchanged.
- `at_times()` on a time-free output, and a `Target` on one, raise a message that
  says the output was reduced over time and how to score a whole-period total.
  A `Target` on an array output points at `with_values`.
- `defer(f)(...)` with no output argument (only `array()` values, parameters and
  numbers) returns `f`'s value as an array output instead of raising.
- The documentation is rebuilt from scratch on the unified idiom: a short guide
  (`docs/guide/`), the textbook chapters migrated to the same idiom, a roadmap page
  and release notes. The old tree (`docs/getting-started/`, `docs/user/`,
  `docs/cookbook/`, `docs/case-studies/`, `docs/summer2/`) is gone
  (`feat/clean-surface`, step 48).
- `InitialPopulation` and `Split` accept numbers, `Param` expressions or run-stage
  expressions only. Callable base values and weights are removed; use
  `defer(f)(Param(...))` instead.

### Removed

Each removal has a replacement that already works. No deprecation period: the
old spelling is gone from this release. See the
[migration table](docs/releases.md) for a before-and-after example of each.

- `Output.sum_over(prop)` and `Output.total()`: use `Output.sum_by(...)`.
- `sum_over` and `total` on `OutputSet` expressions, and on save specs
  (`total()`, `owner(sum_over=...)`): use `sum_by` and `owner(group_by=...)`.
- `PropertyData.sum_over(prop)`: use `PropertyData.sum_by(prop)`.
- `FlowRef.sum()` and `FlowRef.sum_over(prop)`: use `FlowRef.sum_by(...)`.
- `Reduce(sum_over=...)`: use `Reduce(group_by=...)`.
- `Output.cumulative()` on all three surfaces (`Output`, output expressions and save
  specs): use `cumulative_integral()` for a rate (people per unit time), or
  `cumsum()` for amounts per interval (for example after `integrate_intervals()`).
  On a rate, `cumulative()` gave about half the true count with saves two days apart.
- Callable base values and weights in `InitialPopulation` and `Split`: use a number,
  a `Param`, or `defer(f)(Param(...))`.
- `src/summer4/_deprecation.py` and the `warn_renamed` helper.

## [0.2.0a7] — 2026-10-29

Tag: [`v0.2.0a7`](https://github.com/monash-emu/summer4/releases/tag/v0.2.0a7)

### Added

- `ForceOfInfection(group_by=None)`, now the default, is homogeneous mixing over
  the whole population (`contact_rate * I_total / N_total` for the frequency
  kind), as in summer2. The dummy one-trait `pop` property and `[[1.0]]` matrix
  are no longer needed. The flow rate is a scalar and the saved force of
  infection is a `("time",)` series (`feat/foi-whole-population`, PR #65; `CX15`,
  in part).

### Fixed

- A calibration target can name an output of `BayesianModel(..., outputs=)`,
  such as a 7-day rolling sum of infections. It used to raise `KeyError`, and
  giving it the output's quantity instead pinned the output's input to the
  observation times, so time-window outputs (`rolling`, `cumulative`) were
  computed over the observations rather than the run. Outputs now save on the
  run's grid and such a target is read from the evaluated output;
  `BayesianModel.save_plan` shows the plan. `posterior_runs` takes the same plan
  (`fix/bayesian-output-targets`, PR #67).

### Changed

- `sum_by` is the one name for summing across strata, named for what it keeps
  (`feat/sum-by`, PR #68). `sum_by()` keeps only time (was `total()`, which kept time
  despite its name); `sum_by(age)` keeps one value per age (was `sum_over(age)`,
  which kept age despite its name); `sum_by(age, strain)` keeps each (age, strain)
  pair and is new. On `Output`, output expressions, save specs, `PropertyData` and
  `FlowRef` (`FlowRef.sum()` is now `sum_by()`). Time is never summed by
  `sum_by`; use `integrate`, `cumulative`, `rolling` or `at`.
- The save-time and rate-node keyword is `group_by=`: `Compartments(group_by=age)`,
  `FlowMass(..., group_by=(age, "source"))`, `Reduce(age)` / `Reduce(group_by=age)`.

### Deprecated

- `total()`, `sum_over()`, `FlowRef.sum()` and the `sum_over=` keyword still work
  and return the same results, with a `DeprecationWarning` naming the
  replacement. They will be removed in a later release.

## [0.2.0a6] — 2026-10-29

Tag: [`v0.2.0a6`](https://github.com/monash-emu/summer4/releases/tag/v0.2.0a6)

### Highlights

- **Lazy-idiom bridges (Phase L, steps 31–35).** The spellings that work on
  rates now work on likelihood scales and derived outputs: parameter
  expressions as scales, NumPy ufuncs on output expressions, `defer()` over
  outputs, `named()` for stable cache keys, and one walk over the rate tree.
- **Composable solving and calibration.** Solver backends hold your own diffrax
  objects; workflow stages take any `Posterior`; a NUTS tuning stage picks
  sampler settings from short pilots.

### Added

- Likelihood scales accept parameter-only rate expressions such as
  `np.exp(Param("log_k"))` (`fix/likelihood-scale-expr`, PR #55; `CX26`).
- NumPy ufuncs and the named helpers on `outputs.ref(...)`, and
  `Param * outputs.ref(...)` (`feat/output-ufuncs`, PR #56; `CX27`).
- `defer(f)(outputs.ref(...), Param(...))` builds a post-solve derived output
  (`feat/output-defer`, PR #57; `CX28`).
- `named(fn, name)` / `Named`: digest keys on the name, not `id(fn)`, for every
  callable hook (`feat/named-callables`, PR #58; `CX7`).
- `rate_children`, `rate_stage` and `build_hoist_table` exported from
  `summer4.flows`; custom nodes need only `__rate_children__` and
  `__rate_bytes__` (`fix/rate-children-fold`, PR #59; `CX29`).
- `summer4.solvers`: `Diffrax`, `Euler`, the `SolverBackend` protocol, public
  `SolveSpec`, `CompiledModel.assemble_result`, `Result.final_state` /
  `final_time` (`feat/solve-composable`, PR #44; `CX1`–`CX5`).
- `wf.OptimizeRun` (resumable), public `wf.OptimizeBackend` /
  `wf.OptimizeMethod`, `init_point`, `Candidates.from_z`
  (`feat/calib-composable`, PR #45).
- `Posterior` protocol and `NumpyroPosterior`; every workflow stage takes a
  `Posterior`; `BayesianModel.log_likelihood` (`feat/calib-posterior-protocol`,
  PR #60; `CX21`, `CX22`, `CX35`).
- NUTS tuning: `wf.tune_nuts`, `wf.NutsTuning`, `wf.NutsSetting`, `wf.pilot`,
  `wf.laplace_metric`, `wf.time_gradient`, `wf.metric_screen`,
  `wf.trajectory_screen`, `wf.warmup_until(abort=)` (`feat/nuts-tuning`,
  PR #61; `CX36`–`CX38`).
- The composability survey, `docs/evaluation/composability.md`, checked by
  `pixi run composability` (PR #43).

### Changed

- **Breaking:** `BayesianModel.sample` takes a numpyro `kernel` instead of
  `kind=` and returns a `wf.MCMCRun`; `BayesianModel.find_map` returns a
  `wf.OptimizeRun`; `OptimizeResult` is renamed `OptimizeRun` (PR #45).
- `ForceOfInfection` declares its children and step stage, so its
  parameter-only children hoist (PR #59).

### Fixed

- `SavePlan(ts=...)` is the save grid again (PR #49).
- A `FlowRef` inside a `ForceOfInfection` is ordered before the FOI flow, and
  capture metadata under `TableInterp`, `Lookup` and `Capture` is collected
  (PR #59).

### Known issues

- Every `MCMC.run` call, and so every `wf.sample_until` chunk, still pays a
  full XLA compile (`futureplans/mcmc-chunk-recompile.md`).

## [0.2.0a5] — 2026-09-28

Tag: [`v0.2.0a5`](https://github.com/monash-emu/summer4/releases/tag/v0.2.0a5)

### Highlights

- **Bayesian calibration (WP10).** Priors, target likelihoods, a
  `BayesianModel` with MAP and numpyro MCMC, and posterior scenario runs close
  the Kiribati and tb_macro calibration rows (`KI18`–`KI21`, `TM8`).
- **Composable calibration workflow (WP19, steps 24–27).** `Candidates`, Latin
  hypercube designs, batched scoring, multi-start Optax and CMA-ES
  optimisation, and seeded MCMC with automated warmup and run length — each a
  separate piece built on the underlying library's own objects
  (`from summer4.epi.calibration import workflow as wf`).
- **Solver safety.** Adaptive solves report failure through a traced
  `SolverInfo.ok` instead of silently returning partial results.

### Added

- Priors (`Uniform`, `Normal`, `LogNormal`, `TruncatedNormal`, `Beta`,
  `Gamma`) with `to_numpyro()`, `bounds` and `priors_from_frame`; target
  likelihoods (`Normal`, `Poisson`, `NegativeBinomial`) with hierarchical
  scales; traceable `TargetSet.log_likelihood` (`feat/epi-priors-likelihoods`,
  PR #32).
- `BayesianModel`: `log_density`, `find_map` (optax), `sample` (NUTS / AIES /
  ESS / SA → arviz) (`feat/epi-sampling`, PR #33).
- `BayesianModel.posterior_runs` with `Scenario`, per-draw samples, quantiles
  and differences (`feat/epi-posterior-runs`, PR #34).
- `wf.Candidates`, `wf.lhs`, `wf.prior_draws`, batched `wf.evaluate`,
  `Prior.icdf`, `BayesianModel.constrain` / `unconstrain`
  (`feat/calib-candidates`, PR #35).
- `wf.optimize` with `wf.Optax` and `wf.AutoTune` (chunked, vmapped
  multi-start with learning-rate probe, convergence freeze and restarts);
  public `BayesianModel.potential_fn` (`feat/calib-multistart`, PR #37).
- `wf.CMAES` gradient-free backend via evosax, behind a new `gradient-free`
  extra (`feat/calib-gradient-free`, PR #38).
- Seeded chunked MCMC (`feat/calib-seeded-mcmc`, PR #39): the caller builds
  numpyro's `MCMC`, and summer4 adds `Candidates.init_params`,
  `wf.warmup_until` + `wf.WarmupRule` (grow NUTS warmup until R-hat, step
  size agreement, acceptance, divergences and tree depth pass),
  `wf.sample_until` → resumable `wf.MCMCRun` with a per-chunk `progress`
  record, `wf.StopRule`, `wf.replay`, and `wf.run_mcmc` as a one-call
  convenience.
- `SolverInfo.ok` for adaptive solves; `run(..., throw=)` passes through to
  diffrax (`feat/solver-safety`, PR #28).
- TB-scale Kiribati-shaped benchmark baseline in `benchmarks/`
  (`feat/tb-scale-bench`, PR #27).

### Changed

- `compile(fuse_compartment_updates=True)` (the default) fuses per-flow `dy`
  scatters into one sparse scatter-add (`feat/fuse-incidence-scatter`, PR #31).
- Default `max_steps` is derived from `(t1 - t0) / dt` with headroom instead of
  a fixed 4096 (`feat/solver-safety`).
- `MixingMatrix` defaults to `check_reciprocal=False`; the reciprocity check
  is vmap-safe, and `MixingMatrix.validate` runs it on the host
  (`feat/solver-safety`).
- `AGENTS.md` records composability as a project-wide design goal.

### Known issues

- Every `MCMC.run` call, and so every `wf.sample_until` chunk, pays a full XLA
  compile inside numpyro. Prefer large chunks
  (`futureplans/mcmc-chunk-recompile.md`).

## [0.2.0a4] — 2026-09-22

Tag: [`v0.2.0a4`](https://github.com/monash-emu/summer4/releases/tag/v0.2.0a4)

### Highlights

- **TB-port surface.** Ageing sugar, generalised FOI (with per-compartment
  infectiousness), and output algebra / `OutputSet` are on a pinnable tag so
  Kiribati and tb_macro can leave commit SHA pins (`feat/ageing-sugar`,
  `feat/epi-generalised-foi`, `feat/epi-compartment-infectiousness`,
  `feat/trace-algebra`, `feat/output-sets`).

### Added

- `TraitChain.from_breakpoints` for uneven age-band ageing (`feat/ageing-sugar`,
  PR #15).
- `FOIKind.GENERALISED` force of infection with a population exponent
  (`feat/epi-generalised-foi`, PR #19).
- Selector-keyed infectiousness weights applied per compartment before the
  group sum (`feat/epi-compartment-infectiousness`, PR #24).
- Name-aligned `Output` arithmetic, `cumulative`, `rolling`, and
  `integrate_intervals` (`feat/trace-algebra`, PR #25).
- `OutputSet` so named derived outputs are a DAG over saved leaves
  (`feat/output-sets`, PR #26).
- NumPy ufunc / `__array_function__` dispatch on rate wrappers; `defer(fn)` for
  ordinary callables in a rate slot (`feat/rate-array-dispatch`,
  `feat/rate-defer`, PRs #21–#22).

### Changed

- **`Trace` renamed to `Output`.** Import `Output` (and `OutputSet`); there is
  no `Trace` alias on this tag (`feat/trace-algebra`).
- `np.ndarray * rate` builds one `BinOp` with an `ArrayConst`, not an
  object-dtype array of per-element nodes (`feat/rate-array-dispatch`).
- Custom rate nodes must define `__rate_bytes__` for the value-keyed jit digest
  (`fix/custom-rate-node-digest`, PR #20).

### Fixed

- Bare `adjust` traits on a split property bind to the destination
  (`fix/split-adjust-selector-side`, PR #23).

## [0.2.0a3] — 2026-09-21

Tag: [`v0.2.0a3`](https://github.com/monash-emu/summer4/releases/tag/v0.2.0a3)

### Highlights

- **`prepare()` boxes float params.** Plain Python `float` leaves in the params
  pytree are promoted to floating JAX arrays so Diffrax's equinox `filter_jit`
  cache hits across draws with `{str: float}` dicts. a2 fixed Module identity;
  a3 fixes the remaining static-float miss (`fix/prepare-box-float-params`).

### Fixed

- `CompiledModel.prepare` promotes Python `float` parameter leaves to floating
  JAX arrays so Diffrax's equinox `filter_jit` cache hits across draws with
  plain `{str: float}` dicts (`fix/prepare-box-float-params`, PR #16).

## [0.2.0a2] — 2026-09-20

Tag: [`v0.2.0a2`](https://github.com/monash-emu/summer4/releases/tag/v0.2.0a2)

### Highlights

- **Diffrax `run()` JIT cache.** Repeated `CompiledModel.run()` calls with the
  same save plan and solver settings now hit equinox's `filter_jit` cache. The
  Diffrax backend keeps process-stable equinox Modules for the vector field and
  `SubSaveAt` callbacks, and reads prepared parameters from Diffrax `args`, so
  a parameter draw no longer forces a recompile. On the SIR Euler 200-step
  probe this is one cold compile, then warm solves on the order of **1000×**
  faster.

### Added

- Rate-tree math (`tanh`, `clip`, `pow` and friends) that evaluates a
  parameter-only expression and can also scale a saved `Output`
  (`feat/rate-math`).
- A batched table interpolator so a per-age or yearly series is one node, not a
  rebuilt gather on every vector-field call (`feat/table-interp`).
- Example notebooks rewritten as readable user gates: summer2-style prose, a
  titled plot of each claim, and an assertion of the same claim
  (`docs/example-notebook-style`).
- Downstream smoke CI that installs summer4 from the git tag into a scratch
  pixi project (`chore/downstream-smoke-ci`).

### Fixed

- Diffrax `run()` rebuilt `ODETerm` / `SubSaveAt` callables on every call,
  keyed by Python identity, so equinox never reused a compiled solve
  (`fix/diffrax-run-jit-cache`, PR #13).

## [0.2.0a1] — 2026-09-18

Tag: [`v0.2.0a1`](https://github.com/monash-emu/summer4/releases/tag/v0.2.0a1)

First pinnable release of the flows stack on `main`: compartment taxonomy,
`FlowModel` / `CompiledModel`, results (`SavePlan`, `Result`, `Output`),
Diffrax solvers, epidemiology (`ForceOfInfection`, `MixingMatrix`),
time-varying rates, stratification, and honest packaging (JAX as a core
dependency; `frames` extra for polars / pyarrow).
