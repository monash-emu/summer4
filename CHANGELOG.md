# Changelog

All notable changes to summer4 are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers
follow [PEP 440](https://peps.python.org/pep-0440/). summer4 is an alpha: the
API is not stable.

Install from the matching git tag (see
[Installation](docs/getting-started/installation.md)). Narrative release notes
for the Sphinx site live under [docs/releases/](docs/releases/).

## [Unreleased]

### Added

- `ForceOfInfection(group_by=None)`, now the default, is homogeneous mixing over
  the whole population (`contact_rate * I_total / N_total` for the frequency
  kind), as in summer2. The dummy one-trait `pop` property and `[[1.0]]` matrix
  are no longer needed. The flow rate is a scalar and the saved force of
  infection is a `("time",)` series (`feat/foi-whole-population`; `CX15`, in
  part).

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
