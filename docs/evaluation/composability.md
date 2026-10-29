# Composability survey

**This page is the authoritative record of where summer4's public API falls
short of the composability design goal** in `AGENTS.md`
(§ *Composability is a design goal*), and of the order in which to fix it. Read
it before designing or extending a public API, alongside
{doc}`coverage-ledger` (what summer4 can do) and {doc}`../dev/roadmap` (what to
do next).

Cite findings by ID (`CX1`) and work packages by ID (`CP1`) in plans and pull
request descriptions. IDs are stable: a fixed finding keeps its row and its ID
and changes status. **A branch that fixes a finding sets its status to `done` in
the same commit**, then runs:

```bash
pixi run composability-write   # recompute the totals line
pixi run composability         # statuses, evidence anchors, packages, totals
```

`tests/test_composability_survey.py` runs the same check in `pixi run test`.

## The four principles

| ID | Principle | In short |
|---|---|---|
| P1 | Accept the underlying library's objects | Take the numpyro kernel, optax optimiser or diffrax controller the user built, not a string `kind=` or a hand-picked subset of its arguments |
| P2 | One policy per piece | Seeding, stopping, retrying, recording and validation are separate public pieces; a private helper that tests reach into should be public |
| P3 | Long-running work is inspectable and resumable | Return an object that holds the caller's objects and can be extended, not a sealed result |
| P4 | Conveniences are built from the pieces | One-call entry points for modellers are implemented from the public pieces, document the lines they stand for, and test that equivalence |

## Vocabulary

| Column | Values |
|---|---|
| Severity | `high` — blocks a real modelling or calibration use case, or forces private access; `medium` — awkward, a workaround exists; `low` — consistency or discoverability |
| Status | `open` — not scheduled; `planned` — a plan under `plans/` cites it; `done` — fixed (keep the row); `deferred` — tracked by a `futureplans/` note it cites; `rejected` — deliberately kept, with the reason in the row |
| Evidence | `path::anchor` — the anchor is literal text the checker finds in that file while the finding is not `done`, so a fix that removes it forces the row to be updated; a bare `futureplans/…md` path for deferred rows |

## Where things stand

<!-- composability:totals -->
Open findings: **7** (0 high, 2 medium, 5 low); planned: 6; deferred: 4; done: 22; rejected: 0.
<!-- /composability:totals -->

The calibration workflow's MCMC stage (`CX25`), the solve (`CP1`, roadmap step
29) and the calibration entry points (`CP2`, step 30) meet the goal: the caller
builds the diffrax, numpyro and optax objects; `run`, `sample`, `find_map` and
`optimize` are documented compositions of public steps and return objects that
can be inspected and extended. What remains:

- extension points (`CP3`; step 46 closed `CX21` and `CX35`);
- tests that still reach private helpers (`CP4`);
- the staged unification of summer4's lazy idioms: bridges between the rate tree and the
  post-solve outputs (`CP5`, roadmap steps 31-35), then one node set for rates, outputs and
  parameter-dependent values (`CP6`, steps 36-44).

## Findings

<!-- composability:findings -->
| ID | Area | Principle | Severity | Status | Finding | Proposed shape | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CX1 | run | P1 | high | done | Before step 29: `run()` builds the stepsize controller itself: `PIDController(rtol, atol)` when either is set, else `ConstantStepSize`. A caller's own controller (PID coefficients, `dtmin`, jump times) cannot be passed. | Done in step 29: `Diffrax(solver, stepsize_controller=...)`; `rtol` / `atol` build a `PIDController` via `resolve_solver`. | `src/summer4/solvers/diffrax_backend.py::class Diffrax` |
| CX2 | run | P1 | high | done | Before step 29: `diffeqsolve` is called with no `adjoint=`, `event=` or `progress_meter=`. Calibration gradients always use diffrax's default adjoint, and a run cannot stop on a state event. | Done in step 29: `Diffrax(..., adjoint=, event=, progress_meter=)`; `SolverInfo.event`, and `ok` counts an event stop as success. | `src/summer4/solvers/diffrax_backend.py::extra["adjoint"]` |
| CX3 | run | P1, P2 | medium | done | Before step 29: Solver choice is one argument that is either a name or a diffrax instance; summer4's own Euler backend is reachable only by the string `"euler"`, and the solve settings (`SolveSpec`) are internal. | Done in step 29: public `SolverBackend` protocol, `Euler()` / `Diffrax(...)` backends, public `SolveSpec.window`, `resolve_solver` for names. | `src/summer4/solvers/base.py::class SolverBackend` |
| CX4 | run | P2, P4 | medium | done | Before step 29: `CompiledModel.run` is one long method: prepare, expand the save plan, group requests, build the time axis, build a `SolveSpec`, dispatch, assemble the `Result`. No intermediate step is public, so changing one (say, the `SaveAt`) means forking `run`. | Done in step 29: `run` is `prepare` → `expand` → `SolveSpec.window` → `resolve_solver` → `backend.solve` → `assemble_result`, documented and tested. | `src/summer4/flows/compiled.py::def assemble_result` |
| CX5 | run | P3 | medium | done | Before step 29: A `Result` carries no end state. Continuing a run needs a hand-saved `Compartments()` at `t1`, re-wrapped as `y0`. | Done in step 29: `Result.final_state` / `final_time`; `run(params, r.final_state, t0=r.final_time, ...)` continues a run. | `src/summer4/results/result.py::final_state` |
| CX6 | rates | P2 | medium | open | Extension and staging points are not exported from `summer4.flows`: `register_rate_eval` (the custom-rates cookbook imports it from `summer4.flows.rates`), and `rate_stage` / `build_hoist_table` (tests import them from `summer4.flows.stages`). A modeller cannot ask which stage a rate will run in. | Export them, or add a `CompiledModel.hoist_table` / stage-report view. | `src/summer4/flows/rates.py::def register_rate_eval`; `src/summer4/flows/stages.py::def rate_stage` |
| CX7 | rates | P1 | medium | done | Callables are keyed by identity with no stable-name option: `SaveFn.fn` digests `id(fn)`, as do `Transform.fn` and `derived_fn`, so a lambda built in a loop retraces every call. `defer` already solves this with `name=`. | Done in step 34: `named(fn, name)` is accepted by every callable hook; digests key on the name via `_callable_key`. | `src/summer4/flows/rates.py::def _callable_key` |
| CX8 | rates | P2 | low | open | Tests import private flows helpers (`_align_rate`, `_align_grouped_rate`, `_eval_interp`, `_rate_bytes`). Most are true internals. | Triage: promote what a custom-rate author needs, test the rest through public behaviour. | `src/summer4/flows/compiled.py::def _align_rate` |
| CX9 | results | P1 | low | deferred | `Output.plot` hard-codes matplotlib calls around a pandas plot whose backend may be Plotly. | See the note. | `futureplans/trace-plot-backend-coupling.md`; `src/summer4/results/output.py::def plot` |
| CX10 | results | P2 | low | open | The test loss helper rebuilds a state with private `PropertyData._with_data` where public `PropertyData` arithmetic already does it. | Use `y0 * weights` in the helper; no API change. | `tests/helpers/loss.py::_with_data` |
| CX11 | epi | P1 | medium | planned | A custom `ForceOfInfection` `kind` callable receives no parameters, so a parameterised custom shape (learned exponents, saturation) is impossible unless it is `FOIKind.GENERALISED`. | Let `kind` build a rate expression, `kind(infectious, denominator) -> RateOps`, so `Param`s flow through the rate tree and stage normally. Planned as `custom_kind(fn, *rate_args)` (roadmap step 42). | `src/summer4/epi/infection.py::is the custom path` |
| CX12 | epi | P1 | low | open | `normalize_infectiousness` and `MixingMatrix(normalize=)` are closed `StrEnum`s; any other normalisation means forking. | Accept a callable alongside the enum members. | `src/summer4/epi/infection.py::class InfectiousnessNormalize`; `src/summer4/epi/mixing.py::class MixingNormalize` |
| CX13 | epi | P2 | low | open | `apply_compartment_weights` and `coerce_compartment_weights` are in `summer4.epi.__all__`, but the `eval_child` callback contract is undocumented and no user-facing page uses them. | Document the contract with an example, or drop them from `__all__` until the susceptibility surface lands. | `src/summer4/epi/__init__.py::"apply_compartment_weights"` |
| CX14 | epi | P2 | medium | deferred | `MixingMatrix` row-normalises a `FieldRef` matrix inside the vector field on every call instead of once at run start. | See the note. | `futureplans/mixing-matrix-per-call-normalisation.md`; `src/summer4/epi/mixing.py::def resolved_matrix` |
| CX15 | epi | P1 | medium | deferred | `ForceOfInfection(group_by=)` takes one `Property`, so mixing across several properties at once cannot be expressed. (The whole-population case no longer needs a dummy property: `group_by=None`, `plans/foi-whole-population.plan.md`.) | See the note. | `futureplans/foi-multi-property-mixing.md` |
| CX16 | calibration | P1, P4 | high | done | Before step 30: `BayesianModel.sample(kind=...)` dispatches on a string, passes through only `**kernel_kwargs`, cannot be seeded, and duplicates the composable MCMC stage. | Done in step 30: `sample(kernel=, init=, stop=, ...)` takes a caller-built numpyro kernel (names are sugar), seeds from `Candidates`, and is exactly `wf.run_mcmc`; returns the `MCMCRun`. | `src/summer4/epi/calibration/model.py::make_kernel=lambda: kernel_object` |
| CX17 | calibration | P2 | high | done | Before step 30: The `wf.optimize` backend protocol is private (`_Method`) and `method=` is typed as the two built-ins, so a user cannot plug in another optimiser and reuse the chunk loop, convergence and restarts. Tests reach `_chunk_program` and `_tree_set`. | Done in step 30: public `OptimizeBackend` and `OptimizeMethod` protocols; any object with `make` runs through `wf.optimize`; `run.chunk_program` replaces the private helpers tests used. | `src/summer4/epi/calibration/workflow/optimize.py::class OptimizeBackend(Protocol)` |
| CX18 | calibration | P3 | medium | done | Before step 30: `wf.optimize` returns no optimiser state, so a run cannot be extended; restarting from `result.candidates` resets Adam moments or the CMA-ES covariance. | Done in step 30: `wf.optimize` returns a `wf.OptimizeRun` holding the optimiser states; `run.extend(max_steps)` continues with Adam moments / CMA-ES state intact. | `src/summer4/epi/calibration/workflow/optimize.py::def extend(self, max_steps: int)` |
| CX19 | calibration | P3, P4 | medium | done | Before step 30: `BayesianModel.find_map` returns only the final parameters (no loss trace, no state, no resume) and duplicates single-start `wf.optimize`. | Done in step 30: `find_map` is a one-start `wf.optimize` returning the `OptimizeRun` (`best_params` for the dict); `init_point` and `Candidates.from_z` are public. | `src/summer4/epi/calibration/model.py::def init_point(` |
| CX20 | calibration | P1, P2 | medium | open | Likelihoods have no protocol, and nested prior scales are found only through attributes named `sd` or `dispersion`. A custom likelihood with a differently named prior scale silently loses that prior. | A `Likelihood` protocol with `log_prob(...)` and `prior_sites()`. | `src/summer4/epi/calibration/likelihoods.py::def prior_sites` |
| CX21 | calibration | P1 | low | done | Before step 46: A prior must implement the `Prior` protocol; a bare numpyro distribution needs a hand-written wrapper. | Done in step 46: `NumpyroPrior(name, dist)` wraps any numpyro distribution; `icdf` uses the distribution's own and `has_icdf` says when there is none. Not `Prior.from_numpyro`: a static method on the `runtime_checkable` Protocol would become a protocol member that every structural prior lacks. | `src/summer4/epi/calibration/priors.py::class NumpyroPrior` |
| CX22 | calibration | P2 | low | done | Before step 46: Tests call private `BayesianModel._ensure_potential()` for the initial point and the postprocess transform. | Done in step 46: public `postprocess_fn` beside `potential_fn` and `init_point`, on `BayesianModel` and `NumpyroPosterior`; tests use them. | `src/summer4/epi/calibration/model.py::def postprocess_fn` |
| CX23 | calibration | P2 | low | open | How a `Scenario` rebinds params, model and outputs lives in private `_scenario_bindings`, so a scenario cannot be applied outside `posterior_runs`. | A public binding function, or a method on `Scenario`. | `src/summer4/epi/calibration/posterior_runs.py::def _scenario_bindings` |
| CX24 | calibration | P3 | low | done | Before step 30: `OptimizeResult.history` holds only the optimise record, while its `candidates.history` is cumulative. | Done in step 30: `run.history` is the candidates' cumulative history, one record per `optimize` / `extend` call. | `src/summer4/epi/calibration/workflow/optimize.py::history=self._input_history + tuple(self._records)` |
| CX25 | calibration | P1, P2, P3, P4 | high | done | The MCMC stage took `kind=` strings, bundled seeding, retries and stopping, and returned a sealed `SampleResult`. It is now the reference pattern: caller-built numpyro `MCMC`, `Candidates.init_params`, `wf.warmup_until`, `wf.sample_until` → `MCMCRun`, `wf.StopRule`, `wf.replay`, and `wf.run_mcmc` (PR #39). | — | `src/summer4/epi/calibration/workflow/mcmc.py::def sample_until` |
| CX26 | calibration | P1 | low | done | `resolve_scale` resolves a `Prior`, a bare `Param`/`FieldRef` or a number; any other rate expression (`np.exp(Param("log_k"))`, `Param("a") * 2`) is returned unevaluated as the likelihood scale. | Done in step 31: every rate expression is evaluated through `eval_closed`; a step-stage expression raises. | `src/summer4/epi/calibration/likelihoods.py::def resolve_scale` |
| CX27 | results | P1 | medium | done | `OutputExpr` has no `__array_ufunc__` / `__array_function__`, so `np.exp(outputs.ref("x"))` raises, and the named helpers (`tanh`, `maximum`) raise on it too; only `neg` and `abs` exist as unary nodes. `Param("s") * outputs.ref("x")` raises from `as_rate`. | Done in step 32: Dispatch through the shared `algebra` body in a new `output_expr` mode (roadmap step 32). | `src/summer4/results/outputset.py::class OutputExpr` |
| CX28 | results | P1 | medium | done | No way to apply an arbitrary function to post-solve outputs and parameters lazily: summer2's `Function(f, [DerivedOutput(...), Parameter(...)])` has no port, and `defer` accepts only rate expressions. | Done in step 33: `defer(f)(outputs.ref(...), Param(...))` builds a post-solve node; an Output or a same-shape array is returned. | `src/summer4/results/outputset.py::_CALLS = frozenset(` |
| CX29 | rates | P2 | medium | done | The child walk is hand-written in each pass (`_flow_refs`, `_field_paths`, `_collect_capture_meta`, `rate_stage`, `build_hoist_table`), each with a different dunder fallback. `ForceOfInfection.__flow_refs__` returns an empty set, so a `FlowRef` under a FOI is never ordered; `_collect_capture_meta` skips `TableInterp` and `Lookup` children. | Done in step 35: one `rate_children` walk; custom nodes implement `__rate_children__` and `__rate_bytes__`. | `src/summer4/flows/rates.py::def rate_children` |
| CX30 | results | P1, P4 | high | planned | Two lazy expression languages: `RateOps` for rates and `OutputExpr` for post-solve outputs, with separate walks, separate parameter handling (`eval_closed` bridges only run-stage leaves) and operators hand-written in five classes. | A post-solve stage on one node set: `Saved`, `OutputRef` and `PostOp` are `RateOps`; `OutputSet` stores and evaluates `RateOps`; `OutputExpr` is deleted (roadmap steps 36-38). | `src/summer4/results/outputset.py::type _Kind = Literal` |
| CX31 | rates | P1 | high | planned | `derived_fn` is the only multi-output derived-quantity door, and setting it makes every `FieldRef` step-stage, disabling parameter hoisting model-wide (R3). A step-stage sub-expression shared by several flows is re-traced per flow. | Per-call shared-node evaluation plus `field_refs(defer(f)(Params(), y=StateArray(), t=Time()), Schema)`; `derived_fn` deprecated (roadmap steps 39 and 41). | `src/summer4/flows/compiled.py::params_are_static=derived_fn is None` |
| CX32 | results | P1 | medium | planned | An arbitrary step-stage expression (a seasonal rate, a per-age FOI) cannot be saved as an output without a `SaveFn` callable or a `derived_fn` path read by `ComputedValue`. | `outputs[name] = expr` saves any step-stage expression during the solve; `SaveFn` and `ComputedValue` deprecated (roadmap step 40). | `src/summer4/results/plan.py::class SaveFn` |
| CX33 | rates | P1 | medium | planned | InitialPopulation base values and weights accept a `callable(params)` that bypasses the rate tree: it is keyed by `id()`, cannot be staged or inspected, and duplicates what a run-stage expression already does. | Deprecate the callables in favour of `defer(f)(Params())` (roadmap step 42). | `src/summer4/flows/initial.py::def _base_digest` |
| CX34 | rates | P2 | medium | planned | Stage is a runtime property only: `Time()` in an initial population, a step-stage likelihood scale or (after CX30) a post-solve expression in a rate slot fails at compile or run time, never under mypy. | `RateOps` generic in a covariant stage marker (`Run <: Step <: Post`), with each slot annotated (roadmap step 43). | `src/summer4/flows/stages.py::Stage = Literal` |
| CX35 | calibration | P1, P2 | medium | done | Before step 46: the workflow stages were bound to `BayesianModel`: `wf.lhs`, `prior_draws`, `evaluate`, `Candidates`, `optimize`, `run_mcmc` and `MCMCRun.candidates` read `numpyro_model`, `potential_fn`, `log_density`, `constrain`, `unconstrain`, `prior_names` and `site_priors`. A hand-written numpyro model (hierarchical latents, two likelihoods, sites shared between them) runs under `MCMC(NUTS(model))`, but not through LHS, multi-start optimisation, `Candidates.init_params` or replay. | Done in step 46: a `Posterior` protocol of exactly those members, which every stage takes; `NumpyroPosterior(model)` implements it for any model function; public `BayesianModel.log_likelihood(params)` so a joint model can `numpyro.factor` a compiled model's targets; `wf.lhs` / `prior_draws` draw sites without a stratifiable prior by `Predictive`. | `src/summer4/epi/calibration/workflow/design.py::def lhs(post: Posterior` |
| CX36 | calibration | P2, P3 | medium | done | Before step 47: `wf.sample_until` / `MCMCRun` request only `diverging` from numpyro, so `num_steps` (gradient evaluations), `accept_prob` and the adapted step size of each draw are lost; tuning or costing a sampler means rerunning it outside the stage (tbcore `docs/summer4-workarounds.md` W8). | Done in step 47: `sample_until(..., extra_fields=)`; `MCMCRun.extra[name]` is `(chain, draw)` across chunks and `extend`, and lands in `idata.sample_stats`. | `src/summer4/epi/calibration/workflow/mcmc.py::extra = ("diverging",) if self._divergences else ()` |
| CX37 | calibration | P2 | medium | done | Before step 47: `MCMCRun` reports one R-hat and ESS per site, averaged over a plate site's elements (`np.mean(split_gelman_rubin(x))`), so one unconverged element is hidden by its converged siblings and `StopRule` cannot judge on the worst element. | Done in step 47: progress rows and `MCMCRun.diagnostics` add `rhat_element_max` / `ess_bulk_element_min`; `StopRule(per_element=True)` judges on them (the default still averages, so existing runs stop where they did). | `src/summer4/epi/calibration/workflow/mcmc.py::float(np.mean(split_gelman_rubin(x)))` |
| CX38 | calibration | P2, P3 | medium | done | Before step 47: `wf.warmup_until` cannot start from a given inverse mass matrix, restarts every doubling round from the factory's metric (discarding the previous round's adaptation), and reports nothing until it returns, so a long dense warm-up is a black box. | Done in step 47: `warmup_until(..., inverse_mass_matrix=, carry_metric=, on_round=)` calls `make_mcmc(n, inverse_mass_matrix=M)` when it has a metric; rows record `grad_evals`. | `src/summer4/epi/calibration/workflow/warmup.py::mcmc = make_mcmc(n)` |
| CX39 | run | P1 | low | deferred | Number width is process-wide: importing summer4 turns on `jax_enable_x64` for every JAX library in the process, and float32 means turning it off for all of them. A float32 model and a float64 model cannot run side by side, and a solve cannot be float32 while its likelihood sums in float64. | See the note. | `futureplans/per-model-dtype.md`; `src/summer4/_precision/__init__.py::jax.config).update("jax_enable_x64", True)` |
<!-- /composability:findings -->

## Work packages

Every `open` or `planned` finding belongs to exactly one package; deferred
findings are tracked by their `futureplans/` notes instead. **CP1** landed as
roadmap step 29 and **CP2** as step 30. **CP3** and **CP4** are independent
and can run alongside the roadmap. **CP5** and **CP6** are the staged
unification of the lazy idioms (`plans/unified-lazy-idiom.plan.md`): CP5 is
roadmap steps 31-35, CP6 steps 36-44. **CP7** is roadmap step 47, the NUTS
tuning stage (`plans/nuts-tuning.plan.md`).

<!-- composability:packages -->
| Package | Title | Closes |
| --- | --- | --- |
| CP1 | The solve takes the caller's diffrax objects | CX1 CX2 CX3 CX4 CX5 |
| CP2 | Calibration entry points built on the workflow pieces | CX16 CX17 CX18 CX19 CX24 |
| CP3 | Public extension points | CX6 CX12 CX13 CX20 CX21 CX23 CX35 |
| CP4 | Tests use public API | CX8 CX10 CX22 |
| CP5 | Bridges between the lazy idioms (roadmap steps 31-35) | CX7 CX26 CX27 CX28 CX29 |
| CP6 | One lazy node set for rates, outputs and parameters (roadmap steps 36-44) | CX11 CX30 CX31 CX32 CX33 CX34 |
| CP7 | NUTS tuning: sampler statistics, worst-element diagnostics, steerable warm-up (roadmap step 47) | CX36 CX37 CX38 |
<!-- /composability:packages -->

## Patterns to copy

These already meet the goal and are the templates for new work:

- **The MCMC stage** (`CX25`, `src/summer4/epi/calibration/workflow/mcmc.py`
  and `warmup.py`): the caller builds the library object; summer4 adds small
  public pieces; the one-call convenience states and tests its expansion.
- **`Candidates` stages** (`workflow/design.py`, `evaluate.py`,
  `candidates.py`): every stage takes and returns `Candidates`, with cumulative
  `history`.
- **Rate algebra** (`summer4.flows.rates`): typed nodes, NumPy ufunc dispatch,
  `defer(fn, name=)` for ordinary callables, and `register_rate_eval` for
  package-level custom nodes.
- **`Stratification` objects** (`summer4.propertymap`): immutable, reusable,
  and recorded in `PropertyMap.history`.
- **`Target.from_series`**: takes a pandas `Series` directly.

## How this survey was made

On 2026-09-28, on `main` after `v0.2.0a5`, four read-only passes covered
flows, solvers and results, epi and taxonomy, and calibration, checking each
public API against P1–P4. Every finding above was checked against the source
before it was recorded; three claims from those passes were corrected on
checking (`find_map` already accepts any optax transformation; a callable FOI
`kind` already works, but receives no parameters; `Prior` is a runtime-checkable
protocol, so user classes already work). Expressiveness gaps with existing
`futureplans/` notes are recorded as `deferred` rather than restated.
