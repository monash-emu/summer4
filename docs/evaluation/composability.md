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
Open findings: **21** (4 high, 9 medium, 8 low); planned: 0; deferred: 3; done: 1; rejected: 0.
<!-- /composability:totals -->

The calibration workflow's MCMC stage (`CX25`) is the reference pattern, and
the rest of the calibration layer is uneven against it: `BayesianModel.sample`
and `find_map` predate it and duplicate it (`CX16`, `CX19`), and the
optimisation stage keeps its backend protocol private (`CX17`). The largest
gap outside calibration is the solve itself. `run()` accepts a diffrax solver
but builds everything around it — stepsize controller, adjoint, events — so a
calibration that needs a different adjoint for its gradients cannot get one
(`CX1`, `CX2`).

## Findings

<!-- composability:findings -->
| ID | Area | Principle | Severity | Status | Finding | Proposed shape | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CX1 | run | P1 | high | open | `run()` builds the stepsize controller itself: `PIDController(rtol, atol)` when either is set, else `ConstantStepSize`. A caller's own controller (PID coefficients, `dtmin`, jump times) cannot be passed. | `run(..., stepsize_controller=diffrax.PIDController(...))`; `rtol` / `atol` stay as sugar that builds one. | `src/summer4/solvers/diffrax_backend.py::controller: Any = diffrax.PIDController(` |
| CX2 | run | P1 | high | open | `diffeqsolve` is called with no `adjoint=`, `event=` or `progress_meter=`. Calibration gradients always use diffrax's default adjoint, and a run cannot stop on a state event. | Pass caller-built `adjoint=` / `event=` objects through, for example on a public `SolveSpec` (`CX3`). | `src/summer4/solvers/diffrax_backend.py::sol = diffrax.diffeqsolve(` |
| CX3 | run | P1, P2 | medium | open | Solver choice is one argument that is either a name or a diffrax instance; summer4's own Euler backend is reachable only by the string `"euler"`, and the solve settings (`SolveSpec`) are internal. | Public backend objects (`summer4.solvers.Euler()`, any diffrax solver) and a public `SolveSpec`; names stay as sugar. | `src/summer4/flows/compiled.py::solver: str`; `src/summer4/solvers/base.py::class SolveSpec` |
| CX4 | run | P2, P4 | medium | open | `CompiledModel.run` is one long method: prepare, expand the save plan, group requests, build the time axis, build a `SolveSpec`, dispatch, assemble the `Result`. No intermediate step is public, so changing one (say, the `SaveAt`) means forking `run`. | Public steps (`expand_plan`, `solve`, `assemble_result`) with `run` as their documented and tested composition. | `src/summer4/flows/compiled.py::def run(` |
| CX5 | run | P3 | medium | open | A `Result` carries no end state. Continuing a run needs a hand-saved `Compartments()` at `t1`, re-wrapped as `y0`. | `Result.final_state` (a `PropertyData`) usable directly as the next `y0`. | `src/summer4/results/result.py::class Result:` |
| CX6 | rates | P2 | medium | open | Extension and staging points are not exported from `summer4.flows`: `register_rate_eval` (the custom-rates cookbook imports it from `summer4.flows.rates`), and `rate_stage` / `build_hoist_table` (tests import them from `summer4.flows.stages`). A modeller cannot ask which stage a rate will run in. | Export them, or add a `CompiledModel.hoist_table` / stage-report view. | `src/summer4/flows/rates.py::def register_rate_eval`; `src/summer4/flows/stages.py::def rate_stage` |
| CX7 | rates | P1 | medium | open | Callables are keyed by identity with no stable-name option: `SaveFn.fn` digests `id(fn)`, as do `Transform.fn` and `derived_fn`, so a lambda built in a loop retraces every call. `defer` already solves this with `name=`. | An optional `name=` on `SaveFn`, `Transform` and `derived_fn`, with the `defer(name=)` contract. | `src/summer4/results/plan.py::str(id(fn))`; `src/summer4/flows/rates.py::class Transform` |
| CX8 | rates | P2 | low | open | Tests import private flows helpers (`_align_rate`, `_align_grouped_rate`, `_eval_interp`, `_rate_bytes`). Most are true internals. | Triage: promote what a custom-rate author needs, test the rest through public behaviour. | `src/summer4/flows/compiled.py::def _align_rate` |
| CX9 | results | P1 | low | deferred | `Output.plot` hard-codes matplotlib calls around a pandas plot whose backend may be Plotly. | See the note. | `futureplans/trace-plot-backend-coupling.md`; `src/summer4/results/output.py::def plot` |
| CX10 | results | P2 | low | open | The test loss helper rebuilds a state with private `PropertyData._with_data` where public `PropertyData` arithmetic already does it. | Use `y0 * weights` in the helper; no API change. | `tests/helpers/loss.py::_with_data` |
| CX11 | epi | P1 | medium | open | A custom `ForceOfInfection` `kind` callable receives no parameters, so a parameterised custom shape (learned exponents, saturation) is impossible unless it is `FOIKind.GENERALISED`. | Let `kind` build a rate expression, `kind(infectious, denominator) -> RateOps`, so `Param`s flow through the rate tree and stage normally. | `src/summer4/epi/infection.py::is the custom path` |
| CX12 | epi | P1 | low | open | `normalize_infectiousness` and `MixingMatrix(normalize=)` are closed `StrEnum`s; any other normalisation means forking. | Accept a callable alongside the enum members. | `src/summer4/epi/infection.py::class InfectiousnessNormalize`; `src/summer4/epi/mixing.py::class MixingNormalize` |
| CX13 | epi | P2 | low | open | `apply_compartment_weights` and `coerce_compartment_weights` are in `summer4.epi.__all__`, but the `eval_child` callback contract is undocumented and no user-facing page uses them. | Document the contract with an example, or drop them from `__all__` until the susceptibility surface lands. | `src/summer4/epi/__init__.py::"apply_compartment_weights"` |
| CX14 | epi | P2 | medium | deferred | `MixingMatrix` row-normalises a `FieldRef` matrix inside the vector field on every call instead of once at run start. | See the note. | `futureplans/mixing-matrix-per-call-normalisation.md`; `src/summer4/epi/mixing.py::def resolved_matrix` |
| CX15 | epi | P1 | medium | deferred | `ForceOfInfection(group_by=)` takes one `Property`, and the whole-population case needs a dummy property and a `[[1.0]]` matrix. | See the notes. | `futureplans/foi-multi-property-mixing.md`; `futureplans/foi-unstratified-dummy-pop.md` |
| CX16 | calibration | P1, P4 | high | open | `BayesianModel.sample(kind=...)` dispatches on a string, passes through only `**kernel_kwargs`, cannot be seeded, and duplicates the composable MCMC stage. | Rebuild it as a documented convenience over `wf.sample_until` (take `make_kernel=` and `init=`, return an `MCMCRun`), or deprecate it for `wf.run_mcmc`. | `src/summer4/epi/calibration/model.py::def sample(` |
| CX17 | calibration | P2 | high | open | The `wf.optimize` backend protocol is private (`_Method`) and `method=` is typed as the two built-ins, so a user cannot plug in another optimiser and reuse the chunk loop, convergence and restarts. Tests reach `_chunk_program` and `_tree_set`. | A public `OptimizeMethod` protocol (`init(z0, key)`, `step(state)`), with `Optax` and `CMAES` as two implementations. | `src/summer4/epi/calibration/workflow/optimize.py::class _Method` |
| CX18 | calibration | P3 | medium | open | `wf.optimize` returns no optimiser state, so a run cannot be extended; restarting from `result.candidates` resets Adam moments or the CMA-ES covariance. | Keep method state on the result and add `extend(max_steps=)`, as `MCMCRun.extend` does. | `src/summer4/epi/calibration/workflow/optimize.py::class OptimizeResult` |
| CX19 | calibration | P3, P4 | medium | open | `BayesianModel.find_map` returns only the final parameters (no loss trace, no state, no resume) and duplicates single-start `wf.optimize`. | Build it on `wf.optimize` with one start and return an `OptimizeResult`. | `src/summer4/epi/calibration/model.py::def find_map(` |
| CX20 | calibration | P1, P2 | medium | open | Likelihoods have no protocol, and nested prior scales are found only through attributes named `sd` or `dispersion`. A custom likelihood with a differently named prior scale silently loses that prior. | A `Likelihood` protocol with `log_prob(...)` and `prior_sites()`. | `src/summer4/epi/calibration/likelihoods.py::def prior_sites` |
| CX21 | calibration | P1 | low | open | A prior must implement the `Prior` protocol; a bare numpyro distribution needs a hand-written wrapper. | `Prior.from_numpyro(name, dist)`, with `icdf` only where the distribution provides one. | `src/summer4/epi/calibration/priors.py::class Prior(Protocol)` |
| CX22 | calibration | P2 | low | open | Tests call private `BayesianModel._ensure_potential()` for the initial point and the postprocess transform. | Public `init_point` / `postprocess_fn` beside the public `potential_fn`. | `tests/test_epi_calibration.py::_ensure_potential` |
| CX23 | calibration | P2 | low | open | How a `Scenario` rebinds params, model and outputs lives in private `_scenario_bindings`, so a scenario cannot be applied outside `posterior_runs`. | A public binding function, or a method on `Scenario`. | `src/summer4/epi/calibration/posterior_runs.py::def _scenario_bindings` |
| CX24 | calibration | P3 | low | open | `OptimizeResult.history` holds only the optimise record, while its `candidates.history` is cumulative. | Make it cumulative, or drop it in favour of `candidates.history`. | `src/summer4/epi/calibration/workflow/optimize.py::history=(record,)` |
| CX25 | calibration | P1, P2, P3, P4 | high | done | The MCMC stage took `kind=` strings, bundled seeding, retries and stopping, and returned a sealed `SampleResult`. It is now the reference pattern: caller-built numpyro `MCMC`, `Candidates.init_params`, `wf.warmup_until`, `wf.sample_until` → `MCMCRun`, `wf.StopRule`, `wf.replay`, and `wf.run_mcmc` (PR #39). | — | `src/summer4/epi/calibration/workflow/mcmc.py::def sample_until` |
<!-- /composability:findings -->

## Work packages

Every `open` or `planned` finding belongs to exactly one package; deferred
findings are tracked by their `futureplans/` notes instead. Take packages in
this order. **CP1** first: `CX1`–`CX2` block calibrations that need a
different adjoint or an event, and every later package builds on a public
`SolveSpec`. **CP2** next, because it retires the duplicate calibration entry
points before step 28's plots are written against them. **CP3** and **CP4**
are independent and can run alongside the roadmap.

<!-- composability:packages -->
| Package | Title | Closes |
| --- | --- | --- |
| CP1 | The solve takes the caller's diffrax objects | CX1 CX2 CX3 CX4 CX5 |
| CP2 | Calibration entry points built on the workflow pieces | CX16 CX17 CX18 CX19 CX24 |
| CP3 | Public extension points | CX6 CX7 CX11 CX12 CX13 CX20 CX21 CX23 |
| CP4 | Tests use public API | CX8 CX10 CX22 |
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
