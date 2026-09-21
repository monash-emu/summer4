# Coverage ledger

**This file is the authoritative record of what summer4 covers and what 100%
would require.** Other branches, contributors and agents should read it before
planning work, and update it in the same commit that changes coverage.

```{admonition} Contract for anyone changing coverage
:class: important

1. The tables below are the single source of truth. Prose elsewhere on this
   site quotes their totals; it does not restate their contents.
2. Status values are exactly `full`, `partial` or `none`. Nothing else parses.
3. `Status` is what is importable from `summer4` today.
4. Run `pixi run coverage` after editing. `tests/test_coverage_ledger.py`
   fails if a status is misspelled or a total is stale.
5. When you land a feature, change its row in the same commit. A branch that
   moves coverage without moving this file is incomplete.
```

## Definitions

| Term | Meaning |
|---|---|
| `full` | A documented, working way to achieve what the summer2 symbol does |
| `partial` | Achievable, but the user supplies something summer2 supplied, or a material limitation applies |
| `none` | No way to do it |
| **Complete** | Count of `full` rows |
| **Covered** | Count of `full` + `partial` rows |

The summer2 symbol list is not all of summer2. It is every symbol exercised by
the summer2 documentation notebooks (`docs/examples/`, `docs/detailed/`) or by
the summer textbook, extracted mechanically. That is the surface a port has to
reach, and it is the denominator for every percentage on this site.

## API ledger

<!-- ledger:api -->
| ID | summer2 symbol | Area | Status | summer4 equivalent | Notes |
| --- | --- | --- | --- | --- | --- |
| L1 | `CompartmentalModel` | lifecycle | `partial` | `FlowModel` | No times, timestep or infectious_compartments |
| L2 | `model.finalize()` | lifecycle | `full` | `FlowModel.compile()` | Actualizes every join once |
| L3 | `model.run()` | lifecycle | `full` | `CompiledModel.run()` | Returns a `Result`; Euler backend in Phase 2 |
| L4 | `model.set_initial_population()` | lifecycle | `full` | `FlowModel.set_initial_population` / `InitialPopulation` |  |
| L5 | `model.get_initial_population()` | lifecycle | `full` | `CompiledModel.initial_state` |  |
| L6 | `model.get_outputs_df()` | lifecycle | `full` | `Output.to_frame` / `to_pandas` | Polars default; pandas optional |
| L7 | `model.get_derived_outputs_df()` | lifecycle | `full` | `Result` outputs via `SavePlan` | Flat named outputs; no parallel df namespace |
| S1 | `Stratification(name, strata)` | stratification | `full` | `Property + PropertyMap.stratify` |  |
| S2 | `model.stratify_with()` | stratification | `full` | `PropertyMap.stratify` | `FlowModel.stratify` (in place, like summer2) or `PropertyMap.stratify` (new map) |
| S3 | `Stratification(compartments=[...])` | stratification | `full` | `stratify(prop, where=selector)` | Generalised from names to a query |
| S4 | `model.get_stratification()` | stratification | `full` | `PropertyMap.get_property / history` | History also records the where= selector |
| S5 | `Compartment` | stratification | `partial` | `rows; labels(), to_dicts()` | No per-compartment object by design |
| S6 | `AgeStratification` | stratification | `partial` | `Property + TraitChain flow` | Ageing flows exist; no bundled convenience class |
| S7 | `StrainStratification` | stratification | `partial` | `Property + per-strain flows` | Multi-strain FOI via `ForceOfInfection.per_trait`; no bundled StrainStratification class |
| S8 | `set_population_split / adjust_population_split` | stratification | `full` | `Split(prop, weights, by=, where=)` | Weights normalised; unspecified properties split evenly over carriers |
| Q1 | `model.query_compartments()` | queries | `full` | `PropertyMap.select / mask` | Algebra, not a conjunction dict |
| Q2 | `model.get_matching_compartments()` | queries | `full` | `PropertyMap.select / select_one` |  |
| Q3 | `model.query_flows()` | queries | `full` | `EdgeMap` / `CompiledModel.edges` / `Source`/`Dest` | Query API over flow edges |
| F1 | `add_transition_flow` | flows | `full` | `TransitionFlow` |  |
| F2 | `add_death_flow` | flows | `full` | `ExitFlow` |  |
| F3 | `add_universal_death_flows` | flows | `full` | `ExitFlow(Everything(), rate)` |  |
| F4 | `add_crude_birth_flow` | flows | `full` | `EntryFlow` | Absolute rate from a derived total |
| F5 | `add_replacement_birth_flow` | flows | `full` | `EntryFlow(dest, death.sum_over(...))` |  |
| F6 | `add_importation_flow` | flows | `full` | `EntryFlow` | Absolute rate |
| F7 | `add_infection_frequency_flow` | flows | `full` | `ForceOfInfection(kind=FOIKind.FREQUENCY)` | Same object as F8; kind selects frequency vs density |
| F8 | `add_infection_density_flow` | flows | `full` | `ForceOfInfection(kind=FOIKind.DENSITY)` | Same object as F7; kind selects frequency vs density |
| A1 | `Stratification.set_flow_adjustments` | adjustments | `full` | `adjust= with where=` | Flow-owned, deliberately not on Stratification; `adjust_flow` after `stratify`; `Source`/`Dest` where |
| A2 | `Multiply` | adjustments | `full` | `Multiply` | Default for a bare value in adjust= |
| A3 | `Overwrite` | adjustments | `full` | `Overwrite` | Supports where=Selector; precedence levels (summer2 applies in stratification order) |
| A4 | `Stratification.add_infectiousness_adjustments` | adjustments | `full` | `ForceOfInfection(infectiousness=...)` | FOI-owned, deliberately not on Stratification (as A1) |
| M1 | `Stratification.set_mixing_matrix` | mixing | `full` | `summer4.epi.MixingMatrix` | Weights transmission; `TraitMatrix` still moves people |
| P1 | `Parameter` | parameters | `full` | `FieldRef via derived_refs` | Schema-checked, IDE-completable |
| P2 | `Function` | parameters | `full` | `defer / Transform / callables in derived_fn` | `defer` is the rate-slot door; `Transform` only adjusts an existing rate |
| P3 | `Time` | parameters | `full` | `Time()` rate node; `t` on `derived_fn` | |
| P4 | `DerivedOutput (as a rate input)` | parameters | `full` | `FlowRef, .sum(), .sum_over()` | Topologically ordered, cycles detected |
| P5 | `Data` | parameters | `full` | `summer4.data.Data` | Dated series → `Interp` via `Epoch` |
| P6 | `get_linear_interpolation_function` | parameters | `full` | `summer4.timevarying.linear` | Structural `Interp`; x and y knots may be `FieldRef`s; clamps outside range |
| P7 | `get_sigmoidal_interpolation_function` | parameters | `full` | `summer4.timevarying.sigmoidal` | `sharpness` is summer2 curvature; breakpoints may be `FieldRef`s |
| P8 | `get_piecewise_function` | parameters | `full` | `summer4.timevarying.step` / `piecewise` | Right-continuous; breakpoints may be `FieldRef`s |
| P9 | `get_time_callable` | parameters | `partial` | `compile() -> vf(t, y, params)` | A time callable, but not summer2's graph wrapper |
| D1 | `request_output_for_flow` | outputs | `full` | `FlowMass` / `Output` edge queries | `sum_over(..., side=)`, `integrate_intervals`, `integrate` |
| D2 | `request_output_for_compartments` | outputs | `full` | `Compartments(where=)` / `Output.select` |  |
| D3 | `request_aggregate_output` | outputs | `full` | `Output.sum_over` / `total` / `partition` |  |
| D4 | `request_cumulative_output` | outputs | `full` | `Output.cumulative()` |  |
| D5 | `request_function_output` | outputs | `full` | `SaveFn`; arithmetic on `Output.values` | `Output` has no operators yet; they arrive in WP15 |
| D6 | `request_computed_value_output` | outputs | `full` | `ComputedValue` | Path validated against `derived_fn` return schema |
| D7 | `request_track_modelled_value` | outputs | `full` | `ComputedValue` | Same capture path as D6 |
| D8 | `add_computed_value_func` | outputs | `full` | `derived_fn hook` | compute_derived_params runs every step; run-start work goes in `prepare_fn` |
| V1 | `solve_ode` | solver | `full` | `CompiledModel.run` over `euler` | Fixed step; adaptive is V2 |
| V2 | `SolverType / solver selection` | solver | `full` | `solver=` name or diffrax instance | Euler kept as reference stepper |
| T1 | `ref_date / Epoch` | time | `full` | `Epoch` |  |
| T2 | `model.get_epoch()` | time | `full` | `Result.times.epoch` / `TimeAxis.epoch` |  |
<!-- /ledger:api -->

## Textbook ledger

Chapters of the [summer textbook](https://github.com/monash-emu/summer-textbook).
`partial` means the modelling content is expressible but the chapter cannot be
published as written.

<!-- ledger:textbook -->
| Ch | Title | Status | Blocker | Ported |
| --- | --- | --- | --- | --- |
| 1 | Infectious disease modelling | `full` | None - prose | `textbook/01-introduction.md` |
| 2 | Basic model construction | `full` | None | `textbook/02-model-structures.ipynb` |
| 3 | Thinking about flows | `full` | None | `textbook/03-thinking-about-flows.ipynb` |
| 4 | Thinking about flow rates | `full` | None | `textbook/04-thinking-about-flow-rates.ipynb` |
| 5 | Series compartments and latency | `full` | None | `textbook/05-series-compartments-latency.ipynb` |
| 6 | Post-infection immunity | `full` | None | `textbook/06-post-infection-immunity.ipynb` |
| 7 | Obtaining numerical solutions | `full` | None | `textbook/07-numerical-solutions.ipynb` |
| 8 | Derived outputs | `full` | None | `textbook/08-derived-outputs.ipynb` |
| 9 | Transmission assumptions | `full` | None | `textbook/09-transmission-assumptions.ipynb` |
| 10 | The reproduction number | `full` | None (unstratified $R_t$; no next-gen matrix helper) | `textbook/10-reproduction-number.ipynb` |
| 11 | Cyclical epidemic dynamics | `full` | None | `textbook/11-cyclical-epidemic-dynamics.ipynb` |
| 12 | Heterogeneous mixing introduction | `full` | None | `textbook/12-heterogeneous-mixing-intro.ipynb` |
| 13 | Mixing and transmission types | `full` | None | `textbook/13-mixing-and-transmission-types.ipynb` |
| 14 | Assortative mixing | `full` | None | `textbook/14-assortative-mixing.ipynb` |
| 15 | Susceptibility and infectiousness matrices | `partial` | No FOI susceptibility surface symmetric to infectiousness | `textbook/15-susceptibility-infectiousness-matrices.ipynb` |
| 16 | Thinking about contact surveys | `none` | Contact-survey data (WP9) | — |
| 17 | Understanding empiric contact data | `none` | Contact-survey data | — |
| 18 | Implementing empiric survey data | `none` | Contact-survey data | — |
| 19 | Adapting mixing matrices | `none` | Contact-survey data, matrix scaling | — |
| 20 | Calibration and uncertainty | `none` | Calibration workflow | — |
<!-- /ledger:textbook -->

## summer2 documentation ledger

<!-- ledger:summer2docs -->
| Page | Status | Blocker | Ported |
| --- | --- | --- | --- |
| `examples/01-basic-model` | `full` | None | `summer2/01-basic-model.ipynb` |
| `examples/03-derived-outputs` | `full` | None | `summer2/03-derived-outputs.ipynb` |
| `examples/04-flow-types` | `full` | None | `summer2/04-flow-types.ipynb` |
| `examples/06-stratification-introduction` | `full` | None | `summer2/06-stratification-introduction.ipynb` |
| `examples/07-age-stratification` | `full` | None | `summer2/07-age-stratification.ipynb` |
| `examples/08-strain-stratification` | `full` | None (`per_trait` FOI; no `StrainStratification` class) | `summer2/08-strain-stratification.ipynb` |
| `examples/09-mixing-matrices` | `full` | None | `summer2/09-mixing-matrices.ipynb` |
| `examples/10-derived-outputs-stratified` | `full` | None | `summer2/10-derived-outputs-stratified.ipynb` |
| `examples/11-flows-between-strata` | `full` | None | `summer2/11-flows-between-strata.ipynb` |
| `detailed/time-varying-functions` | `full` | None | `summer2/time-varying-functions.ipynb` |
| `detailed/InitialPopulationGraphobject` | `full` | None | `summer2/initial-population-graphobject.ipynb` |
<!-- /ledger:summer2docs -->

## Delivery status

Where the landed work physically lives, and which of the remaining packages have
a written plan. This section records *delivery*, not capability — the `Status`
column of the API ledger above stays the authority on what summer4 can do.

### Landed phases

Phases of `plans/flows-derived-outputs.plan.md`, in stack order. Each branch
contains every branch above it.

| Phase | Work package | Branch | Tip | Plan on the branch |
| --- | --- | --- | --- | --- |
| 0 | Taxonomy prerequisites | `feat/taxonomy-prereqs-phase0` | `285365c` | `plans/taxonomy-prereqs-phase0.plan.md` |
| 1 | WP1 — flows core | `feat/flows-core` | `fd10502` | `plans/flows-core.plan.md` |
| 2 | WP2 — trajectories and `Result` | `feat/results` | `2e66774` | `plans/results.plan.md` |
| 2a | WP2 follow-up — `Output.select` gather | `feat/trace-select-submap` | `50e0a31` | `plans/trace-select-submap.plan.md` |
| 3 | WP7 — diffrax backend | `feat/diffrax-solver` | `32b3e00` | `plans/diffrax-solver.plan.md` |
| 4 | WP4 — flow outputs and polarity | `feat/flow-outputs` | `088814a` | `plans/flow-outputs.plan.md` |
| 5 | WP11 — sparse targets | `feat/sparse-targets` | `3c0139c` | `plans/sparse-targets.plan.md` |
| — | Case study (not a work package) | `docs/age-stratified-seirs-case-study` | `2b1a6ab` | `plans/age-stratified-seirs-case-study.plan.md` |
| 5.1 | WP5 — `describe(params=...)` | `fix/describe-params` | *(this stack)* | `plans/time-varying.plan.md` |
| 5.2–5.5 | WP5 — time-varying library + harvest | *(stacked on 5.1)* | *(this stack)* | `plans/time-varying.plan.md` |
| 6.1–6.6 | WP6 — FOI and mixing | `feat/epi-infection-mixing` | *(this stack)* | `plans/epi-infection-mixing.plan.md` |
| C1–C4 | Textbook / summer2 / evaluation catch-up | `docs/textbook-catchup` | *(this stack)* | `plans/textbook-catchup.plan.md` |
| 3.0–3.7 | WP3 — initial population and run stages | `feat/initial-population` | *(this stack)* | `plans/initial-population.plan.md` |

Every phase in the table is now on `main`, as of the merge of PR #7
(`eceba1e`) and PR #8 (`769f9f1`). The first pinnable tag was `v0.2.0a1`; the
current tag is `v0.2.0a3` (`prepare()` boxes float params for Diffrax JIT cache). The table is kept as
history of the stack order.

Every phase shipped its notebook: `examples/notebooks/01-taxonomy.ipynb`
through `11-run-stages.ipynb`, plus textbook chapters 1–12 and 14–15 (13
ported on this stack), the summer2 pages under `docs/summer2/`, and
`docs/case-studies/age-stratified-seirs.ipynb`.

### Planning status of the remaining packages

WP5, WP6 and the textbook catch-up sweep
(`plans/textbook-catchup.plan.md`) are **applied** on this stack.

WP9 now has a detailed plan, `plans/wp9-contact-surveys.plan.md`; the paragraph
in *The path to 100%* below is its scope statement, not its design. Every
remaining package is planned.

WP10 and WP15–WP16 remain planned in
`plans/tb-ports-feature-completeness.plan.md`. WP12, WP13 and WP14 have shipped.
That plan exists to make two
tuberculosis models portable to summer4; which of their capabilities each
package closes is recorded, by row ID, in {doc}`tb-ports`.

| WP | Name | Planning artifact | What a plan must still settle |
| --- | --- | --- | --- |
| WP3 | Initial population | `plans/initial-population.plan.md` | **Applied.** |
| WP5 | Time-varying function library | `plans/time-varying.plan.md` (5.1–5.5) | **Applied.** `Time()`, `summer4.timevarying`, `summer4.data`, summer2 time-varying page |
| WP6 | Force of infection and mixing | `plans/epi-infection-mixing.plan.md` (6.1–6.6) | **Applied.** `GroupedRate`, `Reduce`, `summer4.epi` (`MixingMatrix`, `ForceOfInfection`, `EpiModel`) (`EpiModel` later removed: `plans/remove-epimodel.plan.md`) |
| — | Textbook / docs catch-up | `plans/textbook-catchup.plan.md` (C1–C4) | **Applied.** Ports for unblocked chapters and summer2 pages; evaluation prose refresh |
| WP9 | Contact survey data | `plans/wp9-contact-surveys.plan.md` | Settled there: `ContactMatrix` loading and validation, rebinning, reciprocity, population adaptation, scaling, textbook 16-19 |
| WP10 | Calibration | `plans/tb-ports-feature-completeness.plan.md` | Settled there: priors, likelihoods on `Target`, numpyro samplers (NUTS and gradient-free ensemble), MAP, posterior runs; lives in `summer4.epi` |
| WP12 | Pinnable release | `plans/wp12-release.plan.md` | **Applied.** First tag `v0.2.0a1`; current tag `v0.2.0a3`. JAX is a core dependency; `frames` declares polars and pyarrow. Downstream smoke CI is step 3 of {doc}`../dev/roadmap` |
| WP13 | Rate-tree math and tabular time series | `plans/tb-ports-feature-completeness.plan.md` | Math nodes, vector-valued table interpolation, `Lookup`, ageing sugar |
| WP14 | Generalised force of infection | `plans/tb-ports-feature-completeness.plan.md` | **Applied.** `FOIKind.GENERALISED` with an exponent; selector-keyed infectiousness |
| WP15 | Output algebra | `plans/tb-ports-feature-completeness.plan.md` | `Output` operators, windowed cumulative, multi-flow outputs, `OutputSet`, frames |
| WP16 | Scale and solver safety | `plans/tb-ports-feature-completeness.plan.md` | TB-scale benchmark, surfaced `max_steps` failure, vmap-safe reciprocity check |
| WP17 | FOI susceptibility surface | `plans/wp17-foi-susceptibility.plan.md` | Settled there: `ForceOfInfection(susceptibility=...)`; raises textbook 15 to `full` |
| WP18 | Rate expression dispatch and deferred callables | `plans/rate-dispatch-and-defer.plan.md` | Settled there: `__array_ufunc__` / `__array_function__` on the four wrapper types with canonical op names; a `Defer` node and `defer(fn)` curry for arbitrary user code in the rate slot |

WP5 lands before WP6, and before WP3, because every other part of a model may be
parameterised in a time-varying fashion. WP6 is then the ordering constraint for
what follows: WP9 and WP10 both sit behind it, and it closes four API rows
(F7 F8 A4 M1) plus textbook chapters 12, 14 and 15 — chapter 13 also needs WP3's
population split. Deferred gotchas that a plan for the unplanned packages should
read first are in `futureplans/`.

**Where the work has got to is recorded in {doc}`../dev/roadmap`**, not here.
That file sequences every remaining package into numbered steps — one branch
each — names the next one, and tells a session with no other context how to run
it. This ledger stays the authority on *capability*; the roadmap is the
authority on *position*. Track
`futureplans/derived-fn-blocks-hoisting.md`,
`futureplans/mixing-matrix-per-call-normalisation.md`, and
`futureplans/wp10-preprocess-is-prepare-fn.md` when planning WP10 / mixing work.

## The path to 100%

Work packages in the order that maximises coverage gained per unit of effort.
Each names the ledger IDs it closes, so progress is checkable against the tables
above rather than against a narrative.

WP1 (promote the flows spike into `summer4`) is **applied**. Flows, rates,
adjustments, `EdgeMap`, `CompiledModel` and a JAX Euler ship in
`summer4.flows`. WP2 (trajectories and a results object, including real-world
time) is **applied**: `CompiledModel.run` returns a queryable `Result`. WP7
(adaptive solver selection via diffrax) is **applied**. WP4 (flow outputs and
polarity queries) and WP11 (sparse outputs and calibration targets) are
**applied**. See [Delivery status](#delivery-status) for the branch each one
landed on and for which of the remaining packages have a written plan.

`Present` and `Absent` are **non-binding** in flow pairing: they name a property
(`selector_properties`) but do not bind it (`selector_values`). Binding is
`Trait` / `IsIn` only. `strict_pairing=True` raises when an unbound property
would move people.

### WP2 — Trajectories and a results object (applied)

**Closes:** L3 L6 L7 V1 T1 T2 D2 D3 D4 D5 (to `full`) · **Unblocks:** textbook
3, 5, 6, 9, 11 and deepens 2, 4, 8, 10; summer2 `11-flows-between-strata`

`CompiledModel.run` returns a `Result` of named `Output`s. `Epoch` / `TimeAxis`
map calendar dates; `SavePlan` names what to keep. Query surface covers select,
aggregate, cumulative, calendar resample, rolling, and interpolated `at_times`.

### WP3 — Initial population (applied)

**Closes:** L4 L5 S8 · **Unblocks:** textbook 2 and 13; summer2 `01-basic-model`,
`07-age-stratification`, `InitialPopulationGraphobject`

Declarative `InitialPopulation` / `Split` / `REMAINDER` with ragged-aware even
defaults and summer2-style `where=` adjusts. Run-start staging
(`prepare_fn`, hoisted rate subtrees) is documented in
{doc}`../dev/run-stages`. Example notebooks: `10-initial-population`,
`11-run-stages`.

### WP4 — Flow outputs and polarity queries (applied)

**Closes:** D1 D6 D7 · **Unblocks:** textbook 4, 8 (API); textbook 10's
port remains to write (time-varying and FOI are applied);
summer2 `03-derived-outputs`,
`04-flow-types`, `10-derived-outputs-stratified`

`FlowMass` outputs are `PropertyData` over the edge table. `sum_over(..., side=)`,
`.integrate()`, `.integrate_intervals()`, and validated `ComputedValue` paths are live.
Textbook ports for 4 and 8 remain to write; chapter 10's honest blocker was WP5.
Plan phase: `feat/flow-outputs`.

### WP5 — Time-varying function library (applied)

**Closes:** P5 P6 P7 P8 · **Unblocks:** summer2 `detailed/time-varying-functions`

`Time()` in the rate tree, structural `Interp` / `GaussianPulse` nodes via
`summer4.timevarying`, and dated-series `summer4.data.Data`. Ported at
`docs/summer2/time-varying-functions.ipynb`.

### WP6 — Force of infection and mixing (applied)

**Closes:** F7 F8 (to `full`) A4 M1 · **Unblocks:** textbook 12, 13, 14, 15

`GroupedRate` and `Reduce` in core; `summer4.epi` ships `MixingMatrix`,
`ForceOfInfection` (frequency / density / custom), infectiousness weights,
`per_trait` multi-strain FOIs, and an `EpiModel` frontend. (`EpiModel` later
removed: `plans/remove-epimodel.plan.md`) Example notebook:
`examples/notebooks/09-epi-models.ipynb`. `TraitMatrix` remains people-movement,
not transmission weighting.

### WP7 — Adaptive solver (applied)

**Closes:** V2 · **Unblocks:** textbook 7 fully

A diffrax backend behind the solver seam, with `solver=` selecting Euler,
Heun, Tsit5, Dopri5, or a diffrax solver instance. Textbook chapter 7 is
ported at `docs/textbook/07-numerical-solutions.ipynb`.

### WP9 — Contact survey data

**Unblocks:** textbook 16, 17, 18, 19

Loading, validating, inspecting and scaling empirical contact matrices.
Depends on WP6.

### WP10 — Calibration

**Unblocks:** textbook 20

A Bayesian workflow over JAX-differentiable models. `numpyro` and `optax` are
declared extras; `optax` is exercised by sparse-target fits (WP11). Depends on
everything above, including WP11's declarative targets.

### WP12–WP16 — Packages for the TB model ports

**Closes:** *(no API rows)* · **Unblocks:** the Kiribati and tb_macro ports;
see {doc}`tb-ports` for the rows each closes and the computed readiness after each.

WP12 is **applied**: first tag `v0.2.0a1`, current tag `v0.2.0a3`. WP13 is
**applied**: math nodes, vector-valued table interpolation, `Lookup`, and ageing
sugar. WP14 is **applied**: `FOIKind.GENERALISED` with a calibratable exponent,
and infectiousness weights keyed by any selector, applied per compartment
before the group sum. Where the work has got to after that is
{doc}`../dev/roadmap`, not this page. WP15 adds output algebra: `Output`
operators, a windowed `cumulative`, multi-flow `FlowMass`, and named output
sets to frames. WP16 benchmarks a TB-scale model and makes solver failure
visible. Plan: `plans/tb-ports-feature-completeness.plan.md`.

### WP11 — Sparse outputs and calibration targets (applied)

**Closes:** *(no API rows)* · **Unblocks:** WP10 / textbook 20

`Target` / `TargetSet` merge observation times into a `SavePlan` so a
likelihood run never materialises dense outputs. Gathering and residuals ship
here; probabilistic likelihoods stay in WP10. Plan phase: `feat/sparse-targets`.

## Work packages, declared

`Closes` lists the ledger IDs each package raises to `full`. The progression
table below is **computed** from these declarations by
`scripts/coverage_report.py`; do not edit it by hand.

<!-- ledger:packages -->
| WP | Name | Closes |
| --- | --- | --- |
| WP2 | Trajectories and a results object | L3 L6 L7 V1 T1 T2 D2 D3 D4 D5 |
| WP3 | Initial population | L4 L5 S8 |
| WP4 | Flow outputs and polarity queries | D1 D6 D7 |
| WP5 | Time-varying function library | P5 P6 P7 P8 |
| WP6 | Force of infection and mixing | F7 F8 A4 M1 |
| WP7 | Adaptive solver | V2 |
| WP9 | Contact survey data | *(no API rows; unblocks textbook 16-19)* |
| WP10 | Calibration | *(no API rows; unblocks textbook 20)* |
| WP11 | Sparse outputs and calibration targets | *(no API rows; unblocks WP10 / textbook 20)* |
| WP12 | Pinnable release | *(no API rows; unblocks the TB ports)* |
| WP13 | Rate-tree math and tabular time series | *(no API rows; unblocks the TB ports)* |
| WP14 | Generalised force of infection | *(no API rows; unblocks the TB ports)* |
| WP15 | Output algebra | *(no API rows; unblocks the TB ports)* |
| WP16 | Scale and solver safety | *(no API rows; unblocks the TB ports)* |
| WP17 | Force-of-infection susceptibility surface | *(no API rows; unblocks textbook 15)* |
| WP18 | Rate expression dispatch and deferred callables | *(no API rows; opens the operator set and the `defer` on-ramp)* |
<!-- /ledger:packages -->

## Coverage after each package

<!-- ledger:progression -->
| After | API rows at `full` | Share |
| --- | --- | --- |
| today | 47 / 52 | 90% |
| WP2 | 47 / 52 | 90% |
| WP3 | 47 / 52 | 90% |
| WP4 | 47 / 52 | 90% |
| WP5 | 47 / 52 | 90% |
| WP6 | 47 / 52 | 90% |
| WP7 | 47 / 52 | 90% |
| WP9 | 47 / 52 | 90% |
| WP10 | 47 / 52 | 90% |
| WP11 | 47 / 52 | 90% |
| WP12 | 47 / 52 | 90% |
| WP13 | 47 / 52 | 90% |
| WP14 | 47 / 52 | 90% |
| WP15 | 47 / 52 | 90% |
| WP16 | 47 / 52 | 90% |
| WP17 | 47 / 52 | 90% |
| WP18 | 47 / 52 | 90% |
<!-- /ledger:progression -->

## What never reaches `full`, and why that is fine

Five rows stay below `full` even after every package, because they are summer2
*shapes* that summer4 has deliberately rejected rather than capabilities it
lacks. F7 and F8 are now `full` via `ForceOfInfection(kind=...)` (WP6 applied);
they are not in this table because they were capabilities to build, not shapes
rejected.

| ID | summer2 symbol | Why |
|---|---|---|
| L1 | `CompartmentalModel` | `FlowModel` owns flows over a map; times and solver options live elsewhere |
| S5 | `Compartment` | Compartments are rows, not objects |
| S6 | `AgeStratification` | A property plus a `TraitChain` flow, not a bundled class |
| S7 | `StrainStratification` | A property plus per-strain flows |
| P9 | `get_time_callable` | `compile()` returns the callable; there is no graph to wrap |

**100% capability does not require 100% symbol parity.** Read this ledger as a
statement of what a user can accomplish, not as a mandate to reproduce summer2's
API shapes.

## Notes carried from the session that produced this ledger

- The denominator is the summer2 surface **exercised by documentation**, not all
  of summer2. It was extracted mechanically from the notebooks; re-extract
  rather than editing from memory.
- Ragged stratification with three-valued logic is ahead of summer2, not merely
  equivalent: partial stratification there is a list of compartment names, here
  it is any selector, and `~p[t]` correctly excludes compartments where `p` does
  not apply.
- `SI` and `SIS` share a compartment map; the distinction lives in the flows.
  With `FlowModel` they are different objects.
- The flows layer originally scoped out timeseries; WP2 lands `CompiledModel.run`
  and a queryable `Result`, so trajectories are now first-class.
- No user research exists. Every satisfaction claim on this site is a heuristic
  evaluation; see {doc}`user-satisfaction`.
