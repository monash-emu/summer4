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
| L3 | `model.run()` | lifecycle | `partial` | `euler()` | Fixed step; returns final state only, no trajectory |
| L4 | `model.set_initial_population()` | lifecycle | `none` | — | Deferred as a separate later API |
| L5 | `model.get_initial_population()` | lifecycle | `none` | — |  |
| L6 | `model.get_outputs_df()` | lifecycle | `none` | — | No results object, no dataframe convention |
| L7 | `model.get_derived_outputs_df()` | lifecycle | `none` | — |  |
| S1 | `Stratification(name, strata)` | stratification | `full` | `Property + PropertyMap.stratify` |  |
| S2 | `model.stratify_with()` | stratification | `full` | `PropertyMap.stratify` | Returns a new map; summer2 mutates |
| S3 | `Stratification(compartments=[...])` | stratification | `full` | `stratify(prop, where=selector)` | Generalised from names to a query |
| S4 | `model.get_stratification()` | stratification | `full` | `PropertyMap.get_property / history` | History also records the where= selector |
| S5 | `Compartment` | stratification | `partial` | `rows; labels(), to_dicts()` | No per-compartment object by design |
| S6 | `AgeStratification` | stratification | `partial` | `Property + TraitChain flow` | Ageing flows exist; no bundled convenience class |
| S7 | `StrainStratification` | stratification | `partial` | `Property + per-strain flows` | Strain-aware force of infection is hand-written |
| S8 | `set_population_split / adjust_population_split` | stratification | `none` | — | Flow split= is fan-out only, not initial population |
| Q1 | `model.query_compartments()` | queries | `full` | `PropertyMap.select / mask` | Algebra, not a conjunction dict |
| Q2 | `model.get_matching_compartments()` | queries | `full` | `PropertyMap.select / select_one` |  |
| Q3 | `model.query_flows()` | queries | `full` | `EdgeMap` / `CompiledModel.edges` / `Source`/`Dest` | Query API over flow edges |
| F1 | `add_transition_flow` | flows | `full` | `TransitionFlow` |  |
| F2 | `add_death_flow` | flows | `full` | `ExitFlow` |  |
| F3 | `add_universal_death_flows` | flows | `full` | `ExitFlow(Everything(), rate)` |  |
| F4 | `add_crude_birth_flow` | flows | `full` | `EntryFlow` | Absolute rate from a derived total |
| F5 | `add_replacement_birth_flow` | flows | `full` | `EntryFlow(dest, death.sum_over(...))` |  |
| F6 | `add_importation_flow` | flows | `full` | `EntryFlow` | Absolute rate |
| F7 | `add_infection_frequency_flow` | flows | `partial` | `TransitionFlow + derived FOI` | User writes contact*I/N; not a primitive |
| F8 | `add_infection_density_flow` | flows | `partial` | `TransitionFlow + derived FOI` | User writes contact*I; not a primitive |
| A1 | `Stratification.set_flow_adjustments` | adjustments | `full` | `adjust= with where=` | Flow-owned, deliberately not on Stratification |
| A2 | `Multiply` | adjustments | `full` | `Multiply` | Default for a bare value in adjust= |
| A3 | `Overwrite` | adjustments | `full` | `Overwrite` | Supports where=Selector |
| A4 | `Stratification.add_infectiousness_adjustments` | adjustments | `none` | — | Needs a force-of-infection concept |
| M1 | `Stratification.set_mixing_matrix` | mixing | `none` | — | TraitMatrix moves people, it does not weight transmission |
| P1 | `Parameter` | parameters | `full` | `FieldRef via derived_refs` | Schema-checked, IDE-completable |
| P2 | `Function` | parameters | `full` | `Transform / callables in derived_fn` |  |
| P3 | `Time` | parameters | `full` | `t passed to derived_fn` |  |
| P4 | `DerivedOutput (as a rate input)` | parameters | `full` | `FlowRef, .sum(), .sum_over()` | Topologically ordered, cycles detected |
| P5 | `Data` | parameters | `none` | — | No data-loading surface |
| P6 | `get_linear_interpolation_function` | parameters | `none` | — |  |
| P7 | `get_sigmoidal_interpolation_function` | parameters | `none` | — |  |
| P8 | `get_piecewise_function` | parameters | `none` | — |  |
| P9 | `get_time_callable` | parameters | `partial` | `compile() -> vf(t, y, params)` | A time callable, but not summer2's graph wrapper |
| D1 | `request_output_for_flow` | outputs | `none` | — | FlowRef exposes mass to flows, not to the caller |
| D2 | `request_output_for_compartments` | outputs | `none` | — | partition/group_by give the index sets only |
| D3 | `request_aggregate_output` | outputs | `none` | — |  |
| D4 | `request_cumulative_output` | outputs | `none` | — |  |
| D5 | `request_function_output` | outputs | `none` | — |  |
| D6 | `request_computed_value_output` | outputs | `none` | — |  |
| D7 | `request_track_modelled_value` | outputs | `none` | — |  |
| D8 | `add_computed_value_func` | outputs | `full` | `derived_fn hook` | compute_derived_params runs every step |
| V1 | `solve_ode` | solver | `partial` | `euler()` | Fixed step, final state only |
| V2 | `SolverType / solver selection` | solver | `none` | — | No adaptive backend, no diffrax |
| T1 | `ref_date / Epoch` | time | `none` | — |  |
| T2 | `model.get_epoch()` | time | `none` | — |  |
<!-- /ledger:api -->

## Textbook ledger

Chapters of the [summer textbook](https://github.com/monash-emu/summer-textbook).
`partial` means the modelling content is expressible but the chapter cannot be
published as written.

<!-- ledger:textbook -->
| Ch | Title | Status | Blocker |
| --- | --- | --- | --- |
| 1 | Infectious disease modelling | `full` | None - prose |
| 2 | Basic model construction | `partial` | Trajectory, initial population, results frame |
| 3 | Thinking about flows | `partial` | Trajectory |
| 4 | Thinking about flow rates | `partial` | Trajectory, sojourn-time outputs |
| 5 | Series compartments and latency | `partial` | Trajectory |
| 6 | Post-infection immunity | `partial` | Trajectory |
| 7 | Obtaining numerical solutions | `partial` | Runge-Kutta backend and solver selection |
| 8 | Derived outputs | `none` | Derived outputs, results object |
| 9 | Transmission assumptions | `partial` | Trajectory |
| 10 | The reproduction number | `partial` | Trajectory, Rt as a derived output |
| 11 | Cyclical epidemic dynamics | `partial` | Trajectory, phase-plane outputs |
| 12 | Heterogeneous mixing introduction | `none` | Mixing matrices |
| 13 | Mixing and transmission types | `none` | Mixing matrices, population split |
| 14 | Assortative mixing | `none` | Mixing matrices, infectiousness adjustments |
| 15 | Susceptibility and infectiousness matrices | `none` | Mixing matrices, infectiousness adjustments |
| 16 | Thinking about contact surveys | `none` | Mixing matrices, contact-survey data |
| 17 | Understanding empiric contact data | `none` | Contact-survey data |
| 18 | Implementing empiric survey data | `none` | Contact-survey data |
| 19 | Adapting mixing matrices | `none` | Contact-survey data, matrix scaling |
| 20 | Calibration and uncertainty | `none` | Calibration workflow |
<!-- /ledger:textbook -->

## summer2 documentation ledger

<!-- ledger:summer2docs -->
| Page | Status | Blocker |
| --- | --- | --- |
| `examples/01-basic-model` | `partial` | Trajectory, initial population, results frame |
| `examples/03-derived-outputs` | `none` | Derived outputs |
| `examples/04-flow-types` | `partial` | Trajectory, flow outputs |
| `examples/06-stratification-introduction` | `partial` | Infectiousness adjustments, trajectory |
| `examples/07-age-stratification` | `partial` | Population split, trajectory |
| `examples/08-strain-stratification` | `partial` | Strain-aware FOI primitive, trajectory |
| `examples/09-mixing-matrices` | `none` | Mixing matrices |
| `examples/10-derived-outputs-stratified` | `none` | Derived outputs |
| `examples/11-flows-between-strata` | `partial` | Trajectory, compartment outputs |
| `detailed/time-varying-functions` | `partial` | Interpolation and piecewise helpers |
| `detailed/InitialPopulationGraphobject` | `none` | Initial population, parameters |
<!-- /ledger:summer2docs -->

## The path to 100%

Work packages in the order that maximises coverage gained per unit of effort.
Each names the ledger IDs it closes, so progress is checkable against the tables
above rather than against a narrative.

WP1 (promote the flows spike into `summer4`) is **applied**. Flows, rates,
adjustments, `EdgeMap`, `CompiledModel` and a JAX Euler ship in
`summer4.flows`. The "today" row below is post-promotion.

`Present` and `Absent` are **non-binding** in flow pairing: they name a property
(`selector_properties`) but do not bind it (`selector_values`). Binding is
`Trait` / `IsIn` only. `strict_pairing=True` raises when an unbound property
would move people.

### WP2 — Trajectories and a results object

**Closes:** L3 L6 L7 V1 (to `full`) · **Unblocks:** textbook 3, 4, 7, 9, 10, 11
and deepens 2, 5, 6; summer2 `01-basic-model`, `04-flow-types`,
`11-flows-between-strata`

The highest-value package in the list, and among the smallest. The compiled Euler
already uses `lax.scan`; returning stacked per-step states instead of only the
carry is a few lines. What is missing is the **convention**: a results type, one
dataframe library, one plotting idiom. Nothing else in this ledger converts
"expressible" into "publishable". Plan phase: `feat/results`.

### WP3 — Initial population

**Closes:** L4 L5 S8 · **Unblocks:** summer2 `InitialPopulationGraphobject`

`parent_row` already makes redistribution a gather and a divide — see
{doc}`../user/06-immutability-and-provenance`. This is an API wrapper over
mechanics that exist.

### WP4 — Derived outputs

**Closes:** D1–D7 · **Unblocks:** textbook 8, 10; summer2 `03-derived-outputs`,
`10-derived-outputs-stratified`

`partition` and `group_by` already produce the index sets that compartment
outputs aggregate over. The request mechanism, the post-integration stage and
the whitelist are the new work. Depends on WP2. Plan phases: flow-outputs
(`feat/flow-outputs`) then sparse-targets (`feat/sparse-targets`, no API-row
delta).

### WP5 — Time-varying function library

**Closes:** P5 P6 P7 P8 · **Unblocks:** summer2 `detailed/time-varying-functions`

Linear and sigmoidal interpolation, piecewise functions, and a data-loading
surface. Mechanically straightforward; must be JAX-traceable.

### WP6 — Force of infection and mixing

**Closes:** F7 F8 (to `full`) A4 M1 · **Unblocks:** textbook 12, 13, 14, 15

The largest genuinely unprototyped design problem remaining. A force of
infection is a reduction over a grouping fed back into a rate, and a mixing
matrix weights that coupling between strata. `TraitMatrix` does **not** do this:
it moves people, not transmission.

### WP7 — Adaptive solver

**Closes:** V2 · **Unblocks:** textbook 7 fully

A diffrax backend behind the existing solver seam, plus solver selection.
`pyproject.toml` already declares the extra. Plan phase: `feat/diffrax-solver`.

### WP8 — Real-world time

**Closes:** T1 T2

Epoch handling so model times can be dates. Needed by every applied example.

### WP9 — Contact survey data

**Unblocks:** textbook 16, 17, 18, 19

Loading, validating, inspecting and scaling empirical contact matrices.
Depends on WP6.

### WP10 — Calibration

**Unblocks:** textbook 20

A Bayesian workflow over JAX-differentiable models. `numpyro` and `optax` are
declared extras and unused. Depends on everything above.

## Work packages, declared

`Closes` lists the ledger IDs each package raises to `full`. The progression
table below is **computed** from these declarations by
`scripts/coverage_report.py`; do not edit it by hand.

<!-- ledger:packages -->
| WP | Name | Closes |
| --- | --- | --- |
| WP2 | Trajectories and a results object | L3 L6 L7 V1 |
| WP3 | Initial population | L4 L5 S8 |
| WP4 | Derived outputs | D1 D2 D3 D4 D5 D6 D7 |
| WP5 | Time-varying function library | P5 P6 P7 P8 |
| WP6 | Force of infection and mixing | F7 F8 A4 M1 |
| WP7 | Adaptive solver | V2 |
| WP8 | Real-world time | T1 T2 |
| WP9 | Contact survey data | *(no API rows; unblocks textbook 16-19)* |
| WP10 | Calibration | *(no API rows; unblocks textbook 20)* |
<!-- /ledger:packages -->

## Coverage after each package

<!-- ledger:progression -->
| After | API rows at `full` | Share |
| --- | --- | --- |
| today | 22 / 52 | 42% |
| WP2 | 26 / 52 | 50% |
| WP3 | 29 / 52 | 56% |
| WP4 | 36 / 52 | 69% |
| WP5 | 40 / 52 | 77% |
| WP6 | 44 / 52 | 85% |
| WP7 | 45 / 52 | 87% |
| WP8 | 47 / 52 | 90% |
| WP9 | 47 / 52 | 90% |
| WP10 | 47 / 52 | 90% |
<!-- /ledger:progression -->

## What never reaches `full`, and why that is fine

Five rows stay below `full` even after every package, because they are summer2
*shapes* that summer4 has deliberately rejected rather than capabilities it
lacks. Infection-frequency/density constructors (F7, F8) stay `partial` until
WP6 lands a force-of-infection primitive; they are not in this table because
they are capabilities still to build, not shapes rejected.

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
- The flows layer scoped out timeseries, so `euler` returns only the final
  state. That was a deliberate decision, and it is why WP2 is the binding
  constraint: every summer2 notebook and almost every textbook chapter ends by
  plotting a trajectory.
- No user research exists. Every satisfaction claim on this site is a heuristic
  evaluation; see {doc}`user-satisfaction`.
