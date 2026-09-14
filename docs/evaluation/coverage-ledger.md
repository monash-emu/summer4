# Coverage ledger

**This file is the authoritative record of what summer4 covers and what 100%
would require.** Other branches, contributors and agents should read it before
planning work, and update it in the same commit that changes coverage.

```{admonition} Contract for anyone changing coverage
:class: important

1. The tables below are the single source of truth. Prose elsewhere on this
   site quotes their totals; it does not restate their contents.
2. Status values are exactly `full`, `partial` or `none`. Nothing else parses.
3. `Shipped` means importable from `summer4`. `Spike` means achievable with
   `explorations/flows/` promoted as-is.
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
| ID | summer2 symbol | Area | Shipped | Spike | summer4 equivalent | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | `CompartmentalModel` | lifecycle | `none` | `partial` | `FlowModel` | No times, timestep or infectious_compartments |
| L2 | `model.finalize()` | lifecycle | `none` | `full` | `FlowModel.compile()` | Actualizes every join once |
| L3 | `model.run()` | lifecycle | `none` | `partial` | `euler()` | Fixed step; returns final state only, no trajectory |
| L4 | `model.set_initial_population()` | lifecycle | `none` | `none` | — | Deferred by the spike as a separate later API |
| L5 | `model.get_initial_population()` | lifecycle | `none` | `none` | — |  |
| L6 | `model.get_outputs_df()` | lifecycle | `none` | `none` | — | No results object, no dataframe convention |
| L7 | `model.get_derived_outputs_df()` | lifecycle | `none` | `none` | — |  |
| S1 | `Stratification(name, strata)` | stratification | `full` | `full` | `Property + PropertyMap.stratify` |  |
| S2 | `model.stratify_with()` | stratification | `full` | `full` | `PropertyMap.stratify` | Returns a new map; summer2 mutates |
| S3 | `Stratification(compartments=[...])` | stratification | `full` | `full` | `stratify(prop, where=selector)` | Generalised from names to a query |
| S4 | `model.get_stratification()` | stratification | `full` | `full` | `PropertyMap.get_property / history` | History also records the where= selector |
| S5 | `Compartment` | stratification | `partial` | `partial` | `rows; labels(), to_dicts()` | No per-compartment object by design |
| S6 | `AgeStratification` | stratification | `none` | `partial` | `Property + TraitChain flow` | Ageing flows exist; no bundled convenience class |
| S7 | `StrainStratification` | stratification | `none` | `partial` | `Property + per-strain flows` | Strain-aware force of infection is hand-written |
| S8 | `set_population_split / adjust_population_split` | stratification | `none` | `none` | — | Flow split= is fan-out only, not initial population |
| Q1 | `model.query_compartments()` | queries | `full` | `full` | `PropertyMap.select / mask` | Algebra, not a conjunction dict |
| Q2 | `model.get_matching_compartments()` | queries | `full` | `full` | `PropertyMap.select / select_one` |  |
| Q3 | `model.query_flows()` | queries | `none` | `partial` | `actualize() -> ActualizedFlow` | Inspection of edges; not a query API |
| F1 | `add_transition_flow` | flows | `none` | `full` | `TransitionFlow` |  |
| F2 | `add_death_flow` | flows | `none` | `full` | `ExitFlow` |  |
| F3 | `add_universal_death_flows` | flows | `none` | `full` | `ExitFlow(Everything(), rate)` |  |
| F4 | `add_crude_birth_flow` | flows | `none` | `full` | `EntryFlow` | Absolute rate from a derived total |
| F5 | `add_replacement_birth_flow` | flows | `none` | `full` | `EntryFlow(dest, death.sum_over(...))` | Demonstrated in flows spike I |
| F6 | `add_importation_flow` | flows | `none` | `full` | `EntryFlow` | Absolute rate |
| F7 | `add_infection_frequency_flow` | flows | `none` | `partial` | `TransitionFlow + derived FOI` | User writes contact*I/N; not a primitive |
| F8 | `add_infection_density_flow` | flows | `none` | `partial` | `TransitionFlow + derived FOI` | User writes contact*I; not a primitive |
| A1 | `Stratification.set_flow_adjustments` | adjustments | `none` | `full` | `adjust= with where=` | Flow-owned, deliberately not on Stratification |
| A2 | `Multiply` | adjustments | `none` | `full` | `Multiply` | Default for a bare value in adjust= |
| A3 | `Overwrite` | adjustments | `none` | `full` | `Overwrite` | Supports where=Selector |
| A4 | `Stratification.add_infectiousness_adjustments` | adjustments | `none` | `none` | — | Needs a force-of-infection concept |
| M1 | `Stratification.set_mixing_matrix` | mixing | `none` | `none` | — | TraitMatrix moves people, it does not weight transmission |
| P1 | `Parameter` | parameters | `none` | `full` | `FieldRef via derived_refs` | Schema-checked, IDE-completable |
| P2 | `Function` | parameters | `none` | `full` | `Transform / callables in derived_fn` |  |
| P3 | `Time` | parameters | `none` | `full` | `t passed to derived_fn` |  |
| P4 | `DerivedOutput (as a rate input)` | parameters | `none` | `full` | `FlowRef, .sum(), .sum_over()` | Topologically ordered, cycles detected |
| P5 | `Data` | parameters | `none` | `none` | — | No data-loading surface |
| P6 | `get_linear_interpolation_function` | parameters | `none` | `none` | — |  |
| P7 | `get_sigmoidal_interpolation_function` | parameters | `none` | `none` | — |  |
| P8 | `get_piecewise_function` | parameters | `none` | `none` | — |  |
| P9 | `get_time_callable` | parameters | `none` | `partial` | `compile() -> vf(t, y, params)` | A time callable, but not summer2's graph wrapper |
| D1 | `request_output_for_flow` | outputs | `none` | `none` | — | FlowRef exposes mass to flows, not to the caller |
| D2 | `request_output_for_compartments` | outputs | `none` | `none` | — | partition/group_by give the index sets only |
| D3 | `request_aggregate_output` | outputs | `none` | `none` | — |  |
| D4 | `request_cumulative_output` | outputs | `none` | `none` | — |  |
| D5 | `request_function_output` | outputs | `none` | `none` | — |  |
| D6 | `request_computed_value_output` | outputs | `none` | `none` | — |  |
| D7 | `request_track_modelled_value` | outputs | `none` | `none` | — |  |
| D8 | `add_computed_value_func` | outputs | `none` | `full` | `derived_fn hook` | compute_derived_params runs every step |
| V1 | `solve_ode` | solver | `none` | `partial` | `euler()` | Fixed step, final state only |
| V2 | `SolverType / solver selection` | solver | `none` | `none` | — | No adaptive backend, no diffrax |
| T1 | `ref_date / Epoch` | time | `none` | `none` | — |  |
| T2 | `model.get_epoch()` | time | `none` | `none` | — |  |
<!-- /ledger:api -->

## Textbook ledger

Chapters of the [summer textbook](https://github.com/monash-emu/summer-textbook).
`partial` means the modelling content is expressible but the chapter cannot be
published as written.

<!-- ledger:textbook -->
| Ch | Title | Shipped | Spike | Blocker |
| --- | --- | --- | --- | --- |
| 1 | Infectious disease modelling | `full` | `full` | None - prose |
| 2 | Basic model construction | `partial` | `partial` | Trajectory, initial population, results frame |
| 3 | Thinking about flows | `none` | `partial` | Trajectory |
| 4 | Thinking about flow rates | `none` | `partial` | Trajectory, sojourn-time outputs |
| 5 | Series compartments and latency | `partial` | `partial` | Trajectory |
| 6 | Post-infection immunity | `partial` | `partial` | Trajectory |
| 7 | Obtaining numerical solutions | `none` | `partial` | Runge-Kutta backend and solver selection |
| 8 | Derived outputs | `none` | `none` | Derived outputs, results object |
| 9 | Transmission assumptions | `none` | `partial` | Trajectory |
| 10 | The reproduction number | `none` | `partial` | Trajectory, Rt as a derived output |
| 11 | Cyclical epidemic dynamics | `none` | `partial` | Trajectory, phase-plane outputs |
| 12 | Heterogeneous mixing introduction | `none` | `none` | Mixing matrices |
| 13 | Mixing and transmission types | `none` | `none` | Mixing matrices, population split |
| 14 | Assortative mixing | `none` | `none` | Mixing matrices, infectiousness adjustments |
| 15 | Susceptibility and infectiousness matrices | `none` | `none` | Mixing matrices, infectiousness adjustments |
| 16 | Thinking about contact surveys | `none` | `none` | Mixing matrices, contact-survey data |
| 17 | Understanding empiric contact data | `none` | `none` | Contact-survey data |
| 18 | Implementing empiric survey data | `none` | `none` | Contact-survey data |
| 19 | Adapting mixing matrices | `none` | `none` | Contact-survey data, matrix scaling |
| 20 | Calibration and uncertainty | `none` | `none` | Calibration workflow |
<!-- /ledger:textbook -->

## summer2 documentation ledger

<!-- ledger:summer2docs -->
| Page | Shipped | Spike | Blocker |
| --- | --- | --- | --- |
| `examples/01-basic-model` | `none` | `partial` | Trajectory, initial population, results frame |
| `examples/03-derived-outputs` | `none` | `none` | Derived outputs |
| `examples/04-flow-types` | `none` | `partial` | Trajectory, flow outputs |
| `examples/06-stratification-introduction` | `partial` | `partial` | Infectiousness adjustments, trajectory |
| `examples/07-age-stratification` | `partial` | `partial` | Population split, trajectory |
| `examples/08-strain-stratification` | `partial` | `partial` | Strain-aware FOI primitive, trajectory |
| `examples/09-mixing-matrices` | `none` | `none` | Mixing matrices |
| `examples/10-derived-outputs-stratified` | `none` | `none` | Derived outputs |
| `examples/11-flows-between-strata` | `none` | `partial` | Trajectory, compartment outputs |
| `detailed/time-varying-functions` | `none` | `partial` | Interpolation and piecewise helpers |
| `detailed/InitialPopulationGraphobject` | `none` | `none` | Initial population, parameters |
<!-- /ledger:summer2docs -->

## The path to 100%

Work packages in the order that maximises coverage gained per unit of effort.
Each names the ledger IDs it closes, so progress is checkable against the tables
above rather than against a narrative.

### WP1 — Promote the flows spike

**Closes:** L1 L2 L3 Q3 F1–F8 A1 A2 A3 P1 P2 P3 P4 P9 D8 V1 (to their `Spike`
status) · **Prerequisite for every later package**

The thirteen-item list in {doc}`../dev/explorations`, starting with making
`PropertyMap` hashable. The design is settled and the code is tested; see
{doc}`../dev/flows/index`. Takes the API ledger from 6 complete to 21, and 7
covered to 31.

Settle one open question first, because it affects the already-public API:
should `Present` and `Absent` bind their property in flow pairing?

### WP2 — Trajectories and a results object

**Closes:** L3 L6 L7 V1 (to `full`) · **Unblocks:** textbook 3, 4, 7, 9, 10, 11
and deepens 2, 5, 6; summer2 `01-basic-model`, `04-flow-types`,
`11-flows-between-strata`

The highest-value package in the list, and among the smallest. `_euler_jax`
already uses `lax.scan`; returning stacked per-step states instead of only the
carry is a few lines. What is missing is the **convention**: a results type, one
dataframe library, one plotting idiom. Nothing else in this ledger converts
"expressible" into "publishable".

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
the whitelist are the new work. Depends on WP2.

### WP5 — Time-varying function library

**Closes:** P5 P6 P7 P8 · **Unblocks:** summer2 `detailed/time-varying-functions`

Linear and sigmoidal interpolation, piecewise functions, and a data-loading
surface. Mechanically straightforward; must be JAX-traceable.

### WP6 — Force of infection and mixing

**Closes:** F7 F8 (to `full`) A4 M1 · **Unblocks:** textbook 12, 13, 14, 15

The largest genuinely unprototyped design problem remaining. A force of
infection is a reduction over a grouping fed back into a rate, and a mixing
matrix weights that coupling between strata. `TraitMatrix` does **not** do this:
it moves people, not transmission. Expect this to need its own spike.

### WP7 — Adaptive solver

**Closes:** V2 · **Unblocks:** textbook 7 fully

A diffrax backend behind the existing solver seam, plus solver selection.
`pyproject.toml` already declares the extra.

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
| WP1 | Promote the flows spike | *(applies every `Spike` status)* |
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
| today | 6 / 52 | 12% |
| WP1 | 21 / 52 | 40% |
| WP2 | 25 / 52 | 48% |
| WP3 | 28 / 52 | 54% |
| WP4 | 35 / 52 | 67% |
| WP5 | 39 / 52 | 75% |
| WP6 | 43 / 52 | 83% |
| WP7 | 44 / 52 | 85% |
| WP8 | 46 / 52 | 88% |
| WP9 | 46 / 52 | 88% |
| WP10 | 46 / 52 | 88% |
<!-- /ledger:progression -->

## What never reaches `full`, and why that is fine

Six rows stay below `full` even after every package, because they are summer2
*shapes* that summer4 has deliberately rejected rather than capabilities it
lacks:

| ID | summer2 symbol | Why |
|---|---|---|
| L1 | `CompartmentalModel` | `FlowModel` owns flows over a map; times and solver options live elsewhere |
| S5 | `Compartment` | Compartments are rows, not objects |
| S6 | `AgeStratification` | A property plus a `TraitChain` flow, not a bundled class |
| S7 | `StrainStratification` | A property plus per-strain flows |
| Q3 | `query_flows` | Edge inspection via `ActualizedFlow`, not a query API |
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
- `SI` and `SIS` are currently the same object in summer4, because the
  distinction lives entirely in the flows. WP1 fixes this.
- The spike's exclusion of timeseries is why WP1 alone publishes nothing. That
  was a deliberate scoping decision in `refine-flows.plan.md`, not an oversight,
  but it is the single most consequential line in the plan.
- No user research exists. Every satisfaction claim on this site is a heuristic
  evaluation; see {doc}`user-satisfaction`.
