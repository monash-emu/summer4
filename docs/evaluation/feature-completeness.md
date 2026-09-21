# Feature completeness

```{admonition} The ledger is authoritative
:class: note

{doc}`coverage-ledger` holds the row-by-row record and computes every total on
this page. If the two disagree, the ledger is right and this page is stale —
`pixi run coverage` says so.
```

## Method

Two comparisons, both derived from source rather than from memory:

1. **Against summer2.** Every summer2 API symbol exercised by the summer2
   documentation notebooks (`docs/examples/*.ipynb`, `docs/detailed/*.ipynb`) or
   by the summer textbook notebooks, extracted mechanically, then checked for a
   summer4 equivalent.
2. **Against summer4's own intent.** The layers named in the project's README
   and plans.

## 1. Against the summer2 API

### Model lifecycle — 4 of 7 complete (5 covered)

| summer2 | summer4 |
|---|---|
| `CompartmentalModel(times, compartments, infectious_compartments, timestep)` | `FlowModel(pmap)` — no times, timestep or infectious_compartments |
| `model.finalize()` | `FlowModel.compile()` → `CompiledModel` |
| `model.run(parameters)` | `CompiledModel.run(...)` → `Result` |
| `model.set_initial_population(distribution)` | `FlowModel.set_initial_population` / `InitialPopulation` |
| `model.get_initial_population(parameters)` | `CompiledModel.initial_state` |
| `model.get_outputs_df()` | `Output.to_frame` / `to_pandas` |
| `model.get_derived_outputs_df()` | `Result` outputs via `SavePlan` |

### Compartments and stratification — 4 of 8 complete (7 covered)

| summer2 | summer4 |
|---|---|
| `Stratification(name, strata)` | `Property(name, traits)` + `PropertyMap.stratify` |
| `model.stratify_with(strat)` | `PropertyMap.stratify(prop, where=...)` — returns a new map |
| `Stratification(..., compartments=[...])` | `where=` selector — **more general** |
| `model.get_stratification(name)` | `PropertyMap.get_property(name)`, `PropertyMap.history` |
| `Compartment` (name + strata object) | rows; `labels()`, `to_dicts()` — no per-compartment object |
| `AgeStratification` | `Property` plus a `TraitChain` flow; no bundled class |
| `StrainStratification` | `Property` plus per-strain flows; multi-strain FOI via `ForceOfInfection.per_trait` (no bundled class) |
| `Stratification.set_population_split` / `model.adjust_population_split` | `Split(..., where=)` |

This is the one area where summer4 is ahead. Partial stratification in summer2
is a list of compartment names; in summer4 it is any selector, and the resulting
ragged structure is handled by three-valued logic rather than by special cases.

### Compartment queries — 3 of 3

| summer2 | summer4 |
|---|---|
| `model.query_compartments(dict)` | `PropertyMap.select` / `mask` / `select_one` |
| `model.get_matching_compartments(name, strata)` | `PropertyMap.select` |
| `model.query_flows(...)` | `CompiledModel.edges` / `EdgeMap` with `Source` / `Dest` |

summer2 queries are conjunction-only dictionaries. summer4 queries are an
algebra with `&`, `|`, `~`, multi-trait membership, and explicit presence and
absence. Flow-edge queries use the same algebra, polarity-wrapped.

### Flows — 8 of 8 complete (8 covered)

`TransitionFlow`, `ExitFlow` and `EntryFlow` cover transition, death, universal
death, crude birth, replacement birth and importation. Infection frequency and
density are `ForceOfInfection(kind=FOIKind.FREQUENCY|DENSITY)` (F7, F8 `full`).

### Flow adjustments — 4 of 4

`adjust=` with `Multiply` / `Overwrite` / `Transform` and a selector `where`
covers `set_flow_adjustments`. Adjustments belong to the flow, not to a
`Stratification`. Infectiousness weights are FOI-owned:
`ForceOfInfection(infectiousness=...)` (A4 `full`).

### Mixing — 1 of 1

`summer4.epi.MixingMatrix` weights transmission between strata (M1 `full`).
`TraitMatrix` remains a different thing: it moves *people* between strata.

### Parameters and time-varying functions — 8 of 9 complete (9 covered)

`derived_refs` over a `NamedTuple` gives schema-checked `Parameter` equivalents;
`Time()` and `t` on `derived_fn` supply model time; `FlowRef` lets one flow's
mass feed another. WP5 applied: `summer4.timevarying` (`linear`, `sigmoidal`,
`step` / `piecewise`) and dated-series `summer4.data.Data`. `get_time_callable`
remains `partial` — `compile()` returns `vf(t, y, params)`, not summer2's graph
wrapper (P9).

### Derived outputs — 8 of 8

`FlowMass` / `Output` edge queries, `Compartments(where=)`, `Output.sum_over` /
`total` / `partition`, `Output.cumulative()`, `SaveFn` plus output arithmetic,
and `ComputedValue` / `derived_fn` cover D1–D8.

### Solver — 2 of 2

`CompiledModel.run` over `euler` (V1) and `solver=` name or diffrax instance
(V2). Euler remains the reference stepper.

### Real-world time — 2 of 2

`Epoch` and `Result.times.epoch` / `TimeAxis.epoch` (T1, T2).

### Totals

| Area | summer4 / summer2 |
|---|---|
| Model lifecycle | 4 / 7 |
| Compartments and stratification | 4 / 8 |
| Compartment queries | 3 / 3 |
| Flows | 8 / 8 |
| Flow adjustments | 4 / 4 |
| Mixing | 1 / 1 |
| Parameters and time-varying functions | 8 / 9 |
| Derived outputs | 8 / 8 |
| Solver | 2 / 2 |
| Real-world time | 2 / 2 |
| **Total (complete)** | **47 / 52 (90%)** |
| **Total (covered, incl. partial)** | 52 / 52 (100%) |

The remaining API gaps are five deliberate shape mismatches that never reach
`full` (L1, S5, S6, S7, P9). See {doc}`coverage-ledger`.

## 2. Against summer4's own intended stack

| Layer | State | Evidence |
|---|---|---|
| Compartment taxonomy | **Implemented** | `src/summer4` |
| Query join (source × destination pairing) | **Implemented** | `summer4.flows.join` |
| Flows (transition / entry / exit, chains, matrices) | **Implemented** | `TransitionFlow` / `ExitFlow` / `EntryFlow` |
| Lazy rates, derived parameters, adjustments | **Implemented** | `FieldRef`, `FlowRef`, `adjust=` |
| Array container over a map (`PropertyData`) | **Implemented** | `summer4.jax.propertydata` |
| Edge queries | **Implemented** | `EdgeMap`, `Source` / `Dest` |
| Compiled model + fixed-step integrator | **Implemented** | `CompiledModel`, `euler` |
| Adaptive solver seam (diffrax) | **Implemented** | `solver=` name or diffrax instance |
| Force of infection / mixing | **Implemented** | `summer4.epi` (`ForceOfInfection`, `MixingMatrix`) |
| Derived outputs and results | **Implemented** | `Result`, `Output`, `SavePlan`, `FlowMass` |
| Real-world time | **Implemented** | `Epoch`, `TimeAxis` |
| Time-varying rates / dated data | **Implemented** | `summer4.timevarying`, `summer4.data` |
| Calibration | Partial | Sparse `Target` / `TargetSet` (WP11); Bayesian workflow (WP10) ahead |

## What is genuinely strong

- **Ragged stratification is a real advance.** summer2 handles partial
  stratification by listing compartment names; summer4 handles it with a
  selector and a three-valued algebra in which `~p[t]` correctly excludes
  compartments where `p` does not apply.
- **The representation is solver-ready.** A frozen, contiguous `int16` table
  plus an `int32` `parent_row` is exactly the shape a JAX kernel wants.
- **Flow edges are queryable.** `CompiledModel.edges` returns an `EdgeMap`;
  `Source` / `Dest` reuse the compartment algebra on polarity-mangled columns.
- **The invariants are tested as laws, not examples.** The Hypothesis suite
  asserts three-way exhaustion, partition coverage, stratification arithmetic
  and history replay for arbitrary maps.
