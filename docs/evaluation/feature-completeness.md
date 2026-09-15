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

### Model lifecycle — 1 of 7 complete (3 covered)

| summer2 | summer4 |
|---|---|
| `CompartmentalModel(times, compartments, infectious_compartments, timestep)` | `FlowModel(pmap)` — no times, timestep or infectious_compartments |
| `model.finalize()` | `FlowModel.compile()` → `CompiledModel` |
| `model.run(parameters)` | `euler(...)` — final state only, no trajectory |
| `model.set_initial_population(distribution)` | — |
| `model.get_initial_population(parameters)` | — |
| `model.get_outputs_df()` | — |
| `model.get_derived_outputs_df()` | — |

### Compartments and stratification — 4 of 8 complete (7 covered)

| summer2 | summer4 |
|---|---|
| `Stratification(name, strata)` | `Property(name, traits)` + `PropertyMap.stratify` |
| `model.stratify_with(strat)` | `PropertyMap.stratify(prop, where=...)` — returns a new map |
| `Stratification(..., compartments=[...])` | `where=` selector — **more general** |
| `model.get_stratification(name)` | `PropertyMap.get_property(name)`, `PropertyMap.history` |
| `Compartment` (name + strata object) | rows; `labels()`, `to_dicts()` — no per-compartment object |
| `AgeStratification` | `Property` plus a `TraitChain` flow; no bundled class |
| `StrainStratification` | `Property` plus per-strain flows; strain-aware FOI is hand-written |
| `Stratification.set_population_split` / `model.adjust_population_split` | — |

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

### Flows — 6 of 8 complete (8 covered)

`TransitionFlow`, `ExitFlow` and `EntryFlow` cover transition, death, universal
death, crude birth, replacement birth and importation. The two infection
constructors are expressible but not primitives: the user writes
`contact * I / N` (or `contact * I`) in `derived_fn` and passes a `FieldRef` as
the rate (F7, F8 `partial`).

### Flow adjustments — 3 of 4

`adjust=` with `Multiply` / `Overwrite` / `Transform` and a selector `where`
covers `set_flow_adjustments`. Adjustments belong to the flow, not to a
`Stratification`. `add_infectiousness_adjustments` does not translate: it
weights a compartment's contribution to a force of infection that is not a
library concept.

### Mixing — 0 of 1

`Stratification.set_mixing_matrix`. `TraitMatrix` looks superficially similar
and is a different thing: it moves *people* between strata (migration), whereas
a mixing matrix weights *transmission* between strata.

### Parameters and time-varying functions — 4 of 9 complete (5 covered)

`derived_refs` over a `NamedTuple` gives schema-checked `Parameter` equivalents;
`t` reaches `derived_fn` every step; `FlowRef` lets one flow's mass feed
another. Missing is the *library* of time functions — linear and sigmoidal
interpolation, piecewise — and a data-loading surface.

### Derived outputs — 1 of 8

`derived_fn` ≈ `add_computed_value_func`. There is no request mechanism, no
post-integration stage and no results object. `FlowRef` exposes mass to *other
flows* inside the vector field, not to the caller.

### Solver — 0 of 2 complete (1 covered)

`euler` is a fixed-step stepper returning the final state (`partial` vs
`solve_ode`). No adaptive backend, no solver selection.

### Real-world time — 0 of 2

`ref_date` / epoch handling and `get_epoch`. Not started.

### Totals

| Area | summer4 / summer2 |
|---|---|
| Model lifecycle | 1 / 7 |
| Compartments and stratification | 4 / 8 |
| Compartment queries | 3 / 3 |
| Flows | 6 / 8 |
| Flow adjustments | 3 / 4 |
| Mixing | 0 / 1 |
| Parameters and time-varying functions | 4 / 9 |
| Derived outputs | 1 / 8 |
| Solver | 0 / 2 |
| Real-world time | 0 / 2 |
| **Total (complete)** | **22 / 52 (42%)** |
| **Total (covered, incl. partial)** | 31 / 52 (60%) |

The 42% is two contiguous areas — declaring a compartment space, then compiling
named flows over it — with everything above the vector field (trajectories,
outputs, mixing, calibration) still absent or partial.

## 2. Against summer4's own intended stack

| Layer | State | Evidence |
|---|---|---|
| Compartment taxonomy | **Implemented** | `src/summer4` |
| Query join (source × destination pairing) | **Implemented** | `summer4.flows.join` |
| Flows (transition / entry / exit, chains, matrices) | **Implemented** | `TransitionFlow` / `ExitFlow` / `EntryFlow` |
| Lazy rates, derived parameters, adjustments | **Implemented** | `FieldRef`, `FlowRef`, `adjust=` |
| Array container over a map (`PropertyData`) | **Implemented** | `summer4.jax.propertydata` |
| Edge queries | **Implemented** | `EdgeMap`, `Source` / `Dest` |
| Compiled model + fixed-step integrator | **Implemented** | `CompiledModel`, `euler` (final state) |
| Adaptive solver seam (diffrax) | Not started | `pyproject.toml` extra only |
| Force of infection / mixing | Not started | F7/F8 hand-written |
| Derived outputs and results | Not started | — |
| Real-world time | Not started | — |
| Calibration | Not started | `pyproject.toml` extra only |

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
