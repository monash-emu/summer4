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
2. **Against summer4's own intent.** The layers named in the project's README,
   plans and `explorations/flows/FINDINGS.md`.

## 1. Against the summer2 API

### Model lifecycle — 0 of 7

| summer2 | summer4 |
|---|---|
| `CompartmentalModel(times, compartments, infectious_compartments, timestep)` | — |
| `model.finalize()` | — |
| `model.run(parameters)` | — |
| `model.set_initial_population(distribution)` | — |
| `model.get_initial_population(parameters)` | — |
| `model.get_outputs_df()` | — |
| `model.get_derived_outputs_df()` | — |

There is no object that owns a model, and no representation of time at all.

### Compartments and stratification — 4 of 8 complete (5 covered)

| summer2 | summer4 |
|---|---|
| `Stratification(name, strata)` | `Property(name, traits)` + `PropertyMap.stratify` |
| `model.stratify_with(strat)` | `PropertyMap.stratify(prop, where=...)` — returns a new map |
| `Stratification(..., compartments=[...])` | `where=` selector — **more general** |
| `model.get_stratification(name)` | `PropertyMap.get_property(name)`, `PropertyMap.history` |
| `Compartment` (name + strata object) | rows; `labels()`, `to_dicts()` — no per-compartment object |
| `AgeStratification` | — (it was a stratification *plus* ageing flows) |
| `StrainStratification` | — (a stratification *plus* strain-aware force of infection) |
| `Stratification.set_population_split` / `model.adjust_population_split` | — |

This is the one area where summer4 is ahead. Partial stratification in summer2
is a list of compartment names; in summer4 it is any selector, and the resulting
ragged structure is handled by three-valued logic rather than by special cases.

### Compartment queries — 2 of 3

| summer2 | summer4 |
|---|---|
| `model.query_compartments(dict)` | `PropertyMap.select` / `mask` / `select_one` |
| `model.get_matching_compartments(name, strata)` | `PropertyMap.select` |
| `model.query_flows(...)` | — (there are no flows to query) |

summer2 queries are conjunction-only dictionaries. summer4 queries are an
algebra with `&`, `|`, `~`, multi-trait membership, and explicit presence and
absence. This is a genuine capability increase, not a translation.

### Flows — 0 of 8

`add_transition_flow`, `add_infection_frequency_flow`,
`add_infection_density_flow`, `add_death_flow`, `add_universal_death_flows`,
`add_crude_birth_flow`, `add_replacement_birth_flow`, `add_importation_flow`.

None exist. Prototypes of the transition/entry/exit trio exist in
`explorations/flows/`; the two infection flows — which require a force of
infection computed over a grouping — have no prototype.

### Flow adjustments — 0 of 4

`Stratification.set_flow_adjustments`, `Stratification.add_infectiousness_adjustments`,
`Multiply`, `Overwrite`.

The spike prototypes flow-owned `Multiply` / `Overwrite` / `Transform` with a
selector `where`, and concludes explicitly that adjustments should **not** hang
off `Stratification` as they do in summer2.

### Mixing — 0 of 1

`Stratification.set_mixing_matrix`. No prototype. This is the hardest remaining
item: it requires the force of infection to be a matrix-weighted coupling across
a grouping rather than a per-edge rate.

### Parameters and time-varying functions — 0 of 9

`Parameter`, `Function`, `Time`, `Data`, `DerivedOutput`,
`get_linear_interpolation_function`, `get_sigmoidal_interpolation_function`,
`get_piecewise_function`, `get_time_callable`.

The spike prototypes schema-built parameter references over a `NamedTuple` and a
small lazy rate expression tree, and concludes that a full compute graph
(summer2 uses `computegraph`) is not warranted yet. Nothing about time-varying
functions or interpolation has been prototyped.

### Derived outputs — 0 of 8

`request_output_for_flow`, `request_output_for_compartments`,
`request_aggregate_output`, `request_cumulative_output`,
`request_function_output`, `request_computed_value_output`,
`request_track_modelled_value`, `add_computed_value_func`.

`PropertyMap.partition` and `group_by` provide the *index sets* that
compartment-based outputs would aggregate over, which is a real head start, but
there is no request mechanism, no post-integration stage and no results object.

### Solver — 0 of 2

Solver selection and ODE integration. The spike contains a fixed-step Euler
shared between a NumPy loop and a JAX `lax.scan`, checked to agree over eight
steps. There is no adaptive solver, no diffrax integration, and no solver seam
in the package.

### Real-world time — 0 of 2

`ref_date` / epoch handling and `get_epoch`. Not started, and not prototyped.

### Totals

| Area | summer4 / summer2 |
|---|---|
| Model lifecycle | 0 / 7 |
| Compartments and stratification | 4 / 8 |
| Compartment queries | 2 / 3 |
| Flows | 0 / 8 |
| Flow adjustments | 0 / 4 |
| Mixing | 0 / 1 |
| Parameters and time-varying functions | 0 / 9 |
| Derived outputs | 0 / 8 |
| Solver | 0 / 2 |
| Real-world time | 0 / 2 |
| **Total (complete)** | **6 / 52 (12%)** |
| **Total (covered, incl. partial)** | 7 / 52 (13%) |

The 12% is not evenly distributed: it is one contiguous area — declaring and
querying a compartment space — implemented to a higher standard than summer2's
equivalent, with everything else absent.

## 2. Against summer4's own intended stack

| Layer | State | Evidence |
|---|---|---|
| Compartment taxonomy | **Implemented** | `src/summer4`, 570 lines, 4 test modules, benchmark suite |
| Query join (source × destination pairing) | Prototyped | `explorations/flows/prototype.py` |
| Flows (transition / entry / exit, chains, matrices) | Prototyped | as above |
| Lazy rates, derived parameters, adjustments | Prototyped | as above, plus `refine-flows.plan.md` |
| Array container over a map (`PropertyData`) | Prototyped | `explorations/flows/propertydata.py` |
| Fixed-step integrator | Prototyped | Euler, NumPy + `lax.scan` |
| Adaptive solver seam (diffrax) | Not started | `pyproject.toml` extra only |
| Force of infection / mixing | Not started | — |
| Derived outputs and results | Not started | — |
| Real-world time | Not started | — |
| Calibration | Not started | `pyproject.toml` extra only |

Three of the four committed plans in `plans/` are exploratory. The project has
done substantial design work on tier 1 and has not yet promoted any of it.

## What is genuinely strong

It is worth being specific, because "12%" undersells the quality of what exists:

- **Ragged stratification is a real advance.** summer2 handles partial
  stratification by listing compartment names; summer4 handles it with a
  selector and a three-valued algebra in which `~p[t]` correctly excludes
  compartments where `p` does not apply. That property is asserted by
  Hypothesis tests and is the kind of thing that is very hard to retrofit.
- **The representation is solver-ready.** A frozen, contiguous `int16` table
  plus an `int32` `parent_row` is exactly the shape a JAX kernel wants, and
  population redistribution is a gather rather than a join.
- **Error messages are good.** Every failure path names the offending value and
  lists the valid alternatives.
- **The invariants are tested as laws, not examples.** The Hypothesis suite
  asserts three-way exhaustion, partition coverage, stratification arithmetic
  and history replay for arbitrary maps.
