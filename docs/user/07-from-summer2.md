# 7. Coming from summer2

This chapter is a translation table. The left-hand column is the summer2 API
that the [summer2 documentation](https://summer2.readthedocs.io) and the
[summer textbook](https://github.com/monash-emu/summer-textbook) use; the
right-hand column is what summer4 offers today.

## What translates

### Declaring compartments

summer2 names compartments as strings on the model constructor:

```python
# summer2
model = CompartmentalModel(
    times=(0.0, 100.0),
    compartments=["S", "I", "R"],
    infectious_compartments=["I"],
)
```

summer4 treats the compartment name as one property among many:

```python
# summer4
from summer4 import Property, PropertyMap

state = Property("state", ("S", "I", "R"))
pmap = PropertyMap.from_property(state)
```

There is no `infectious_compartments` argument, because there is nothing yet
that computes a force of infection. In summer4 that role will be played by a
selector (`state["I"]`) rather than a constructor list.

### Stratifying

```python
# summer2
strat = Stratification(name="age", strata=["0-4", "5-9", "10+"], compartments=["S", "I", "R"])
model.stratify_with(strat)
```

```python
# summer4
age = Property("age", ("0-4", "5-9", "10+"))
pmap = pmap.stratify(age)
```

The summer2 `compartments=` argument restricts a stratification to named
compartments. summer4 generalises it to a selector:

```python
# summer2: severity only on the infectious compartment
Stratification("severity", ["mild", "severe"], compartments=["I"])

# summer4: any query, including across several axes
pmap.stratify(severity, where=state["I"] & age["10+"])
```

Note that `stratify` **returns a new map** — summer2's `stratify_with` mutates
the model in place.

### Finding compartments

```python
# summer2
model.query_compartments({"age": "0-4", "name": "I"}, as_idx=True)
model.get_matching_compartments("I", {"age": "0-4"})
```

```python
# summer4
pmap.select(state["I"] & age["0-4"])   # int32 indices
pmap.mask(state["I"] & age["0-4"])     # boolean mask
pmap.select_one(state["I"] & age["0-4"])
```

summer2 queries are dictionaries, which can express conjunction only. summer4
queries are an algebra: `&`, `|`, `~`, multi-trait `isin`, plus `present()` and
`absent()`.

### Listing compartments

```python
# summer2
[str(c) for c in model.compartments]     # 'IXageX0-4'
c.name, c.strata                          # 'I', {'age': '0-4'}
```

```python
# summer4
pmap.labels()      # 'state=I_age=0-4'
pmap.to_dicts()    # {'state': 'I', 'age': '0-4'}
```

## Summary table

| summer2 | summer4 today | Notes |
|---|---|---|
| `CompartmentalModel(compartments=[...])` | `PropertyMap.from_property(Property(...))` | Compartment name becomes an ordinary property |
| `Compartment` | row of a `PropertyMap`; `labels()`, `to_dicts()` | No per-compartment object |
| `Stratification(name, strata)` | `Property(name, traits)` + `PropertyMap.stratify` | Returns a new map |
| `Stratification(..., compartments=[...])` | `stratify(prop, where=selector)` | Generalised from names to a query |
| `AgeStratification` | — | Was a `Stratification` plus ageing flows; the flow half does not exist |
| `StrainStratification` | — | Was a `Stratification` plus strain-aware force of infection |
| `model.query_compartments(dict)` | `pmap.select(selector)` | Algebra instead of a conjunction dict |
| `model.get_matching_compartments` | `pmap.select` / `pmap.select_one` | |
| `model.get_stratification(name)` | `pmap.get_property(name)`, `pmap.history` | History records the `where=` selector too |

## What does not translate

Everything below has **no summer4 equivalent at any level of the public API**.
This is the list that blocks reproducing the summer2 documentation and the
textbook as runnable material.

| Area | summer2 API |
|---|---|
| Model object | `CompartmentalModel`, `finalize`, `run`, `get_runner` |
| Initial conditions | `set_initial_population`, `get_initial_population`, `adjust_population_split`, `Stratification.set_population_split` |
| Transition flows | `add_transition_flow` |
| Infection flows | `add_infection_frequency_flow`, `add_infection_density_flow` |
| Entry flows | `add_crude_birth_flow`, `add_replacement_birth_flow`, `add_importation_flow` |
| Exit flows | `add_death_flow`, `add_universal_death_flows` |
| Flow adjustment | `Stratification.set_flow_adjustments`, `Multiply`, `Overwrite` |
| Infectiousness | `Stratification.add_infectiousness_adjustments` |
| Mixing | `Stratification.set_mixing_matrix` |
| Parameters | `Parameter`, `Function`, `Time`, `Data`, `DerivedOutput`, `set_default_parameters` |
| Time-varying rates | `get_linear_interpolation_function`, `get_sigmoidal_interpolation_function`, `get_piecewise_function`, `get_time_callable` |
| Solver | `SolverType`, `solve_ode`, adaptive/fixed-step integration |
| Derived outputs | `request_output_for_flow`, `request_output_for_compartments`, `request_aggregate_output`, `request_cumulative_output`, `request_function_output`, `request_computed_value_output`, `request_track_modelled_value`, `add_computed_value_func` |
| Results | `get_outputs_df`, `get_derived_outputs_df` |
| Real-world time | `ref_date`, `get_epoch` |
| Flow introspection | `query_flows`, `model.flows`, `summer2.inspect` |
| Calibration | everything in the summer textbook chapter 20 |

Design work for the flows, rates and adjustment portion of that list exists as a
spike under `explorations/flows/`; see {doc}`../dev/flows/index`, and
{doc}`../evaluation/if-promoted` for what promoting it would be worth. It is not
importable as part of `summer4` and nothing on this site depends on it.
