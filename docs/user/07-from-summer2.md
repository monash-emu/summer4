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

There is no `infectious_compartments` argument on the map constructor.
Infectious compartments are named when you build a force of infection in
`summer4.epi` (`ForceOfInfection`) — typically a selector such as
`state["I"]`, not a constructor list. summer2's `add_infection_*_flow`
becomes a `TransitionFlow` whose rate is a `ForceOfInfection`.

### Stratifying

```python
# summer2
strat = Stratification(name="age", strata=["0-4", "5-9", "10+"], compartments=["S", "I", "R"])
model.stratify_with(strat)
```

```python
# summer4 — map first
age = Property("age", ("0-4", "5-9", "10+"))
pmap = pmap.stratify(age)

# summer4 — or stratify the model after declaring flows (like stratify_with)
model.stratify(age)
```

The summer2 `compartments=` argument restricts a stratification to named
compartments. summer4 generalises it to a selector:

```python
# summer2: severity only on the infectious compartment
Stratification("severity", ["mild", "severe"], compartments=["I"])

# summer4: any query, including across several axes
pmap.stratify(severity, where=state["I"] & age["10+"])
# or: model.stratify(severity, where=state["I"] & age["10+"])
```

`PropertyMap.stratify` **returns a new map**. `FlowModel.stratify` mutates the
model in place like summer2's `stratify_with`, and declared flows re-resolve
when you `compile()`.

### Adjustments after stratifying

summer2 applies flow adjustments in **stratification order** (later
stratifications overwrite earlier ones where they overlap). summer4 orders
adjustments by **precedence level** (`Overwrite` → `Multiply` → `Transform`),
independent of when you declared them. Pass `precedence=` on an adjustment to
reproduce "later overwrite wins".

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
| `AgeStratification` | `Property` + `TraitChain` flow | No bundled convenience class |
| `StrainStratification` | `Property` + per-strain flows | Multi-strain FOI via `ForceOfInfection.per_trait`; no bundled class |
| `model.query_compartments(dict)` | `pmap.select(selector)` | Algebra instead of a conjunction dict |
| `model.get_matching_compartments` | `pmap.select` / `pmap.select_one` | |
| `model.get_stratification(name)` | `pmap.get_property(name)`, `pmap.history` | History records the `where=` selector too |
| `add_transition_flow` | `TransitionFlow` | {doc}`08-flows` |
| `add_death_flow` / `add_universal_death_flows` | `ExitFlow` | |
| `add_crude_birth_flow` / `add_replacement_birth_flow` / `add_importation_flow` | `EntryFlow` | Replacement births via `death.sum_over(...)` |
| `query_flows` | `CompiledModel.edges` / `Source` / `Dest` | |
| `Multiply` / `Overwrite` | same names on `adjust=` | Flow-owned, not on `Stratification` |
| `np.exp(graph_object)` and other ufuncs | `np.exp(rate)` — the same spelling | `jnp.exp(rate)` does not work; JAX has no dispatch hook |
| `finalize` | `FlowModel.compile()` | Returns a `CompiledModel` |
| `set_initial_population` / `get_initial_population` | `FlowModel.set_initial_population` / `CompiledModel.initial_state` | Declarative `InitialPopulation` |
| `set_population_split` / `adjust_population_split` | `Split(prop, weights, by=, where=)` | Weights normalised; even default on carriers |

## What does not translate

Infection FOI, mixing, infectiousness weights, interpolation helpers, adaptive
solvers, derived outputs, results and initial populations all have summer4
equivalents (see the table above and {doc}`../evaluation/coverage-ledger`).
What still has **no** library wrapper that would let every summer2 notebook run
as written:

| Area | summer2 API |
|---|---|
| Calibration | Bayesian workflow in the summer textbook chapter 20 (sparse `Target` / `TargetSet` fits exist; WP10 does not) |
