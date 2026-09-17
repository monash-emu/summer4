# Architecture

summer4 is a JAX-native compartmental modelling platform, built bottom-up.
The taxonomy and flows layers are implemented.

## The intended stack

```{mermaid}
flowchart TB
    subgraph done ["Implemented — src/summer4"]
        T["Taxonomy<br/>Property · Trait · Selector<br/>PropertyMap · Stratification"]
        J["Query join<br/>source × dest pairing"]
        F["Flows<br/>Transition · Entry · Exit"]
        R["Lazy rates<br/>FieldRef · FlowRef · adjust"]
        E["CompiledModel<br/>JAX vector field · euler"]
    end
    subgraph todo ["Not started"]
        P["Time-varying function library"]
        S["Solver seam<br/>diffrax"]
        D["Derived outputs"]
        X["Mixing matrices /<br/>force of infection"]
        C["Calibration"]
        O["Results / trajectories"]
    end

    T --> J --> F --> R --> E
    R --> P
    E --> S --> O
    O --> D --> C
    F --> X
```

Taxonomy, join, flows, rates and the compiled Euler are importable from
`summer4`. JAX is required to evaluate `CompiledModel.vector_field` and
`euler`; the taxonomy itself stays NumPy-only.

## The taxonomy layer

The implemented taxonomy answers one question — *what are the compartments, and
how do I refer to a subset of them?* — with three ideas.

### 1. A compartment is a row of trait codes

There is no `Compartment` class. A compartment is a row index into a single
`int16` matrix whose columns are properties. Names are derived on demand
(`labels()`, `to_dicts()`); they are never the identity of a compartment.

This is a departure from summer2, where `Compartment` is an object carrying a
name and a strata dictionary, and where compartment lookup is string matching.

### 2. Stratification is a pure function

`PropertyMap.stratify(prop, where=sel)` returns a new map. The old map remains
valid and queryable. Two consequences matter for the layers above:

- A model's compartment space can be built as an expression, branched, and
  compared, without a mutable builder object.
- Every map carries the full `history` of `Stratification` values that produced
  it, so the construction is replayable and auditable.

### 3. Queries are a three-valued algebra, not a dictionary

Selectors form a small expression tree (`Trait`, `IsIn`, `Present`, `Absent`,
`Everything`, `Nothing`, `And`, `Or`, `Not`, `Source`, `Dest`) that a map
compiles to an `int8` Kleene array and then to `int32` indices. Ragged maps —
a property present on some compartments only — are the reason for the third
truth value. `Source` / `Dest` evaluate on an `EdgeMap`, not a compartment
map.

## The flows layer

A {class}`~summer4.flows.compiled.FlowModel` owns named flows over one map.
`compile()` actualizes every join once and returns a static
{class}`~summer4.flows.compiled.CompiledModel` with a JAX vector field and
{meth}`~summer4.flows.compiled.CompiledModel.edges` for inspection.
{func}`~summer4.euler` is the reference fixed-step backend;
{meth}`~summer4.flows.compiled.CompiledModel.run` returns a queryable
{class}`~summer4.results.result.Result` (Euler or a diffrax solver).

## Dependency policy

The taxonomy (`properties`, `selectors`, `propertymap`) depends on **NumPy
only**. The compiled vector field and `summer4.jax.PropertyData` import JAX
lazily. Optional extras declare `jax`, `diffrax` and `equinox`.

The documentation environment installs JAX because the user-guide flows
notebook compiles a vector field at build time.

## Where the seams are

| Decision | Consequence |
|---|---|
| `parent_row` on every stratified map | Initial-population splitting and coarse/fine aggregation are gathers, not joins |
| Frozen, contiguous `codes` | The table can be handed to a JAX kernel without a copy |
| Value-compared frozen selectors | The per-map query cache is keyed by selector value, so equal queries built independently share work |
| `PropertyMap` hashes by content digest | Equal rebuilt maps share a jit cache as static arguments |
| `Present` / `Absent` are non-binding | Pairing binds `Trait` / `IsIn` only; `strict_pairing` guards leftover movement |
