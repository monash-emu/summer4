# Architecture

summer4 is planned as a JAX-native compartmental modelling platform. It is being
built bottom-up, and exactly one layer is implemented.

## The intended stack

```{mermaid}
flowchart TB
    subgraph done ["Implemented — src/summer4"]
        T["Taxonomy<br/>Property · Trait · Selector<br/>PropertyMap · Stratification"]
    end
    subgraph spike ["Spiked — explorations/flows (not public)"]
        J["Query join<br/>source x dest pairing"]
        F["Flows<br/>Transition · Entry · Exit"]
        R["Lazy rates<br/>FieldRef · FlowRef · adjust"]
        E["Euler step<br/>numpy + lax.scan"]
    end
    subgraph todo ["Not started"]
        P["Parameters &<br/>time-varying functions"]
        S["Solver seam<br/>diffrax"]
        D["Derived outputs"]
        X["Mixing matrices /<br/>force of infection"]
        C["Calibration"]
        O["Results / xarray"]
    end

    T --> J --> F --> R --> E
    R --> P
    E --> S --> O
    O --> D --> C
    F --> X
```

Only the top box is importable from `summer4`. The middle box lives in
`explorations/flows/` and is covered by its own tests, but it is deliberately
outside the package: see {doc}`explorations`.

## The taxonomy layer

The implemented layer answers one question — *what are the compartments, and how
do I refer to a subset of them?* — and answers it with three ideas.

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
`Everything`, `Nothing`, `And`, `Or`, `Not`) that a map compiles to an `int8`
Kleene array and then to `int32` indices. Ragged maps — a property present on
some compartments only — are the reason for the third truth value, and they are
the central design commitment of this layer.

## Dependency policy

`src/summer4` depends on **NumPy only**. JAX appears solely in the pixi
environment matrix so later stages can be benchmarked across JAX versions
without forcing the dependency on users of the taxonomy.

The documentation environment (`pixi run -e docs docs`) does not install JAX at
all, and the site still builds — a concrete measure of how much of the platform
is currently NumPy-level structure.

## Where the seams are

Three decisions in the current code are load-bearing for the layers that follow:

| Decision | Consequence |
|---|---|
| `parent_row` on every stratified map | Initial-population splitting and coarse/fine aggregation are gathers, not joins |
| Frozen, contiguous `codes` | The table can be handed to a JAX kernel without a copy |
| Value-compared frozen selectors | The per-map query cache is keyed by selector value, so equal queries built independently share work |
| `PropertyMap.__eq__` without `__hash__` | Maps **cannot** yet be pytree aux data; this is the first item on the promotion list in {doc}`explorations` |
