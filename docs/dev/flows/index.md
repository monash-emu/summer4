# The flows spike

```{admonition} Not the summer4 API
:class: warning

`explorations/flows/` is a prototype. It imports `summer4` but is not part of
it, is not installed with the package, and carries no stability promise.
Published here because it is the largest body of design work the project has
done, and because it runs.
```

Two notebooks, executed on every documentation build, covering the
`explore-flows` and `refine-flows` spikes end to end.

```{toctree}
:maxdepth: 2

01-flow-types
02-derived-rates-and-adjustments
```

## What the spike covers

| Concept | Type | Spike |
|---|---|---|
| Query join — pairing source and destination across leftover strata | `identity_join`, `EdgeArrays` | I |
| Movement between compartments | `TransitionFlow` | I |
| Leaving / entering the system | `ExitFlow`, `EntryFlow` | I |
| A named bag of flows compiled to `vf(t, y, params)` | `FlowModel` | I |
| Ageing as one flow over a chain of trait pairs | `TraitChain` | I |
| Migration over a sparse destination × source matrix | `TraitMatrix` | I |
| Destination fan-out weights | `split=`, default `1/N` | I |
| Lazy rates over a parameter schema | `derived_refs`, `FieldRef` | I |
| One flow's output as another's rate | `FlowRef`, `.sum()`, `.sum_over()` | I |
| JAX backend and pytree state | `backend="jax"`, `PropertyData` | I |
| Nested parameter bundles | `derived_refs` recursion | II |
| Static topology, time-varying matrix rates | `TraitMatrix` mask + 2-D rate | II |
| Sequential rate adjustment | `adjust=`, `Multiply`, `Overwrite`, `Transform` | II |
| Stratum-scoped adjustment | `where=` on an adjustment | II |
| Time stepping | `euler`, NumPy loop and `lax.scan` | II |

## The load-bearing rules

**A flow is one object, not one object per edge.** `TransitionFlow` stores
selectors; `actualize` resolves them against a `PropertyMap` once, producing
index arrays. summer2 expanded flows into a Python object per compartment pair.

**Bound and free properties.** Properties named in either selector are *bound*;
the rest, present on both sides, are *free* and matched pairwise. That single
rule gives infection across leftover age and location, many-to-one recovery
across a ragged severity axis, and per-cell migration, with no special cases.

**Pairings are overrides, not subclasses.** `TraitChain` and `TraitMatrix`
replace the identity pairing on one property and leave the free-property
matching intact. Ageing is one named flow, not one per band.

**Rates are a lazy expression tree.** `FieldRef` paths are built from a
`NamedTuple` schema, so an IDE completes real field names; `FlowRef` lets a
flow's rate depend on another flow's computed mass, resolved in topological
order with cycle detection.

**`adjust=` is a pipeline, not sugar for `*`.** Each step sees the previous
per-edge rate, runs after pairing `scale` and before `* y[src]`, and may be
masked to a stratum with `where=`. Writing `D.foi * 0.5` bakes a multiply into
the definition; `adjust=` is the ordered, previous-valued layer.

## Keeping these pages honest

The code cells here are byte-identical to `explorations/flows/01-flows.ipynb`
and `02-flows.ipynb`. Only the title cell differs, plus one Plotly renderer
setup cell in spike II. `tests/test_flows_docs_sync.py` compares them and fails
if they drift, so the published walkthrough cannot fall behind the tested spike.

```bash
pixi run explore-flows   # run the spike's tests and its own notebooks
pixi run -e docs docs    # execute these pages
```
