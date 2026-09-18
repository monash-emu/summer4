# Explorations (historical)

The flows layer used to live as a prototype under `explorations/flows/`,
outside the package. That spike is **retired**. Joins, flows, lazy rates,
adjustments, `EdgeMap` and a compiled JAX vector field are importable from
{mod}`summer4` — see {doc}`../user/08-flows` and {mod}`summer4.flows`.

There is no `explorations/` tree in this repository. Historical design notes
live in committed plans (`explore-flows.plan.md`, `refine-flows.plan.md`,
`flows-derived-outputs.plan.md`).

## Settled design (now the public API)

**One named flow plus a query join is enough.** Properties bound by a flow are
those named by a `Trait` or `IsIn` in either selector (`selector_values`).
`Present` and `Absent` name a property but do **not** bind it.
`strict_pairing=True` raises when an unbound property would move people.
Leftover properties present on both sides are matched pairwise.

**Destination split belongs on the flow, not the stratification.** Default
`1/N` over destinations sharing a free key, with an optional explicit `split=`
mapping. Initial-population split remains a separate API.

**Ageing and migration are pairing overrides, not flow subclasses.**
`TraitChain` and `TraitMatrix` replace the identity pairing on one property.

**Rate laziness is a small expression tree.** `derived_refs` over a
`NamedTuple` produce `FieldRef`s; `add_flow` returns a `FlowRef`; topological
evaluation catches cycles.

**A `lax.scan` Euler step proves the seam.** `compile()` returns a
`CompiledModel`; `CompiledModel.run` returns a `Result`. Diffrax sits behind
the same `solver=` seam.

## What is still open

- **Mass-level limiting.** `min(mass, y[src])` after multiplying by the source
  compartment is not the same as a rate `Transform`.
- **A `__rate__` protocol** on derived bundles, versus always writing
  `baseline * seasonal` explicitly.
