# Flows spike findings

Drafted on the `explore-flows` branch, 2026-09-09. The spike lives in
`explorations/flows/` and does not change `src/summer4`.

## What worked

- **One named flow + query join** is enough for S→I across leftover age and
  location. `select` / packed leftover codes are the only join inputs.
- **Bound = properties mentioned in either selector.** Leftover properties
  present on both selected sets are matched. That is the summer3 /
  summer4-A rule, expressed over Kleene selectors instead of query dicts.
- **Dest-split belongs on the flow.** Default `1/N` over dests that share a
  free key; optional `split={severity: {"mild": 0.8, "severe": 0.2}}`.
  Product then renormalise handles two dest-only properties and ragged
  holes. This replaces summer2 `Stratification.set_population_split` for
  flow fan-out only.
- **TraitChain and TraitMatrix** are pairing overrides, not flow subclasses.
  Ageing is **one** named flow: the chain lists every band step (the top band
  is simply omitted) and optional per-pair ``rates`` hold ``1/width``. Several
  named ageing flows are unnecessary. Migration is a sparse dest×source
  matrix; leftover properties still match.
- **Rate laziness is a small expression tree**, not a PropertyData op graph
  and not computegraph. `derived_refs(NamedTuple)` builds `FieldRef`s named
  after the schema (IDE completion on real fields). `add_flow` returns a
  `FlowRef` used in later rates. `death.sum()` is a scalar; `death.sum_over(location)`
  reduces contribution onto that property's traits and aligns by dest codes.
  Topological evaluation of `FlowRef` edges is cheap and catches cycles.
  Open-ended singleton proxies (`DerivedParamStructProxy`, `Flows`) were
  retired.
- **NumPy and JAX backends share actualized index arrays.** jit over a
  `PropertyData` state works with the digest static wrapper copied from
  explore-datatypes.

## What to promote on `feat/flows`

1. Make `PropertyMap` hashable (digest of `codes` / `parent_row` plus
   `properties` / `history`) so it can be pytree aux without a wrapper.
2. Land `PropertyData` in a JAX module; keep the taxonomy NumPy-only.
3. Add `PropertyMap.join(source, dest, *, extra_bound=..., split=...)` as a
   real method. The packed-key implementation here is the starting point.
4. Public `TransitionFlow` / `ExitFlow` / `EntryFlow` with flow-owned
   `split`. Do **not** reattach population split to `Stratification`.
5. Initial-population split is a separate later API (a one-shot
   redistribution of `y`, not a flow property).
6. Keep rate proxies as this expression tree until a compile step needs a
   real graph (shared subexpressions, recorded computed values). Do not
   introduce computegraph for FOI + replacement births alone.
7. Leave 1/N as the implicit dest-split default. Explicit `split` is the
   escape hatch; an `allow_split` flag is unnecessary if weights always
   exist on the edge array.
8. Public derived-param refs should be schema-built (`derived_refs` or
   equivalent). Validate `FieldRef` paths at compile time against the
   NamedTuple / dataclass. Do not revive a module-level `__getattr__` proxy.
9. `add_flow` should return a `FlowRef`. Partial reduces (`sum_over`) are
   reduced `PropertyData` (a property-only map), not a separate CategoryData
   type.

## Open questions

- Syntactic bound names treat `age.present()` as binding `age`, so it drops
  out of the free set. That is correct for a chain/matrix override and
  surprising for a “match leftover age” reading of `present()`. A later API
  may want `Present` / `Absent` to be non-binding.
- `Present` / `Absent` binding is the main selector-design leftover. Ageing
  does not need multiple flow names: `TraitChain.rates` (or a source-gathered
  PropertyData) covers unequal band widths on one flow.
- `FieldRef` lookup on a NamedTuple works under `jax.jit`. A public API
  should still validate paths at compile time against that schema.

## Refine-flows follow-up (2026-09-09)

Drafted on `refine-flows`. Still no `src/summer4` changes.

### What worked

- **Nested `derived_refs`.** A NamedTuple field whose annotation is another
  NamedTuple becomes a nested ref tree (`D.migration.baseline`). Non-NamedTuple
  annotations stay leaf `FieldRef`s. Passing “all this” as one argument is
  either the already-combined `D.migration_rates` matrix or ordinary arithmetic
  on the nested proxy (`D.migration.baseline * D.migration.seasonal`). Auto-
  multiplying a looked-up bundle was not needed; a later `__rate__` protocol
  is optional sugar, not required for compactness.
- **Static topology + dynamic dest×source rates.** `TraitMatrix` still
  actualizes edges once. A 2-D `(n_traits, n_traits)` rate gathers
  `rate[dest_code, src_code]` per edge. Time-varying migration is
  `baseline * seasonal(t)` in `compute_derived_params`. Changing sparsity
  after compile would break `jit`.
- **`adjust=` pipeline.** Sequential Multiply (default for a bare value),
  Overwrite, and Transform on the aligned rate (after pairing `scale`, before
  `* y[src]`). Optional `where=Selector` is a static mask on gather indices.
  `Transform(fn, *args)` takes the previous numeric rate plus any `RateOps`
  (including `FlowRef`s; topo walks `adjust`). `*` on the rate expression
  remains the baked-in multiply; `adjust=` is the ordered / masked / prev-
  valued layer.
- **Minimal Euler.** NumPy loop and JAX `lax.scan` share one `euler(...)`.
  `jax.jit` around the scan stepper matches NumPy over 8 steps. A time-varying
  derived rate changes the final `y` versus `amp=0`. Replacement births keep
  `sum(y)` constant. `Transform` callables on the jitted path must be
  JAX-traceable (`jnp.minimum`, not Python branching over edges).

### What to promote later

10. Recurse public `derived_refs` into nested schemas. Do not auto-unpack a
    bundle into a rate.
11. Align dest×source matrices onto `TraitMatrix` edges; keep topology static.
12. Flow-owned `adjust=` (Multiply / Overwrite / Transform) with selector
    `where`. Keep it off `Stratification`.
13. A tiny `lax.scan` Euler is enough to prove jit of the compiled field.
    Diffrax can wrap the same `vf` later.

### Still open

- Mass-level limiting (`min(mass, y[src])`) is not the same as a rate
  Transform. It would run after `* y[src]`.
- A `__rate__` protocol on derived bundles vs always writing `baseline * seasonal`.
- `Present` / `Absent` binding (unchanged from the first spike).
