# Explorations: the flows spike

`explorations/flows/` holds a 1 500-line prototype of the layer that comes after
the taxonomy: joins, flows, lazy rates, adjustments and a minimal integrator. It
imports `summer4` but nothing in `summer4` imports it, and it is not installed
as part of the package.

```{admonition} Not an API
:class: warning

Nothing in `explorations/` is public, stable, or documented as user-facing.
It exists so that design questions can be settled with working code before a
public API is frozen. `pixi run explore-flows` runs its tests
(`tests/test_explore_flows.py`, 620 lines) and its two notebooks, which are
published — executed on every documentation build — under {doc}`flows/index`.
```

## What the spike contains

| Module | Contents |
|---|---|
| `prototype.py` | Selector introspection, query join, `TransitionFlow` / `ExitFlow` / `EntryFlow`, `TraitChain`, `TraitMatrix`, rate expression tree, `Multiply` / `Overwrite` / `Transform`, Euler stepper |
| `propertydata.py` | Array container keyed by a `PropertyMap`, NumPy and JAX backends |
| `01-flows.ipynb`, `02-flows.ipynb` | The two spike walkthroughs, published as {doc}`flows/01-flow-types` and {doc}`flows/02-derived-rates-and-adjustments` |
| `FINDINGS.md` | The written conclusions, reproduced in summary below |

## Settled design questions

These are the load-bearing conclusions from `FINDINGS.md`, and they are the
reason the taxonomy looks the way it does.

**One named flow plus a query join is enough.** An S→I flow across leftover age
and location strata needs only `select` and packed leftover codes as join
inputs. The properties *bound* by a flow are those mentioned in either
selector; leftover properties present on both sides are matched pairwise. That
is the same rule as summer3 and summer4-A, expressed over Kleene selectors
rather than query dictionaries.

**Destination split belongs on the flow, not the stratification.** Default
`1/N` over destinations sharing a free key, with an optional explicit `split=`
mapping, product-then-renormalise for two destination-only properties. This
deliberately replaces summer2's `Stratification.set_population_split` for flow
fan-out. Initial-population split remains a separate, later API — a one-shot
redistribution of the state vector, not a flow property.

**Ageing and migration are pairing overrides, not flow subclasses.**
`TraitChain` expresses ageing as one named flow listing every band step, with
optional per-pair rates carrying `1/width`. `TraitMatrix` expresses migration as
a sparse destination × source matrix. Leftover properties still match in both.
Several named ageing flows are unnecessary.

**Rate laziness is a small expression tree, not a compute graph.** Schema-built
`derived_refs` over a `NamedTuple` produce `FieldRef`s named after real fields,
so an IDE completes them; `add_flow` returns a `FlowRef` usable in later rates;
topological evaluation catches cycles. Nested schemas recurse into a ref tree
(`D.migration.baseline`). The conclusion was explicit: do **not** introduce
computegraph for force of infection plus replacement births alone.

**A `lax.scan` Euler step is enough to prove the seam.** A NumPy loop and a JAX
scan share one `euler(...)`; `jax.jit` around the scan matches NumPy over eight
steps; time-varying derived rates change the result and replacement births keep
`sum(y)` constant. Diffrax can wrap the same vector field later.

## The promotion list

`FINDINGS.md` names thirteen items to promote onto a `feat/flows` branch. The
first is a prerequisite that touches the already-public taxonomy:

1. **Make `PropertyMap` hashable** — digest of `codes` / `parent_row` plus
   `properties` / `history` — so it can be pytree aux data without a wrapper.
   Today `PropertyMap` defines `__eq__` with `eq=False` on the dataclass and no
   `__hash__`, so maps are unhashable.
2. Land `PropertyData` in a JAX module, keeping the taxonomy NumPy-only.
3. Add `PropertyMap.join(source, dest, *, extra_bound=..., split=...)`.
4. Public `TransitionFlow` / `ExitFlow` / `EntryFlow` with flow-owned `split`.
5. Initial-population split as a separate later API.
6. Keep rate proxies as an expression tree until a compile step needs a real
   graph.
7. Leave `1/N` as the implicit destination-split default.
8. Schema-built derived-parameter refs, validated at compile time.
9. `add_flow` returns a `FlowRef`; partial reduces are reduced `PropertyData`.
10. Recurse `derived_refs` into nested schemas; do not auto-unpack a bundle.
11. Align destination × source matrices onto `TraitMatrix` edges; topology stays
    static.
12. Flow-owned `adjust=` (`Multiply` / `Overwrite` / `Transform`) with a
    selector `where`; keep it off `Stratification`.
13. A tiny `lax.scan` Euler is enough to prove `jit` of the compiled field.

## Open questions

- **`Present` / `Absent` binding.** Syntactic bound-name extraction treats
  `age.present()` as binding `age`, which drops it out of the free set. That is
  right for a chain or matrix override and surprising for a "match leftover age"
  reading. A later API may want `Present` and `Absent` to be non-binding. This
  is the main unresolved selector-design question and it affects the *public*
  taxonomy, not just the spike.
- **Mass-level limiting.** `min(mass, y[src])` runs after multiplication by the
  source compartment and is therefore not the same thing as a rate `Transform`.
- **A `__rate__` protocol** on derived bundles, versus always writing
  `baseline * seasonal` explicitly.

## Why this is not in `src/summer4`

The repository's feature bar (see {doc}`contributing`) requires that a branch
adding public API also add tests and a runnable example notebook. The spike
satisfies the test and notebook requirements but has not settled the questions
above, and promoting it would freeze `PropertyMap.join`, flow classes and the
rate expression tree before the `Present` / `Absent` binding question is
answered. Keeping it outside the package is the deliberate cost of not shipping
an API that would then need to be broken.
