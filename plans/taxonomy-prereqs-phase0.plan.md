---
name: taxonomy-prereqs-phase0
overview: "Phase 0 of flows/derived-outputs — taxonomy prerequisites on already-public API: hashable PropertyMap, Groups from group_by, from_properties/to_frame/__len__, identifier property names, Source/Dest selector stubs."
todos:
  - id: propertymap-hash
    content: "_digest blake2b-16 + __hash__; jit-cache and Hypothesis hash==eq tests"
    status: completed
  - id: public-accessors
    content: "column_index, column, kleene, label public API"
    status: completed
  - id: groups-api
    content: "group_by returns Groups[V] Mapping; update docs/tests call sites"
    status: completed
  - id: ergonomics
    content: "__len__, from_properties, to_frame (lazy polars)"
    status: completed
  - id: property-names
    content: "Property name must be isidentifier() (blocks @ mangling collisions)"
    status: completed
  - id: source-dest
    content: "Source/Dest on Selector union; raise on compartment maps"
    status: completed
  - id: notebook-checks
    content: "examples/notebooks/02-taxonomy-ergonomics.ipynb + required pixi checks"
    status: completed
isProject: true
---

# Phase 0 — Taxonomy prerequisites (`feat/taxonomy-prereqs`)

Parent design: [flows-derived-outputs.plan.md](../../plans/flows-derived-outputs.plan.md).

Three changes to **already-public** taxonomy code that everything else needs.
Separated so the public-API break is reviewed on its own.

## `src/summer4/propertymap.py`

- `__hash__` via `_digest: bytes` (blake2b-16 over `codes.tobytes()` plus
  `parent_row`); `hash((properties, history, _digest))`.
- Public accessors: `column_index`, `column`, `kleene`, `label`.
- `group_by` returns `Groups[V](Mapping[tuple[Trait, ...], V])`; `partition`
  stays `dict[Trait, V]` (includes empty groups). **Breaks**
  `for traits, idx in pmap.group_by(age)` — use `.items()`.
- Papercuts: `__len__`, `from_properties([...])`, `to_frame()` (lazy polars).

## `src/summer4/properties.py`

Reject `@` in a property name (`isidentifier()`), so Phase 1 `@source` /
`@dest` mangling cannot collide with a user property.

## `src/summer4/selectors.py`

Add `Source` and `Dest` to the `Selector` union now, with raising arms on
compartment maps (`"Source()/Dest() select flow edges, not compartments"`).

## Acceptance

- **Ledger:** no API rows move (stays 6/52). Closes `gaps.md` 1.6 and four
  Priority-4 rows.
- **Notebook:** `examples/notebooks/02-taxonomy-ergonomics.ipynb`
- **Gates:** property-based `hash(a) == hash(b)` whenever `a == b`; jit-cache
  hit for a rebuilt-but-equal map.

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch && pixi run coverage
```
