# Dense PropertyMap unstack (flat ↔ product axes)

## Concern

`PropertyData` keeps a single last axis aligned to `PropertyMap` rows. When the
map is a full Cartesian product (no partial `stratify(..., where=)`, no `-1`
codes, `size == ∏ n_traits`), that flat vector is a perfect tile of
`(age × state × loc × …)` in stratification order (newest property varies
fastest). There is no helper to prove density and reshape for inspection,
plotting, or hand linear algebra — callers must know the layout and
`reshape` by hand.

This is fine for solvers and the Output query surface (flat last-axis is the
invariant). It is awkward when a notebook or FOI derivation wants the cube
view without committing to first-class multi-axis `PropertyData`.

## Pointers

- `PropertyMap.from_properties` / unconstrained `stratify` — dense product
  layout (`src/summer4/propertymap.py`)
- Ragged counterexample: `stratify(prop, where=…)` (`tests/test_propertymap.py`)
- `PropertyData.sum_over` / `broadcast_over` — 1-property gather/scatter on
  the flat axis, not ND reshape (`src/summer4/jax/propertydata.py`)
- Explicitly out of scope: making `(…, age, state, loc)` a native
  `PropertyData` / `Output.dims` layout (would break flows, saves, and edge maps)

## Done looks like

Opt-in, fail-closed helpers only:

1. Host-side `PropertyMap` predicate: dense product for an ordered property
   list (codes match expected product order, or equal a static permutation).
2. `PropertyData.unstack(*props) -> array` with shape
   `data.shape[:-1] + (n0, n1, …)`; inverse `stack` flattens back.
3. Raise on ragged maps, `take`/`select` subsets, edge tables, or wrong prop
   order.

No change to solvers, `SavePlan`, or Output query semantics. Promote to a real
`plans/` entry only if TB ports or textbook work needs the cube view often
enough to justify the API.
