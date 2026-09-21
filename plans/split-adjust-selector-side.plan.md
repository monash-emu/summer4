# Bare `adjust=` traits on a `split=` property

Accepted fix for `futureplans/split-adjust-selector-side.md`. The note is
deleted when this lands.

## Decision

`_bind_adjust_masks` resolves a polarity-free `where=` one subtree at a time.
A property that **this flow's** `split=` introduces is read on the destination.
Every other bare trait keeps the flow's default side (`Source` on a transition,
`Dest` on an entry). An explicit `Source(...)` or `Dest(...)` is left alone.

`age["young"] & clinical[c]` therefore becomes `Source(age["young"]) &
Dest(clinical[c])` when `clinical` is the split key, including when pairing
moves `age` from source to a different destination trait.

A dest-only property that is **not** a split key still binds to the source and
raises, as today. The error may still suggest `split=`.

## Done when

- The summer2 page `docs/summer2/10-derived-outputs-stratified.ipynb` executes.
- `pixi run -e docs docs-strict` passes.
- A regression test shows the bare trait matches `Dest(...)`, and a pairing
  test shows the non-split property stays on the source.
