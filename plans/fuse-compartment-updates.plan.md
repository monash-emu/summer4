---
name: fuse-compartment-updates
description: Fuse per-flow dy scatters into one sparse scatter-add; keep a compile switch for A/B. Source gathers stay per-flow.
---

# Fuse compartment updates

## Problem

`CompiledModel.observe` did one (or two) `dy.at[idx].add(±mass)` per named
flow. On TB-macro (~30 flows) profiling attributed ~25% of named XLA time to
`wrapped_scatter`. Each op is tiny; the cost is op count × Dopri5 stages.

## Approach

1. Keep the per-flow rate / mass loop (needed for `FlowRef` order).
2. When `fuse_compartment_updates=True` (default): one `scatter-add` of all
   signed masses into `dy`.
3. Keep `fuse_compartment_updates=False` as the previous per-flow path for
   parity tests and A/B benchmarks.

**Not shipped:** concatenating relative `src` gathers then slicing. That cut
forward gather count but added `slice`→`pad` under `grad` (~15% slower AD on
TB-scale). Details in `futureplans/vf-gather-and-mul-followups.md`.

## Exit

- Numerical parity tests (entry / exit / transition, FlowRef, TB-scale spot).
- Bench prints VF `scatter-add` / `gather` counts and warm dopri5
  solve / grad / value_and_grad for both modes (synced, interleaved median).
- No new user-facing module; `compile(fuse_compartment_updates=...)` mirrors
  `hoist=`.
