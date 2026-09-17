---
name: model-stratify-landing
description: Land FlowModel.stratify / adjustment precedence on feat/model-stratify (plan was already on the parent branch).
---

# Land model-stratify (`feat/model-stratify`)

## Why this plan exists

`plans/model-stratify.plan.md` was added on the parent branch
`feat/remove-epimodel`, so `pixi run check-branch -- --base origin/feat/remove-epimodel`
does not see a plan file in this PR's diff even though `src/summer4` changes.
AGENTS.md requires a plan under `plans/` (or `.cursor/plans/`) in the branch
diff when `src/summer4` changes. This file is that landing record.

## Scope (already implemented on this branch)

Implements `plans/model-stratify.plan.md`:

- `FlowModel.stratify` / `copy` / `update_flow` / `adjust_flow`
- Adjustment precedence (`Overwrite` → `Multiply` → `Transform`, `precedence=`)
- `Source` / `Dest` edge masks; dead `where=` raises
- Tests, example notebook `12-model-stratification.ipynb`, docs / ledger notes

## Follow-up noticed while fixing CI (`latest`)

JAX 0.11+ lowers `jnp.stack` of K scalars to **one** `stack` op (width K).
JAX 0.6 unrolled that stack into many eqns. Hoist tests that asserted
`off40 > off4` on raw in-loop primitive counts fail on `latest` even though
hoisting still pulls knot stacks out of the euler `scan`. Fixed by measuring
in-loop stack width (with primitive-count fallback) in
`tests/helpers/jaxpr.py` (`loop_knot_cost`).

```bash
pixi run check-branch -- --base origin/feat/remove-epimodel
pixi run test
pixi run --environment latest test
```
