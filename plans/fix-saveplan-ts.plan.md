---
name: fix-saveplan-ts
overview: Restore SavePlan.ts as the default save grid, lost in the CP1 solver-backend refactor (PR #44).
isProject: false
---

# Fix: a plan-level `SavePlan.ts` is ignored by `run`

## Symptom

Since `f85d718` (roadmap step 29, `CP1`, PR #44), `CompiledModel.run` saves on
every Euler step (or the `dt` grid) even when the `SavePlan` carries its own
`ts`. Before the refactor `run` used `plan.ts` as the default grid and fell back
to `t0 + dt * arange(steps + 1)` only when it was `None`. The refactor moved
that choice into `SolveSpec.default_ts()`, which never sees the plan, and the
three callers (`EulerBackend.solve`, `Diffrax.solve`,
`CompiledModel.assemble_result`) passed its result straight to
`group_requests`. A request-level `SaveRequest(ts=...)` still worked.

Found because `pixi run -e docs docs-strict` fails on `main`:
`docs/summer2/12-concurrent-diseases.ipynb` builds a frame on `TS` (81 points)
from outputs that now have 401.

## Fix

`SolveSpec.save_ts(plan)` returns `plan.ts` when set and `default_ts()`
otherwise; the three call sites use it. `tests/test_saveplan_ts.py` pins the
plan-level grid under Euler and Tsit5, that a request's own `ts` still wins,
and the fallback.
