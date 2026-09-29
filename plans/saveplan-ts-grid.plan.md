---
name: saveplan-ts-grid
overview: Restore SavePlan.ts as the save grid after step 29 grouped requests on SolveSpec.default_ts().
todos:
  - id: fix
    content: SolveSpec.save_ts(plan); use it in Euler.solve, Diffrax.solve and CompiledModel.assemble_result
    status: completed
  - id: tests
    content: tests/test_saveplan_ts_grid.py — plan ts honoured by euler and tsit5; request ts still overrides; fallback to the step grid
    status: completed
  - id: notebook
    content: Section G of examples/notebooks/23-solve-backends.ipynb — a weekly plan grid saves weekly
    status: completed
isProject: false
---

# SavePlan.ts is the save grid again

## Bug

Before step 29 (`feat/solve-composable`, PR #44), `CompiledModel.run` grouped
save requests on `plan.ts` when the plan set one, else on every step. Step 29
moved grouping into the backends and `assemble_result`, which all call
`group_requests(plan, spec.default_ts())`. `SolveSpec` does not know the plan, so
`SavePlan(ts=...)` was silently dropped: an Euler run with `dt=0.1` over 40 days
saved 401 rows instead of the 81 requested. Requests with their own `ts` were
unaffected. `docs/summer2/12-concurrent-diseases.ipynb` builds a frame on the
plan grid and fails under `pixi run -e docs docs-strict`.

## Fix

`SolveSpec.save_ts(plan)` returns `plan.ts` when set, else `default_ts()`. The
three grouping sites call it, so every backend (including third-party ones that
follow the built-ins) and `assemble_result` agree on the grid.

Composability: the rule is a public method on the public `SolveSpec`, so a
custom `SolverBackend` gets the same grid by calling it rather than
re-deriving it.
