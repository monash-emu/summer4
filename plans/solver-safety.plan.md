---
name: solver-safety
overview: Surface adaptive solver failure via SolverInfo.ok and make MixingMatrix reciprocity vmap-safe.
todos:
  - id: solver-ok
    content: SolverInfo.ok traced bool; throw= on run; default max_steps from span/dt with headroom
    status: completed
  - id: reciprocity
    content: MixingMatrix check_reciprocal default False; batch-safe callback; host validate()
    status: completed
  - id: gate
    content: Extend 05-solvers.ipynb with max_steps=16 partial trajectory vs healthy ok; close KI22
    status: completed
isProject: false
---

# Solver safety (WP16 §16b / §16c)

Follows `plans/tb-ports-feature-completeness.plan.md` §16b and §16c. Roadmap
step 12 on `feat/solver-safety`. Closes `KI22`.

## Solver failure must be visible

diffrax previously always used `throw=False` and `max_steps=4096`. A long run
that hits the ceiling returned silently wrong trajectories.

- `SolverInfo.ok` — traced bool, `result_code == 0` (Euler sets `result_code=0`).
- `CompiledModel.run(..., throw=None)` — `None` → `False` (failure via `ok`);
  a bool passes through to diffrax.
- When `max_steps` is `None`, derive
  `max(4096, ceil((t1 - t0) / dt) * 64)` instead of a fixed 4096.

WP10 / step 14 uses `ok` to return `-inf` log-likelihood inside `jit`.

## vmap-safe reciprocity check

`MixingMatrix.check_reciprocity` called `float(d)` on a debug-callback residual,
which fails under `vmap`.

- Constructor default `check_reciprocal=False`.
- Callback reduces with `np.max` so batched residuals are safe.
- Host-side `MixingMatrix.validate(matrix, population)` for eager checks.

## Notebook

Extend `examples/notebooks/05-solvers.ipynb` with a deliberately too-small
`max_steps` (enough for some accepted steps, not enough to finish) and assert
`SolverInfo.ok` plus partial finite saves.
