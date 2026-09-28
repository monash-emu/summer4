# CP1 — the solve takes the caller's diffrax objects

Closes composability findings `CX1`–`CX5` (`docs/evaluation/composability.md`,
work package `CP1`). Roadmap step 29, branch `feat/solve-composable`.

## Problem

`CompiledModel.run` accepts a diffrax solver instance but builds everything
around it: the stepsize controller is `PIDController(rtol, atol)` or
`ConstantStepSize`, no `adjoint=` / `event=` / `progress_meter=` reaches
`diffeqsolve`, summer4's own Euler stepper is reachable only by the string
`"euler"`, `SolveSpec` is undocumented, `run` is one long method with no
public intermediate steps, and a `Result` has no end state to continue from.
A calibration that needs a different adjoint for its gradients, or a run that
should stop on a state event, cannot be expressed.

## Design

The seam is a **solver backend object**: a public protocol that the two
built-in backends implement and a user can implement too.

```python
from summer4.solvers import Diffrax, Euler, SolveSpec, SolverBackend

class SolverBackend(Protocol):
    name: str
    def solve(self, model, *, y0, params, spec: SolveSpec, plan: SavePlan) -> SolveOutput: ...

Euler()                                   # summer4's fixed-step stepper
Diffrax(
    diffrax.Tsit5(),                      # any diffrax.AbstractSolver
    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),  # None -> ConstantStepSize
    adjoint=diffrax.DirectAdjoint(),      # None -> diffrax's default
    event=diffrax.Event(cond_fn),         # None -> no event
    progress_meter=None,                  # None -> diffrax's default
)
```

- `SolveSpec` is public and holds only the window: `t0`, `t1`, `steps`,
  `dt`, `max_steps`, `dense`, `throw`. `SolveSpec.window(t0=, dt=, t1= | steps=,
  ...)` validates and fills `steps`; `spec.default_ts()` is the default save
  grid. The old `rtol` / `atol` fields move onto the controller.
- `resolve_solver(solver, *, rtol=None, atol=None) -> SolverBackend` is the
  public sugar: `"euler"` → `Euler()`; `"heun"` / `"tsit5"` / `"dopri5"` or a
  bare diffrax solver → `Diffrax(...)`, with `PIDController(rtol, atol)` when
  either is given; a backend object passes through. `rtol` / `atol` with
  `Euler()` or with a `Diffrax` backend raise (set the controller on the
  backend instead).
- `SolveOutput` gains `final_time` and `final_state` (the raw state array at
  the end of the integration, or where an event stopped it).
- `CompiledModel.assemble_result(plan, out, *, spec, epoch=None) -> Result` is
  public. `Result` gains `final_time` and `final_state` (a `PropertyData`),
  both pytree children, so `run(params, result.final_state, t0=result.final_time, ...)`
  continues a run.
- `SolverInfo.ok` is true for a successful solve **or** one an event
  terminated; `SolverInfo.event` says which.
- `CompiledModel.run` keeps its signature (`solver` widens to
  `str | SolverBackend | diffrax.AbstractSolver`) and becomes exactly:

  ```python
  prepared = model.prepare(params)
  y0 = model.initial_state(prepared) if y0 is None else y0
  plan = model.expand(EVERYTHING if save is None else save)
  spec = SolveSpec.window(t0=t0, t1=t1, dt=dt, steps=steps,
                          max_steps=max_steps, throw=throw, dense=plan.dense)
  backend = resolve_solver(solver, rtol=rtol, atol=atol)
  out = backend.solve(model, y0=y0, params=prepared, spec=spec, plan=plan)
  return model.assemble_result(plan, out, spec=spec, epoch=epoch)
  ```

  documented in its docstring and tested for equality.

### Final state

- **diffrax:** one extra `SubSaveAt(t1=True)` on the existing `SaveAt(subs=...)`;
  `sol.ts` / `sol.ys` of that sub give the end time and state, including the
  time an event stopped at.
- **Euler:** the lerp path already holds the whole trajectory; the fast path
  returns its last carry, and when the last save time is before `t0 + dt*steps`
  the remaining steps run in one `fori_loop`.

## Tests (`tests/test_solve_composable.py`)

- A `Diffrax(Tsit5(), stepsize_controller=PIDController(...))` run equals
  `solver="tsit5", rtol=, atol=` exactly; a different controller changes the
  step count.
- `jax.grad` of a loss through `run` agrees between the default adjoint and
  `DirectAdjoint()`.
- An `Event` stops the run early: `final_time < t1`, `solver.ok` and
  `solver.event` are true, and the final state is past the threshold.
- Resuming from `final_state` / `final_time` equals one longer run (Euler
  exactly, diffrax to tolerance), including an Euler save grid that ends
  before `t1`.
- `resolve_solver` errors, a user-written backend used by `run`, `SolveSpec.window`
  validation, and `run` equal to its documented expansion.

## Notebook

`examples/notebooks/23-solve-backends.ipynb`, an acceptance page in the style of
notebook 22 (*What to check* boxes, plots and objects, no asserts).

## Out of scope

- The per-chunk MCMC recompile (`futureplans/mcmc-chunk-recompile.md`).
- `CX6`–`CX8` (rate extension points), `CP2`–`CP4`.
