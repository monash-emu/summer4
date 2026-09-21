---
name: output-sets
overview: OutputSet as a DAG over save leaves, Result.to_frame, and Target.reduce for an aggregate series.
todos:
  - id: dag
    content: OutputSet leaves, refs, cycles, and a plan that saves each spec once
    status: completed
  - id: frames
    content: Result.to_frame wide and long, pandas and polars
    status: completed
  - id: reduce
    content: Target.reduce sum, mean, or a callable before residuals
    status: completed
  - id: tests
    content: jit and grad, frames, notebook 06, KI13–KI17
    status: completed
isProject: false
---

# Named output sets

Follows `plans/tb-ports-feature-completeness.plan.md` §15c, the
`targetset-residual-reduction` half of §15d, and §15e. The value type is
`Output`, not `Trace`.

## Contract

- A leaf is a save spec. `Compartments().total()` and `FlowMass(...).midpoint()`
  record post-ops; the solver still saves the spec.
- `outputs.ref(name)` and arithmetic, including `* Param(...)` and
  `* tanh(Param(...))`, are interior nodes. Cycles raise and name the loop.
- `plan(base)` adds one request per distinct spec. The save key is the first
  output name that uses it; a second spec in that same expression is
  `name__2`. A key already in `base` is kept, times included, when the
  quantity matches, and rejected when it does not.
- The solved `Result` holds the raw leaf under that key. `evaluate` returns
  the named outputs, post-ops included. Read those, not the solve result, when
  the name has a chain.
- `Result.to_frame(shape="wide"|"long", backend="polars"|"pandas")` is
  host-side. Wide needs one time axis. A polars frame writes parquet.
- `Target(reduce="sum"|"mean"|callable)` reduces every axis after the leading
  time axis before the residual is formed.

`cumulative(start=)` still needs a concrete time axis, including when
`evaluate` runs under `jit`. The DAG itself unrolls once per named output.
