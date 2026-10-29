# posterior_runs assumes every named output is one time series

**Status:** open. Not scheduled. Noticed on `feat/time-free-outputs`.  
**Where:** `_run_scenario_draws` and `_series_1d` in
`src/summer4/epi/calibration/posterior_runs.py`.

## Problem

`posterior_runs` stacks every named output of the `OutputSet` as one `(time,)`
column per draw (`jnp.stack(cols)` after `_series_1d`). Two kinds of named output
do not fit:

- a time-free output (`ref.integrate()`, `ref.at(t)`), now useful because it
  broadcasts against series (`plans/time-free-outputs.plan.md`);
- an array-valued output (`ref.array()`, `ref.scalar()`), which lands in
  `Result.arrays`, not `Result.outputs`.

Either one in `BayesianModel(outputs=)` makes the stack fail or misalign. The
workaround is a separate `Scenario(outputs=)` that names only series.

## Done looks like

`posterior_runs` keeps series in the existing `(draw, output, time)` block and
returns time-free and array outputs in a second mapping of `(draw, ...)` arrays,
with a test of a whole-period total's posterior.
