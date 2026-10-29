<!--
Accepted 2026-10-29 (fast-tracked into 0.2.0a7 for the summer4template release). Historical
record once merged: do not edit. Follow-up work goes in a new plan.
-->

# Calibration targets on transformed outputs

## Context

`BayesianModel(..., outputs=OutputSet)` evaluates the outputs before scoring, and
`TargetSet.log_likelihood` reads any key of the evaluated result. So a target
named after an output — `outputs["weekly"] = FlowMass("infection").total().rolling(7, how="sum")`
with `Target(key="weekly", ...)` — should be scorable. It is not, because of how
`BayesianModel._build_save_plan` (`src/summer4/epi/calibration/model.py`) builds the
solver's save plan: `targets.plan(...)` first, then `outputs.plan(...)`.

1. A target keyed to an output name with no `quantity` raises `KeyError` (the
   key is not in the plan yet). This is what the summer4template port hit, so
   the template could not use summer3's weekly rolling-sum target.
2. Giving the target the output's leaf quantity, or swapping the order, is
   worse. An `OutputSet` saves each leaf under the output's own name, so the
   target's `contribute` pins that leaf to the observation times. A time-window
   output (`rolling`, `cumulative`, `integrate_intervals`, `resample`) is then
   computed over the sparse observation points instead of the run's grid, with
   no error.

`posterior_runs` already avoids this (`posterior_runs.py::_build_save_plan`
plans the `OutputSet` alone, "targets.plan would pin shared leaves to
observation times").

## Shape

`BayesianModel._build_save_plan`:

- With `outputs`, plan the `OutputSet` first, on the run's own save grid.
- A target whose key is an output name adds **no** save request. It is scored
  from the evaluated output, read at its times by `Output.at_times` (linear
  interpolation between saved points; exact for times on the grid). Its
  `quantity`, if given, is not used for saving.
- Any other target plans as before. If its key collides with a leaf key the
  `OutputSet` uses (`name__2` and similar), raise a `ValueError` naming the key
  rather than narrowing that leaf.

Plain-leaf outputs targeted by name (notebook 18, textbook chapter 20) now save
on the run's grid rather than only at the observation times. Their scores are
unchanged where the observation times lie on that grid, which they do.

`rolling(n)` counts saved points, not days. It is a 7-day window only when the
grid is daily. The `BayesianModel` docstring says so.

## Files

- `src/summer4/epi/calibration/model.py`
- `tests/test_bayesian_output_targets.py` — new
- `examples/notebooks/31-calibration-output-targets.ipynb` — acceptance notebook
- `CHANGELOG.md`

## Tests

- A rolling-sum output target: the log-likelihood equals a hand computation
  from a dense run, and the save plan has no observation-time narrowing.
- The pre-fix narrowing is detectable: scoring the same target from a plan pinned
  to observation times gives a different (wrong) value — characterises the bug.
- The log-likelihood is higher at the true parameters than nearby.
- A target with no `quantity` keyed to an output no longer raises.
- A plain-leaf output target scores the same as the equivalent quantity target
  without outputs (observation times on the grid).
- A non-output target whose key collides with an output leaf key raises.
- `posterior_runs` and `sample` run on a model with an output target.

## Acceptance notebook

`31-calibration-output-targets.ipynb`, the summer4template model (age-stratified
SIR, homogeneous mixing), target = 7-day rolling sum of infections, as summer3:

1. The target and the model's weekly output on the full run.
2. What the old pinned plan would have computed against the correct output.
3. The posterior from `bm.sample(..., progress_bar=True)` against the true
   parameters.
