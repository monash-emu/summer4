# BayesianModel saves every outputs leaf on every `dt` step

## Seen on

`perf/gradient-performance`, profiling the Kiribati TB port's calibration gradient
(`docs/gradient-performance.md` on the port's `feat/gradient-performance`).

## What is wrong

`BayesianModel._build_save_plan` (`src/summer4/epi/calibration/model.py`) starts from
`SavePlan()` with `ts=None`, so `SolveSpec.save_ts` falls back to every step of the window:
186 yearly saves for 1850–2035. The Kiribati targets read 15 years (2011, 2024, 2025 and the
notification years), plus the year before each for summer2's flow-output midpoint. Every
save is an `observe` call inside the solve and a cotangent in the reverse pass.

There is no public way to say which times a calibration needs: `run_kwargs` may not carry
`save=`, `TargetSet.plan` cannot narrow `ts` without also narrowing `posterior_runs` (which
builds its plan from `targets.plan(SavePlan())` too, and needs every year), and `Target.times`
does not know that a derived output (a midpoint, a cumulative) reads other save times.

## Measured gain (small; not adopted in the port)

Overriding the private `_build_save_plan` to save only the target years and the year before
each (the port's `scripts/bench/levers.py`, variant `sparse_saves`) gives a bit-identical log
density and gradient. The saves cost about 7% of one log-density evaluation (solve with 186
saves 0.124 s against one save 0.115 s, `profile_original_run0.json`), but no gradient speedup
was measurable on a loaded machine (0.92x–1.12x across three runs, `levers_*.json`), so the
port does not carry the private override. This is an API gap more than a performance one.

## Done when

- `BayesianModel(..., save_ts=...)` (or a `TargetSet`/`OutputSet` method that returns the
  times its evaluation reads, including the previous save for `midpoint` and the window for
  `cumulative`) sets the calibration save grid without touching `posterior_runs`.
- A test checks the log density is unchanged and the solve saves only those times.
