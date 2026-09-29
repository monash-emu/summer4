# Posterior runs drop the calendar axis

## What is wrong today

`PosteriorRuns.times` (`src/summer4/epi/calibration/posterior_runs.py`) is a
plain float array taken from the probe run
(`probe[names[0]].times.values`). The `Epoch` a `Result` may carry is not
kept, so `PosteriorRuns.quantiles()` is indexed by model time, and every
`wf.plot_*` figure (`src/summer4/epi/calibration/workflow/plots.py`) labels its
x axis `"time"` in model units. `Target.times` are also floats, so the target
overlay lines up, but neither shows dates.

## Why it hurts

Kiribati-style analyses are calendar-dated (`Epoch`, `Output.to_pandas()`
already returns a date index). A modeller reading a scenario ribbon wants
"March 2026", not "day 812", and currently has to convert the x values by hand
after the fact.

## What a fix looks like

- Keep the probe result's `Epoch` (if any) on `PosteriorRuns` as `epoch`.
- `PosteriorRuns.quantiles()` returns a date index when `epoch` is set
  (mirroring `Output.to_pandas`).
- The plot functions map x through `epoch` when present (one private helper,
  `_x(runs)`), and `add_targets` does the same for target times.
- A test with an `Epoch`-carrying model: the ribbon's x values are
  `datetime64` and the target markers share them.
