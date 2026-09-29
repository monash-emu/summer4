---
name: calib-outputs
overview: Post-calibration figures — spaghetti, ribbons, scenario differences and stage diagnostics — WP19 / roadmap step 28.
todos:
  - id: api
    content: wf.plot_spaghetti / plot_ribbons / plot_scenarios / plot_design / plot_optimisation / plot_chains / add_targets; PosteriorRuns.difference
    status: completed
  - id: tests
    content: tests/test_calib_workflow_plots.py — trace counts and names, target markers, difference signs, every stage's points, lazy plotly
    status: completed
  - id: notebook
    content: examples/notebooks/25-calibration-workflow.ipynb (acceptance page)
    status: completed
  - id: ledger
    content: Close CW10–CW11 (11 of 11)
    status: completed
isProject: false
---

# Calibration outputs (WP19 / step 28)

Follows `plans/calibration-toolkit.plan.md` *Step 28*, as corrected by the
roadmap's step 28 section. Closes `CW10`–`CW11`. Stacked on
`feat/calib-composable` (step 30, PR #45), which was not yet merged when this
step started.

## What ships

`summer4.epi.calibration.workflow.plots`, re-exported on `wf`. Every function
returns a `plotly.graph_objects.Figure` and takes `fig=` to draw onto an
existing figure, so layers compose (a posterior ribbon, optimised fits as
spaghetti, and the data on one canvas) instead of each function growing
options. Plotly is imported lazily and added to the `calibration` extra.

| Function | Input | Draws |
| --- | --- | --- |
| `plot_spaghetti(runs, output, scenario=, n=, targets=, fig=, name=, color=)` | `PosteriorRuns` | one trace, one line per draw (gaps between draws) |
| `plot_ribbons(runs, output, q=, scenarios=, targets=, fig=)` | `PosteriorRuns` | a filled band per quantile pair plus the median, per scenario |
| `plot_scenarios(runs, output, ref=, scenarios=, q=, relative=, fig=)` | `PosteriorRuns` | ribbons of the paired per-draw difference from `ref` |
| `add_targets(fig, targets, output=)` | `Target`, sequence, or `TargetSet` (matched by key) | black markers |
| `plot_design(candidates, fig=)` | evaluated `Candidates` | log density against each parameter |
| `plot_optimisation(run)` | `OptimizeRun` | best loss per start |
| `plot_chains(source)` | `MCMCRun`, numpyro `MCMC`, or a `(chain, draw)` mapping | per-chain traces, R-hat / ESS per panel, chunk boundaries |

The data behind each figure is public: `PosteriorRuns.samples`, the new
`PosteriorRuns.difference(scenario, ref, output, relative=)` (which
`differences()` now uses), `OptimizeRun.loss_trace`, `MCMCRun.samples`.

## Deviations from the toolkit plan

- `plot_chains` takes an `MCMCRun` (there is no `SampleResult`), and also a
  numpyro `MCMC` or a plain mapping (P1).
- `plot_optimisation` takes an `OptimizeRun` (`OptimizeResult` is gone).
- `Candidates.from_idata` and `posterior_runs` accepting `Candidates` / run
  objects had already landed in steps 24 and 30; nothing to add.
- Targets are overlaid by a separate public piece, `add_targets`, which the
  `targets=` arguments call (P2).
- Notebook number is 25, not 23. It is an acceptance page with no asserts
  (notebook 22's style); the plan's "posterior median covers the targets"
  assertion is replaced by what the figures show, because the ribbons are
  trajectory bands, not posterior-predictive intervals — see
  `futureplans/posterior-predictive-observation-noise.md`.
