# Posterior runs have no observation noise (no posterior predictive)

## What is wrong today

`BayesianModel.posterior_runs` (`src/summer4/epi/calibration/posterior_runs.py`,
`run_posterior`) evaluates the model's **trajectories** for each draw. The
likelihood on each `Target` (`NegativeBinomial`, `Poisson`, `NormalLikelihood`
in `src/summer4/epi/calibration/likelihoods.py`) is never sampled from, so
`wf.plot_ribbons` draws a band of model means, not a posterior-predictive
interval for the data.

## Why it hurts

A modeller checking the fit expects the 95% band to contain about 95% of the
observations. With noisy data and a well-identified early epidemic the
trajectory band is narrow and individual data points sit outside it — in
`examples/notebooks/25-calibration-workflow.ipynb` the band has zero width at
day 0 while the day-0 observation is a negative-binomial draw. Step 28's plan
wanted the notebook to assert "the posterior median covers the targets"; that
check is not meaningful without predictive noise, so the notebook says so
instead. estival offered predictive intervals for this reason.

## What a fix looks like

- A `sample(key, mean, params)` method on each likelihood (numpyro already has
  the distributions: `to_numpyro`-style construction plus `.sample`), and a
  public `Likelihood` protocol carrying it (composability finding `CX20`).
- `PosteriorRuns.predictive(target_key, seed=)` → `(n, len(target.times))`
  draws of observations at the target times, reusing the per-draw series.
- `wf.plot_ribbons(..., predictive=True)` or, more composably, a
  `PosteriorRuns`-shaped object for the predictive draws that the existing
  ribbon function draws unchanged.
- A test that about 95% of simulated observations from the true parameters
  fall inside the 95% predictive band.
