# Start a NUTS warm-up from a previous run's draws

**Status:** open. Deliberately left out of step 47 (`plans/nuts-tuning.plan.md`).
**Where:** `src/summer4/epi/calibration/workflow/tune.py` (`LaplaceMetric`,
`NutsSetting.initial_metric`); `warmup.py::_adapted_metric`.

## Problem

`wf.laplace_metric` primes the metric from the Hessian at one optimum. After a
model change, the previous calibration's posterior draws (an `MCMCRun` or
`InferenceData`) are usually a better starting metric than a Laplace fit, and
the previous run's adapted `inverse_mass_matrix` (in
`mcmc.last_state.adapt_state`) is better still. Today a caller can pass either
to `warmup_until(inverse_mass_matrix=...)` by hand, but must know numpyro's
block order (`tuple(sorted(sites))`, each site raveled) and that the draws must
be unconstrained first.

## Done looks like

`wf.empirical_metric(post, draws) -> LaplaceMetric`-shaped object (same
`inverse_mass_matrix(dense=)`), built from unconstrained draws in
`ravel_pytree` order with optional shrinkage, and `NutsSetting` accepting any
such metric. A test that a run warmed from the previous posterior needs fewer
warm-up gradient evaluations than one warmed from the identity.
