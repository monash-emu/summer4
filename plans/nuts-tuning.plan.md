# Step 47 — NUTS hyperparameter tuning stage (design spike)

Roadmap step 47 (Phase K, package `CP7`; closes `CX36`, `CX37`, `CX38`). Asked for by the TB
natural-history joint calibration (tbcore NH4, `scripts/fit_nh_hierarchical.py`), but the stage is
domain-neutral: it takes any `Posterior` and any `Candidates`.

## The problem

Every new calibration picks NUTS settings by hand: diagonal or dense mass matrix, how to start the
metric, `max_tree_depth`, `target_accept_prob`, initial step size, warm-up length. The right answer
changes as the model changes, and the costs are lopsided. Evidence from tbcore NH4 (194-dimensional
hierarchical posterior, one log-density+gradient ~4.4 ms):

| | diagonal | dense from identity |
|---|---|---|
| warm-up | 1 round, 5 min | 2 rounds, 54 min; 8% of iterations hit max tree depth |
| leapfrog steps per draw after warm-up | ~115 | ~60 |
| divergences, R-hat | 0, 1.004 | 0, 1.004 |

Dense samples twice as fast but its warm-up costs ten times more. Which is better depends on how
many effective draws are wanted, so the tuner must report warm-up cost separately from sampling
efficiency, and rank for a stated target ESS. Dense from identity is slow to warm up because the
first adaptation window integrates with a unit metric on a badly scaled posterior. A Laplace
metric (inverse negative Hessian at the best optimum, eigenvalues clipped) gives the first window
the right scales; tbcore uses this already (`tbcore.fit.pipeline.laplace_covariance`), but for
scalar sites only, sorted by name.

## What numpyro does (0.16, checked in the source)

- `NUTS(..., inverse_mass_matrix=M)` is the **initial** metric of warm-up. `dense_mass=True`
  becomes the one block `tuple(sorted(z))`, and an array `M` becomes `{tuple(sorted(z)): M}`. The
  block's vector is the sites in sorted-name order, each raveled: that is
  `jax.flatten_util.ravel_pytree` of the site dict. A 2-D `M` given to a diagonal kernel is
  reduced to its diagonal.
- With `adapt_mass_matrix=True`, the metric is re-estimated at the end of each slow window from
  the window's draws (regularised towards `1e-3 * I`); `M` governs the initial buffer and the first
  window, which is where dense-from-identity wastes its time. With `adapt_mass_matrix=False`, `M`
  is used throughout and warm-up adapts only the step size.
- Initial step size: `step_size` (default 1.0) seeds dual averaging at `log(10 * step_size)`.
  `find_heuristic_step_size=True` searches for a reasonable value at the start and after each
  window. Neither is searched by default.
- `extra_fields=("num_steps", "accept_prob", "adapt_state.step_size", "diverging")` are available
  per draw; `num_steps` is the number of leapfrog steps, that is gradient evaluations.

## Gaps closed first (each a composability finding)

| ID | Today | After |
|---|---|---|
| `CX36` | `sample_until` / `MCMCRun` request only `diverging`; `num_steps` and `accept_prob` are lost (tbcore `docs/summer4-workarounds.md` W8). | `sample_until(..., extra_fields=(...))` and `MCMCRun.extra[name]`, `(chain, draw)` per field, concatenated across chunks and `extend`; included in `MCMCRun.idata` `sample_stats`. |
| `CX37` | `_site_diagnostics` reports `np.mean(split_gelman_rubin(x))` over a plate site's elements, so one bad element is hidden by good ones. | Progress rows and `MCMCRun.diagnostics` add `rhat_element_max` and `ess_bulk_element_min` (worst element). `StopRule(per_element=True)` judges on those. The default stays the mean so existing runs stop where they did. |
| `CX38` | `warmup_until` cannot start from a given metric, restarts each doubling round from the factory's metric (throwing away the previous round's adaptation), and reports nothing until it returns. | `warmup_until(..., inverse_mass_matrix=, carry_metric=, on_round=)`. When a metric is supplied or carried, the factory is called as `make_mcmc(n, inverse_mass_matrix=M)`; otherwise as `make_mcmc(n)`, so existing factories keep working. `carry_metric=True` hands round k+1 the chain-mean of round k's adapted metric. `on_round(row)` is called as each row lands. Rows gain `grad_evals` (sum of `num_steps` over the whole round). |

## The tuning stage

New module `src/summer4/epi/calibration/workflow/tune.py`, re-exported from `wf`. Every piece is
public and individually testable; `wf.tune_nuts` is the convenience.

```python
metric = wf.laplace_metric(post, z, min_var=1e-4, max_var=25.0)  # LaplaceMetric
seconds = wf.time_gradient(post, z, repeats=20)                   # one log density + gradient

setting = wf.NutsSetting("dense+laplace", {"dense_mass": True}, laplace=True, num_warmup=100)
pilot = wf.pilot(post, init, setting, metric=metric, num_chains=4, pilot_samples=200)

tuning = wf.NutsTuning(post, init, num_chains=4, pilot_samples=200)  # computes metric, timing
tuning.run(wf.metric_screen())                                       # stage A
tuning.run(wf.trajectory_screen(tuning.best(target_ess=1000).setting))  # stage B
tuning.table(target_ess=1000)          # ranked DataFrame
make_mcmc = tuning.make_mcmc(num_samples=500, num_chains=4)  # the recommendation, as a factory
tuning.best().run.extend(wf.StopRule(...))  # or keep sampling the winning pilot: no new warm-up
```

`wf.tune_nuts(post, init, ...)` is exactly the four `tuning` lines above (tested).

### `laplace_metric(post, z)`

- `z`: one unconstrained point (site dict without a batch axis) or `Candidates` (uses
  `best(1)` if evaluated, else row 0).
- Flatten with `ravel_pytree(z)` (sorted site names, each raveled: numpyro's block order), so plate
  and vector sites work. Hessian of `post.log_density(unravel(v))` by `jax.hessian` (forward over
  reverse). This differentiates the discretised solve exactly; tbcore's central difference of the
  gradient was needed only for its loose adaptive-solver tolerance. A caller with that problem
  passes a `Posterior` with a tighter solver.
- Eigen-decompose `-H`; non-positive eigenvalues become `max_var`; variances `1/λ` are clipped to
  `[min_var, max_var]`; record how many were clipped.
- Returns frozen `LaplaceMetric(sites, mode, covariance, eigenvalues, clipped)` with
  `inverse_mass_matrix(dense: bool)` → `{tuple(sorted(sites)): cov or diag(cov)}`, ready for
  `NUTS(inverse_mass_matrix=...)`, and `std` (marginal standard deviations, for display).

### `time_gradient(post, z)`

`jax.jit(jax.value_and_grad(f))` over the raveled vector; one warm call, then `repeats` timed
calls each ending in `jax.block_until_ready` (JAX dispatch is asynchronous, so an unblocked timer
measures dispatch only). Returns the median seconds. Used to turn gradients into seconds and to
show how much of the wall time is the model.

### `NutsSetting`

Frozen dataclass: `label`, `kwargs` (passed verbatim to `numpyro.infer.NUTS`, per AGENTS.md
*accept the underlying library's objects*: anything NUTS accepts is a candidate), `laplace: bool`
(start from the Laplace metric), `num_warmup` (first warm-up round). Methods:
`kernel(model, inverse_mass_matrix=None)` and
`make_mcmc(model, *, num_samples, num_chains, chain_method="vectorized", metric=None)` returning a
factory `(n, inverse_mass_matrix=None) -> MCMC` that `warmup_until` accepts.

### Default screens (one factor at a time, two stages)

A full factorial (4 metrics × 3 depths × 3 targets × step sizes) is ~50 pilots; most of it is
uninformative because the metric dominates. Two short stages instead:

- `metric_screen()` at numpyro defaults (depth 10, target 0.8): `diag`, `diag+laplace`, `dense`,
  `dense+laplace`, `dense+laplace fixed` (`adapt_mass_matrix=False`). Laplace settings are skipped
  (with a recorded reason) when no metric was computed.
- `trajectory_screen(base)` around the stage-A winner: `max_tree_depth` 6 and 8,
  `target_accept_prob` 0.9 and 0.95, and `find_heuristic_step_size=True` (the initial-step-size
  factor; a fixed `step_size` is a one-line custom `NutsSetting`).

### `pilot(...)` and what each records

Seeded identically (`init.init_params(num_chains, jitter, seed)`), one pilot is:

1. `warmup_until(setting.make_mcmc(...), setting.num_warmup, warmup_check, carry_metric=True,
   inverse_mass_matrix=<Laplace if setting.laplace>)`. Warm-up length is chosen by the doubling
   rounds; record rounds, final `num_warmup`, seconds and gradient evaluations (all rounds).
2. `sample_until(warm.mcmc, StopRule(rhat=None, ess=None, max_samples=2*pilot_samples),
   extra_fields=("num_steps", "accept_prob"))`: two chunks. The first chunk compiles the sampler;
   **speed is timed on the second chunk only**. ESS is over both.

Pilot row: `label`, `warmup_rounds`, `num_warmup`, `warmup_passed`, `warmup_seconds`,
`warmup_grad_evals`, `step_size` (chain mean of adapted), `accept_mean`, `num_steps_mean`,
`num_steps_max`, `treedepth_frac` (draws with `num_steps >= 2**depth - 1`), `divergence_frac`,
`rhat_max` and `ess_min` (worst element over all sites), `ess_per_grad`, `draws_per_second`,
`ess_per_second`, `model_frac` (`grad_seconds * grad_evals / sample seconds`: the share of
sampling spent in the model; low means overhead dominates and gradient counts mislead).

### Ranking

`table(target_ess)` adds `cost_seconds = warmup_seconds + target_ess / ess_per_second` and
`cost_grads = warmup_grad_evals + target_ess / ess_per_grad`, then sorts usable pilots (warm-up
passed, `divergence_frac <= max_divergence_frac`, `rhat_max <= max_rhat`, defaults 0.01 and 1.1)
by `cost_seconds`, unusable ones last with a `why` column. A small `target_ess` favours a cheap
warm-up; a large one favours the fastest sampler — the NH4 trade-off made explicit. Wall seconds
include JIT compilation, which is a real cost of each new setting and similar across settings;
`cost_grads` is the hardware- and compile-free comparison.

`recommended(target_ess)` returns the winning `NutsSetting` with `num_warmup` set to the length
its warm-up passed at, so a rerun warms up once. `make_mcmc(...)` wraps it with the Laplace metric
when it needs one.

## Tests (`tests/test_calib_workflow_tune.py`, plus additions to the MCMC tests)

- `extra_fields` arrive per chunk and concatenate across `extend`; `idata.sample_stats` has them.
- A plate site with one bad element: `rhat_element_max` exceeds the mean; `StopRule(per_element=True)`
  refuses to stop where the default stops.
- `warmup_until`: `on_round` sees each row as it lands; a supplied metric reaches the kernel;
  `carry_metric=True` hands round 2 the chain-mean of round 1's metric; old one-argument factories
  still work.
- `laplace_metric` on a Gaussian posterior with known covariance (plate and scalar sites) recovers
  it in `ravel_pytree` order; clipping of a flat direction; numpyro accepts the result for both
  dense and diagonal kernels.
- `time_gradient` returns positive seconds and blocks (a deliberately slow function is timed at
  least its sleep).
- `pilot` row fields are consistent (`ess_per_grad * grad_evals == ess`, `treedepth_frac` from
  `num_steps`); `table` ranks by cost and moves a divergent pilot last; `tune_nuts` equals its
  documented expansion.
- Gate claims (`test_gate_*`, on the notebook's two models): on the correlated hierarchical
  posterior a Laplace-primed dense metric needs fewer gradient evaluations per effective draw than
  diagonal, and its warm-up costs fewer gradient evaluations than dense from identity.

## Gate notebook: `29-calibration-nuts-tuning.ipynb`

Acceptance page. Two posteriors: notebook 28's joint SIR + partially pooled outbreaks (a plate,
a non-centred latent, strong correlation between `mu`, `tau` and `z_main`), and the scalar
two-parameter SIR from notebook 22. Sections: the Laplace metric (correlation heat map and
eigenvalue clipping), gradient timing, stage A pilots (bar chart of ESS per gradient and per second,
warm-up cost), stage B around the winner (tree-depth saturation and step size vs depth/target),
cost against target ESS (the crossing curves that make the NH4 trade-off visible), and the
recommended factory used for a real run. Each with a *What to check* box; asserts in `tests/`.

## Not in this step

- Tuning other kernels (`HMCECS`, ensemble samplers) or `chain_method` choice.
- Re-warming from a sampled run's draws (`MCMCRun` → metric): possible with `laplace_metric`'s
  sibling `empirical_metric`, noted in `futureplans/nuts-metric-from-draws.md`.
- Pathfinder initialisation.
