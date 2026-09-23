---
name: epi-sampling
overview: BayesianModel with log_density, MAP, and NUTS/AIES/ESS sampling — WP10 §10.2 / roadmap step 14.
todos:
  - id: model
    content: BayesianModel wiring priors, run, outputs, likelihood factor, -inf on solver failure
    status: pending
  - id: prepare_fn
    content: Reject preprocess=; use CompiledModel.prepare_fn (delete futureplans note)
    status: pending
  - id: map_sample
    content: find_map via optax; sample via numpyro MCMC → arviz.InferenceData
    status: pending
  - id: tests_nb
    content: Extend test_epi_calibration (§10.2 parts) and 17-priors-and-likelihoods.ipynb
    status: pending
isProject: false
---

# Bayesian sampling (WP10 §10.2)

Follows `plans/tb-ports-feature-completeness.plan.md` §10.2. Roadmap step 14 on
`feat/epi-sampling`. Closes no port row by itself — step 15 closes
`KI18`–`KI21` and `TM8`.

## prepare_fn, not preprocess

`futureplans/wp10-preprocess-is-prepare-fn.md`: do **not** add a parallel
`preprocess=` hook. Run-start work (yearly mixing stacks, knots) belongs on
`FlowModel.compile(prepare_fn=...)`. `BayesianModel(..., preprocess=...)` raises
and names `prepare_fn`. Delete that futureplans note when this lands.

## API

```python
bm = BayesianModel(
    compiled, fixed_params, priors, targets,
    outputs=OutputSet | None, init=InitialPopulation | None, y0=...,
    run_kwargs=dict(t0=..., t1=..., dt=..., solver="dopri5", ...),
)
bm.log_density(unconstrained)  # jit-able; unconstrained site dict
bm.find_map(init=None, steps=..., optimizer=optax.adam(...)) -> dict
bm.sample(kind="nuts"|"aies"|"ess"|"sa", ...) -> arviz.InferenceData
```

Default `chain_method="vectorized"`. Solver `ok` is false → likelihood factor
`-inf`.
