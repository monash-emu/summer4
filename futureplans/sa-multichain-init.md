# `bm.sample("sa", num_chains>1)` fails in numpyro's vectorised init

Found while fixing `tests/test_epi_calibration.py::test_sa_recovers_infection_in_95_interval`
on `feat/float64-default`. Same failure in float32 and float64.

```python
bm = BayesianModel(cm, params, priors=(Uniform("infection", 0.05, 0.8),), targets=targets,
                   y0=y0, run_kwargs=dict(t0=0.0, t1=80.0, dt=1.0, solver="euler"))
bm.sample("sa", num_warmup=2000, num_samples=1000, num_chains=4, seed=2, progress_bar=False)
```

raises from `numpyro/infer/sa.py` `init`: `vmap(sa_init_fn)(init_params, rng_key)` —
"vmap was requested to map its argument along axis 0 … but is only 0 (its shape is ())".
The init params handed to a multi-chain SA kernel are not batched per chain (the NUTS path
batches them). Likely in how `BayesianModel.sample` / `Candidates.init_params` builds
`init_params` for `num_chains > 1` with a non-HMC kernel
(`src/summer4/epi/calibration/model.py`, `src/summer4/epi/calibration/workflow/mcmc.py`).

Done when multi-chain SA samples, with a test, or `sample` raises a clear error for it.
