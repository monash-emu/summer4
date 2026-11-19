# A joint coordinate map on `BayesianModel` (Kiribati W9 and W11, `CX62`)

## What is wrong today

`BayesianModel.numpyro_model` samples every prior in its own bijector (`numpyro.sample(prior.name,
prior.to_numpyro())` in `src/summer4/epi/calibration/model.py`), with no hook to change
coordinates jointly. Two Kiribati workarounds need one:

- **W9.** The transmission ridge (`raw_transmission_rate` against `infection_pop_scale`) is thin
  and curved in those coordinates. The port runs NUTS on `NUTS(potential_fn=shear.potential(bm))`
  in sheared coordinates and maps draws back by `bm.constrain(shear.inverse(w))`
  (`kiribati-tb-summer4/src/kiribati_tb/pipeline.py`, `RidgeShear`; now in `tb-core`). Running on
  a potential function loses numpyro's `postprocess_fn`, so constrained draws are rebuilt by hand.
- **W11.** NeuTra: the flow is fitted, and NUTS run, on a one-site model
  `x ~ ImproperUniform(real^19)` with `factor(-bm.potential_fn(coords.to_unconstrained(x)))`
  (`kiribati-tb-summer4/src/kiribati_tb/neutra.py`, `WhitenedCoordinates`). Applying
  `NeuTraReparam` to `bm.numpyro_model()` directly starts the flow at a standard normal in the logit
  coordinates, far from a ridge 0.02 wide there (loss about 4,900 nats, then NaN). Starting at the
  Laplace approximation needs a joint affine (and sheared) change of coordinates before the flow.

## What done looks like

- `BayesianModel(..., coordinates=)` takes a bijection of the unconstrained vector (forward and
  inverse with log-det, for example a numpyro `Transform` on the flattened sites, or a numpyro
  `Reparam` applied jointly). `numpyro_model()` samples one latent site in those coordinates and
  records the constrained sites as deterministic, so `MCMC(NUTS(bm.numpyro_model()))`,
  `postprocess_fn`, `wf.sample_until` and `posterior_runs` all return constrained draws.
- A helper builds the affine whitening map from a Laplace fit (`find_map` plus the Hessian).
- A test on a banana-shaped two-parameter posterior shows fewer divergences and a higher ESS per
  gradient than the default coordinates, and identical constrained draws' moments.
