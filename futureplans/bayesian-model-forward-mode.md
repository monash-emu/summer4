# Forward-mode differentiation for a `BayesianModel` (Kiribati W6, `CX59`)

## What is wrong today

A model solved with `Diffrax(adjoint=diffrax.ForwardMode())` cannot be calibrated through
summer4's own entry points. `Posterior._ensure_initialized`
(`src/summer4/epi/calibration/posterior.py`) calls numpyro's `initialize_model` without
`forward_mode_differentiation=True`, and its valid-start check takes a reverse-mode gradient, which
fails through a forward-mode solve:

```text
ValueError: Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop with
dynamic start/stop values.
```

So `potential_fn`, `log_density`, `wf.evaluate` and `wf.optimize` (whose `Optax` step is
`jax.value_and_grad`) are unusable with the forward-mode adjoint. NUTS has the same problem unless
the caller passes `forward_mode_differentiation=True` to the kernel themselves.

## Why it matters

The Kiribati first port needed forward mode where reverse-mode gradients of the adaptive solve
were NaN (its sharp edge S5: 17 % of 256 prior design points with the original Dopri5 setting). It
rebuilt the log density by hand (`kiribati-tb-summer4/src/kiribati_tb/pipeline.py`,
`forward_log_density`; the same code now lives in `tb-core`). The port's later solver choice
(Bosh3 with a PI controller and `jump_ts`, its S10) made reverse mode finite again, so this is not
blocking it today, but any model whose reverse-mode gradient fails has no summer4 route.

## What done looks like

- `BayesianModel(..., forward_mode=True)`, or the mode read off the run's solver adjoint, passed to
  `initialize_model(forward_mode_differentiation=...)`.
- `wf.Optax` uses `jax.jacfwd` (or `jax.linearize`) for such a model; `wf.evaluate` and
  `wf.optimize` work unchanged.
- The NUTS factory documented in the calibration guide shows passing
  `forward_mode_differentiation=True` to `NUTS`.
- A test calibrates a small model solved with `ForwardMode()` through `find_map` and `run_mcmc`.
