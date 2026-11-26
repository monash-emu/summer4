<!--
Accepted 2026-11-26 as proposed in conversation. Historical record once committed: do not edit it.
-->

# `laplace_metric` falls back to a finite-difference Hessian

Branch: `feat/laplace-fd-hessian`, cut from `main` at `1ccf7e4`.

## Context

tb_acf found that `wf.laplace_metric` cannot differentiate a posterior whose ODE is solved with
summer4's `default_solver()` (`Diffrax(Tsit5, PIDController(rtol=1e-6, atol=1e-9))` with diffrax's
default `RecursiveCheckpointAdjoint`). `jax.hessian` is forward over reverse, and the adjoint is a
`custom_vjp`, so JAX raises `TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp
function`. Reverse over reverse (`jax.jacrev(jax.grad(f))`) fails too, on the adaptive
`while_loop`. tb_acf keeps a second posterior solved with `DirectAdjoint` just for the metric.

Options weighed:

- Swap the adjoint automatically: not feasible, the solver is inside the model function.
- Make `DirectAdjoint` the default: every gradient (so every NUTS step) costs about twice as much
  (0.07 ms to 0.14 ms on a small SIR model).
- Central differences of the reverse-mode gradient (chosen): match the exact Hessian to about
  2e-8 relative for relative steps 1e-3 to 1e-6 on the same model, cost 2n gradients once per
  metric, and need nothing from the model.

## What to do

1. `laplace_metric(..., hessian="auto", fd_step=1e-4)` in `workflow/tune.py`:
   - `"exact"`: `jax.hessian` as today; the forward-mode `TypeError` (checked by message) becomes
     a `ValueError` naming `hessian="fd"` and `adjoint=diffrax.DirectAdjoint()`.
   - `"fd"`: one jitted `jax.vmap(jax.grad(f))` over the (2n, n) stack of `x ± h_i e_i`,
     `h_i = fd_step * max(1, |x_i|)`, with a per-point loop if the model does not vmap;
     symmetrised.
   - `"auto"`: exact, falling back to fd only on that failure. Other values raise.
   - The non-finite check applies to both.
2. `LaplaceMetric.hessian` (last field, default `"exact"`) records the method used.
3. Callers: `NutsTuning` and `tune_nuts` stay on `"auto"`. `NutsTuning(metric=...)` is already
   the composable way to choose (AGENTS.md: accept the caller's objects), so no pass-through
   keyword; the docstring says so.
4. Docs: docstrings, changelog, notebook 29's Laplace section, new acceptance notebook 43.

## Acceptance

- `tests/test_calib_laplace_fd.py`: on the default-solver SIR posterior `"auto"` uses fd and its
  covariance matches the `DirectAdjoint` exact one to `rtol=1e-5`; `"exact"` raises the new
  `ValueError` chained from the `TypeError`.
- `tests/test_calib_workflow_tune.py`: fd on the Gaussian model matches the exact covariance
  (vmapped and looped); `"auto"` on it uses exact; a bad `hessian` or `fd_step` raises.
- Notebook 43 shows the failure, the fallback, the comparison, timings and a short NUTS run
  from the fd metric.
