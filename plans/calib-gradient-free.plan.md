---
name: calib-gradient-free
overview: CMA-ES backend (evosax) on the multi-start driver — WP19 / roadmap step 26.
todos:
  - id: api
    content: wf.CMAES + gradient-free extra; wired into wf.optimize
    status: completed
  - id: tests
    content: CMA-ES recovery, stop_gradient case, Optax vs CMA-ES descent
    status: completed
  - id: notebook
    content: examples/notebooks/21-calibration-gradient-free.ipynb
    status: completed
  - id: ledger
    content: Close CW7
    status: completed
isProject: false
---

# Calibration gradient-free (WP19 / step 26)

Follows `plans/calibration-toolkit.plan.md` *Step 26*. Closes ledger row `CW7`.
Evosax `>=0.3.1,<0.4` resolves in the jax06 env.
