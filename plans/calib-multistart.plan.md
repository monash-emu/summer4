---
name: calib-multistart
overview: Multi-start optax optimisation with AutoTune — WP19 / roadmap step 25.
todos:
  - id: api
    content: bm.potential_fn; wf.Optax / AutoTune / optimize / OptimizeResult; _Method protocol
    status: completed
  - id: tests
    content: MAP parity, freeze, restart, LR probe, jaxpr size
    status: completed
  - id: notebook
    content: examples/notebooks/20-calibration-multistart.ipynb
    status: completed
  - id: ledger
    content: Close CW5–CW6
    status: completed
isProject: false
---

# Calibration multi-start (WP19 / step 25)

Follows `plans/calibration-toolkit.plan.md` *Step 25*. Closes ledger rows
`CW5`–`CW6`.
