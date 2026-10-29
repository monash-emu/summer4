# NUTS tuning gate test is not reproducible across CI runners

`tests/test_calib_workflow_tune.py::test_gate_recommended_run_converges_on_every_element`
failed once in CI's `default` environment (JAX 0.7.2) on PR #63, the 0.2.0a6 release, which
changed no code. The same seeded test had passed on #61 and #55. A rerun of the same job passed.

## What is wrong

The gate run (`tests/gate_nuts_tuning.py::run_gate`) is fully seeded, but the result still
differs between runners (XLA CPU codegen or thread scheduling changes floating-point
reduction order). The final `wf.sample_until` stops on
`wf.StopRule(rhat=1.01, ess=TARGET_ESS, per_element=True, max_samples=RUN_MAX_SAMPLES)`. On an
unlucky runner it hits `max_samples` before every element reaches R-hat ≤ 1.01, so
`gate.run.converged` is false and `reason` is not `"diagnostics"`.

The project's development threshold is R-hat < 1.05; 1.01 is for final results.

## Why it hurts

The test takes about 10 minutes of a 40-minute CI job, so a flake costs a full rerun and
reads as a regression on unrelated PRs.

## What done looks like

One of these, chosen with the notebook 29 gate claim in mind (the notebook says the
recommended run converges on every element):

- give the gate run more headroom (`RUN_MAX_SAMPLES`), and check the margin by running the gate
  under several `RUN_SEED`s;
- or assert the claim at the development threshold (1.05) in the test while the notebook
  reports the measured values;
- and in either case, log `rhat_element_max` / `ess_bulk_element_min` in the assertion message,
  so a failure shows how far off it was.
