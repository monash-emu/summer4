# NUTS tuning gate test is not reproducible across CI runners

**Resolved** on `fix/nuts-gate-divergences` (2026-11-26). Kept as a record of what was measured.

`tests/test_calib_workflow_tune.py::test_gate_recommended_run_converges_on_every_element`
failed on PR #63 (the 0.2.0a6 release, reason `max_samples`), on `feat/nuts-tuning`, and on PR #91
(the 0.3.0a6 release, reason `divergences` after one chunk). None of these changed the code
under test.

## What was wrong

The gate run (`tests/gate_nuts_tuning.py::run_gate`) is fully seeded, and its results are
deterministic for a given CPU family but differ between them. A sweep on four GitHub runners
(three Intel Xeon of two models, one AMD EPYC 7763) gave identical results on every Intel
runner and different ones on AMD, where tuning even recommends a different setting
(`diag+laplace` rather than `dense+laplace`). With `RUN_SEED = 1`, Intel always stopped at 1.5%
divergent draws and AMD passed, so the test's outcome depended on which runner a job got.

Underneath, the recommended setting diverges on about 0.5-0.9% of draws while the final
`StopRule` stopped above 1%. A pilot is usable at up to 1%, so the cheapest usable setting tends
to sit near that limit. Over five seeds on each CPU family, 2 of 10 final runs stopped on
divergences.

## The fix

The final run raises `target_accept_prob` to 0.9 (`RUN_TARGET_ACCEPT`) and allows 2% divergent
draws (`RUN_MAX_DIVERGENCE`), in both the gate and notebook 29. In the same sweep all 10 runs
converged with diagnostics, the worst at 1.0%. The assertion message now prints the run's
progress (divergences, worst-element R-hat and ESS), so a future failure shows how far off it
was.

## Still open

`NutsTuning`'s usability check uses the same 1% divergence limit as a typical final run, so a
recommendation can be marginal by construction. A stricter pilot limit, or recommending a
higher `target_accept_prob` when the best pilot is near the limit, would fix this in the
library rather than in each caller.
