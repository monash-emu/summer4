---
name: epi-posterior-runs
overview: Batched posterior scenario runs, quantile/difference frames, 18-calibration notebook and textbook ch.20 — WP10 §10.3 / roadmap step 15.
todos:
  - id: api
    content: Scenario + PosteriorRuns + BayesianModel.posterior_runs (per-draw samples, quantiles, differences)
    status: completed
  - id: tests
    content: Quantile parity vs looped draws; differences schema; plain site-dict input
    status: completed
  - id: notebooks
    content: examples/notebooks/18-calibration.ipynb + docs/textbook/20-calibration.ipynb
    status: in_progress
  - id: ledger
    content: Close KI18–KI21 TM8; chapter 20 full; handoff to step 24
    status: pending
isProject: false
---

# Posterior runs (WP10 §10.3)

Follows `plans/tb-ports-feature-completeness.plan.md` §10.3–§10.4 with roadmap
corrections: keep per-draw outputs (`samples(scenario, output) → (n, t)`); accept
plain constrained site dicts (and later `Candidates`) as well as
`InferenceData`. Notebook is `18-calibration.ipynb`. Handoff goes to Phase J
step 24, not Track G.
