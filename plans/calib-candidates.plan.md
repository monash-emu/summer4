---
name: calib-candidates
overview: Candidates, Prior.icdf, LHS design, batched evaluate, best-N, save/load — WP19 / roadmap step 24.
todos:
  - id: api
    content: Candidates + StageRecord; Prior.icdf; bm.constrain/unconstrain; wf.lhs/prior_draws/evaluate
    status: completed
  - id: tests
    content: tests/test_calib_workflow_candidates.py (LHS strata, icdf, jaxpr, best, save/load)
    status: completed
  - id: notebook
    content: examples/notebooks/19-calibration-design.ipynb — best 16 bracket true infection
    status: completed
  - id: ledger
    content: Close CW1–CW4
    status: completed
isProject: false
---

# Calibration candidates (WP19 / step 24)

Follows `plans/calibration-toolkit.plan.md` *Step 24*. Shared value
`Candidates` plus the first three stages: Latin-hypercube design, memory-bounded
scoring, and keep-the-N-best. Closes ledger rows `CW1`–`CW4`.
