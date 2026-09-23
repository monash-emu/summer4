---
name: epi-priors-likelihoods
overview: Priors, target likelihoods, and TargetSet.log_likelihood — WP10 §10.1 / roadmap step 13.
todos:
  - id: priors
    content: Frozen prior dataclasses with to_numpyro() and bounds; priors_from_frame
    status: pending
  - id: likelihoods
    content: Normal/Poisson/NegativeBinomial on Target; hierarchical sd; from_tolerance
    status: pending
  - id: loglik
    content: Traceable TargetSet.log_likelihood summing likelihood.log_prob
    status: pending
  - id: gate
    content: tests/test_epi_calibration.py (§10.1 parts) and 17-priors-and-likelihoods.ipynb
    status: pending
isProject: false
---

# Priors and likelihoods (WP10 §10.1)

Follows `plans/tb-ports-feature-completeness.plan.md` §10.1. Roadmap step 13 on
`feat/epi-priors-likelihoods`. Closes no port row by itself — steps 14–15 close
`KI18`–`KI21` and `TM8`.

Calibration lives in `summer4.epi.calibration` (separation rule). `numpyro` and
`optax` stay in the `calibration` extra.

## Priors

Frozen dataclasses naming a parameter: `Uniform`, `Normal`, `LogNormal`,
`TruncatedNormal`, `Beta`, `Gamma`. Each exposes `.to_numpyro()` and `.bounds`
for unconstrained transforms. `priors_from_frame(df, name_col, dist_col, p1_col,
p2_col)` covers Kiribati's `parameters.xlsx` constant sheet.

## Likelihoods

`Normal(sd)`, `Poisson()`, `NegativeBinomial(dispersion)` set on
`Target(..., likelihood=...)`. `sd` may be a float, a `Param` / `FieldRef`, or a
prior (hierarchical target scale). `Normal.from_tolerance(data, tol_pct)` sets
`sd = (tol_pct/100) * mean / 1.96`.

## Log likelihood

`TargetSet.log_likelihood(result, params) -> scalar`, traceable, summing
`likelihood.log_prob(observed | predicted)`.

Sampling (`BayesianModel`), MAP, and posterior runs are steps 14–15.
