<!--
Accepted 2026-11-23 as proposed in conversation. Historical record once committed: do not edit it.
-->

# `warmup_until` leaves numpyro's step-size-restart draws out of its checks

Branch: `fix/warmup-skip-resets`, cut from `main` at `5dab1b5`.

## Context

A step-by-step trace of NUTS warmup on a 4×4 grid model found most warmup divergences are
numpyro's adaptation, not the posterior. At the end of each middle adaptation window numpyro
installs the new mass matrix and restarts dual averaging from ten times a freshly found step
size (`_update_at_window_end` in `numpyro.infer.hmc_util`); the next draw diverges in every
chain. Reproduced on a 3-d standard normal: windows ending at 99, 149 and 249 give divergences in
all 4 chains at draws 101, 151 and 251 (step sizes 6–13 against about 1.4), plus draw 1.

`WarmupRule` checks the second half of each round, and numpyro's final window always starts 50
draws before the end, so every round has at least one restart draw in the checked half: one per
chain per restart, 1.3% of checked draws at `num_warmup=150` or 300 and 2% at 100 or 200, over
the 1% divergence limit with no help from the posterior.

## What to do

1. `_reset_draws(num_warmup, skip)` in `workflow/warmup.py`: the `skip` draws starting one after
   each restart (draw 0 and the end of each middle window of
   `numpyro.infer.hmc_util.build_adaptation_schedule`).
2. `warmup_until(skip_after_reset=1)`: leave those draws (in the checked half) out of
   `accept_mean`, `divergence_frac` and `treedepth_frac`; record `reset_divergences`. No masking
   when the kernel does not adapt its step size. R-hat and step-size statistics unchanged.
3. Docs: `warmup_until` and `WarmupRule` docstrings, `PILOT_WARMUP_RULE` docstring, changelog,
   notebook 22's description of the warmup check.

Not in scope: the background divergence rate while the step size re-adapts (about 1 per 100
draws per chain on the grid model, clustered in setting A's tail), which may be geometry and
needs its own look (post-warmup divergences there; dense metric).

## Acceptance

- A 150-step warmup of a standard normal passes the divergence criterion by default and fails it
  with `skip_after_reset=0` (`tests/test_calib_warmup_resets.py`).
- `_reset_draws` matches numpyro's windows for 10, 150 and 300 draws.
- Existing calibration workflow, tuning, multi-model and posterior gate tests pass; notebooks
  that use `warmup_until` (22, 28, 38, guide 11) still tell their story.
