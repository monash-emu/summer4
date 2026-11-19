# `MCMCRun` save/load and tail ESS in `StopRule` (Kiribati W8, `CX61`)

## What is on `main` already

- `sample_until(..., extra_fields=)` keeps numpyro's per-draw fields on `MCMCRun.extra` (step 47,
  `CX36`).
- `StopRule(max_seconds=)` stops before a wall-clock deadline.
- `StopRule(per_element=True)` judges the worst element of a plate site (`CX37`).

## What is still missing

- **No persistence.** `wf.MCMCRun` lives only in memory: there is no `save` / `load`, so resuming
  in a new process means reaching into its private `_sample` / `_decide`, or losing the earlier
  draws from the stop rule's diagnostics. `warmup_until` is not checkpointed either. The Kiribati
  port writes `idata.nc`, the pickled sampler state and `progress.csv` from inside its stop
  callable after every chunk (`kiribati-tb-summer4/src/kiribati_tb/pipeline.py`, `Checkpoint`;
  now in `tb-core`), and `Checkpoint.resume` gives a fresh `MCMC` the saved state as
  `post_warmup_state`.
- **No tail ESS or rank-normalised R-hat.** `_site_diagnostics`
  (`src/summer4/epi/calibration/workflow/mcmc.py`) uses numpyro's classic `split_gelman_rubin`
  and bulk `effective_sample_size`. The port decides on arviz's rank-normalised R-hat and bulk
  **and tail** ESS, because tail quantiles are what it reports.

## What done looks like

- `MCMCRun.save(path)` writes the draws, extra fields, progress table and the sampler's last state;
  `MCMCRun.load(path, mcmc)` rebuilds a run that `extend` continues, with the stop rule seeing every
  draw since the first chunk. A test saves after two chunks, loads in a fresh object and checks the
  third chunk continues the chain.
- The progress row carries `ess_tail_min` (and its worst-element twin), from arviz when installed,
  and `StopRule(ess_tail=)` requires it; R-hat can be rank-normalised (`StopRule(rhat_kind=)` or a
  diagnostics callable).
