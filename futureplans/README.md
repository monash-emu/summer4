# Future plans (deferred gotchas)

Short, agent-written notes about problems or risks spotted while building a
feature plan or reviewing code, when fixing them is **out of scope for the
current branch**.

This is not the design-plan archive (`plans/`). Notes here are reminders:
jaxpr/XLA compile-time traps, sharp edges, and follow-up refactors. Promote a
note into `plans/<slug>.plan.md` when it becomes scheduled work.

## Conventions

- One concern per file; prefer a stable slug (`trace-rolling-jaxpr.md`).
- Link to concrete symbols or paths (`Trace._apply_rolling`,
  `src/summer4/results/trace.py`).
- Say what is wrong today, why it hurts (e.g. jaxpr size ∝ T), and what a fix
  should look like.
- Agents: read this folder when planning; append or update notes when you flag
  something for later. See `AGENTS.md` (§ Future plans, § JAX).

## Notes

- [`trace-rolling-jaxpr.md`](trace-rolling-jaxpr.md) — Python loop in
  `Trace._apply_rolling` grows the jaxpr with trajectory length.
- [`state-ledgers-incidence.md`](state-ledgers-incidence.md) — opt-in exact
  incidence via `State.ledgers` (post-hoc trapezoid is biased for calibration).
- [`targetset-residual-reduction.md`](targetset-residual-reduction.md) —
  `TargetSet.residuals` cannot reduce a stratified save onto an aggregate
  series.
- [`trace-plot-backend-coupling.md`](trace-plot-backend-coupling.md) —
  `Trace.plot` hardcodes matplotlib and forwards backend-specific kwargs.
- [`foi-unstratified-dummy-pop.md`](foi-unstratified-dummy-pop.md) —
  unstratified FOI still needs a singleton `pop` property + `[[1.0]]` matrix.
- [`foi-susceptibility-surface.md`](foi-susceptibility-surface.md) —
  infectiousness has `ForceOfInfection(infectiousness=...)`; susceptibility does not.
- [`next-generation-matrix-r0.md`](next-generation-matrix-r0.md) —
  unstratified $R_t$ is ported; no next-generation-matrix / spectral $R_0$ helper.
- [`propertydata-dense-unstack.md`](propertydata-dense-unstack.md) — opt-in
  reshape of a dense Cartesian `PropertyMap` from a flat last axis to
  `(age × state × …)` (not first-class multi-axis `PropertyData`).
- [`derived-fn-blocks-hoisting.md`](derived-fn-blocks-hoisting.md) — R3: a
  `derived_fn` makes every `FieldRef` step-stage and disables param hoisting.
- [`mixing-matrix-per-call-normalisation.md`](mixing-matrix-per-call-normalisation.md)
  — `MixingMatrix.resolved_matrix` re-normalises FieldRef matrices every step.
- [`wp10-preprocess-is-prepare-fn.md`](wp10-preprocess-is-prepare-fn.md) —
  WP10 `preprocess=` should be `CompiledModel.prepare_fn`, not a second hook.
- [`adjustment-expanding-arrays.md`](adjustment-expanding-arrays.md) — evaluate
  adjustment chains over unique value combinations (summer3proto `polarized`)
  instead of folding over every edge; benchmark-gated.
