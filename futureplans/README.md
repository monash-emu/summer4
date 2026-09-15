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
