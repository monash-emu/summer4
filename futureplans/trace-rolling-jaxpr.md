# `Output.rolling` may explode the jaxpr

**Status:** open  
**Where:** `Output.rolling` → `Output._apply_rolling` in
`src/summer4/results/output.py`

## Problem

`_apply_rolling` builds each output time with a **Python** `for i in range(n)`
over the time axis, appending JAX ops and then `jnp.stack`ing. Under `jax.jit`,
that loop unrolls: jaxpr (and XLA compile time) grow with trajectory length
`T`, not as a single vectorized window op.

Host-side index arithmetic plus one batched gather/segment-reduce (as in
`at_times` / `reduce_by`) does *not* have this shape. Rolling’s *intent*
(cumsum / window difference via `RollingSpec`) is vectorizable; the current
body is not.

## Why it matters

`jax.jit` is the primary target (`AGENTS.md`). A calibration loss that calls
`.rolling(...)` on a long daily grid can pay a large compile cost even when the
math is O(T) at runtime.

## What “done” looks like

- Rewrite `_apply_rolling` as a closed-form / vectorized prefix-sum window (or
  `lax.scan` with a fixed window body) so jaxpr size is O(1) in `T`.
- Guard with `jax.make_jaxpr` (compare node count or pretty-print size vs `T`)
  plus a timing benchmark if useful.
- Keep behaviour: `how`, `center`, `min_periods`, NaN when under-filled.

## Related

Phase 2 “static index / traced gather” rule in
`plans/flows-derived-outputs.plan.md` §2d — rolling should follow that pattern,
not the “Python loop over targets unrolls” pattern (that applies to a *small*
static set of named targets, not the time axis).
