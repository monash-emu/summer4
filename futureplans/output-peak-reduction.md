# A lazy peak (max / min over time) for outputs

**Status:** open. Not scheduled. Left out of `plans/time-free-outputs.plan.md` on purpose.  
**Where:** `Output` in `src/summer4/results/output.py`; `_CALLS` in
`src/summer4/results/outputset.py`.

## Problem

The time reductions that drop `time` are `at(when)` and `integrate()`. Peak
prevalence or peak weekly incidence (and the time of the peak) is a common epi
quantity with no verb. Today it is spelled through the array escape:

```python
outputs["peak"] = defer(lambda a: jnp.max(a, axis=0))(outputs.ref("prev").array())
```

That works lazily and under `jit`, but returns a plain array (no dims or map), so a
per-age peak loses its `PropertyMap` and has to be re-wrapped with
`outputs.ref("prev").at(t).with_values(...)`.

The time-free plan rejected an `over_time(how="sum" | "mean" | "max" | ...)` verb:
`sum` and `mean` would add a third total that disagrees with `integrate()` and
`cumulative().at(t_end)`, and a `how=` string is the shape the composability goal
(P1) argues against.

## Done looks like

`Output.peak()` / `Output.trough()` (or `max_over_time()`), dropping `time` and
keeping the other dims and the map like `integrate()`, in `_CALLS`, with an
`argmax` companion that returns the model time of the peak. One `jnp.max` over the
time axis: no per-time equations. Port as a `PostOp` with roadmap steps 36-38.
