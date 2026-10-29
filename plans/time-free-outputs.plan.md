# Time-free outputs broadcast over time; a lazy escape to plain arrays

Branch: `feat/time-free-outputs`, cut from `main` at `d08db9a` (after v0.2.0a7).
Closes composability findings **CX39** and **CX40** (recorded by this branch).
Not a roadmap step; it does not move the *Current position* block. It touches the
`OutputExpr` surface that roadmap steps 36-38 (**CX30**) replace, and says below how
each new verb ports.

## Problem

`Output.integrate()` drops the `time` dim but keeps a one-point `TimeAxis` at the last
save time. `Output.at(t)` does the same with `[t]`. Any arithmetic between that
result and a series goes through `_require_same_times`, which compares a one-point
axis with the full save grid and raises *Output time axes differ*:

```python
rate = FlowMass("infection").sum_by()
outputs["share"] = outputs.ref("rate") / outputs.ref("rate").integrate()  # raises
```

The same happens eagerly (`r / r.integrate()`). The only way round it today is
`defer(f)(...)`, which is the wrong tool for "divide a series by a number the model
produced". Two related gaps:

- `at(when)` is an `Output` method but not a lazy post-op (`_CALLS`), so a named
  output cannot use the value at a time.
- The lazy surface has no way out to a plain array. `OutputSet.evaluate` rejects any
  named output that is not an `Output`, and once a value is an `Output` the only
  array-level operations are the pointwise ones. Getting back from an array to an
  output with times (for a target or a plot) needs `defer` too.

## Design

### 1. Time-free outputs broadcast over time (CX39)

An `Output` whose `dims` lack `"time"` is **time-free**. Its `times` is an anchor for
display (`to_frame`, `to_pandas`) and nothing else.

`Output._combine_outputs`:

| left has time | right has time | check | result `times` |
|---|---|---|---|
| yes | yes | `_require_same_times` (unchanged) | left |
| yes | no | none | left |
| no | yes | none | right |
| no | no | none | left (anchor only) |

The named-dim broadcast (`_broadcast_dim_order`, `_broadcast_named`) already treats
`time` as one more name, so `("time",) ∘ ()`, `("time", "group") ∘ ("group",)` and
`("time", "group") ∘ ()` need no new code. PropertyMap retention (`_shared_pmap`) is
unchanged. The alignment stays host-side and static: no per-time equations.

This covers, with no new method:

- `rate / rate.integrate()` — share of the whole-period total at each time;
- `rate - rate.at(t)` — difference from a reference time;
- `per_age / per_age.integrate().sum_by()` — per-age series over the all-age total;
- `per_age.integrate() / per_age.integrate().sum_by()` — time-free over time-free.

**`at_times` on a time-free output raises** a `ValueError` that says the output was
reduced over time, gives its anchor, and names the remedy. **Target scoring** goes
through `at_times`, so it raises the same error, prefixed with the target key.
Broadcasting to the requested times was rejected: a time-free value compared with a
series of observations is almost always a mistake (a total scored against weekly
counts), and silently repeating it hides that. A whole-period quantity is scored with
the existing time-aware outputs: `rate.cumulative()` with `Target(times=[t_end])`
reads the total at the end, or `with_values` (below) gives an array its times.

`Output.at` / `integrate` on a time-free output already raise (`no 'time' axis`);
the message is kept.

### 2. Lazy whole-window reductions

Add `at(when)` to `_CALLS`, `OutputExpr.at` and `_OutputExprChain.at` (so
`FlowMass("x").sum_by().at(10.0)` and `outputs.ref("x").at(10.0)` both work).

**No `over_time(how)`.** Each candidate already has a spelling:

| want | spelling |
|---|---|
| last value | `.at(t_end)` |
| total | `.integrate()` (quadrature) or `.cumulative().at(t_end)` (sum of saves) |
| mean | `.integrate() / (t_end - t_start)` |
| sum of samples | `.cumulative().at(t_end)` |
| peak / trough | `defer(lambda a: jnp.max(a, axis=0))(ref.array())` (see 3) |

An `over_time("sum")` would add a third total that disagrees with both by the
quadrature rule; `"mean"` hides the same choice. Peak is the one real gap; it is
recorded in `futureplans/output-peak-reduction.md` rather than added as a string
`how=` (P1). One verb per policy is what the composability goal asks for.

### 3. Escape hatch to plain lazy arrays (CX40)

| Verb | Eager (`Output`) | Lazy (`OutputExpr`, save-spec chain) | Evaluates to |
|---|---|---|---|
| `array()` | `Output.array()` | node kind `"array"` | the raw values, `PropertyData` unwrapped, axes in `dims` order |
| `scalar()` | `Output.scalar()` | node kind `"scalar"` | same, after checking shape `()` |
| `with_values(arr)` | `Output.with_values(arr)` | node kind `"with_values"`, kids `(template, values)` | the template's times, dims and map with new values; shape must equal the template's |

Array-valued nodes compose with numbers, other arrays, `Param` expressions and NumPy
ufuncs through the existing `"unary"` / `"binary"` nodes, which already call
`apply_unary` / `apply_binary` when neither side is an `Output`. An array combined
with an `Output` broadcasts positionally (NumPy rules on the raw values), as an array
constant does today.

`defer(f)(...)` with no `Output` argument now returns `f`'s value as is (an array
node) instead of raising, so `defer` composes with `array()`.

Array-valued named outputs are allowed in an `OutputSet`. `OutputSet.evaluate` puts
them in a new `Result.arrays` mapping (pytree children, so `jit` and `vmap` work).
`result[name]` looks in `outputs` then `arrays`; `keys()`, `in` and iteration see
both. `to_frame` rejects an array name with a clear message. A `Target` on an array
key raises and points at `with_values`.

### 4. Porting to the post-solve stage (CX30, steps 36-38)

Each new verb is a single-child (or, for `with_values`, two-child) post-op with
static arguments, the same shape as the existing `"call"` nodes:

- `at(when)` → `PostOp("at", when)`; it is in `_CALLS`, so it ports with the others.
- `array()`, `scalar()` → `PostOp("array")`, `PostOp("scalar")` whose value type is
  an array rather than an `Output`.
- `with_values(v)` → `PostOp("with_values")` over `(template, v)`.

The broadcasting rule lives in `Output._combine_outputs`, which the post stage keeps
as its evaluation kernel, so it ports unchanged. Step 36's plan must list these four
verbs; a note goes into the step's *What the previous worker left you* only if this
branch merges while step 36 is in progress (it is not today).

## Files

- `src/summer4/results/output.py` — `_combine_outputs` time rule; `at_times` message;
  `Output.array`, `scalar`, `with_values`.
- `src/summer4/results/outputset.py` — `_CALLS` gains `at`; `OutputExpr.at`, `array`,
  `scalar`, `with_values`; new node kinds; `evaluate` keeps array values;
  `_apply_deferred` returns raw values when no `Output` argument.
- `src/summer4/results/plan.py` — `_OutputExprChain.at`, `array`, `scalar`,
  `with_values`.
- `src/summer4/results/result.py` — `Result.arrays`, lookup, pytree, `to_frame` guard.
- `src/summer4/results/targets.py` — clear errors for time-free and array keys.
- `tests/test_time_free_outputs.py` — new.
- `examples/notebooks/33-time-free-outputs.ipynb` — acceptance notebook.
- `docs/evaluation/composability.md` — CX39, CX40 (`done`).
- `futureplans/output-peak-reduction.md`, `futureplans/posterior-runs-non-series-outputs.md`.
- `CHANGELOG.md`.

## Tests (`tests/test_time_free_outputs.py`)

1. Eager `rate / rate.integrate()` on a hand-built `Output` equals the hand values
   (trapezoid total computed in NumPy); result `times` is the series axis.
2. Lazy `outputs.ref("rate") / outputs.ref("rate").integrate()` on a solved model
   equals the eager result.
3. `rate - rate.at(t)`, eager and lazy; `at` from the save-spec chain.
4. `per_age / per_age.integrate().sum_by()` keeps the `PropertyMap`;
   `per_age.integrate() / per_age.integrate().sum_by()` sums to one.
5. Two timed outputs on different grids still raise.
6. `at_times` on a time-free output raises the new message; a `Target` on a
   time-free output raises with its key.
7. `array()` / `scalar()` eager and lazy; `scalar()` on a non-scalar raises;
   `array()` composes with `Param`, numbers and `np.exp`; array-valued named outputs
   land in `Result.arrays`; `result[name]`, `keys()` see them; `to_frame` refuses.
8. `with_values` round trip equals the dimensioned path; wrong shape raises; an
   `Output` passed as values raises.
9. A `BayesianModel` target on a `with_values` output: `log_likelihood` equals a
   hand-computed Normal log density.
10. `defer` over arrays only returns the array.
11. Everything above under `jax.jit` (evaluate inside jit).
12. `jax.make_jaxpr`: the equation count of `rate / rate.integrate()` and of
    `array()` / `with_values` does not grow with trajectory length (50 vs 500 saves),
    and `array()` adds no equations.

## Acceptance notebook (`33-time-free-outputs.ipynb`)

An age-stratified SIR (children mix more, so the age shares differ from the
population shares and the epidemic peaks at different times by age). Sections:

1. Share of the whole-period total each week, lazily: `weekly / total` where
   `total` is time-free. Plot weekly share (bars) and the cumulative share curve that
   reaches 1 at the end. *What to check*: the curve ends at 1; the bars sum to 1.
2. Per-age attack share: `by_age.integrate() / by_age.integrate().sum_by()` against
   population share. *What to check*: children's infection share exceeds their
   population share.
3. Difference from a reference week: `prevalence - prevalence.at(t_ref)`.
4. The array escape: `share_arr = weekly.array() / total.scalar()` then
   `weekly.with_values(share_arr)`; plot it over the dimensioned result.
   *What to check*: the two lines coincide.
5. Under `jit`, and a jaxpr size plot: equation count against trajectory length for
   the broadcasting share. *What to check*: flat.

## Checks

`pixi run lint format-check check-notebooks check-branch coverage roadmap
composability`, `pixi run composability-write`; the full `pixi run test` and
`pixi run -e docs docs-strict` on the mini via `mini build`.
