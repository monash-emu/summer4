# Sharp edges a modeller hits writing the guide

Found while writing `docs/guide/01`–`08` on step 48 (`feat/clean-surface`). Each one either made
a page longer than it should be or needed a sentence of explanation. None of them is fixed on that
branch. Judge each fix against the non-programmer on-ramp: what is the shortest spelling for
someone who is not a programmer?

## Seeing values

1. **A rate cannot be plotted over time.** There is no public way to save a rate expression as an
   output. `Capture` is not exported, and `GroupedOutput` only works on a force-of-infection
   capture. Page 03 plots the user's own `defer` function directly instead. Roadmap step 40
   (save expressions) is the fix; check page 03 when it lands.
2. **Getting a named output takes two calls.** `cm.run(..., save=outputs.plan())` must be followed
   by `outputs.evaluate(result, params)`, and both must be passed the same arguments. Adding an
   output to an evaluated `OutputSet` does nothing until the model is run again (pages 04 and 08).
3. **Frames and plots are awkward.**
   - A one-column `Output.to_pandas()` has a column called `value`. Plotly then labels the axis
     `_value` and ignores `labels={"value": ...}`.
   - Compartment columns are named `state=S`, and so on.
   - `to_frame(shape="wide")` raises when outputs are on different time grids. This includes
     `integrate_intervals()` and `cumsum()`, which have one row fewer.
4. **Values come back wrapped.**
   - `cm.initial_state(params)` returns a `PropertyData`; `.data` gets the array.
   - `find_map`'s `best_params` are numpy scalars.
5. **Builders print noisily.** `model.add_flow(...)` and `set_initial_population(...)` return
   objects whose reprs print when they are the last line of a notebook cell.

## Naming and defaults

6. **Flow outputs need `side=`.** `sum_by(prop)` on a flow output raises unless
   `side="source"` or `side="dest"` is given. Also, `sum_by(age)` reads as "keep age", which
   surprises people who expect "sum over age".
7. **`Poisson()` defaults to the mean.** It uses `aggregate="mean"`, so the default
   log-likelihood is on a different scale from the summed one a modeller expects.
8. **Time-varying helpers are hidden.**
   - `TableInterp` needs `over=Property`.
   - A plain single series needs `summer4.data.Data(times, values).interp()` or
     `summer4.timevarying`, and neither is exported from `summer4`.
9. **`set_initial_population(base, splits)` does not take an `InitialPopulation`.** Callers
   unpack `initial.base` and `initial.splits`.
10. **`EntryFlow` is a count, not a rate.** Its rate is people per day. Births proportional to
    the population need `FlowRef` and `sum_by`, which the guide does not show.
11. **`defer` functions must use `jax.numpy`.** Branching with `np.where` or `if` fails inside a
    deferred function, so it needs `jnp.where`. Page 03 says so.
12. **`TraitChain` ageing reads oddly.** It is written as an `age.present()` → `age.present()`
    transition with rate 1.0, and the real rates sit inside the chain.
13. **`Multiply(where=...)` applies by source stratum.** On an infection flow it therefore
    changes susceptibility. This is correct, but the guide has to say so.

## Tooling

14. **No `nbconvert` in the docs environment.** `jupyter nbconvert --execute` fails there, so
    workers executed notebooks with `nbclient`. Add `nbconvert` to the `docs` feature, or document
    the `nbclient` route in `docs/dev/documentation.md`.

**Done when** each item is fixed, or explicitly accepted as a known wart, and the guide page that
works around it is simplified.
