Implements Step 33 of `unified-lazy-idiom.plan.md`.

## Deviations

- Section C of notebook 26 defines its own `OutputSet` with the `incidence` leaf repeated, rather than adding `notifications` to the section A/B set. The section then runs on its own and does not need `delay` in the params of earlier sections.
- The notebook's "area" claim is worded as "close to the incidence sum (within about 4%)", not "about the same". The kernel sums to 1, but incidence is still rising at the last week, so mass delayed past the end of the series is lost. The "What to check" box says so.
- Section B's opening drops the `tanh` mention rather than adding a `tanh` line, as the step brief allowed.
- The eval logic for a `defer` node is a module-level helper, `_apply_deferred` in `outputset.py`, not inline in `eval_node`.
- Two tests were added beyond the brief's list: a direct `Defer(f, outputs.ref(...))` still raises (the docstring claims this), and a synthetic pulse checks the peak and area claim behind the notebook's section C figure.
- `defer()` is annotated `-> Callable[..., Defer | OutputExpr]`, and `wrapped` returns `Defer | OutputExpr`, as the master plan says. `OutputExpr` is imported under `TYPE_CHECKING` in `flows/rates.py`, so there is no runtime import cycle (`results` imports `flows` at module level, and `outputset` is still imported lazily inside `wrapped`). `from __future__ import annotations` keeps the annotation unevaluated at runtime. `pixi run lint` (mypy `--strict`) passes.
- `_apply_deferred` also raises `TypeError` when `fn` returns a non-`Output` and none of its arguments is an `Output`, so there is no shape to wrap the result in. The master plan did not specify this. Every public spelling that builds a `defer` node (a save spec, an `outputs.ref`, or an expression built on one) evaluates to an `Output`, so the branch is reached only by passing a parameter-only `OutputExpr` built directly (`OutputExpr(kind="param", param=...)`). `tests/test_output_defer.py` covers it that way.
