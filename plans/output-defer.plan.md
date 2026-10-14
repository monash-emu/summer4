Implements Step 33 of `unified-lazy-idiom.plan.md`.

## Deviations

- Section C of notebook 26 defines its own `OutputSet` with the `incidence` leaf repeated, rather than adding `notifications` to the section A/B set. The section then runs on its own and does not need `delay` in the params of earlier sections.
- The notebook's "area" claim is worded as "close to the incidence sum (within about 4%)", not "about the same". The kernel sums to 1, but incidence is still rising at the last week, so mass delayed past the end of the series is lost. The "What to check" box says so.
- Section B's opening drops the `tanh` mention rather than adding a `tanh` line, as the step brief allowed.
- The eval logic for a `defer` node is a module-level helper, `_apply_deferred` in `outputset.py`, not inline in `eval_node`.
- Four tests were added beyond the brief's list:
  - `test_direct_defer_node_still_rejects_an_output`: a direct `Defer(f, outputs.ref(...))` still raises (the docstring claims this);
  - `test_longer_delay_moves_the_peak_later_lower_and_keeps_the_area`: a synthetic pulse checks the peak and area claim behind the notebook's section C figure;
  - `test_array_return_without_an_output_argument_raises_type_error`: the no-Output `TypeError`;
  - `test_mixed_time_and_output_arguments_raise_value_error_at_evaluate`: the mixed `Time()`-plus-output `ValueError`.
- `defer()` returns `Callable[..., Any]`, not the master plan's `Callable[..., Defer | OutputExpr]`. Which node it builds depends on the arguments, and a union return breaks every documented spelling under mypy `--strict`. With `Defer | OutputExpr` there were four errors: `outputs["notified"] = defer(f)(outputs.ref("incidence"), Param("delay"))` fails because `OutputSet.__setitem__` takes `Quantity | OutputExpr`, not `Defer`; `eval_closed(defer(f)(Param("x")), params)` fails because `eval_closed` expects `RateOps`; and `r: RateOps = defer(f)(Time())` and `defer(f)(Time()).fn` fail, although both type-checked before this step. The return stays `Any` until steps 36–38 merge the two expression languages. Inside, `wrapped` still returns `Defer | OutputExpr`. `OutputExpr` is imported under `TYPE_CHECKING` in `flows/rates.py` for that annotation, so there is no runtime import cycle.
- `_apply_deferred` also raises `TypeError` when `fn` returns a non-`Output` and none of its arguments is an `Output`, so there is no shape to wrap the result in. The master plan did not specify this. Every public spelling that builds a `defer` node (a save spec, an `outputs.ref`, or an expression built on one) evaluates to an `Output`, so the branch is reached only by passing a parameter-only `OutputExpr` built directly (`OutputExpr(kind="param", param=...)`). `tests/test_output_defer.py` covers it that way.
