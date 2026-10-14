Implements Step 33 of `unified-lazy-idiom.plan.md`.

## Deviations

- Section C of notebook 26 defines its own `OutputSet` with the `incidence` leaf repeated, rather than adding `notifications` to the section A/B set. The section then runs on its own and does not need `delay` in the params of earlier sections.
- The notebook's "area" claim is worded as "close to the incidence sum (within about 4%)", not "about the same". The kernel sums to 1, but incidence is still rising at the last week, so mass delayed past the end of the series is lost. The "What to check" box says so.
- Section B's opening drops the `tanh` mention rather than adding a `tanh` line, as the step brief allowed.
- The eval logic for a `defer` node is a module-level helper, `_apply_deferred` in `outputset.py`, not inline in `eval_node`.
- Two tests were added beyond the brief's list: a direct `Defer(f, outputs.ref(...))` still raises (the docstring claims this), and a synthetic pulse checks the peak and area claim behind the notebook's section C figure.
