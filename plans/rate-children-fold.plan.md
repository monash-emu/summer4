Implements Step 35 of `unified-lazy-idiom.plan.md`.

Deviations:

- **Corpus size.** The characterisation corpus in `tests/test_rate_children.py` has 28
  expressions, not "about 15". The extras are three FOI variants, plus a FOI wrapped in each
  container type (the `foi_*` cases). They are there so that every fix shows up as a changed
  expectation.
- **A fifth `# fixed in step 35` line.** The capture meta of `foi_in_capture` is outside the three
  categories in plan item 9. It follows from plan item 4: `_collect_capture_meta` now folds over
  `rate_children`, so `Capture` folds over its `inner`. The orchestrator approved it.
- **Section E content differs from plan item 12.** The plan asked for "a plot of FOI and
  incidence against a hand-ordered reference model". Notebook 26 section E instead plots
  incidence with `hoist=True`, `hoist=False` and `vax_effect=0`. The comparison with a
  hand-ordered reference is in
  `tests/test_rate_children.py::test_foi_flow_ref_trajectory_matches_hand_ordered_reference`.
- **User-visible behaviour change for custom nodes.** A node that defines `__rate_children__` but
  not `__flow_refs__` / `__field_paths__` now reports its children's flow refs and field paths.
  Before, it reported none. Capture meta now also folds through `__rate_children__`.
- **Defer's redundant dunders kept.** `__flow_refs__`, `__field_paths__` and
  `__capture_children__` on `Defer` (`src/summer4/flows/rates.py`) are now redundant with
  `rate_children`. They are left in place.
- **A smaller FOI hoist-equivalence test** than the `test_hoisting.py:138` pattern:
  `test_foi_hoist_on_and_off_agree_and_slot_the_parameter_children` runs euler only, with no
  tsit5 run and no gradient check.
- **`_flow_refs` now also walks `Lookup.table`.** This has no effect, because `Lookup` accepts
  only a `FieldRef` there.
- **Custom-node headline tests, added after review.** `tests/test_custom_rate_nodes.py` gains a
  `Halved` node that defines only `__rate_children__` and `__rate_bytes__`, plus a
  `register_rate_eval` evaluator, with one rate-expression child.
  `test_rate_children_alone_orders_flow_refs_and_finds_params` checks that a `FlowRef` inside the
  child orders `vaccination` before `infection`, that the child's `Param`s reach
  `CompiledModel.computed_paths`, and that the vector field matches a hand-computed value.
  `test_rate_children_alone_hoists_and_matches_unhoisted` checks that the child's parameter-only
  `BinOp` is slotted, and that the vector fields with hoisting on and off agree.
