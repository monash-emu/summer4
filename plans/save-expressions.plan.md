# Save any expression; delete `derived_fn` (steps 40 and 41, standalone)

Roadmap steps 40 and 41 on `feat/save-expressions`, cut from step 51 (`feat/output-requests`).
Closes `CX32` and `CX31`.

## Why standalone

`plans/unified-lazy-idiom.plan.md` puts steps 40 and 41 after steps 36-39: the stage lattice
(36), post-stage `RateOps` nodes (37), `OutputSet` on rate nodes (38), and shared nodes plus
`field_refs` (39). None of those had landed. The user chose (2026-11-19) to do 40 and 41 now
on today's `OutputExpr`, pulling in only the part of step 39 that step 41 needs. That plan is
history and is not edited; the roadmap's corrections table records what this changes for steps
36-39.

The roadmap correction of 2026-10-30 applies: **delete outright, no deprecation cycle**.

## Part A: shared nodes and `field_refs` (the part of step 39 that 41 needs)

1. **Per-call memo.** `_eval_rate` takes `memo: dict[int, tuple[RateOps, Any]] | None`. A
   non-leaf node's value is stored under `id(node)` with the node itself, and a hit requires
   `memo[id][0] is node` (so a node built and freed during evaluation can never alias another).
   The hoist-slot lookup still runs first. `observe` makes one memo per vector-field call and
   threads it through `_eval_aligned`, `_apply_adjustments`, the dynamic-split evaluator and
   `SaveContext.evaluate`. `prepare` uses one memo across all hoist entries.
2. **Leaves.** `Params()` (run stage) evaluates to the prepared parameters, the object
   `derived_fn` received as its first argument. `StateArray()` (step stage) evaluates to the raw
   compartment array, what `derived_fn` received as `y=`. Both are in `_is_leaf` (never slotted
   alone) and have `__rate_bytes__`.
3. **`Item(arg, key)`.** Reads `arg`'s value by `getattr` (a str key on a NamedTuple or object)
   or `[key]`. Its stage is `arg`'s; its child is `arg`.
4. **`field_refs(expr, Schema) -> Schema`.** A NamedTuple of nodes named after `Schema`'s fields,
   nested schemas becoming nested trees. With `expr` a `Params()`, the nodes are the
   `FieldRef(path)`s `derived_refs(Schema)` built, so typed parameter access is unchanged and
   hoists exactly as before. Any other `expr` gives `Item` chains on it.

The replacement for `compile(derived_fn=f)` plus `derived_refs(Schema)` is therefore:

```python
D = field_refs(defer(f)(Params(), y=StateArray(), t=Time()), Schema)
```

With the memo, `f` runs once per vector-field call however many rates read `D`, as `derived_fn`
did, and parameter-only subtrees elsewhere in the model still hoist.

## Part B: delete `derived_fn` (step 41)

Delete `FlowModel.compile(derived_fn=)`, `CompiledModel.derived_fn`, `computed_paths`,
`DerivedFn`, `_eval_derived`, `derived_refs`, `derived_return_schema`,
`validate_computed_path`, `_validate_plan_computed`, rule R3 and the `params_are_static`
keyword on `rate_stage`, `build_hoist_table` and `Defer.__rate_stage__` (a `FieldRef` is always
run stage). `__rate_stage__` hooks are called with no arguments. The `derived` keyword that
registered evaluators receive is kept and now always holds the prepared parameters.

The `EVERYTHING` plan no longer saves every parameter path as a `ComputedValue`; it saves
compartments, every flow and every capture.

## Part C: save any expression (step 40)

1. **`SavedExpr(expr: RateOps, over: Property | None = None)`** in `results/plan.py`, a save
   quantity. Equality and hash use `_rate_bytes(expr)` and `over`'s name. `_quantity_bytes`
   encodes the same. `SavePlan.flow_reads()` includes `_flow_refs(expr)`.
2. **`SaveContext.evaluate`**: `Callable[[RateOps], Any] | None = None`. `observe` sets it to a
   closure over the same `t`, state, prepared parameters, flow values, captures, hoist table
   and memo, so a saved expression shared with a flow is not recomputed.
3. **Evaluation.** A scalar value is returned as is. A `GroupedRate` whose properties are
   `(over,)` becomes `PropertyData(PropertyMap.from_property(over), data)`; a whole-population
   capture (no properties) is a scalar. Any other value, or a grouping that does not match
   `over`, raises `TypeError` naming `over=` or `Compartments` / `FlowMass`.
4. **Inferring `over`.** `CompiledModel.expand` fills `over` when it is `None`, from the
   expression: `TableInterp.over`, `Reduce(group_by=)`, `FlowRef(reduce=("sum_by", p))`,
   a `group_by` property on a custom node (such as `ForceOfInfection`), through `Capture`,
   `BinOp`, `UnaryOp`, `Defer` and `Item`. `values_for` and `dims_for_quantity` give
   `("time",)` or `("time", over.name)`.
5. **`OutputSet` lifting.** Wherever an output expression takes a rate expression
   (`_coerce`, so arithmetic, `defer` arguments and `with_values`): a step-stage expression
   becomes a `SavedExpr` leaf, saved during the solve; a run-stage one stays a post-solve
   parameter. `outputs[name] = expr` accepts a rate expression: step stage is saved; run stage
   raises (`a constant is not a time series; multiply a saved quantity`). Because Python
   builds `Param("a") * Time()` before it meets an output, the largest step-stage subtrees
   are what get lifted. Requests from step 51 are unchanged.
6. **Delete `SaveFn` and `ComputedValue`** (and the `SaveFn` arm in `describe`'s error message),
   and migrate every caller.

## Migration

Every caller in `src`, `tests`, `examples/notebooks`, `docs/guide` and `docs/dev` moves to the
replacements; equivalence tests become tests of the replacement against hand-computed values.
`tests/test_docs_idiom.py` keeps banning the deleted names in the docs. Downstream repositories
(`kiribati-tb-summer4` uses `SaveFn` for workaround W2) break and move to `outputs[name] = expr`
when they upgrade; the CHANGELOG says so.

## Tests

`tests/test_save_expressions.py`:
- a seasonal beta saved by expression equals NumPy on the save times (Euler and Diffrax, and
  under `jit`);
- FOI by age saved by expression equals the `GroupedOutput(Capture)` route, with `over`
  inferred and given;
- an expression reading a `FlowRef` saves with a reduced plan (`flow_reads`);
- `Param("a") * Time() * outputs.ref("inc")` lifts only the step-stage part;
- `outputs[name] = Param("x")` raises; a per-compartment expression raises; a wrong `over`
  raises.

`tests/test_shared_nodes.py`:
- one `defer(sin)(Time())` object shared by three flows traces one `sin` in
  `jax.make_jaxpr(cm.vector_field)` (three before the memo);
- a `field_refs(defer(f)(Params(), y=StateArray(), t=Time()), S)` model equals hand-computed
  values, with `hoist` on and off;
- `Param("a") * Param("b")` beside it is in the hoist table;
- `field_refs(Params(), S)` builds the same `FieldRef`s as the old `derived_refs`.

Gate notebook `examples/notebooks/40-save-expressions.ipynb`: a saved seasonal rate, FOI by age
saved by expression against the capture route, the `derived_fn` replacement with its hoist table
and `sin` count.
