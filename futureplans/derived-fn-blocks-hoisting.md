# derived_fn blocks parameter hoisting (R3)

When `FlowModel.compile(derived_fn=...)` is set, every `FieldRef` is
**step-stage** because it reads `derived_fn` output. That disables automatic
hoisting of parameter-only rate subtrees (`Param("a") * Param("b")`, Interp
knot stacks of FieldRefs, etc.) into the run-start stage.

**Today:** users must put `t`/`y`-independent work in `prepare_fn` (or accept
re-evaluating it every vector-field call).

**Done when:** either (a) `derived_fn` is split into run-start and per-step
halves, or (b) the user can declare a static path through derived params so
those FieldRefs remain hoistable.

See `docs/dev/run-stages.md` (R3) and `summer4.flows.stages.rate_stage`.
