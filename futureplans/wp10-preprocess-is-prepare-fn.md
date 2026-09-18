# WP10 preprocess should be prepare_fn

`plans/tb-ports-feature-completeness.plan.md` WP10.2 sketches
`BayesianModel(preprocess=...)` as a second run-start hook. WP3 already ships
`CompiledModel.prepare_fn` / `Prepared` for exactly that role (derived
constants, interpolator knots, yearly mixing stacks).

**Today:** the WP10 sketch still names a separate `preprocess`.

**Done when:** WP10 uses `prepare_fn` (possibly aliased) instead of inventing a
parallel mechanism; Kiribati's yearly mixing stack is documented as a
`prepare_fn` example.
