Implements Step 32 of `unified-lazy-idiom.plan.md`.

Closes `CX27`.

Before this step, on the parent branch (`fix/likelihood-scale-expr`):

- `Param("s") * outputs.ref("x")` raised `TypeError: Cannot use OutputExpr as a flow rate.` from `as_rate`.
- `2.0 * outputs.ref("x") * Param("s")` already worked, because `OutputExpr.__mul__` coerces the `Param`.
- `np.array(2.0) * outputs.ref("x")` already worked, through `OutputExpr.__rmul__`.
- `np.exp(outputs.ref("x"))` raised a ufunc `TypeError`: `OutputExpr` had no `exp` method and no `__array_ufunc__`.

Deviations:

- The plan's `rates.py` note suggested detecting `OutputExpr` by a class-name check or a cached import. `_is_output_expr` uses a local import, as the step's brief specifies. The cost is a `sys.modules` lookup per call, paid only when an expression is built. No jitted code reaches it.
- `algebra._apply_mode` calls the private helpers `outputset._coerce` and `outputset._binary`, rather than adding a public constructor. Their only callers are the dispatch bodies and the `rates` helpers.
- `RateOps.__array_function__` is unchanged. `np.clip(Param("s"), 0.0, outputs.ref("x"))` still works: the rate-side hook calls `rates.clip`, which now builds an `OutputExpr` when a bound is one.
- `OutputExpr.__array_priority__` is a `ClassVar[int]`, so the `slots=True` dataclass does not treat it as a field.
- Comparison ufuncs on an `OutputExpr` (`np.greater(ref, 0.0)`) are still refused, by `resolve_op` through `DENY_OPS`, with `ValueError`. They are not `TypeError`.
- Notebook 26 section B reuses the `incidence` output defined in the model cell and does not redefine it. Redefining it would only re-assign the same expression.
- Notebook 26: the opening table gains row B, and the sentence about later sections now reads "Sections C–E are added by later steps."
- Save specs (`Compartments`, `FlowMass`, ...) still have no `__array_ufunc__`. `np.tanh(FlowMass(...))` stays unsupported until step 38, as the plan says.
