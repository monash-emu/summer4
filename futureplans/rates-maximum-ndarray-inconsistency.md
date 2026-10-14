# `rates.maximum` and `np.maximum` disagree on ndarray operands

`summer4.maximum` / `minimum` / `clip` (`src/summer4/flows/rates.py`, the
`_binary` path) treat an `np.ndarray` as an evaluated value (`_is_value`), so
`maximum(Param("a"), np.array([...]))` raises the mixed-expression `TypeError`.
`np.maximum(Param("a"), np.array([...]))` goes through `dispatch_ufunc` and
`operands_are_mixed` (`src/summer4/flows/algebra.py`), which does not count an
ndarray as evaluated, so it builds a `BinOp` with an `ArrayConst`.

The two spellings are documented as aliases (`docs/dev/rate-expressions.md`,
*The operator surface*), so one of them is wrong. Found while planning
`plans/unified-lazy-idiom.plan.md`; out of scope there.

**Done when:** both spellings agree (most likely: an ndarray beside a rate
expression becomes an `ArrayConst`, as `np.maximum` does), with a test pairing
each named helper with its `np.*` spelling on an ndarray operand.
