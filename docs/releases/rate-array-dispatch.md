# NumPy ufuncs on rate expressions

**Status:** shipped in {doc}`v0.2.0a4` (`feat/rate-array-dispatch`).

`np.sin(Param("phase"))` builds a rate node. `jnp.sin` still does not: JAX
implements no dispatch hook, so the spelling is `np.*`.

## Behaviour change

`np.array([1.0, 2.0]) * Param("x")` used to return an object-dtype ndarray of
one `BinOp` per element. It now builds a single node:

```python
BinOp("mul", ArrayConst([1.0, 2.0]), FieldRef(("x",)))
```

NumPy scalars are unchanged: `np.float64(2.0) * Param("x")` is still
`BinOp("mul", Const(2.0), FieldRef(("x",)))`.

`np.multiply(a, b)` and `a * b` digest identically. The op string stays
summer4's short name (`mul`, not `multiply`) so existing jit cache keys do not
split.
