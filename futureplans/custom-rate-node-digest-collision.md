# Custom rate nodes without `__rate_bytes__` silently share a jit cache entry

**Status:** open, unscheduled. Reproduced on `4e52277`.
**Where:** `_rate_bytes` fallback, `src/summer4/flows/rates.py:879-886`;
`_model_digest` / `CompiledModel.__hash__` / `__eq__`,
`src/summer4/flows/compiled.py`.

## What is wrong today

`_rate_bytes` matches the built-in node types and, for anything else, tries
`expr.__rate_bytes__()` before falling back to `type(expr).__name__.encode()`.
The fallback ignores every field on the node.

`CompiledModel.__hash__` and `__eq__` are both its `_digest`, and the model is
passed to `jax.jit` as a static argument. So two models that differ *only* in a
custom node's field values are equal, hash equal, and **share a compiled
program**. The second model silently returns the first model's numerics.

The code comment at that fallback already says the behaviour "is unsafe;
require `__rate_bytes__` when registered evaluators exist" — but nothing
enforces it.

## Reproduction

A `RateOps` subclass registered via `register_rate_eval` that omits
`__rate_bytes__`:

```python
@dataclass(frozen=True, slots=True)
class Scaled(RateOps):
    inner: RateOps
    k: float

@register_rate_eval(Scaled)
def _eval(expr, *, eval_child, **_):
    return expr.k * eval_child(expr.inner)

a = build(Scaled(0.1, k=1.0))   # compiled model
b = build(Scaled(0.1, k=7.0))
```

Unjitted, `a.vector_field` and `b.vector_field` differ correctly
(`-1.0` vs `-7.0`). Under `jax.jit(..., static_argnums=0)` both return `-1.0`:
`a == b` is `True`, so `b` hits `a`'s cache entry.

## Why it hurts

This is a wrong-answer bug, not a performance bug, and it has no symptom. The
affected user is an extension author following
`docs/cookbook/01-custom-rates.ipynb` — the cookbook's own `ForceOfPredation`
example implements `__rate_bytes__`, so anyone who copies it is fine, and
anyone who writes their own node from the `register_rate_eval` docstring alone
is not.

The same identity-keyed weakness exists for `Transform.fn` and `derived_fn`
(`id()` in the digest), but those fail *safe*: a fresh function object is a new
id, so the failure mode is a redundant recompile, not a wrong result. The
custom-node fallback is the one direction that fails unsafe.

## Done when

`_rate_bytes` is **total**: a node type that reaches the `case _` arm without a
callable `__rate_bytes__` raises `TypeError` naming the class and the dunder it
must implement. Registration is the natural enforcement point — `register_rate_eval`
can reject a class that does not define `__rate_bytes__`, which moves the error
from "wrong numbers at run time" to "import-time failure at the definition".

A regression test should assert that two models differing only in a custom
node's fields compare unequal, and that a node without the dunder fails loudly.

Related: `docs/dev/rate-expressions.md` (§ What it costs) argues the wider point
— six silent-default dunders on the extension protocol is the real cost of the
typed-node design, and this is the one that is actively unsafe.
