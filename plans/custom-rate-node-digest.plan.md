# Custom rate node digest collision

## Problem

`_rate_bytes` (`src/summer4/flows/rates.py`) matched the built-in node types
and, for anything else, tried `expr.__rate_bytes__()` before falling back to
`type(expr).__name__.encode()`. That fallback ignored every field on the node.

`CompiledModel.__hash__` and `__eq__` are both its `_digest`
(`src/summer4/flows/compiled.py`), and the model is passed to `jax.jit` as a
static argument. So two models that differed *only* in a custom node's field
values were equal, hashed equal, and **shared a compiled program** — the second
model silently returned the first model's numerics.

Reproduced on `4e52277` with a frozen-dataclass `RateOps` subclass
`Scaled(inner, k)` registered via `register_rate_eval` and no `__rate_bytes__`:
unjitted vector fields correctly gave `-1.0` and `-7.0`, but under
`jax.jit(..., static_argnums=0)` both gave `-1.0`.

This is a wrong-answer bug with no symptom. It hits an extension author who
writes a node from the `register_rate_eval` docstring alone; anyone who copies
`ForceOfPredation` from `docs/cookbook/01-custom-rates.ipynb` is fine, because
that example implements the dunder.

The same identity keying exists for `Transform.fn` and `derived_fn` (`id()` in
the digest), but those fail *safe*: a fresh function object is a new id, so the
failure mode is a redundant recompile. The custom-node fallback was the one
direction that failed unsafe.

## Approach

1. **Enforce at registration.** `register_rate_eval(cls)` raises `TypeError`
   when `cls` has no callable `__rate_bytes__`, naming the class and the method.
   The failure moves from wrong numbers at run time to an import-time error at
   the class definition, and nothing is added to `_RATE_EVALUATORS`.
2. **Make `_rate_bytes` total.** Its `case _` arm still prefers
   `__rate_bytes__`, but raises `TypeError` instead of falling back to the class
   name — the backstop for a node that reaches evaluation without going through
   registration.
3. Document the requirement in the `register_rate_eval` docstring, in the
   cookbook appendix, and in `docs/dev/rate-expressions.md` (§ What it costs,
   § Could we have a better compute graph?), which argued for exactly this.

Deliberately *not* done: deriving a default digest from the dataclass fields.
A node's fields can be arbitrary objects with no stable encoding, and a
plausible-looking automatic digest would reintroduce the same class of silent
error. Requiring the author to say what identifies the node is the point.

## Exit checks

- `summer4.epi.ForceOfInfection` still registers; it implements the dunder.
- `docs/cookbook/01-custom-rates.ipynb` still executes — `ForceOfPredation`
  implements the dunder.
- `tests/test_custom_rate_nodes.py`: two models differing only in a custom
  node's fields compare unequal, hash unequal, and return their own numerics
  under `jax.jit(..., static_argnums=0)`; registration and `_rate_bytes` both
  raise for a node without the dunder.
- `examples/notebooks/14-custom-rate-nodes.ipynb` is the user gate: two
  shedding rates through one jitted function, a sweep showing one cache entry
  per model, and the refused registration.

Promoted from `futureplans/custom-rate-node-digest-collision.md`, which this
branch removes.
