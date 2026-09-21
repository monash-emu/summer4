---
name: rate-dispatch-and-defer
description: Open the rate-expression operator set via NumPy's dispatch protocols, and add a first-class Defer node so arbitrary user code can be a flow rate — steps 22 and 23 of docs/dev/roadmap.md.
---

# WP18 — array dispatch and deferred callables on rate expressions

## Context

Two complaints about `summer4.flows.rates`, both raised in review, both
answered by the same underlying observation. The full argument is
[`docs/dev/rate-expressions.md`](../docs/dev/rate-expressions.md) — read it, but this section carries enough
to work from.

**Complaint 1 — "why the hand-written arithmetic helpers?"** `summer4` exports
`exp`, `log`, `tanh`, `sqrt`, `floor`, `maximum`, `minimum` and `clip`. They
look like duplicates of `jnp`. They are not: you cannot write `jnp` here.
`jnp.exp(Param("beta"))` raises `TypeError`, because `jnp.exp` is a
`PjitFunction` rather than a `numpy.ufunc` and JAX implements **neither**
`__array_ufunc__` nor `__array_function__`. Operators (`Param("b") * 2`) work
only because Python routes them through `__mul__`. So the helpers are the only
way to get a non-operator op into a rate tree.

What they get wrong is the *shape*. The helper set is a hand-maintained
allowlist: `tanh` is in, `sin` is not, and seasonal forcing is a first-order
epidemiological need. `clip` is sugar over `maximum` + `minimum` and owns no
node. Adding `sin` today means editing `UNARY_OPS`, the `Literal[...]`
annotation on `UnaryOp.op`, a new export, and the docs.

summer2 had no such list because `computegraph.GraphObject` implements
`__array_ufunc__` and `__array_function__`, so `np.exp(obj)` built a node
automatically. That worked because summer2 was **NumPy-first at the API
boundary** — `jaxify.get_modules()` swapped `jnp` in underneath.

**Complaint 2 — "what replaces `defer()`?"** summer2 shipped a two-line helper:

```python
def defer(func):
    def wrapped_maker(*args, **kwargs):
        return Function(func, args, kwargs)
    return wrapped_maker
```

A modeller writes an ordinary Python function, wraps it, calls it with a mix of
parameters and literals. No class, no registration. For users who are not
software developers this was the main door into the graph. summer4 has the
capability three times over and no comparable door:

- `Transform` **cannot occupy the rate slot** —
  `TransitionFlow(..., rate=Transform(...))` raises
  `TypeError: Cannot use Transform as a flow rate`. The working incantation is
  a dummy `rate=1.0` plus a callable that discards its first argument.
- `Transform` is documented *only* as an adjustment precedence level
  (`Overwrite` → `Multiply` → `Transform`) in `docs/user/08-flows.ipynb` and
  `docs/user/07-from-summer2.md`. No page says "put arbitrary code here".
- `derived_fn` needs a `NamedTuple` schema and, per
  [`docs/dev/run-stages.md`](../docs/dev/run-stages.md) R3, disables parameter hoisting model-wide.
- `docs/cookbook/01-custom-rates.ipynb` goes `Reduce` arithmetic → FOI-specific
  `kind=` callable → subclass `RateOps`. Nothing in between.

**Why a generic node does not cost what it looks like it costs.** A
`Defer(fn, *args)` takes its dependencies as **explicit arguments**, so its
stage is exactly the join of its arguments' stages. This was prototyped:
`defer(f)(Param("x"), Param("y"))` classifies `run` and hoists;
`defer(f)(Time(), Param("amp"))` classifies `step`. The thing typed nodes
uniquely buy is **sub-node** hoisting — `Interp` hoists its breakpoint stack
and value stack independently of its argument, where a `Defer` is
all-or-nothing. That is a reason to keep `Interp`, not to withhold `Defer`.

**Closes (API ledger):** no rows move status. Note that row **`P2`
(`Function`, parameters)** is already `full` with the route
`Transform / callables in derived_fn`. Given that `Transform` cannot be a rate,
that route note is misleading; step 23 corrects the note (see its *Ledger*
subsection). Do **not** change `P2`'s status — it would move the progression
totals and it is not what this work is about. **Closes (ports ledger):**
nothing. **Declared as:** `WP18` in the coverage ledger's packages block.

## Ordering

The two steps are **independent of each other and of WP13–WP17**. Either order
works; the roadmap runs dispatch (step 22) first because it touches shared
machinery (`flows/algebra.py`, `flows/rates.py`) that step 23 then builds on
unchanged.

They are independent of steps 8–21 and may run at any time from `main`.

**One real dependency, in step 23.** A separate piece of work makes
`_rate_bytes` **total** — raising for a registered `RateOps` subclass that does
not implement `__rate_bytes__`, instead of silently digesting by class name
(`futureplans/custom-rate-node-digest-collision.md`). `Defer` carries a
callable, so it is exactly the kind of node that trap catches.

Check before starting step 23:

```bash
grep -n "return type(expr).__name__.encode()" src/summer4/flows/rates.py
```

- **No match** → the totality fix has landed. `register_rate_eval` may now
  reject a class without `__rate_bytes__`. `Defer` implements it, so this is
  fine; just confirm registration succeeds.
- **A match** → the fix has not landed. Implement `Defer.__rate_bytes__`
  anyway (it is in the plan below) and say in the handoff that the trap is
  still open.

Either way, **do not implement the totality fix as part of these steps.**

## Read first

Both steps, in this order:

1. `AGENTS.md` — branch naming, style, the JAX rules, the futureplans
   convention. Not optional.
2. `docs/dev/roadmap.md` — *How to run a step*, then your step's section.
3. `docs/dev/rate-expressions.md` — the whole page. It is the rationale this
   plan implements and it carries the measurements quoted here.
4. `docs/dev/run-stages.md` — the stage contract (R1–R6) and the
   `rate_stage` / `build_hoist_table` rules.
5. `src/summer4/flows/rates.py` — every node type and the six traversals.
6. `src/summer4/flows/algebra.py` — `UNARY_OPS`, `BINARY_OPS`, `apply_unary`,
   `apply_binary`, and how they preserve `GroupedRate` / `PropertyData`.

Step 22 additionally: `src/summer4/results/trace.py` (`Trace._map_unary`,
`Trace._combine`, `__array_priority__`) and
`src/summer4/jax/propertydata.py`.

Step 23 additionally: `src/summer4/flows/stages.py` (all of it),
`src/summer4/flows/compiled.py` `_eval_rate` and `_collect_capture_meta`,
`futureplans/no-defer-equivalent.md`, and
`docs/cookbook/01-custom-rates.ipynb` (the rung structure you are inserting
into).

---

## Step 22 — `feat/rate-array-dispatch`

Make `np.<ufunc>(...)` build rate nodes and operate on evaluated wrappers, so
the operator set is open instead of an allowlist.

### 22.1 What ships

- `__array_ufunc__` and `__array_function__` on **four** types: `RateOps`
  (symbolic) and `Trace`, `GroupedRate`, `PropertyData` (evaluated).
- Op identity carried as a **canonical string**, resolved to a `jnp` callable
  at evaluation — computegraph's `getattr(fnp, func_str)` trick.
- `UnaryOp.op` / `BinOp.op` widened from `Literal[...]` to `str`, with
  **construction-time validation** replacing the static check.
- Every existing named export kept, reimplemented on the new machinery.
- `examples/notebooks/16-array-dispatch.ipynb` (see 22.8 for numbering).

### 22.2 The canonical-name table — get this right first

This is the single easiest thing to get wrong, and getting it wrong silently
doubles the jit cache.

summer4's existing op strings are **not** NumPy ufunc names. Six of fourteen
differ:

| summer4 op | NumPy ufunc | |
|---|---|---|
| `add` | `add` | same |
| `sub` | `subtract` | **differs** |
| `mul` | `multiply` | **differs** |
| `div` | `divide` (and `true_divide`) | **differs** |
| `pow` | `power` | **differs** |
| `neg` | `negative` | **differs** |
| `abs` | `absolute` | **differs** |
| `maximum`, `minimum`, `exp`, `log`, `tanh`, `sqrt`, `floor` | same | same |

`_rate_bytes` encodes the op string directly (`b"binop:" + op.encode()`). If
`np.multiply(a, b)` produced `BinOp("multiply", …)` while `a * b` produces
`BinOp("mul", …)`, the two would digest differently, compile twice, and be
impossible to tell apart in a jaxpr. They are the same computation and must
produce byte-identical digests.

**Therefore:** add an alias map that canonicalises an incoming ufunc name to
summer4's existing short name where one exists, and leaves it alone otherwise.
`np.multiply(a, b)` must produce `BinOp("mul", …)`, identical to `a * b`.

```python
# in summer4/flows/algebra.py
UFUNC_ALIASES: dict[str, str] = {
    "subtract": "sub",
    "multiply": "mul",
    "divide": "div",
    "true_divide": "div",   # NumPy < 2 spells `/` this way
    "power": "pow",
    "float_power": "pow",
    "negative": "neg",
    "absolute": "abs",
    "fabs": "abs",
}
```

New ops (`sin`, `cos`, `expm1`, `sign`, …) keep their NumPy name verbatim, so
`np.sin(x)` → `UnaryOp("sin", x)`.

**Record goldens before you change anything.** These three hex digests were
captured from `main` at `f367923`; they must be byte-identical when the step
lands. Put them in the test suite as literals:

```
Param("x") * 2   62696e6f703a6d756c6669656c64282778272c29636f6e73740000000000000040
exp(Param("x"))  756e6172793a6578706669656c64282778272c29
Param("x") + Param("x")
                 62696e6f703a6164646669656c64282778272c296669656c64282778272c29
```

Reproduce with:

```python
from summer4.flows.rates import _rate_bytes
_rate_bytes(expr).hex()
```

### 22.3 Resolving an op to a callable

Replace the two hand-written dicts as the *source of truth*. Keep them as a
cache, but populate by lookup:

```python
def resolve_op(name: str) -> Callable[..., Any]:
    """Return the jnp implementation of a canonical op name."""
```

Rules, in order:

1. If `name` is in `DENY_OPS`, raise `ValueError` naming it. `DENY_OPS` exists
   because some NumPy ufuncs are not elementwise or are already spoken for:
   `matmul` (`GroupedRate.__matmul__` owns `@` and has its own grouping
   semantics), plus anything that returns a boolean or integer mask rather than
   a rate — `greater`, `less`, `equal`, `not_equal`, `greater_equal`,
   `less_equal`, `logical_and`, `logical_or`, `logical_not`, `isnan`, `isinf`,
   `isfinite`, `signbit`. Masks belong to `Selector`, not to rate arithmetic;
   admitting them would produce non-differentiable rates with no error.
2. Map through `UFUNC_ALIASES` first, then look up a small table of summer4's
   own spellings (`sub` → `jnp.subtract`, `mul` → `jnp.multiply`, `div` →
   `jnp.divide`, `pow` → `jnp.power`, `neg` → `jnp.negative`, `abs` →
   `jnp.abs`).
3. Otherwise `getattr(jnp, name)`; if absent, raise `ValueError` saying the op
   has no `jax.numpy` equivalent.

Import `jax.numpy` **lazily inside the function**, not at module import. The
taxonomy layer must stay JAX-free at import time (`AGENTS.md`, *Dependency
policy*); `algebra.py` currently imports `jnp` at module level, which is
acceptable there, but `rates.py` must not gain a module-level `jnp` import.

### 22.4 The dunders

One shared implementation, mixed into all four types. Put the generic half in
`summer4/flows/algebra.py` so `rates.py`, `trace.py`, `compiled.py`
(`GroupedRate`) and `jax/propertydata.py` all call it.

```python
def dispatch_ufunc(ufunc, method, inputs, kwargs, *, build):
    """Shared __array_ufunc__ body. `build` makes a node or applies a value."""
```

Required behaviour:

- **`method != "__call__"` → `NotImplemented`.** Rejects `reduce`,
  `accumulate`, `outer`, `reduceat`, `at`. NumPy then raises its own
  `TypeError`, which is the right outcome: `np.add.reduce(rate_expr)` has no
  meaning here.
- **`out=` present → `NotImplemented`.** Nodes are frozen; there is no in-place.
- **`where=`, `casting=`, `order=`, `dtype=` present → `NotImplemented`.**
  Silently ignoring them would be wrong, and `where=` in particular collides
  with summer4's own selector-shaped `where=`.
- **Arity other than 1 or 2 → `NotImplemented`.**
- For `RateOps`: coerce each input with `as_rate` and return
  `UnaryOp(canonical, arg)` / `BinOp(canonical, left, right)`. Validate the
  canonical name via `resolve_op` at construction so a bad op fails at build
  time, not at trace time.
- For the three evaluated types: route to `apply_unary` / `apply_binary`, which
  already preserve the wrapper. `Trace` routes to `Trace._map_unary` /
  `Trace._combine`.
- **Mixed symbolic and evaluated → raise the existing `_MIXED_EXPR`
  `TypeError`**, not `NotImplemented`. `np.multiply(trace, Param("x"))` must
  give the same error as `trace * Param("x")` does today, pointing at
  `eval_closed`.

`__array_function__` covers the non-ufunc NumPy API. Support exactly
`np.clip`, `np.maximum`, `np.minimum`, `np.power`, `np.absolute`, and return
`NotImplemented` for everything else — an open `__array_function__` would
claim the entire NumPy namespace, including reductions and reshapes that have
no meaning on a rate node. `np.clip(x, lo, hi)` composes `maximum` then
`minimum`, reusing the existing `clip` logic including its `None`-bound
handling.

### 22.5 Behaviour that must not change, and one that must

Add a test for each of these. They are the interop edges that defining
`__array_ufunc__` disturbs.

**Must not change** (verified on `f367923`):

| Expression | Must still give |
|---|---|
| `np.float64(2.0) * Param("x")` | `BinOp("mul", Const(2.0), FieldRef(("x",)))` |
| `Param("x") * np.float64(2.0)` | `BinOp("mul", FieldRef(("x",)), Const(2.0))` |
| `np.int32(3) * Param("x")` | `BinOp("mul", Const(3.0), FieldRef(("x",)))` |
| `2.0 * Param("x")` | `BinOp("mul", Const(2.0), FieldRef(("x",)))` |
| `exp(Param("x"))` | digest byte-identical to the golden above |

NumPy scalars currently route through `RateOps.__rmul__`. Once
`__array_ufunc__` exists NumPy will call it instead. The *result* must be the
same object graph, which is why these are pinned as tests rather than left to
inspection.

**Must change, deliberately.** Today `np.array([1.0, 2.0]) * Param("x")`
returns an **object-dtype ndarray of `BinOp`s** — NumPy broadcasts elementwise
because `RateOps` defines neither `__array_ufunc__` nor `__array_priority__`.
That is not useful and is almost certainly nobody's intent. After this step it
must build a single node:

```python
BinOp("mul", ArrayConst([1.0, 2.0]), FieldRef(("x",)))
```

Achieve it by having `as_rate` accept `np.ndarray` by wrapping in
`ArrayConst` — check whether `as_rate` already does; at `f367923` it raises
`TypeError: Cannot use ndarray as a flow rate`. Widening `as_rate` is in scope
for this step. Call the change out in the release note and the handoff: it is
the only user-visible behaviour difference.

### 22.6 Typing

`UnaryOp.op` and `BinOp.op` widen from `Literal[...]` to `str`. This loses a
static check, so replace it with a runtime one: validate in `__post_init__` via
`resolve_op`, raising `ValueError` with the offending name and a hint. That is
strictly better than the `Literal` was, because it also catches ops built from
runtime strings.

`mypy --strict` runs on `src/summer4` (`pixi run lint`). The dunders are
untyped in NumPy's stubs; annotate them as
`(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any` and keep
every helper fully annotated per `AGENTS.md`.

### 22.7 Tests — `tests/test_rate_dispatch.py`

1. **Digest stability.** The three goldens in 22.2, as literal hex strings.
2. **Alias canonicalisation.** `np.multiply(p, q)` and `p * q` produce equal
   nodes *and* equal `_rate_bytes`. Same for subtract, divide, power, negative,
   absolute.
3. **A genuinely new op.** `np.sin(Param("phase"))` builds
   `UnaryOp("sin", …)`; `0.1 * (1.0 + 0.5 * np.sin(Param("phase")))` compiles
   and its vector field matches `0.1 * (1 + 0.5 * sin(phase))` computed in
   NumPy.
4. **Staging is unaffected.** `rate_stage(np.sin(Param("p")), params_are_static=True)`
   is `"run"`; with `Time()` inside, `"step"`.
5. **Value-keyed digests survive.** Two separately built models using
   `np.sin(Param("phase"))` compare equal and share a jit cache entry; a
   `np.cos` model compares distinct. Use trace counting (see 23.7 for the
   pattern).
6. **All four wrapper types.** `np.exp` on a `Trace` returns a `Trace`, on a
   `GroupedRate` a `GroupedRate`, on a `PropertyData` a `PropertyData`, on a
   `RateOps` a `UnaryOp`. Assert the wrapper metadata survives — `dims` on the
   `Trace`, `properties` on the `GroupedRate`, `pmap` on the `PropertyData`.
7. **Mixed operands raise.** `np.multiply(trace, Param("x"))` raises
   `TypeError` mentioning `eval_closed`.
8. **Rejections.** `np.add.reduce(...)`, `np.exp(p, out=...)`,
   `np.greater(p, 1.0)`, `np.matmul(p, q)` each raise rather than silently
   producing something.
9. **The interop table in 22.5**, every row.
10. **`np.clip`** matches `summer4.clip` for both bounds, lower only, upper
    only; both-`None` raises.

### 22.8 Docs

- `examples/notebooks/16-array-dispatch.ipynb` — **required** by
  `pixi run check-branch`. Notebook numbering is first-come: 14 and 15 are
  claimed by planned steps 13–15 (`14-calibration`) and 17–19
  (`15-contact-matrices`). If those have not landed, take the next free
  two-digit prefix and correct the roadmap in the same commit. summer2
  documentation style: explain the claim, plot it, assert it. The story is
  seasonal forcing with `np.sin` — something that was impossible before.
- `docs/cookbook/01-custom-rates.ipynb` — a sentence in *What to reach for*
  saying any NumPy ufunc now works on a rate expression, and that `np.*` is the
  spelling (not `jnp.*`).
- `docs/dev/rate-expressions.md` — the *operator surface* section says
  "prototyped"; change it to applied, keeping the explanation. Move item 2 of
  *Where this leaves the recommendation* to done.
- `docs/user/07-from-summer2.md` — map summer2's `np.*` on a `GraphObject` to
  the same spelling here.
- `docs/releases/` — a note covering the `np.ndarray * RateOps` change.
- **State the `jnp` wart explicitly** in the notebook and the cookbook:
  `np.sin(expr)` works, `jnp.sin(expr)` does not, because JAX offers no
  dispatch hook. Users will hit this.

### 22.9 Out of scope

- The `__children__()` traversal refactor (item 3 of the recommendation).
- Making `_rate_bytes` total.
- Any grouping marker in the type system (item 4).
- `__array_function__` beyond the five named functions.

---

## Step 23 — `feat/rate-defer`

Give arbitrary user code a first-class door into the rate slot.

### 23.1 What ships

- `Defer` — a `RateOps` node carrying a callable and explicit rate-expression
  arguments.
- `defer(fn)` — the curry, exported from `summer4`.
- Descent into a custom node's children during hoisting, so a param-only
  argument inside a step-stage `Defer` still hoists.
- `examples/notebooks/17-deferred-functions.ipynb`.
- The cookbook gains the missing rung.

### 23.2 The node

```python
@dataclass(frozen=True, slots=True)
class Defer(RateOps):
    """Call ``fn`` on evaluated rate arguments. The general escape hatch."""

    fn: Callable[..., Any]
    args: tuple[RateOps, ...]
    kwargs: tuple[tuple[str, RateOps], ...]
    name: str | None = None
```

Notes a low-context worker needs:

- `kwargs` is a **tuple of pairs**, not a dict, because the node must stay
  hashable and frozen. Sort by key at construction so two equal calls build
  equal nodes regardless of keyword order.
- **Support kwargs.** computegraph's `Function` did, and the ergonomic appeal
  of `defer` is that you call the function the way you wrote it —
  `defer(f)(t=Time(), amp=Param("amp"))`. It costs one extra loop in each
  traversal.
- `name` is optional and affects **only** the digest (23.5).
- The constructor coerces every argument with `as_rate`, so literals and
  (after step 22) arrays are accepted alongside nodes.

The curry:

```python
def defer(fn: Callable[..., Any], *, name: str | None = None) -> Callable[..., Defer]:
    """Wrap ``fn`` so calling it builds a :class:`Defer` node."""
```

Both `defer(f)(Time(), Param("amp"))` and `Defer(f, Time(), Param("amp"))` must
work; the curry is the documented spelling.

### 23.3 Evaluation

```python
@register_rate_eval(Defer)
def _eval_defer(expr, *, eval_child, **_):
    args = [eval_child(a) for a in expr.args]
    kwargs = {k: eval_child(v) for k, v in expr.kwargs}
    return expr.fn(*args, **kwargs)
```

`fn` receives **traced JAX values** — arrays, or `GroupedRate` /
`PropertyData` wrappers when an argument evaluates to one. Say this in the
docstring; it is the thing users get wrong. Whatever it returns goes through
`_align_rate` like any other rate, so the existing alignment errors apply
unchanged.

### 23.4 Staging

```python
def __rate_stage__(self) -> Stage:
    """`step` if any argument is step-stage, else `run`."""
```

This is *exact*, not conservative, because every dependency is an explicit
argument. Verified in the prototype. `rate_stage` already consults
`__rate_stage__` for unknown nodes (`flows/stages.py`), so no change is needed
to `rate_stage` itself.

**One thing that does need changing.** `build_hoist_table`'s `walk` ends with:

```python
# Capture and unknown custom nodes: do not descend.
```

So a step-stage `Defer` with a param-only argument never has that argument
hoisted. Fix it by adding an optional `__rate_children__()` returning the
node's child expressions, and having `walk` descend into it for unknown nodes
that provide it. `Defer` implements it as `(*args, *[v for _, v in kwargs])`.

Keep the change minimal — this is **not** the general `__children__` refactor
(item 3 of the recommendation in `rate-expressions.md`), just the hook
`build_hoist_table` needs. Leave `Capture`'s non-descent as it is.

### 23.5 The digest, and why `id()` is correct

`__rate_bytes__` encodes `name` when given, otherwise `str(id(self.fn))`, plus
the encoded args and kwargs.

Identity keying is **not** a concession — it is the safe choice. Measured on
`f367923` by trace counting under `jax.jit(..., static_argnums=0)`:

| Pattern | Traces |
|---|---|
| One model, 50 parameter draws | 1 |
| 50 rebuilds, callable at module level | 0 |
| 50 rebuilds, callable defined inside the builder | 50 |
| One closure-built model, 50 runs | 1 |
| Two closures capturing `scale=1.0` vs `2.0` | 2, results differ |

The last row is the reason. Two closures with identical source but different
captured values are **different programs** and must not share a cache entry;
`id()` separates them for free. Keying on `fn.__qualname__` would give both
`build_closure.<locals>.cl`, collide, and silently return the first closure's
numbers — the same unsafe failure as
`futureplans/custom-rate-node-digest-collision.md`.

The cost lands on model **rebuild**, never on `run`, and calibration is
untouched because parameters are dynamic. `name=` is the opt-in for users who
rebuild in a sweep and can assert "these are the same program". Document it as
an assertion the user makes, not a performance knob — a wrong `name=` is a
wrong-answer bug.

**Do not** default `name` to anything derived from the function. The default
must be `id()`.

### 23.6 Where `Defer` may appear

- **Rate slot of every flow type** — `TransitionFlow`, `EntryFlow`, `ExitFlow`.
  This is the point of the step.
- **Adjustment values** — `Multiply(defer(f)(...))`, `Overwrite(...)`. Free,
  since those take any `RateOps`.
- **Initial population** — allowed **iff** every argument is allowed there. R2
  forbids `Time()`, `FlowRef`, `Reduce` and `Capture` in initial-population
  expressions. Since `Defer`'s dependencies are explicit, the existing check
  finds them by walking `__rate_children__` — confirm this works and test it,
  rather than adding a second rule.

`_flow_refs` and `_field_paths` must both descend into args and kwargs, via
`__flow_refs__` and `__field_paths__`. A `FlowRef` inside a `Defer` argument
must participate in `topo_sort`, or the flow will be evaluated out of order.
Test this explicitly — it is the subtlest correctness risk in the step.

Likewise `__capture_children__` so a `Capture` inside a `Defer` argument is
found by `_collect_capture_meta` and becomes saveable.

### 23.7 Tests — `tests/test_rate_defer.py`

1. **Rate slot.** `defer(f)(Time(), Param("amp"))` as a `TransitionFlow` rate
   gives the same vector field as the `rate=1.0` + `Transform` workaround.
2. **Staging.** `defer(f)(Param("x"), Param("y"))` → `run`;
   `defer(f)(Time(), Param("amp"))` → `step`.
3. **Hoisting.** A param-only `Defer` gets a slot
   (`build_hoist_table([expr], params_are_static=True).entries` has length 1).
   A step-stage `Defer` with a param-only *argument* hoists that argument.
4. **Closure distinctness.** Two closures capturing different values produce
   unequal digests and different results. This is the anti-regression test for
   the `__qualname__` trap; comment it as such.
5. **`name=` shares a cache entry.** Two rebuilds with distinct function
   objects but the same `name=` compare equal and trace once.
6. **Trace counting.** Reproduce the table in 23.5 as a test. Pattern:

   ```python
   traces = {"n": 0}

   @functools.partial(jax.jit, static_argnums=0)
   def _vf(model, y, params):
       traces["n"] += 1          # Python body runs once per trace
       return model.vector_field(5.0, y, params)
   ```

7. **kwargs.** `defer(f)(t=Time(), amp=Param("amp"))` works; keyword order does
   not change the node or its digest.
8. **`FlowRef` inside a `Defer` argument** forces the correct topological
   order; a cycle through a `Defer` raises.
9. **`Capture` inside a `Defer` argument** is saveable via `GroupedOutput`.
10. **Nested `Defer`** evaluates and stages correctly.
11. **Initial population**: a param-only `Defer` is accepted; one containing
    `Time()` raises the existing R2 error naming `Time`.
12. **Registration.** If `_rate_bytes` is total (see *Ordering*), assert
    `Defer` registers without error.

### 23.8 Docs

- `examples/notebooks/17-deferred-functions.ipynb` — required by
  `check-branch`; numbering per 22.8. Write it **for a modeller, not a package
  author**: an ordinary Python function, wrapped, used as a rate, plotted,
  asserted.
- `docs/cookbook/01-custom-rates.ipynb` — insert `defer` as the **new rung 2**,
  between `Reduce` arithmetic and the FOI `kind=` callable, renumbering the
  rungs below it. Update the *What to reach for* table: "one-off custom hazard
  that is easier to write as code" → `defer`.
- `docs/user/07-from-summer2.md` — map `computegraph.defer` → `summer4.defer`,
  and note the `name=` caveat.
- `docs/user/08-flows.ipynb` — say what `Transform` is *for* (adjusting an
  existing rate) now that `defer` is the answer for producing one.
- `docs/dev/rate-expressions.md` — the *There is no good on-ramp* section
  becomes a description of what shipped.
- **Ledger.** In `docs/evaluation/coverage-ledger.md`, row `P2`
  (`Function`, parameters) currently reads `Transform / callables in
  derived_fn`. Change the route to name `defer` first. **Leave the status
  `full`** — it does not move, and changing it would shift the progression
  totals. Run `pixi run coverage-write` then `pixi run coverage`.
- Remove `futureplans/no-defer-equivalent.md` and its entry in
  `futureplans/README.md`, per the folder's convention that a note is deleted
  when its concern lands.

### 23.9 Out of scope

- Removing or deprecating `Transform`. It keeps its job — adjusting a rate
  that already exists — and removing it would break users.
- Making `derived_fn` unnecessary. It still owns multi-output derived
  quantities and `ComputedValue` saving.
- Tracing a `Defer` to a jaxpr for structural analysis (option (c) in
  `rate-expressions.md`). Explicitly rejected there; do not attempt it.
- The `_rate_bytes` totality fix.

---

## Risks

| Risk | Where | Mitigation |
|---|---|---|
| Digest churn doubles the jit cache | 22.2 | Alias table; golden hex digests as tests |
| `np.ndarray * RateOps` behaviour change | 22.5 | Deliberate and documented; release note |
| `__array_function__` claims all of NumPy | 22.4 | Allowlist of five functions; `NotImplemented` otherwise |
| Boolean ufuncs produce silently non-differentiable rates | 22.3 | `DENY_OPS` |
| `FlowRef` inside a `Defer` breaks topological order | 23.6 | `__flow_refs__` descent, with a dedicated test |
| `name=` used carelessly gives wrong numbers | 23.5 | Document as a user assertion; default is `id()` |
| A `Defer` hides a dependency by closing over it | 23.3 | Docstring: pass dependencies as arguments. Cannot be enforced |

## Definition of done

Both steps: the [standard exit checks](../docs/dev/roadmap.md#exit-checks-every-step) pass, the
step's notebook has been **run by the user and signed off** (the blocking
manual gate — automation only proves the notebook executes, not that the story
reads), and the handoff commit named in the step's roadmap section is on the
branch before the merge.
