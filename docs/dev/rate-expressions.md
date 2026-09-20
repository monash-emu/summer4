# Rate expressions: why typed nodes instead of a compute graph

Design rationale for `summer4.flows.rates`, written as an answer to a recurring
question: *summer2 expressed the same things with two generic types
(`Variable` and `Function`) from `computegraph`; summer4 has thirteen rate node
classes and a dunder protocol. What does the extra machinery buy, and could a
better compute graph buy it more cheaply?*

This page is normative about **why**. The stage contract itself is
[run-stages](run-stages.md); the user-facing ladder for custom processes is
{doc}`../cookbook/01-custom-rates`.

## The two designs

**computegraph** (summer2, `monash-emu/computegraph`) has two node types.
`Variable(key, source)` reads `sources[source][key]` at evaluation.
`Function(func, args, kwargs)` calls an arbitrary Python callable on evaluated
children. Operators on `GraphObject` build `Function(jnp.add, …)`; `__getitem__`
builds `Function(getitem, …)`. A `ComputeGraph` is a `networkx` DAG of named
nodes, topologically sorted into one Python closure. Everything a model needs
to compute is a `Function` over `Variable`s, and the framework is closed: two
types cover every case, and a user never extends it.

**summer4** has an expression tree of frozen dataclasses — `Const`, `Time`,
`FieldRef`, `FlowRef`, `UnaryOp`, `BinOp`, `Interp`, `GaussianPulse`, `Reduce`,
`Capture`, `ArrayConst`, `TableInterp`, `Lookup` — plus `ForceOfInfection` in
`summer4.epi` and three adjustment kinds (`Multiply`, `Overwrite`,
`Transform`). Evaluation is one `match` in `_eval_rate`. The framework is open:
`register_rate_eval` admits nodes defined outside the package.

The difference is not "more classes for the sake of it". It is a choice about
**where a user's intent is legible**. In computegraph, intent is inside an
opaque `func` — the graph knows a call happens and knows its children, but
nothing about what the call means. In summer4 the intent *is* the node type, so
the compiler can read it.

## What the node zoo buys: seven compile-time passes

The cost of typed nodes is that every node participates in seven traversals.
The benefit is that those seven traversals exist at all — each one is a
question you cannot answer about an opaque callable.

| Pass | Where | Question it answers | Impossible over an opaque `Function` because… |
|---|---|---|---|
| `_eval_rate` | `flows/compiled.py` | what is the value? | (this one is possible — it is the only one) |
| `rate_stage` | `flows/stages.py` | does this depend on `t`, `y`, or another flow? | a callable's free variables are not introspectable |
| `build_hoist_table` | `flows/stages.py` | which subtrees can move from per-step to run-start? | you cannot hoist what you cannot classify |
| `_rate_bytes` | `flows/rates.py` | are two models structurally identical? | callables have no value identity, only `id()` |
| `_flow_refs` | `flows/rates.py` | which flows must be evaluated first? | `topo_sort` needs the edge before the call runs |
| `_field_paths` | `flows/rates.py` | which parameter paths does the model read? | needed to validate `derived_fn` output |
| `_collect_capture_meta` | `flows/compiled.py` | which named grouped quantities are saveable? | `GroupedOutput` needs the properties before the solve |

Three of those are load-bearing in a way worth spelling out.

### Staging (`rate_stage` → `build_hoist_table`)

A rate that depends only on parameters is evaluated **once per `run`**, not once
per solver step. With a 4000-step solve, an `Interp` knot stack built from
calibrated `FieldRef`s is the difference between one `jnp.stack` and four
thousand. That optimisation needs a compile-time predicate over the expression
— "does this subtree read `t` or `y`?" — and that predicate is a structural walk
of typed nodes. Over a `Function(some_closure, …)` the answer is always
"assume yes", because the closure might read anything. computegraph has no
staging distinction for exactly this reason: every node runs every step.

### Digest and jit cache (`_rate_bytes`)

`CompiledModel` is passed to `jax.jit` as a **static** argument, so `__hash__`
and `__eq__` decide cache hits. Both are its `_digest`, which folds
`_rate_bytes` of every rate. Typed nodes hash by *value*: two models built in
separate cells from `linear(Time(), [0, 10], [0.1, 0.4])` share a compiled
program. A `Function` wrapping a closure hashes by `id()`, so a model rebuilt
inside a builder function is a cache miss every time. This is observable today
in the one place summer4 still keys on identity — see
[the escape hatches](#the-escape-hatches-today).

### Shape and grouping (`Reduce`, `TableInterp`, `ForceOfInfection`)

Nodes do not all return scalars. `Reduce` and `ForceOfInfection` return a
`GroupedRate` — an array plus the `Property` tuple it is indexed by. That
wrapper is what lets `_align_rate` gather a per-age force of infection onto
edges without the user writing index arithmetic, and what makes
`GroupedRate / GroupedRate` refuse to combine mismatched groupings.
computegraph's `Function` returns whatever the callable returned, so the
alignment has to be done by hand in the callable, correctly, every time.

## What it costs

Honest accounting, because the cost is real:

- **Seven touch points per node.** Adding `Lookup` meant editing `_eval_rate`,
  `rate_stage`, `build_hoist_table`, `_rate_bytes`, `_flow_refs`, `_field_paths`
  — six files-worth of `match` arms for one feature.
- **Six silent-default dunders for extension nodes.** `__rate_bytes__`,
  `__field_paths__`, `__flow_refs__`, `__rate_stage__`, `__capture_meta__`,
  `__capture_children__`. Every one of them has a fallback that does something
  reasonable-looking and wrong. `_rate_bytes` falling back to the class name is
  a **silent miscompile**: two custom nodes of the same class with different
  fields digest identically and share a jit cache entry. See
  [`futureplans/custom-rate-node-digest-collision.md`](https://github.com/monash-emu/summer4/blob/main/futureplans/custom-rate-node-digest-collision.md).
- **A closed operator set behind hand-written wrappers.** `UnaryOp` allows
  seven operations and `BinOp` seven, exposed as the named functions `exp`,
  `log`, `tanh`, `sqrt`, `floor`, `maximum`, `minimum`, `clip`. Anything else —
  `sin` for seasonality, `sigmoid`, `erf` — needs a node, an evaluator and the
  dunders, or an escape hatch. See
  [the operator surface](#the-operator-surface-a-fixable-mistake).

The trade is deliberate: summer4 pays authoring cost at the framework boundary
to keep the compiled program small, cacheable and statically analysable. But
the cost lands on *extension authors*, which is exactly who the dunder protocol
is failing.

## The escape hatches today

Ordered simplest-first. If you want "defer this arbitrary code and call it
later to produce a rate", these are the three answers, and the first is the one
most people want.

### 1. `Transform` adjustment — the closest thing to `Function()`

`Transform(fn, *args)` calls `fn(prev, *evaluated_args)` where `prev` is the
rate so far and each arg is any rate expression. It is an adjustment, not a
rate, so it sits on the flow's `adjust=` list rather than in the rate slot:

```python
def seasonal(prev, t, amp):
    return prev * (1.0 + amp * jnp.sin(t / 10.0) ** 2)

TransitionFlow(
    "recovery", state["I"], state["R"], 0.1,
    adjust=[Transform(seasonal, Time(), Param("amp"))],
)
```

`fn` receives traced JAX values and may do anything traceable. No registration,
no dunders.

**The catch:** `_adjust_bytes` encodes a `Transform` as `id(adj.fn)`. A module-level
`fn` is stable, so rebuilding the same model hits the jit cache. A `fn` defined
*inside a builder function* is a new object per call, and every rebuild misses.
If you reach for `Transform`, define the callable at module level.

### 2. `derived_fn` — arbitrary code, once per step, named outputs

`compile(derived_fn=...)` runs `derived_fn(params, y=, t=)` before the rates
and exposes its result to every `FieldRef`. Annotate the return as a
`NamedTuple` and `derived_refs(Schema)` hands back a typed accessor:

```python
class D(NamedTuple):
    rec: Any

def derived(params, *, y, t) -> D:
    return D(rec=0.1 * (1.0 + params["amp"] * jnp.sin(t / 10.0) ** 2))

D_ = derived_refs(D)
model.add_flow(TransitionFlow("recovery", state["I"], state["R"], D_.rec))
cm = model.compile(derived_fn=derived)
```

This is the general-purpose hook, and `ComputedValue` can save its outputs. Two
costs: it is keyed by `id(derived_fn)` in the digest (same rebuild caveat), and
per [R3](run-stages.md) it makes **every** `FieldRef` step-stage, disabling
parameter hoisting model-wide. Put `t`/`y`-independent work in `prepare_fn`
instead.

### 3. `RateOps` subclass + `register_rate_eval` — for reuse

The package-author path, documented in
{doc}`../cookbook/01-custom-rates`. Worth it when the same named process
appears in many models; overkill for one hazard. If you take it, **implement
`__rate_bytes__`** — the default is unsafe.

## The operator surface: a fixable mistake

The named arithmetic helpers (`exp`, `log`, `tanh`, `sqrt`, `floor`, `maximum`,
`minimum`, `clip`) attract the obvious objection: *why not just write the `jnp`
functions?* The objection is right about the design and wrong about the
mechanism, and the difference matters.

**You cannot just write `jnp`.** Every one of these raises `TypeError` today:

```python
jnp.exp(Param("beta"))          # Error interpreting argument … as an abstract array
jnp.clip(Param("beta"), 0, 1)
jnp.maximum(Param("beta"), 0.0)
jnp.exp(some_trace)             # also fails on Trace, GroupedRate, PropertyData
```

Operators work (`Param("b") * 2` builds a `BinOp`) because Python routes them
through `__mul__`. Functions do not, because `jnp.exp` is a `PjitFunction`, not
a `numpy.ufunc`, and JAX implements **neither** `__array_ufunc__` nor
`__array_function__`. There is no protocol for a third-party object to
intercept a `jnp` call. So the wrappers are not duplicating `jnp` — they are
the only way to get a non-operator op into a rate tree at all.

**This is exactly what computegraph got right.** `GraphObject` implements
`__array_ufunc__` and `__array_function__`, so `np.exp(graph_obj)` auto-builds
`Function(jnp.exp, [obj])` and summer2 never needed a wrapper list. That worked
because summer2 was **NumPy-first at the API boundary** — `jaxify.get_modules()`
swapped `jnp` in underneath. The user always wrote `np.*`.

**The wrappers do earn something `jnp` cannot give.** They dual-dispatch. One
symbol works on an unevaluated rate tree *and* on each of the four evaluated
wrapper types, preserving the wrapper:

| call | `summer4` helper | `jnp` equivalent |
|---|---|---|
| on `RateOps` | builds `UnaryOp` | `TypeError` |
| on `Trace` | returns `Trace` | `TypeError` |
| on `GroupedRate` | returns `GroupedRate` | `TypeError` |
| on `PropertyData` | returns `PropertyData` | `TypeError` |

Writing `jnp` by hand means unwrapping `.data` / `.values`, losing the `pmap`,
`properties` or `dims` that make the result queryable, and rewrapping. That is
the actual gain — not the arithmetic.

**But the shape is wrong.** The gain is a *dispatch* concern, and it is being
paid for with a hand-maintained allowlist: adding `sin` means editing
`UNARY_OPS`, the `Literal[...]` annotation on `UnaryOp.op`, a new exported
wrapper, and the docs. `clip` is pure sugar over `maximum` + `minimum` and
carries no node of its own. The set is arbitrary — `tanh` is in, `sin` is not,
and seasonal forcing is a first-order epidemiological need.

### The fix, prototyped

Implement `__array_ufunc__` / `__array_function__` on `RateOps`, `Trace`,
`GroupedRate` and `PropertyData`, and key the op by **name** rather than by
callable — computegraph's `getattr(fnp, func_str)` trick. A working prototype
against `c4d3534`:

```python
def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kw):
    if method != "__call__" or out is not None:
        return NotImplemented
    name = ufunc.__name__
    if len(inputs) == 1 and name in _UNARY:
        return UnaryOp(name, as_rate(inputs[0]))
    if len(inputs) == 2 and name in _BINARY:
        return BinOp(name, as_rate(inputs[0]), as_rate(inputs[1]))
    return NotImplemented
```

with `UNARY_OPS` populated as `{n: getattr(jnp, n) for n in _UNARY}`. Verified:

- `np.sin(Param("phase"))` builds `UnaryOp(op='sin', arg=FieldRef(('phase',)))`.
- `rate_stage` classifies it `run` with no change — staging reads the argument,
  not the op.
- `0.1 * (1.0 + 0.5 * np.sin(Param("phase")))` compiles and evaluates to the
  expected vector field.
- **Value-keyed digests survive**, which is the property that matters most:
  the op is a *string*, not a callable id, so `_rate_bytes` already encodes it
  (`b"unary:" + op.encode()`). Two separately-built `np.sin` models compare
  equal and share a jit cache entry; `np.sin` and `np.cos` compare distinct.

So the whole numpy API opens up, the hand-maintained list disappears, and
nothing the typed-node design exists for is lost. The named exports stay as
thin aliases for discoverability and for users who prefer them.

One caveat to state plainly in the user docs if this lands: `np.sin(expr)`
would work and `jnp.sin(expr)` still would not, because JAX offers no dispatch
hook. That is summer2's bargain — write `np.*`, get `jnp` execution — and it is
a genuine wart, not a design we would choose if JAX gave us the option.

## Could we have a better compute graph?

The question contains its own answer, and it is not "adopt computegraph". It is:
*can the typed-node benefits be kept while collapsing the seven traversals into
one?* Three shapes, worst to best.

### (a) Generic `Function` node — rejected

Adding `Function(fn, args)` as a `RateOps` alongside the typed nodes is the
straight computegraph port. It costs staging (always step), value-identity
digests (always `id()`), and grouping (always hand-aligned) for every
expression that uses it. `Transform` already occupies this niche at the
adjustment layer, with those exact costs, and its `id()` keying is a known
sharp edge rather than a model to extend.

### (b) Proxy-with-intent — already half-built, worth finishing

A typed proxy that records operations while preserving static type information
is not hypothetical here: `derived_refs(Schema)` is one. It returns something
`cast` to `Schema`, so mypy and IDE completion see `D.migration.baseline` as its
declared type, while the runtime value is a `FieldRef(("migration", "baseline"))`
path. The user writes ordinary attribute access; the framework receives a typed
node.

Generalising that is the real proposal behind the question. It looks like:

- **A shape/grouping protocol on nodes** — `GroupedRate` already carries
  `properties` as *values*. Lifting that to the type level (a generic
  `Rate[Grouping]`, or at minimum a `__grouping__()` classmethod every node
  implements) would let `mypy --strict` reject a per-age FOI used where an
  edge-aligned rate is required, instead of raising at trace time.
- **One traversal instead of seven.** Every node already knows its children.
  A single `__children__()` on `RateOps` would let `_flow_refs`,
  `_field_paths`, `_collect_capture_meta` and the `build_hoist_table` walk
  become one generic fold plus a per-node leaf function, cutting the extension
  surface from six dunders to two (`__children__`, `__rate_bytes__`) and
  removing four of the silent-default failure modes.
- **A required, total digest.** `_rate_bytes` should raise for a registered
  node that does not encode itself, not fall back to the class name.

That is a refactor of the existing design, not a replacement of it — which is
the point. The node zoo's problem is *duplicated traversal*, not *typed nodes*.

### (c) Structural tracing — the bigger idea, not yet justified

JAX already has a system for turning arbitrary Python into an inspectable,
cacheable, stageable program: `jax.make_jaxpr`. A rate could in principle be
"any traceable callable", traced once at compile to a jaxpr, with staging
recovered by inspecting which of `t` / `y` the jaxpr actually reads and the
digest taken over the jaxpr rather than the callable.

This would genuinely subsume the zoo: arbitrary user code *and* structural
analysis. It is not proposed as work because the pieces summer4 needs most —
`FlowRef` topological ordering, `GroupedRate` alignment, `Capture` naming — are
*framework* concepts that a jaxpr has erased by the time you can read it. A
jaxpr tells you which arrays were touched, not which flow depends on which.

## Where this leaves the recommendation

Keep typed nodes. They are why staging, value-keyed jit caching, and grouping
alignment exist at all, and none of those survive a generic `Function`.

Attack the actual cost, which is traversal duplication and unsafe extension
defaults, in this order:

1. Make `_rate_bytes` **total** — raise, do not fall back. It is a correctness
   bug today, not a papercut.
2. Replace the hand-written arithmetic wrappers with `__array_ufunc__` /
   `__array_function__` on the four wrapper types, keying ops by name.
   Prototyped above; opens the numpy API, deletes the allowlist, keeps
   value-keyed digests and staging intact.
3. Add `__children__()` to `RateOps` and refold the four child-walking passes
   onto it.
4. Consider a grouping marker on the node type so `mypy` catches alignment
   errors that currently surface at trace time.

Steps 1–3 remove most of the reason the question gets asked. The design would
then be "typed nodes, two dunders, one traversal, open operators" — a smaller
surface than the compute graph it is being compared to, and with the one thing
computegraph genuinely did better (protocol dispatch) adopted rather than
reinvented.
