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
| `rate_stage` | `flows/stages.py` | does this depend on `t`, `y`, or another flow? | **partly possible** — see the correction below |
| `build_hoist_table` | `flows/stages.py` | which subtrees can move from per-step to run-start? | sub-node hoisting (an `Interp` knot stack) needs the node's internal structure |
| `_rate_bytes` | `flows/rates.py` | are two models structurally identical? | callables have no value identity, only `id()` |
| `_flow_refs` | `flows/rates.py` | which flows must be evaluated first? | `topo_sort` needs the edge before the call runs |
| `_field_paths` | `flows/rates.py` | which parameter paths does the model read? | needed to validate `derived_fn` output |
| `_collect_capture_meta` | `flows/compiled.py` | which named grouped quantities are saveable? | `GroupedOutput` needs the properties before the solve |

Three of those are load-bearing in a way worth spelling out.

(staging-pass)=
### Staging (`rate_stage` → `build_hoist_table`)

A rate that depends only on parameters is evaluated **once per `run`**, not once
per solver step. With a 4000-step solve, an `Interp` knot stack built from
calibrated `FieldRef`s is the difference between one `jnp.stack` and four
thousand. That optimisation needs a compile-time predicate over the expression:
"does this subtree read `t` or `y`?"

**Correction to an earlier version of this page**, which claimed the predicate
is impossible over a `Function`-style node because a callable's free variables
are not introspectable. That is too strong, and the distinction it misses is
the one that matters for the rest of this page.

A `Function(func, args)` — computegraph's, or a `Defer(fn, *args)` in summer4 —
takes its dependencies as **explicit arguments**. Nothing is hidden, so the
node's stage is exactly the join of its arguments' stages. Verified with a
~35-line `Defer` prototype: `defer(f)(Param("x"), Param("y"))` classifies as
`run` and hoists; `defer(f)(Time(), Param("amp"))` classifies as `step`. The
predicate is only defeated by a callable that *closes over* model state rather
than receiving it, which is a property of how the callable is written, not of
the node being generic.

computegraph therefore *could* have staged; it simply never did — every node
runs every step. That is a missing feature in summer2, not a consequence of its
design.

What typed nodes uniquely buy here is finer: **sub-node** hoisting. `Interp` is
hoisted in three independent pieces — the breakpoint stack, the value stack and
the argument — so knots built from calibrated `FieldRef`s stack once per run
even though the argument is `Time()` and the node as a whole is step-stage. A
generic node is all-or-nothing: wrap that same interpolation in a `Defer` whose
argument is `Time()` and the knot stack rebuilds on every one of the 4000 steps.
That is a real benefit, and a narrower one than originally claimed.

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
- **Six dunders for extension nodes, five of them silent-defaulting.**
  `__rate_bytes__`, `__field_paths__`, `__flow_refs__`, `__rate_stage__`,
  `__capture_meta__`, `__capture_children__`. The last five have a fallback that
  does something reasonable-looking and can be wrong. `__rate_bytes__` used to
  be the sixth, and it was the dangerous one: falling back to the class name was
  a **silent miscompile**, because two custom nodes of the same class with
  different fields digested identically and shared a jit cache entry. It is now
  mandatory — `register_rate_eval` rejects a class that omits it and
  `_rate_bytes` raises rather than guessing — so that failure is an import-time
  `TypeError` instead of wrong numbers. The remaining five fail conservatively
  (a redundant recompile, a step-stage node that could have hoisted), not
  silently wrong.
- **An open operator set.** NumPy ufuncs on a rate expression build a node
  (`np.sin(Param("phase"))`). The named helpers (`exp`, `log`, `tanh`, `sqrt`,
  `floor`, `maximum`, `minimum`, `clip`) remain as aliases. `jnp.sin` still
  raises: JAX has no dispatch hook. See
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

(identity-keying-cost)=
### What identity keying actually costs

The caveat is narrower than "inline functions recompile". It bites on model
**rebuild**, never on `run`. Measured by counting traces — a jitted function's
Python body executes once per trace, so a cache hit counts zero — with
`jax.jit(..., static_argnums=0)` over a `CompiledModel`:

| Pattern | Traces |
|---|---|
| One model, 50 parameter draws | **1** |
| 50 rebuilds, callable defined at module level | **0** (hits the entry an identical earlier model compiled) |
| 50 rebuilds, callable defined inside the builder | **50** |
| One closure-built model, 50 runs | **1** |
| Two closures capturing `scale=1.0` and `scale=2.0` | **2**, and the results differ |

Three things follow.

**Calibration is unaffected.** Parameters are dynamic — `prepare` promotes
Python `float` leaves to arrays precisely so equinox does not key on their
values — so thousands of draws against one model object compile once. The only
way to pay is to construct a new `CompiledModel` per iteration, which a
calibration loop does not do.

**Closures are not ruled out**, and identity keying is the *correct* choice for
them rather than a concession. The last row is the reason: two closures with
identical source but different captured values are different programs and
**must not** share a cache entry. `id()` separates them for free. Keying on
`fn.__qualname__` would give both the same key
(`build_closure.<locals>.cl`) and silently return the first closure's numbers
for the second — the unsafe direction, and the same trap as
[`custom-rate-node-digest-collision.md`](https://github.com/monash-emu/summer4/blob/main/futureplans/custom-rate-node-digest-collision.md).

**What it does cost** is re-*building* with a freshly created callable: a
notebook cell re-executed after editing the function, or a scenario sweep that
rebuilds the model per variant. Both recompile, and neither is wrong — just
slower. Note also that summer4 does not jit `run` itself; the model is
documented as a static argument for callers who jit, and diffrax's `filter_jit`
caches internally. A bare `cm.run(...)` in a script pays no summer4-level
cache miss at all.

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
appears in many models; overkill for one hazard. If you take it you **must
implement `__rate_bytes__`**; `register_rate_eval` raises without it.

### The on-ramp is `defer`

summer2's door was `computegraph.defer(f)(param("y"), 5.1)`: an ordinary
function, wrapped, called with parameters and literals. summer4's equivalent
is `defer`:

```python
model.add_flow(
    TransitionFlow(
        "infection",
        state["S"],
        state["I"],
        defer(my_own_function)(Time(), Param("amp")),
    )
)
```

`fn` is called with traced JAX values (arrays, or a `GroupedRate` /
`PropertyData` when an argument evaluates to one). Pass every dependency as
an argument. A value closed over is invisible to staging and to the digest.

`defer(f)(Param("x"), Param("y"))` is run-stage and hoists.
`defer(f)(Time(), Param("amp"))` is step-stage, and a param-only argument
inside it still hoists. The digest keys on `id(fn)` unless you pass `name=`.
`name=` asserts that two function objects are the same program. A wrong name
is a wrong answer. See
[What identity keying actually costs](#identity-keying-cost).

`Transform` is not this door. It cannot occupy the rate slot, it is an
adjustment of a rate that already exists, and its callable takes the previous
rate as a first argument. `derived_fn` still owns multi-output derived
quantities; it is not the way to write one hazard.

## The operator surface: a fixable mistake

The named arithmetic helpers (`exp`, `log`, `tanh`, `sqrt`, `floor`, `maximum`,
`minimum`, `clip`) attract the obvious objection: *why not just write the `jnp`
functions?* The objection is right about the design and wrong about the
mechanism, and the difference matters.

**You cannot just write `jnp`.** Every one of these raises `TypeError`:

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
intercept a `jnp` call.

**This is exactly what computegraph got right.** `GraphObject` implements
`__array_ufunc__` and `__array_function__`, so `np.exp(graph_obj)` auto-builds
a node and summer2 never needed a wrapper list. That worked because summer2 was
**NumPy-first at the API boundary** — `jaxify.get_modules()` swapped `jnp` in
underneath. The user always wrote `np.*`. summer4 now does the same:
`np.sin(Param("phase"))` builds a `UnaryOp`. The named helpers stay as aliases.

**The wrappers still earn something `jnp` cannot give.** They dual-dispatch, and
so does `np.*`. One symbol works on an unevaluated rate tree *and* on each of
the evaluated wrapper types, preserving the wrapper:

| call | `np.*` or a `summer4` helper | `jnp` equivalent |
|---|---|---|
| on `RateOps` | builds `UnaryOp` / `BinOp` | `TypeError` |
| on `Trace` | returns `Trace` | `TypeError` |
| on `GroupedRate` | returns `GroupedRate` | `TypeError` |
| on `PropertyData` | returns `PropertyData` | `TypeError` |

Writing `jnp` by hand means unwrapping `.data` / `.values`, losing the `pmap`,
`properties` or `dims` that make the result queryable, and rewrapping. That is
the actual gain — not the arithmetic.

The old shape was a hand-maintained allowlist: adding `sin` meant editing
`UNARY_OPS`, the `Literal[...]` annotation on `UnaryOp.op`, a new exported
wrapper, and the docs. `tanh` was in, `sin` was not, and seasonal forcing is a
first-order epidemiological need. `clip` is still sugar over `maximum` +
`minimum` and owns no node.

### The fix, applied

`__array_ufunc__` and `__array_function__` are on `RateOps`, `Trace`,
`GroupedRate` and `PropertyData`. The op is a **canonical string**, resolved to
a `jax.numpy` callable at evaluation — computegraph's `getattr(fnp, func_str)`
trick. Six of summer4's existing spellings are not NumPy's (`mul` vs
`multiply`, `sub` vs `subtract`, and the same for divide, power, negative and
absolute). An alias map canonicalises those, so `np.multiply(a, b)` and `a * b`
digest identically and share a jit cache entry. New ops keep the NumPy name:
`np.sin(x)` is `UnaryOp("sin", x)`.

`__array_function__` allows `np.clip`, `np.maximum`, `np.minimum`, `np.power`
and `np.absolute` only. Comparisons and other masks are refused: they belong
to `Selector`. `matmul` is refused on a rate node because `GroupedRate @`
already owns that operator.

One deliberate behaviour change: `np.array([1.0, 2.0]) * Param("x")` used to
return an object-dtype array of `BinOp`s. It now builds one
`BinOp("mul", ArrayConst(...), ...)`.

The wart to state in user docs: `np.sin(expr)` works and `jnp.sin(expr)` does
not, because JAX offers no dispatch hook. That is summer2's bargain — write
`np.*`, get `jnp` execution — and it is a genuine wart, not a design we would
choose if JAX gave us the option.

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
- **A required, total digest.** Done: `_rate_bytes` raises for a node that does
  not encode itself, and `register_rate_eval` refuses to register one, so the
  error lands at the class definition rather than in the numerics.

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

1. **Done.** `_rate_bytes` is total: `register_rate_eval` rejects a class
   without `__rate_bytes__`, and the walk raises rather than digesting by class
   name.
2. **Done.** `__array_ufunc__` / `__array_function__` on the four wrapper
   types, ops keyed by canonical name. `np.sin` works; the named helpers are
   aliases; value-keyed digests and staging are unchanged.
3. Add `__children__()` to `RateOps` and refold the four child-walking passes
   onto it.
4. Consider a grouping marker on the node type so `mypy` catches alignment
   errors that currently surface at trace time.

Items 1 and 2 have landed. Items 3 and 4 are what is left: one traversal, and
a grouping marker so `mypy` can catch alignment errors. The operator set is
open, which is the thing computegraph did better, adopted rather than
reinvented.
