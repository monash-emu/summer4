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
- **A closed operator set.** `UnaryOp` allows seven operations and `BinOp`
  seven. Anything else needs a node, an evaluator, and the dunders — or an
  escape hatch.

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
appears in many models; overkill for one hazard. If you take it you **must
implement `__rate_bytes__`**; `register_rate_eval` raises without it.

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

Attack the actual cost, which is traversal duplication and unsafe extension
defaults, in this order:

1. Make `_rate_bytes` **total** — raise, do not fall back. It is a correctness
   bug today, not a papercut.
2. Add `__children__()` to `RateOps` and refold the four child-walking passes
   onto it.
3. Consider a grouping marker on the node type so `mypy` catches alignment
   errors that currently surface at trace time.

Steps 1 and 2 remove most of the reason the question gets asked: the design
would then be "typed nodes, two dunders, one traversal", which is a smaller
surface than the compute graph it is being compared to.
