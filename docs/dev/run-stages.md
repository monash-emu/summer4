# Run stages (compile / run start / per step)

Normative contract for parameter transforms and rate evaluation in summer4.
Implemented in `summer4.flows.stages` and wired through `CompiledModel`.

## Stage contract

| Stage | When | Runs on | Inputs | Produces |
|---|---|---|---|---|
| **0 — compile** | `FlowModel.compile()`, once per model structure | host (NumPy) | `PropertyMap`, flows, `InitialPopulation` | index arrays, `HoistTable`, `InitPlan`, digest |
| **1 — run start** | once per `CompiledModel.run` / `initial_state` / loss evaluation, **before** the solve | traced JAX | raw `params` | `Prepared(params=box_float_leaves(prepare_fn(params) or params), hoisted=(...))`; the initial state `y0`. Python `float` leaves are promoted to float64 arrays so equinox's `filter_jit` treats them as dynamic across draws |
| **2 — per step** | every vector-field call | traced JAX, inside `scan`/`while` | `t`, `y`, `Prepared` | rates, reading hoisted values and the prepared parameters |

Rules:

- **R1.** `prepare_fn(params) -> params'` is a user hook: `FlowModel.compile(prepare_fn=...)`. It must not depend on `t` or `y`. It hashes by `id` in the digest. After the hook (or on raw params), `prepare` promotes plain Python `float` leaves to float64 arrays so Diffrax's equinox cache does not key on the numeric values.
- **R2.** Initial-population expressions and callables read `Prepared.params`, never the output of a deferred function. A `Time()`, `FlowRef`, `Reduce` or `Capture` in one is a compile-time `ValueError`.
- **R3.** Removed in step 41. A `FieldRef` (and a `Param`) is always run stage, so parameter-only subtrees always hoist. Code that reads `t` or `y` as well goes through `field_refs(defer(f)(Params(), y=StateArray(), t=Time()), Schema)`; that subtree is step stage, and parameter-only subtrees beside it still hoist.
- **R4.** Hoisting is an optimisation only. `compile(hoist=False)` must give identical results. The digest includes the `hoist` flag.
- **R5.** `observe` / `vector_field` still accept raw params. If `params` is not a `Prepared`, they call `self.prepare(params)` themselves, which is correct but not hoisted. Solvers always call `prepare` once and pass the `Prepared` down.
- **R6.** In `run`, an explicit `y0` overrides a model-attached initial population (warm starts). With no `y0` and no initial population, `run` raises `ValueError`.

## Stage classification of rate nodes (`rate_stage`)

| Node | Stage |
|---|---|
| `Const`, `ArrayConst` | run (leaf) |
| `FieldRef`, `Param`, `Params()` | run (leaf) |
| `StateArray()` | step (leaf) |
| `Time`, `FlowRef`, `Reduce`, `Capture` | step |
| `BinOp` | step if any child is step, else run |
| `UnaryOp` | same stage as its argument |
| `Interp` | step if any of breakpoints/values/arg is step, else run |
| `GaussianPulse` | step if any of arg/centre/width/height is step, else run |
| any other `RateOps` | `expr.__rate_stage__()` when the node defines it, else **step** (conservative; `ForceOfInfection` stays step). `Defer` is run stage when every argument is |

## Hoisting rule (`build_hoist_table`)

Walk the roots depth-first. The roots are every flow's `rate`, every
`Multiply.value`/`Overwrite.value`, and every `Transform.args` item, in
`order`. At each node:

1. If the node is **run-stage and not a leaf** (a leaf is `Const`,
   `ArrayConst` or `FieldRef`), give it a slot keyed `(id(node), "value")`.
   Do not descend.
2. Otherwise, if the node is an `Interp`:
   - if every breakpoint is run-stage, add slot `(id(node), "breakpoints")`;
   - if every value is run-stage, add slot `(id(node), "values")`.

   Then descend into `arg`, and into any breakpoints/values not covered by a
   slot.
3. Otherwise descend into the children of `BinOp`, `UnaryOp` and `GaussianPulse`, into a
   `Lookup` index, and into `__rate_children__()` when a node defines it (so a step-stage
   `Defer` still hoists a run-stage argument). Do not descend into `Capture` or unknown
   custom nodes.
4. Skip a key that already has a slot, because shared node objects are
   evaluated once.

The `HoistTable` keeps a reference to every slotted node, so `id()` stays valid
for the lifetime of the `CompiledModel`.
