---
name: results
overview: "Phase 2 of flows/derived-outputs — time and epochs, a State pytree, observe()/SavePlan, and a queryable Result with a JIT-traceable time surface (calendar resample, rolling, cumulative, interpolated at_times)."
todos:
  - id: time
    content: "summer4.time: Epoch, TimeAxis, TimeGrouping, RollingSpec; numpy datetime64 calendar core"
    status: pending
  - id: state
    content: "State pytree (compartments + empty ledgers); _unpack_state in the vector field"
    status: pending
  - id: observe
    content: "observe() -> SaveContext; vector_field becomes observe(...).dy"
    status: pending
  - id: plan
    content: "SavePlan / SaveRequest / Quantity; expand(); describe() via jax.eval_shape"
    status: pending
  - id: result
    content: "Trace + Result + SolverInfo; traceable query surface; to_frame / to_pandas"
    status: pending
  - id: run
    content: "CompiledModel.run(); nested-scan Euler fast path; notebook, tests, ledger, docs"
    status: pending
isProject: true
---

# Phase 2 — Time, trajectories and `Result` (`feat/results`)

Parent design: [flows-derived-outputs.plan.md](flows-derived-outputs.plan.md) §Phase 2.
Predecessor: [flows-core.plan.md](flows-core.plan.md) (Phase 1, complete).

## Context

Phase 1 promoted flows into `summer4`: `FlowModel.compile()` returns a
`CompiledModel` with `EdgeMap` queries and a JAX vector field, and the ledger
stands at **22 / 52 (42%)**. But `euler()` still returns **the final state
only** — `src/summer4/flows/compiled.py:546`. Every summer2 notebook and nearly
every textbook chapter ends by running a model and plotting a trajectory, so
"expressible" has not yet become "publishable". The binding constraint is now
results, exactly as the parent plan predicted.

Phase 2 builds the outputs mechanism: real-world time, a `Result` object queried
with the selector architecture that already exists, and a `SavePlan` that says
what to keep. It is deliberately **not** summer2's `request_output_for_flow`
registry — no flat global string namespace, no declaration-order dependency, no
mutable whitelist. The user names the keys; the plan *is* the request.

Two decisions taken for this phase, both widening the parent plan:

1. **Calendar time operations are JIT-traceable.** The parent plan filed
   `resample_calendar` as host-side-only. Instead, every time operation splits
   into a *static* index computation at trace time and a *traced* gather or
   segment reduction. Monthly resampling, rolling windows, cumulative sums and
   interpolated `at_times` are all usable inside a calibration loss.
2. **Both dataframe libraries, explicitly.** `to_frame()` returns polars
   (consistent with `PropertyMap.to_frame`, `propertymap.py:300`); `to_pandas()`
   returns a `DatetimeIndex`-indexed frame. Both are lazy imports, and **neither
   is required by the traced path** — the calendar core is numpy `datetime64`.

Target: **32 / 52 (62%)**.

## Baseline and branch

Cut `feat/results` from `feat/flows-core` (or from `main` once Phase 1 merges —
`main` has none of the flows work). Copy this plan to `plans/results.plan.md`
and `.cursor/plans/results.plan.md` on the branch as the first commit;
`check-branch` fails a feature branch with no plan under `plans/`.

---

## Stage 1 — `src/summer4/time.py` (NumPy + stdlib only)

New module, no JAX import at module scope, no pandas in the traced path.

**`Epoch`** — a pure host-side affine map, ported from `summer3wip`'s
`utils.py:56`: `ref_date: date`, `unit: timedelta = timedelta(days=1)`.
`to_model(dates) -> NDArray[float64]`, `from_model(values) -> NDArray[datetime64[ns]]`.
Relax both predecessor limits: summer2 hard-codes `timedelta(1)` so its `unit`
is dead, and summer3wip's `dti_to_epoch` assumes a uniform step.

**`TimeAxis`** — `values: NDArray[float64] | Array`, `epoch: Epoch | None`,
`kind: Literal["grid", "explicit", "steps"]`. `values` is the only pytree leaf;
`epoch` and `kind` are aux. Methods:

| Method | Returns | When |
|---|---|---|
| `locate(when)` | `int` | host-side; accepts `float \| date \| datetime` |
| `window(t0, t1)` | `slice` | host-side |
| `weights_for(ts)` | `(idx (n,2) int32, w (n,2) float64)` | static lerp pairs |
| `grouping(rule, *, origin=None)` | `TimeGrouping` | static, cached |
| `rolling(window, *, how, center, min_periods)` | `RollingSpec` | static |
| `as_dates()` | `NDArray[datetime64[ns]]` | needs `epoch` |
| `as_index()` | `pd.Index \| pd.DatetimeIndex` | lazy pandas |

Every host-side method must raise a clear error when `values` is a tracer
("time values are traced; `locate` needs a concrete axis") rather than failing
deep inside numpy. A `date` is **never** a traced value — say so in the module
docstring and enforce it.

**`TimeGrouping`** — frozen, hashable, entirely static; this is the reified
"plan" half of the API:

```python
@dataclass(frozen=True, slots=True)
class TimeGrouping:
    segment_ids: NDArray[np.int32]   # len == n_times; -1 drops a point
    n_groups: int
    counts: NDArray[np.int32]        # per group, for mean and min_periods
    starts: NDArray[np.float64]      # group start in model time
    labels: tuple[str, ...]
    rule: str | int
```

Calendar rules are computed with numpy `datetime64` and stdlib only. Support
exactly `"D"`, `"W"` / `"W-<DAY>"`, `"ME"`, `"QE"`, `"YE"`, plus an integer
factor (fixed-stride reshape). Anything else raises with a message pointing at
`to_pandas()` for exotic offsets — a small documented vocabulary that never
drags pandas into a jitted loss is worth more than full offset-alias parity.
Cache groupings on the axis keyed by `(rule, origin)`.

**`RollingSpec`** — static window length, `how`, `center`, `min_periods`, plus
the precomputed per-position valid counts. Sum/mean apply as a cumulative-sum
difference (O(n), one pass); document that choice next to the alternative
(`jnp.convolve`) so nobody "optimises" it back.

**The rule, stated once here and repeated in the user guide:**

> All index arithmetic is host-side and static, computed from `TimeAxis` at
> trace time. Only gathers, segment reductions, lerps and arithmetic are traced.

---

## Stage 2 — `State` pytree

`src/summer4/jax/state.py`:

```python
@register_pytree_node_class
@dataclass(frozen=True, slots=True)
class State:
    compartments: PropertyData
    ledgers: Mapping[str, Any] = field(default_factory=dict)   # empty in Phase 2
```

This is the extension point the parent plan insists on reserving. Instantaneous
rates plus post-run reductions are right for *outputs* and wrong for *feedback*:
a cumulative quantity that feeds back into a rate (a vaccination coverage
ledger) must be state. Retrofitting that after `Result` and `SavePlan` freeze is
expensive, so introduce it now even though Phase 2 populates only the first
field.

Concretely: replace the `y_pd` / `y_arr` dance at `compiled.py:398` and `:518`
with one `_unpack_state(y) -> tuple[Any, Callable[[Any], Any]]` helper accepting
`State | PropertyData | Array`, and **do not** let `y.shape[-1] == pmap.size`
become a hard invariant in `_scatter_add`, `PropertyData.check` or the Euler
carry.

---

## Stage 3 — `observe()` and `SaveContext`

`flow_values[name] = mass` is currently a local in the `vector_field` loop
(`compiled.py:431`, `:437`, `:444`) and is discarded. Returning `(dy, aux)` from
the vector field is not an option — Phase 3's `ODETerm` relies on a vector field
returning a `y`-shaped pytree.

Factor the loop body out:

```python
@dataclass(frozen=True, slots=True)
class SaveContext:
    t: Any
    y: Any
    dy: Any
    derived: Any
    flows: Mapping[str, Any]      # per-edge mass, pruned to what the plan reads
```

`CompiledModel.observe(t, y, params) -> SaveContext`, and `vector_field` becomes
`observe(t, y, params).dy`. One code path, no duplication.

Pruning: `CompiledModel._observer(needs: frozenset[str])` returns a closure whose
`flows` mapping is restricted. Be honest about what this buys — every flow's
mass is computed anyway inside the loop, and flows referenced by a later flow's
rate must be computed regardless (`_flow_rate_refs`, `actualize.py:294`). The
saving is in **not materialising the buffers**, which is `SavePlan`'s job, not
`observe`'s. `needs` = requested ∪ transitive `FlowRef` dependencies.

Cost of the general path: one extra field evaluation per *save point* (~10%
under diffrax at `dt≈0.1` with daily saves). That is what summer3wip pays via
`jax.vmap(get_flow_values)(sol.ts, sol.ys)`, and it is fine.

---

## Stage 4 — `SavePlan` — `src/summer4/results/plan.py`

Named `SavePlan`, not `SaveAt`: `diffrax.SaveAt` lands in this module in Phase 3.

```python
type Quantity = Compartments | FlowMass | ComputedValue | SaveFn

@dataclass(frozen=True, slots=True)
class Compartments:
    where: Selector | None = None
    sum_over: Property | None = None

@dataclass(frozen=True, slots=True)
class FlowMass:
    """Per-edge mass. Reduce *here*: the buffer is (n_saves, n_kept), so a
    200k-edge flow over 3650 days is 5.8 GB dense and 470 kB summed over age."""
    flow: str
    where: Selector | NDArray[np.bool_] | None = None
    sum_over: tuple[Property, Side] | None = None

@dataclass(frozen=True, slots=True)
class ComputedValue:
    path: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class SaveFn:
    fn: Callable[[SaveContext], Any]
    reads: frozenset[str] = frozenset()

@dataclass(frozen=True, slots=True)
class SaveRequest:
    what: Quantity
    ts: NDArray[np.float64] | None = None    # None -> the plan's default grid

@dataclass(frozen=True, slots=True)
class SavePlan:
    requests: Mapping[str, SaveRequest] = ...   # empty == EVERYTHING
    ts: NDArray[np.float64] | None = None       # None -> the solve grid
    dense: bool = False                          # Phase 3
    solver_stats: bool = True
```

- **The trivial case desugars into the general one.** An empty `requests` is the
  sentinel (`EVERYTHING = SavePlan()`); `CompiledModel.expand(plan)` fills it
  with `compartments`, every flow, and every entry of `self.computed_paths`
  (`compiled.py:494`). There is exactly one code path below `expand`.
- **Hashable**, since it is a static argument: digest over request keys,
  quantity fields and `ts.tobytes()`. Document — next to the existing
  `derived_fn` / `Transform.fn` note at `compiled.py:359` — that `SaveFn.fn`
  hashes by identity, so a lambda defined in a loop retraces every call.
- **`describe(plan) -> PlanDescription`** via `jax.eval_shape` on each group's
  save function (diffrax's own trick). This does three jobs and the third pays
  for it: it sizes the Euler buffers; it validates that a user's `SaveFn` returns
  statically-shaped output instead of failing as an opaque `lax.scan` structure
  mismatch mid-solve; and it prints every output's shape and memory footprint
  **without running the model**.

**Phase boundary, to be stated in the code:** `FlowMass` requests land in Phase 2
and produce a `Trace` over `EdgeMap.table`. The *edge query surface* —
`sum_over(prop, side=)` on a trace, `.integrate()`, `.incidence()` — is Phase 4.
Phase 2 offers reduction inside the request only. Do not drift.

---

## Stage 5 — `Trace` and `Result` — `src/summer4/results/`

```python
@dataclass(frozen=True, slots=True)
class Trace:
    times: TimeAxis
    values: PropertyData | Any
    dims: tuple[str, ...]        # ("time", "compartment"), ("draw", "time"), ...

@register_pytree_node_class
@dataclass(frozen=True, slots=True)
class Result:
    times: TimeAxis
    traces: Mapping[str, Trace]
    solver: SolverInfo | None = None
```

**The axis rule:** time is the second-to-last axis, the aligned (compartment or
edge) axis is last, anything left is free. This extends `PropertyData`'s existing
"compartments last" invariant and composes — `vmap` over draws gives
`(draw, time, comp)` and time stays at `-2`. summer3wip's
`dims=["time","compartment"]` assumes time is axis 0 and breaks under `vmap`.
Even so, **never address axes by position in query code**; resolve through
`dims.index("time")`, because reductions move them.

**One flat mapping, no privileged `.compartments` / `.flows` namespaces.** The
user chose the keys in the save plan; `result["compartments"]` is a key like any
other, and it is simply what the default plan names it.

Query surface on `Trace`:

| Traceable | Host-side only |
|---|---|
| `select`, `sum_over`, `total`, `partition`, `group_by` | `to_frame()` (polars), `to_pandas()` (DatetimeIndex) |
| `between`, `at`, `at_times` (interpolated) | `plot()` |
| `resample(7 \| "ME", how=)`, `reduce_by(grouping, how=)` | `partition` / `group_by` **keys** |
| `rolling(7, how="mean")`, `cumulative()` | |

`reduce_by(grouping, how=)` is the primitive; `resample(...)` is sugar that
builds and caches a `TimeGrouping`. Because both integer and calendar rules end
in the same static-segment-id path, traceability never depends on an argument's
*value* — the objection the parent plan raised against a polymorphic `resample`
dissolves once calendar grouping is traceable.

`partition` / `group_by` hand back traced arrays with static group membership, so
they work inside a loss provided the loop unrolls; document that they must not be
called inside `lax.scan`.

`SolverInfo` is **typed optional fields, not a free-form `extras` dict**
(`solver`, `num_steps`, `num_accepted_steps`, `num_rejected_steps`,
`result_code`, `max_steps`, `dense`). estival's untyped `extras` became
load-bearing within one release because nothing forced a name to be documented.
Phase 2 fills `solver` and `num_steps`; Phase 3 fills the rest.

Deliberately **not** on `Result`: `params` (drags the parameter tree through
every jit boundary — offer host-side `with_params()` for provenance) and `model`
(same, plus it would make `Result` unpickleable).

---

## Stage 6 — `CompiledModel.run()` and the Euler fast path

```python
def run(self, params, y0, *, t0, t1=None, dt, steps=None,
        save=EVERYTHING, solver="euler", epoch=None) -> Result: ...
```

Exactly one of `t1` / `steps`. Times stay off `FlowModel` — L1 is in the ledger's
"never reaches `full`" table precisely because `FlowModel` owns flows over a map
and solver options live elsewhere. `euler` and `numpy_euler` stay public and
unchanged as the low-level seam Phase 3 swaps diffrax into behind `solver=`.

Two save mechanisms, driven by the same plan:

- **General path.** A save `fn` calls `observe`. Works for any `ts`.
- **Euler fast path.** Emit the wanted quantities in the `lax.scan` `ys` slot at
  zero extra cost. `ys` must be uniform across iterations, so use a **nested
  scan** — outer over `n_saves`, inner `fori_loop` over `stride` Euler steps —
  keeping the buffer at `n_saves` rather than `n_steps`. Requires the save grid
  to be an arithmetic subgrid of the step grid: check host-side that
  `(ts - t0) / dt` is near-integer and equally spaced, and include `t0` as the
  first save point. Otherwise fall back to the general path plus `at_times`.

---

## Ledger, docs and the notebook

**Ledger** (`docs/evaluation/coverage-ledger.md`) — restate the packages block in
terms of the plan's phases, as agreed:

- `<!-- ledger:packages -->`: **WP2 closes `L3 L6 L7 V1 T1 T2 D2 D3 D4 D5`**;
  WP4 keeps `D1 D6 D7`; WP8 (real-world time) disappears into WP2. Update the
  WP2/WP4 prose sections to match.
- API rows to `full`: L3 (`run()` → `Result`), L6/L7 (`to_frame` / `to_pandas`),
  V1 (`run()` over `euler`; **V2 stays `none`** until Phase 3), T1/T2 (`Epoch`;
  `Result.times.epoch`), D2 (`Compartments(where=)`), D3 (`sum_over` / `total`),
  D4 (`cumulative()`), D5 (`SaveFn` plus trace arithmetic).
- Textbook and summer2docs rows: **re-derive from each row's `Blocker` column**
  rather than trusting a list here. Expected: chapters 3, 5, 6, 9, 11 → `full`;
  8 `none` → `partial` (compartment outputs yes, flow outputs Phase 4); 2, 4, 10
  stay `partial` (initial population is WP3, sojourn-time and Rt are Phase 4);
  7 stays `partial` (needs Runge–Kutta). `examples/11-flows-between-strata` →
  `full`; `01-basic-model` stays `partial` on initial population.
- Then `pixi run coverage-write` and update the quoted totals in
  `docs/evaluation/index.md`. `tests/test_coverage_ledger.py` fails on a stale
  progression table or an unsupported quoted total.

**Docs** — `AGENTS.md` forbids describing shipped behaviour in the future tense.
Every page that says "final state only" or "no results object" needs rewriting:
`docs/user/index.md` (the admonition), `docs/api/index.md` ("What is not here"),
`docs/index.md`, `README.md`, `docs/user/07-from-summer2.md`,
`docs/textbook/index.md` and `roadmap.md`, `docs/evaluation/*`, and the two
notebooks that assert the limitation (`docs/getting-started/quickstart.ipynb`,
`docs/textbook/02-model-structures.ipynb`). Add a user-guide chapter
`docs/user/09-running-and-results.ipynb` and the new symbols to
`docs/api/summer4.rst`.

**Notebook** (the acceptance bar) —
`examples/notebooks/04-running-and-results.ipynb`: run an SIR with age, plot the
trajectory, select by date, aggregate by age, resample to calendar months,
take a 7-day rolling mean, show `describe(plan)` reporting the footprint
*before* running, and close with a `jax.jit` + `jax.grad` loss ending in
`at_times`. `y0` is still built by hand with `pm.select` — initial population is
WP3.

**Dependencies** — add `pandas` to `[feature.dev.dependencies]` and
`[feature.docs.pypi-dependencies]` in `pixi.toml` and as an optional extra in
`pyproject.toml`. Guard `to_pandas` tests with `pytest.importorskip` so the
traced path never acquires a hard pandas dependency.

---

## Verification

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch && pixi run coverage
pixi run -e docs docs-strict
pixi run test-all          # both pinned JAX 0.6.x and latest
```

New test modules `tests/test_time.py`, `tests/test_results.py`, plus additions to
`tests/test_flows.py`. Gates:

- **Shapes before solving.** `describe(plan)` returns correct shapes for a plan
  with mixed `ts`, with no solve; a `SaveFn` returning a dynamically-shaped
  array fails there with a readable message, not inside `lax.scan`.
- **Euler equivalence.** The nested-scan trajectory's final state equals
  Phase 1's `euler` final state, and every step matches `numpy_euler`.
- **Differentiability.** `jax.jit` and `jax.grad` of a loss ending in
  `at_times` both trace.
- **Time.** `Epoch` round-trips exactly for integer steps; `locate` / `window`
  accept dates; a host-side call on a traced axis raises clearly; an unsupported
  calendar rule raises pointing at `to_pandas()`.
- **Traceable time ops.** `resample("ME", "sum")` inside `jit` equals a
  host-side reference computed with stdlib `datetime` (**not** pandas — the
  reference must not import the thing it is validating); `rolling(7, "mean")`
  matches a numpy sliding window including the `min_periods` edges;
  `cumulative()` matches `np.cumsum`; `at_times` interpolates an off-grid time
  and is an exact no-op gather on-grid.
- **Axis rule.** `vmap` over parameter draws yields `dims == ("draw", "time",
  "compartment")` with time still at `-2`.
- **State.** An empty `ledgers` mapping round-trips through `jit` and `vmap`.
- **Hashing.** Two `SavePlan`s differing only in `ts` do not collide; a `SaveFn`
  lambda rebuilt in a loop retraces (assert the documented trap).
- **`Result` pytree.** Flattens and unflattens; carries no params and no model.

Manual validation is the notebook: run it in `pixi run notebook`, read it as a
user would, and check the story is one a modeller would actually write.
