# Promote flows to core summer4, and build a queryable `Result`

## Context

`summer4` ships exactly one layer today: the compartment taxonomy (`Property`,
`Trait`, the Kleene selector algebra, `PropertyMap`). Everything above it —
flows, rates, adjustments, a solver — exists as a tested 1 285-line prototype in
`explorations/flows/prototype.py`, deliberately outside the package.
`docs/evaluation/coverage-ledger.md` is the authoritative record of what that
costs: **6 of 52** summer2 API symbols are complete; promoting the spike as-is
reaches **21 of 52** and still publishes nothing, because the spike scoped out
timeseries — `euler()` returns the final state only. Every summer2 notebook and
nearly every textbook chapter ends by running a model and plotting a trajectory.

So the binding constraint moves from *flows* to *results* the moment the spike
lands. This plan does both: promote the flows work into core `summer4`, and
build the derived-outputs mechanism that makes the library able to complete a
user task.

The outputs design is deliberately **not** summer2's declarative
`request_output_for_flow(...)` registry. That is a flat global string namespace,
strictly declaration-order dependent (`assert source in
self._derived_output_requests`), build-time-only — a new question means
rebuilding and re-solving — and carries two overlapping, mutually-clobbering
pruning mechanisms (`save_results=` and a mutable global whitelist that silently
overrides it). Instead this is a summer3-style **`Result` object** returned from
a run and queried afterwards with the selector architecture that already exists,
plus a `SaveAt`-style save plan for the calibration case where dense outputs are
not wanted at all.

### Settled design decisions

1. **Flow polarity is an edge map plus Kleene absence.** A flow's edges become
   their own `PropertyMap`-shaped table — one row per edge, compartment
   properties doubled under `@source` / `@dest` namespaces. `Source(sel)` /
   `Dest(sel)` wrap ordinary selectors and compose with `&`, `|`, `~`. On an
   `ExitFlow` the dest columns are `-1`, so `Dest(...)` is Kleene *unknown*
   rather than an error — which is exactly why `Source(s) | Dest(s)` and
   polarity-agnostic requests work uniformly across exit, entry and transition
   flows. This fixes the two things that sank `summer3proto`'s `polarized`
   experiment: no either/both, and selectors that `KeyError` on the wrong flow
   kind.
2. **Flow outputs are instantaneous rates**, with explicit reductions
   (`.integrate()`, `.incidence()`) on the time axis — not accumulated in solver
   state. summer3wip shipped instantaneous rates *without saying so*, which
   silently differs from summer2's midpoint-integrated flow outputs.
3. **diffrax lands in Phase 3**, right after `Result` is defined against the
   existing Euler. Later phases then inherit `SaveAt`/`SubSaveAt`, off-grid
   interpolated save times, dense output and solver statistics rather than
   re-implementing them. This pulls ledger WP7 forward, out of its planned order.
4. **One package.** `summer4.flows`, `summer4.results`, `summer4.jax`; one
   `feat/` branch per phase; JAX stays an optional extra, the taxonomy stays
   NumPy-only, and pandas is never a hard dependency (lazy import inside
   host-side methods only).

### Baseline

All work is cut from **`feat/coverage-ledger`** (not `main` — `main` has none of
the flows work or the ledger). Each phase is one `feat/` branch merged before the
next begins, and each ships tests, an `examples/notebooks/` entry, a
`plans/<slug>.plan.md`, and a coverage-ledger update in the same commit.

---

## Phase 0 — Taxonomy prerequisites (`feat/taxonomy-prereqs`)

Three changes to **already-public** taxonomy code that everything else needs.
Separated so the public-API break is reviewed on its own.

**`src/summer4/propertymap.py`**

- `__hash__`. `PropertyMap` sets `eq=False` with a custom `__eq__`, so Python
  sets `__hash__ = None` and maps cannot be pytree aux, dict keys or jit static
  args. Add a `_digest: bytes` slot computed in `__post_init__` (blake2b-16 over
  `codes.tobytes()` plus `parent_row`), and `__hash__ = hash((properties,
  history, _digest))`. `Property` and `Stratification` are already hashable
  (`Property._index` has `compare=False`). This is the recipe already proven as
  `DigestStatic` in `explorations/flows/propertydata.py:35`, which then
  disappears as a separate type. Measured in the `explore-datatypes` spike:
  digest equality is lost in jit-dispatch noise, whereas identity hashing costs
  ~7.4 ms on a rebuilt-but-equal map because it retraces.
- Public accessors, replacing the four privates the spike reaches into
  (`_prop_index`, `_kleene`, `_validate_selector`, `_label`):
  `column_index(prop) -> int`, `column(prop) -> NDArray[int16]`,
  `kleene(sel) -> NDArray[int8]`, `label(row) -> str`.
- Resolve the `partition` / `group_by` asymmetry flagged in
  `docs/evaluation/gaps.md:120`. `group_by` returns a `Groups[V](Mapping[tuple[Trait,
  ...], V])` instead of a generator; `partition` stays `dict[Trait, V]` (single
  property, keys unwrapped, and — its distinguishing contract — includes empty
  groups). This **breaks** `for traits, idx in pmap.group_by(age)`; take the
  break now at `0.1.0a0`. Call sites: `explorations/`,
  `docs/user/05-partitions-and-groups.ipynb`.
- While here, the cheap papercuts from `docs/evaluation/user-satisfaction.md`:
  `__len__`, `PropertyMap.from_properties([...])`, `to_frame()` (polars is
  already a dev dependency).

**`src/summer4/properties.py`** — reject `@` in a property name (require
`isidentifier()`). Phase 1 mangles column names with `@source` / `@dest`; a user
property literally called `age@dest` would silently alias a real column.

**`src/summer4/selectors.py`** — add `Source` and `Dest` to the `Selector` union
now, with raising arms in `PropertyMap._validate_selector` and `_evaluate`
("`Source()`/`Dest()` select flow edges, not compartments"). Widening the
existing union is the only option that type-checks: a separate `EdgeSelector`
union cannot reuse `And`/`Or`/`Not`, whose fields are typed `Selector`, so
`And(Source(x), Dest(y))` would fail mypy on the first realistic call site. The
alternative — making `And[S]`/`Or[S]`/`Not[S]` PEP-695-generic — is a recursive
generic alias that mypy handles badly and that changes a public signature.

**Ledger:** no API rows move (stays 6/52). Closes `gaps.md` 1.6 and four
Priority-4 rows.
**Notebook:** `examples/notebooks/02-taxonomy-ergonomics.ipynb` — hashing a map,
`from_properties`, `to_frame`, the new `Groups` mapping.

---

## Phase 1 — Promote the flows spike, with edge maps (`feat/flows-core`)

The thirteen-item promotion list from `explorations/flows/FINDINGS.md`, plus the
edge map, plus a compiled-model object. `explorations/flows/prototype.py` moves
to `src/summer4/flows/` and `propertydata.py` to `src/summer4/jax/`.

### 1a. Settle the `Present` / `Absent` binding question

This is the one open question that affects the already-public API
(`gaps.md:156`, `coverage-ledger.md:160`). **Decide: `Present` and `Absent` are
non-binding for pairing.** Binding requires a *value* to correspond with;
`Source(age.present())` is a pure filter on the source side and cannot determine
a pairing. Split `selector_properties` (`prototype.py:39`) into
`selector_values()` (binding — `Trait`/`IsIn` only) and `selector_properties()`
(mentioned — what `_validate_selector` wants). Nothing regresses: the case that
motivated binding, `TraitChain`/`TraitMatrix`, already passes
`extra_bound=frozenset({pairing.property.name})` explicitly at
`prototype.py:300`.

Ship a compile-time invariant that makes the wrong answer loud, behind
`strict_pairing: bool = True` with `extra_bound=` as the opt-out:

> If a property is named by no `Trait`/`IsIn` leaf on either side and is not in
> `extra_bound`, then `edge_map.moves_mask(prop)` must be all-`False`.

Under "Present binds", `TransitionFlow(source=age.present() & state["S"],
dest=state["I"])` drops `age` from the free set and silently fans people across
age bands; the invariant catches it at compile time instead of six months later
in a trajectory.

### 1b. `EdgeMap` — `src/summer4/flows/edges.py`

A genuine doubled `PropertyMap`, held by composition (not subclassed — an
`EdgeMap`'s rows are edges, and one passed where a compartment map is expected
would type-check and produce nonsense).

```python
@dataclass(frozen=True, slots=True)
class EdgeRoles:
    """How the join classified each property. Provenance, not truth."""
    bound: frozenset[str]; free: frozenset[str]
    dest_only: frozenset[str]; src_only: frozenset[str]; paired: frozenset[str]

@dataclass(frozen=True, slots=True)
class EdgeMap:
    pmap: PropertyMap                    # the compartment map the edges live over
    table: PropertyMap                   # n_edges rows: {p}@source, {p}@dest, + 2 markers
    src_idx: NDArray[np.int32] | None
    dest_idx: NDArray[np.int32] | None
    roles: EdgeRoles
```

Because `table` is a real `PropertyMap`, `_kleene`, `_evaluate`,
`_validate_selector`, the per-selector cache, `partition` and `group_by` are all
inherited unchanged — and **there is no `EdgeData` type**: a flow result is a
`PropertyData` whose map is `EdgeMap.table`. That reuse is the whole reason for
this shape.

`Source`/`Dest` lower by a **pure syntactic rewrite** to the mangled names, then
run on the existing evaluator. Two marker properties (`@source`, `@dest`) gate
each side:

```python
def rewrite(self, sel):
    match sel:
        case Source(inner=i): return _gate(self._src_gate, _rename(i, "@source"))
        case Dest(inner=i):   return _gate(self._dest_gate, _rename(i, "@dest"))
        case And(l, r): return And(self.rewrite(l), self.rewrite(r))   # Or, Not likewise
        case Everything() | Nothing(): return sel
        case _: raise TypeError(f"{type(sel).__name__} is a compartment selector; on a "
                                "flow it must be wrapped in Source(...) or Dest(...).")
```

The gate is load-bearing and easy to omit. Without it, `Dest(Everything())` on an
exit flow rewrites to `Everything()` and is TRUE on every edge — the opposite of
"no dest ⇒ unknown". Gating uniformly fixes it with no leaf special-casing,
because it is a no-op precisely on the leaves already UNKNOWN-on-absent. The gate
is `None` (skipped entirely) when every edge has that endpoint.

Two warts to document rather than hide: `Dest(Nothing())` is FALSE, not UNKNOWN
(harmless — both select no edges); and `~(Source(s) | Dest(s))` on a
single-endpoint flow is UNKNOWN everywhere, so "neither endpoint is s" selects
nothing. Users write `~Source(s)`. Both are inherent to Kleene.

Validate **before** rewriting, so errors name `age` and not `age@dest`. A bare
unwrapped `age["0-4"]` on an edge map is an **error**, not an implicit `Source` —
implicit-Source is a silent empty result on entry flows. `Everything()` /
`Nothing()` pass through.

`roles` records the join's classification (which `identity_join` currently
computes at `prototype.py:179-190` and throws away) for provenance and error
messages. It does **not** answer "which property does this flow move along?" —
those disagree, since `source=age["0-4"] & state["S"], dest=age["0-4"] &
state["I"]` binds `age` while moving along nothing. That answer is computed from
the table: `moves_mask(prop)` is `(src_code != dest_code) & both present`.

Cross-side predicates ("edges that cross an age band") are relational between two
columns and are not expressible in a per-column Kleene algebra. Rather than add a
`SameAs` selector node — which would be the first selector whose meaning depends
on the map's *shape* — let `select()` accept `Selector | NDArray[np.bool_]` and
have `moves_mask()` return the mask. One union member, no new node type.

`EdgeMap.labels()` must demangle (`"S_age=0-4 -> I_age=0-4"`); `table.labels()`
would render `age@source=0-4_age@dest=0-4`. `summer3wip`'s
`transition_flow_labeller` (`summer3/runners.py:22`) is the template.

### 1c. Fix the join while promoting it

- **Silent overflow in `_pack_keys` (`prototype.py:62`).** Each property consumes
  a full 16-bit field regardless of cardinality; `int64` is signed, so **3 free
  properties is the safe limit**. A fourth wraps silently — two compartments
  collide onto one key and are joined to the wrong destination, producing a
  plausible-looking trajectory. Delete `_pack_keys` and reuse
  `np.unique(..., axis=0)`, which `PropertyMap.group_by` already uses
  (`propertymap.py:206`), uniquing over the concatenated source and destination
  rows so both share a key space. No limit, and one fewer grouping
  implementation.
- **Vectorise the group-expansion loop** (`prototype.py:203-219`), a Python
  double loop over groups × members building lists — O(n_edges) in the
  interpreter. `compile()` becomes a user-visible step in this design, so on a
  large model this is minutes of apparent hang. It is `np.repeat` / `np.tile`.
- **Split `ActualizedFlow` by kind.** Today `src_idx` and `dest_idx` are both
  `NDArray | None` and four sites re-assert non-`None` with `cast` plus
  `raise RuntimeError` (`prototype.py:648, 886, 1164, 1192`).
  `TransitionEdges` / `ExitEdges` / `EntryEdges` make each field non-optional
  where it exists and the `match` exhaustive. This is the single largest
  `mypy --strict` win available and it is free.

### 1d. `CompiledModel` — `src/summer4/flows/compiled.py`

`FlowModel.compile()` currently hides `actualized` and `flow_meta` in a closure
and returns a bare callable, so nothing can ask "what edges does flow X have?"
without re-running `actualize` — which is exactly what the spike notebooks do.

```python
@dataclass(frozen=True, slots=True)
class CompiledModel:
    pmap: PropertyMap
    order: tuple[str, ...]                  # topologically sorted flow names
    flows: Mapping[str, FlowEdges]
    edge_maps: Mapping[str, EdgeMap]
    derived_fn: DerivedFn | None
    computed_paths: tuple[tuple[str, ...], ...]
    _digest: bytes = field(init=False, repr=False, compare=False)

    def edges(self, name: str) -> EdgeMap: ...
    def observe(self, t, y, params) -> SaveContext: ...      # Phase 2
    def vector_field(self, t, y, params) -> Array: ...
    def __hash__(self) -> int: return hash(self._digest)
```

Entirely static, zero pytree leaves — pass it as `jax.jit(..., static_argnums=0)`
rather than registering it. **The digest must cover the float arrays, not just
structure**: two models identical in topology but differing in `split`
proportions have identical `src_idx`/`dest_idx` and different `weight`; hashing
only names and shapes collides them and silently reuses a jit cache entry
compiled with the wrong split. Hash per flow: name, kind, the edge table digest,
`weight.tobytes()`, `scale.tobytes()`, the rate expression tree, and the adjust
masks. Document two jit-cache traps: `derived_fn` and `Transform.fn` hash by
identity, so a lambda defined inside a loop recompiles every call.

### 1e. `mypy --strict` on the promoted code

`explorations/` is linted but not type-checked; `src/summer4` is
`mypy --strict`. Ranked by pain:

- **`xp: Any` backend dispatch is the worst.** It threads through `_eval_rate`,
  `_align_rate`, `_apply_adjustments`, `_sum_mass_over`, `_scatter_add` and
  `vector_field`; every `xp.` call returns `Any`, so strict mode checks nothing
  in the hot path — including `_scatter_add`, exactly where a
  `np.add.at` / `.at[].add` divergence would hide. **Drop the dual backend**: the
  compiled field is JAX-only in `summer4.jax`, and the NumPy path survives as a
  reference implementation in tests. (Fallback if dual must stay: a
  `Protocol ArrayNamespace` over the ~12 functions actually used.)
- `_is_property_data` comparing `type(value).__name__` (`prototype.py:1119`) and
  `_is_jax_value` sniffing `__module__` (`:1217`) exist only because the spike
  cannot import the JAX module. Promotion *removes* `Any` here: `isinstance`.
- Contain `derived: Any` with PEP 695 type params — `FlowModel[P, D]`,
  `FieldRef` paths validated against `D` at compile time — so `Any` survives only
  inside one `cast` in `_lookup_path`'s body. Replace
  `DerivedFn = Callable[..., Any]` with a `Protocol` carrying the real keyword
  parameters.
- **Not all 1 285 lines are worth promoting.** `derived_refs` /
  `_nested_schema` / `_field_ref_tree` / `_field_annotations`
  (`prototype.py:463-514`) are the most `Any`-dense and the most fragile —
  string-annotation resolution via `sys.modules` will fight
  `from __future__ import annotations` under strict mode. The flows do not depend
  on them; land them last, or replace with explicit schema registration.

### 1f. Retire the spike (easy to miss, and `pixi run test` fails without it)

- Delete `explorations/flows/`, `tests/test_explore_flows.py`,
  `tests/test_flows_docs_sync.py`, and the `explore-flows` pixi task.
  `test_flows_docs_sync.py` asserts the published `docs/dev/flows/` cells are
  byte-identical to the spike notebooks; with the spike gone it fails.
- Move `docs/dev/flows/*` into the user guide, rewritten against the public API.
- Rewrite `docs/dev/explorations.md` (the promotion list is now history) and
  `docs/evaluation/if-promoted.md` (its whole premise — "what if the spike were
  promoted" — is answered). Update `docs/index.md` and the "What is not here"
  section of `docs/api/index.md`; `AGENTS.md:92` forbids describing shipped
  behaviour in the future tense or planned behaviour in the present.
- **The ledger's `Spike` column becomes meaningless.** Collapse the API,
  textbook and summer2docs tables to a single status column; update
  `scripts/coverage_report.py` (`read_block` expects exactly 7 API cells, and
  `progression` special-cases WP1 by replacing the whole state with the `Spike`
  column) and `tests/test_coverage_ledger.py` (which asserts
  `rank[Spike] >= rank[Shipped]` per row). Rewrite the
  `<!-- ledger:packages -->` block so the WP rows match these phases, keeping
  WP3/WP5/WP6/WP9/WP10 as the unclaimed remainder.

**Ledger:** WP1 applied — every `Spike` status becomes the shipped status.
6 → 21 complete, 7 → 31 covered. Plus **Q3 `query_flows` → `full`**: the edge map
*is* a genuine query API, so remove Q3 from the "never reaches `full`" table.
**22 / 52 (42%)**.
**Notebook:** `examples/notebooks/03-flows.ipynb` — an SIR with ragged severity,
ageing via `TraitChain`, an adjustment with `where=`, and `model.edges("infection")`
queried with `Source`/`Dest`.

---

## Phase 2 — Time, trajectories and `Result` (`feat/results`)

### 2a. Time — `src/summer4/time.py` (NumPy + stdlib only)

`Epoch` is a pure host-side affine map, ported essentially as-is from
`summer3wip/summer3/utils.py:56` — `ref_date` plus a `unit: timedelta`. Dates
never enter the JAX path; a `DatetimeIndex` is attached once at result-wrapping
time. Relax two limits the predecessors carried: summer2 hard-codes
`timedelta(1)` so its `unit` parameter is dead, and summer3wip's `dti_to_epoch`
assumes a uniform step.

```python
@dataclass(frozen=True, slots=True)
class TimeAxis:
    values: NDArray[np.float64] | Array     # numeric model time; may be a tracer
    epoch: Epoch | None = None
    kind: Literal["grid", "explicit", "steps"] = "grid"

    def locate(self, when) -> int: ...            # host-side
    def window(self, t0, t1) -> slice: ...        # host-side
    def weights_for(self, ts) -> tuple[NDArray[np.int32], NDArray[np.float64]]: ...
    def as_index(self) -> "pd.Index | pd.DatetimeIndex": ...   # lazy pandas import
```

`values` is a pytree leaf (an adaptive solver's `sol.ts` is traced); `epoch` and
`kind` are aux.

### 2b. The axis rule

> **Time is the second-to-last axis; the aligned (compartment or edge) axis is
> last; anything to the left is free** (vmap draws, chains, scenarios).

This extends `PropertyData`'s existing "compartments last" invariant and
composes: `vmap` over parameter draws gives `(draw, time, comp)` and time stays
at `-2`. summer3wip's `dims=["time","compartment"]` assumes time is axis 0 and
breaks the moment you vmap.

Even so, **never address axes by position in query code** — reductions move them.
Carry `dims` explicitly:

```python
@dataclass(frozen=True, slots=True)
class Trace:
    """One named quantity over time. The unit the query surface operates on."""
    times: TimeAxis
    values: PropertyData | Series
    dims: tuple[str, ...]            # ("time", "compartment") / ("draw", "time")
```

### 2c. `Result` — `src/summer4/results/`

```python
@register_pytree_node_class
@dataclass(frozen=True, slots=True)
class Result:
    times: TimeAxis
    traces: Mapping[str, Trace]      # flat; keys are the ones the user chose
    solver: SolverInfo | None = None
```

**One flat mapping, no privileged `.compartments` / `.flows` namespaces** — the
user chose the keys in the save plan, and a parallel `result.flows[...]`
namespace reintroduces the stringly-typed lookup this design exists to avoid.
`result["compartments"]` is a key like any other; it is simply what the default
plan names it.

`SolverInfo` is **typed optional fields, not a free-form `extras` dict**
(`solver`, `num_steps`, `num_accepted_steps`, `num_rejected_steps`,
`result_code`, `max_steps`, `dense`). estival's untyped `extras` became
load-bearing within one release because nothing forces a name to be documented.
If a backend has something new to report, add a field.

Deliberately **not** on `Result`: `params` (would drag the whole parameter tree
through every jit boundary and vmap — offer `with_params()` host-side for
provenance) and `model` (same, plus it would make `Result` unpickleable).

**Reserve one extension point now.** Decision 2 — instantaneous rates, reductions
on the `Result` — is right for *outputs* and wrong for *feedback*: a cumulative
quantity that feeds back into a rate (a vaccination coverage ledger) must be
state, not a post-run reduction. So do not let `y.shape[-1] == pmap.size` become
a hard invariant baked into `_scatter_add`, `PropertyData.check` and the Euler
carry. Introduce a `State` pytree with `compartments: PropertyData` and a
currently-empty `ledgers: Mapping[str, Array]`, even though Phase 2 only
populates the first. Retrofitting this after `Result` and `SavePlan` freeze is
expensive.

### 2d. The query surface

State the split as a rule, because it is what users get wrong:

> **All index arithmetic is host-side, computed at trace time from static
> `PropertyMap` / `TimeAxis` data. Only gathers, sums, lerps and arithmetic are
> traced.**

That is estival's `TargetEvaluator` lesson generalised: the Python loop over
targets unrolls at trace time and the integer index array is a constant.

| Traceable (usable in a calibration loss) | Host-side only |
|---|---|
| `select`, `sum_over`, `total` | `to_frame`, `to_series`, `.plot()` |
| `between`, `at`, `at_times` | `resample_calendar(rule)` (pandas offsets) |
| `integrate`, `incidence` | `partition` / `group_by` *keys* |
| `resample(factor, how)` | |

`resample(7, "sum")` on a daily grid is a reshape-and-reduce and is traceable;
calendar resampling is pandas. **Two methods, not one with a polymorphic
argument** — otherwise the traceability of a loss depends on the *value* of a
string. `partition` / `group_by` hand back traced arrays with static group
membership, so they are usable inside a loss provided the loop unrolls; document
that they must not be called inside `lax.scan`.

`side: Side | None` on `sum_over` / `partition` / `group_by` is `None` for
compartment traces and **required** for edge traces. This is the honest cost of
the mangling scheme: a `Property` is not a column on an edge map, `age@source`
is. Ship `side=` now; `sum_over(Source(age))` sugar needs `Source` overloaded
over `Property | Selector` and can come later once proven under `mypy --strict`.

Representative call sites, which double as the notebook's outline:

```python
# Prevalence over a date range
res["compartments"].select(state["I"]).total().between(date(2021,1,1), date(2021,6,30))

# Per-location trajectories
res["compartments"].select(state["I"]).partition(location)

# Two-property grouping
for (band, loc), tr in res["compartments"].select(state["I"]).group_by(age, location).items():
    ...

# A scalar at one date
res["compartments"].select(state["I"]).total().at(date(2021,3,14))
```

### 2e. The save plan — `src/summer4/results/plan.py`

Name it **`SavePlan`, not `SaveAt`** — `diffrax.SaveAt` will be in the same
module in Phase 3.

```python
type Quantity = Compartments | FlowMass | ComputedValue | SaveFn

@dataclass(frozen=True, slots=True)
class SaveRequest:
    what: Quantity
    ts: NDArray[np.float64] | None = None     # None -> the plan's default grid

@dataclass(frozen=True, slots=True)
class SavePlan:
    requests: Mapping[str, SaveRequest] = field(default_factory=dict)
    ts: NDArray[np.float64] | None = None     # None -> the solve grid
    dense: bool = False                        # Phase 3
    solver_stats: bool = True
```

`result[key]` is request-shaped because **the user wrote the key** — `traces` is
built from `plan.requests` at trace time, with no runtime name resolution and no
`get_derived_outputs_df()["incidenceXage_0"]` convention.

The trivial case desugars into the general one rather than being a parallel path:
an empty `requests` is the sentinel, and `CompiledModel.expand()` fills it with
`compartments`, every flow, and every computed value. `model.run(params)` with no
`save=` passes `EVERYTHING`; there is exactly one code path below `expand`.

**Getting per-flow mass out at all.** Today `flow_values[name] = mass`
(`prototype.py:1189, 1199`) is a local in the `vector_field` closure — the
quantity we want is already computed every step and then discarded, and
`_euler_jax`'s `lax.scan` emits `None` in its `ys` slot (`:1255`). Returning
`(dy, aux)` from `vf` is not an option: it breaks the contract that a vector
field returns a `y`-shaped pytree, which Phase 3's `ODETerm` relies on. Use two
mechanisms driven by the same `SavePlan`:

- **General:** factor the closure body into
  `observe(t, y, params) -> SaveContext` (fields `t`, `y`, `dy`, `derived`,
  `flows: Mapping[str, Array]`), with `vector_field` returning `observe(...).dy`.
  A save `fn` calls `observe`. Costs one extra field evaluation per *save point*
  — ~10% under diffrax at `dt≈0.1` with daily saves. This is what summer3wip
  pays via `jax.vmap(get_flow_values)(sol.ts, sol.ys)`, and it is fine.
- **Euler fast path:** emit the wanted quantities in the `lax.scan` `ys` slot at
  zero extra cost. `ys` must be uniform across iterations, so use a nested scan —
  outer over `n_saves`, inner `fori_loop` over `stride` Euler steps — which keeps
  the buffer at `n_saves` rather than `n_steps`. Requires the save grid to be an
  arithmetic subgrid of the step grid; otherwise fall back to the general path
  plus `at_times`.

**Shapes are known before running**, via `jax.eval_shape` on each group's save
`fn` (diffrax's own trick at `_integrate.py:1246`). This does three jobs, and the
third pays for it: it sizes the Euler buffers; it validates that a user's
`SaveFn` escape hatch returns statically-shaped output, rather than failing as an
opaque `lax.scan` structure mismatch mid-solve; and it powers
`model.describe(plan)`, which prints every output's shape and memory footprint
**without running the model**. A 200k-edge flow over 3 650 days is
`200000 × 3650 × 8 B ≈ 5.8 GB` — a number the user should see before they wait
for it. Which is the argument for `sum_over` living *in the request*: the
reduction happens inside the save `fn`, so the buffer is `(3650, 16) ≈ 470 kB`.
That is the entire reason `SubSaveAt.fn` exists and it belongs in the first line
of the `FlowMass` docstring.

Prune the `SaveContext`: a `SaveFn` touching one flow would otherwise force all
of them to be materialised. Walk the plan for the flows actually read and
restrict `observe`'s `flows` mapping — `_topo_sort` already computes the
dependency edges this needs.

**Ledger:** L3 L6 L7 V1 T1 T2 D2 D3 D4 D5 → `full`. **32 / 52 (62%)**. Textbook
2–6, 9, 11 become publishable; summer2 `01-basic-model` becomes runnable.
**Notebook:** `examples/notebooks/04-running-and-results.ipynb` — run an SIR,
plot the trajectory, select by date, aggregate by age, resample weekly. `y0` is
still built by hand with `pm.select` (initial population is ledger WP3, a natural
small follow-on).

---

## Phase 3 — diffrax backend and solver info (`feat/diffrax-solver`)

Put diffrax behind the solver seam that `euler` already defines. `diffrax>=0.7.2`
is declared in `pyproject.toml`'s `jax` extra and in `pixi.toml` today, and is
entirely unused; declare `equinox` explicitly rather than relying on it arriving
transitively.

- Each `SavePlan` group (requests sharing a `ts`) becomes one `SubSaveAt`, so
  `saveat = SaveAt(subs={name: SubSaveAt(ts=g.ts, fn=g.fn)}, dense=plan.dense)`.
  `sol.ys` returns `{group: {key: array}}` and is restructured into
  `Result.traces` at trace time from the static spec. Group on
  `blake2b(ts.tobytes())`, not `id()`, or two targets with equal-but-distinct
  arrays pay two full save passes.
- Off-grid save times now work properly: diffrax drains every requested `ts`
  falling inside `(tprev, tnext]` and **interpolates** via the solver's local
  interpolant. This is the fix for estival's sharp edge, where
  `model_times.get_loc(t)` is an exact lookup and a target time not landing
  exactly on a gridpoint is silently dropped by set intersection.
- `SolverInfo` is populated from `sol.stats` and `sol.result`. diffrax's
  `RESULTS` enum carries human-readable messages on the members — worth mirroring
  so a failure survives jit as an integer but prints as prose. summer2 discards
  the entire `Solution` except `ys`, so a truncated solve produces silently wrong
  arrays.
- `dense=True` exposes `result.evaluate(t)`. Note its cost honestly: it allocates
  `max_steps` worth of interpolation coefficients and requires a finite
  `max_steps` — strictly heavier than any `ts`-based saving.
- Keep `euler` as a supported reference stepper. Textbook chapter 7 compares
  manual evaluation, Euler and Runge–Kutta and needs all three.
- **The acceptance test for this phase is that Phase 2's loss-function source is
  unchanged.** Both backends end in `.at_times(TARGET_TS)`, which is a no-op
  gather when the times already coincide.

**Ledger:** V2 → `full`. **33 / 52 (63%)**. Textbook 7 fully portable.
**Notebook:** `examples/notebooks/05-solvers.ipynb` — Euler vs Dopri5 against an
analytic solution, adaptive stepping, `solver.num_steps`, dense evaluation.

---

## Phase 4 — Flow outputs and polarity queries (`feat/flow-outputs`)

Wire the Phase 1 `EdgeMap` through the Phase 2 save plan and Phase 3 solver.

- `FlowMass` requests produce a `Trace` whose `values` is a `PropertyData` over
  `EdgeMap.table`, so the entire Phase 2 query surface applies with `side=`.
- `sum_over(prop, side="source" | "dest")` replaces the spike's `_sum_mass_over`
  (`prototype.py:581`), whose call site hard-codes the side by flow kind
  (`:648`) so that `transition.sum_over(location)` always groups by the *source*
  compartment. Both directions are needed.
- `.integrate()` and `.incidence()` on the time axis, with the semantics stated
  in the docstring. **Be explicit about the error.** Trapezoid over a save
  interval is second-order accurate, whereas summer2's accumulated incidence is
  exact for the solver's own quadrature; for calibration against case counts this
  is a real bias and it is *not* fixable on the `Result`, because the information
  is gone. Mitigations, in order: save on a finer grid than you calibrate on and
  `resample`; `integrate(method="simpson")`; eventually an opt-in accumulator
  state — the same machinery the coverage ledger in 2c reserves. Do not silently
  ship trapezoid and call it incidence.
- `ComputedValue` requests capture `derived_fn` outputs into the `Result`,
  covering D6/D7. Validate `ComputedValue.path` against the derived schema at
  compile time with `get_type_hints`, as `derived_refs` already does for
  `FieldRef` — otherwise a typo surfaces as an `AttributeError` inside a traced
  scan body.

```python
# Weekly incidence by the age of the newly infected
(res["incidence"]                       # (time, edge) instantaneous rates
   .incidence()                         # (time-1, edge) counts per interval
   .sum_over(age, side="dest")
   .resample(7, how="sum")
   .to_frame())

# A polarity-restricted flow
res["transfers"].select(Source(age["0-4"]) & Dest(~state["R"])).total()

# Only the edges that actually cross a band
res["ageing"].select(model.edges("ageing").moves_mask(age))
```

**Ledger:** D1 D6 D7 → `full`. **36 / 52 (69%)**. The API-row delta understates
this phase: it is the largest design chunk and it is what moves textbook 3, 4, 8
and 10 and summer2 `03-derived-outputs`, `04-flow-types`,
`10-derived-outputs-stratified` and `11-flows-between-strata`.
**Notebook:** `examples/notebooks/06-flow-outputs.ipynb` — incidence by age,
source-vs-dest aggregation side by side, an exit flow queried with `Dest(...)` to
show Kleene-unknown, and an explicit rate-vs-incidence comparison. **This is the
notebook to user-test hardest** — the polarity surface is the least proven part
of the design.

---

## Phase 5 — Sparse outputs and calibration targets (`feat/sparse-targets`)

The calibration case: sparse observed data, and no wish to materialise dense
outputs at all.

```python
@dataclass(frozen=True, slots=True)
class Target:
    key: str
    times: NDArray[np.float64]
    values: NDArray[np.float64]

    @classmethod
    def from_series(cls, key, s: "pd.Series", epoch: Epoch) -> Target: ...
    def contribute(self, plan: SavePlan) -> SavePlan: ...   # merges its ts into plan[key]
```

Two strategies, chosen per backend, with the *same* user-facing code: under
diffrax the target's times are merged into the group's `ts` and hit exactly;
under Euler the model saves on its grid and `TimeAxis.weights_for` supplies a
precomputed `(idx, weight)` pair applied by `at_times`.

Adopt estival's two structural ideas and drop its two problems. Keep: compiling a
declarative `Target` (a pandas Series plus a key) into an evaluator holding a
**precomputed integer index array**, and unrolling the Python loop over targets
at trace time so each target becomes a static gather in the jaxpr. Keep also the
touch where a target contributes its own nuisance parameter (a dispersion prior).
Drop: the exact-index requirement that silently discards off-grid observations
(Phase 3 interpolates), and the mutable global whitelist that estival has to flip
on and off around two `get_runner` calls — here, the plan *is* the request, so a
likelihood run and a reporting run are two `SavePlan`s, not two transient states
of one shared model.

**Ledger:** no API rows — sparse saving has no summer2 equivalent, so the
percentage does not move, and the plan should say so rather than inflating it.
Its value is that it unblocks WP10 (calibration) and textbook 20, and makes
D1–D7 usable at calibration scale. Add it to the packages block as a declared WP
with no `Closes` IDs, the way WP9 and WP10 already are.
**Notebook:** `examples/notebooks/07-targets.ipynb` — fit an SIR to ~40 scattered
observations, showing `model.describe(plan)` reporting the memory the sparse plan
saves against the dense one.

---

## Remaining to the 46/52 ceiling

Unchanged from the ledger, and not part of this plan: WP3 initial population
(L4 L5 S8), WP5 time-varying function library (P5–P8), WP6 force of infection and
mixing (F7 F8 A4 M1 — the largest genuinely unprototyped design problem left),
WP9 contact survey data, WP10 calibration. Six rows stay below `full` by design;
this plan removes **Q3** from that list, leaving five.

---

## Verification

Per phase, before asking for a merge:

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch
pixi run coverage
pixi run -e docs docs-strict
```

- `check-branch` fails unless a branch touching `src/summer4/*.py` also touches
  `tests/`, `examples/notebooks/` **and** `plans/` — so every phase needs its
  notebook and its plan file on the branch.
- `pixi run coverage` (`scripts/coverage_report.py --check`) fails if the
  generated progression block is stale **or** if `docs/evaluation/index.md` no
  longer contains the literal `**{n} of {total}` strings. Run
  `pixi run coverage-write` and update that page's quoted totals in the same
  commit.
- `pixi run -e docs docs-strict` executes every notebook on the site with
  `nb_execution_raise_on_error = True`; a docs build is a test run.
- `pixi run test-all` runs the suite against both the pinned JAX 0.6.x and the
  latest JAX environment.

Phase-specific gates:

- **Phase 0** — a property-based test that `hash(a) == hash(b)` whenever
  `a == b`, over Hypothesis-generated maps including rebuilt-but-equal ones, plus
  a jit-cache-hit test proving a rebuilt map does not retrace.
- **Phase 1** — the `strict_pairing` invariant is tested directly:
  `source=age.present() & state["S"], dest=state["I"]` must raise. A Kleene truth
  table for `Source`/`Dest` over all three flow kinds, including
  `Dest(Everything())` on an exit flow being UNKNOWN. A regression test for the
  `_pack_keys` overflow — a four-free-property model whose edges are verified
  against a brute-force join.
- **Phase 2** — `output_shapes()` returns correct shapes **before** any solve,
  for a plan with mixed `ts`. A trajectory from the nested-scan Euler matches the
  Phase-1 `euler` final state. `jax.jit` of a loss that ends in `at_times` traces
  and differentiates.
- **Phase 3** — diffrax and Euler agree to `rtol=1e-4` on an analytic SIR; that
  same loss function's **source is byte-identical** to Phase 2's; a target time
  deliberately placed off-grid is hit by interpolation rather than dropped.
- **Phase 4** — `sum_over(side="source")` and `side="dest"` differ on a flow that
  crosses a stratum, and each matches a hand-computed expectation.
  `.incidence()` on a fine grid converges to the analytic integral.
- **Phase 5** — a fit against ~40 sparse observations reaches the known
  parameters; `describe()` reports a smaller footprint for the sparse plan than
  the dense one, and the two agree at the observation times.

Manual validation between phases is the notebook: run it in the `nb` environment
(`pixi run notebook`), read it as a user would, and check that the story it tells
is one a modeller would actually write.
