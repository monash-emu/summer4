---
name: initial-population
description: WP3. Adds a parameterisable, declarative initial population (base totals, ragged-aware balanced splits, splits by arbitrary functions, summer2-style adjusted splits) and makes the two per-run evaluation stages explicit — run start (a prepare hook plus hoisted parameter-only rate subtrees, including interpolator knots) and per timestep.
---

# WP3 — Initial population, and the run-start / per-step stage contract

> **Who this is for.** An implementing agent with little context. Follow the
> phases in order. Do exactly what each step says. Where this plan says **STOP**,
> stop and ask the user; do not improvise a design change. Every phase ends in a
> commit. Never commit on `main`.

## Context

`docs/evaluation/coverage-ledger.md` lists three API rows at `none` that WP3
closes: `L4` `set_initial_population`, `L5` `get_initial_population`, and `S8`
`set_population_split / adjust_population_split`. Today every notebook builds
`y0` by hand (`np.zeros` plus `pmap.select`, or `PropertyData.at[...].set`). The
TB-port ledger (`docs/evaluation/tb-ports.md`, branch `docs/tb-ports-plans`)
records the same gap as `KI3` (Kiribati: seed in `clin_inf`, reachability split
by `Param`) and `TM6` (tb_macro: even split over ragged strata). Textbook
chapter 13 and summer2 pages `01-basic-model`, `07-age-stratification` and
`InitialPopulationGraphobject` are blocked on it.

The user added two requirements beyond the ledger rows:

1. **Initial populations must be parameterisable**, including split ratios
   computed by arbitrary functions of parameters, and **split balancing**: the
   summer3 `build_istate` rule, where properties nobody specified are divided
   evenly among the compartments that carry them (ragged-aware), plus
   summer2's `adjust_population_split`, which redistributes a group's total
   under a filter.
2. **The two parameter-transform stages must be explicit**:
   - **Run start.** Evaluated once per `run` call, before any timestep. The
     values stay constant for the whole run: the initial state, derived
     constants, and interpolator knots.
   - **Per timestep.** Evaluated on every vector-field call. These read the
     run-start values, for example evaluating an interpolator whose knots were
     built at run start.

Today nothing runs at run start. `derived_fn`, every `FieldRef`, and every
`Interp` knot stack (`jnp.stack` in `_eval_rate`,
`src/summer4/flows/compiled.py:299-308`) are re-evaluated on every vector-field
call. The user chose the full option: write the contract down, add a run-start
`prepare_fn` hook, **and** hoist parameter-only rate subtrees (including
`Interp` knots) into the run-start stage automatically.

**Closes:** `L4` `L5` `S8` in the API ledger (44/52 → 47/52), and `KI3` `TM6`
in the TB-port ledger.
**Unblocks:** textbook 2 (to `full`) and 13, and summer2 `01-basic-model`,
`07-age-stratification` and `InitialPopulationGraphobject`.
**Supersedes:** the *WP3* section of `plans/tb-ports-feature-completeness.plan.md`
(on `docs/tb-ports-plans`). That sketch had two gaps, which this plan fixes:
- Its base-entry rule ("divided among the compartments its selector matches")
  double-divides when combined with its even default.
- It had no staging and no `where=` or `by=` splits.

WP3 no longer waits on WP13. Arbitrary math goes in callables, not new rate
nodes.

---

## Background the agent must know (read once)

| Thing | Where | Fact you need |
|---|---|---|
| `PropertyMap` | `src/summer4/propertymap.py` | `codes: int16[N, P]`, where `-1` means the row does not carry that property. `select(sel)` returns row indices, `column(prop)` returns a codes column, `get_property(name)`, `labels()`. `stratify(prop, where=)` makes ragged maps: only matched rows gain the property. |
| Rate nodes | `src/summer4/flows/rates.py` | `Const`, `Time`, `Interp`, `GaussianPulse`, `FieldRef`/`Param`, `FlowRef`, `BinOp`, `Reduce`, `Capture`, `ArrayConst`, plus `Multiply`/`Overwrite`/`Transform` adjustments. Custom nodes register via `register_rate_eval`. Tree walkers are `_flow_refs`, `_field_paths` and `_rate_bytes`. |
| Evaluator | `src/summer4/flows/compiled.py:222` `_eval_rate(expr, *, derived, flow_values, flow_meta, pmap, t, y_arr, captures)` | `FieldRef` resolves against `derived`, which is `derived_fn(params, y=, t=)` if a `derived_fn` is set, otherwise `params` (`_eval_derived`, `:622`). |
| Per-step loop | `CompiledModel.observe` (`compiled.py:683`) | Called for every vector-field evaluation. `vector_field` wraps it. |
| Solvers | `src/summer4/solvers/euler_backend.py:170` `euler_solve`, `src/summer4/solvers/diffrax_backend.py:102` `diffrax_solve` | Both take `params` and pass it into `observe` / `vector_field` / `_snapshot_factory` / `_group_fn`. |
| Model build | `FlowModel.compile(*, derived_fn, strict_pairing)` (`compiled.py:1050`) → `CompiledModel` (frozen, hashed by `_model_digest`, `:602`) | `CompiledModel.run(params, y0, *, t0, t1, dt, steps, save, solver, ...)` (`:894`) |
| Frontend | `src/summer4/epi/model.py` `EpiModel` | Sugar over `FlowModel`. The rule is that builder path and declarative path give **equal digests**. |
| Pytree pattern | `src/summer4/jax/state.py` `State` | `@register_pytree_node_class` + frozen slots dataclass + `tree_flatten`/`tree_unflatten`. Copy this pattern. |
| Selector helpers | `src/summer4/flows/join.py` `selector_properties(sel)` | The property names a selector mentions. |
| jit test pattern | `tests/test_time_varying.py` `test_jit_value_and_grad_through_parametric_xy` (~`:455`) | How existing tests jit/grad through `run`. Copy it. |
| summer3 split rule | `/Users/s/dev/EMU/summer3wip/summer3/epi.py:76-125` `build_istate` | Base sets, splits multiply, unmentioned properties get `×1/K` on carriers only. |
| summer2 adjust rule | `/Users/s/dev/EMU/summer2/code/summer2/population.py:37-86`; fixture `/Users/s/dev/EMU/summer2/tests/test_population_restribution.py` | Within each sibling group differing only in `strat` and matching `dest_filter`, the total is kept and redistributed as `total × p[k]`. |

The repo rules in `AGENTS.md` apply throughout: Black at line length 100, a type
annotation on every function including tests, `mypy --strict` on `src/summer4`,
and no epidemiology in core. `InitialPopulation` and the stages are
domain-neutral, so they belong in `summer4.flows`.

---

## The stage contract (normative — implement exactly this)

| Stage | When | Runs on | Inputs | Produces |
|---|---|---|---|---|
| **0 — compile** | `FlowModel.compile()`, once per model structure | host (NumPy) | `PropertyMap`, flows, `InitialPopulation` | index arrays, `HoistTable`, `InitPlan`, digest |
| **1 — run start** | once per `CompiledModel.run` / `initial_state` / loss evaluation, **before** the solve | traced JAX | raw `params` | `Prepared(params=prepare_fn(params) or params, hoisted=(...))`; the initial state `y0` |
| **2 — per step** | every vector-field call | traced JAX, inside `scan`/`while` | `t`, `y`, `Prepared` | `derived = derived_fn(prepared.params, y=, t=)`; rates, reading hoisted values |

Rules:

- **R1.** `prepare_fn(params) -> params'` is a user hook: `FlowModel.compile(prepare_fn=...)`. It must not depend on `t` or `y`. It hashes by `id` in the digest, like `derived_fn`.
- **R2.** Initial-population expressions and callables read `Prepared.params`, never `derived_fn` output. A `Time()`, `FlowRef`, `Reduce` or `Capture` in one is a compile-time `ValueError`.
- **R3.** A rate subtree is **run-stage** when it depends only on constants and (if `derived_fn is None`) on `FieldRef`s. When `derived_fn` is set, `FieldRef`s are **step-stage**, because they read `derived_fn` output. So adding a `derived_fn` disables parameter hoisting. The documented remedy is to move `t`/`y`-independent work into `prepare_fn`.
- **R4.** Hoisting is an optimisation only. `compile(hoist=False)` must give identical results. The digest includes the `hoist` flag.
- **R5.** `observe` / `vector_field` still accept raw params. If `params` is not a `Prepared`, they call `self.prepare(params)` themselves, which is correct but not hoisted. Solvers always call `prepare` once and pass the `Prepared` down.
- **R6.** In `run`, an explicit `y0` overrides a model-attached initial population (warm starts). With no `y0` and no initial population, `run` raises `ValueError`.

### Stage classification of rate nodes (`rate_stage`)

| Node | Stage |
|---|---|
| `Const`, `ArrayConst` | run (leaf) |
| `FieldRef` | run if `params_are_static` (i.e. `derived_fn is None`), else step (leaf) |
| `Time`, `FlowRef`, `Reduce`, `Capture` | step |
| `BinOp` | step if any child is step, else run |
| `Interp` | step if any of breakpoints/values/arg is step, else run |
| `GaussianPulse` | step if any of arg/centre/width/height is step, else run |
| any other `RateOps` | `expr.__rate_stage__()` if that attribute exists and is callable, else **step** (conservative; `ForceOfInfection` stays step) |

### Hoisting rule (`build_hoist_table`)

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
3. Otherwise descend into the children of `BinOp` and `GaussianPulse`. Do not
   descend into `Capture` or unknown custom nodes.
4. Skip a key that already has a slot, because shared node objects are
   evaluated once.

The `HoistTable` keeps a reference to every slotted node, so `id()` stays valid
for the lifetime of the `CompiledModel`.

---

## The initial-population semantics (normative)

```python
from summer4 import InitialPopulation, Split, REMAINDER

init = InitialPopulation(
    base={state["S"]: Param("pop") - Param("seed"), state["I"]: Param("seed")},
    splits=(
        Split(reach, {"reachable": Param("frac"), "unreachable": REMAINDER}),
        Split(vacc, lambda p: vacc_table(p), by=(age,)),            # arbitrary function
        Split(vacc, {"one": 0.7, "two": 0.2, "none": 0.1}, where=age["old"]),  # summer2 adjust
    ),
)
```

Definitions, for a `PropertyMap` with `N` rows and properties `P_1..P_m`,
where property `P` has `K_P` traits:

1. **Base entries.** Each `(selector S_e, value v_e)` is the **total**
   population of the rows it matches, `M_e = pmap.select(S_e)`. The value is a
   `float`, a run-stage `RateOps`, or a callable `fn(prepared_params) -> scalar`.
   An empty `M_e` is a `ValueError`. Entries **add** where they overlap.
2. **Share of a row for property `P`, written `s_P[r]`:**
   - If row `r` does not carry `P`: `1`.
   - Else, if a split governs `(r, P)`: that split's normalised weight for the
     row's trait. The governing split is the **last** `Split` for `P`, in
     declaration order, whose `where` matches `r` (`where=None` matches all).
   - Else: `1 / K_P`. This is the summer3 even default, and it is ragged-aware
     automatically because non-carriers get `1`.
3. **Split weights** are **always normalised** along the trait axis
   (`w / w.sum(-1, keepdims=True)`). This is the balancing rule. `REMAINDER`
   evaluates to `1 - sum(other traits)` before normalisation, and at most one
   trait may be `REMAINDER`. The weight forms are:
   - **`Mapping[str, float | RateOps | REMAINDER]`**: every trait of `P` must be
     a key. Not allowed with `by`.
   - **`RateOps` or `Callable[[params], array]`**: evaluates to shape `(K_P,)`,
     or `(K_B1, ..., K_Bn, K_P)` when `by=(B1, ..., Bn)`. The row's weight is
     gathered at `[code_B1[r], ..., code_Bn[r], code_P[r]]`. Every row governed
     by the split must carry every `by` property, or `ValueError` at compile.
     The wrong evaluated shape is a `ValueError` at trace time.
4. **Free properties of an entry.** `P` is *free* for entry `e` iff the rows of
   `M_e` that carry `P` take **≥ 2 distinct codes**. A property the selector
   pins, e.g. `age["0"]`, is not free, so its share is not applied. This means
   `base={age["0"] & state["S"]: 100}` puts all 100 in age 0 even if an age
   split gives age 0 weight zero.
5. **Row weight:** `W[e, r] = Π_{P free for e} s_P[r]` for `r ∈ M_e`, else `0`.
6. **Result:** `y[r] = Σ_e v_e · W[e, r] / Σ_q W[e, q]`. A zero denominator
   gives `0` via a safe divide. This is documented as the one silent case
   (traced all-zero weights).

Consequences to state in docs: mass is conserved per entry. With complete
splits and full-coverage selectors, this equals summer3 `build_istate`, and
`Split(..., where=)` equals summer2 `adjust_population_split`.

Host validation at compile (`InitPlan`), each a `ValueError` with the property
or trait named:
- unknown property;
- a `by` property equal to `P`;
- `where` mentioning `P` (use `selector_properties`);
- missing or extra mapping keys;
- more than one `REMAINDER`;
- a negative concrete weight;
- concrete weights (all `float`/`Const`, no `REMAINDER`) whose sum differs
  from 1 by more than `1e-9` **unless** `Split(normalize=True)`;
- concrete non-`REMAINDER` weights summing to more than 1 when `REMAINDER` is
  present;
- a step-stage `RateOps` in a base value or weights (R2).

JAX shape of the evaluation, which must not unroll per row:
- **Host arrays:**
  - `member: bool[E, N]`
  - `free: bool[E, m]`
  - per property: `codes`, `carries`, `K`
  - per split: `rows_j: int32[...]` (rows it governs after later overrides)
    and `by_codes_j: int32[len(rows_j), n_by]`
- **Traced:**
  1. For each property, `s = where(carries, 1/K, 1)`; for each governing split
     in order, `s = s.at[rows_j].set(table[(*by_codes_j.T, codes[rows_j])])`.
  2. `S = stack(...)` of shape `[m, N]`.
  3. `W = prod(where(free[:, :, None], S[None], 1), axis=1) * member`.
  4. `v = stack(E values)`.
  5. `y = (v[:, None] * W / safe(W.sum(-1))[:, None]).sum(0)`.
- The Python loops are over properties, splits and entries only. None of these
  grows with `N`.

---

## Phase 3.0 — Branch, merge the TB-port plans, baseline

1. `git fetch origin && git switch docs/textbook-catchup && git pull --ff-only`
2. `git switch -c feat/initial-population`
3. `git merge origin/docs/tb-ports-plans`. Expect exactly two textual conflicts:
   - **`docs/evaluation/index.md`**:
     - keep *ours* for the `8 of 11`, `12 of 20` and "Implemented layers …
       timevarying" rows;
     - take *theirs* for the "Ceiling of the planned roadmap (WP2–WP16)" row;
     - keep the `tb-ports` toctree line (it auto-merges).
   - **`futureplans/README.md`**: keep **both** sides' bullet lists (ours
     first, then `propertydata-dense-unstack.md`).
   - **STOP** if any other file conflicts.
4. `pixi run coverage-write && pixi run coverage && pixi run test-quick`, all
   green. **STOP** if not.
5. Commit the merge. Copy this plan to `plans/initial-population.plan.md` and
   commit ("Plan WP3 initial population and run stages.").
6. Read before coding:
   - `AGENTS.md`
   - `futureplans/*.md`
   - `docs/evaluation/tb-ports.md`
   - the WP3 section of `plans/tb-ports-feature-completeness.plan.md`
   - `src/summer4/flows/compiled.py` lines 222–330 and 575–780
   - both solver backends

## Phase 3.1 — `Prepared` and the `prepare_fn` hook (no hoisting yet)

**Create `src/summer4/flows/stages.py`:**
- `PrepareFn = Callable[[Any], Any]`
- `Prepared` pytree (fields `params: Any`, `hoisted: tuple[Any, ...]`). Use
  the `State` pattern; aux data is `None`.
- `Stage = Literal["run", "step"]`

**Edit `compiled.py`:**
- `CompiledModel` gets fields `prepare_fn: PrepareFn | None = None` and `hoist_table: HoistTable | None = None`. Add them **before** `capture_meta`, with defaults, and keep every existing keyword call working.
- Add `def prepare(self, params: object) -> Prepared`. For now, `p = params if self.prepare_fn is None else self.prepare_fn(params)` and return `Prepared(p, ())`.
- `observe(t, y, params)`: if `not isinstance(params, Prepared)`, set `params = self.prepare(params)`. Then compute `derived` from `params.params`.
- `_model_digest`: hash `id(prepare_fn)` like `derived_fn`.
- `FlowModel.compile(*, derived_fn=None, prepare_fn=None, strict_pairing=True)`.

**Edit the solvers:**
- In `euler_solve` and `diffrax_solve`, call `prepared = model.prepare(params)` once at the top.
- Pass `prepared` everywhere `params` went: `_snapshot_factory`, `_group_fn`, and diffrax `args`.

**Create `tests/helpers/jaxpr.py`:**
```python
def _sub_jaxprs(eqn: Any) -> list[Any]:
    out = []
    for v in eqn.params.values():
        for item in (v if isinstance(v, (tuple, list)) else (v,)):
            if hasattr(item, "jaxpr") and hasattr(item.jaxpr, "eqns"):
                out.append(item.jaxpr)
            elif hasattr(item, "eqns"):
                out.append(item)
    return out

def _all_eqns(jaxpr: Any) -> Iterator[Any]:  # recursive
    for eqn in jaxpr.eqns:
        yield eqn
        for sub in _sub_jaxprs(eqn):
            yield from _all_eqns(sub)

def loop_body_primitives(closed: Any) -> Counter[str]:
    """Primitive names inside any scan/while body, at any depth."""
    # walk; when eqn.primitive.name in {"scan", "while"}, count every eqn in _all_eqns(sub)

def outside_loop_primitives(closed: Any) -> Counter[str]:
    """Primitive names not inside any scan/while body."""
```

**Create `tests/test_stages.py`:**
- **`prepare_fn` feeds rates.** `prepare_fn=lambda p: {**p, "beta": p["r0"] / p["period"]}` with a rate `Param("beta")`. `run` must equal a model without `prepare_fn` given precomputed `beta`, for Euler and `"tsit5"`.
- **`prepare_fn` runs at run start.** Its body calls `jnp.cumsum`. Using the jit pattern from `test_time_varying.py`, build `jax.make_jaxpr` of a scalar function of params that calls `run`. Assert `"cumsum"` is absent from `loop_body_primitives` and present in `outside_loop_primitives`, for Euler and `"tsit5"`.
- **Raw params still work.** `observe`/`vector_field` with raw params equal `observe` with `cm.prepare(params)`.
- **Digest.** The digest differs with and without `prepare_fn`.

**Checks:** `pixi run lint && pixi run format-check && pixi run test-quick`.
All pre-existing tests must pass unchanged. **STOP** after two failed fix
attempts. Commit: "Add a run-start prepare stage ahead of the per-step vector
field."

## Phase 3.2 — Stage classification and hoisting

**In `stages.py`, add:**
- `rate_stage(expr: RateOps, *, params_are_static: bool) -> Stage`, per the table above.
- `HoistEntry(node: RateOps, part: Literal["value", "breakpoints", "values"])`
- `HoistTable(entries: tuple[HoistEntry, ...], slot: Mapping[tuple[int, str], int])` (frozen, `eq=False`)
- `build_hoist_table(roots: Sequence[RateOps], *, params_are_static: bool) -> HoistTable`, per the hoisting rule.
- `def roots_of(flows: Mapping[str, FlowEdges], order: tuple[str, ...]) -> list[RateOps]`

**In `compiled.py`:**
- `FlowModel.compile(..., hoist: bool = True)` builds `hoist_table = build_hoist_table(roots_of(...), params_are_static=derived_fn is None)` if `hoist`, else an empty table. The digest hashes `b"hoist1"`/`b"hoist0"`.
- `prepare()` fills `hoisted`. For each entry, evaluate with `_eval_rate(node, derived=p, flow_values={}, flow_meta={}, pmap=self.pmap, t=None, y_arr=None, captures={})`:
  - for part `"value"`, that is the result;
  - for `"breakpoints"`/`"values"`, `jnp.stack([jnp.asarray(eval(c)) for c in node.breakpoints/values])`.
- `_eval_rate` gains the keyword `hoisted: Prepared | None = None` and `table: HoistTable | None = None`. Thread both through `child`, `_eval_aligned`, `_apply_adjustments`, `observe`, and custom evaluators (they receive `eval_child`, which already closes over them).
  - At the top of `_eval_rate`, if `(id(expr), "value")` is in `table.slot`, return `hoisted.hoisted[slot]`.
  - In the `Interp` case, use the `"breakpoints"`/`"values"` slots when present instead of `jnp.stack(...)`.

**Create `tests/test_hoisting.py`:**
- **Classification unit tests** for each table row, including `derived_fn` set making `FieldRef` step-stage and a custom node without `__rate_stage__` being step.
- **Equivalence.** A model with `Param("a") * Param("b") + 1.0` rates, `linear(Time(), (refs.t0, refs.t1, 10.0), (refs.y0, refs.y1, 0.0))`, a `GaussianPulse` with param centre, and `Multiply`/`Transform` adjustments.
  - `compile(hoist=True)` vs `hoist=False`: Euler `assert_allclose(rtol=1e-12)`, `"tsit5"` `rtol=1e-9`.
  - `jax.grad` with respect to the knot params matches between the two.
- **Knot count does not grow the loop body.** An `Interp` with `K` `FieldRef` knots, `K=4` vs `K=40`.
  - `hoist=True`: `sum(loop_body_primitives(...).values())` is **equal**.
  - `hoist=False`: it is **larger** for `K=40`. This proves the test measures something.
- **Epi unaffected.** An `EpiModel` SIR from `tests/test_epi_model.py` gives equal results with `hoist=True` and `hoist=False`.
- **Digest.** The digest differs between `hoist=True` and `hoist=False`.

Never weaken the jaxpr assertions to make them pass. **STOP** instead.

**Checks as in 3.1, plus `pixi run test`.** Commit: "Hoist parameter-only rate
subtrees and interpolator knots into the run-start stage."

## Phase 3.3 — `InitialPopulation` types and the compiled `InitPlan`

**Create `src/summer4/flows/initial.py`:**
- `class _Remainder` (singleton, with `__repr__` `"REMAINDER"`) and `REMAINDER: Final = _Remainder()`
- `@dataclass(frozen=True, slots=True) class Split`: `prop: Property`; `weights: Mapping[str, float | RateOps | _Remainder] | RateOps | Callable[[Any], Any]`; `by: tuple[Property, ...] = ()`; `where: Selector | None = None`; `normalize: bool = False`
- `@dataclass(frozen=True, slots=True, eq=False) class InitialPopulation`:
  - `base: tuple[tuple[Selector, float | RateOps | Callable[[Any], Any]], ...]`
  - `splits: tuple[Split, ...] = ()`
  - `__post_init__` accepts a `Mapping` for `base` and converts it to a tuple of pairs with `object.__setattr__`
  - `compile(self, pmap: PropertyMap) -> InitPlan`
- `@dataclass(frozen=True, slots=True, eq=False) class InitPlan`:
  - holds the host arrays listed under *JAX shape of the evaluation*
  - `digest_bytes() -> bytes`: hash codes, member, free, property/by names, `repr(where)`, `normalize`, `_rate_bytes` of `RateOps`, `id()` of callables, mapping key order
  - `evaluate(self, params: object) -> PropertyData` (implemented in 3.4)

Implement every host validation listed in the semantics. Use `rate_stage(...,
params_are_static=True)` for R2.

**Create `tests/test_initial_population.py`, part 1 (host only):**
- Each validation error, with `pytest.raises(ValueError, match=...)`.
- `free` matrix:
  - `base={age["0"] & state["S"]: ...}` makes age not free;
  - `base={state["S"]: ...}` on an age-stratified map makes age free;
  - on a ragged map, a property carried by a single code among members is not free.
- `rows_j` override: a later `Split(vacc, ..., where=age["old"])` takes the old rows away from an earlier unrestricted `Split(vacc, ...)`.

Commit: "Add declarative InitialPopulation with host-side validation."

## Phase 3.4 — Evaluation, model wiring, `get_initial_population`

1. Implement `InitPlan.evaluate(params)` exactly as specified. Evaluate `RateOps` values with `_eval_rate(..., derived=params, t=None, y_arr=None, flow_values={}, flow_meta={}, captures={})`. Call callables as `fn(params)`. Normalise tables, apply `REMAINDER`, use a safe divide.
2. Wire the model:
   - `FlowModel.set_initial_population(self, base, splits=()) -> InitialPopulation` stores the object and returns it. Calling it twice replaces the stored object.
   - `FlowModel.compile(..., init: InitialPopulation | None = None)`:
     - if both `init` and a stored population are present, raise `ValueError`;
     - otherwise compile to `InitPlan` and store it on `CompiledModel.init_plan: InitPlan | None = None`;
     - the digest includes `init_plan.digest_bytes()` or `b"noinit"`.
3. Add `CompiledModel.initial_state(self, params: object) -> PropertyData` (ledger `L5`). It accepts raw params or `Prepared`: it prepares if needed, then evaluates `init_plan` on `prepared.params`. With no plan it raises `ValueError("No initial population: call FlowModel.set_initial_population or pass y0.")`.
4. `CompiledModel.run(self, params, y0=None, *, ...)`:
   - Compute `prepared = self.prepare(params)` once.
   - If `y0 is None`, use `self.initial_state(prepared)`.
   - Pass `prepared` as `params` to the solvers. They already accept `Prepared` and must not re-prepare, so guard with `isinstance`.
5. Add `EpiModel.set_initial_population(self, base, splits=())`, delegating to `self._flow_model`.
6. Exports: add `InitialPopulation`, `Split`, `REMAINDER`, `Prepared` to `src/summer4/flows/__init__.py` and `src/summer4/__init__.py` `__all__`.

**`tests/test_initial_population.py`, part 2.** Hard-code expected vectors;
never import summer2 or summer3.
- **Ragged even default (summer3 / `TM6`).** Properties are `age(0,5,15)`, `state(naive, incipient, active)`, and `infect(low, high)` stratified `where=state["active"]`. `base={age[a]: 1000.0 for a in ages}` with no splits gives, per age: naive 333.33…, incipient 333.33…, active-low 166.66…, active-high 166.66…. Totals: 1000 per age.
- **Absolute by age (`TM6` port shape).** `base={state["naive"] & age[a]: 1000.0}` puts 1000 in each naive age row and 0 elsewhere, including under infect.
- **summer2 adjust fixture.** Rebuild the map from `/Users/s/dev/EMU/summer2/tests/test_population_restribution.py`: `state(S,I,R)`, `age(young,old)`, `loc(urban,rural)`, `vacc(one_dose,two_dose,unvacc)`.
  - `base={state["S"]: 990.0, state["I"]: 10.0}`.
  - Splits `age {young: .6, old: .4}`, `loc {urban: .8, rural: .2}`, `Split(vacc, {one_dose: .7, two_dose: .2, unvacc: .1}, where=age["old"])`.
  - Assert S-old-urban equals `990 * 0.4 * 0.8 * [0.7, 0.2, 0.1]`.
  - Assert young rows equal `990 * 0.6 * loc * (1/3)`.
  - Assert the total is 1000.
  - Repeat the multi-filter case from that file with `where=age["old"] & loc["urban"]`.
- **Kiribati shape (`KI3`).** `state(mtb_naive, clin_inf, other)`, 8 age bands, `reach(reachable, unreachable)`. `base={state["mtb_naive"]: Param("pop") - Param("seed"), state["clin_inf"]: Param("seed")}`, `Split(reach, {"reachable": Param("frac"), "unreachable": REMAINDER})`.
  - Each naive/age/reachable row equals `(pop - seed) / 8 * frac`.
  - `jax.grad` of the sum of reachable rows with respect to `frac` equals `pop`.
- **By-function (`InitialPopulationGraphobject`).** `Split(imm, lambda p: jnp.array([[p["vy"], 1 - p["vy"]], [p["vo"], 1 - p["vo"]]]), by=(age,))` gives the expected rows. Also a wrong-shape callable raises `ValueError`.
- **Selector pinning.** `base={age["0"] & state["S"]: 100.0}` with `Split(age, {"0": 0.0, "5": 1.0}, normalize=True)` still puts 100 in age 0.
- **Overlap adds.** `base={state["S"]: 10.0, Everything(): 30.0}`.
- **`normalize=True`.** Raw weights `{a: 2.0, b: 6.0}` become 0.25/0.75.
- **Model wiring.**
  - `run(params)` with `set_initial_population` is bit-identical to `run(params, cm.initial_state(params))`.
  - An explicit `y0` overrides it.
  - No init and no `y0` raises.
  - The `set_initial_population` digest equals the `compile(init=...)` digest.
  - The `EpiModel` path digest equals the `FlowModel` path digest.
- **Under jit.** `jax.jit(value_and_grad)` of a loss through `run` with `y0=None` works (copy the `test_time_varying.py` pattern).
- **Jaxpr size independent of `N`.** `len(jax.make_jaxpr(plan.evaluate)(params).jaxpr.eqns)` is equal for age with 3 traits vs 30 traits, with the same entries and splits.
- **Run-start stage.** In a `run` jaxpr, the init evaluation's `reduce_prod` (from `prod`) is absent from `loop_body_primitives`. If the flows also use `prod`, pick another primitive unique to the init path and say which in a comment.

Checks: all of the *Required checks* in `AGENTS.md`. Commit: "Evaluate initial
populations at run start and wire them into FlowModel, EpiModel and run."

## Phase 3.5 — Example notebooks and developer doc

Take the next free two-digit prefixes in `examples/notebooks/` (expected
`10-`, `11-`; check with `ls`). Code cells must be plain Python, assert their
claims, and use `from summer4 import ...`. Plot with Plotly via
`Trace.to_pandas()`. Clear the outputs.

- **`10-initial-population.ipynb`.** The story, in order:
  1. total plus seed;
  2. the even default on a ragged map (show `pmap.to_frame()` next to the values);
  3. a parameter split with `REMAINDER`, and `jax.grad` through it;
  4. a split by an arbitrary function with `by=`;
  5. `where=` as summer2's `adjust_population_split`;
  6. `initial_state` as `get_initial_population`;
  7. explicit `y0` overriding for warm starts.

  Assert totals and the key rows.
- **`11-run-stages.ipynb`.** The story, in order:
  1. the three-stage table;
  2. `prepare_fn` computing a derived constant;
  3. show that `Interp` knots are hoisted by printing the loop-body primitive counts for `hoist=True/False` (a small inline walker is fine here);
  4. why `derived_fn` disables parameter hoisting (R3), and how to move work into `prepare_fn`.

  Assert equal results between `hoist=True` and `hoist=False`.
- **`docs/dev/run-stages.md`.** Copy the normative *stage contract*, *classification* and *hoisting* sections verbatim, and link it from `docs/dev/index.md` (toctree) and `docs/dev/architecture.md`.
- **`AGENTS.md`**, under "JAX is the primary runtime target", add one paragraph: "Parameter work is staged (compile / run start / per step); see `docs/dev/run-stages.md`. Put `t`/`y`-independent computation in `prepare_fn` or a hoistable rate subtree, never in `derived_fn`."

Checks: `pixi run check-notebooks && pixi run test && pixi run -e docs
docs-strict`. Commit: "Document initial populations and run stages with example
notebooks."

## Phase 3.6 — Ledgers and evaluation prose (same rules as `AGENTS.md`)

1. **`docs/evaluation/coverage-ledger.md`:**
   - `L4` → `full`, route `` `FlowModel.set_initial_population` / `InitialPopulation` ``
   - `L5` → `full`, route `` `CompiledModel.initial_state` ``
   - `S8` → `full`, route `` `Split(prop, weights, by=, where=)` ``, note "Weights normalised; unspecified properties split evenly over carriers"
   - `D8` note: append "; run-start work goes in `prepare_fn`"
   - *Delivery status*: add row `3.0–3.7 | WP3 — initial population and run stages | feat/initial-population | (this stack) | plans/initial-population.plan.md`
   - *Planning status* table: WP3 row → `plans/initial-population.plan.md` | **Applied.**
   - WP3 section heading → "(applied)", with a two-sentence summary
2. **`docs/evaluation/tb-ports.md`:** `KI3` and `TM6` → `full`, `Closed by` `—`, `Route today` `` `InitialPopulation` ``.
   - **Do not edit** the `KI2`, `KI12`, `KI23` or `TM9` rows. `tests/test_coverage_ledger.py` string-matches them.
3. Run `pixi run coverage-write`, then `pixi run coverage`. Fix every stale quote it reports:
   - `**N of M**` in `docs/evaluation/index.md`
   - `<model> rows complete today: **N of M**` in `tb-ports.md`

   Use the numbers the checker prints. Never hand-compute them.
4. Update the prose that says WP3 is missing:
   - `docs/user/07-from-summer2.md`: move "Initial conditions" out of *What does not translate* into the mapping table.
   - `docs/evaluation/gaps.md` §1.4 and the lines around 76 and 142–147
   - `docs/evaluation/feature-completeness.md` (L4/L5/S8 rows)
   - `docs/evaluation/docs-coverage.md`
   - `docs/textbook/roadmap.md`
   - `docs/index.md`
   - `docs/dev/architecture.md:93`
   - `docs/user/06-immutability-and-provenance.ipynb` cells 13–14 ("Not yet an API" → point to `InitialPopulation`, keeping the `parent_row` explanation)

   Apply the no-present-tense-for-planned-work rule.
5. **`futureplans/`:** add one note per concern, and index each in `futureplans/README.md`:
   - `derived-fn-blocks-hoisting.md`: R3. "Done" would be a split `derived_fn` into run and step halves, or a static-path declaration.
   - `mixing-matrix-per-call-normalisation.md`: `MixingMatrix.resolved_matrix` (`src/summer4/epi/mixing.py:76-93`) re-normalises a `FieldRef` matrix every step. It could declare `__rate_stage__` or move to `prepare_fn`.
   - `wp10-preprocess-is-prepare-fn.md`: `BayesianModel(preprocess=...)` in `plans/tb-ports-feature-completeness.plan.md` WP10.2 should be `CompiledModel.prepare_fn`, not a second mechanism. The Kiribati yearly mixing stack belongs there.

Commit: "Record WP3 in the coverage and TB-port ledgers."

## Phase 3.7 — Harvest the notebooks WP3 unblocked (increment only)

Port **exactly** these, no others. Rows already `full` but unported are not in
scope.

| Ledger row | Source | Target |
|---|---|---|
| textbook 2 `partial` | already ported | `docs/textbook/02-model-structures.ipynb`: remove the WP3 admonition, run the model using `set_initial_population` |
| textbook 13 `none` | `monash-emu/summer-textbook` @ `fd97783474789e50ace5ea420aec20147f9bbd76`, `textbook/13-mixing-and-transmission-types.ipynb` | `docs/textbook/13-mixing-and-transmission-types.ipynb` |
| summer2 `examples/01-basic-model` | `/Users/s/dev/EMU/summer2/docs/examples/01-basic-model.ipynb` | `docs/summer2/01-basic-model.ipynb` |
| summer2 `examples/07-age-stratification` | `/Users/s/dev/EMU/summer2/docs/examples/07-age-stratification.ipynb` | `docs/summer2/07-age-stratification.ipynb` |
| summer2 `detailed/InitialPopulationGraphobject` | `/Users/s/dev/EMU/summer2/docs/detailed/InitialPopulationGraphobject.ipynb` | `docs/summer2/initial-population-graphobject.ipynb` |

- If `/private/tmp/summer-sources/summer-textbook` is missing, clone the textbook into the scratchpad and `git checkout` the SHA above.
- Follow `docs/textbook/porting.md`:
  - carry the prose, figures and BSD-2-Clause notice, and pin the SHA;
  - rewrite **all** code in summer4 idiom;
  - use Plotly.
- Chapter 13's closing assertion (all four prevalence curves identical for random `prop1`) must be a real `assert`. Split the seed with the groups, as the source does.
- Add each page to its `index.md` toctree.
- Set each ledger row's `Status` and `Ported` path (relative to `docs/`), then `pixi run coverage-write && pixi run coverage`.
- Update `docs/textbook/12-heterogeneous-mixing-intro.ipynb` cell 0, which says chapter 13 is not ported.
- **Honesty rule:** if a page hits a real blocker other than WP3, leave its row at the true status with the blocker named and the `Ported` cell empty, and list it in next steps. If a port needs an API this plan does not provide, **STOP**.
- Do not modify chapters 14/15 (their hand-built `y0` splits); list them in next steps.

Checks: all *Required checks* plus `pixi run -e docs docs-strict`. Commit: "Port
the textbook and summer2 pages WP3 unblocks." Push the branch
(`git push -u origin feat/initial-population`). **Do not open or merge a pull
request.** Ask the user which base branch it targets.

## Phase 3.8 — Generate explicit next steps (mandatory final output)

Before you report, re-run `pixi run coverage` and read its output, the updated
*Planning status* table, `tb-ports.md` *port order* / *readiness*, and
`futureplans/README.md`. Then produce a **Next steps** list, in chat **and** as
the last section of the PR body draft you give the user. Use exactly this
template for each item:

```
N. <imperative title>
   Why: <ledger IDs / futureplans note / failing check that motivates it>
   Branch: <suggested branch name>   Depends on: <branch or WP, or "none">
   First action: <one concrete command or file to open>
   Done when: <observable check, e.g. "pixi run coverage shows KI… full">
```

The list must include, in this order:
1. **User notebook gate.** One checklist line per notebook from 3.5 and 3.7, each with the specific claim to check. For example: "`10-initial-population`: the ragged-map table shows active rows at half the naive rows". The user runs them with `pixi run notebook` or the docs build. Merging waits for their sign-off.
2. **PR and merge order.** This branch contains the `docs/tb-ports-plans` merge; state which base branch you propose and why.
3. **Any honesty-rule leftovers** from 3.7.
4. **The next package** from `tb-ports.md` port order and the ledger (expected WP12 → WP13 → WP14 …). Name the plan section to promote and the rows it closes.
5. **Each new `futureplans/` note** from 3.6, one item per note.
6. **Optional clean-up:** refactor textbook 14/15 hand-built splits onto `Split`.
7. **Anything that failed, was skipped, or surprised you,** stated plainly with command output.

Also add a one-paragraph "Next steps after WP3" to the *Planning status* section
of `docs/evaluation/coverage-ledger.md` naming items 4–5, so other branches see
it. Run `pixi run coverage` again, commit ("Record next steps after WP3."), and
push.

---

## Files touched (summary)

- **New:**
  - `src/summer4/flows/stages.py`, `src/summer4/flows/initial.py`
  - `tests/helpers/jaxpr.py`, `tests/test_stages.py`, `tests/test_hoisting.py`, `tests/test_initial_population.py`
  - `examples/notebooks/10-initial-population.ipynb`, `examples/notebooks/11-run-stages.ipynb`
  - `docs/dev/run-stages.md`
  - `plans/initial-population.plan.md`
  - three `futureplans/*.md` notes
  - four ported pages
- **Modified:**
  - `src/summer4/flows/compiled.py`, `src/summer4/solvers/euler_backend.py`, `src/summer4/solvers/diffrax_backend.py`
  - `src/summer4/epi/model.py`
  - `src/summer4/flows/__init__.py`, `src/summer4/__init__.py`
  - `docs/evaluation/{coverage-ledger,tb-ports,index,gaps,feature-completeness,docs-coverage}.md`
  - `docs/user/07-from-summer2.md`, `docs/user/06-immutability-and-provenance.ipynb`
  - `docs/textbook/{02,12}-*.ipynb`, `docs/textbook/roadmap.md`, `docs/textbook/index.md`, `docs/summer2/index.md`
  - `docs/dev/{index,architecture}.md`, `docs/index.md`
  - `AGENTS.md`, `futureplans/README.md`

## Verification (end to end)

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch && pixi run coverage
pixi run -e docs docs-strict
```

Expected results:
- `pixi run coverage` reports **47 of 52** API rows at `full` and `KI3` `TM6` at `full`.
- `tests/test_hoisting.py::*knot*` proves the loop-body jaxpr does not grow with knot count.
- `tests/test_initial_population.py` reproduces the summer2 adjust fixture and the summer3 ragged even split with hard-coded numbers.
- `tests/test_stages.py` proves `prepare_fn` work sits outside the solver loop for Euler and diffrax.
- The notebooks pass the user gate before merge.
