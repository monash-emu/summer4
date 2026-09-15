---
name: flow-outputs
description: Phase 4 of flows/derived-outputs — flow mass as a queryable edge trace, source/dest polarity aggregation, explicit incidence and integration, and computed-value capture.
---

# Phase 4 — Flow outputs and polarity queries (`feat/flow-outputs`)

## Context

`plans/flows-derived-outputs.plan.md` §"Phase 4" is the accepted design; this is
its execution plan. It wires the Phase 1 `EdgeMap` through the Phase 2 save plan
and the Phase 3 solver seam.

The pieces exist and are not connected. `EdgeMap` (`src/summer4/flows/edges.py`)
already gives every flow a doubled `PropertyMap` table with `@source` / `@dest`
columns and a `Source`/`Dest` rewrite. `FlowMass` requests already materialise
per-edge mass. But `results/eval.py:84` returns a bare array, so `Trace.select`
raises immediately (`results/trace.py:87`), and there is no way to ask which
*side* an aggregation grouped by. Ledger `D1` (`request_output_for_flow`), `D6`
(`request_computed_value_output`) and `D7` (`request_track_modelled_value`) are
all `none`.

**Closes:** `D1` `D6` `D7` → `full` (WP4). 33 → 36 / 52 (69%).
**Unblocks:** textbook 4, 8, 10; summer2 `03-derived-outputs`, `04-flow-types`,
`10-derived-outputs-stratified`.

The API-row delta understates the phase. This is the largest design chunk of the
three, and the roadmap's own warning stands: **the polarity surface is the least
proven part of the design.**

Depends on Phase 3 (`feat/diffrax-solver`) for the solver seam and save groups.

## 4a. `FlowMass` traces become `PropertyData` over the edge table

In `src/summer4/results/eval.py`, the `FlowMass` branch returns
`PropertyData(emap.table, mass)`. When `where` is given, gather onto
`emap.table.take(idx)` so the sub-table travels with the data — the same `take`
primitive `Trace.select` already uses (`propertymap.take`, landed in `8d481de`).
The whole Phase 2 query surface then applies to flows for free.

**Fix a live bug while here.** `eval.py:93` calls
`_sum_over_edge(ctx.flows[flow], emap, prop, side)` — the *unfiltered* mass —
silently discarding the `where` filter computed on line 89. Sum over the
filtered mass against the filtered table, and cover it with a test (§4g).

Then generalise trace construction. `CompiledModel.run` special-cases
`Compartments` inline (`flows/compiled.py:632`) and hands everything else
through raw. Replace it with one `values_for(req, raw, model)` helper in
`results/eval.py`, used by both backends, so the Euler and diffrax paths cannot
disagree about what a trace's `values` should be.

## 4b. Edge-aware `Trace`

An edge trace's `PropertyData.pmap` **is** `EdgeMap.table`, so nothing new
enters the pytree. But `PropertyMap._validate_selector` rejects `Source`/`Dest`
by design (Phase 0 added those raising arms), so `Trace.select` has to route
through the edge rewrite (`edges.py:181`).

**Do not store the `EdgeMap` on the `Trace`.** It holds NumPy arrays, so it is
unhashable and cannot be pytree aux; `Result.tree_unflatten` would not survive
it. Instead:

- add a free function in `flows/edges.py` that rewrites a `Source`/`Dest`
  selector against a **table alone**, deriving the `@source` / `@dest` marker
  gates from the table's own marker columns — a gate is needed exactly when its
  marker column is `-1` throughout, which is the same condition
  `EdgeMap.from_indices` uses at construction (`edges.py:165-166`). `EdgeMap`'s
  own methods then delegate to it, so there is one implementation.
- have `Trace` detect an edge trace from `dims` containing `"edge"`. `dims` is
  already aux and already survives `tree_unflatten` (`results/result.py:78`).

With that, the roadmap's examples work:

```python
res["transfers"].select(Source(age["0-4"]) & Dest(~state["R"])).total()
res["ageing"].select(model.edges("ageing").moves_mask(age))
```

and `Dest(Everything())` on an exit flow stays Kleene-**unknown** rather than
raising — the property that makes polarity-agnostic requests work uniformly
across exit, entry and transition flows.

## 4c. `sum_over(prop, side=)`

On an edge trace, `side` is **required**; omitting it raises an error naming
both options.

This is the entire point of the row. The retired spike hard-coded the side by
flow kind (`explorations/flows/prototype.py:648` as cited in the roadmap; the
file no longer exists), so `transition.sum_over(location)` always grouped by the
*source* compartment. Both directions are needed, and neither is a safe default
— a silent default here is the bug, not the ergonomics.

Promote `eval.py:_sum_over_edge` to a shared helper in `flows/edges.py` so the
save-time reduction (`FlowMass.sum_over`) and the post-hoc reduction
(`Trace.sum_over`) are one implementation with one set of semantics. Return a
`PropertyData` over `PropertyMap.from_property(prop)` so `to_frame()` labels the
group columns by trait name; apply the same treatment to the compartment
`sum_over` path so both kinds of trace behave alike.

`dims` goes `("time", "edge")` → `("time", "group")`, as `dims_for_quantity`
already declares.

## 4d. `.integrate()` and `.incidence()` on the time axis

Flow outputs are **instantaneous rates**. That is settled design decision 2 in
the roadmap, and the reductions are explicit rather than accumulated in solver
state. On `Trace`:

- `incidence(method="trapezoid" | "simpson") -> Trace` — counts per interval;
  `T-1` rows, times set to the interval right edges.
- `integrate(method=...) -> Trace` — one value per column over the whole window;
  drops `"time"` from `dims`.

Both compute from the actual `times.values` spacing, because Phase 3's adaptive
`ts` need not be uniform. Simpson requires a uniform grid with an odd point
count and raises a clear error otherwise. Both must be **vectorized** — do not
add a second Python-loop unroll next to the one already filed in
`futureplans/trace-rolling-jaxpr.md`; validate with `jax.make_jaxpr` that the
op count does not grow with `T`.

**Be explicit about the error** — in the docstring, in the notebook, and in the
ported chapter. Trapezoid over a save interval is second-order accurate, whereas
summer2's accumulated incidence is exact for the solver's own quadrature. For
calibration against case counts that is a real bias, and it is **not** fixable
on the `Result`, because the information is gone by then. Mitigations, in order:

1. save on a finer grid than you calibrate on, and `resample`;
2. `method="simpson"`;
3. eventually an opt-in accumulator in `State.ledgers`, which Phase 2 reserved
   and which is **not** implemented — file it as a `futureplans/` note naming
   `src/summer4/jax/state.py` and what "done" looks like.

Do not silently ship trapezoid and call it incidence.

## 4e. `ComputedValue` validated at compile time

Validate `ComputedValue.path` against the `derived_fn` return annotation using
`get_type_hints`, recursing into nested `NamedTuple`s exactly as `derived_refs`
already does (`src/summer4/flows/rates.py`). Raise at `expand` / `describe` /
`run` time with the available paths listed. Otherwise a typo surfaces as an
`AttributeError` from `_lookup_path` inside a traced scan body, which is a
miserable place to read an error from.

When `derived_fn` carries no return annotation, skip validation rather than
guessing — an unannotated hook is legal, just unchecked.

Closes `D6` and `D7`.

## 4f. Prune `SaveContext.flows` to what the plan reads

`SavePlan.flow_reads()` (`results/plan.py:138`) is computed and never used.
`observe` keeps every flow's full edge mass (`flows/compiled.py:413`, 441/448/455)
at every save point — the 5.8 GB case written down in `FlowMass`'s own
docstring.

Give `observe` an optional `keep: frozenset[str] | None`. Rate expressions still
need `flow_values` for `FlowRef` dependencies *within* the step, so pruning
happens at the end: drop every entry not in `keep` ∪ the set the rate
expressions genuinely depend on (derivable from the actualized flows' `FlowRef`
edges, which `topo_sort` already walks). `run` passes `plan.flow_reads()`.

This does nothing for a default `EVERYTHING` run, which asks for every flow by
construction — it is what makes an explicit plan actually cheap, and it is the
difference between flow outputs being usable at scale and not.

## 4g. Tests — `tests/test_flow_outputs.py`

- `sum_over(side="source")` and `side="dest"` differ on a `TraitChain` ageing
  flow that crosses a band, and each matches a hand-computed expectation. This
  is the phase's headline gate.
- `sum_over` without `side` on an edge trace raises, naming both options.
- `select(Source(...) & Dest(...))` on a transition flow.
- `Dest(Everything())` on an exit flow selects nothing — Kleene-unknown, not an
  error — and the same request runs unchanged against a transition flow.
- `select(model.edges("ageing").moves_mask(age))` — the boolean-mask path.
- `.incidence()` on a refining grid converges to the analytic integral at
  second order (halving `dt` quarters the error).
- Simpson beats trapezoid on a smooth analytic rate, and raises on an even or
  non-uniform grid.
- `where` combined with `sum_over` against a hand computation — the §4a bug.
- A bad `ComputedValue.path` raises before any solve, not inside the scan; a
  good one round-trips into `Result`.
- Pruning: a plan naming one flow does not materialise the others. Assert both
  the `observe` keys and the `describe()` byte total.
- `jax.jit` and `jax.grad` through a loss ending in
  `.sum_over(age, side="dest").at_times(...)`.
- `jax.make_jaxpr` on `incidence()` — op count independent of `T`.
- `to_frame()` on an edge trace labels columns with `EdgeMap.labels()`
  (`state=S_age=0-4 -> state=I_age=0-4`).

## 4h. Notebooks and the user gate

**Feature notebook** — `examples/notebooks/06-flow-outputs.ipynb`:

```python
# Weekly incidence by the age of the newly infected
(res["incidence"]              # (time, edge) instantaneous rates
   .incidence()                # (time-1, edge) counts per interval
   .sum_over(age, side="dest")
   .resample(7, how="sum")
   .to_frame())
```

plus source-vs-dest aggregation side by side on the *same* flow, an exit flow
queried with `Dest(...)` to show Kleene-unknown, and an explicit
rate-vs-incidence comparison that **shows** the quadrature error as a number
rather than describing it.

**Harvest**, increment only. Run `pixi run coverage`, read the "publishable, not
yet ported" line, and port what this phase unblocked:

- textbook 4 — *Thinking about flow rates* (sojourn times);
- textbook 8 — *Derived outputs*;
- textbook 10 — *The reproduction number*;
- summer2 `03-derived-outputs`, `04-flow-types`,
  `10-derived-outputs-stratified`, into a new `docs/summer2/` section with its
  own index carrying the same attribution block as `docs/textbook/`.

Prose and figures carried under the Phase 3 porting convention
(`docs/textbook/porting.md`); code in current summer4 idiom.

**The harvest is evidence-driven.** `docs/textbook/roadmap.md:34` says chapter
10 also needs time-varying parameters for $R_t$ — that is WP5 (`P5`–`P8`), not
this phase, and it disagrees with the ledger's blocker for row 10. Attempt the
port. If a genuine blocker appears, leave the row `partial` with the *real*
blocker named and the `Ported` cell empty, rather than forcing a port or
quietly moving the status. Recording the truth is what the ledger is for.

> ### Notebook gate — blocking
>
> This is the notebook to user-test hardest. Do not merge until the user has run
> these in `pixi run notebook` and ticked:
>
> - [ ] `examples/notebooks/06-flow-outputs.ipynb` — a modeller can tell **from
>   the page alone** which side an aggregation grouped by, and the
>   rate-vs-incidence difference is visible, quantified and explained.
> - [ ] each ported textbook chapter and summer2 page reads as its source and
>   runs.
> - [ ] any ledger row left `partial` names a blocker the user agrees is real.

## Verification

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch
pixi run coverage
pixi run -e docs docs-strict
pixi run test-all
```

Ledger discipline in the same commit as the code: `D1` `D6` `D7` → `full`;
textbook rows 4, 8, 10 and summer2 rows `03-derived-outputs`, `04-flow-types`,
`10-derived-outputs-stratified` updated with their `Ported` paths; then
`pixi run coverage-write` and the totals in `docs/evaluation/index.md` (33 → 36).
