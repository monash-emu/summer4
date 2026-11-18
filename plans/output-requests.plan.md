# OutputSet requests: summer2-style one-call outputs, with escape hatches

Roadmap step 51, `feat/output-requests`. Closes `CX49` and `CX50` (package `CP10`).

## Context

summer2's derived outputs are one call each: `request_output_for_compartments(name, comps,
strata=)`, `request_output_for_flow(name, flow, source_strata=)`, `request_aggregate_output`,
`request_cumulative_output`, `request_function_output`, with `save_results=False` for
intermediates and `model.run(params)` evaluating them. `plans/flows-derived-outputs.plan.md`
deliberately did not port that registry (a flat global namespace, declaration order, outputs fixed
at build time, two pruning mechanisms that clobber each other), and `OutputSet` fixes all four.

Every summer2 capability is `full` in the coverage ledger (D1–D5), but porting a real model shows
the cost. `kiribati-tb-summer4/src/kiribati_tb/outputs.py` is 358 lines, most of it helpers
summer2 supplied:

| The port writes | summer2 had |
|---|---|
| `_compartments(where)` = `Compartments().select(sel).total()` | `request_output_for_compartments` |
| `_flow(flow, where)` = `FlowMass(f).select(Source(sel)).total().midpoint()` | `request_output_for_flow` |
| `_total(outputs, names)`, a loop of `ref + ref` | `request_aggregate_output` |
| `for age … for reach …` loops building `prevXage_15Xreach_r` names | the same loops |
| `restrict_outputs`, inlining intermediates (workaround W5) | `save_results=False` |
| `plan()` → `run(save=)` → `evaluate(result, params)` | `model.run(params)` |

## Design

Requests are **methods on `OutputSet`**, not a second registry. Each is one `set(name, expr,
keep=)` call; its docstring gives that expression, and a test checks the equivalence (`P4`).
Each returns the expression, so it can be chained or referenced like any other.

```python
out = OutputSet(flow_values="rate")                 # or "midpoint" for summer2 parity
out.compartments("infectious", state["I"], by=[age, reach, (age, reach)])
out.flow("incidence", "infection", source=age["15"], dest=None, values=None, by=age)
out.aggregate("tb_incidence", ["inc_lowinf", "inc_inf"])
out.cumulative("cum_inc", "incidence", start=2026.0, method="integral")   # or "sum"
out.function("ratio", f, "a", "b", Param("x"))      # str args are refs: defer(f)(refs…)
out.set("tmp", expr, keep=False)                    # an intermediate
out.fan_out("s", Compartments().select(state["S"]), age)
named = cm.run(params, t0=..., t1=..., dt=..., outputs=out)
```

**Intermediates.** `keep=False` (on `set` and every request) marks a name as an intermediate.
`keys()` still lists every name; `kept()` lists the rest. `evaluate(result, params, names=None)`
returns `names`, defaulting to `kept()`, and computes only those names and what they read.
`subset(names)` copies the set with only `names` kept (the port's `restrict_outputs`, made public
and without inlining). `posterior_runs` records `kept()`. `BayesianModel` evaluates every name, so
a target may score an intermediate. This is the one pruning mechanism.

**Fan-out.** `by=` is a property, or a sequence of groupings, each a property or a tuple of
properties. Each grouping is one hidden output `{name}[age,reach]` = `expr.sum_by(*grouping)`
(with `side=` for a flow), and each stratum is a row of it:
`ref(hidden).select(age[a] & reach[r]).sum_by()`. Each stratum then reads the small grouped array,
not the full save. It is not a smaller jaxpr: `evaluate` unrolls one step per named output either
way, so fan-out traces the same equations per stratum as a hand loop, plus one grouped sum per
grouping (checked in the notebook, section E). Names come from the set's `naming`: a template formatted
per property with `prop` and `trait` and appended (default `"X{prop}_{trait}"`, summer2-port
style), or a callable `(name, traits) -> str`.

**Shared saves.** Requests select and sum as post-ops on one `Compartments()` save and one
`FlowMass(flow)` save per flow, as the Kiribati port does. A model whose full edge save is too
large assigns a save-time-reduced leaf (`FlowMass(flow, where=, group_by=)`) directly.

**`run(outputs=)`** is exactly `outputs.evaluate(model.run(params, ..., save=outputs.plan(save)),
params)`; `save=` still sets `ts`, `dense` and `solver_stats`.

**Escape hatches.** Everything that worked still works and mixes freely with requests:
`outputs[name] = <any leaf or expression>`, `ref`, NumPy ufuncs, `defer`, `array()` /
`with_values()`, and `plan()` / `evaluate()` for a custom save plan, `jit` and calibration.

### Decisions (user, 2026-11-19)

1. Methods on `OutputSet`, not a façade.
2. Selectors only, no summer2 dict strata (`{"age": "15"}`); dicts would need a property map
   and are a second spelling.
3. `flow_values` defaults to `"rate"` (summer4's convention); ports set `"midpoint"`.
4. Lands now, before `CP6`. It uses only the public surface (`set`, `ref`, leaf constructors,
   `defer`), so step 38's rewrite of `OutputExpr` onto rate nodes must keep
   `tests/test_output_requests.py` green unchanged.

### Not in this step

- `request_computed_value_output` for any rate expression waits for step 40 (`CX32`, save any
  expression). After it, `out.set(name, expr)` saves one, so no extra request is added; that
  retires Kiribati workaround W2.
- `request_param_output`: `out.set(name, ...)` of a `Param` is a time-free array today. No request
  until there is a need.
- Porting `kiribati-tb-summer4/src/kiribati_tb/outputs.py` to requests is downstream work in that
  repository (golden values unchanged is the acceptance), after this is released.

## Files

`src/summer4/results/outputset.py` (requests, `set`, `kept`, `subset`, `fan_out`,
`evaluate(names=)`), `src/summer4/flows/compiled.py` (`run(outputs=)`),
`src/summer4/epi/calibration/{model.py,posterior_runs.py}`, `tests/test_output_requests.py`,
`examples/notebooks/39-output-requests.ipynb`, `docs/guide/04-derived-outputs.ipynb`,
`docs/evaluation/{composability.md,coverage-ledger.md}`, `docs/dev/roadmap.md`, `CHANGELOG.md`.

## Tests

`tests/test_output_requests.py`: each request equals its documented expression (values and, for
`compartments`, the expression itself) and NumPy on the raw save; requests on one flow share one
save; `keep=False` is computed but not returned; `evaluate(names=)` computes only what they read;
`subset`; fan-out names, values, totals, `side="dest"`, midpoint, naming template and callable;
fan-out's jaxpr grows per stratum no faster than a select per stratum; `run(outputs=)` equals plan, run and
evaluate.
