# Kiribati port gaps: fix the small ones, record the rest

Roadmap step 52, `feat/kiribati-gaps`. Closes `CX51`–`CX56` and `CX58` (package `CP11`); records
`CX57` and `CX59`–`CX62` as `deferred`, each with a `futureplans/` note.

## Context

Two ports of the Kiribati TB model to summer4 found the gaps below:

- the first port, `kiribati-tb-summer4` (and the shared core `tb-core`), whose remaining
  workarounds are listed in their `docs/summer4-workarounds.md` (`W*`);
- the clean, summer2-familiar port `monash-emu/tb-kiribati-s2s4` (local
  `kiribati-tb-summer4-clean`), which matches the summer2 goldens and reported gaps 1–4 in its
  hand-over. Gaps 5 and 6 come from its `model_summer4.py`, which uses `TraitChain` within the
  state property.

Gap 7 was found running gate notebook 40 (`40-save-expressions.ipynb`).

## Decisions

| Gap | What | Decision | Finding |
|---|---|---|---|
| 1 | `Lookup` reads its table only from a `Param` / `FieldRef` | **Fix** | `CX51` |
| 2 | `jnp` functions fail on `GroupedRate` arguments to `defer` | **Fix** | `CX52` |
| 3 | No exact running total of a flow | **Record** (too large) | `CX57`, `futureplans/state-ledgers-flow-integral.md` |
| 4 | Fan-out naming is per `OutputSet`, not per request | **Fix** | `CX53` |
| 5 | `TraitChain`'s docstring describes only ageing | **Fix** (docs) | part of `CX54` |
| 6 | `TraitChain` per-pair `rates` are plain numbers only | **Fix** | `CX54` |
| 7 | A `GroupedOutput` of a plain `Capture` loses its property map | **Fix** | `CX55` |
| W4 | No `sign=` on `PosteriorRuns.differences` (averted burden) | **Fix** (small) | `CX56` |
| W6 | No forward-mode `BayesianModel` | **Record** | `CX59`, `futureplans/bayesian-model-forward-mode.md` |
| W7 | `optimize` maps starts only by `vmap`; `Optax` passes no `value_fn` | **Fix** `value_fn` (small, `CX58`); **record** the start mapping | `CX58` done; `CX60`, `futureplans/optimize-start-mapping.md` |
| W8 | `MCMCRun` save/load; tail ESS in `StopRule` | **Record**. `extra_fields` (step 47, `CX36`) and the deadline (`StopRule(max_seconds=)`) are already on `main` | `CX61`, `futureplans/mcmc-run-save-load-tail-ess.md` |
| W9, W11 | No joint coordinate map on `BayesianModel` | **Record** (one finding for both) | `CX62`, `futureplans/bayesian-model-coordinate-map.md` |

## Design of the fixes

### Gap 1: `Lookup(table=<any rate expression>)`

`Lookup.__init__` takes any rate expression (via `as_rate`, so an array literal is an
`ArrayConst`), and rejects only a scalar `Const`. The hoist walk (`build_hoist_table`) walks the
table as well as the index, so a parameter-only table, such as
`field_refs(defer(stack_fn)(Params()), S).stack`, is evaluated once per run and slotted; the
per-step work is the gather. The Kiribati yearly mixing stack then needs no second `defer` and no
`prepare_fn`.

`BayesianModel`'s docstring stops pointing to `prepare_fn` for a parameter-derived stack and names
this spelling.

**Test** (`tests/test_kiribati_gaps.py`): a computed stack read by `Lookup` gives the same
trajectory as the stack passed as a parameter; the table's `Item` node is in the hoist table; a
`Const` table raises.

### Gap 2: `jnp` on `GroupedRate`

`GroupedRate.__jax_array__` returns `jnp.asarray(self.data)`, so `jnp.exp(g)`, `jnp.where(g > 0,
…)` and friends accept it. Their result is a plain array without the grouping. `_eval_defer` puts
it back: when `fn` returns a plain one-dimensional array, every `GroupedRate` argument has the same
grouping, and the array's length is that grouping's trait count, the result is wrapped as a
`GroupedRate` with that grouping. A result of any other shape is returned as is (a `(K, K)`
mixing matrix built from a grouped population stays a matrix). NumPy functions keep working through
`__array_function__` and keep the grouping themselves.

`defer`'s and `Defer`'s docstrings state the contract: which argument types `fn` receives, that
NumPy and `jnp` both work on a `GroupedRate`, and the re-wrapping rule.

**Test:** `defer(lambda g: jnp.exp(-g))(TableInterp…)` gives the same flow rates as
`np.exp(-…)`; a function using `jnp.maximum` on two grouped arguments works; a `(K, K)` result is
not wrapped.

### Gap 4: per-request `naming=`

`compartments`, `flow` and `fan_out` take `naming: Naming | None = None`; `None` uses the set's
`naming`. `Naming` gains a third form: a **whole-name template**. A string containing `{prop}` or
`{trait}` is the existing per-property suffix; any other string with a placeholder is formatted
once per stratum with `name` and one field per property name, holding that property's trait:

```python
out.flow("all_cause_mortality", "all_cause_mortality", by=[(state, age)],
         naming="{name}_from_{state}_age_{age}")
```

A template naming a property that is not in the grouping raises `KeyError` with the known fields.

**Test:** the per-request naming wins over the set's; the whole-name template produces the
Kiribati names; a set without per-request naming is unchanged.

### Gaps 5 and 6: `TraitChain` within one property; rate-expression `rates`

The docstring describes `TraitChain` as any explicit list of `(source trait, dest trait)` pairs
within one property, not only ageing, with an example of paired state moves (clinical progression
over `(subclin_inf, clin_inf), (subclin_lowinf, clin_lowinf)`) and of a split from one source
(`incipient` to two states with `rates=(1 - p, p)`).

`rates` entries may be numbers or rate expressions. Numbers stay on the static edge `scale`.
For a non-number entry, `actualize` leaves that pair's `scale` at 1 and adds a
`Multiply(rate, where=Source(prop[src]) & Dest(prop[dest]), precedence=-1)` adjustment, which is
exactly the hand-written `Dest` adjustment the clean port uses. Precedence `-1` runs it before any
`Overwrite` (level 0), matching a numeric scale, which is applied before every adjustment. The
user's declared `flow.adjust` is unchanged; only the actualized edges carry the extra adjustment,
so it hoists and digests like any other.

**Test:** a chain with `rates=(1 - Param("p"), Param("p"))` matches the same flow written with
`Dest` adjustments; numeric rates are unchanged; a non-rate entry raises `TypeError`.

### Gap 7: a plain `Capture` keeps its property map

`_collect_capture_meta` takes the property map. For a `Capture(name, inner)` it records
`(_infer_over(inner, pmap),)` when the inner expression has a grouping, so `values_for` returns
`PropertyData` and `.select()` works, as it does for `ForceOfInfection`'s own capture.

**Test:** `GroupedOutput("cap")` of `Capture("cap", Reduce(age))` evaluates to `PropertyData` over
`age` and `.select(age["0"])` works.

### W4: `differences(sign=)`

`PosteriorRuns.differences(..., sign=1)`; `sign=-1` gives `ref - scenario` (averted burden,
positive when the scenario helps), quantiled per draw, and the relative column divides that by the
reference draw. The docstring stops calling the default "averted".

**Test:** `sign=-1` quantiles equal the quantiles of the negated per-draw differences, not the
negated quantiles.

### W7 (part): `Optax` passes `value_fn`

`_OptaxMethod.step` calls `tx.update(grads, state, z, value=loss, grad=grads,
value_fn=potential_fn)`, so `optax.lbfgs()` (zoom line search) runs under `wf.Optax`. optax's
`GradientTransformationExtraArgs` ignores the extra keywords for Adam and the plateau chain.

**Test:** `wf.optimize(..., method=wf.Optax(optax.lbfgs()))` reaches the optimum of a small
model.

## Recorded, not fixed

Each has a `CX` finding with evidence and a `futureplans/` note that says what done looks like:

- **Gap 3** (`CX57`): confirm `futureplans/state-ledgers-flow-integral.md` and add the Kiribati
  evidence: in screening scenario 3, notifications from midpoint counts were up to 41 % off
  (2027) and the campaign was overstated 2.8× and split across 2026/2027; only putting the 0.01-year
  ramp times on the save grid plus `jump_ts` got within 0.7 %.
- **W6** (`CX59`), **W7 start mapping** (`CX60`), **W8 save/load and tail ESS** (`CX61`), **W9 and
  W11** (`CX62`).

## Notebook

`examples/notebooks/41-kiribati-gaps.ipynb`, an acceptance notebook: one section per fix, each with
a *What to check* box and a titled Plotly figure or displayed object.

## Files

`src/summer4/flows/{rates.py,stages.py,compiled.py,join.py,actualize.py}`,
`src/summer4/results/outputset.py`, `src/summer4/epi/calibration/{model.py,posterior_runs.py}`,
`src/summer4/epi/calibration/workflow/optimize.py`, `tests/test_kiribati_gaps.py` (new),
`examples/notebooks/41-kiribati-gaps.ipynb` (new), `docs/evaluation/{composability.md,tb-ports.md}`,
`docs/dev/roadmap.md`, `futureplans/*.md`, `CHANGELOG.md`.
