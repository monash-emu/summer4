---
name: time-varying
description: WP5 — make time first-class in the rate tree and ship a JAX-traceable library of interpolation, sigmoidal, piecewise and step functions plus a dated-series data surface.
---

# WP5 — Time-varying functions (`feat/time-*`)

## Context

Every other part of a summer4 model may be parameterised in a time-varying
fashion, so this package lands before WP3 and WP6.

Today it cannot. A rate expression cannot see `t`. `_eval_rate`
(`src/summer4/flows/compiled.py:109`) receives only `derived`, `flow_values` and
the static `pmap`; `as_rate` (`src/summer4/flows/rates.py:191`) rejects
callables; and `Transform.fn` is invoked as `fn(prev, *args)` with no time
argument. The single escape hatch is `derived_fn(params, *, y, t)`, whose result
is reached by `FieldRef`. So a seasonal contact rate, an intervention step, or
the Gaussian importation pulse in `docs/case-studies/age-stratified-seirs.ipynb`
must all be hand-written inside one `derived_fn` closure:

```python
pulse = jnp.exp(-0.5 * ((t - params["seed_time"]) / params["seed_width"]) ** 2)
```

**Closes:** `P5` `P6` `P7` `P8` → `full` (WP5). 36 → **40 / 52 (77%)**.
**Unblocks:** summer2 `detailed/time-varying-functions`; textbook 10, whose
$R_t$ treatment needs time-varying parameters (`docs/textbook/roadmap.md`).
**Depends on:** the flows stack through `feat/sparse-targets`. Branch from the
tip of that stack, **not** from `main` — `main` carries only the taxonomy and
cannot `import summer4.flows` at all. See the *Delivery status* section of
`docs/evaluation/coverage-ledger.md`.

Five subphases, each one branch, in order. 5.1 is deliberately the smallest: it
is the warm-up that teaches the landing procedure.

| # | Branch | Ships | Ledger |
|---|---|---|---|
| 5.1 | `fix/describe-params` | `describe(params=...)` | — |
| 5.2 | `feat/time-node` | `Time()` in the rate tree | — |
| 5.3 | `feat/time-functions` | `linear` `sigmoidal` `piecewise` `step` | `P6` `P7` `P8`, 36 → 39 |
| 5.4 | `feat/timeseries-data` | `Data` from dated series | `P5`, 39 → 40 |
| 5.5 | `docs/time-varying-harvest` | the summer2 page, `docs/summer2/` | summer2docs row |

---

## Before you start — read this even if you have read it before

Read, in this order: `AGENTS.md`, `docs/evaluation/coverage-ledger.md`
(especially *Delivery status*), and every note in `futureplans/`.

**One subphase is one branch.** Branch from the tip of the previous subphase.
Copy this plan file onto your branch as `plans/time-varying.plan.md` before you
touch `src/` — `pixi run check-branch` fails without a plan.

`check-branch` (`scripts/check_feature_branch.py`) also fails unless a branch
that changes any `src/summer4/*.py` **also** touches `tests/` **and**
`examples/notebooks/`. This is unconditional; the "for new or changed public
modules" qualifier in `AGENTS.md` is not what the script implements. Where a
subphase has no new notebook of its own, it extends the one its sibling created.

Finish every subphase with, in order:

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch
pixi run coverage
pixi run -e docs docs-strict
pixi run test-all
```

**Ledger discipline, in the same commit as the code.** Change the affected rows'
`Status` in `docs/evaluation/coverage-ledger.md`, run `pixi run coverage-write`,
then update the totals quoted in `docs/evaluation/index.md` — both the
`**N of 52` complete line and the "any working route" covered line. `P5`–`P8` are
all `none` today, so each one you close raises *both* counts.

No new `Area` value is needed anywhere in this package: `parameters` already
exists in `AREA_TITLES` in `scripts/coverage_report.py`, and its title is already
"Parameters and time-varying functions". Inventing a new area is a code change,
and `tests/test_coverage_ledger.py::test_every_area_is_named` will fail.

**If you are blocked, record the truth.** Leave the row `partial`, name the real
blocker, leave `Ported` empty. Do not force a port and do not move a status the
code does not support. If you find a problem you are not fixing, write a
`futureplans/<slug>.md` note naming the symbol and what "done" looks like, and
add it to that folder's README list.

**The notebook is a blocking human gate.** Do not merge a subphase until the user
has run its notebook in `pixi run notebook` and signed it off.

### The five-site rule

5.2 and 5.3 both add `RateOps` subclasses. A new node must be added to **five**
places or it fails silently:

| Site | File | Failure if missed |
|---|---|---|
| `_eval_rate` | `flows/compiled.py:109` | `TypeError: Unsupported rate expression` |
| `_flow_refs` | `flows/rates.py:213` | `case _` returns `set()`; topo-sort silently misorders flows |
| `_field_paths` | `flows/rates.py:239` | `case _` returns `set()`; `computed_paths` silently incomplete |
| `_rate_bytes` | `flows/rates.py:328` | fallback is the class name — **two distinct nodes collide in the jit cache** |
| `__all__` | `flows/__init__.py`, `summer4/__init__.py` | not importable |

The `_rate_bytes` failure is the dangerous one: it produces wrong results from a
stale compiled model rather than an error.

---

## 5.1 — `fix/describe-params`

`CompiledModel.describe` (`src/summer4/flows/compiled.py:548`) sizes a `SavePlan`
by calling `jax.eval_shape` on a closure that passes `params=None`. Any
`derived_fn` that indexes params therefore raises:

```
ValueError: SavePlan failed shape inference (SaveFn must return statically
shaped arrays): 'NoneType' object is not subscriptable
```

This package makes params-reading models the normal case, and WP6 makes them
universal, so fix it first. It is also why
`docs/case-studies/age-stratified-seirs.ipynb` counts rows out of the plans' `ts`
arrays instead of calling `describe`, and why
`docs/user/10-targets-and-fitting.ipynb` can only demo sparse-vs-dense saving on
a model with no `derived_fn` at all.

### 5.1a. The change

Add an optional `params: object | None = None` to `describe`, threaded exactly
where `run` threads it. `jax.eval_shape` accepts `ShapeDtypeStruct` pytrees, so a
caller may pass either real params or a shape-only stand-in. Keep `None` working
for models without a `derived_fn`.

### 5.1b. Tests — `tests/test_results.py`

- `describe(plan, params=...)` on a model whose `derived_fn` indexes params
  returns a footprint instead of raising. This is the regression.
- `total_nbytes` from `describe` equals the actual bytes of the arrays a real
  `run` with the same plan produces.
- `describe(plan)` with no params still works on a model with no `derived_fn`.

### 5.1c. Notebook

Extend `docs/user/10-targets-and-fitting.ipynb` so the sparse-vs-dense comparison
runs on a model that *does* have a `derived_fn`, which is what a reader will
actually have. Touch `examples/notebooks/` as `check-branch` requires — the
natural edit is `examples/notebooks/07-targets.ipynb`, showing `describe` sizing
a plan for a params-reading model.

### 5.1d. Close the note

Delete `futureplans/describe-requires-params.md` and its bullet in
`futureplans/README.md` in the same commit. A note whose fix has landed is worse
than no note.

**Ledger discipline:** no rows change. `pixi run coverage` output is unchanged;
run it anyway.

---

## 5.2 — `feat/time-node`

Make `t` first-class in the rate tree, so any rate or adjustment can be a
function of time directly:

```python
TransitionFlow(
    "infection", state["S"], state["I"],
    rate=sigmoidal(Time(), t0=100.0, t1=140.0, lo=0.1, hi=0.4),
)
```

### 5.2a. The node — `src/summer4/flows/rates.py`

```python
@dataclass(frozen=True, slots=True)
class Time(RateOps):
    """The current model time, usable anywhere a rate expression is."""
```

Field-less is deliberate: every instance then compares and hashes equal, so two
models built from separately-constructed `Time()` nodes share a jit cache entry.
Do not give it a name, a unit, or an offset — an offset is `Time() - t0`, which
the `RateOps` arithmetic already builds as a `BinOp`.

### 5.2b. Threading `t`

`t` is **already in scope** at the top of `CompiledModel.observe`
(`compiled.py:428`) — it is the first parameter, and `_eval_derived` already
receives it. Pass it down through `_eval_aligned` (`:229`), `_eval_rate`
(`:109`) and `_apply_adjustments` (`:258`) as a keyword argument, matching how
`derived` and `flow_values` are already threaded.

`_eval_rate` gains one case:

```python
case Time():
    return t
```

`Time()` evaluates to a scalar, so it lands on the `shape == ()` branch of
`_align_rate` (`:193`) unchanged. No alignment work is needed.

Under `jax.jit` and under the diffrax backend, `t` is a **tracer**. Nothing may
branch on it in Python — no `if t > x`, no `float(t)`. This is the rule that
5.3's library also has to obey.

### 5.2c. The five-site rule, applied

- `_eval_rate` — the case above.
- `_flow_refs` — `Time` holds no `FlowRef`, so the existing `case _` fallback is
  correct. **Add an explicit `case Time(): return set()` anyway**, so a reader
  can see it was considered rather than missed.
- `_field_paths` — same, explicitly.
- `_rate_bytes` — `case Time(): return b"time"`.
- `__all__` — export `Time` from `summer4.flows` and from `summer4`.

### 5.2d. Tests — `tests/test_time_varying.py` (new)

- A flow with `rate=Time() * 0.1` on a one-compartment exit integrates to the
  analytic solution of $\dot y = -0.1\,t\,y$, to `rtol=1e-6` under `tsit5`.
- `Time()` works inside a `Transform` argument and inside a
  `Multiply(value=..., where=...)` adjustment — both go through
  `_eval_aligned`, and a missed thread in `_apply_adjustments` shows up only
  here.
- `Time() == Time()` and `hash(Time()) == hash(Time())`.
- **jit-cache hit:** compile the same model twice from separately-built `Time()`
  nodes and assert the second `run` does not retrace (count traces with a
  side-effecting `derived_fn` or compare `CompiledModel` digests directly).
- `jax.make_jaxpr` of a loss over a trajectory: the jaxpr node count does not
  grow with the number of save times. This is the `AGENTS.md` §JAX rule 2 gate
  every phase carries.

### 5.2e. Notebook

Create `examples/notebooks/08-time-varying.ipynb` with the `Time()` section — a
linearly ramping rate, asserted against the analytic solution. 5.3 and 5.4 extend
this same notebook rather than adding their own.

**Ledger discipline:** no rows change. `Time()` alone closes no summer2 symbol;
say so in the PR rather than implying progress.

---

## 5.3 — `feat/time-functions`

New module `src/summer4/timevarying.py`. Public: `linear`, `sigmoidal`,
`piecewise`, `step`, `gaussian_pulse`.

### 5.3a. The constraint that decides the design

`_adjust_bytes` (`rates.py:342`) hashes `Transform.fn` by `id()`:

```python
return b"tf" + str(id(adj.fn)).encode() + b"".join(_rate_bytes(arg) for arg in adj.args)
```

A library built out of closures would therefore produce a fresh digest on every
compile and **retrace the whole model every call**. `CompiledModel`'s own
docstring already warns about this for lambdas defined in loops.

So these are **structural nodes**, not closures.

### 5.3b. The node — `src/summer4/timevarying.py`

```python
@dataclass(frozen=True, slots=True)
class Interp(RateOps):
    """Interpolation between knots, evaluated at a rate expression."""

    kind: Literal["linear", "sigmoidal", "step"]
    breakpoints: tuple[float, ...]     # static and hashable
    values: tuple[RateOps, ...]        # Const or FieldRef — calibratable
    arg: RateOps                       # usually Time()
```

`values` being rate expressions rather than floats is the whole point:
summer2's `get_sigmoidal_interpolation_function(x_pts, y_pts)` accepts
`Parameter`s, and a modeller calibrating an intervention is calibrating the
*knots*. `breakpoints` stay static floats — they index the gather and cannot be
traced.

The public functions are thin constructors over it:

```python
def linear(arg, breakpoints, values) -> Interp: ...
def sigmoidal(arg, breakpoints, values, *, sharpness=1.0) -> Interp: ...
def step(arg, breakpoints, values) -> Interp: ...
def piecewise(arg, breakpoints, values) -> Interp:   # alias of step, summer2's P8 name
```

Also ship `gaussian_pulse(arg, centre, width, height)` as its own small node.
`docs/case-studies/age-stratified-seirs.ipynb` hand-writes exactly this shape for
importation seeding, and it is the most reusable single expression in the repo.
Its `centre`, `width` and `height` are rate expressions, so they calibrate.

### 5.3c. The five-site rule, and the trap in it

`Interp` holds rate expressions in **two** places — `values` and `arg`.
`_field_paths` and `_flow_refs` **must recurse into both**. If they do not, a
`FieldRef` knot never reaches `CompiledModel.computed_paths`, and a
`SavePlan(EVERYTHING)` silently omits it while the model still runs and produces
plausible numbers.

`_rate_bytes` must fold in `kind`, the breakpoints as bytes
(`np.asarray(breakpoints).tobytes()`, not `repr`), and the recursive bytes of
every `values` entry and of `arg`.

Because `Interp` lives in `summer4/timevarying.py` but the traversals live in
`summer4/flows/rates.py`, decide and state the import direction in the PR.
Recommended: define `Interp` in `flows/rates.py` alongside the other nodes and
re-export it from `summer4/timevarying.py`, which keeps every traversal's `match`
statement in one file and makes a missed site visible in review.

### 5.3d. Evaluation — all traceable

- `linear` — `jnp.interp(x, breakpoints, stacked_values)`.
- `step` / `piecewise` — `jnp.searchsorted(breakpoints, x)` then a gather.
- `sigmoidal` — a logistic blend between neighbouring knots. State the
  parameterisation of `sharpness` in the docstring in terms of the transition
  width, not as a bare slope, and say what it does at the ends.
- `gaussian_pulse` — `height * jnp.exp(-0.5 * ((x - centre) / width) ** 2)`.

No Python conditionals on `arg`. `jnp.interp` clamps outside the range rather
than extrapolating; say so in the docstring, because silently clamping a
calibration input is a real trap.

### 5.3e. Tests — `tests/test_time_varying.py`

- Each function matches a hand-computed reference at the knots and at midpoints.
- `step` is right-continuous at a breakpoint, and the test pins which side.
- **Calibratability:** `jax.grad` of a trajectory with respect to a `FieldRef`
  knot value is finite and matches a central-difference estimate. The claim of
  this subphase is that knots calibrate, so test it rather than assuming it.
- Two structurally equal `Interp`s hash equal; a model rebuilt from them hits the
  jit cache.
- Two `Interp`s differing only in a breakpoint value produce **different**
  digests. (This is the `_rate_bytes` collision test; it fails if you used
  `repr` of a float tuple carelessly or dropped a field.)
- `jax.make_jaxpr` node count is independent of the number of breakpoints times
  trajectory length.

### 5.3f. Notebook

Extend `examples/notebooks/08-time-varying.ipynb`: a seasonal contact rate, a
step-change intervention, and a sigmoidal ramp — each plotted and each asserted,
not just displayed.

**Ledger discipline:** `P6` (`get_linear_interpolation_function`), `P7`
(`get_sigmoidal_interpolation_function`) and `P8` (`get_piecewise_function`) →
`full`, with the summer4 equivalent column naming `summer4.timevarying`. 36 → 39.
Run `pixi run coverage-write` and update both quoted totals in
`docs/evaluation/index.md`.

---

## 5.4 — `feat/timeseries-data`

`P5` is summer2's `Data`. New module `src/summer4/data.py`: turn an observed,
dated series into interpolation knots on the model's own time axis.

### 5.4a. The surface

```python
Data.from_series(s, epoch)          # pandas Series indexed by date
Data.from_csv(path, epoch, *, time_column=..., value_column=...)
data.interp(kind="linear")          # -> Interp node from 5.3
```

Reuse `Epoch` from `src/summer4/time.py`. **Do not invent a second
date-conversion path** — `Target.from_series` in
`src/summer4/results/targets.py` already establishes how a dated pandas series
becomes model times, and the two must agree. If `Target.from_series` has logic
worth sharing, factor it out rather than duplicating it.

pandas is an optional extra (`pyproject.toml`'s `pandas` group). The module must
import without pandas installed; `from_series` raises a clear error naming the
extra if it is missing.

### 5.4b. Tests — `tests/test_time_varying.py`

- A round trip — dated series in, `interp` out, evaluated at those same dates —
  is exact.
- A time outside the observed range clamps to the end value, and the docstring
  says so. Add the test even though it is `jnp.interp` behaviour; it is the
  contract users will rely on.
- `Data.from_csv` and `Data.from_series` on the same data give equal `Interp`
  nodes, including equal digests.
- Importing `summer4.data` with pandas absent does not raise (simulate with a
  monkeypatched import if needed).

### 5.4c. Notebook

Extend `examples/notebooks/08-time-varying.ipynb` with a real dated series
driving a flow rate, dated through an `Epoch`, and plotted against the model
output on calendar dates.

**Ledger discipline:** `P5` (`Data`) → `full`. 39 → 40. `pixi run coverage-write`
plus both totals in `docs/evaluation/index.md`.

---

## 5.5 — `docs/time-varying-harvest`

Docs-only. No `src/summer4` change, so `check-branch` passes without a notebook
under `examples/`.

### 5.5a. Create `docs/summer2/`

`plans/flow-outputs.plan.md` §4h specified a `docs/summer2/` tree with "its own
index carrying the same attribution block as `docs/textbook/`". It was never
created. Create it now: `docs/summer2/index.md` with the attribution block, and
add it to the toctree in `docs/index.md`.

### 5.5b. Port the page

Port summer2's `detailed/time-varying-functions` to
`docs/summer2/time-varying-functions.ipynb`.

Follow `docs/textbook/porting.md`: carry and adapt the source prose with its
licence notice and a pinned source commit SHA, but **write every line of code in
current summer4 idiom** — `from summer4.timevarying import ...`,
`CompiledModel.run`, `SavePlan`, `Result`. Never transliterate summer2 calls.

Plot with Plotly through `Trace.to_pandas()`, **not** `Trace.plot` — the latter
hardcodes matplotlib and raises under the Plotly backend
(`futureplans/trace-plot-backend-coupling.md`).

### 5.5c. Harvest honestly

Set the summer2docs row `detailed/time-varying-functions` to `full` with its
`Ported` path. Declared `Ported` paths must exist under `docs/` or
`pixi run coverage` fails.

Then run `pixi run coverage` and read its *publishable, not yet ported* line.
Textbook chapter 10 (the reproduction number) lists time-varying parameters for
$R_t$ as its blocker and becomes attemptable. **Attempt it.** If a genuine
blocker appears, leave the row `partial`, name the real blocker, and leave
`Ported` empty — recording the truth is what the ledger is for. Do not force the
port to fill the slot.

### 5.5d. Update the ledger's own planning table

In the *Delivery status* section of `docs/evaluation/coverage-ledger.md`, move
WP5's row from "Ledger paragraph only" to `plans/time-varying.plan.md`, and add
the landed subphase branches to the *Landed phases* table.

> ### Notebook gate — blocking
>
> Do not merge any subphase of WP5 until the user has run these in
> `pixi run notebook` and ticked:
>
> - [ ] `examples/notebooks/08-time-varying.ipynb` — a modeller can tell from the
>       page alone how to make any rate time-varying, and the calibratable-knot
>       claim is visible rather than asserted in prose.
> - [ ] `docs/summer2/time-varying-functions.ipynb` reads as its source and runs.
> - [ ] any ledger row left `partial` names a blocker the user agrees is real.

---

## Verification

Per subphase, before asking for a merge:

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch
pixi run coverage
pixi run -e docs docs-strict
pixi run test-all
```

Phase-specific gates, restated:

- **5.1** — `describe` sizes a plan for a params-reading model, and its
  `total_nbytes` matches a real run.
- **5.2** — the `Time() * 0.1` model matches the analytic solution; a rebuilt
  model hits the jit cache; `Time()` works inside an adjustment, not only as a
  bare rate.
- **5.3** — `jax.grad` through a knot matches central differences; two `Interp`s
  differing only in a breakpoint have different digests.
- **5.4** — the dated round trip is exact; `summer4.data` imports without pandas.
- **5.5** — every declared `Ported` path exists; chapter 10 was attempted and its
  outcome recorded honestly either way.

**Ledger discipline across the package:** `P5` `P6` `P7` `P8` → `full`, 36 → 40
of 52. WP6 (`plans/epi-infection-mixing.plan.md`) starts from 40 and depends on
5.2's `t` threading, which it extends to thread `y`.
