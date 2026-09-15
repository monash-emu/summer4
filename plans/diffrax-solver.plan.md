---
name: diffrax-solver
description: Phase 3 of flows/derived-outputs — a real solver seam with a diffrax backend, per-request save groups, pytree solver stats, and the documentation-porting infrastructure the later harvests need.
---

# Phase 3 — diffrax backend and solver info (`feat/diffrax-solver`)

## Context

`plans/flows-derived-outputs.plan.md` §"Phase 3" is the accepted design; this is
its execution plan. Phases 0–2 landed flows, `CompiledModel.run`, `Epoch` /
`TimeAxis`, `SavePlan` and a queryable `Result` over a fixed-step Euler.

Ledger `V2` (`SolverType / solver selection`) is `none`: there is no adaptive
backend and no solver selection. `CompiledModel.run` raises on anything but
`"euler"` (`src/summer4/flows/compiled.py:588`), and `diffrax>=0.7.2` plus
`equinox` are declared in `pyproject.toml:17-18` and `pixi.toml:16-19` and
entirely unused.

The roadmap pulls this work forward out of its natural ledger order (WP7) on
purpose: once diffrax is behind the seam, Phases 4 and 5 inherit
`SaveAt`/`SubSaveAt`, off-grid interpolated save times, dense output and solver
statistics instead of re-implementing them.

**Closes:** `V2` → `full` (WP7). 32 → 33 / 52 (63%).
**Unblocks:** textbook chapter 7, *Obtaining numerical solutions*.

Phase 3 also carries one-off documentation infrastructure, because it is the
first phase that harvests a chapter and the convention should be settled once.

## Where this plan departs from the roadmap

Three things the roadmap did not anticipate. Each is a real constraint found in
the Phase 2 code, not a change of design.

1. **`SolverInfo` has to become a pytree node** (§3d). `Result.tree_flatten`
   puts it in aux, and aux must be static; diffrax's stats are tracers under
   `jax.jit`.
2. **`SaveRequest.ts` is declared and never read** (§3b). `run` uses the plan's
   single `ts`. Phase 5 depends on per-request times, and `SubSaveAt` is the
   natural moment to close it.
3. **`describe()` sizes every output from one global `n`** (§3b), which is wrong
   the moment per-request `ts` works, and Phase 5's sparse-vs-dense footprint
   story depends on it.

`plans/flows-derived-outputs.plan.md` is committed history and is not edited.

## 3a. A real solver seam — `src/summer4/solvers/`

Today `euler`, `numpy_euler` and `_euler_save` all live in
`src/summer4/flows/compiled.py`. Give solvers their own subpackage, the way
`summer4.results` owns plan/eval/trace/result:

- **`solvers/base.py`** — the boundary both backends return across:

  ```python
  @dataclass(frozen=True, slots=True)
  class SolveSpec:
      t0: float
      t1: float | None
      steps: int | None
      dt: float
      rtol: float | None = None
      atol: float | None = None
      max_steps: int | None = None
      dense: bool = False

  @dataclass(frozen=True, slots=True)
  class SolveOutput:
      saved: dict[str, Any]
      stats: SolverInfo
      dense: Any | None = None
  ```

- **`solvers/euler_backend.py`** — `_euler_save` moves here with **no behaviour
  change**, including the `_is_arithmetic_subgrid` nested-scan fast path
  (`compiled.py:657`–`748`). It gains only the per-group loop from §3b.
- **`solvers/diffrax_backend.py`** — new (§3c).
- **`flows/compiled.py`** keeps exporting the public `euler` and `numpy_euler`
  (both are in `summer4.__all__`; moving them is a gratuitous API break).
  `CompiledModel.run` dispatches on `solver=`.

`run` gains `rtol`, `atol` and `max_steps`, and `solver=` accepts either a name
— `"euler"`, `"heun"`, `"tsit5"`, `"dopri5"` — or a diffrax solver instance so
an expert is not boxed in by the name table. An unknown name raises listing the
known ones. Importing the diffrax backend without the extra installed raises a
message naming `pip install summer4[jax]` rather than a bare `ModuleNotFoundError`.

Keep `euler` a supported reference stepper permanently: textbook chapter 7
compares manual evaluation, Euler and Runge–Kutta and needs all three.

## 3b. Save groups, and honouring `SaveRequest.ts`

`SaveRequest.ts` (`src/summer4/results/plan.py:67`) is documented as
"`None` -> the plan's default grid" and is never read — `run` builds one grid
from `expanded.ts` (`compiled.py:603`) and evaluates every request on it.

New module **`src/summer4/results/groups.py`**:

```python
@dataclass(frozen=True, slots=True)
class SaveGroup:
    name: str                       # "g0", "g1", ... stable for a given plan
    ts: NDArray[np.float64]
    keys: tuple[str, ...]

def group_requests(plan: SavePlan, default_ts: NDArray[np.float64]) -> tuple[SaveGroup, ...]:
    ...
```

Group requests by `blake2b(np.ascontiguousarray(ts, np.float64).tobytes())` —
**not** by `id()`. Two targets carrying equal-but-distinct arrays must land in
one group, or Phase 5 pays two full save passes for the same times. Requests
with `ts=None` join the group for `default_ts`. Order groups by first
appearance of their keys so `name` is deterministic.

Both backends consume groups:

- diffrax maps each group to one `SubSaveAt` (§3c);
- Euler runs one save pass per group, reusing the existing fast path when that
  group's `ts` is an arithmetic subgrid and the general lerp path otherwise. The
  state trajectory is computed once and shared across groups on the general
  path.

Then fix `CompiledModel.describe` (`compiled.py:538`–`556`), which currently
takes `n` from `n_saves` or `expanded.ts` and applies it to every output. Each
`OutputShape` must be sized from its own group's `ts`. Without this, Phase 5's
"sparse plan is smaller than the dense one" claim is unverifiable.

## 3c. The diffrax backend

- **The term.** `ODETerm(lambda t, y, args: model.vector_field(t, y, args))`
  over the raw array, using the existing `unpack_state` / `rebox` seam
  (`src/summer4/jax/state.py`) exactly as `_euler_save` does. `args` is `params`.
- **Saving.**
  `saveat = SaveAt(subs={g.name: SubSaveAt(ts=g.ts, fn=g.fn) for g in groups}, dense=plan.dense)`,
  where `g.fn(t, y, args)` calls `model.observe(t, rebox(y), args)` and evaluates
  only that group's keys through `build_save_fn` over a filtered plan. `sol.ys`
  returns `{group: {key: array}}` and is restructured into `Result.traces` at
  trace time from the static group spec — no traced dictionary manipulation.
- **Stepping.** `PIDController(rtol=rtol, atol=atol)` when either is given, else
  `ConstantStepSize()` with `dt0=dt`. `max_steps` passes through.
- **Off-grid save times now work properly.** diffrax drains every requested `ts`
  falling inside `(tprev, tnext]` and interpolates through the solver's local
  interpolant. This is the fix for estival's sharp edge, where
  `model_times.get_loc(t)` is an exact lookup and a target time not landing on a
  gridpoint is silently dropped by set intersection.
- **Dense output.** `dense=True` keeps the solution's interpolation on
  `Result.dense` and adds `Result.evaluate(t) -> PropertyData`, raising a clear
  error when the plan was not dense. State the cost honestly in the docstring:
  it allocates `max_steps` worth of interpolation coefficients and requires a
  finite `max_steps` — strictly heavier than any `ts`-based saving.

## 3d. `SolverInfo` becomes a pytree node

**The easy-to-miss change.** `Result.tree_flatten` puts `solver` in the aux
tuple (`src/summer4/results/result.py:78`). Aux must be hashable and static.
Under `jax.jit`, `sol.stats` values are *tracers*, so the first jitted `run`
that populates `SolverInfo` from diffrax breaks the pytree contract.

- Register `SolverInfo` with `register_pytree_node_class`: `num_steps`,
  `num_accepted_steps`, `num_rejected_steps` and `result_code` are **children**;
  `solver` (a name) and `dense` (a bool) stay aux.
- Move `solver` out of `Result`'s aux tuple and into its children.
- Add `SolverInfo.message` as a **host-side property** mapping `result_code`
  through diffrax's `RESULTS` enum, which carries human-readable messages on its
  members. A failure then survives jit as an integer and prints as prose.
  summer2 discards the entire `Solution` except `ys`, so a truncated solve there
  produces silently wrong arrays; this is the thing to not reproduce.
- Euler keeps returning a Python `int` for `num_steps`; that is a valid leaf.

`solver_stats=False` still omits the whole object (`solver=None`).

## 3e. Documentation infrastructure

Landed once here, because Phase 3 is the first phase to harvest a chapter.

### The ledger gains a `Ported` column

Append it to the **end** of both blocks:

```
| Ch | Title | Status | Blocker | Ported |
| Page | Status | Blocker | Ported |
```

Appending is safe. `scripts/coverage_report.py:36` indexes status at a fixed
*leading* position (`textbook` 2, `summer2docs` 1), which does not shift. The
cell holds a path relative to `docs/`, or an em dash when unported.

Extend `scripts/coverage_report.py`:

- `PORTED_COLUMN = {"textbook": 4, "summer2docs": 3}`;
- `--check` **fails** when a declared port names a file that does not exist
  under `docs/` — a claimed port must be real;
- `--check` **reports but does not fail** rows that are `full` with no port,
  printing them as a "publishable, not yet ported" backlog with a count.

The report-don't-fail split is deliberate. Textbook chapters 3, 5, 6, 9 and 11
are `full` with no port today, and the harvest policy for this work is
increment-only: a phase ports what it unblocked, and the standing backlog
belongs to a separate documentation branch. The printed line is what keeps that
backlog visible to the branch that will drain it.

Extend `tests/test_coverage_ledger.py` with a cell-count test for both blocks
(mirroring `test_api_ledger_shape`) and a `test_declared_ports_exist`.

### A porting convention — `docs/textbook/porting.md`

States, once, what every port must do:

- the BSD-2-Clause notice carried, with the source repository and a **pinned
  commit SHA** so a later reader can diff against what was adapted;
- vendored figures live in `docs/textbook/figures/<chapter>/`, with
  `docs/textbook/figures/LICENSE` carrying the copyright line and licence text;
- plotting is Plotly — already in the `docs` and `nb` pixi environments
  (`pixi.toml:53,68`) — driven from `Trace.to_pandas()`;
- prose is carried and adapted; **code is written in current summer4 idiom**,
  never transliterated from summer2.

### Two stale pages fixed

- `docs/textbook/index.md:14-25` still says "`euler` returns a final state only,
  so those chapters are not published as runnable notebooks yet". False since
  `941086b`, and AGENTS.md's "do not describe planned behaviour in the present
  tense" is exactly this class of error in reverse.
- `docs/textbook/roadmap.md` restates per-chapter ledger status, which the ledger
  contract forbids ("prose elsewhere quotes their totals; it does not restate
  their contents") and which has already drifted — it still says chapters 2–7
  and 9–11 are "blocked on results". Slim it to the content the ledger does not
  hold: the plotting convention, the figures, the licence decision, and the
  dependency-tier diagram. Point at the ledger for status.

## 3f. Tests — `tests/test_solvers.py`

- Euler and Dopri5 agree to `rtol=1e-4` on an analytic SIR.
- **The phase acceptance test.** One loss function, defined once in
  `tests/helpers/loss.py`, differentiated under both backends from identical
  source; `jax.grad` agrees to `rtol=1e-3`. Both paths end in `.at_times(TS)`,
  which is a no-op gather when the times already coincide. If this test needs
  two versions of the loss, the seam is wrong.
- A save time deliberately off-grid (`t=3.37` with `dt=1.0`) matches the
  analytic value through the solver's interpolant, and is not dropped.
- `SolverInfo` populated: `result_code == 0`, `.message` a non-empty string,
  accepted + rejected steps consistent with the total.
- `jax.jit(run)` with the model and plan static returns a `Result` that flattens
  and unflattens with traced solver stats — the §3d regression.
- Two equal-but-distinct `ts` arrays yield exactly one `SaveGroup`.
- Per-request `ts` is honoured by **both** backends — the §3b regression.
- `dense=True` then `result.evaluate(t)` matches the saved values at a save
  point; `dense=False` then `evaluate` raises.
- An unknown solver name raises listing the known names; a diffrax solver
  instance is accepted.
- `describe()` sizes a mixed-`ts` plan correctly before any solve.

`diffrax` and `equinox` are already declared — verify, do not re-add.

## 3g. Notebooks and the user gate

**Feature notebook** — `examples/notebooks/05-solvers.ipynb`: Euler vs Dopri5
against an analytic solution, adaptive stepping, `result.solver.num_steps`,
dense evaluation, and one deliberately off-grid save time. Plain Python, no
magics, asserting each claim (AGENTS.md notebook rules).

**Harvest.** Run `pixi run coverage`, read the "publishable, not yet ported"
line, and port what *this* phase unblocked: textbook chapter 7, *Obtaining
numerical solutions*, as `docs/textbook/07-numerical-solutions.ipynb`. It
compares manual evaluation, Euler and Runge–Kutta on one model — which is
precisely why `euler` and `numpy_euler` stay supported. Prose and figures
carried under the §3e convention; code in summer4 idiom. Set ledger row 7
`partial` → `full` and fill its `Ported` cell.

> ### Notebook gate — blocking
>
> Do not merge until the user has run these in `pixi run notebook` and ticked:
>
> - [ ] `examples/notebooks/05-solvers.ipynb` — the adaptive solver visibly takes
>   a different number of steps from Euler, and the off-grid save time is
>   *right*, not merely present.
> - [ ] `docs/textbook/07-numerical-solutions.ipynb` — reads as the chapter, and
>   the three methods are compared on the same model.
>
> Automation only proves the notebooks execute. The gate is whether the story a
> modeller reads makes sense.

## Verification

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch
pixi run coverage
pixi run -e docs docs-strict
pixi run test-all          # pinned JAX 0.6.x and latest
```

Ledger discipline: change `V2` and textbook row 7 in the same commit as the
code, then `pixi run coverage-write` and update the totals quoted in
`docs/evaluation/index.md` (32 → 33).

File in `futureplans/` rather than fixing here: anything noticed about the
rolling unroll already tracked in `futureplans/trace-rolling-jaxpr.md`.
