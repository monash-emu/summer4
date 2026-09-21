# Gaps and recommendations

Every blocker found while attempting to reproduce the summer2 documentation and
the summer textbook, in priority order.

## Priority 1 — blocks ten textbook chapters and eight summer2 notebooks

### 1.1 Flows

`TransitionFlow`, `EntryFlow`, `ExitFlow` as public API, with the query join
that pairs source and destination compartments across leftover strata.

**State:** **done.** Import from `summer4`. `Present` / `Absent` are
non-binding in pairing; `strict_pairing=True` raises when an unbound property
would move people. Inspect edges with {class}`~summer4.flows.edges.EdgeMap`
and `Source` / `Dest` on {meth}`~summer4.flows.compiled.CompiledModel.edges`.

### 1.2 Rates and parameters

A parameter representation and a way to attach time- and state-dependent rates
to flows.

**State:** **done** (WP5). Expression tree (`FieldRef`, `FlowRef`, `Multiply` /
`Overwrite` / `Transform`, `derived_fn`) plus the time-varying library
(`Time()`, `summer4.timevarying`, `summer4.data`). A full compute graph is not
warranted yet.

### 1.3 A solver seam

**State:** **done** (WP2 + WP7). `CompiledModel.run` returns a `Result`
trajectory; Euler is the reference stepper; `solver=` selects adaptive
diffrax backends (Heun, Tsit5, Dopri5, or a diffrax instance). Textbook
chapter 7 is ported at `docs/textbook/07-numerical-solutions.ipynb`.

### 1.4 Initial population

`set_initial_population`, and a redistribution API for splitting a population
across a stratification.

**State:** **done** (WP3: L4, L5, S8 `full`). `InitialPopulation` / `Split` /
`REMAINDER` with ragged-aware even defaults; `CompiledModel.initial_state` and
`run` without `y0`. See {doc}`../dev/run-stages` and
`examples/notebooks/10-initial-population.ipynb`.

### 1.5 Derived outputs and a results object

Request mechanisms for compartment-based, flow-based, aggregate, cumulative and
function outputs; a results type; a dataframe view.

**State:** done (ledger WP2 and WP4). `CompiledModel.run` returns a `Result` of
named `Trace`s; `SavePlan` declares what to keep; the query surface covers
select, aggregate, cumulative, calendar resample, rolling and interpolated
`at_times`, and `to_frame` / `to_pandas` give the dataframe view. Flow-output
polarity queries (`sum_over(..., side=)`, `incidence`, `integrate`) and
validated `ComputedValue` capture ship alongside.

### 1.6 `PropertyMap` must become hashable

**State:** done in Phase 0 (`feat/taxonomy-prereqs`). Maps hash by
`(properties, history, blake2b-16 digest of codes and parent_row)`, so equal
rebuilt maps share a hash and work as `jax.jit` static arguments.

## Priority 2 — blocks seven more chapters

### 2.1 Force of infection

**State:** **done** (WP6). `ForceOfInfection(kind=FOIKind.FREQUENCY|DENSITY)`
(F7, F8 `full`). Example: `examples/notebooks/09-epi-models.ipynb`.

### 2.2 Mixing matrices

**State:** **done** (WP6). `summer4.epi.MixingMatrix` weights transmission
between strata (M1 `full`). `TraitMatrix` still moves *people*, not
transmission. Textbook chapters 12 and 14 are ported; chapter 13 is unblocked
by WP3's population split (port on this stack).

### 2.3 Infectiousness and susceptibility adjustments

**State:** **infectiousness done** (A4 `full`) via
`ForceOfInfection(infectiousness=...)`. **Susceptibility still open** —
there is no FOI surface symmetric to infectiousness; chapter 15 is therefore
`partial` and scales susceptibility via flow `adjust=` or matrix row scaling.
See `futureplans/foi-susceptibility-surface.md`.

## Priority 3 — blocks the remaining chapters

### 3.1 Contact-survey data handling

Loading, validating, inspecting and scaling empirical contact matrices.
Chapters 16–19 (WP9). Depends on WP6 (applied).

### 3.2 Calibration

A Bayesian workflow over JAX-differentiable models. Sparse `Target` /
`TargetSet` fits exist (WP11); probabilistic likelihoods / priors are WP10.
Chapter 20.

### 3.3 Real-world time

**State:** **done** (WP2). `Epoch` / `TimeAxis` map calendar dates on
`Result.times`.

## Priority 4 — quality of life

Detailed in {doc}`user-satisfaction`; summarised here.

| Gap | Effort | Impact |
|---|---|---|
| `__hash__` on `PropertyMap` | ~~Small~~ | **Done** (Phase 0); see 1.6 |
| `__len__`, `__getitem__`, `__iter__` | Small | `__len__` done (Phase 0); `__getitem__` / `__iter__` still open |
| `PropertyMap.from_properties([...])` | ~~Small~~ | **Done** (Phase 0) |
| `to_frame()` for polars/pandas | ~~Small~~ | **Done** (Phase 0; polars, lazy import) |
| Serialise a map or its `history` | Medium | Reproducible model structures |
| A `filter` / `drop` operation | Medium | Excluding impossible combinations after the fact |
| Consistent `partition` / `group_by` return types | ~~Small~~ | **Done** (Phase 0; `group_by` returns `Groups`) |

## Priority 5 — documentation infrastructure

Previously absent, now in place:

- `docs/` with Sphinx, myst-nb, pydata-sphinx-theme and executed notebooks;
- a `docs` pixi environment and `docs` / `docs-strict` / `docs-serve` /
  `docs-clean` tasks;
- `.readthedocs.yaml`.

Still missing / partial:

- **Vendored textbook figures.** Partially done for ported chapters (figures
  under `docs/textbook/figures/` for chapters 3–6 and 8). Remaining chapters
  that need diagrams (especially 12–19 when unblocked) still need copies under
  the BSD-2-Clause notice, or redraws.
- ~~**A plotting convention.**~~ Settled: Plotly through the pandas plotting
  backend over `Trace.to_pandas()`, as in the summer2 documentation. See
  {doc}`../dev/plotting` for the decision and the Sphinx renderer it requires.
- **A documentation CI check.** `pixi run -e docs docs-strict` should run on
  every pull request, since the site executes its own claims.

## The single highest-leverage move

**WP3 (initial population) is applied.** WP2–WP7, WP4, WP5, WP6 and WP11 are
also applied: a model can be built with infection and mixing, given a
declarative initial population, run to a `Result`, queried, time-parameterised
and fitted to sparse targets.

The next leverage is **WP15** (output algebra), then WP16 and WP10. WP12
(pinnable release), WP13 (rate-tree math, tables, ageing sugar — `KI2` `KI6`
`KI10` `KI11` `TM3` `TM4`) and WP14 (generalised FOI and compartment × age
infectiousness — `KI4` `KI5` `TM5`) are applied. See
`plans/tb-ports-feature-completeness.plan.md` and the notes in `futureplans/`
(`derived-fn-blocks-hoisting`, `mixing-matrix-per-call-normalisation`,
`wp10-preprocess-is-prepare-fn`).

`Present` / `Absent` binding is **settled**: they are non-binding in flow
pairing.
