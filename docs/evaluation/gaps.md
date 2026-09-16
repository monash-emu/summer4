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

**State:** **done** for the expression tree (`FieldRef`, `FlowRef`, `Multiply` /
`Overwrite` / `Transform`, `derived_fn`) and for the time-varying library
(`Time()`, `summer4.timevarying`, `summer4.data`). A full compute graph is not
warranted yet.

### 1.3 A solver seam

**State:** a fixed-step Euler via JAX `lax.scan`, plus a NumPy reference loop
in tests. `euler` returns the final state only.

**Needed:** trajectories (WP2), an adaptive backend (diffrax; WP7), and solver
selection. Textbook chapter 7 compares hand-stepped Euler against Runge-Kutta
and needs all three.

### 1.4 Initial population

`set_initial_population`, and a redistribution API for splitting a population
across a stratification.

**State:** the mechanism is trivially available — `parent_row` makes it a gather
and a divide, as demonstrated in {doc}`../user/06-immutability-and-provenance` —
but no API wraps it. Deferred as "a one-shot redistribution of `y`, not a flow
property".

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

`add_infection_frequency_flow` and `add_infection_density_flow`. This is
**not** a small addition to the flows layer: rates are per-edge scalars,
whereas a force of infection is a reduction over a grouping fed back into a
rate. Expressible today by writing `contact * I / N` in `derived_fn` (F7/F8
`partial`).

### 2.2 Mixing matrices

`set_mixing_matrix`, with the infection flow becoming a matrix-weighted coupling
between strata. Textbook chapters 12–15 and 19 are entirely about this, as is
summer2's `examples/09-mixing-matrices`. `TraitMatrix` moves *people*, not
transmission.

### 2.3 Infectiousness and susceptibility adjustments

`add_infectiousness_adjustments`, plus flow adjustments interacting with the
mixing matrix. Textbook chapter 15 exists specifically to explain the three
equivalent ways of expressing these.

## Priority 3 — blocks the remaining chapters

### 3.1 Contact-survey data handling

Loading, validating, inspecting and scaling empirical contact matrices.
Chapters 16–19.

### 3.2 Calibration

A Bayesian workflow over JAX-differentiable models. `numpyro` and `optax` are
already declared as an optional extra and are unused. Chapter 20.

### 3.3 Real-world time

`ref_date` / epoch handling, so model times can be dates. Every applied example
in both corpora eventually needs this.

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

Still missing:

- **Vendored textbook figures.** Chapters 2–6 and 12–19 embed SVGs from the
  source repository under BSD-2-Clause.
- ~~**A plotting convention.**~~ Settled: Plotly through the pandas plotting
  backend over `Trace.to_pandas()`, as in the summer2 documentation. See
  {doc}`../dev/plotting` for the decision and the Sphinx renderer it requires.
- **A documentation CI check.** `pixi run -e docs docs-strict` should run on
  every pull request, since the site executes its own claims.

## The single highest-leverage move

**Force of infection and mixing** (ledger WP6). WP2 landed trajectories and a
results object, WP4 landed flow outputs, WP7 landed an adaptive solver, and
WP11 landed sparse calibration targets, so a model can now be built, run,
queried and fitted. WP6 is what remains between that and the eight textbook
chapters on heterogeneous mixing.

Its row count (`F7` `F8` `A4` `M1`) understates it.
{doc}`age-stratified-seirs-case-study` measures the reason: a hand-written
homogeneous mixing matrix gives every stratum an *identical* force of infection,
so age-varying infectiousness is not identifiable — a fit recovers an inverted
age gradient at a lower loss than the parameters that generated the data.
Off-diagonal mixing structure is not a convenience over hand-written coupling;
without it a whole class of stratum-specific parameters cannot be estimated at
all.

`Present` / `Absent` binding is **settled**: they are non-binding in flow
pairing.
