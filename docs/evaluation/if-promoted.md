# If the flows spike were promoted

{doc}`coverage-ledger` is the authoritative row-by-row record; this page reads
it in one direction. {doc}`feature-completeness` measures summer4 as shipped:
6 of 52 summer2 API symbols complete, 12%. This page asks a different question — **what would the numbers be
if `explorations/flows/` became `summer4` proper?**

The spike is real, tested and executable ({doc}`../dev/flows/index`), so this is
not speculation about a design; it is an accounting of working code that happens
to sit outside the package.

```{admonition} The headline
:class: important

Promoting the spike takes the API ledger from **6/52 complete (12%) to 21/52
(40%)**, and from 7/52 covered to **31/52 (60%)**. The textbook chapters whose
*modelling content* becomes expressible go from 4 to 10.

It moves the number of **fully runnable** summer2 notebooks from 0 to 0, and
fully runnable textbook chapters from 1 to 1.

The reason is a single deliberate scoping decision: the spike has no timeseries.
`euler` returns the final state only, and `refine-flows.plan.md` lists "save-at /
timeseries results" as out of scope. Every summer2 notebook and almost every
textbook chapter ends by running a model and plotting a trajectory.
```

## API coverage, before and after

Per-area counts are `complete` rows; see {doc}`coverage-ledger` for the
row-by-row detail and the `covered` figures.

| Area | Now | Promoted | Change |
|---|---|---|---|
| Model lifecycle | 0 / 7 | **1 / 7** | `FlowModel` ≈ a model object; `compile()` ≈ `finalize()` |
| Compartments and stratification | 4 / 8 | 4 / 8 | unchanged — initial-population split stays deferred |
| Compartment queries | 2 / 3 | 2 / 3 | `actualize()` / `ActualizedFlow` answers `query_flows` |
| Flows | 0 / 8 | **6 / 8** | all eight constructors become expressible |
| Flow adjustments | 0 / 4 | **3 / 4** | `adjust=` with `Multiply` / `Overwrite` / `Transform` / `where=` |
| Mixing | 0 / 1 | 0 / 1 | unchanged |
| Parameters and time-varying functions | 0 / 9 | **4 / 9** | `FieldRef`, `FlowRef`, `t`, callables; no interpolation helpers |
| Derived outputs | 0 / 8 | **1 / 8** | `derived_fn` ≈ `add_computed_value_func` |
| Solver | 0 / 2 | 0 / 2 | `euler` is `partial`: fixed step, final state only |
| Real-world time | 0 / 2 | 0 / 2 | unchanged |
| **Total (complete)** | **6 / 52 (12%)** | **21 / 52 (40%)** | **+15** |
| **Total (covered)** | 7 / 52 (13%) | 31 / 52 (60%) | +24 |

### Where the gains are

**All eight flow constructors become usable (0 → 8 covered, 6 complete).** `TransitionFlow`, `ExitFlow` and
`EntryFlow` cover transition, death, universal death, crude birth, replacement
birth and importation directly — {doc}`../dev/flows/01-flow-types` demonstrates
replacement birth reading another flow's mass via
`death.sum_over(location)`. The two infection flows are expressible but not
primitives: the user writes `contact * I / N` in `compute_derived_params` and
passes `D.foi` as the rate. That is more flexible than summer2's
`add_infection_frequency_flow`, and more work.

**Flow adjustments (0 → 3).** `adjust=[D.seasonal, Overwrite(0.0,
where=age["0-4"]), Transform(np.minimum, D.foi_cap)]` covers what summer2 does
with `set_flow_adjustments`, `Multiply` and `Overwrite` — with the deliberate
difference that adjustments belong to the flow, not to a `Stratification`.
`add_infectiousness_adjustments` is the one that does not translate, because it
weights a compartment's contribution to a force of infection that does not exist
as a library concept.

**Parameters (0 → 4).** `derived_refs` over a `NamedTuple` gives schema-checked,
IDE-completable `Parameter` equivalents; nested bundles give
`D.migration.baseline * D.migration.seasonal`; `t` reaches `derived_fn` every
step, so time-varying rates work. What is missing is the *library* of time
functions — linear and sigmoidal interpolation, piecewise — which summer2's
`detailed/time-varying-functions` page is entirely about.

**Flow introspection (2 → 3 covered).** `actualize(flow, pmap)` returns an
`ActualizedFlow` with `src_idx`, `dest_idx`, `weight`, `scale` and `n_edges`,
which is a better answer to "what does this flow actually do?" than
`query_flows`.

### Where nothing changes

- **Mixing matrices.** `TraitMatrix` looks superficially like
  `set_mixing_matrix` and is a different thing: it moves *people* between strata
  (migration), whereas a mixing matrix weights *transmission* between strata.
  Nothing in either spike addresses the latter.
- **Derived outputs.** `FlowRef` exposes a flow's mass to *other flows* inside
  the vector field. Nothing exposes it to the caller, and there is no request,
  no aggregation, no cumulative output and no results object.
- **Initial population.** Still built by hand with `pm.select`, exactly as the
  notebooks do. The spike deferred this explicitly.
- **Real-world time.** Untouched.

## Documentation coverage, before and after

### summer2 notebooks

| Page | Now | Promoted | What still blocks it |
|---|---|---|---|
| `01-basic-model` | No | **Partial** | Timeseries, initial population, results frame |
| `03-derived-outputs` | No | No | Derived outputs |
| `04-flow-types` | No | **Partial** | All eight flows expressible; running and flow outputs are not |
| `06-stratification-introduction` | Partial | **Partial +** | Adjustments now work; infectiousness and running do not |
| `07-age-stratification` | Partial | **Partial +** | `TraitChain` gives ageing; population split and running do not |
| `08-strain-stratification` | Partial | **Partial +** | Strain flows expressible; strain-aware FOI is hand-written |
| `09-mixing-matrices` | No | No | Mixing matrices |
| `10-derived-outputs-stratified` | No | No | Derived outputs |
| `11-flows-between-strata` | No | **Partial** | Exactly `TraitChain` / `TraitMatrix`; running and outputs are not |
| `detailed/time-varying-functions` | No | **Partial** | `t` reaches rates; interpolation helpers do not exist |
| `detailed/InitialPopulationGraphobject` | No | No | Initial population, parameters |

**0 → 0 fully reproducible. 3 → 7 partially reproducible.**

### Textbook chapters

| Outcome | Now | Promoted |
|---|---|---|
| Fully ported | 1 (ch 1) | 1 (ch 1) |
| Modelling content expressible, trajectory missing | 3 (ch 2, 5, 6) | **11** (ch 2–11) |
| Blocked on mixing | 4 (ch 12–15) | 4 (ch 12–15) |
| Blocked on contact-survey data | 4 (ch 16–19) | 4 (ch 16–19) |
| Blocked on calibration | 1 (ch 20) | 1 (ch 20) |

The chapters that move are 3, 4, 7, 8, 9, 10 and 11, plus a substantial
deepening of 2, 5 and 6. Two are worth calling out:

**Chapter 6 stops being embarrassing.** {doc}`../textbook/02-model-structures`
currently has to admit that `SI` and `SIS` are *the same model* to summer4,
because the distinction is entirely in the flows. With the spike promoted, that
chapter's four immunity structures become four genuinely different objects.

**Chapter 7 comes closest to fully portable.** It is about obtaining numerical
solutions: manual evaluation of the system, then Euler, then Runge–Kutta. The
spike supplies a manually evaluable `vf(t, y, params)` and a forward Euler, which
is two of its three sections. Only the Runge–Kutta comparison is missing.

## The binding constraint moves

Today the binding constraint is **flows**. After promotion it is **results**:

```{mermaid}
flowchart LR
    A["Now<br/>blocked on flows"] -->|promote the spike| B["Then<br/>blocked on trajectories"]
    B -->|timeseries + outputs| C["10 textbook chapters<br/>7 summer2 notebooks"]
    C -->|mixing matrices| D["4 more chapters"]
    D -->|survey data| E["4 more"]
    E -->|calibration| F["1 more"]
```

Every summer2 example notebook and nearly every textbook chapter ends the same
way: `model.run(...)`, `get_outputs_df()`, plot. The spike produces a vector
field and steps it, and then throws away everything but the final state.

That is a small gap in code and a large gap in outcome. `_euler_jax` already uses
`lax.scan`; returning the stacked per-step states instead of only the carry is a
few lines, and a NumPy loop that appends is fewer. What is genuinely missing is
the *convention* — a results type, the choice of a dataframe library, and a
plotting idiom — none of which the project has decided.

## Recommendation

The sequencing that maximises documentation coverage per unit of work:

1. **Promote the spike** (the thirteen items in {doc}`../dev/explorations`),
   starting with making `PropertyMap` hashable. Takes API coverage to ~48%.
2. **Add trajectories and a results object** — stacked `lax.scan` output, a
   results type, one dataframe and plotting convention. This is what converts
   "expressible" into "publishable" for ten textbook chapters and seven summer2
   notebooks.
3. **Then** derived-output requests, which are mostly aggregation over index sets
   that `partition` and `group_by` already provide.
4. **Then** force of infection and mixing matrices — the largest genuinely
   unprototyped design problem remaining.

Steps 1 and 2 together would take the project from "cannot complete a user task"
to "can complete most of the tasks the summer textbook sets", which is a
different kind of project.
