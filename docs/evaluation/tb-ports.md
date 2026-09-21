# TB model ports

**Can two existing tuberculosis models be built in summer4, and what stands in
the way?** This page is the authoritative, machine-checked answer. Plans on other
branches and in the downstream port repositories cite its row IDs (`KI4`, `TM6`)
and the work packages (`WP13`) that close them.

```{admonition} Contract for anyone changing this page
:class: important

1. The `ports` table is the single source of truth. Prose quotes its totals; it
   does not restate its rows.
2. `Status` uses the definitions of the {doc}`coverage-ledger`: `full`,
   `partial` (achievable, but the modeller hand-writes something summer4 should
   supply) or `none`.
3. `Closed by` is `—` for a `full` row, and otherwise exactly one work package
   declared in the ledger's packages table *and* listed in the port order below.
4. Run `pixi run coverage-write` after editing, then `pixi run coverage`.
   `tests/test_coverage_ledger.py` fails on a bad status, an undeclared work
   package, a stale readiness table, or a stale quoted total.
5. When a work package lands, move its rows to `full` in the same commit.
```

## The two models

| Model | Source | Library | Scale | Calibration |
|---|---|---|---|---|
| Kiribati TB screening | [`monash-emu/kiribati_tb_modelling`](https://github.com/monash-emu/kiribati_tb_modelling) | summer2gen (a summer2 fork adding `add_infection_generalised_flow`) | 10 states × 8 uneven age bands × 2 reachability strata = 160 compartments; 1850–2035; ~150 derived outputs; 12+ screening scenarios | estival + pymc `DEMetropolisZ`, hierarchical target sd, nevergrad MAP, posterior full runs per scenario |
| TB macroeconomics | [`monash-emu/tb_macroeconomics`](https://github.com/monash-emu/tb_macroeconomics) | summer3wip | 7 states, ragged clinical × infectious strata on `active`, 3 age bands; 1800–1999 | numpyro NUTS, Poisson likelihood |

Both ports land in new repositories that depend on summer4 from GitHub:
`monash-emu/kiribati-tb-summer4` (`plans/kiribati-tb-summer4-port.plan.md`) and
`monash-emu/tb-macro-summer4` (`plans/tb-macro-summer4-port.plan.md`). The
summer4 work that closes the rows below is
`plans/tb-ports-feature-completeness.plan.md`. "Fully implemented" means
numerical parity against golden outputs from the original library, then the
original analyses re-run on numpyro.

## Verdict

- **tb_macro** — tb_macro rows complete today: **8 of 9**, and every other row is `partial`. The force of infection is `ForceOfInfection(kind=FOIKind.GENERALISED)`; calibration is still a hand-written numpyro model (the original hand-writes that too).
- **Kiribati** — Kiribati rows complete today: **13 of 23**. Every capability except calibration and scale has *some* route, but a faithful, calibratable port would re-implement several layers by hand. Remaining blockers:
  1. **Output algebra** at summer2 scale — `KI13`–`KI17`. `Trace` has no operators, `cumulative()` has no start, `FlowMass` names one flow, there is no summer2 midpoint convention and no `Result` → frame.
  2. **Calibration workflow** — `KI18`–`KI21`. No priors, likelihoods, gradient-free sampler or posterior-run tooling exist.
  3. **Scale** — `KI22`. Nothing measures a 160-compartment, 185-year model, and a solver that exceeds its step ceiling fails silently.

## Ports ledger

<!-- ledger:ports -->
| ID | Model | Capability | Status | Route today | Closed by |
| --- | --- | --- | --- | --- | --- |
| KI1 | Kiribati | State × 8 uneven age bands × reachability map | `full` | `PropertyMap.from_property(state).stratify(age).stratify(reach)` | — |
| KI2 | Kiribati | Ageing between uneven bands at rate 1/width | `full` | `TraitChain.from_breakpoints(age)` | — |
| KI3 | Kiribati | Initial population: seed in `clin_inf`, reachability split by `Param` | `full` | `InitialPopulation` | — |
| KI4 | Kiribati | Generalised FOI `M @ (I_g / N_g**exp)` with a calibrated exponent | `full` | `ForceOfInfection(kind=FOIKind.GENERALISED, exponent=Param(...), mixing=...)` | — |
| KI5 | Kiribati | Infectiousness weights by compartment × age | `full` | `infectiousness=[(selector, weight), ...]` multiplied per compartment before the group sum; a `group_by` trait map is sugar for the same pairs | — |
| KI6 | Kiribati | Time-varying, parameterised mixing matrix (UN weights, fertility age gaps, spectral normalisation) | `full` | `Lookup(Param("mixing"), floor(Time() - year0))` inside `MixingMatrix`; the yearly stack is a parameter | — |
| KI7 | Kiribati | Flow adjustments by age and reachability with params and time functions | `full` | Stacked `Multiply(..., where=)` and `Transform` | — |
| KI8 | Kiribati | Deaths recycled to (`mtb_naive`, age 0) preserving reachability | `full` | `TransitionFlow(state[x] & age[a], state["mtb_naive"] & age["0"], ...)` | — |
| KI9 | Kiribati | Births into one stratum from sigmoidal interpolation × param | `full` | `EntryFlow` with `sigmoidal(...) * Param(...)` | — |
| KI10 | Kiribati | Per-age time series from UN tables (death rates, treatment outcomes) | `full` | `Data.table(times, values, over=age).interp()` — one `TableInterp`, a `GroupedRate` over age | — |
| KI11 | Kiribati | Interpolation knots derived with log / max (screening rate, outcome floors) | `full` | `log` / `maximum` on a `TableInterp` or on scalar `Interp` knots | — |
| KI12 | Kiribati | Computed values (detection rates, matrix distance) | `full` | `ComputedValue` over `derived_fn` | — |
| KI13 | Kiribati | Outputs summed over several flows and filtered by strata | `partial` | One `FlowMass` per flow added by hand, or `SaveFn` | WP15 |
| KI14 | Kiribati | Output arithmetic: per-capita, percentages, × parameter | `partial` | `Trace` × scalar/array, and `eval_closed` for a parameter expression such as `tanh(Param(...))`; no name-aligned `Trace` ÷ `Trace` | WP15 |
| KI15 | Kiribati | Cumulative output from a start year | `partial` | `between(...).cumulative()` then re-padding | WP15 |
| KI16 | Kiribati | summer2 midpoint flow-output convention (parity) | `partial` | Hand-written averaging of saved rates | WP15 |
| KI17 | Kiribati | Named output set to a frame / parquet | `partial` | Per-trace `to_pandas` concatenated by hand | WP15 |
| KI18 | Kiribati | Priors and Normal targets, including a prior-distributed sd | `none` | — | WP10 |
| KI19 | Kiribati | Gradient-free posterior sampling (replacing `DEMetropolisZ`) | `none` | — | WP10 |
| KI20 | Kiribati | MAP fit (replacing nevergrad) | `partial` | Hand-written optax loop, as in the case study | WP10 |
| KI21 | Kiribati | Posterior full runs × scenarios, quantiles, averted differences | `none` | — | WP10 |
| KI22 | Kiribati | Verified compile time, step cost and solver safety at TB scale | `none` | — | WP16 |
| KI23 | Kiribati | summer4 installable from a tagged GitHub release | `full` | Pin `tag = "v0.2.0a3"` | — |
| TM1 | tb_macro | Ragged map: clinical × infectious only on `active` | `full` | `stratify(prop, where=state["active"])` | — |
| TM2 | tb_macro | Partial destination (even split), collapse, expand | `full` | `identity_join` equal split; source-only properties dropped | — |
| TM3 | tb_macro | Ageing 0 → 5 → 15 | `full` | `TraitChain.from_breakpoints(age)` | — |
| TM4 | tb_macro | Rates as functions of `t` and params (triangular seed, tanh scale-up) | `full` | `clip(h * (1 - abs(Time() - peak) / w), 0)` and a `tanh` scale-up rate tree | — |
| TM5 | tb_macro | Per-age FOI `I / N**exp` with rel_sus per source compartment | `full` | `kind=FOIKind.GENERALISED` plus `adjust=(Multiply(Param("rel_sus"), where=source),)` on the infection flow | — |
| TM6 | tb_macro | Initial population with even split over ragged strata | `full` | `InitialPopulation` | — |
| TM7 | tb_macro | Rolling-sum flow target queried at times | `full` | `FlowMass` → `rolling(7, how="sum")` → `at_times` (unrolled jaxpr, fixed in WP15) | — |
| TM8 | tb_macro | Poisson likelihood, uniform prior, NUTS | `partial` | Hand-written numpyro model over `run` | WP10 |
| TM9 | tb_macro | summer4 installable from a tagged GitHub release | `full` | Pin `tag = "v0.2.0a3"` | — |
<!-- /ledger:ports -->

## Port order

The order in which the feature plan lands the packages. The readiness table
below is computed in this order.

<!-- ledger:port-order -->
| Step | WP | Why here |
| --- | --- | --- |
| 1 | WP12 | Downstream repos need a tag to pin before anything else |
| 2 | WP13 | Math nodes and tables are used by WP3, WP14 and both ports |
| 3 | WP3 | Initial population; parallel with WP14 |
| 4 | WP14 | Generalised FOI needs WP13's `Pow` |
| 5 | WP15 | Output algebra for Kiribati's ~150 outputs |
| 6 | WP16 | Scale and solver safety before long calibrations |
| 7 | WP10 | Calibration consumes everything above |
<!-- /ledger:port-order -->

## Readiness after each package

Computed by `scripts/coverage_report.py`; do not edit by hand.

<!-- ledger:port-readiness -->
| After | Kiribati | tb_macro |
| --- | --- | --- |
| today | 13 / 23 | 8 / 9 |
| WP12 | 13 / 23 | 8 / 9 |
| WP13 | 13 / 23 | 8 / 9 |
| WP3 | 13 / 23 | 8 / 9 |
| WP14 | 13 / 23 | 8 / 9 |
| WP15 | 18 / 23 | 8 / 9 |
| WP16 | 19 / 23 | 8 / 9 |
| WP10 | 23 / 23 | 9 / 9 |
<!-- /ledger:port-readiness -->

## Limitations that stay in the ports

These are not summer4 gaps; the port plans own them.

- **Spectral normalisation is not reverse-differentiable as written.** The
  Kiribati mixing builder calls `jnp.linalg.eigvals` on `C = S·diag(pop)`.
  Because `S` is symmetric and `pop` positive, `C` has the spectrum of the
  symmetric `diag(√pop)·S·diag(√pop)`, so `eigvalsh` gives the same radius with
  gradients.
- **The mixing builder loops in Python** over 36 age-band pairs. Vectorise it and
  build one matrix per year before the solve.
- **Duplicate detection flows in tb_macro.** `passive_detection` (constant) and
  `detection` (time-varying) both move `active` to `treatment`. The port keeps
  both for parity and documents it.
- **No ordinal age selectors** (`age >= 15`). Age groups stay explicit lists; this
  is a convenience, not a capability gap.
