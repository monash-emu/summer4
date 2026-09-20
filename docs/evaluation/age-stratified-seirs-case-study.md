# Case study: an age-stratified SEIRS model

An assessment of one concrete modelling problem against the
{doc}`coverage-ledger`, carried out by building and running the model rather
than by reading the ledger. The runnable version is
{doc}`../case-studies/age-stratified-seirs`.

```{admonition} Verdict
:class: important

The problem is **fully expressible on the shipped feature set**, with no
additions to `summer4`. Six pieces are hand-written rather than declared, and
every one of them is an open work package — WP3, WP5, WP6 and WP10. Nothing
was unreachable.
```

## The problem

An age-stratified SEIRS model with an ageing population that skews old;
age mixing through a contact matrix, homogeneous to begin with; a parameterised
introduction of infection into a population that starts with none; and
age-varying infectiousness as a parameter. Generate outputs at chosen
parameters, add noise, thin them to a sparse sample, and calibrate back against
those targets.

## Requirement by requirement

| Requirement | Mechanism today | Ledger |
|---|---|---|
| SEIRS crossed with three age bands | `PropertyMap.from_property(state).stratify(age)`, 12 rows | `S1` `S2` `full` |
| Ageing between bands | `TraitChain.from_breakpoints(age)` (or explicit pairs) — one flow | `S6` `partial` |
| Births replacing deaths | `EntryFlow(..., death.sum())`; flow-to-flow rates are topologically sorted | `F4` `F5` `full` |
| Universal death | `ExitFlow("death", Everything(), rate)` | `F3` `full` |
| Population that skews old | Emergent: integrate the demography alone to its steady state | — |
| Initial state with `I == 0` | `PropertyData.wrap(...).at[sel].set(...)` passed as `run`'s `y0` | `L4` `L5` `S8` `none` |
| Mixing matrix | A `numpy` array in `params`; `K @ shedding` inside `derived_fn` | `M1` `none` |
| Age-varying infectiousness | `params["infectiousness"] * i_by_age` before the matmul | `A4` `none` |
| Per-band force of infection | `PropertyData.broadcast_over(age, foi)`, then rate alignment | `F7` `F8` `partial` |
| Parameterised infection inflow | `EntryFlow` with a derived rate and `split=` across bands | `F6` `full` |
| Time-varying inflow shape | A Gaussian pulse written in JAX | `P6`-`P8` `none` |
| Outputs at target parameters | `run(..., solver="tsit5")` → `Result`; `FlowMass(where=Dest(...))` | `D1` `D2` `V2` `full` |
| Capturing the force of infection | `ComputedValue(path=("foi",))` | `D6` `full` |
| Noise, then sparse selection | Host-side `numpy`; `Target(..., quantity=...)` | WP11 |
| Merging observation times | `TargetSet.plan(SavePlan())`; equal times share one save group | WP11 |
| Calibration | `jax.jit(jax.value_and_grad(loss))` + `optax.adam` | WP10 for the likelihood |

### What "none" means for the initial population

`L4`, `L5` and `S8` read as the hardest blockers in the ledger, and they are
the least binding in practice. `CompiledModel.run` takes the initial state as
its second positional argument, and it accepts a bare array, a `PropertyData`
or a `State`. `PropertyData.at[selector].set(...)` builds one from a selector.
An arbitrary initial population — including one that skews old and holds no
infection — has always been available. What is missing is a *declarative*
`set_initial_population` and a population-split helper, which is what WP3 adds.

In this case study the initial population was not written down at all. The
demography was integrated on its own for three hundred years with transmission
switched off, and its final state became `y0`. That steady state is checkable
in closed form, and the run reproduces it to float32 precision:

| Band | Model, after 300 years | Analytic steady state |
|---|---|---|
| `0-14` | 0.1579 | 0.1579 |
| `15-64` | 0.3239 | 0.3239 |
| `65+` | 0.5182 | 0.5182 |

## Measured calibration results

Weekly observations between days 21 and 175 in each of three bands (23 per
band), multiplied by lognormal noise with a log standard deviation of 0.10.
Adam for 600 iterations at a learning rate of 0.05, on
`jax.jit(jax.value_and_grad(...))` through an adaptive `tsit5` solve. The whole
notebook executes in about 31 seconds.

Fitting the two parameters that set the size and timing of the wave, with the
infectiousness gradient held at its known value:

| Parameter | Target | Recovered |
|---|---|---|
| `contact_rate` | 0.550 | 0.551 |
| `seed_rate` | 5.000 | 4.893 |

The likelihood plan solves 69 rows against 753 for a daily grid, and the three
targets collapse into a single solver save group because they share
observation times.

## The substantive finding: homogeneous mixing cannot identify infectiousness

Freeing the two outer infectiousness weights as well, under homogeneous
mixing, recovers an **inverted** age gradient — decreasing with age, where the
data were generated with it increasing — at a loss slightly *below* the loss at
the parameters that generated the data:

| | `0-14` | `15-64` | `65+` | loss |
|---|---|---|---|---|
| Target | 0.603 | 0.862 | 1.207 | 0.04088 |
| Recovered | 1.401 | 1.259 | 0.716 | 0.04071 |

This is not an optimiser failure. Capturing the force of infection with
`ComputedValue` shows the mechanism exactly: under a homogeneous matrix the
largest difference in $\lambda_a$ between bands, over all times, is **0.0**.
Every band experiences the same force of infection, so the infectiousness
weights enter only through one scalar sum, and many gradients give the same
sum.

Swapping in a row-normalised assortative matrix — a one-line change, because
the matrix is a parameter — makes $\lambda_a$ band-specific (0.0052, 0.0070,
0.0096 at day 60) and the gradient becomes identifiable: refitting recovers
0.487 / 0.982 / 1.168, monotone increasing as generated, with `contact_rate` at
0.543 against a target of 0.550.

This is the clearest available argument for WP6. Mixing matrices are not a
convenience over hand-written coupling; without off-diagonal structure, a
whole class of age-specific parameters is unidentifiable in principle.

### A conditioning note, not a limitation

`TargetSet.residuals` subtracts on the raw scale. Notification counts span
three orders of magnitude across an epidemic wave, so a raw-scale sum of
squares is dominated by the peak, and Adam stalls on it — an early attempt
converged to `contact_rate` 0.195 against a target of 0.550. Least squares on
the log scale, which is the estimator multiplicative noise implies, recovers
`contact_rate` to under 1% from the same start. `residuals` cannot express
that, so the case study uses `TargetSet.gather` and writes the comparison out.
That is precisely the boundary WP10 is meant to move.

## Sharp edges encountered

Each is recorded as a note in
[`futureplans/`](https://github.com/monash-emu/summer4/tree/main/futureplans).

| Edge | Where | Cost |
|---|---|---|
| `PropertyData.where(sel, x)` **replaces** matching compartments | `jax/propertydata.py:112` | Read as "keep", it silently gives a force of infection over $S+E+R$. The model still integrates; the epidemic saturates on day one. |
| `CompiledModel.describe` passes `None` for params | `flows/compiled.py:548` | Cannot size a plan for any model whose `derived_fn` indexes `params` — i.e. this one. Raises `'NoneType' object is not subscriptable`. |
| `TargetSet.residuals` reshapes but cannot reduce | `results/targets.py:166` | A stratified save cannot meet a one-dimensional series; targets must be one band each, or go through `SaveFn`. |
| `Trace.plot` hardcodes matplotlib | `results/trace.py:469` | `plot(legend=False)` raises under the Plotly backend. |

Two further notes already in `futureplans/` bear directly on calibrating
against case counts: `state-ledgers-incidence.md` (quadrature incidence is not
the solver's accumulated flow mass) and `trace-rolling-jaxpr.md`
(`Trace.rolling` unrolls under `jit`).

## Consequences for the roadmap

No ledger row moves: this is documentation over shipped features, and
`pixi run coverage` totals are unchanged. The case study does sharpen two
priorities:

- **WP6 is worth more than its row count suggests.** It closes four rows
  (`F7` `F8` `A4` `M1`), but the identifiability result above shows it also
  determines whether age-specific parameters can be estimated at all.
- **WP3 is worth less than its `none`s suggest.** Three rows are marked `none`,
  but the capability is present and was used here without difficulty. WP3 buys
  ergonomics and discoverability, not new reach.
