# Textbook roadmap

The [summer textbook](https://github.com/monash-emu/summer-textbook) has twenty
chapters. This page states, for each one, what summer4 would need in order to
publish it as a runnable notebook.

## Status summary

| Status | Chapters | Count |
|---|---|---|
| Ported | 1, and the structural half of 2 | 2 |
| Blocked on flows, rates and a solver | 2–11 | 10 |
| Blocked on the above **plus** mixing matrices | 12–16, 18, 19 | 7 |
| Blocked on the above **plus** contact-survey data handling | 16–19 | 4 |
| Blocked on the above **plus** calibration | 20 | 1 |

Ten of twenty chapters are blocked by a single tier of missing functionality:
flows, rates, and an integrator. That tier is the highest-leverage thing the
project could build.

## Chapter by chapter

| # | Title | Needs | Status |
|---|---|---|---|
| 1 | Infectious disease modelling | Nothing — prose | **Ported** ({doc}`01-introduction`) |
| 2 | Basic model construction | Compartments (have), `add_transition_flow`, `add_infection_frequency_flow`, `set_initial_population`, `run`, `get_outputs_df` | **Partial** ({doc}`02-model-structures`) |
| 3 | Thinking about flows | `add_transition_flow`, solver, outputs | Blocked |
| 4 | Thinking about flow rates | As above, plus competing-flow bookkeeping and sojourn-time outputs | Blocked |
| 5 | Series compartments and latency | Ragged stratification (have), chained transition flows, solver | **Partial** — structure only |
| 6 | Post-infection immunity | Waning and recovery flows; the SI/SIS/SIR/SIRS distinction *is* the flow set | **Partial** — structure only |
| 7 | Obtaining numerical solutions | A solver seam with selectable Euler and Runge-Kutta backends, plus a manually evaluable vector field | Blocked |
| 8 | Derived outputs | `request_output_for_compartments`, `request_output_for_flow`, `request_function_output`, results frames | Blocked |
| 9 | Transmission assumptions | `add_infection_frequency_flow` **and** `add_infection_density_flow`, `add_universal_death_flows` | Blocked |
| 10 | The reproduction number | Solver, outputs, and time-varying parameters for $R_t$ | Blocked |
| 11 | Cyclical epidemic dynamics | Waning flows, `add_replacement_birth_flow`, `add_universal_death_flows`, phase-plane outputs | Blocked |
| 12 | Heterogeneous mixing introduction | `Stratification.set_mixing_matrix`, density-dependent infection flow | Blocked |
| 13 | Mixing and transmission types | As 12, plus `set_population_split` and stratified derived outputs | Blocked |
| 14 | Assortative mixing | As 13, plus `add_infectiousness_adjustments` and `get_stratification` | Blocked |
| 15 | Susceptibility and infectiousness matrices | As 14, plus `set_flow_adjustments` with `Multiply` / `Overwrite` | Blocked |
| 16 | Thinking about contact surveys | As 15, plus loading and applying an empirical contact matrix | Blocked |
| 17 | Understanding empiric contact data | Contact-survey data utilities (largely data analysis, little model API) | Blocked |
| 18 | Implementing empiric survey data | As 16, plus matrix construction from survey records | Blocked |
| 19 | Adapting mixing matrices | As 18, plus matrix scaling against population data | Blocked |
| 20 | Calibration and uncertainty | Everything above, plus a Bayesian calibration workflow over JAX-differentiable models | Blocked |

## The dependency tiers

```{mermaid}
flowchart TB
    T0["Tier 0 — have<br/>Compartments, stratification, queries"]
    T1["Tier 1<br/>Flows · rates · parameters · solver · outputs"]
    T2["Tier 2<br/>Mixing matrices · infectiousness adjustments · population split"]
    T3["Tier 3<br/>Contact survey data handling"]
    T4["Tier 4<br/>Calibration and uncertainty"]

    T0 --> T1 --> T2 --> T3 --> T4

    T0 -.->|"ch 1, part of 2, 5, 6"| C0["2 chapters"]
    T1 -.->|"ch 2-11"| C1["10 chapters"]
    T2 -.->|"ch 12-15"| C2["4 chapters"]
    T3 -.->|"ch 16-19"| C3["4 chapters"]
    T4 -.->|"ch 20"| C4["1 chapter"]
```

Tier 1 is where the spike in `explorations/flows/` has already done design work:
joins, flow classes, a rate expression tree, adjustments and a `lax.scan` Euler
step all exist as prototypes. Promoting that spike — the thirteen-item list in
{doc}`../dev/explorations` — would unblock ten chapters in one move.

Tier 2 requires a force-of-infection mechanism with matrix-weighted coupling
between strata. Nothing in the current spike addresses that: the prototype's
rates are per-edge scalars and small expression trees, not a coupling across a
grouping.

## What a port would also need beyond the API

Even with tier 1 complete, publishing the textbook here requires three things
the repository does not have:

1. **A plotting convention.** The source textbook uses Plotly throughout, with
   `get_outputs_df()` returning a pandas frame. summer4 has no results object,
   no dataframe conversion, and the `docs` environment currently installs
   neither pandas nor Plotly.
2. **The chapter figures.** Chapters 2–6 and 12–19 depend on SVG diagrams
   (`sir_structure.svg`, `seir_transition.svg`, `immunity_structures.svg` and
   others) that live in the source repository and are not vendored here.
3. **A licence and attribution decision.** The source is BSD-2-Clause; a port
   needs the copyright notice carried and the adaptation stated, which
   {doc}`index` does for the two chapters present.
