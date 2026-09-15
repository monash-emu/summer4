# Textbook roadmap

The [summer textbook](https://github.com/monash-emu/summer-textbook) has twenty
chapters. This page states, for each one, what summer4 would need in order to
publish it as a runnable notebook.

## Status summary

| Status | Chapters | Count |
|---|---|---|
| Ported | 1, and the structural half of 2 | 2 |
| Modelling content expressible; blocked on a trajectory / results | 2–7, 9–11 | 9 |
| Blocked on derived outputs | 8 | 1 |
| Blocked on mixing matrices | 12–15 | 4 |
| Blocked on contact-survey data handling | 16–19 | 4 |
| Blocked on calibration | 20 | 1 |

Flows, rates and a fixed-step Euler ship in `summer4`. The binding constraint
for publishing chapters 2–11 is a **results object** (ledger WP2).

## Chapter by chapter

| # | Title | Needs | Status |
|---|---|---|---|
| 1 | Infectious disease modelling | Nothing — prose | **Ported** ({doc}`01-introduction`) |
| 2 | Basic model construction | Compartments (have), flows (have), `set_initial_population`, trajectory, `get_outputs_df` | **Partial** ({doc}`02-model-structures`) |
| 3 | Thinking about flows | Trajectory, outputs | Blocked on results |
| 4 | Thinking about flow rates | As above, plus sojourn-time outputs | Blocked on results |
| 5 | Series compartments and latency | `TraitChain` (have), trajectory | **Partial** — structure only |
| 6 | Post-infection immunity | Distinct `FlowModel`s (have); trajectory | **Partial** — structure only |
| 7 | Obtaining numerical solutions | Selectable Euler and Runge-Kutta, plus a manually evaluable vector field (the field exists) | Blocked on WP7 |
| 8 | Derived outputs | Request API, results frames | Blocked |
| 9 | Transmission assumptions | Hand-written FOI (have as `partial`), deaths, trajectory | Blocked on results |
| 10 | The reproduction number | Trajectory, outputs, time-varying parameters for $R_t$ | Blocked |
| 11 | Cyclical epidemic dynamics | Replacement births (have), deaths (have), phase-plane outputs | Blocked on results |
| 12 | Heterogeneous mixing introduction | `Stratification.set_mixing_matrix`, density-dependent infection flow | Blocked |
| 13 | Mixing and transmission types | As 12, plus `set_population_split` and stratified derived outputs | Blocked |
| 14 | Assortative mixing | As 13, plus `add_infectiousness_adjustments` | Blocked |
| 15 | Susceptibility and infectiousness matrices | As 14 | Blocked |
| 16 | Thinking about contact surveys | As 15, plus loading and applying an empirical contact matrix | Blocked |
| 17 | Understanding empiric contact data | Contact-survey data utilities | Blocked |
| 18 | Implementing empiric survey data | As 16, plus matrix construction from survey records | Blocked |
| 19 | Adapting mixing matrices | As 18, plus matrix scaling against population data | Blocked |
| 20 | Calibration and uncertainty | Everything above, plus a Bayesian calibration workflow | Blocked |

## The dependency tiers

```{mermaid}
flowchart TB
    T0["Tier 0 — have<br/>Compartments, stratification, queries"]
    T1["Tier 1 — have<br/>Flows · rates · CompiledModel · euler"]
    T2["Tier 2<br/>Trajectories · results · outputs"]
    T3["Tier 3<br/>Mixing matrices · infectiousness · population split"]
    T4["Tier 4<br/>Contact survey data handling"]
    T5["Tier 5<br/>Calibration and uncertainty"]

    T0 --> T1 --> T2 --> T3 --> T4 --> T5

    T0 -.->|"ch 1, part of 2, 5, 6"| C0["2 chapters"]
    T1 -.->|"expressible: ch 2-11"| C1["content"]
    T2 -.->|"publishable: ch 2-11"| C2["10 chapters"]
    T3 -.->|"ch 12-15"| C3["4 chapters"]
    T4 -.->|"ch 16-19"| C4["4 chapters"]
    T5 -.->|"ch 20"| C5["1 chapter"]
```

Tier 1 is in the package. See {doc}`../user/08-flows`.

Tier 3 requires a force-of-infection mechanism with matrix-weighted coupling
between strata. `TraitMatrix` moves people, not transmission.

## What a port would also need beyond the API

Even with trajectories, publishing the textbook here requires:

1. **A plotting convention.** The source textbook uses Plotly throughout, with
   `get_outputs_df()` returning a pandas frame.
2. **The chapter figures.** Chapters 2–6 and 12–19 depend on SVG diagrams
   that live in the source repository and are not vendored here.
3. **A licence and attribution decision.** The source is BSD-2-Clause; a port
   needs the copyright notice carried and the adaptation stated, which
   {doc}`index` does for the two chapters present.
