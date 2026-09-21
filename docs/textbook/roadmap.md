# Textbook roadmap

The [summer textbook](https://github.com/monash-emu/summer-textbook) has twenty
chapters. Per-chapter status and blockers live only in the
{doc}`coverage ledger <../evaluation/coverage-ledger>` — this page holds what
the ledger does not: how a port is done, and how the chapters depend on the
stack.

## Porting

See {doc}`porting` for the licence notice, pinned source SHA, figure layout,
and Plotly convention. Fill the ledger `Ported` cell when a chapter ships.

## What a port also needs beyond the API

1. **A plotting convention.** Plotly via `Output.to_pandas()` (see {doc}`porting`).
2. **Chapter figures.** Vendored under `docs/textbook/figures/<chapter>/` with
   `docs/textbook/figures/LICENSE`.
3. **Licence and attribution.** BSD-2-Clause notice carried; adaptation stated.

## The dependency tiers

```{mermaid}
flowchart TB
    T0["Tier 0 — have<br/>Compartments, stratification, queries"]
    T1["Tier 1 — have<br/>Flows · rates · CompiledModel · euler"]
    T2["Tier 2 — have<br/>Trajectories · Result · adaptive solvers"]
    T3["Tier 3<br/>Mixing matrices · infectiousness · population split"]
    T4["Tier 4<br/>Contact survey data handling"]
    T5["Tier 5<br/>Calibration and uncertainty"]

    T0 --> T1 --> T2 --> T3 --> T4 --> T5

    T0 -.->|"ch 1, part of 2, 5, 6"| C0["taxonomy chapters"]
    T2 -.->|"publishable: ch 2-11"| C2["results + solvers"]
    T3 -.->|"ch 12-15"| C3["mixing"]
    T4 -.->|"ch 16-19"| C4["contact surveys"]
    T5 -.->|"ch 20"| C5["calibration"]
```

Tier 2 ships: `CompiledModel.run` returns a queryable `Result`, and `solver=`
selects Euler or a diffrax method. See {doc}`../user/08-flows` for flows and
`examples/notebooks/05-solvers.ipynb` for solver selection.

Tier 3 ships: `summer4.epi` provides `MixingMatrix` and `ForceOfInfection`.
`TraitMatrix` still moves people, not transmission.
WP3 (initial population / population split) is applied; chapter 13 is unblocked
for porting. Chapters 14–15 are ported (15 partial — no first-class
susceptibility surface).
