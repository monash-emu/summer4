# Project evaluation

An assessment of summer4's progress against two questions: **how complete is the
feature set**, and **how well is it likely to serve its users**.

```{toctree}
:maxdepth: 2

coverage-ledger
feature-completeness
docs-coverage
user-satisfaction
age-stratified-seirs-case-study
tb-ports
gaps
```

## Headline

| Measure | Result |
|---|---|
| Public API symbols | 47 |
| Package source | taxonomy (NumPy) plus `summer4.flows`, `summer4.jax`, `summer4.epi`, and `summer4.timevarying` |
| summer2 API surface exercised by its own docs, complete in summer4 | **47 of 52 (90%)** |
| … with any working route (complete or partial) | 52 of 52 (100%) |
| summer2 documentation notebooks reproducible as runnable summer4 | **8 of 11** (2 more partially) |
| summer textbook chapters reproducible as runnable summer4 | **12 of 20** (2 more partially) |
| Implemented layers of the intended stack | taxonomy + flows + results + epi + timevarying + initial population |
| Ceiling of the planned roadmap (WP2–WP16) | 47 of 52 (90%) — see {doc}`coverage-ledger` |

summer4 ships a compartment taxonomy, flows, results, epidemiology
(`summer4.epi`), time-varying rates (`summer4.timevarying` /
`summer4.data`), and declarative initial populations. Declare a map, attach
named flows (including infection and mixing), set an initial population,
compile to a `CompiledModel`, and `run()` to a queryable `Result`
with selectable Euler or adaptive diffrax solvers. What remains is mainly
a first-class FOI
susceptibility surface (chapter 15 partial), WP9 (contact-survey data —
chapters 16–19), and WP10 (Bayesian calibration — chapter 20).

## How to read this section

- {doc}`coverage-ledger` is the **authoritative, machine-checked record**:
  every summer2 symbol, every textbook chapter, and the ordered work packages
  that lead to full coverage. Every number quoted anywhere else on this site
  comes from it. Read this one first.
- {doc}`feature-completeness` compares the public API against summer2's,
  symbol by symbol, and against the layers summer4's own architecture implies.
- {doc}`docs-coverage` takes the two documentation corpora the project wants to
  reproduce — the summer2 readthedocs site and the summer textbook — and states
  page by page whether they can be published as runnable material.
- {doc}`user-satisfaction` is a heuristic evaluation: task walkthroughs for three
  user types, and an ergonomics review of the API that does exist. There is no
  user research behind it and it does not pretend otherwise.
- {doc}`age-stratified-seirs-case-study` takes one concrete modelling problem
  end to end and reports what it cost, requirement by requirement, against the
  ledger. It is the only page here whose claims come from building the thing
  rather than from reading the API.
- {doc}`tb-ports` asks whether two existing tuberculosis models (Kiribati on
  summer2gen, tb_macro on summer3wip) can be built in summer4, row by row, and
  which work package closes each gap. It is machine-checked like the ledger.
- {doc}`gaps` collects the blockers in priority order.
