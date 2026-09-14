# Project evaluation

An assessment of summer4's progress against two questions: **how complete is the
feature set**, and **how well is it likely to serve its users**.

```{toctree}
:maxdepth: 2

coverage-ledger
feature-completeness
docs-coverage
user-satisfaction
gaps
```

## Headline

| Measure | Result |
|---|---|
| Public API symbols | 40 |
| Package source | taxonomy (NumPy) plus `summer4.flows` and `summer4.jax` |
| summer2 API surface exercised by its own docs, complete in summer4 | **22 of 52 (42%)** |
| … with any working route (complete or partial) | 31 of 52 (60%) |
| summer2 documentation notebooks reproducible as runnable summer4 | **0 of 11** (7 partially) |
| summer textbook chapters reproducible as runnable summer4 | **1 of 20** (9 more partially) |
| Implemented layers of the intended stack | taxonomy + flows; no results object |
| Ceiling of the planned roadmap (WP2–WP10) | 47 of 52 (90%) — see {doc}`coverage-ledger` |

summer4 ships a compartment taxonomy and a flows layer: declare a map, attach
named flows, compile to a `CompiledModel`, evaluate a JAX vector field, and
take a fixed-step Euler to a **final state**. There is no trajectory, results
object or derived-output request API, so no summer2 notebook or textbook
chapter that ends by plotting a run is publishable yet.

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
- {doc}`gaps` collects the blockers in priority order.
