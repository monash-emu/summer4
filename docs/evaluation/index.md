# Project evaluation

An assessment of summer4's progress against two questions: **how complete is the
feature set**, and **how well is it likely to serve its users**.

```{toctree}
:maxdepth: 2

coverage-ledger
feature-completeness
if-promoted
docs-coverage
user-satisfaction
gaps
```

## Headline

| Measure | Result |
|---|---|
| Public API symbols | 13 |
| Package source | 570 lines, NumPy only |
| summer2 API surface exercised by its own docs, complete in summer4 | **6 of 52 (12%)** |
| … with any working route (complete or partial) | 7 of 52 (13%) |
| summer2 documentation notebooks reproducible as runnable summer4 | **0 of 11** (3 partially, structure only) |
| summer textbook chapters reproducible as runnable summer4 | **1 of 20** (3 more partially) |
| Implemented layers of the intended stack | 1 of ~7 |
| Same measure **if the flows spike were promoted** | **21 of 52 (40%)** complete, 31 of 52 (60%) covered |
| Ceiling of the planned roadmap (WP1–WP10) | 46 of 52 (88%) — see {doc}`coverage-ledger` |

summer4 is one well-built layer of a platform. The compartment taxonomy is
complete, tested, benchmarked and — as of this site — documented. Nothing above
it exists in the public API, so no end-to-end modelling task can be completed
with the library today.

## How to read this section

- {doc}`coverage-ledger` is the **authoritative, machine-checked record**:
  every summer2 symbol, every textbook chapter, and the ordered work packages
  that lead to full coverage. Every number quoted anywhere else on this site
  comes from it. Read this one first.
- {doc}`feature-completeness` compares the public API against summer2's,
  symbol by symbol, and against the layers summer4's own architecture implies.
- {doc}`if-promoted` re-runs that comparison on the assumption that the
  `explorations/flows/` spike becomes part of the package.
- {doc}`docs-coverage` takes the two documentation corpora the project wants to
  reproduce — the summer2 readthedocs site and the summer textbook — and states
  page by page whether they can be published as runnable material.
- {doc}`user-satisfaction` is a heuristic evaluation: task walkthroughs for three
  user types, and an ergonomics review of the API that does exist. There is no
  user research behind it and it does not pretend otherwise.
- {doc}`gaps` collects the blockers in priority order.
