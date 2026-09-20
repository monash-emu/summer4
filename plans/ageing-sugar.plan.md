---
name: ageing-sugar
overview: Build TraitChain.from_breakpoints so uneven age bands age at 1/width without hand-written pairs.
todos:
  - id: from-breakpoints
    content: Parse numeric lower-bound traits; consecutive pairs; rates 1/(width*unit); reject bad input
    status: completed
  - id: digest
    content: Acceptance test — sugar compiles to the same digest as the hand-written chain
    status: completed
  - id: notebook-ledger
    content: Finish 13-rate-math-and-tables.ipynb ageing section; move KI2 and TM3 to full
    status: completed
isProject: false
---

# Ageing sugar

Follows `plans/tb-ports-feature-completeness.plan.md` §13.3. The roadmap step is
authoritative where they differ: the notebook is
`examples/notebooks/13-rate-math-and-tables.ipynb`, and `S6`
(`AgeStratification`) stays `partial` — a bundled stratification class remains a
rejected summer2 shape.

## API

```python
TraitChain.from_breakpoints(age: Property, *, unit: float = 1.0) -> TraitChain
```

Traits must parse as numeric lower bounds (`"0"`, `"5"`, `"15"`). Consecutive
traits form pairs; rates are `1 / (width * unit)`. The last band has no outgoing
edge. Raise if a trait does not parse, bounds are not strictly increasing, there
are fewer than two traits, or `unit` is not positive.

## Acceptance

The sugar is equal to the equivalent hand-written `TraitChain`, and a model that
uses either compiles to the same digest. Port rows `KI2` and `TM3` move to
`full`; no API-ledger row moves.
