---
name: foi-susceptibility
overview: ForceOfInfection(susceptibility=...) on the recipient side, after mixing, never normalised; textbook 15 rewritten onto it.
todos:
  - id: api
    content: susceptibility= keyword, coerced with coerce_compartment_weights; per-group pairs fold into the GroupedRate, other pairs make the rate compartment-aligned
    status: completed
  - id: tests
    content: tests/test_epi_susceptibility.py — hand formula with asymmetric K, Multiply and row-scaled equivalence, not-normalised, normalised infectiousness untouched, grad, jit, jaxpr size, validation
    status: completed
  - id: chapter
    content: Rewrite docs/textbook/15 onto susceptibility=; ledger row 15 to full
    status: completed
  - id: notebook
    content: Extend examples/notebooks/09-epi-models with a susceptibility section
    status: completed
isProject: false
---

# FOI susceptibility surface (roadmap step 16)

Follows `plans/wp17-foi-susceptibility.plan.md` (WP17). This file records the
decisions that plan left open, and the one place the plan could not be met.

## API

```python
ForceOfInfection(..., susceptibility={age["child"]: 0.5})           # trait map
ForceOfInfection(..., susceptibility=[(state["R"], 0.3), ...])      # selector pairs
```

Both shapes are coerced by `coerce_compartment_weights(..., what="susceptibility")`,
so validation and digests match infectiousness. `__field_paths__` and
`__rate_bytes__` recurse into the weights (`b"sus"` tag in the bytes).

## Where a weight is applied

`split_susceptibility(pairs, group_by=)` (public in `summer4.epi`) divides the
pairs:

- **Per group** — the selector is a single trait of `group_by` (everything a
  trait map produces). The weight multiplies λ_a in the `GroupedRate`, so the
  rate stays per group, and the value captured under the FOI's name includes it.
  This is exactly a row scale of the mixing matrix.
- **Per compartment** — any other selector. The per-group λ is gathered onto
  compartments with one constant-index gather, and each pair is one masked
  multiply (`apply_compartment_weights`). The rate is then compartment-aligned.
  The capture stays per group and excludes these weights (it is the force on a
  member of per-compartment susceptibility 1), because `GroupedOutput` is
  single-property and a within-group weight has no per-group value.
  `ForceOfInfection.per_compartment` reports this mode; `captured()` refuses it.

Compartments lacking `group_by` receive NaN in per-compartment mode; an infection
flow leaving them is already an error on the per-group path.

No `normalize_susceptibility=` (WP17 Part A).

## Digest equivalence (WP17 Part B) — not achievable

A digest hashes the rate tree and the adjustment list structurally, so
`susceptibility={age[a]: s}` and `Multiply(s, where=age[a])` on the flow can never
share one. The tests assert identical flows, `dy` and trajectories instead, and
assert the digests differ with the reason in the docstring.
