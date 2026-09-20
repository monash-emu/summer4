---
name: epi-generalised-foi
overview: Add ForceOfInfection kind="generalised" with a calibratable population exponent.
todos:
  - id: api
    content: exponent on ForceOfInfection; kind generalised; reject bad kind/exponent pairs
    status: completed
  - id: eval
    content: shedding = i_grp / n_grp ** exponent via GroupedRate pow; digest includes exponent
    status: completed
  - id: tests
    content: frequency≡exp1 and density≡exp0 bit-identical; raise paths; gradient w.r.t. exponent
    status: completed
isProject: false
---

# Generalised force of infection

Follows `plans/tb-ports-feature-completeness.plan.md` §14a. Ignore §14c (`EpiModel`
was removed). Compartment-level infectiousness (§14b) and the TB-shaped notebook
section (§14e) are roadmap step 8.

## API

```python
ForceOfInfection(
    ...,
    kind="generalised",
    exponent=Param("infection_pop_scale"),  # RateOps | float
)
```

Raise if `kind="generalised"` without `exponent`, or `exponent` with any other
kind (including a custom callable). Frequency dependence must stay bit-identical
to generalised with `exponent=1.0`; density to `exponent=0.0`.
