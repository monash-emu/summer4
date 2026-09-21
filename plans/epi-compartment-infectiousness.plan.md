---
name: epi-compartment-infectiousness
overview: Selector-keyed infectiousness weights on ForceOfInfection, applied per compartment before the group sum.
todos:
  - id: weights
    content: Coerce trait maps and selector pairs; multiply before the group sum; keep normalisation on the effective weight
    status: completed
  - id: tests
    content: Digest equality, compartment x age zeros, reachability, Lookup mixing, gradient
    status: completed
  - id: notebook
    content: Extend 09-epi-models with compartment weights and the four-source reinfection model
    status: completed
isProject: false
---

# Compartment-level infectiousness

Follows `plans/tb-ports-feature-completeness.plan.md` §14b, §14d and §14e.
Ignore §14c (`EpiModel` was removed). `adjust=` already exists on `TransitionFlow`
as a sequence of adjustments.

## API

`infectiousness` accepts either shape. Both store the same sorted
`(selector, rate)` pairs (`coerce_compartment_weights`), so a trait map and the
pairs it expands to digest equally.

```python
ForceOfInfection(
    ...,
    infectiousness=[(state["sub"], Param("rel_sub")), (age["0"], 0.0)],
)
# sugar, same digest as [(age["young"], 0.5), (age["old"], 2.0)]:
ForceOfInfection(..., infectiousness={age["young"]: 0.5, age["old"]: 2.0})
```

Pairs multiply matching compartments before the infectious group sum. A
compartment no pair matches keeps weight 1. A selector that matches nothing
raises. The pair list is unrolled; keep it small.

`normalize_infectiousness` still defaults to `None`. `population` divides the
weighted pool by the population-weighted mean of the compartment weights.
`mean` divides by the unweighted mean of the per-group effective weights (the
unweighted mean of compartment weights in the group; an empty group counts as
1). When every compartment in a group shares one weight, both match the
previous per-trait formulas.

## Reuse for susceptibility (step 16)

`coerce_compartment_weights` and `apply_compartment_weights` are the shared
machinery. Susceptibility multiplies the recipient after mixing and is not
normalised, so it must not call `scale_infectious_pool`. A selector that is
not a trait of `group_by` does not fit in the `GroupedRate`; apply the weights
to the compartment-aligned rate.
