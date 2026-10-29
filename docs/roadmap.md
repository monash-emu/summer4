# Roadmap

This page says what summer4 does today, which older ways of writing a model are still
available but on their way out, and what is coming next. The detailed, step-by-step plan
is kept in the [project roadmap on GitHub](https://github.com/monash-emu/summer4/blob/main/docs/dev/roadmap.md).

## What summer4 does today

The guide builds up one model at a time:

- {doc}`guide/01-basic-model` builds a small SIR model with compartments and flows, runs it and plots the result.
- {doc}`guide/02-flow-types` covers transition, entry and exit flows, and how parameters set their rates.
- {doc}`guide/03-rates` shows the kinds of rate you can write: parameters, time, arithmetic, mathematical functions, your own functions, and tables of values.
- {doc}`guide/04-derived-outputs` derives new quantities from a run, such as totals, counts over time, integrals and cumulative sums.
- {doc}`guide/05-stratification` splits compartments into groups such as age or risk bands, with flows between the groups.
- {doc}`guide/06-mixing` adds ageing and mixing matrices, so that infection depends on who mixes with whom.
- {doc}`guide/07-initial-population` sets how many people start in each compartment, including shares that depend on parameters.
- {doc}`guide/08-calibration` fits parameters to observed data with a Bayesian model and a best-fitting (MAP) estimate.

The textbook chapters show the same ideas applied to fuller models.

## Still here, on the way out

These older ways of writing a model still work, and nothing in the guide uses them. Each one
has a newer replacement, which is being built in the steps shown.

| Older way | What it is for | Replaced in | Newer way (will be) |
| --- | --- | --- | --- |
| `compile(derived_fn=...)` and `derived_refs` | A quantity that depends on the state of the model, and is shared between flows | Step 39, removed in step 41 | Fields written with `field_refs` and `defer`, which will read the state directly |
| `SaveFn` and `ComputedValue` | Saving a value worked out at each step of a run | Step 40 | Assigning any expression to an entry in an output set, which will be saved like any other output |
| `ForceOfInfection(kind=...)` with a plain Python function | A custom force of infection | Step 42 | `custom_kind(fn, Param(...))`, which will pass your parameters to the function |

When a step replaces one of these, it removes the old way at the same time, and this table
loses that row.

## Coming next

- **Phase M (steps 36 to 44)** will give rates and outputs one expression type, so any quantity can be combined and saved in the same way. It will also let you save any expression as an output, and check the stages of a run when the model is built, so that mistakes are caught before the model runs.
- **Step 28** will draw plots of calibrated outputs: many fitted runs as overlaid lines, and quantile ribbons with the observed targets on top.
- **Track G (steps 16 to 19)** will add susceptibility to the force of infection, contact-survey matrices, and textbook chapters 15 to 19.

For the full plan, see the [project roadmap on GitHub](https://github.com/monash-emu/summer4/blob/main/docs/dev/roadmap.md).
