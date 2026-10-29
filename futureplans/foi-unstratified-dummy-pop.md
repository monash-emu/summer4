# ForceOfInfection unstratified FOI needs a dummy population property

**Resolved** on `feat/foi-whole-population` (`plans/foi-whole-population.plan.md`):
`ForceOfInfection(group_by=None)` — now the default — is one whole-population
group, `contact_rate * I_total / N_total` for the frequency kind. Acceptance
notebook: `examples/notebooks/30-foi-whole-population.ipynb`.

Published pages written before the fix still use the dummy property; migrating
them is `futureplans/foi-dummy-pop-docs-migration.md`.

## Original concern

`ForceOfInfection` always grouped by a property and expected a `MixingMatrix` on
that property. For an unstratified SIR (textbook chapter 9,
`examples/notebooks/09-epi-models.ipynb`) the user had to introduce a singleton
`Property("pop", ("all",))`, stratify onto it, and attach a `[[1.0]]` mixing
matrix with `check_reciprocal=False`. Summer2's `add_infection_frequency_flow`
did not require this ceremony.
