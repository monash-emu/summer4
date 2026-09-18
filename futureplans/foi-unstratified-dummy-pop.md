# ForceOfInfection unstratified FOI needs a dummy population property

## Concern

`ForceOfInfection` always groups by a property and expects a `MixingMatrix` on
that property. For an unstratified SIR (textbook chapter 9,
`examples/notebooks/09-epi-models.ipynb`) the user must introduce a singleton
`Property("pop", ("all",))`, stratify onto it, and attach a `[[1.0]]` mixing
matrix with `check_reciprocal=False`. Summer2's `add_infection_frequency_flow`
did not require this ceremony.

Noticed while porting chapter 9 in the textbook catch-up sweep
(`plans/textbook-catchup.plan.md`).

## Pointers

- `src/summer4/epi/infection.py` — `ForceOfInfection` requires `group_by`
- `docs/textbook/09-transmission-assumptions.ipynb` — the dummy-`pop` pattern
  in a published page

## Done looks like

Either:

1. a documented convenience on `ForceOfInfection` for the whole-population case
   (implicit unit group + identity mixing), or
2. an explicit decision that the dummy property is the supported idiom, with
   a short user-guide note so ports stop rediscovering it.
