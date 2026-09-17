# Next-generation matrix / spectral $R_0$ helper

## Concern

Textbook chapter 10 and mixed-population modelling need $R_0$ as the spectral
radius of a next-generation matrix. summer4 can express unstratified
$R_0 = \beta/\gamma$ and $R_t = R_0\,S/N$ (ported at
`docs/textbook/10-reproduction-number.ipynb`), but has no public helper that
builds or eigendecomposes an NGM from a `ForceOfInfection` + mixing matrix.

Noticed during the textbook catch-up sweep (`plans/textbook-catchup.plan.md`).

## Pointers

- `docs/textbook/10-reproduction-number.ipynb` — unstratified path + admonition
- `src/summer4/epi/infection.py` — `ForceOfInfection` / mixing already encode
  the contact structure an NGM would use

## Done looks like

A documented, tested function (or `EpiModel` method) that returns $R_0$ (and
optionally per-group contributions) from the compiled infection structure for
at least the frequency-dependent SIR/SEIR case with a mixing matrix — without
requiring the user to assemble the NGM by hand in NumPy.
