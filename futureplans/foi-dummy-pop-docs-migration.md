# Published pages still use the dummy `pop` property for homogeneous mixing

## Concern

`ForceOfInfection(group_by=None)` (`plans/foi-whole-population.plan.md`) made the
singleton `Property("pop", ("all",))` + `[[1.0]]` matrix idiom unnecessary, but
the pages written before it still teach it. Readers copy the longer idiom, and
every compartment label in those pages carries `pop=all`.

Not migrated on `feat/foi-whole-population` to keep that branch small and its
docs build unchanged.

## Pointers

`grep -lE 'Property\(\\?"pop\\?"' docs examples` (excluding `docs/_build`), at the
time of writing:

- `examples/notebooks/09-epi-models.ipynb`
- `docs/cookbook/01-custom-rates.ipynb`
- `docs/textbook/02, 05, 06, 08, 09, 10, 11, 13, 14` (chapter notebooks)
- `docs/summer2/01, 03, 04, 06, 08, 09, 10` (summer2 doc ports)

Pages that need a real grouping property for something else (a mixing matrix,
per-age infectiousness) keep it; only the whole-population uses change.

## Done looks like

Each whole-population use is `ForceOfInfection(..., infectious=...)` with no
`group_by`, no `pop` stratification and no `[[1.0]]` matrix; outputs are
unchanged (the trajectories are identical, see
`tests/test_foi_whole_population.py::test_matches_dummy_population_idiom_trajectory`);
`pixi run -e docs docs-strict` passes.
