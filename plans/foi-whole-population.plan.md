<!--
Accepted 2026-10-29 (fast-tracked for the summer4template release). Historical record once
merged: do not edit. Follow-up work goes in a new plan.
-->

# Whole-population force of infection: `ForceOfInfection(group_by=None)`

Closes the dummy-population half of `CX15`
(`futureplans/foi-unstratified-dummy-pop.md`). The multi-property half
(`futureplans/foi-multi-property-mixing.md`) stays deferred.

## Context

`ForceOfInfection` always groups by one `Property`. For homogeneous mixing over
the whole population — summer2's default, where an unadjusted model has one
mixing category and a `[[1.0]]` matrix — a summer4 user has to add a singleton
property (`Property("pop", ("all",))`), stratify every compartment by it, and
group the FOI by it. Every compartment label then carries `pop=all`.

Grouping by a real property without a mixing matrix is *not* a substitute: it
makes each group mix only with itself (`I_g / N_g`). It matches homogeneous
mixing only while every group has the same prevalence, which is how the
summer4template port first went wrong.

## Shape

`group_by` becomes optional and defaults to `None`, meaning one group that is
the whole population:

```python
ForceOfInfection("foi", infectious=state["I"], contact_rate=Param("beta"))
# frequency: beta * I_total / N_total, on any map, however stratified
```

- **Rate.** The node evaluates to a scalar, so it broadcasts to every edge of
  the flow (a scalar rate already does). `kind`, `denominator`, `exponent`,
  `contact_rate`, `infectiousness` pairs and `normalize_infectiousness` keep
  their meaning with one group.
- **Capture.** The FOI is still captured under its name. Its capture has no
  properties, so the saved output has dims `("time",)` and is a plain series.
- **Custom `kind`.** The callable receives `GroupedRate` values with
  `properties=()` and must return one.
- **Rejected combinations.** `mixing=` with `group_by=None` raises (there is
  nothing to mix between). A trait-keyed `infectiousness` mapping raises with
  `group_by=None`; selector pairs work.
- **Digest.** The node's bytes mark the whole-population case explicitly, so
  it never collides with a property name.

`GroupedOutput` evaluation and `values_for` accept a capture with no
properties; `CompiledModel` gives it dims `("time",)`.

## Files

- `src/summer4/epi/infection.py` — optional `group_by`, evaluation, validation.
- `src/summer4/results/eval.py`, `src/summer4/flows/compiled.py` — captures
  with no properties.
- `tests/test_foi_whole_population.py` — new.
- `examples/notebooks/30-foi-whole-population.ipynb` — acceptance notebook.
- `docs/evaluation/composability.md` — narrow `CX15` to the multi-property case.
- `futureplans/foi-unstratified-dummy-pop.md` — removed (done);
  `futureplans/foi-dummy-pop-docs-migration.md` — the published pages that still
  use the dummy property.
- `CHANGELOG.md` — Unreleased entry.

## Tests

- Frequency FOI with `group_by=None` equals `contact_rate * I_total / N_total`
  on an age-stratified map with unequal prevalences, for every age.
- It equals the dummy-population model's trajectory exactly.
- It differs from `group_by=age` without a mixing matrix when prevalences differ.
- Density, generalised, denominator, infectiousness pairs and both
  normalisations agree with hand computations.
- A custom `kind` callable receives `properties=()`.
- The default save has the FOI as a `("time",)` output equal to the hand value.
- `mixing=` and a trait-keyed `infectiousness` mapping raise with `group_by=None`.
- Digests differ between `group_by=None` and a property, and are stable.
- Works under `jax.jit` and with `BayesianModel`.

## Acceptance notebook

`30-foi-whole-population.ipynb`: an age-stratified SIR seeded only in children.

1. Whole-population FOI vs the dummy-property idiom: identical curves.
2. Whole-population FOI vs per-age grouping with no matrix: adults are infected
   under homogeneous mixing and never under per-age grouping.
3. The saved FOI series against `contact_rate * I_total / N_total` from the
   compartments.
