# Documentation coverage

The brief was to reproduce most of the summer2 readthedocs site and the summer
textbook as runnable documentation. This page states, page by page, how much of
that is achievable against the current API.

## summer2 readthedocs

The site has four notebook-bearing sections plus prose and an API reference.

| Page | Content | Reproducible? | Blocker |
|---|---|---|---|
| `install.ipynb` | Installation | **Replaced** | Superseded by {doc}`../getting-started/installation` |
| `examples/01-basic-model` | SIR model, run, plot | **Partial** | Trajectory, initial population, results frame |
| `examples/03-derived-outputs` | Five `request_*` output types | **No** | Derived outputs |
| `examples/04-flow-types` | All eight flow constructors | **Partial** | Trajectory, flow outputs |
| `examples/06-stratification-introduction` | Stratify, adjust flows, infectiousness | **Partial** | Infectiousness adjustments, trajectory |
| `examples/07-age-stratification` | Age strata, population split, ageing | **Partial** | Population split, trajectory |
| `examples/08-strain-stratification` | Strain strata, strain-specific flows | **Partial** | Strain-aware FOI primitive, trajectory |
| `examples/09-mixing-matrices` | Contact matrices | **No** | Mixing matrices |
| `examples/10-derived-outputs-stratified` | Stratified outputs | **No** | Derived outputs |
| `examples/11-flows-between-strata` | Inter-stratum flows | **Partial** | Trajectory, compartment outputs |
| `detailed/time-varying-functions` | Interpolation, piecewise, `Time` | **Partial** | Interpolation and piecewise helpers |
| `detailed/InitialPopulationGraphobject` | Parameterised initial population | **No** | Initial population, parameters |
| `rationale.md` | Why summer exists | **Portable** | Prose; adapted into {doc}`../dev/architecture` |
| `api/*.rst` | Autodoc for six modules | **N/A** | summer4 has its own; see {doc}`../api/index` |
| `dev-setup.md` | Development setup | **Replaced** | Superseded by {doc}`../dev/pixi` |

**0 of 11 notebooks reproduce in full. 7 reproduce in part.** Structure,
flows and a final-state Euler exist; every page that plots a run is still
blocked on a results object. {doc}`../user/08-flows` is the summer4-native
walkthrough.

## The summer textbook

Twenty chapters. Full detail in {doc}`../textbook/roadmap`; the summary:

| Outcome | Chapters | Count |
|---|---|---|
| Ported in full | 1 | 1 |
| Modelling content expressible, trajectory missing | 2, 3, 4, 5, 6, 7, 9, 10, 11 | 9 |
| Blocked on derived outputs / results | 8 | 1 |
| Blocked additionally on mixing matrices | 12, 13, 14, 15 | 4 |
| Blocked additionally on contact-survey data | 16, 17, 18, 19 | 4 |
| Blocked additionally on calibration | 20 | 1 |

The structural half of chapters 2, 5 and 6 is consolidated into one page here,
{doc}`../textbook/02-model-structures`, rather than published as three stubs.

## Non-API blockers

Even with flows in the package, four things still stand between the project
and a published port of those corpora.

### 1. No results object or dataframe convention

Both corpora call `model.get_outputs_df()` and plot the result. summer4 has no
results type, no dataframe conversion, and no plotting convention. The
`dev` feature installs `polars`, `plotly` and `matplotlib`, but nothing in
`src/summer4` produces anything for them to consume.

### 2. Source figures are not vendored

Textbook chapters 2–6 and 12–19 embed SVG diagrams from the source repository
(`sir_structure.svg`, `seir_transition.svg`, `immunity_structures.svg`,
`incubation_terminology.svg`, `source_dest_structure.svg` and others). A port
needs them copied in under the BSD-2-Clause notice, or redrawn.

### 3. No contact-survey data layer

Chapters 16–19 load empirical contact matrices, inspect them and scale them
against population data. summer2 leans on external data utilities for this;
summer4 has no data-loading surface at all.

### 4. No documentation infrastructure existed

Before this change the repository had no `docs/` directory, no Sphinx
configuration, no `.readthedocs.yaml`, and no documentation task in
`pixi.toml` — while summer2 publishes to readthedocs. That infrastructure is
now in place (`pixi run -e docs docs`), so this blocker is closed.

## What this site does instead

Rather than publishing stubs for material that cannot run, the site documents
the implemented layer completely and states the gap explicitly:

| Section | Pages | All code executed at build time |
|---|---|---|
| Getting started | 2 | Yes |
| User guide | 9 (8 notebooks + migration guide) | Yes |
| Developer guide | (Markdown + performance notebook) | Yes |
| Textbook | 3 (1 notebook) | Yes |
| Evaluation | 5 | n/a — prose |
| API reference | generated from `__all__` | n/a |

`nb_execution_raise_on_error = True`, so every assertion on this site is a test.
