# Documentation coverage

The brief was to reproduce most of the summer2 readthedocs site and the summer
textbook as runnable documentation. This page states, page by page, how much of
that is achievable against the current API. Counts come from
{doc}`coverage-ledger`.

## summer2 readthedocs

The site has four notebook-bearing sections plus prose and an API reference.

| Page | Status | Ported | Blocker |
|---|---|---|---|
| `install.ipynb` | **Replaced** | — | Superseded by {doc}`../getting-started/installation` |
| `examples/01-basic-model` | `full` | `summer2/01-basic-model.ipynb` | None |
| `examples/03-derived-outputs` | `full` | `summer2/03-derived-outputs.ipynb` | None |
| `examples/04-flow-types` | `full` | `summer2/04-flow-types.ipynb` | None |
| `examples/06-stratification-introduction` | `full` | `summer2/06-stratification-introduction.ipynb` | None |
| `examples/07-age-stratification` | `full` | `summer2/07-age-stratification.ipynb` | None |
| `examples/08-strain-stratification` | `full` | `summer2/08-strain-stratification.ipynb` | None (`per_trait` FOI; no `StrainStratification` class) |
| `examples/09-mixing-matrices` | `full` | `summer2/09-mixing-matrices.ipynb` | None |
| `examples/10-derived-outputs-stratified` | `full` | `summer2/10-derived-outputs-stratified.ipynb` | None |
| `examples/11-flows-between-strata` | `full` | `summer2/11-flows-between-strata.ipynb` | None |
| `detailed/time-varying-functions` | `full` | `summer2/time-varying-functions.ipynb` | None |
| `detailed/InitialPopulationGraphobject` | `full` | `summer2/initial-population-graphobject.ipynb` | None |
| `rationale.md` | **Portable** | — | Prose; adapted into {doc}`../dev/architecture` |
| `api/*.rst` | **N/A** | — | summer4 has its own; see {doc}`../api/index` |
| `dev-setup.md` | **Replaced** | — | Superseded by {doc}`../dev/pixi` |

**11 of 11 notebooks reproduce in full.** Structure, flows, infection / mixing,
trajectories, results and initial population are no longer the blocker for this
corpus. {doc}`../user/08-flows` is the summer4-native
walkthrough.

## The summer textbook

Twenty chapters. Full detail in {doc}`../textbook/roadmap` and the ledger; the
summary:

| Outcome | Chapters | Count |
|---|---|---|
| Ported in full | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 | 14 |
| Ported, modelling content partial | 15 (no FOI susceptibility surface) | 1 |
| Blocked on contact-survey data (WP9) | 16, 17, 18, 19 | 4 |
| Blocked on calibration (WP10) | 20 | 1 |

Ported paths live under `docs/textbook/` (see the ledger `Ported` column).
Chapter 10 notes unstratified $R_t$ without a next-generation-matrix helper.

## Non-API blockers

### 1. Results object — closed

Both corpora call `model.get_outputs_df()` and plot the result.
`CompiledModel.run` returns a `Result`; `Trace.to_frame` / `to_pandas` and the
Plotly-via-pandas convention cover the publishable-run path. See
{doc}`../dev/plotting`.

### 2. Source figures are partially vendored

Textbook chapters under `docs/textbook/figures/` already include SVGs for
ported chapters 3–6 and 8 (BSD-2-Clause). Remaining chapters that embed
diagrams (especially 12–19 when unblocked) still need copies or redraws.

### 3. No contact-survey data layer

Chapters 16–19 load empirical contact matrices, inspect them and scale them
against population data. WP9; depends on WP6 (applied).

### 4. Documentation infrastructure — closed

`docs/` with Sphinx, myst-nb, `.readthedocs.yaml`, and
`pixi run -e docs docs` is in place.

## What this site does instead

Rather than publishing stubs for material that cannot run, the site documents
the implemented layer completely, ports what the API supports, and states the
remaining gaps explicitly:

| Section | Pages | All code executed at build time |
|---|---|---|
| Getting started | 2 | Yes |
| User guide | 9 (8 notebooks + migration guide) | Yes |
| Developer guide | (Markdown + performance notebook) | Yes |
| Textbook | Ported chapters under `docs/textbook/` | Yes |
| summer2 ports | Ported pages under `docs/summer2/` | Yes |
| Evaluation | 6 | n/a — prose |
| API reference | generated from `__all__` | n/a |

`nb_execution_raise_on_error = True`, so every assertion on this site is a test.
