# Documentation coverage

The brief was to reproduce most of the summer2 readthedocs site and the summer
textbook as runnable documentation. This page states, page by page, how much of
that is achievable against the current API. Counts come from
{doc}`coverage-ledger`.

## summer2 readthedocs

The site has four notebook-bearing sections plus prose and an API reference.

| Page | Status | Ported | Blocker |
|---|---|---|---|
| `install.ipynb` | **Replaced** | — | Superseded by {doc}`../index` (Install) |
| `examples/01-basic-model` | `full` | `guide/01-basic-model.ipynb` | None |
| `examples/03-derived-outputs` | `full` | `guide/04-derived-outputs.ipynb` | None |
| `examples/04-flow-types` | `full` | `guide/02-flow-types.ipynb` | None |
| `examples/06-stratification-introduction` | `full` | `guide/05-stratification.ipynb` | None |
| `examples/07-age-stratification` | `full` | `guide/06-mixing.ipynb` | None |
| `examples/08-strain-stratification` | `partial` | — | `per_trait` FOI works and is tested (`tests/test_epi_multistrain.py`) but no built page documents it (the port is archived unbuilt in `guide/_sources/`); no `StrainStratification` class |
| `examples/09-mixing-matrices` | `full` | `guide/06-mixing.ipynb` | None |
| `examples/10-derived-outputs-stratified` | `full` | `guide/05-stratification.ipynb` | None |
| `examples/11-flows-between-strata` | `full` | `guide/05-stratification.ipynb` | None |
| `detailed/time-varying-functions` | `full` | `guide/03-rates.ipynb` | None |
| `detailed/InitialPopulationGraphobject` | `full` | `guide/07-initial-population.ipynb` | None |
| `rationale.md` | **Portable** | — | Prose; adapted into {doc}`../dev/architecture` |
| `api/*.rst` | **N/A** | — | summer4 has its own; see {doc}`../api/index` |
| `dev-setup.md` | **Replaced** | — | Superseded by {doc}`../dev/pixi` |

**10 of 11 notebooks reproduce in full, and the strain page partially (it is tested but no built page documents it).** Structure, flows, infection / mixing,
trajectories, results and initial population are no longer the blocker for this
corpus. {doc}`../guide/index` is the summer4-native
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
`CompiledModel.run` returns a `Result`; `Output.to_frame` / `to_pandas` and the
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
| Guide | 8 (`docs/guide/01` to `08`) | Yes |
| Developer guide | (Markdown + performance notebook) | Yes |
| Textbook | Ported chapters under `docs/textbook/` | Yes |
| Evaluation | 6 | n/a — prose |
| API reference | generated from `__all__` | n/a |

`nb_execution_raise_on_error = True`, so every assertion on this site is a test.
