---
name: contact-survey-data
overview: Load, validate and inspect empirical contact-survey matrices — WP9 / roadmap step 17.
todos:
  - id: api
    content: ContactMatrix (from_array, from_frame, to_frame, to_pandas, total, mean_contacts, reciprocity_error), SettingStack, summer4.data.frame_columns
    status: completed
  - id: tests
    content: tests/test_contact_matrices.py on the asymmetric fixture in tests/helpers/contacts.py
    status: completed
  - id: notebook
    content: examples/notebooks/26-contact-matrices.ipynb, loading and inspection sections (step 18 extends it)
    status: completed
isProject: false
---

# Contact survey data (WP9 / step 17)

Follows `plans/wp9-contact-surveys.plan.md` *Step 17*. Closes no ledger row on
its own; textbook rows 16–19 move in step 19.

## The data question — answered

The user chose **option 1** on 2026-09-29: **ship no data, embed the extract.**
summer4 ships loaders plus a small synthetic fixture under `tests/helpers/`
(`tests/helpers/contacts.py`). Notebooks embed the small public extract they
need inline — hard-coded, with its source and licence stated — and **nothing is
fetched over the network at docs build time.** The extract used is the POLYMOD
Great Britain all-settings matrix (Mossong et al. 2008, *PLoS Medicine* 5(3):
e74, CC BY), as transcribed in the summer textbook (BSD-2-Clause) chapter 17,
with that textbook's UK 2006 population.

## What ships

`src/summer4/epi/contacts.py`:

- `ContactMatrix` — frozen; `values[i, j]` = contacts a member of band `i`
  (participant) has with band `j` (contact). `bands` has `K + 1` strictly
  increasing edges, only the last may be `inf`. `prop` aligns by position;
  numeric trait names must equal the lower edges. `source_population` is the
  survey population and the default for every method that needs one.
- Constructors `from_array`, `from_frame` (pandas or polars long frame; labels
  `"[0,5)"`, `"0-4"`, `"75+"` or numeric lower bounds), `from_settings`.
- One `_validate`; every failure names the offending value (plan 17c table).
- Inspection: `total()`, `mean_contacts(population)`,
  `reciprocity_error(population) -> Reciprocity(matrix, relative)`.
- Export: `to_frame()` (long, round-trips through `from_frame`) and
  `to_pandas()` (square, participant × contact).
- `SettingStack` — read-only mapping of setting name to `ContactMatrix` sharing
  bands; `total()`, `map(fn)`, `from_frame(..., setting="setting")`.

`summer4.data.frame_columns` is the generic "read these columns of a pandas or
polars frame" helper `contacts.py` calls, per the plan's separation rule.

## Deviations from `wp9-contact-surveys` *Step 17*

- **No `ContactMatrix.plot`.** No plotting seam exists yet (`CX9`,
  `futureplans/trace-plot-backend-coupling.md`). `to_pandas()` returns the
  square labelled frame instead, and the notebook plots it with
  `frame.plot(kind="imshow")` under the pandas Plotly backend. The futureplans
  note records this.
- **Notebook number** is `26`, not `15` (`15` is taken twice; `24` and `25` are
  claimed by roadmap steps 30 and 28). Step 18 extends the same notebook.
- **Gate style.** The notebook follows the current sign-off convention
  (`23-solve-backends.ipynb`): each claim has a figure or displayed table and a
  *What to check* box; the assertions live in `tests/test_contact_matrices.py`.
- Added `SettingStack.from_frame` and `SettingStack.map` (composability: one
  transformation, applied per setting, rather than per-method loops in step 18).
