<!--
Accepted 2026-10-30. Commit as plans/clean-surface.plan.md on the
step's branch (step C0). After it merges it is historical record: do not edit it. Corrections go
in the roadmap's "Corrections to committed plans" table.
-->

# Clean surface: one idiom, minimal API, docs rebuilt and published

Branch: `feat/clean-surface`, cut from `main` once all three gating PRs have merged.
Roadmap: a new **step 48** (Phase N), placed in the landing line after the
gating PRs and **before step 28**.

## Gate: do not start until these are merged into `main`

| PR | Branch | Why the clean branch needs it |
|---|---|---|
| monash-emu/summer4#69 | `feat/float64-default` | Docs execute under the precision users will get |
| monash-emu/summer4#70 | `feat/cumulative-integral` | Adds `cumulative_integral` / `cumsum` and deprecates `cumulative()`, which this branch then removes |
| monash-emu/summer4#71 | `feat/time-free-outputs` | Adds lazy `at`, `array`, `scalar`, `with_values`, which the new derived-outputs page teaches |

**Gate met 2026-10-30:** all three are merged; `main` is at `516d05f`. #69's notebook became
`35-float64-default.ipynb` and its composability finding is CX41.

```bash
gh pr view 69 --json state -q .state   # MERGED
gh pr view 70 --json state -q .state   # MERGED
gh pr view 71 --json state -q .state   # MERGED
```

## Context and decisions (user, 2026-10-30)

summer4 has accumulated several spellings for the same thing. Phase L (steps 31–35) added the
unified lazy idiom. Phase M (steps 36–44, `plans/unified-lazy-idiom.plan.md`) has **not started**.
It is the phase that supplies new-idiom replacements for `derived_fn`, `SaveFn`, `ComputedValue`,
callable initial populations and bare-callable FOI kinds.

This branch is a **provisional, work-in-progress clean surface**. It is a demonstration of where
the API is heading, made quickly but not messily. The rules:

1. **Remove a door only where a working replacement exists today.** No capability is lost.
2. **Doors without a replacement stay in the code, unchanged and silent.** They emit no warnings.
   They are tagged *legacy* in their docstrings, never used in docs or examples, and listed on the
   published roadmap page with the Phase M step that replaces each one.
3. **Rebuild the published docs from scratch.** A few short, accessible pages in the summer2
   documentation style, using only the unified idiom. The textbook ports stay, migrated to the
   same idiom. A roadmap page is included.
4. **Publish to GitHub Pages** at <https://monash-emu.github.io/summer4/>. A GitHub Actions
   workflow builds the site on every push to `main`.
5. **Keep the machine-checked records** (roadmap runbook, coverage ledger, composability survey,
   tb-ports). Exclude them from the published site rather than moving them, because the scripts,
   tests, `AGENTS.md` and the immutable plans all cite their current paths.

`define_fn` (the user's wording) is taken to mean `derived_fn`. No `define_fn` exists in the repo.

## What is removed, what stays

### Removed (a replacement works today)

| Door | Replacement | Where |
|---|---|---|
| `Output.sum_over(prop)`, `Output.total()` | `Output.sum_by(...)` | `src/summer4/results/output.py` |
| `OutputSet` / expression `sum_over`, `total` | `sum_by` | `src/summer4/results/outputset.py` |
| Save-spec `total()`, `owner(sum_over=...)` | `sum_by()`, `group_by=` | `src/summer4/results/plan.py` |
| `PropertyData.sum_over(prop)` | `PropertyData.sum_by(prop)` | `src/summer4/jax/propertydata.py` |
| `FlowRef.sum()`, `FlowRef.sum_over(prop)` | `FlowRef.sum_by(...)` | `src/summer4/flows/rates.py` |
| `Reduce(sum_over=...)` | `Reduce(group_by=...)` | `src/summer4/flows/rates.py` |
| `Output.cumulative()` (all three surfaces, from #70) | `cumulative_integral()` for a rate, `cumsum()` for amounts | `results/output.py`, `outputset.py`, `plan.py` |
| `InitialPopulation` / `Split` callable base values and weights | `defer(f)(Param("a"), ...)`, a run-stage rate expression already accepted at `src/summer4/flows/initial.py` (`_validate_base_value`, the weights coercion) | `src/summer4/flows/initial.py` |

`src/summer4/_deprecation.py` is deleted if nothing imports it once these removals are done.

The callable-initial-population row pre-empts the deprecation half of roadmap step 42. Remove it
**only if** step C2's equivalence test passes for base values, per-trait weights and whole-table
weights. Any form that test cannot reproduce stays, and moves to the legacy table below.

### Legacy (stays; no replacement until Phase M)

| Door | Replaced in | Note |
|---|---|---|
| `FlowModel.compile(derived_fn=...)`, `derived_refs` | step 39 (`field_refs`, `StateArray`, `Params`), deprecated in step 41 | State-dependent custom quantities shared across flows have no other route today |
| `SaveFn`, `ComputedValue` | step 40 (save expressions) | Saving an arbitrary per-step value |
| `ForceOfInfection(kind=<callable>)` | step 42 (`custom_kind`) | The callable receives no parameters (CX11) |

`prepare_fn` and `Transform` are not doors on the removal path. The plan keeps both, so they stay
in the API reference, but the new guide pages do not need either.

## How this relates to further unification (roadmap)

- **No more deprecate-then-remove (user decision 2026-10-30).** Each Phase M step deletes its old
  door in the same commit that lands the replacement. The step migrates every caller and deletes
  the door's tests, keeping any equivalence test as a replacement-only test with hand values. The
  runbook carries this as one correction row covering steps 40, 41, 42 and 45:
  - **Step 40** deletes `SaveFn` and `ComputedValue` instead of warning.
  - **Step 41** deletes `compile(derived_fn=...)`, `derived_refs`, rule R3 and the
    `params_are_static` keyword with its 36 test call sites. Its "hoist table unchanged for
    `derived_fn` models" test goes.
  - **Step 42** deletes the bare-callable FOI `kind` (and the initial-population callables, if C2
    did not already remove them).
  - **Step 45** (`chore/remove-deprecated-doors`) is **dropped**: it gets status `done` with a note
    that steps 40–42 absorbed it, and is removed from the landing line.
    `futureplans/deprecated-lazy-doors-removal.md` and its README bullet are deleted in C0.
  - **Notebook 27** (the Phase M gate) can no longer run the old idiom after each step. Its
    "before" sections become text-only cells, and each comparison plot uses the values the old
    idiom produced, saved once in the step that removes it (`tests/data/` or the notebook's own
    data file), or a hand-computed reference.

  This branch's C3 legacy tags say "removed in roadmap step N".
- **Phase M steps 36–44 now write into the new docs.** Each step that adds a replacement:
  - adds the new idiom to the matching guide page;
  - deletes that door's row from the published roadmap page's "still here" table.

  Add this to each step's *Do* in the runbook (a correction row, because the plan is immutable).
- **Step 44** (`docs/unified-lazy-idiom`) shrinks. The docs are already written in the unified
  idiom, so step 44 only converts what Phase M adds.
- **Track G (steps 16–19)** and the open PRs #46–#53 add or edit pages under the old docs tree.
  Each needs a rebase that moves its page into the new layout. Record this under each step's
  *What the previous worker left you*.
- **The landing line** becomes `… 35 → 48 → 28 → 36 …`. Step 48's handoff points to step 28.
  Update `tests/test_roadmap.py` expectations through `pixi run roadmap`.
- The **published roadmap page** (`docs/roadmap.md`) is a short reader-facing summary. The runbook
  (`docs/dev/roadmap.md`) stays the authoritative, agent-facing record. The published page links
  to the runbook on GitHub and must not describe planned behaviour in the present tense.

## New docs layout

```text
docs/
  index.md                    what summer4 is, install, a 15-line SIR, "where next"
  guide/
    index.md
    01-basic-model.ipynb       compartments, a TransitionFlow, run(), plot
    02-flow-types.ipynb        transition / entry / exit, Param rates
    03-rates.ipynb             Param, Time, arithmetic, np.* / exp, defer, named, TableInterp
    04-derived-outputs.ipynb   OutputSet, outputs.ref, FlowMass, sum_by, cumulative_integral,
                               cumsum, integrate, at, array / with_values
    05-stratification.ipynb    Property / Trait / Stratification, adjustments, sum_by over strata
    06-mixing.ipynb            ForceOfInfection, MixingMatrix, age stratification and ageing
    07-initial-population.ipynb  InitialPopulation / Split, including defer(f)(Param(...))
    08-calibration.ipynb       Target, BayesianModel, one short fit
  textbook/                    kept; chapters migrated to the unified idiom
  roadmap.md                   what exists, what is legacy, what Phase M brings
  releases.md                  this release's removals and their replacements
  api/                         autosummary of the kept surface; legacy items marked
  dev/, evaluation/            unchanged on disk, excluded from the build
```

Removed from disk: `docs/getting-started/`, `docs/user/`, `docs/cookbook/`, `docs/case-studies/`,
`docs/summer2/` and the old `docs/releases/` pages.

Guide pages 01–06 are adapted from the existing `docs/summer2/` ports, which are already in the
summer2 style and carry the attribution. They are trimmed and rewritten into the unified idiom,
not written fresh. Each page:

- opens with what it covers;
- uses one small model;
- has a short paragraph before every code cell;
- plots with the pandas Plotly backend, with a title and axis labels;
- ends with "things to try".

These are example docs, not acceptance gates. They carry no *What to check* boxes.

Coverage ledger: the summer2 doc-page rows and some API rows cite `docs/summer2/*` or `docs/user/*`
pages. Step C6 points each one at its new page, or downgrades its `Status` where no new page covers
it, then runs `pixi run coverage-write`.

## Steps

Each step below is one commit on `feat/clean-surface`. Workers are Claude subagents: haiku
implements where the files and spellings are named, sonnet otherwise. Each checker runs one tier
up, read-only and in fresh context. The protocol is in `docs/dev/roadmap.md` § *Running a step
with tiered workers*. Every step ends with a `### Handoff` entry in the runbook's step 48 section,
which records `**Worker:** Claude <tier>; checker <tier>; attempts <n>`.

### C0 — cut the branch, land the plan · orchestrator

- Confirm the gate. Run `git switch -c feat/clean-surface origin/main`.
- Commit this file as `plans/clean-surface.plan.md`.
- Add step 48 to `docs/dev/roadmap.md`: its table row, landing line, *Current position*, and a
  section with *Summary* / *Read first* / *Do* / *Exit checks* / *Handoff*.
- Add the correction rows for steps 40–42, 44, 45 and 16–19 described above. Set step 45 to
  `done` (absorbed) and remove it from the landing line.
- Delete `futureplans/deprecated-lazy-doors-removal.md` and its README bullet.
- **Check:** `pixi run roadmap`.

### C1 — remove renamed spellings · haiku, checker sonnet

**Files:** the five `src/` files in the *Removed* table (all rows except the initial-population
row), plus `_deprecation.py`.

- Delete each deprecated method, keyword branch and `warn_renamed` call.
- Migrate every caller in `tests/`, `examples/notebooks/`, `benchmarks/` and `summer2bench/`.
- Delete the tests that only assert a deprecation warning.
- **Sanity check:**
  - `grep -rnE "sum_over|\.total\(|\.cumulative\(|FlowRef\([^)]*\)\.sum\(\)|warn_renamed" src tests examples benchmarks summer2bench` is empty, except `group_by`-style hits the checker confirms;
  - `pixi run lint` passes;
  - `pixi run test-quick` passes.

### C2 — initial-population callables · sonnet, checker opus

**Files:** `src/summer4/flows/initial.py`, the new `tests/test_initial_defer.py`, and callers.

- For each callable form (a base value, a per-trait weight, a whole-table weight), write an
  equality test: the callable spelling and `defer(f)(Param(...))` give equal `y0`, and equal
  gradients with respect to one parameter.
- If all three pass, delete the callable branches (`initial.py`: `_validate_base_value`, the
  per-trait and whole-table weight coercions, and both evaluation sites). Turn the tests into
  replacement-only tests with hand-computed values.
- If a form cannot be reproduced, keep it, and add it to the *Legacy* table on the roadmap page
  (C7). Stop and report this to the orchestrator.
- **Sanity check:**
  - `grep -n "callable(" src/summer4/flows/initial.py` shows only what the step's report says it
    kept;
  - `pixi run test-quick` passes.

### C3 — tag legacy doors · haiku, checker sonnet

**Files:**

- `src/summer4/flows/compiled.py` (`compile` docstring, `derived_fn`);
- `src/summer4/flows/rates.py` (`derived_refs`);
- `src/summer4/results/plan.py` (`SaveFn`, `ComputedValue`);
- `src/summer4/epi/infection.py` (`kind` docstring).

- Prepend each docstring with `Legacy: removed in roadmap step N, replaced by <new spelling>. Not
  used in the documentation.` Change no behaviour and add no warnings.
- **Sanity check:** `git diff --stat` touches only docstrings, and `pixi run lint` passes.

### C4 — docs skeleton · sonnet, checker opus

**Files:** `docs/conf.py`, `docs/index.md`, `docs/api/*`, the deletions listed above.

- Delete the old trees.
- Add `"dev/**", "evaluation/**"` to `exclude_patterns`.
- Write `index.md` with toctrees for Guide, Textbook, Roadmap, Releases and API.
- Set `html_baseurl = "https://monash-emu.github.io/summer4/"`.
- Trim `docs/api/summer4.rst` to the kept modules.
- Fix or remove every `{doc}` link the deletions broke. `grep -rn "{doc}" docs --include=*.md --include=*.ipynb` resolves, outside `dev/` and `evaluation/`.
- Add `tests/test_docs_idiom.py`. It fails if any `.md` or `.ipynb` under `docs/` (outside the
  excluded folders) contains one of the forbidden idioms: `derived_fn`, `derived_refs`, `SaveFn`,
  `ComputedValue`, `sum_over`, `.total(`, `.cumulative(`, or a callable passed as `kind=`. This
  check keeps the docs clean as Phase M lands.
- **Check:** `pixi run -e docs docs-strict` builds on the mini (`mini build -C dev/EMU/s3a -- pixi run -e docs docs-strict`). The guide toctree may still point at placeholder pages at this stage.

### C5 — guide pages 01–08 · sonnet ×(2–3 in parallel), checker opus

- Split the work into three groups, each one worker: 01–03, 04–05 and 06–08.
- **Files:** only that group's `docs/guide/*.ipynb`. Each page follows *New docs layout*.
- Strip outputs before committing (`pixi run check-notebooks`).
- Write the code from `from summer4 import ...` only, with no private imports.
- **Sanity check:**
  - `pytest tests/test_docs_idiom.py` passes;
  - each page executes with `jupyter nbconvert --execute` on the mini;
  - the checker reads each page against the non-programmer on-ramp: no subclassing, no
    `register_*`, no NamedTuple schemas.

### C6 — textbook migration and ledger · sonnet, checker opus

- Migrate `docs/textbook/07-numerical-solutions`, `08-derived-outputs` and `20-calibration` to
  the unified idiom. These are the chapters that use removed or legacy idioms.
- Grep the other chapters for the C4 list as well.
- Keep the prose and figures. Every line of code must be in current idiom.
- Point the coverage-ledger and tb-ports rows that cite deleted pages at their new pages, or
  downgrade their status. Then run `pixi run coverage-write`.
- **Check:** `pixi run coverage`, and `pytest tests/test_coverage_ledger.py tests/test_docs_idiom.py`.

### C7 — roadmap and release pages · haiku, checker sonnet

**Files:** `docs/roadmap.md`, `docs/releases.md`, `CHANGELOG.md`.

The roadmap page has three short sections:

- **What summer4 does today**: one paragraph per guide page.
- **Still here, on the way out**: the *Legacy* table, in reader language, with the step that
  replaces each door.
- **Coming next**: Phase M in future tense (one expression type for rates and outputs, saving any
  expression, typed stages), then Track G.

It links to `docs/dev/roadmap.md` on GitHub.

The release page lists the *Removed* table as a migration table.

### C8 — publish to GitHub Pages · haiku, checker sonnet

**Files:** the new `.github/workflows/docs.yml`, and `.github/workflows/ci.yml` (one job).

- `docs.yml` runs on `push` to `main` and on `workflow_dispatch`. It needs `permissions: pages:
  write, id-token: write, contents: read` and `concurrency: pages`.
- The `build` job:
  - checkout;
  - `prefix-dev/setup-pixi` with `environments: docs` and the cache keyed like `ci.yml`;
  - `pixi run -e docs docs-strict`;
  - `actions/upload-pages-artifact` with `path: docs/_build/html`.
- The `deploy` job runs `actions/deploy-pages` in environment `github-pages`.
- Add a `docs` job to `ci.yml` that runs `docs-strict` on pull requests and uploads the HTML as a
  normal artifact. Reviewers can download the built site before merge.
- **One-time repo setting (needs the user's go-ahead; admin action):** set Pages to *GitHub
  Actions* as its source. Use `gh api -X POST repos/monash-emu/summer4/pages -f build_type=workflow`,
  or Settings → Pages. Pages is not enabled today (the API returns 404). The org site
  `monash-emu/monash-emu.github.io` exists, so this repository's site appears at `/summer4/`.
- Set the repo homepage to the site URL with
  `gh repo edit monash-emu/summer4 --homepage https://monash-emu.github.io/summer4/`. This also
  needs the user's go-ahead.

### C9 — checks, PR, handoff · orchestrator

- Run the full *Required checks* from `AGENTS.md`. Run `test` and `docs-strict` on the mini.
- Run `pixi run composability-write`. No CX finding changes status unless C2 removed the callable
  initial populations: CX33 then stays open (the FOI half remains), and that is noted in its
  evidence.
- Open the PR into `main`, citing step 48, the removal table and the legacy table.
- Write the handoff: step 48 `done`, step 28 `next`.
- **Acceptance (deviation from the feature bar, approved 2026-10-30):** the gate is the built
  site, downloaded from the PR's `docs` artifact and read page by page. No new
  `examples/notebooks/` gate notebook is added. `check-branch` is satisfied by the migrated
  notebooks. A removal-only change has no new behaviour to plot.

## Decisions on the open questions

1. ~~Acceptance gate~~ **Decided 2026-10-30: the built site is the gate**, read from the PR's
   `docs` artifact. No new gate notebook.
2. ~~How the branch lands~~ **Decided 2026-10-30: a normal PR into `main`.** The site goes live
   when it merges.
3. ~~Should Phase M keep its deprecate-then-remove cycle?~~ **Decided 2026-10-30: delete
   outright.** See *How this relates to further unification*.
