# Agent instructions

## Branches

Do not implement features, fixes, or refactors on `main`. Create a branch from the latest `main` before editing:

- `feat/<short-slug>` — new user-facing or API behaviour
- `fix/<short-slug>` — bug fixes
- `docs/<short-slug>` — documentation-only
- `chore/<short-slug>` — tooling, packaging, CI

Keep the branch focused. Open (or update) a PR into `main`; do not fast-forward feature work onto `main` in-place.

```bash
git fetch origin
git switch -c feat/my-feature origin/main   # or local main if no remote
```

After clone or when hooks are missing, run `pixi run setup` so commit and merge
hooks are installed. The pre-commit hook formats staged Python with Black
(line length 100) and rejects notebooks that still have outputs or execution
counts.

## Style

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
unless this file or the tooling below says otherwise.

- **Line length is 100**, not 80. Black is the formatter (`pixi run format`).
- **Every function definition is type-annotated** (parameters and return type),
  including tests and scripts. mypy `--strict` on `src/summer4` enforces this
  for the package.

## Plans land in the workspace on merge

Cursor plans must travel with the branch so a merge automatically copies them into the repo.

1. As soon as a plan is accepted, copy it onto the feature branch as `plans/<slug>.plan.md` (keep the original filename if it already ends in `.plan.md`).
2. Also keep a copy under `.cursor/plans/` if you are iterating in Cursor; `.gitignore` tracks that folder.
3. Merge the branch into `main`. Git brings `plans/` along. The `post-merge` hook then copies any `*.plan.md` added by the merge from `.cursor/plans/` (or other incoming paths) into `plans/` if they are not already there.

Do not leave the only copy of a plan outside the repository. Do not edit a plan file that was already committed as historical record; add a new plan for follow-up work.

## Before planning feature work: read the coverage ledger

`docs/evaluation/coverage-ledger.md` is the authoritative record of what summer4
covers and what full coverage requires. It holds:

- every summer2 API symbol exercised by the summer2 docs or the summer textbook,
  with its status in the shipped package (`Shipped`) and with the
  `explorations/flows/` spike promoted (`Spike`);
- the same for all twenty textbook chapters and eleven summer2 doc pages;
- ten ordered work packages (WP1–WP10) with the ledger IDs each one closes, and
  a computed table of coverage after each.

Read it before proposing feature work, so a branch lands the next package rather
than a duplicate of one already designed. Quote its IDs (`F1`, `D4`, `WP2`) in
plans and PR descriptions — they are stable, and agents on other branches can
resolve them without this context.

**If your branch changes what summer4 can do, update the ledger in the same
commit.** Change the affected rows' `Shipped` status, then:

```bash
pixi run coverage-write   # recompute the progression table
pixi run coverage         # verify statuses and quoted totals
```

`tests/test_coverage_ledger.py` fails if a status is misspelled, if the spike
column claims less than the shipped column, if the progression table is stale,
or if `docs/evaluation/index.md` quotes totals the ledger no longer supports.

Status values are exactly `full`, `partial` or `none`:

| Status | Meaning |
|---|---|
| `full` | A documented, working way to achieve what the summer2 symbol does |
| `partial` | Achievable, but the user supplies something summer2 supplied, or a material limitation applies |
| `none` | No way to do it |

Six rows are marked as never reaching `full` on purpose — they are summer2
shapes summer4 has rejected, not capabilities it lacks. Do not "fix" them
without changing that decision explicitly.

## Documentation

Documentation lives in `docs/` and builds with `pixi run -e docs docs`. Every
notebook on the site executes at build time (`nb_execution_raise_on_error`), so
a docs build is a test run — add `pixi run -e docs docs-strict` to your checks
when a change touches documented behaviour.

Two rules that are easy to get wrong:

- **Do not describe planned behaviour in the present tense.** Where a summer2 or
  textbook capability has no summer4 equivalent, say so and cite the ledger.
- **`docs/dev/flows/` is generated.** Its code cells are byte-identical to
  `explorations/flows/01-flows.ipynb` and `02-flows.ipynb`, and
  `tests/test_flows_docs_sync.py` enforces that. Edit the spike notebooks and
  regenerate; never edit the published copies in place.

## Feature acceptance bar

A branch that adds or changes **features** (new public API, new modelling capability, or user-visible behaviour) is not complete unless it includes all three:

1. **The implementation** on a feature branch, not `main`.
2. **Tests** in `tests/` that cover the new behaviour (unit and/or property tests; no placeholder `assert True`).
3. **Example documentation as a notebook** in `examples/notebooks/`. Notebooks are runnable smoke tests: `pixi run test` executes every `*.ipynb` there.

Notebook rules:

- One notebook per feature (or a clearly named extension of an existing notebook when the feature is a small addition).
- Code cells must be plain Python: no IPython magics, no hidden manual steps.
- The notebook should tell a short story (what the feature is, a realistic example) and **assert** the outcomes it claims.
- Prefer `from summer4 import ...` over re-implementing library code inline.

Bug-fix branches need regression tests. Add or update a notebook only when the fix changes documented user-facing behaviour.

Docs-only and chore branches do not need notebooks.

## Required checks

Before you ask for a merge:

```bash
pixi run lint
pixi run format-check
pixi run check-notebooks
pixi run test
pixi run check-branch
pixi run coverage
```

`check-branch` fails if this branch changed `src/summer4` but did not also change `tests/` and (for new or changed public modules) `examples/notebooks/`, or if a feature branch has no plan under `plans/`.
