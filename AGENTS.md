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
```

`check-branch` fails if this branch changed `src/summer4` but did not also change `tests/` and (for new or changed public modules) `examples/notebooks/`, or if a feature branch has no plan under `plans/`.
