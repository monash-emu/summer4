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

## JAX is the primary runtime target

Compiled models, the results query surface, and calibration losses are expected
to run under **`jax.jit`** (and friends). NumPy remains fine for host-side index
arithmetic and taxonomy code that must stay JAX-free at import time; it is not
the default path for traced numerics going forward.

When changing or reviewing JAX-facing code (vector fields, `Trace` ops, save
evaluation, losses):

1. Prefer **vectorized** gathers, `segment_*`, and `lax.scan` over Python loops
   that emit one JAX op per iteration.
2. Where possible or useful, validate with **`jax.make_jaxpr`** (or
   `jax.make_jaxpr(... )(...).jaxpr` size / pretty-print) so the program handed
   to XLA does not explode with trajectory length, compartment count, or similar
   static sizes. Complement that with timing / profiling benchmarks when the
   change is performance-sensitive.
3. Host-side Python that builds **constant** index arrays (then one batched
   gather) is fine and does *not* grow the jaxpr with those loops. Python
   iteration over a small static set of named targets/groups inside a jitted
   loss *does* unroll — keep that set small, or fold it into one batched op.

Gotchas and deferred jaxpr/XLA risks live in [`futureplans/`](futureplans/);
read that folder when planning, and add a short note there when you flag
something for later rather than fixing it on the current branch.

## Future plans (deferred gotchas)

[`futureplans/`](futureplans/) is an agent-maintained scratch archive of issues
noticed during normal plan building that are **out of scope for the current
branch** but should not be forgotten (compile-time jaxpr blow-ups, API sharp
edges, follow-up refactors).

- **Read** it before proposing or extending feature work, alongside the coverage
  ledger.
- **Write** a short markdown note when you discover a gotcha you are not fixing
  now: one concern per file, concrete pointers into the code, and what “done”
  would look like.
- Do not treat these notes as accepted design plans — those still belong under
  `plans/`. When a note is ready to implement, promote it into a real
  `plans/<slug>.plan.md` on a feature branch.

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
