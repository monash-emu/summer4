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

Parameter work is staged (compile / run start / per step); see
`docs/dev/run-stages.md`. Put `t`/`y`-independent computation in `prepare_fn`
or a hoistable rate subtree, never in `derived_fn`.

When changing or reviewing JAX-facing code (vector fields, `Output` ops, save
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

## To continue the work: read the roadmap

`docs/dev/roadmap.md` is the authoritative record of **where the work has got
to**. The coverage ledger below says what summer4 can do; the roadmap says what
to do next. If you have been told only "next step", that file is your entire
brief.

1. Read its *Current position* block. It names exactly one step.
2. **Present that step's *Summary* to the user before doing anything else.**
3. Follow *Read first*, *Do*, *Exit checks* and *Handoff* in that step's
   section. Where the step's *Do* contradicts a plan under `plans/`, the step
   wins — plans are immutable history and the roadmap carries the corrections.
4. **Finishing a step means committing its handoff on the step's own branch**,
   so it merges with the work. A step that lands without updating the roadmap
   has left the next session with nothing to start from.

```bash
pixi run roadmap   # the runbook parses, one step is current, its references resolve
```

## Before planning feature work: read the coverage ledger

`docs/evaluation/coverage-ledger.md` is the authoritative record of what summer4
covers and what full coverage requires. It holds:

- every summer2 API symbol exercised by the summer2 docs or the summer textbook,
  with a single `Status` (`full` / `partial` / `none`);
- the same for all twenty textbook chapters and eleven summer2 doc pages;
- remaining work packages (WP2–WP10; WP1 is applied) with the ledger IDs each
  one closes, and a computed table of coverage after each.

Read it before proposing feature work, so a branch lands the next package rather
than a duplicate of one already designed. `docs/evaluation/tb-ports.md` is its companion for
the Kiribati and tb_macro tuberculosis model ports: capability rows `KI*` / `TM*`,
each naming the work package (WP3, WP10, WP12–WP16) that closes it. Quote its IDs (`F1`, `D4`, `WP2`) in
plans and PR descriptions — they are stable, and agents on other branches can
resolve them without this context.

**If your branch changes what summer4 can do, update the ledger in the same
commit.** Change the affected rows' `Status`, then:

```bash
pixi run coverage-write   # recompute the progression table
pixi run coverage         # verify statuses and quoted totals
```

`tests/test_coverage_ledger.py` fails if a status is misspelled, if the
progression table is stale, or if `docs/evaluation/index.md` quotes totals the
ledger no longer supports.

Status values are exactly `full`, `partial` or `none`:

| Status | Meaning |
|---|---|
| `full` | A documented, working way to achieve what the summer2 symbol does |
| `partial` | Achievable, but the user supplies something summer2 supplied, or a material limitation applies |
| `none` | No way to do it |

Five rows are marked as never reaching `full` on purpose — they are summer2
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
- **Flows ship in `summer4`.** Document them in the user guide against
  `from summer4 import ...` / `CompiledModel`. There is no spike copy under
  `explorations/` or `docs/dev/flows/`.

## Feature acceptance bar

A branch that adds or changes **features** (new public API, new modelling capability, or user-visible behaviour) is not complete unless it includes all three:

1. **The implementation** on a feature branch, not `main`.
2. **Tests** in `tests/` that cover the new behaviour (unit and/or property tests; no placeholder `assert True`).
3. **Example documentation as a notebook** in `examples/notebooks/`. Notebooks are runnable smoke tests: `pixi run test` executes every `*.ipynb` there.

Notebook rules:

Example notebooks are the **user gate**, not only a pytest smoke test. A person
runs `pixi run notebook` and signs the feature off by reading them. Write every
gate notebook in the summer2 documentation style (see `docs/summer2/`): prose a
modeller can follow, a figure of the claim, and an assertion of that same claim.
A notebook that only asserts is not a user gate.

- One notebook per feature (or a clearly named extension of an existing notebook when the feature is a small addition).
- Code cells must be plain Python: no IPython magics, no hidden manual steps.
- Open with what the page is for and which claim the reader is checking.
- Before each code section, a markdown heading and a short explanation of what the next cells do and what the figure should show.
- **Plot** every series, comparison, or size the reader is asked to judge. Use the pandas Plotly backend (`pd.options.plotting.backend = "plotly"`, `pio.renderers.default = "notebook_connected"`), with a title and axis labels. A structural page (taxonomy, jaxpr size, hoist-table length) still plots that comparison — compartment counts, edge counts, loop-body size — rather than only printing it.
- **Assert** the outcomes the prose and the figure claim. Do not replace a plot with an assert, or an assert with a plot.
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
pixi run roadmap
```

`check-branch` fails if this branch changed `src/summer4` but did not also change `tests/` and (for new or changed public modules) `examples/notebooks/`, or if a feature branch has no plan under `plans/`.
