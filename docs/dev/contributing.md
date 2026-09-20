# Contributing

The contract is `AGENTS.md` at the repository root. This page summarises it and
explains the reasoning.

## Branches

Nothing is implemented on `main`. Branch from the latest `main` first:

```bash
git fetch origin
git switch -c feat/my-feature origin/main
```

| Prefix | For |
|---|---|
| `feat/` | New user-facing or API behaviour |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `chore/` | Tooling, packaging, CI |

Open or update a PR into `main`; do not fast-forward feature work onto `main`
in place.

## The feature bar

A branch that adds or changes a **feature** — new public API, new modelling
capability, or user-visible behaviour — is incomplete without all three of:

1. the implementation, on a feature branch;
2. tests in `tests/` covering the new behaviour, unit and/or property-based,
   with no placeholder `assert True`;
3. a runnable example notebook in `examples/notebooks/`.

`pixi run check-branch` enforces this: it fails if the branch changed
`src/summer4` without also changing `tests/`, and — for new or changed public
modules — `examples/notebooks/`, or if a feature branch has no plan in `plans/`.

Bug-fix branches need a regression test. They need a notebook only when the fix
changes documented user-facing behaviour. Docs-only and chore branches need
neither.

### Notebook rules

Example notebooks are the user gate. Someone runs `pixi run notebook` and
signs the feature off by reading the page, so write it in the summer2
documentation style (`docs/summer2/`): an explanation, a titled plot of the
claim, and an assertion of that same claim. A notebook that only asserts is
not a gate.

- One notebook per feature, or a clearly named extension of an existing one.
- Plain Python only: no IPython magics, no hidden manual steps.
- Open with what the page is for. Before each code section, say what the next
  cells do and what the figure should show.
- Plot every series, comparison, or size the reader is asked to judge, with
  the pandas Plotly backend
  (`pd.options.plotting.backend = "plotly"`,
  `pio.renderers.default = "notebook_connected"`), a title, and axis labels.
  Structural pages (taxonomy, jaxpr size) plot that comparison too.
- Assert the outcomes the prose and the figure claim.
- Prefer `from summer4 import ...` over re-implementing library code inline.

## Plans

An accepted plan is copied onto the feature branch as `plans/<slug>.plan.md`
as soon as it is accepted, so that merging the branch brings the plan into the
repository. A committed plan is a historical record and is never edited;
follow-up work adds a new plan.

## Style

[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html),
with two deliberate deviations:

- **Line length is 100**, not 80. Black is the formatter.
- **Every function definition is annotated**, parameters and return type,
  including tests and scripts. `mypy --strict` enforces this on
  `src/summer4`.

## Required checks

```bash
pixi run lint
pixi run format-check
pixi run check-notebooks
pixi run test
pixi run check-branch
```

Add `pixi run -e docs docs-strict` when the change touches documented
behaviour — the site executes its own notebooks, so a stale claim on this site
is a build failure, not a cosmetic problem.

## Hooks

```bash
pixi run setup
```

installs a pre-commit hook that Black-formats staged Python and rejects
notebooks carrying outputs or execution counts, plus a post-merge hook that
copies merged plans into `plans/`. Run it after cloning and whenever hooks go
missing.
