# Project layout

```text
summer4/
├── src/summer4/              # the package
│   ├── __init__.py           # the public API
│   ├── properties.py         # Property, Trait
│   ├── selectors.py          # selector node types, SelectorOps
│   ├── propertymap.py        # PropertyMap, Stratification, Kleene evaluation
│   ├── flows/                # joins, rates, CompiledModel
│   └── jax/                  # PropertyData (JAX pytree)
├── examples/notebooks/       # runnable feature notebooks (executed by pytest)
├── docs/                     # this site
├── tests/                    # unit, property-based and workflow tests
├── benchmarks/               # pytest-benchmark suites
├── scripts/                  # repository tooling
├── plans/                    # accepted plans, kept as historical record
├── AGENTS.md                 # the contribution contract
└── pixi.toml                 # environments and task table
```

## `src/summer4`

Taxonomy modules stay NumPy-only. Flows compile a JAX vector field lazily.

```{mermaid}
flowchart LR
    S["selectors.py"] --> P["properties.py"]
    S --> M["propertymap.py"]
    P --> M
    M --> F["flows/"]
    F --> J["jax/"]
```

`selectors.py` imports `Trait` only under `TYPE_CHECKING`, which is what keeps
the cycle out of the runtime import graph while still typing the `Selector`
union precisely.

`__init__.py` re-exports every public name and declares `__all__`. If a name is
not in that list it is not public, and the API reference is generated from it.

## `scripts/`

| Script | Task | Purpose |
|---|---|---|
| `setup_hooks.py` | `pixi run setup` | Install pre-commit and post-merge git hooks |
| `check_feature_branch.py` | `pixi run check-branch` | Enforce the feature bar (tests + notebook + plan) |
| `check_cleared_notebooks.py` | `pixi run check-notebooks` | Reject notebooks with outputs or execution counts |
| `copy_merged_plans.py` | post-merge hook | Copy merged `*.plan.md` into `plans/` |
| `bench_json.py` | `pixi run bench-json` | Write benchmark JSON keyed by JAX version |
| `coverage_report.py` | `pixi run coverage` | Parse the coverage ledger and check quoted totals |

## `plans/`

Accepted plans travel with the branch that implements them and are copied into
`plans/` on merge. They are a historical record: a committed plan is never
edited, and follow-up work adds a new plan instead.

## `docs/`

```text
docs/
├── conf.py                   # Sphinx + myst-nb configuration
├── index.md
├── getting-started/          # installation, quickstart notebook
├── user/                     # chapters 1–8 (notebooks + migration guide)
├── dev/                      # this guide
├── textbook/                 # summer-textbook port and its roadmap
├── evaluation/               # feature completeness, docs coverage, gaps
└── api/                      # autosummary stubs
```

Notebook sources are committed **cleared** (no outputs, no execution counts) and
executed by Sphinx at build time with `nb_execution_raise_on_error = True`. A
documentation build is therefore a test run: if the library changes underneath a
documented claim, the build fails.
