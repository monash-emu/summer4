# Project layout

```text
summer4/
├── src/summer4/              # the package — NumPy only, 570 lines
│   ├── __init__.py           # the entire public API
│   ├── properties.py         # Property, Trait
│   ├── selectors.py          # selector node types, SelectorOps
│   └── propertymap.py        # PropertyMap, Stratification, Kleene evaluation
├── explorations/             # spikes, not importable as summer4
│   └── flows/                # join / flows / rates / Euler prototype
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

Four modules, and the dependency order between them is strict:

```{mermaid}
flowchart LR
    S["selectors.py<br/>node types, SelectorOps"] --> P["properties.py<br/>Property, Trait"]
    S --> M["propertymap.py<br/>PropertyMap, Stratification"]
    P --> M
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

## `plans/`

Accepted plans travel with the branch that implements them and are copied into
`plans/` on merge. They are a historical record: a committed plan is never
edited, and follow-up work adds a new plan instead. The current set is

| Plan | Subject |
|---|---|
| `summer4_compartment_taxonomy_*.plan.md` | The implemented taxonomy layer |
| `explore-flows.plan.md` | The first flows spike |
| `explore-flows-refs.plan.md` | Schema-built derived and flow refs |
| `refine-flows.plan.md` | Nested derived rates, adjustments, JIT Euler |

Three of the four plans are exploratory. That ratio is an accurate picture of
where the project is.

## `docs/`

```text
docs/
├── conf.py                   # Sphinx + myst-nb configuration
├── index.md
├── getting-started/          # installation, quickstart notebook
├── user/                     # chapters 1-7 (notebooks + migration guide)
├── dev/                      # this guide
│   └── flows/                # the flows spike, executed at build time
├── textbook/                 # summer-textbook port and its roadmap
└── api/                      # autosummary stubs
```

Notebook sources are committed **cleared** (no outputs, no execution counts) and
executed by Sphinx at build time with `nb_execution_raise_on_error = True`. A
documentation build is therefore a test run: if the library changes underneath a
documented claim, the build fails.
