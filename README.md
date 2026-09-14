# summer4

JAX-native compartmental modelling. Stage 1 is the compartment taxonomy:
`Property`, `Trait`, a Kleene three-valued selector algebra, and an immutable
NumPy-backed `PropertyMap`.

The taxonomy layer depends on NumPy only. JAX lives in the pixi environment
matrix so later stages can benchmark solvers across versions.

## Concepts

- **Property** — a named group of mutually exclusive traits, e.g. `age: ["0-4", "5-9", "10+"]`.
- **Trait** — one value of a property; also a selector leaf (`age["0-4"]`).
- **Selector** — composable query (`state["I"] & age["0-4"]`, `~sev["severe"]`, `sev.absent()`).
- **PropertyMap** — every compartment is a row in an integer code table. Queries compile to `int32` index arrays.
- **Stratification** — applying a property to a (possibly filtered) subset of an existing map.

Ragged maps are first-class: a property may be absent on some compartments.
Absent is Kleene *unknown*, so both `severity["mild"]` and `~severity["mild"]`
exclude unstratified compartments. Use `severity.absent()` / `severity.present()`
to reach them.

## Usage

```python
from summer4 import Property, PropertyMap

state = Property("state", ("S", "I", "R"))
age = Property("age", ("0-4", "5-9", "10+"))
sev = Property("severity", ("mild", "severe"))

pm = (
    PropertyMap.from_property(state)
    .stratify(age)
    .stratify(sev, where=state["I"])
)

pm.select(state["I"] & age["0-4"])
pm.select(age[("0-4", "5-9")] & ~sev["severe"])
pm.select(sev.absent())
pm.partition(age)
```

## Development

Read [AGENTS.md](AGENTS.md) before changing the code: work on a feature branch,
follow the Google Python Style Guide (function definitions are always
type-annotated; Black line length 100), copy the plan into `plans/`, and land
tests plus a runnable example notebook.

```bash
pixi install
pixi run setup
pixi run register-kernel
pixi run test
pixi run lint
pixi run check-branch
pixi run bench
```

## Coverage

`docs/evaluation/coverage-ledger.md` records what summer4 covers against the
summer2 API and the summer textbook, and the ordered work packages that lead to
full coverage. It is the reference other branches and agents should plan
against.

```bash
pixi run coverage
```

## Documentation

```bash
pixi run -e docs docs         # build HTML into docs/_build/html
pixi run -e docs docs-serve   # http://localhost:8765
pixi run -e docs docs-strict  # warnings become errors
```

The site has a **user guide** (properties, maps, selectors, ragged
stratification, partitions, provenance), a **developer guide** (architecture,
data structures, Kleene evaluation, measured performance, tooling, and the
`explorations/flows/` spike published as two executed walkthroughs), a partial **textbook** port, a **project evaluation** of feature
completeness against summer2 and the summer textbook, and the generated **API
reference**. Every notebook on the site is executed at build time, so a docs
build is also a test run.

The default environment includes `ipykernel` so example notebooks run in VS Code/Cursor.
Select the interpreter at `.pixi/envs/default/bin/python`, or run `pixi run register-kernel`
and choose the **Python (summer4)** kernel.

Environments: `default` (JAX 0.6.x + notebook kernel), `latest` (current JAX), `nb` (JupyterLab).
The package is also pip-installable via hatchling (`pip install .`).
Example notebooks in `examples/notebooks/` are executed as smoke tests.
