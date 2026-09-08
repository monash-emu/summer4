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

```bash
pixi install
pixi run test
pixi run lint
pixi run bench
```

Environments: `default` (JAX 0.6.x), `latest` (current JAX), `nb` (notebooks).
The package is also pip-installable via hatchling (`pip install .`).
