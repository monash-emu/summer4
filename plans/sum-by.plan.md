<!--
Accepted 2026-10-29 (fast-tracked into 0.2.0a7 for the summer4template release). Historical
record once merged: do not edit. Follow-up work goes in a new plan.
-->

# `sum_by`: one name for summing across strata

## Context

Two names for the same operation read wrongly:

- `Output.total()` (and `OutputExpr.total()`, `Compartments(...).total()`) sums the
  aligned axis — every compartment or edge — and **keeps time**. "Total" suggests a
  single number. In the summer4template, `FlowMass("infection").total()` is the
  infection rate per day, not a total.
- `sum_over(age)` (on `Output`, `OutputExpr`, `PropertyData`, `FlowRef`) keeps one
  value per age. Mathematically "sum over age" removes age.

The save-time and rate-tree keyword `sum_over=` (`Compartments`, `FlowMass`,
`Reduce`) has the same inverted reading.

## Shape

Name the operation for what it keeps, as pandas' `groupby(...).sum()` does.

- **Methods:** `sum_by(*props)` on `Output`, `OutputExpr`, the save-spec chain
  (`Compartments(...).sum_by()`), `PropertyData` and `FlowRef`.
  - `sum_by()` keeps only time (what `total()` did). On an `Output` whose only dim
    is `time` it raises: time is never summed by `sum_by`.
  - `sum_by(age)` is the old `sum_over(age)`.
  - `sum_by(age, strain)` keeps one column per (age, strain) pair, over
    `PropertyMap.from_property(age).stratify(strain)` (first property slowest).
    Compartments or edges missing any kept property do not contribute. New for
    `Output`, `OutputExpr` and `PropertyData`.
  - Edge outputs still need `side=` when properties are kept.
  - `FlowRef.sum_by` takes none or one property (a rate node is grouped by at
    most one property); `FlowRef.sum()` becomes `sum_by()`.
- **Constructor keyword:** `group_by=` on `Compartments`, `FlowMass` and `Reduce`.
  It cannot be `sum_by=`: a dataclass field named `sum_by` would hide the
  `sum_by()` method on `Compartments` and `FlowMass`. `group_by=` already means
  "one value per trait of this property" in `ForceOfInfection(group_by=...)`.
- **Deprecation:** `total()`, `sum_over()`, `FlowRef.sum()` and `sum_over=` keep
  working, emit `DeprecationWarning` naming the replacement, and return the same
  result. Removal is a later release.
- Output dims are unchanged: a kept-properties output's last dim is `"group"`, as
  `sum_over` gave.

Reducing over time stays separate: `integrate`, `cumulative`, `rolling`, `at`,
`reduce_by`.

## Files

- `src/summer4/_deprecation.py` (new), `jax/propertydata.py`, `flows/edges.py`,
  `flows/rates.py`, `flows/compiled.py`, `flows/actualize.py`, `results/output.py`,
  `results/outputset.py`, `results/plan.py`, `results/eval.py`, `epi/infection.py`,
  `epi/calibration/posterior_runs.py`.
- Every test, benchmark, example notebook and docs page that used the old
  spellings (excluding plans, release notes and roadmap history).
- `tests/test_sum_by.py` (new), `examples/notebooks/32-sum-by.ipynb` (acceptance).
- `CHANGELOG.md`.

## Tests

`tests/test_sum_by.py`: none/one/two kept properties on a ragged map against hand
sums; product-map labels; edges with `side=`; the `side=` and time-only errors;
repeated properties; `OutputExpr` evaluation; `FlowRef.sum_by` and its
one-property limit; `group_by=` fields; every deprecated spelling warns, names the
replacement, and equals the new one; a save with `sum_over=` equals `group_by=`.
The rest of the suite runs with no rename warning.
