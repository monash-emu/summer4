# Testing

```bash
pixi run test        # everything
pixi run test-all    # everything, in default and latest
```

`pytest` is configured in `pyproject.toml` with `testpaths = ["tests"]` and
`pythonpath = ["src", "."]`, so tests import `summer4` from the source tree.

## What is covered

| File | Covers |
|---|---|
| `test_properties.py` | `Property` validation, trait lookup, `isin`, `present` / `absent` |
| `test_selectors.py` | Node construction, `&` / `|` / `~`, the `__bool__` guard |
| `test_propertymap.py` | Construction, stratification, queries, partition, `group_by`, labels, equality |
| `test_taxonomy_properties.py` | Hypothesis property-based invariants |
| `test_flows.py` | Joins, `EdgeMap`, rates, `CompiledModel`, Euler |
| `test_coverage_ledger.py` | Ledger parse, quoted totals |
| `test_notebooks.py` | Executes every notebook in `examples/notebooks/` |
| `test_cleared_notebooks.py` | The cleared-notebook checker itself |
| `test_branch_workflow.py` | The feature-branch gate itself |

## Property-based tests

`test_taxonomy_properties.py` uses [Hypothesis](https://hypothesis.readthedocs.io)
to assert invariants that must hold for *any* map and *any* selector, rather
than for hand-picked examples. The invariants worth knowing about:

- **Three-way exhaustion.** For any property `p` and trait `t`,
  `select(p[t])`, `select(~p[t])` and `select(p.absent())` are pairwise
  disjoint and together cover every compartment.
- **Partition coverage.** `partition(p)` covers exactly `select(p.present())`.
- **Stratification arithmetic.** Applying `p` with `where=sel` to a map of size
  `n` where `sel` matches `m` rows yields `n - m + m·len(p.traits)` rows.
- **Replay.** Replaying `history` onto a bootstrap map reproduces the map.

## Notebooks are tests

`test_notebooks.py` executes every `examples/notebooks/*.ipynb` in-process,
cell by cell, in a shared namespace. It **rejects** IPython magics and shell
escapes, because a notebook that needs a magic is not a reproducible example.

Notebooks are expected to `assert` the outcomes they claim. A notebook that only
prints is not a test — it is a screenshot that happens to run.

They are also the manual user gate (`pixi run notebook`). Write them in the
summer2 documentation style: prose before each section, a titled Plotly figure
of the claim (including structural comparisons such as compartment counts or
loop-body size), and an assertion of that same claim. See {doc}`contributing`.

The same is true of the documentation: `nb_execution_raise_on_error = True` in
`docs/conf.py` means a broken claim on this site fails the docs build. See
{doc}`documentation`.

## Writing a test for new behaviour

Three things, in this order:

1. A unit test in the relevant `tests/test_*.py` with a concrete, hand-checked
   expectation. Prefer asserting on `labels()` or `to_dicts()` over raw index
   numbers where the intent is structural.
2. A Hypothesis invariant in `test_taxonomy_properties.py` if the behaviour is
   a law rather than a case.
3. A cell in an `examples/notebooks/` notebook that demonstrates it and asserts
   the outcome, per the feature bar in {doc}`contributing`.

## Type checking

`mypy --strict` runs over `src/summer4` as part of `pixi run lint`. Every
function definition in the repository — tests and scripts included — is
annotated in both parameters and return type. `strict` is not negotiable for the
package.
