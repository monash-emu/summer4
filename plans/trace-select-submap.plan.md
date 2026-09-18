---
name: Trace.select submap gather
overview: Replace Trace.select's current "zero non-matching rows, keep full map" behaviour with a real gather onto a smaller sub-PropertyMap, by adding a small PropertyMap.take(idx) primitive and reusing it in both the Trace query surface and the SavePlan-time Compartments(where=) save path.
todos:
  - id: take
    content: Add PropertyMap.take(idx) in src/summer4/propertymap.py + tests in tests/test_propertymap.py
    status: pending
  - id: trace-select
    content: Rewrite Trace.select in src/summer4/results/trace.py to gather via pmap.take(idx)
    status: pending
  - id: eval-fix
    content: Rewrite Compartments(where=, sum_over=) in src/summer4/results/eval.py to gather via pmap.take instead of masking
    status: pending
  - id: tests
    content: Add/update tests in tests/test_results.py for shrink-on-select, partition-after-select, and where= save-path map-awareness
    status: pending
  - id: notebook
    content: Extend examples/notebooks/04-running-and-results.ipynb (+ docs/user copy) with a to_pandas() column-count demonstration
    status: pending
  - id: checks
    content: Run lint/format-check/check-notebooks/test/check-branch/coverage before requesting merge
    status: pending
isProject: false
---


# Trace.select: gather into a sub-PropertyMap instead of masking

## Background

`Trace.select` currently does (`src/summer4/results/trace.py:78-86`):

```python
def select(self, sel: Selector) -> Trace:
    """Zero compartments/edges where ``sel`` is not Kleene-true (keeps the map)."""
    mask = pmap.mask(sel)
    data = xp.where(mask, _as_array(self.values), 0)
    return self._with(values=PropertyData(pmap, data))
```

This keeps the full `pmap` and zeros non-matching rows. Consequences confirmed while researching this plan:

- `res["compartments"].select(state["I"]).to_pandas()` returns every original column (S/I/R x age), not just "I" — the reported issue.
- It's needlessly large: every downstream op (`sum_over`, `total`, `rolling`, `at_times`, `to_frame`) works over the full-size array even for a tight selection.
- It's actually **wrong** for `partition`/`group_by` chained after `select`: both read indices from `self._pmap()`, which is still the *original, unrestricted* map, so `select(state["I"]).partition(age)` returns per-age index arrays that include the (zeroed) S/R rows too, not just I. Gathering to a real sub-map fixes this as a side effect.
- The same masking pattern exists at save time in `results/eval.py`'s `Compartments(where=, sum_over=)` branch (mask-then-`segment_sum`), and `Compartments(where=X, sum_over=None)` currently drops the map entirely — it returns a bare gathered array, not a `PropertyData`, so a saved trace with `where=` can't call `.select()`/`.sum_over()`/`.partition()` afterward.

`PropertyMap` already has everything needed to build a true sub-map with zero new fields, confirmed from `src/summer4/propertymap.py:97-146,209-233`:

- `parent_row: NDArray[np.int32] | None` already means "row of the *immediately previous* map" (one hop) — exactly what `_apply_stratification` sets it to (`parent_row=row_src`, indices into `self`, not further back).
- `__post_init__` only checks `parent_row.shape == (codes.shape[0],)` — no requirement that it be surjective, sorted, or same-length as the parent, so a shrinking subset is a valid `PropertyMap`.
- `__hash__`/`__eq__`/`_digest_bytes` (lines 50-58, 410-426) are driven by `codes` + `parent_row`, both of which naturally differ across different selections, so a sub-map hashes/compares correctly with no extra code.
- `PropertyMap.select(sel) -> NDArray[np.int32]` (line 240) already returns the gather indices; only the *consumer* needs to change.

## Decisions (confirmed with user)

1. Add `PropertyMap.take(idx)` as a small **public** method on `PropertyMap`, not a private/inline construction. It is reused by both `Trace.select` and the `results/eval.py` save path.
2. Fix **both** `Trace.select` and the `results/eval.py` `Compartments(where=, sum_over=)` (and `where=`-only) save-time path, so the two code paths share one gather primitive instead of two divergent (and one incorrect) implementations.

Edge/`FlowMass` traces stay out of scope: `_apply_where` for `FlowMass` already gathers (not masks), and edge-side map-awareness (`EdgeMap.table`, `side=` polarity) is explicitly Phase 4 per the existing boundary comment in `src/summer4/results/plan.py`. `Trace.select` continues to raise `TypeError` for non-`PropertyData` (edge) traces.

## Changes

### 1. `src/summer4/propertymap.py` — add `PropertyMap.take`

```python
def take(self, idx: NDArray[np.int32]) -> PropertyMap:
    """Return a sub-map containing only rows ``idx`` (gather, not stratify).

    ``idx`` indexes rows of this map. ``parent_row`` on the result is ``idx``
    itself — one hop back to this map — matching the convention used by
    :meth:`stratify`. Pair with the same ``idx`` to gather an aligned data array.
    """
    idx = np.asarray(idx, dtype=np.int32)
    return PropertyMap(
        properties=self.properties,
        codes=self.codes[idx],
        history=self.history,
        parent_row=idx,
    )
```

No other `PropertyMap` code changes — `to_frame`, `labels`, `partition`, `group_by`, `kleene`/`select` on the result all already work generically off `properties`/`codes`.

### 2. `src/summer4/results/trace.py` — `Trace.select` gathers

```python
def select(self, sel: Selector) -> Trace:
    """Return a Trace restricted to compartments/edges where ``sel`` is Kleene-true."""
    pmap = self._pmap()
    if pmap is None:
        raise TypeError("select() requires PropertyData values.")
    idx = pmap.select(sel)
    submap = pmap.take(idx)
    data = _as_array(self.values)[..., idx]
    return self._with(values=PropertyData(submap, data))
```

`idx` is computed host-side and static (matches the file's existing "static index / traced gather" convention used by `at_times`/`reduce_by`). `dims` is unchanged (`"compartment"` axis, just smaller). Because `_pmap()` now returns the sub-map, `sum_over`, `partition`, `group_by`, and `to_frame`'s label lookup all automatically operate over the restricted set with no further changes.

### 3. `src/summer4/results/eval.py` — share the same gather

Rewrite the `Compartments` arm of `eval_quantity` (currently lines 70-86) to gather via `pmap.take`, for both the `where`-only and `where + sum_over` cases:

```python
case Compartments(where=where, sum_over=sum_over):
    y = ctx.y
    from summer4.jax.state import State
    if isinstance(y, State):
        y = y.compartments
    data = y.data if isinstance(y, PropertyData) else jnp.asarray(y)
    active = pmap
    if where is not None:
        idx = where if isinstance(where, np.ndarray) else pmap.select(where)
        active = pmap.take(idx)
        data = data[..., idx]
    pd = PropertyData(active, data)
    return pd.sum_over(sum_over) if sum_over is not None else pd
```

This drops the `pmap.mask(where)` + `jnp.where(..., 0)` zeroing entirely, and makes `Compartments(where=X)` (no `sum_over`) return a proper `PropertyData` on the restricted sub-map instead of a bare array — so a saved, `where`-restricted trace remains queryable (`.select`, `.sum_over`, `.partition`) downstream. `_apply_where` stays as-is for the `FlowMass` arm (unaffected, out of scope).

### 4. Tests

- `tests/test_propertymap.py`: new tests for `PropertyMap.take` — correct rows/codes, hash/eq differs by subset, composes with `stratify`/`partition`/`group_by`/`labels`/`to_frame` on the sub-map.
- `tests/test_results.py`:
  - Update `test_query_select_sum_between_dates` (and similar) to assert the selected trace's last axis shrinks to the expected count (e.g. 3 age bands for `state["I"]`), not the full 9.
  - New `test_select_gathers_not_masks`: assert `to_frame()`/`to_pandas()` column count equals the number of selected compartments, and values match a manual NumPy reference gather.
  - New `test_select_then_partition_is_restricted`: catch the previously-latent bug — `select(state["I"]).partition(age)` must only see "I" rows.
  - New `test_compartments_where_in_saveplan_stays_map_aware`: a `SaveRequest(Compartments(where=state["I"]))` plan produces a `Trace` whose `.select()`/`.sum_over()` still work post-save.

### 5. Notebook

Extend `examples/notebooks/04-running-and-results.ipynb` (and its docs copy `docs/user/09-running-and-results.ipynb`) with a cell that calls `res["compartments"].select(state["I"]).to_pandas()` and asserts `list(pdf.columns)` contains only the "I" compartments — directly demonstrating the fix, per the "extend an existing notebook for a small addition" rule in `AGENTS.md`.

### 6. Docs / ledger

- Update the `Trace.select` docstring/any prose that describes the old "zeros, keeps full map" behaviour (grep confirmed no user-guide prose currently advertises the masking as a feature, so this is docstring-only plus the notebook cell above).
- No `docs/evaluation/coverage-ledger.md` status changes expected (D2-D5 stay `full`); confirm no ledger notes text references the old masking behaviour and update if so.

## Branch

Cut from `feat/results` (not `main` — `Trace`/`PropertyMap` don't exist on `main` yet):

```bash
git fetch origin
git switch -c feat/trace-select-submap origin/feat/results
```

Copy this plan to `plans/trace-select-submap.plan.md` on the branch once implementation starts, per the repo's plan-tracking convention.

## Verification

```bash
pixi run lint
pixi run format-check
pixi run check-notebooks
pixi run test
pixi run check-branch
pixi run coverage
```
