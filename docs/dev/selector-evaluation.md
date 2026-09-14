# Selector evaluation

Resolving a selector against a map is two passes: validate, then evaluate.

## Validation

`PropertyMap._validate_selector` walks the tree before any array work and
raises on:

- a non-selector object (`TypeError`);
- a property name not registered on this map (`KeyError`, listing the known
  names);
- a trait name not in the registered property (`KeyError`, listing the known
  traits);
- a `Trait` whose `code` disagrees with the registered property's index
  (`ValueError`) — the stale-trait guard.

Validation runs on every `mask` / `select` call, including cache hits on
sub-expressions, because it is cheap relative to the array pass and because its
error messages are the main debugging affordance of the layer.

## Evaluation

`_evaluate` maps each node onto one NumPy expression over an `int8` array of
length `n`:

| Node | Expression |
|---|---|
| `Trait` | `where(present, where(col == code, 1, -1), 0)` |
| `IsIn` | `where(present, where(isin(col, codes), 1, -1), 0)` |
| `Present` | `where(col != -1, 1, -1)` |
| `Absent` | `where(col == -1, 1, -1)` |
| `Everything` | `full(n, 1)` |
| `Nothing` | `full(n, -1)` |
| `And` | `minimum(left, right)` |
| `Or` | `maximum(left, right)` |
| `Not` | `negative(inner)` |

The last three lines are the reason the encoding is `{-1, 0, 1}` rather than
`{0, 1, 2}`. With true/unknown/false as `+1/0/-1`:

- Kleene conjunction is `min`,
- Kleene disjunction is `max`,
- Kleene negation is arithmetic negation.

No branching, no lookup tables, three NumPy ufuncs. `Present` and `Absent` never
emit `0`, which is what makes them the two-valued escape hatch from raggedness.

`mask` is `kleene == 1`; `select` is `flatnonzero(mask).astype(int32)`.

## Caching

`kleene` memoises on the **selector value**. Every node in a tree is cached
independently, so a shared sub-expression across several queries is evaluated
once per map:

```python
common = state["I"] & age["0-4"]
pmap.select(common & severity["mild"])
pmap.select(common & severity["severe"])   # 'common' is a cache hit
```

Cached arrays are frozen (`writeable = False`) before being stored, so a caller
that reaches into the cache cannot poison it.

## Complexity

For `n` compartments and a tree of `k` nodes, a cold evaluation is `O(n · k)`
time and `O(n · k)` cached bytes; a warm one is a dictionary lookup. `IsIn` over
`m` traits uses `np.isin`, which is `O(n · m)` for small `m` and sorts for large
`m` — preferring `age[("0-4", "5-9")]` over `age["0-4"] | age["5-9"]` therefore
saves one full `int8` pass plus a `maximum`, which is what
{doc}`performance` measures.

## Stratification

`_apply_stratification` is the only operation that changes the table's shape:

1. Resolve `where` to a boolean `matched` array (all-true when `where is None`).
2. `reps = where(matched, k, 1)` — matched rows expand into `k` rows, others
   into one.
3. `row_src = repeat(arange(n), reps)` — this becomes the new `parent_row`.
4. `gathered = codes[row_src]` — existing columns, duplicated.
5. New column: `-1` everywhere, then `tile(arange(k), n_matched)` written into
   the matched positions.
6. Concatenate and construct a new frozen map.

It is a single gather plus a concatenate: no Python loop over compartments, and
the cost is linear in the size of the **output** table.

Because matched rows expand into a contiguous block, the new trait varies
fastest and each parent's children stay adjacent. That is what keeps
`partition` and `group_by` results contiguous for the common case, and it is an
ordering guarantee the solver layer will depend on.
