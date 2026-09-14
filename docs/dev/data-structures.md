# Data structures

Everything in the taxonomy layer reduces to four arrays and two dictionaries.

## `PropertyMap`

```python
@dataclass(frozen=True, slots=True, eq=False)
class PropertyMap:
    properties: tuple[Property, ...]
    codes: NDArray[np.int16]
    history: tuple[Stratification, ...] = ()
    parent_row: NDArray[np.int32] | None = None
    # _cache, _prop_index, _digest built in __post_init__
```

### `codes` — the compartment table

Shape `(n_compartments, n_properties)`, dtype `int16`.

| Value | Meaning |
|---|---|
| `0 .. len(traits)-1` | the trait code, in property declaration order |
| `-1` | the property does not apply to this compartment |

`int16` caps a single property at 32 767 traits, which is far above anything a
compartmental model needs, and halves the table's memory against `int32`. The
array is made C-contiguous and has `writeable = False` set in `__post_init__`,
so a map handed to a caller cannot be corrupted in place.

### `parent_row` — provenance

Shape `(n_compartments,)`, dtype `int32`. `parent_row[i]` is the row of the map
that this map was stratified from. `None` on a map built by `from_property`.

This is what makes redistribution cheap. Splitting a coarse population across a
new stratification is `coarse[parent_row] / bincount(parent_row)[parent_row]` —
a gather and a divide, with no dictionary lookup and no string parsing.

### `history` — the build log

A tuple of `Stratification(property, where)` values, in application order.
Because `Stratification` is a frozen dataclass holding a `Property` and an
optional `Selector`, history is fully serialisable data and can be replayed onto
another map with `step.apply(pmap)`.

### `_cache` — memoised queries

`dict[Selector, NDArray[np.int8]]`, populated by `kleene`. Frozen selectors
compare by value, so a structurally identical selector built elsewhere hits the
same entry. Cached arrays are frozen too, which is why `copy()` exists: it
returns the same table with an empty cache when a long-lived map has accumulated
many one-off queries.

### `_prop_index` / `_digest` — lookup and hashing

`_prop_index` is `dict[str, int]`. Properties are identified by **name**, not
object identity; this is what lets `pmap.partition("age")` work and what makes a
`Property` rebuilt in a helper function interchangeable with the original.
Public accessors `column_index`, `column`, `kleene`, and `label` wrap the table
without exposing these fields. `_digest` is a blake2b-16 of `codes` and
`parent_row`; `__hash__` is `hash((properties, history, _digest))`.

## `Property` and `Trait`

```python
@dataclass(frozen=True, slots=True)
class Property:
    name: str
    traits: tuple[str, ...]
    _index: dict[str, int]      # trait name -> code, built in __post_init__

@dataclass(frozen=True, slots=True)
class Trait(SelectorOps):
    property: str
    name: str
    code: int
```

`Property` names must be Python identifiers (so Phase 1 `@source` / `@dest`
column mangling cannot collide with a user property). `Trait` stores the
property **name**, not the `Property` object. That keeps traits cheap to compare
and hash, and keeps a selector tree free of references to large objects — but it
means a trait is only meaningful relative to a map that registers a property of
that name. `PropertyMap._validate_selector` closes the loop by checking that the
trait's `code` still matches the registered property's index, so a renamed or
reordered property is caught rather than silently selecting the wrong
compartments.

## Selector nodes

All eleven node types are frozen, slotted dataclasses inheriting `SelectorOps`,
which supplies `&`, `|`, `~` and a `__bool__` that raises.

```python
type Selector = (
    Trait | IsIn | Present | Absent | Everything | Nothing | And | Or | Not
    | Source | Dest
)
```

`Source` and `Dest` are in the union so they compose with `&` / `|` / `~` under
`mypy --strict`. On a compartment map they raise
(`"Source()/Dest() select flow edges, not compartments"`); edge evaluation lands
with flows.

The union is a PEP 695 `type` alias, and evaluation is a `match` statement over
it, so adding a node type produces a `TypeError` at the two exhaustive match
sites rather than silently falling through.

## Memory profile

For a map of `n` compartments and `p` properties:

| Structure | Bytes |
|---|---|
| `codes` | `2 · n · p` |
| `parent_row` | `4 · n` |
| each cached query | `n` (int8) |

A 100 000-compartment, 5-property map is 1 MB of table plus 400 kB of parent
rows; each cached selector adds 100 kB. That is the budget the query cache is
spending, and the reason `copy()` is offered as a cache reset rather than the
cache being unbounded-but-weak.
