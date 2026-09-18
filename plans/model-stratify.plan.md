# Stratify a `FlowModel` in place, with adjustment precedence

## Context

Today a model must be built **map first, then flows**. `PropertyMap.stratify` returns
a new map, and `FlowModel(pmap)` holds that map once. To stratify an existing
model (by age, by Erlang stages of E, by strain), users have to re-declare every flow.
The summer2 port `docs/summer2/06-stratification-introduction.ipynb` repeats the same
flow declarations in every cell for this reason.

This branch adds, on `FlowModel` only (`EpiModel` is removed by the previous branch,
`plans/remove-epimodel.plan.md`):

1. `FlowModel.stratify(prop, where=None)`: **stratifies in place**, like summer2's
   `stratify_with`. The model's map is replaced, and all declared flows and the
   initial population stay.
2. `FlowModel.copy()` to branch a model before stratifying it different ways.
3. `FlowModel.update_flow(name, **changes)` and `FlowModel.adjust_flow(name, *adjustments)`
   to edit declared flows after stratifying.
4. **Adjustment precedence levels**: `Overwrite` → `Multiply` → `Transform`, with an
   explicit `precedence=` override. Overwrites in one level that overlap raise an
   error.
5. **Edge-level adjustment masks**: `where=Dest(...)` and `where=Source(...)`. An
   adjustment that can never fire raises instead of silently doing nothing.

Ledger IDs: S2, A1, A3. There are no status changes; Notes change only.

Read `AGENTS.md` first. Read `docs/evaluation/coverage-ledger.md` rows S1–S8 and
A1–A4.

## Why flows already survive stratification (background, verified)

Flows are stored as **declarations**: selectors, rate trees, `pairing`, `split` and
`adjust`. Nothing is bound to row indices until `FlowModel.compile()` calls
`actualize(flow, self.pmap)` (`src/summer4/flows/compiled.py`, in `FlowModel.compile`).
So compiling the same declarations against a stratified map already reproduces
summer2's behaviour. The join is `identity_join` in `src/summer4/flows/join.py`:

| New property ends up on… | Join role | Result on the new map |
|---|---|---|
| source and dest (plain `stratify`) | free key | one copy of the flow per stratum, same rate |
| dest only (`where=` hits only the dest) | `dest_only` | equal split over new strata; `split=` overrides |
| source only | `src_only` | flow leaves every stratum at the same rate |
| entry flow's dest | `dest_only` | equal split (`_entry_indices` in `actualize.py`) |

`EdgeMap.moves_mask` (`src/summer4/flows/edges.py`) counts a move only when the
property is present at **both** ends, so strict pairing accepts the dest-only case.
`FlowRef` refers to flows by name. `InitialPopulation` compiles against the map at
`compile()` and splits evenly over properties it isn't told about.
`ForceOfInfection` groups by one property; after stratifying by another it sums over
it, which is homogeneous mixing across the new strata. **None of these need changes.**

The work is the API, flow editing, and the adjustment semantics below.

## Part A — API on `FlowModel` (`src/summer4/flows/compiled.py`, class `FlowModel`)

Add these methods. Each needs full type annotations (mypy `--strict`) and a
Google-style docstring.

```python
def stratify(self, prop: Property, where: Selector | None = None) -> None:
    """Stratify this model's map in place; declared flows re-resolve at compile()."""
    self.pmap = self.pmap.stratify(prop, where)

def copy(self) -> FlowModel:
    """Return an independent builder with the same map, flows and initial population."""
    other = FlowModel(self.pmap)
    other.flows = list(self.flows)          # declarations are frozen; list is new
    other._initial_population = self._initial_population
    return other

def update_flow(self, name: str, **changes: object) -> FlowRef:
    """Replace fields of a declared flow (source, dest, rate, split, pairing, adjust, ...)."""

def adjust_flow(self, name: str, *adjustments: object) -> FlowRef:
    """Append adjustments to a declared flow; precedence decides evaluation order."""
```

Implementation notes:
- Add a private `_index_of(name) -> int` that raises
  `KeyError(f"Unknown flow {name!r}. Known: {...}")`.
- `update_flow`:
  - `"name" in changes` → `ValueError("Flow names cannot be changed.")`.
  - Allowed keys are `{f.name for f in dataclasses.fields(flow)}`. Any other key →
    `TypeError` listing the allowed keys.
  - Build the new flow with `dataclasses.replace(flow, **changes)`. This calls the
    custom `__init__` of `TransitionFlow` / `ExitFlow` / `EntryFlow`
    (`src/summer4/flows/types.py`), which normalises `rate`, `split` and `adjust`.
    `as_rate`, `_normalize_split` and `_normalize_adjust` all accept values that are
    already normalised.
  - Replace `self.flows[i]` and return `FlowRef(name)`.
  - Write a test that `replace` works for all three flow types before relying on it.
- `adjust_flow`: with no adjustments → `ValueError`. Otherwise
  `return self.update_flow(name, adjust=(*flow.adjust, *adjustments))`.
  `_normalize_adjust` wraps bare numbers as `Multiply`.
- `stratify` does **not** validate flows eagerly. Errors surface at `compile()`, the
  same as a map-first build. Say so in the docstring.
- A `CompiledModel` compiled **before** `stratify` is frozen and keeps the old map.
  Test this.
- Improve an error message, without changing behaviour. In `_align_rate`
  (`compiled.py`), the final `raise ValueError("Rate last axis ... matches neither pmap.size ...")`:
  when `pmap.parent_row is not None` and `shape[-1] == int(pmap.parent_row.max()) + 1`,
  append: `" It matches the size of the map this one was stratified from; lift it with arr[..., pmap.parent_row]."`
  Do not add any other shape heuristics.

Export nothing new from `summer4/__init__.py`; these are methods.

## Part B — Adjustment precedence (`src/summer4/flows/rates.py`, `actualize.py`)

### Semantics (decided with the user)

Adjustments form a chain evaluated by `_apply_adjustments` (`compiled.py`). For each
adjustment it computes `new` from the previous value and applies
`jnp.where(mask, new, prev)`. Today the order is declaration order. After this
branch the order is a **stable sort by `(level, declaration index)`**:

| Kind | Default level | Intent |
|---|---|---|
| `Overwrite` | 0 | set a base rate |
| `Multiply` | 1 | scale it |
| `Transform` | 2 | post-process (caps, additive terms) |

`precedence=<int>` on any adjustment overrides its level.

Rules inside one level:
- `Multiply`s commute, so their relative order doesn't matter.
- `Overwrite`s must have **pairwise disjoint masks**. If any edge is selected by two
  `Overwrite`s at the same level, `actualize` raises `ValueError`. The message names
  both adjustments (kind, value repr, `where` repr), lists up to 5 overlapping edges
  from `edge_map.labels()`, and suggests an explicit `precedence=` on one of them.
  A `None` mask means every edge.
- `Transform`s keep declaration order and are never checked.

What this means after stratifying. "Old" is an adjustment declared before stratifying
by `strain`; "new" is one added afterwards with `adjust_flow`.

| # | Old | New | Result where both apply |
|---|---|---|---|
| 1 | `Multiply(a, where=age["old"])` | `Multiply(b, where=strain["b"])` | `rate·a·b` |
| 2 | `Overwrite(0, where=age["0-4"])` | `Multiply(b, where=strain["b"])` | `0` |
| 3 | `Multiply(a, where=age["old"])` | `Overwrite(β, where=strain["b"])` | `β·a`. The overwrite is level 0 and runs first. summer2 would give `β`; users wanting that pass `precedence=2`. |
| 4 | `Overwrite(0, where=age["0-4"])` | `Overwrite(β, where=strain["b"])` | **raises** (same-level overlap) |
| 5 | any | `Transform(fn, x, where=strain["b"])` | runs after all multiplies; order among transforms is declaration order |

Compatibility check already done: every existing `Overwrite` in the repo is alone in its
chain or disjoint from others (`tests/test_flows.py:527`, `docs/user/08-flows.ipynb`,
`examples/notebooks/03-flows.ipynb`, `docs/summer2/10-derived-outputs-stratified.ipynb`,
`docs/summer2/11-flows-between-strata.ipynb`, which uses disjoint age overwrites). So no
current result should change. If a test or notebook does change, stop and report it.
Do not adjust the expected value.

### Implementation
1. `rates.py`:
   - Add `precedence: int | None = None` as a field on `Multiply`, `Overwrite` and
     `Transform`, and a keyword argument in each `__init__`. For `Transform` it goes
     after `where`:
     `def __init__(self, fn, *args, where=None, precedence=None)`.
   - Add
     ```python
     def adjustment_level(adj: Adjustment) -> int:
         if adj.precedence is not None:
             return adj.precedence
         return 0 if isinstance(adj, Overwrite) else 1 if isinstance(adj, Multiply) else 2

     def canonical_adjustments(adjust: tuple[Adjustment, ...]) -> tuple[Adjustment, ...]:
         order = sorted(range(len(adjust)), key=lambda i: (adjustment_level(adjust[i]), i))
         return tuple(adjust[i] for i in order)
     ```
   - `_adjust_bytes` needs no change: the canonical order is hashed through the stored
     chain.
2. `actualize.py`, `actualize()`: for each of the three flow kinds, compute
   `adjust = canonical_adjustments(flow.adjust)`. Store `adjust=adjust` on the
   `*Edges` object and bind masks from that same tuple (Part C). Then call
   `_check_overwrite_overlap(adjust, masks, edge_map)`, a new private function in
   `actualize.py`.
3. Nothing downstream changes. `_apply_adjustments`, `_flow_digest`, `roots_of`
   (`stages.py`), `_flow_paths` and `_rate_flow_deps` all iterate the stored
   `flow.adjust`, which is now canonical.

## Part C — Edge-level masks with `Source` / `Dest` (`src/summer4/flows/actualize.py`)

Today `_bind_adjust_masks(adjust, gather_idx, pmap)` computes
`np.isin(gather_idx, pmap.select(where))`. The gathered rows are the **source** for
transition and exit flows and the **dest** for entry flows. A `where=` over a
property that exists only on the destination (e.g. `severity` added with
`where=state["I"]`, on an `S → I` flow) is therefore all-false and **silently never
fires**.

Replace it with a function that evaluates on the flow's `EdgeMap`, which already
supports `Source(...)`/`Dest(...)`: `EdgeMap.mask(sel)` in `edges.py`.

```python
def _bind_adjust_masks(
    adjust: tuple[Adjustment, ...],
    edge_map: EdgeMap,
    default_side: type[Source] | type[Dest],
) -> tuple[NDArray[np.bool_] | None, ...]:
```

For each adjustment:
- `where is None` → `None`.
- If `where` contains no `Source`/`Dest` node anywhere (write a small recursive
  `_has_polarity(sel)` over `And`/`Or`/`Not`), wrap it: `sel = default_side(where)`.
  Otherwise use `where` as-is. A mixed selector such as `age["old"] & Dest(x)` makes
  `EdgeMap._validate_edge_selector` raise `TypeError` with a clear message. That is
  correct; don't catch it.
- If the flow has no destination (`edge_map.dest_idx is None`) and `sel` contains
  `Dest`, raise `ValueError("Dest(...) in where= on a flow without a destination")`.
  Do the same for `Source` when `edge_map.src_idx is None`.
- `mask = edge_map.mask(sel)`, then freeze it (`mask.flags.writeable = False`).
- **Dead-mask check:** if `not mask.any()`, and some property named inside a
  `Source(inner)` is absent on every edge's source (all `-1` in column `f"{p}@source"`
  of `edge_map.table`), or likewise a property inside `Dest(inner)` on the destination,
  raise:
  `ValueError(f"Adjustment where={adj.where!r} can never apply on flow {name!r}: property {p!r} is absent on the {side} of every edge. Use Dest(...)/Source(...) or split=.")`.
  Use `selector_properties` (`join.py`) on `inner`. An all-false mask whose
  properties *are* present (e.g. a trait that no edge happens to carry) is legal. Do
  not raise for it.

Call sites in `actualize()`:
- transitions and exits use `default_side=Source`;
- entries use `default_side=Dest`;
- pass the `edge_map` built for that flow (`edges.edge_map` for transitions, the local
  `edge_map` for exits and entries). Add a `name` argument, or pass `flow.name`, for
  the message.

A bare selector gives a mask **identical** to the old
`isin(gather_idx, pmap.select(...))`. `Source(sel)` is renamed to the `@source` columns,
which hold exactly `pmap.codes[src_idx]`. A test pins this (below).

`Source` and `Dest` are already exported from `summer4` (see the imports in
`docs/user/08-flows.ipynb`), so no export change is needed.

## Part D — Tests

Every test function is fully type-annotated. Hypothesis strategies live in
`tests/test_taxonomy_properties.py` (`property_specs`, `built_maps`). Move them into a
new `tests/helpers/strategies.py` and import them from there in both files
(`from tests.helpers.strategies import built_maps`), so they can be reused.

### `tests/test_model_stratify.py`
1. `test_stratify_commutes_with_flow_declaration`: Hypothesis, `max_examples=40`.
   - Draw a map with `built_maps()`, a fresh property `q` (name not already on the
     map), and optionally `where = <first property>[<first trait>]`.
   - Build a flow set: an `ExitFlow` on `Everything()`, an `EntryFlow` into the first
     property's first trait, and a `TransitionFlow` between the first property's first
     and second traits. Give each an `adjust=` entry `Multiply(2.0, where=<second property>[<trait>])`.
   - Build A: `FlowModel(pmap)` + flows, then `A.stratify(q, where)`.
   - Build B: `FlowModel(pmap.stratify(q, where))` + the same flows.
   - Assert `A.compile() == B.compile()` (digest equality) and that each flow's
     `edges(name).table.codes` are equal.
   - Use `strict_pairing=False` in `compile` only if Hypothesis finds
     strict-pairing rejections that happen in both builds. Rejections in both builds
     are fine; assert that both raise the same error type.
2. `test_stratify_mutates_in_place_and_keeps_flows`: `pmap` changes, `flows` is the same
   list object, and the initial population is kept.
3. `test_compiled_before_stratify_is_unchanged`.
4. `test_copy_branches_independently`: `core.copy()`, then stratify each copy differently
   and add a flow to one; the other copy and `core` are unchanged.
5. `test_dest_only_equal_split_and_split_override`: SIR, then
   `stratify(severity, where=state["I"])`. The infection flow splits 50/50; after
   `update_flow("infection", split={severity: {...}})` the weights match.
6. `test_update_flow_errors`: unknown flow name → `KeyError`; unknown field →
   `TypeError`; `name=` → `ValueError`.
7. `test_update_flow_replace_all_kinds`: `dataclasses.replace` round-trips
   `TransitionFlow`, `ExitFlow` and `EntryFlow` with `adjust`, `split` and `pairing` set.
8. `test_flowref_sum_over_survives_stratify`: a rate using `FlowRef(...).sum()` compiles
   after stratify.
9. `test_initial_population_splits_evenly_over_new_property`.
10. `test_erlang_latent_stages`: SEIR, then `stratify(stage, where=state["E"])` with 3
    stages, then:
    - `update_flow("infection", dest=state["E"] & stage["e1"])`;
    - `update_flow("progression", source=state["E"] & stage["e3"])` and
      `adjust_flow("progression", 3.0)`;
    - `add_flow(TransitionFlow("latent_stages", state["E"], state["E"], SIGMA, pairing=TraitChain(stage, (("e1","e2"),("e2","e3"))), adjust=[3.0]))`.

    Assert that compiled equals a map-first build of the same model. Also run both for
    a short Euler horizon and assert the trajectories are equal.
11. `test_align_rate_error_mentions_parent_row`.

### `tests/test_adjust_precedence.py`
1. `test_default_levels_sort`: the canonical order of
   `[Transform, Multiply, Overwrite, Multiply]` is `Overwrite, Multiply, Multiply, Transform`.
   Declaration order is kept within a level.
2. `test_case3_overwrite_before_multiply`: effective per-edge rate is `β·a`. With
   `precedence=2` on the overwrite it is `β`. Compute effective rates through
   `CompiledModel.vector_field` on a state with 1.0 in each source compartment, or by
   evaluating `_apply_adjustments` on a unit rate. Pick one approach and use it
   throughout.
3. `test_case4_same_level_overwrite_overlap_raises`: the message contains an edge label.
   Disjoint overwrites (three different `age` traits) compile.
4. `test_multiply_order_irrelevant`: Hypothesis over permutations of 3 `Multiply` with
   random `where` masks. Rates are equal.
5. `test_transforms_keep_declaration_order`.
6. `test_bare_where_mask_identical_to_legacy`: for every flow in a stratified SIR
   (transition, exit, entry), the new mask equals
   `np.isin(gather_idx, pmap.select(where))`.
7. `test_dest_where_fires_on_dest_only_property`: `S → I` with severity on `I` only;
   `Multiply(2.0, where=Dest(severity["severe"]))` doubles exactly the severe edges.
8. `test_dead_where_raises`: the bare `where=severity["severe"]` on that flow raises and
   the message mentions `Dest(`. `Dest(...)` on an `ExitFlow` raises. `Source(...)` on
   an `EntryFlow` raises.
9. `test_absent_selector_is_not_dead`: `where=Absent(severity)` on the source side
   compiles and selects every edge.

### Existing tests
No existing test should be edited. If one fails, that is a finding: report it.
Pay attention to `tests/test_flows.py` (`test_adjust_*`, ~lines 500–580) and
`tests/test_hoisting.py` (`Transform` at ~line 114).

## Part E — Example notebook (required by `AGENTS.md`)

`examples/notebooks/12-model-stratification.ipynb`. Plain Python, cleared outputs,
assertions on every claim.
1. Build an SEIR `FlowModel` on a `state × pop` map, with an infection
   `ForceOfInfection` flow, progression and recovery. Run it and keep the total `I`
   trajectory.
2. `aged = core.copy(); aged.stratify(age)`. Use equal initial splits and no mixing
   matrix change. Assert that the aggregate `I` trajectory matches the core model
   (`np.testing.assert_allclose`).
3. Add an age adjustment, then `stratify(strain)`, then
   `adjust_flow("infection", Overwrite(beta_b, where=strain["b"]))`. Show that old
   strain-b edges get `beta_b · age multiplier` (case 3), and that `precedence=2`
   instead gives `beta_b`.
4. Show the overlap error with `try/except ValueError` and assert on the message.
5. `erlang = core.copy()`, then the Erlang steps from test 10. Assert the mean latent
   duration is unchanged and the E-residence distribution is narrower: compare the
   variance of time-to-I for a cohort, or simply compare peak timing.
6. `Dest(...)` adjustment on a severity-on-`I` stratification.

## Part F — Documentation

| File | Change |
|---|---|
| `docs/user/08-flows.ipynb` | Retitle "A map, then a model" to "Maps and models". Map-first stays the main path. Add a section "Stratifying a model": in place, `copy()` to branch, `update_flow`, `adjust_flow`. Add "Adjustment order": the level table, `precedence=`, the overlap error, `Source`/`Dest` in `where=` vs `split=`. Every claim gets an `assert`. |
| `docs/user/06-immutability-and-provenance.ipynb` | Add a short cell: maps are immutable, but `FlowModel` is a mutable builder. Use `copy()` to branch it. |
| `docs/user/07-from-summer2.md` § Stratifying | Replace "Note that `stratify` returns a new map — summer2's `stratify_with` mutates" with: `PropertyMap.stratify` returns a new map; `FlowModel.stratify` mutates the model like `stratify_with`. Add a subsection on adjustments: summer2 applies them in stratification order, while summer4 orders them by precedence level; pass `precedence=` to reproduce "later overwrite wins". |
| `docs/dev/architecture.md` (~line 53) | One paragraph: joins are late-bound at `compile()`, which is why `FlowModel.stratify` needs no flow rewriting. Adjustments are canonicalised by level in `actualize`. |
| `docs/summer2/06-stratification-introduction.ipynb` | Rebuild the repeated per-cell models with one base model + `copy()` + `stratify(...)` + `adjust_flow(...)`, mirroring the original summer2 page. Keep all assertions. |
| `docs/evaluation/coverage-ledger.md` | Notes only. S2: "`FlowModel.stratify` (in place, like summer2) or `PropertyMap.stratify` (new map)". A1: append "; `adjust_flow` after `stratify`; `Source`/`Dest` where". A3: append "; precedence levels (summer2 applies in stratification order)". Then run `pixi run coverage-write` and `pixi run coverage`; totals are unchanged. |
| `futureplans/foi-multi-property-mixing.md` (new) | summer2 combines mixing matrices across stratifications (Kronecker). summer4's `ForceOfInfection.group_by` (`src/summer4/epi/infection.py`) takes one property, so after `stratify` mixing across the new property is homogeneous. "Done" = `group_by` accepts several properties with a product mixing matrix. Link it from `futureplans/README.md`. |

## Order of work and commits

1. Part B + Part C + `tests/test_adjust_precedence.py`. Run the full `pixi run test`
   now. This is where existing behaviour could change.
2. Part A + `tests/test_model_stratify.py`.
3. Part E notebook.
4. Part F docs + ledger + futureplans.

Each step is one commit. The PR description quotes S2, A1 and A3.

## Required checks
```bash
pixi run lint
pixi run format-check
pixi run check-notebooks
pixi run test
pixi run check-branch
pixi run coverage
pixi run -e docs docs-strict
```

**Manual gate:** the user signs off on `examples/notebooks/12-model-stratification.ipynb`
and the rewritten `docs/summer2/06-stratification-introduction.ipynb` before merge.

## Out of scope
- Any change to how adjustments are evaluated at run time beyond the order. The
  per-edge `jnp.where` fold stays. Evaluating over unique value combinations is
  tracked separately in `futureplans/adjustment-expanding-arrays.md`.
- Multi-property force-of-infection grouping (futureplans note only).
- Removing flows from a model.
