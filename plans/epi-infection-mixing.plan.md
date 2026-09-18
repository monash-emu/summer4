---
name: epi-infection-mixing
description: WP6 — grouped rate values and state reductions in core, then a summer4.epi submodule with mixing matrices, frequency/density/custom force of infection, infectiousness adjustments, multi-strain and multi-disease models, and an EpiModel frontend.
---

# WP6 — Force of infection and mixing (`feat/grouped-rates` … `feat/epi-model`)

## Context

This is the largest genuinely unprototyped design problem left in the roadmap.
A force of infection is **a reduction over a grouping, fed back into a rate**,
and a mixing matrix **weights that coupling between strata**. `TraitMatrix` does
not do this: it moves people, not transmission.

`docs/case-studies/age-stratified-seirs.ipynb` writes the whole thing by hand in
about six lines, and three of those six are traps:

```python
def derived_fn(params, *, y, t):
    pd_y = PropertyData(pmap, y)
    n_by_age = pd_y.sum_over(age).data
    # `where` REPLACES what it matches, so `~state["I"]` is what KEEPS the
    # infectious ones. Reading it the natural way silently gives a force of
    # infection over S + E + R.
    i_by_age = pd_y.where(~state["I"], 0.0).sum_over(age).data
    shedding = (jnp.asarray(params["infectiousness"]) * i_by_age) / n_by_age
    foi_by_age = params["contact_rate"] * (jnp.asarray(params["mixing"]) @ shedding)
    return Derived(foi=pd_y.broadcast_over(age, foi_by_age))
```

The three traps: the `where` polarity
(`futureplans/propertydata-where-polarity.md`, which cost that case study a full
round of *wrong* identifiability analysis that survived into a written plan
before it was caught); the mandatory `sum_over → broadcast_over` round trip,
because a bare per-trait vector is silently misread as a per-edge rate whenever
the edge count happens to match; and the frequency-versus-density choice, which
is invisibly encoded as "is `/ n_by_age` present or not".

**This package is worth more than its four ledger rows.** The same case study
showed that under a homogeneous mixing matrix the force of infection is
*identical across age bands* — maximum difference over all times: **0.0** — so
age-specific infectiousness is unidentifiable in principle. The fit recovered an
*inverted* age gradient at a loss slightly *below* the loss at the generating
parameters. A row-normalised assortative matrix made $\lambda_a$ band-specific
and the gradient identifiable again. Mixing matrices are not a convenience over
hand-written coupling; without off-diagonal structure a whole class of parameters
cannot be estimated at all.

**Closes:** `F7` `F8` `A4` `M1` → `full` (WP6). 40 → **44 / 52 (85%)**.
**Unblocks:** textbook 12, 13, 14, 15; summer2
`06-stratification-introduction`, `08-strain-stratification`,
`09-mixing-matrices`.
**Depends on:** WP5 (`plans/time-varying.plan.md`). 5.2 threads `t` through the
rate evaluator; 6.2 extends the same seam to thread `y`. Start from 40 / 52.

### The separation rule

> Anything epidemiological lives in `summer4.epi`. Nothing in `summer4.flows`,
> `summer4.results` or `summer4.jax` may mention infection, transmission,
> susceptibility, a pathogen or a strain.
>
> 6.1 and 6.2 are **core** changes anyway, because "a value per grouping" and "a
> reduction over the state" are not epidemiological ideas — they are the generic
> machinery that was missing and that made the case study hand-write JAX. 6.3
> onwards is `summer4.epi`.

### Subphases

| # | Branch | Layer | Ships | Ledger |
|---|---|---|---|---|
| 6.1 | `feat/grouped-rates` | core | `GroupedRate` with arithmetic | — |
| 6.2 | `feat/state-reductions` | core | `Reduce` rate node | — |
| 6.3 | `feat/epi-foi` | `summer4.epi` | `MixingMatrix`, `ForceOfInfection` | `F7` `F8` `M1`, 40 → 43 |
| 6.4 | `feat/epi-infectiousness` | `summer4.epi` | infectiousness weights | `A4`, 43 → 44 |
| 6.5 | `feat/epi-multistrain` | `summer4.epi` | multi-strain, multi-disease | — |
| 6.6 | `feat/epi-model` | `summer4.epi` | `EpiModel` frontend | — |

---

## Before you start — read this even if you have read it before

Read, in this order: `AGENTS.md`, `docs/evaluation/coverage-ledger.md`
(especially *Delivery status*), and every note in `futureplans/`. Then read
`docs/case-studies/age-stratified-seirs.ipynb` and
`docs/evaluation/age-stratified-seirs-case-study.md` — they are the worked
example this package replaces, and the evaluation names the exact friction each
subphase is meant to remove.

**One subphase is one branch.** Branch from the tip of the previous subphase, and
the first from the tip of WP5. Not from `main` — `main` carries only the taxonomy
and cannot `import summer4.flows`. Copy this plan file onto your branch as
`plans/epi-infection-mixing.plan.md` before you touch `src/`.

`check-branch` fails unless a branch changing any `src/summer4/*.py` **also**
touches `tests/` **and** `examples/notebooks/`. Unconditionally. 6.1 and 6.2 have
no notebook of their own; they extend `examples/notebooks/09-epi-models.ipynb`,
which 6.1 creates as a stub and 6.6 finishes.

Finish every subphase with, in order:

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch
pixi run coverage
pixi run -e docs docs-strict
pixi run test-all
```

**Ledger discipline, in the same commit as the code.** Change the affected rows'
`Status`, run `pixi run coverage-write`, then update both totals quoted in
`docs/evaluation/index.md`. Note the asymmetry in this package: `F7` and `F8` are
`partial` today, so closing them raises the complete count but **not** the
covered count; `A4` and `M1` are `none`, so they raise both.

No new `Area` value is needed: `flows` (F7, F8), `adjustments` (A4) and `mixing`
(M1) all already exist in `AREA_TITLES` in `scripts/coverage_report.py`.
Inventing one is a code change and
`tests/test_coverage_ledger.py::test_every_area_is_named` will fail.

**Do not "fix" a row on the never-reaches-`full` list.** `S6`
(`AgeStratification`) and `S7` (`StrainStratification`) are summer2 *shapes*
summer4 has deliberately rejected, not capabilities it lacks. 6.5 makes `S7`'s
Notes cell out of date, and the instruction there is to fix the *note* and leave
the *status* alone.

**If you are blocked, record the truth.** Leave the row `partial`, name the real
blocker, leave `Ported` empty. If you find a problem you are not fixing, write a
`futureplans/<slug>.md` note and add it to that folder's README list.

**The notebook is a blocking human gate.** Do not merge a subphase until the user
has run its notebook in `pixi run notebook` and signed it off.

### The five-site rule

6.1 and 6.2 both touch the rate tree. A new `RateOps` subclass must be added to
**five** places or it fails silently:

| Site | File | Failure if missed |
|---|---|---|
| `_eval_rate` | `flows/compiled.py:109` | `TypeError: Unsupported rate expression` |
| `_flow_refs` | `flows/rates.py:213` | `case _` returns `set()`; topo-sort silently misorders flows |
| `_field_paths` | `flows/rates.py:239` | `case _` returns `set()`; `computed_paths` silently incomplete |
| `_rate_bytes` | `flows/rates.py:328` | fallback is the class name — **two distinct nodes collide in the jit cache** |
| `__all__` | `flows/__init__.py`, `summer4/__init__.py` | not importable |

---

## 6.1 — `feat/grouped-rates` (core)

### 6.1a. What exists, and what is wrong with it

`_SubmapRate` (`src/summer4/flows/compiled.py:53`) is the only "value per
grouping" carrier in the codebase:

```python
@dataclass(frozen=True, slots=True)
class _SubmapRate:
    """Rate whose last axis is aligned to one property's traits."""
    data: object
    properties: tuple[Property, ...]
```

Three limitations, each of which blocks a force of infection:

1. **Produced only by `FlowRef.sum_over`.** Nothing else can make one. A
   `derived_fn` return value goes through `_as_array` and hits the shape-matching
   ladder in `_align_rate` (`:193`) instead — which is exactly why the case study
   needs `broadcast_over` to inflate a 3-vector to 12 rows only for
   `_align_rate`'s `gather_idx` to compress it again.
2. **Arithmetic-lossy.** A `BinOp` combining a `_SubmapRate` with anything falls
   through to Python `*` / `+`, which works only by broadcasting accident and
   **drops the `properties` tag**. So `nu * I_by_age / N_by_age` cannot stay
   grouped, which is the entire body of a force of infection.
3. **Single-property.** `_align_submap_rate` (`:175`) raises at `:180` on
   `len(rate.properties) != 1`.

### 6.1b. The change

Promote it to a real value type, `GroupedRate`, with:

- **Arithmetic that preserves `properties`.** Implement `__mul__`, `__truediv__`,
  `__add__`, `__sub__` and their reflected forms. Two `GroupedRate`s combine only
  if their groupings are equal; a `GroupedRate` combined with a scalar keeps its
  grouping. Mismatched groupings raise with both groupings named — do not
  broadcast silently, which is precisely the failure mode being removed.
- **A matrix product against a matrix aligned to the grouping.** `M @ grouped`
  where `M` is `(n_traits, n_traits)`. This is what a mixing matrix *is*, and it
  belongs here rather than in `summer4.epi`, because weighting a grouped quantity
  by a square matrix is not an epidemiological idea.
- **Multi-property grouping.** Widen `_align_submap_rate` past the
  `len(properties) != 1` guard, reusing `PropertyMap.group_by(*props)`
  (`src/summer4/propertymap.py:279`) for the index arithmetic rather than writing
  new code. Keep the existing error for a gather row that lacks the property.

Decide and state in the PR whether `GroupedRate` is exported from `summer4` or
stays internal with only `Reduce` and the epi layer producing it. Recommended:
**export it**, because 6.3's custom-FOI callable receives and returns one, so it
is already part of the public contract.

### 6.1c. Tests — `tests/test_grouped_rates.py` (new)

- `grouped * scalar`, `scalar * grouped`, `grouped / grouped` all keep the
  grouping; assert on the `properties` tuple, not just the numbers.
- Two `GroupedRate`s over different properties raise on combination, and the
  message names both.
- `M @ grouped` against a hand-computed matrix product.
- A two-property grouping aligns correctly onto a ragged map where one property
  is absent on some compartments.
- The existing `FlowRef.sum_over` path still works unchanged — it is the only
  current producer and the regression risk is real. `tests/test_flows.py` already
  covers it; assert it still passes rather than rewriting it.

### 6.1d. Notebook

Create `examples/notebooks/09-epi-models.ipynb` as a stub with a short
"grouped values" section. 6.2 through 6.6 extend it; 6.6 rewrites the opening.

**Ledger discipline:** no rows change.

---

## 6.2 — `feat/state-reductions` (core)

### 6.2a. The change

A rate node that reduces the *state* over a grouping, so a force of infection
needs no `derived_fn` at all:

```python
Reduce(where=state["I"], sum_over=age)     # per-age infectious count
Reduce(sum_over=age)                       # per-age denominator
```

- Evaluates to a `GroupedRate` from 6.1.
- Requires threading `y_arr` into `_eval_rate`, alongside the `t` that WP5 §5.2
  threaded. `y_arr` is likewise **already in scope** at the top of
  `CompiledModel.observe` (`compiled.py:428`) — `unpack_state` produces it on the
  first line. The plumbing is the same one-line-per-frame change.
- Reuse `PropertyData.sum_over` (`src/summer4/jax/propertydata.py`) for the
  reduction. It is already a `jax.ops.segment_sum` under a `vmap` over flattened
  leading axes. Do not write a second segment-sum.

### 6.2b. `where=` means KEEP, and this is deliberate

`PropertyData.where(sel, other)` **replaces** what `sel` matches — pandas
`Series.mask` semantics. `Reduce(where=sel)` does the opposite: it **keeps** what
`sel` matches, which is how `select`, `sum_over` and `partition` already read,
and how every user expects it to read.

This is the single highest-value correction in the package. Requirements:

- The docstring's **first line** states the polarity.
- A test asserts `Reduce(where=sel, sum_over=p)` equals
  `pd.where(~sel, 0.0).sum_over(p)` numerically, so the relationship between the
  two polarities is pinned in code rather than in prose.
- Land the `PropertyData.keep(sel, other=0.0)` alias proposed in
  `futureplans/propertydata-where-polarity.md` here too, implemented as
  `where(~sel, other)`, with a test asserting the identity. Update that note —
  or delete it and its README bullet if `keep` plus `Reduce` fully discharge it.

### 6.2c. The five-site rule, applied

`Reduce` holds a `Selector` and a `Property`, neither of which is a `RateOps`, so
`_flow_refs` and `_field_paths` return empty sets — **write the cases explicitly**
rather than relying on `case _`. `_rate_bytes` must fold in a stable encoding of
the selector and the property name; do not use `repr` of the selector unless you
have checked it is stable across equal-but-rebuilt selectors.

### 6.2d. Tests — `tests/test_state_reductions.py` (new)

- **Headline gate:** a SIR whose force of infection is built from `Reduce` gives
  a trajectory matching the same model built with a hand-written `derived_fn`, to
  machine precision. If this does not hold, nothing downstream is trustworthy.
- The `keep` / `where` polarity identity above.
- `jax.make_jaxpr` shows one `segment_sum` per `Reduce`, not a Python loop.
- A ragged map where the grouping property is absent on some compartments
  reduces correctly and does not count the absent rows.
- A `Reduce` inside a `BinOp` with an `Interp` from WP5 — the two new node
  families compose.

### 6.2e. Notebook

Extend `examples/notebooks/09-epi-models.ipynb`: a hand-rolled frequency-dependent
force of infection written purely in the rate tree, with **no `derived_fn` at
all**, asserted against the `derived_fn` version. This is the page that shows the
core machinery before any epidemiology is layered on it.

**Ledger discipline:** no rows change. Say so; the temptation to move `F7`/`F8`
here is real and premature — a user still has to assemble the FOI by hand.

---

## 6.3 — `feat/epi-foi` (`summer4.epi`)

New package: `src/summer4/epi/__init__.py`, `epi/mixing.py`, `epi/infection.py`.

### 6.3a. `MixingMatrix` — `src/summer4/epi/mixing.py`

```python
MixingMatrix(age, refs.contacts, normalize="rows", check_reciprocal=True)
```

- **The matrix may be a `FieldRef`, so it stays a params key rather than a
  build-time constant.** This is the one thing the case study got right and it
  must be preserved: it made swapping a homogeneous matrix for an assortative one
  a genuine one-liner with no recompile, and that is what made the identifiability
  demonstration possible at all. A plain NumPy array is also accepted, and is
  then a `Const`-like static value.
- **Validate what was previously done by hand or skipped.** The case study
  asserted row sums by hand and explicitly did not check reciprocity:
  - shape against the trait count of the grouping property — at construction,
    whenever the matrix is static, and otherwise at compile;
  - `normalize="rows" | "none"`, with row sums checked or applied;
  - `check_reciprocal=True` verifies $K_{ab}N_a = K_{ba}N_b$. **This needs the
    population, so it is a run-time check behind a flag, not a construction-time
    one.** Say that in the docstring; a user who expects a constructor error will
    otherwise think it passed. Full empirical-matrix scaling is WP9 and out of
    scope — do not start it.

### 6.3b. `ForceOfInfection` — `src/summer4/epi/infection.py`

```python
ForceOfInfection(
    "infection",
    infectious=state["I"],
    group_by=age,
    mixing=contacts,            # MixingMatrix | None
    kind="frequency",           # | "density" | a callable
    contact_rate=refs.beta,
    denominator=None,           # default: everyone in the group
)
```

- Built from `Reduce` and `GroupedRate`; it evaluates to a `GroupedRate`, so it
  is usable **directly as a flow rate** with no `broadcast_over` round trip:

  ```python
  model.add_flow(TransitionFlow("infection", state["S"], state["E"], foi))
  ```

- `kind="frequency"` divides by the per-group denominator; `kind="density"` does
  not. Making this an explicit named argument closes the worst silent choice in
  the hand-written idiom, where the distinction was whether `/ n_by_age` was
  present. `F7` and `F8` become the same object with one argument different —
  say so in the ledger Notes cells rather than inventing two classes.
- **A callable `kind` is the custom-FOI path**, and it is what keeps generality.
  It receives the grouped infectious term and the grouped denominator and returns
  a `GroupedRate`. Test it by reimplementing `"frequency"` through it and
  asserting bit-identical output — that assertion is the proof that the built-in
  kinds are not privileged.
- `denominator` defaults to everything in the group; a selector narrows it (for
  models where only part of the population is mixing).

### 6.3c. Make the force of infection properly saveable

Today the only way to inspect a FOI is `ComputedValue(path=("foi",))`, which
returns the raw `pmap.size`-wide broadcast array with no property metadata, so
the case study re-slices it with `pmap.select(state["S"])` — an arbitrary state
row that happens to carry one copy per band.

Register each named `ForceOfInfection` as a saveable quantity returning a
**properly dimensioned per-trait trace**, so `res["foi_infection"]` has
`dims == ("time", "age")` and the whole `Trace` query surface applies. Follow how
`FlowMass` already produces a `PropertyData` over a group map in
`src/summer4/results/eval.py`.

### 6.3d. Tests — `tests/test_epi_foi.py` (new)

- Frequency and density FOI differ **exactly** by the denominator on a model
  whose population changes over time.
- A custom callable reimplementing `"frequency"` gives bit-identical output.
- **The identifiability gate.** Reproduce the case study result: under a
  homogeneous mixing matrix, assert the maximum difference in $\lambda_a$ between
  groups over all times is `0.0`; under a row-normalised assortative matrix,
  assert it is not. This test *is* the argument for the package, and it belongs in
  the suite rather than only in a notebook.
- A non-square matrix, or one whose size does not match the trait count, raises at
  construction with the sizes named.
- `check_reciprocal=True` raises on a deliberately non-reciprocal matrix and
  passes on a reciprocal one.
- A model built through `ForceOfInfection` matches the case study's hand-written
  `derived_fn` trajectory to machine precision.
- `jax.grad` of a loss with respect to `contact_rate` is finite and matches
  central differences.

### 6.3e. Notebook

Extend `examples/notebooks/09-epi-models.ipynb`: frequency versus density side by
side on the same model, and the homogeneous-versus-assortative $\lambda_a$
comparison plotted, with the max-difference assertion visible on the page.

**Ledger discipline:** `F7` (`add_infection_frequency_flow`) and `F8`
(`add_infection_density_flow`) → `full`, Notes naming
`ForceOfInfection(kind=...)`; `M1` (`set_mixing_matrix`) → `full`, Notes naming
`summer4.epi.MixingMatrix` and stating that `TraitMatrix` remains a different
thing that moves people. 40 → 43. `pixi run coverage-write`, then both totals in
`docs/evaluation/index.md` — `M1` was `none`, so the covered count moves too.

---

## 6.4 — `feat/epi-infectiousness` (`summer4.epi`)

Infectiousness weights on the force-of-infection reduction — summer2's
`Stratification.add_infectiousness_adjustments`.

### 6.4a. Where the weights attach

**To the force of infection, not to the stratification.** This matches the
decision already recorded for `A1` in the ledger ("flow-owned, deliberately not
on `Stratification`") and keeps `summer4.epi` from reaching back into the
taxonomy. A stratification does not know what a pathogen is.

Accept a mapping of trait → rate expression, so weights are calibratable:

```python
ForceOfInfection(..., infectiousness={age["0-14"]: 0.7, age["65+"]: refs.nu_old})
```

Weights multiply the grouped infectious term before the mixing matmul, which is
where the case study puts them: `(nu * i_by_age) / n_by_age`, then `K @ ...`.

### 6.4b. Own the normalisation policy

The case study normalises weights so their population-weighted mean is exactly 1,
to break the redundancy between `contact_rate` and the overall infectiousness
scale. Its `POPULATION_SHARE` came from the **final state of a prior 300-year
demography run**, so a parameter definition depended on a previous simulation.
That coupling is a real ergonomic cost and this is where it gets owned.

Offer `normalize="population" | "mean" | None` and document in the docstring
**which redundancy each one breaks** and which it does not. `"population"` needs
the population at evaluation time, so compute it from the same `Reduce` the FOI
already builds rather than asking the user for a prior run's output.

### 6.4c. Tests — `tests/test_epi_foi.py`

- Two parameterisations differing only by a constant scale give **identical**
  trajectories under `normalize="population"`, and different ones under `None`.
  This is the claim, so it is the test.
- Weights keyed on a property the FOI does not group by raise, naming both.
- `jax.grad` through a weight is finite and matches central differences.
- With `normalize="population"`, the weighted mean of the applied weights is 1.0
  at every saved time, on a model whose population composition changes.

### 6.4d. Notebook

Extend `examples/notebooks/09-epi-models.ipynb` with an age-varying infectiousness
example, showing the unnormalised/normalised equivalence explicitly.

**Ledger discipline:** `A4` (`Stratification.add_infectiousness_adjustments`) →
`full`, Notes naming `ForceOfInfection(infectiousness=...)` and recording that it
is FOI-owned rather than stratification-owned, as `A1` is. 43 → 44. `A4` was
`none`, so both totals move.

---

## 6.5 — `feat/epi-multistrain` (`summer4.epi`)

Multiple diseases and multiple strains, in the shape chosen for this project: **a
named force of infection per (disease, stratum-group)**, several coexisting on
one compartment map.

### 6.5a. The shape

- A strain-stratified model is built by `ForceOfInfection.per_trait(strain, ...)`,
  returning one named FOI per strain trait, each with its own infectious selector
  (`state["I"] & strain[s]`), its own mixing matrix if given one, and its own
  saved trace.
- Two diseases on one map are simply two independent `ForceOfInfection` objects
  with disjoint infectious selectors and separate names. No new machinery — but
  the plan says so explicitly, and the tests prove it, because "it should just
  work" is not evidence.
- Cross-immunity, waning and strain replacement are ordinary flows between
  compartments and need nothing from this subphase. Do not build a cross-immunity
  abstraction here; if one is wanted later it is a separate package.

### 6.5b. Tests — `tests/test_epi_multistrain.py` (new)

- **Headline gate — non-interference.** Two diseases on one compartment map: each
  calibrated alone, then run together with the other's `contact_rate` at zero,
  gives **bit-identical** trajectories to the solo runs. A leak between FOIs is
  the failure mode that matters and it is otherwise invisible.
- `per_trait` over a strain property builds one FOI per trait, each with the
  right infectious selector; assert on the selectors, not only on the output.
- Strain-specific infectiousness recovers the generating gradient under an
  assortative matrix (the 6.3d identifiability result, now per strain).
- Each FOI saves its own properly dimensioned trace under its own name.

### 6.5c. The ledger row you must not "fix"

`S7` (`StrainStratification`) currently reads "Strain-aware force of infection is
hand-written". After this subphase that Notes cell is false — **update the note**.

**Leave the status `partial`.** `S7` is on the *What never reaches `full`* list as
a summer2 shape summer4 has deliberately rejected (a property plus per-strain
flows, not a bundled class), and `AGENTS.md` forbids changing those without
changing the decision explicitly. The same applies to `S6`. Write this in the PR
description so a reviewer does not ask for it to be moved.

### 6.5d. Notebook

Extend `examples/notebooks/09-epi-models.ipynb` with a two-strain model, and state
the non-interference property on the page with the assertion beside it.

**Ledger discipline:** no `Status` changes. `S7`'s Notes cell only.

---

## 6.6 — `feat/epi-model` (`summer4.epi`)

The simple frontend: something that feels like summer2's `CompartmentalModel`,
without sacrificing generality.

### 6.6a. The surface

```python
from summer4.epi import EpiModel

m = EpiModel(pmap, infectious=state["I"])
m.set_mixing_matrix(age, refs.contacts)
m.add_infectiousness_adjustments(age, {"0-14": 0.7, "65+": 1.4})
m.add_infection_frequency_flow("infection", state["S"], state["E"], Param("beta"))
m.add_infection_density_flow(...)
m.add_transition_flow("progression", state["E"], state["I"], 1 / 5)
compiled = m.compile()
```

- Every builder method is sugar over an object from 6.3–6.5 that the user could
  have constructed directly. Non-infection methods delegate straight to
  `FlowModel`.
- **The escape hatches are part of the design, not an afterthought.**
  `m.flow_model` exposes the underlying `FlowModel`; `m.foi(name)` returns the
  `ForceOfInfection`; `m.add_flow(...)` accepts any declarative flow. A user who
  outgrows the frontend steps down one layer without rebuilding.

### 6.6b. The acceptance gate

**A digest-equality test.** A model built through `EpiModel` and the same model
built declaratively compile to `CompiledModel`s with **equal digests**:

```python
assert m.compile() == declarative_model.compile()
```

`CompiledModel.__eq__` is already defined over `_digest`, which covers topology,
float arrays, split proportions and `derived_fn` identity. This single assertion
is what makes "without sacrificing generality" checkable rather than
aspirational, and it belongs in the notebook as well as the test suite.

### 6.6c. `Param` — an open decision to settle in this subphase

The summer2 muscle memory the frontend exists to serve is `Parameter("beta")`.
summer4's equivalent is `FieldRef(("beta",))`, and `_lookup_path` already resolves
a `FieldRef` against a plain dict as well as a NamedTuple, so a one-element path
already works against ordinary params.

**Recommendation: ship `Param("beta")` as a thin alias for `FieldRef(("beta",))`,
in core, not in `summer4.epi`** — a named parameter is not an epidemiological
idea, and putting it in `epi` would mean the frontend's most-used name is
unavailable to anyone using the declarative API. Cost is one name and one
docstring. If you decide against it, say why in the PR and use `derived_refs` in
every example instead; do not leave the question open.

### 6.6d. Tests — `tests/test_epi_model.py` (new)

- The digest-equality gate above, for a model using mixing, infectiousness and
  both infection kinds.
- `m.flow_model` and `m.foi(name)` return the live objects, and mutating through
  them is visible in the compiled result.
- `EpiModel` with no infection flows compiles to the same thing a bare
  `FlowModel` would.
- A duplicate flow name raises, matching `FlowModel.add_flow`'s existing error.
- Rebuilding the same `EpiModel` twice hits the jit cache.

### 6.6e. Notebook — the package's whole ergonomic claim

Finish `examples/notebooks/09-epi-models.ipynb`:

- The age-stratified SEIRS model from `docs/case-studies/age-stratified-seirs.ipynb`
  rebuilt through `EpiModel`, with the hand-written `derived_fn` gone. State the
  line count honestly, both before and after.
- The digest-equality assertion visible on the page.
- The two-strain model from 6.5.
- The homogeneous-versus-assortative identifiability comparison, which is the
  scientific reason the package exists.

> ### Notebook gate — blocking
>
> **This is the notebook to user-test hardest.** The frontend's entire claim is
> ergonomic, and an ergonomic claim cannot be verified by a test suite. Do not
> merge until the user has run these in `pixi run notebook` and ticked:
>
> - [ ] `examples/notebooks/09-epi-models.ipynb` — a summer2 user can build an
>       age-mixed SEIRS model from the page alone, and can see from the page that
>       the frontend gives up no generality.
> - [ ] the frequency-versus-density distinction is unmissable, not a footnote.
> - [ ] the force of infection is inspectable as a properly dimensioned trace,
>       without re-slicing a broadcast array.
> - [ ] a modeller reading it would write this code, rather than dropping to
>       `derived_fn` out of habit.
> - [ ] any ledger row left `partial` names a blocker the user agrees is real.

---

## Verification

Per subphase, before asking for a merge:

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch
pixi run coverage
pixi run -e docs docs-strict
pixi run test-all
```

Phase-specific gates, restated:

- **6.1** — grouped arithmetic preserves the grouping; mismatched groupings raise
  rather than broadcasting; `FlowRef.sum_over` still works unchanged.
- **6.2** — a `Reduce`-built SIR matches a `derived_fn`-built one to machine
  precision; the `keep` / `where` polarity identity holds; one `segment_sum` per
  `Reduce` in the jaxpr.
- **6.3** — the homogeneous-matrix $\lambda$ difference is exactly `0.0` and the
  assortative one is not; a custom callable reproduces `"frequency"`
  bit-identically; the full FOI matches the case study's hand-written one.
- **6.4** — a constant rescale of the weights changes nothing under
  `normalize="population"`.
- **6.5** — two diseases on one map do not interfere, bit-identically.
- **6.6** — builder and declarative models compile to equal digests.

**Ledger discipline across the package:** `F7` `F8` `A4` `M1` → `full`, 40 → 44
of 52 (85%). `S7`'s Notes cell updated, status untouched. In the *Delivery status*
section of `docs/evaluation/coverage-ledger.md`, move WP6's row from "Ledger
paragraph only" to `plans/epi-infection-mixing.plan.md` and add the landed
subphase branches to the *Landed phases* table.

The remaining distance to the published 47 / 52 ceiling is WP3 (initial
population: `L4` `L5` `S8`), which is not in this plan's scope. The catch-up
sweep that follows is `plans/textbook-catchup.plan.md`.
