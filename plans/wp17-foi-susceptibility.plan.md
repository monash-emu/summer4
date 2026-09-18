---
name: wp17-foi-susceptibility
description: Give ForceOfInfection a susceptibility surface symmetric to its infectiousness weights, and rewrite textbook chapter 15 onto it — step 16 of docs/dev/roadmap.md.
---

# WP17 — a susceptibility surface on the force of infection

## Context

`ForceOfInfection` owns infectiousness weights: `infectiousness=` scales how
much each group contributes to the infectious pool, optionally normalised by
population (`src/summer4/epi/infection.py`). There is **no symmetric
`susceptibility=`**. A user who wants some groups to acquire infection more
readily than others must either attach `Multiply(..., where=group[trait])`
adjustments to the infection `TransitionFlow`, or row-scale a `MixingMatrix`
with `normalize="none"`.

That works, and it is what
`docs/textbook/15-susceptibility-infectiousness-matrices.ipynb` does today. It is
also unsatisfying for two reasons the chapter itself makes: it splits a related
pair of concepts across two unrelated APIs, and the mixing-matrix route
encourages baking susceptibility into the contact structure — which the chapter
explicitly argues against. Chapter 15 is therefore `partial` in the coverage
ledger with the blocker "No FOI susceptibility surface symmetric to
infectiousness". It is the only textbook chapter short of `full` outside the
contact-survey block (16–19) and calibration (20).

This plan promotes `futureplans/foi-susceptibility-surface.md` into scheduled
work, which is how `AGENTS.md` says a deferred note becomes a plan.

**Closes (API ledger):** no API rows. It moves **textbook row 15** from
`partial` to `full`. **Closes (ports ledger):** nothing — neither tuberculosis
model needs it, though tb_macro's per-source `rel_sus` is the same idea done by
hand. **Declared as:** `WP17` in the coverage ledger's packages block.

## Ordering

Independent of WP13–WP16 and WP10, so it may run at any time after the stack is
on `main`. It is **cheapest after step 8** (`feat/epi-compartment-infectiousness`),
which generalises infectiousness weights from "keyed on a trait of `group_by`" to
"keyed on any selector, applied per compartment". Susceptibility wants exactly
that same machinery pointed at the destination side, so running before step 8
means building it twice. The roadmap places this at step 16 for that reason.

If for any reason this runs before step 8, implement only the trait-keyed form
and say so in the docstring and the handoff; do not invent a second selector
mechanism.

## Read first

1. `AGENTS.md`
2. `futureplans/foi-susceptibility-surface.md`
3. `src/summer4/epi/infection.py` — `ForceOfInfection.__init__`,
   `__field_paths__`, and the infectiousness evaluation path
4. `src/summer4/epi/mixing.py` — `MixingMatrix`, `normalize`
5. `docs/textbook/15-susceptibility-infectiousness-matrices.ipynb` — the
   workaround being replaced
6. `futureplans/foi-multi-property-mixing.md` — `group_by` is one property; the
   same limitation applies to susceptibility, and the docstring must not imply
   otherwise

## Part A — the API

Add one keyword to `ForceOfInfection`, mirroring `infectiousness=`:

```python
ForceOfInfection(
    name,
    *,
    infectious: Selector,
    group_by: Property,
    ...
    infectiousness: InfectiousnessMap | None = None,
    normalize_infectiousness: NormalizeWeights = None,
    susceptibility: SusceptibilityMap | None = None,   # new
)
```

`SusceptibilityMap` accepts the same two shapes infectiousness accepts:

1. a mapping of `Trait` (or bare trait name) on `group_by` → `RateOps | float`;
2. after step 8, a sequence of `(Selector, RateOps | float)` pairs applied per
   compartment.

Semantics, and they are the whole point of the feature:

- **Infectiousness weights the source of transmission**; it multiplies each
  group's contribution *into* the infectious pool, before the mixing product.
- **Susceptibility weights the recipient**; it multiplies the per-group force of
  infection $\lambda_a$ *after* the mixing product, or equivalently the infection
  edges leaving that group.

So the evaluated rate is

$$\lambda_a = s_a \cdot c \cdot \sum_b K_{ab} \, f(w_b I_b, N_b)$$

with $w$ the infectiousness weights, $s$ the susceptibility weights, $K$ the
mixing matrix and $f$ the kind (frequency, density, generalised).

Three decisions to encode explicitly, each of which a reader will ask about:

- **Susceptibility is not normalised.** There is no
  `normalize_susceptibility=`, and none should be added: normalisation exists for
  infectiousness because a population-weighted mean of 1 keeps $R_0$
  interpretable, and no analogous invariant applies to the recipient side.
  Say this in the docstring.
- **Validation matches infectiousness.** Keys must name `group_by` (the same
  check that `__init__` already performs for `infectiousness`, lines 76–83);
  raise with the same message shape. A weight that can never apply is an error,
  not a silent no-op.
- **`__field_paths__` must recurse into susceptibility values**, exactly as it
  does for infectiousness. Missing this leaves `computed_paths` incomplete and
  parameters unvalidated, with no error — the same silent-failure class as the
  five-site rule in `plans/tb-ports-feature-completeness.plan.md`.

## Part B — equivalence with the workaround

The acceptance test for the whole feature: a model with
`susceptibility={age["0-14"]: 0.5}` must compile to the **same digest** as the
same model with `Multiply(0.5, where=age["0-14"])` on the infection flow. If the
digests differ, one of the two is doing something else, and the notebook that
teaches the old way is now teaching a different model.

Where a digest match is not achievable (for example, the adjustment path runs
through a precedence level that the FOI path cannot reproduce), assert numerical
equality on a fixed state instead, and **write down why** in the test docstring
and in the handoff.

## Part C — tests, `tests/test_epi_susceptibility.py` (new)

- Trait-keyed susceptibility on a three-band age model with an asymmetric mixing
  matrix, against a hand-computed expected $\lambda$ vector. Asymmetric, so a
  transposed mixing product cannot pass.
- The digest equivalence of Part B, both ways round.
- `jax.grad` through a `Param`-valued susceptibility weight — the point of
  having it in the rate tree rather than in Python.
- Interaction with infectiousness: both set at once, with different weights, and
  `normalize_infectiousness` on; assert susceptibility is untouched by the
  normalisation.
- Validation: a key naming a property other than `group_by` raises, naming both.
- After step 8: a selector-keyed weight over compartment × age.

## Part D — the chapter, and the ledgers

- Rewrite `docs/textbook/15-susceptibility-infectiousness-matrices.ipynb` onto
  `susceptibility=`. Keep the chapter's argument intact — it is *about* the
  distinction between contact structure, infectiousness and susceptibility, so
  the rewrite should make that argument easier to follow, not merely shorter.
  Keep a short paragraph showing the mixing-matrix route and saying why the
  chapter does not take it.
- Add a note to the user guide where mixing is documented: mixing matrices
  describe contact structure; infectiousness and susceptibility describe the
  people. One sentence each.
- `docs/evaluation/coverage-ledger.md`: textbook row 15 → `full`, `Blocker` →
  `None`. Declare `WP17` in the packages block if it is not already there
  (`*(no API rows; unblocks textbook 15)*`). Then
  `pixi run coverage-write && pixi run coverage`.
- Delete `futureplans/foi-susceptibility-surface.md` **and its bullet in
  `futureplans/README.md`** — a note whose work has landed is misinformation.

## Part E — notebook

Extend `examples/notebooks/09-epi-models.ipynb` with a short susceptibility
section rather than adding a new notebook: it is one keyword on an object that
notebook already teaches. Assert the equivalence from Part B in the notebook
itself, so a reader can see the two formulations agree. **User gate.**

## Verification

```bash
pixi run lint && pixi run format-check && pixi run check-notebooks
pixi run test && pixi run check-branch
pixi run coverage && pixi run roadmap
pixi run -e docs docs-strict
pixi run test-all
```

Plus: textbook row 15 reads `full` with no blocker, the futureplans note is gone
from both the folder and its README, and the rewritten chapter executes in the
docs build.

## Deliberate non-goals

- **No `normalize_susceptibility=`.** See Part A.
- **No multi-property susceptibility.** `group_by` is one property; that
  limitation is recorded in `futureplans/foi-multi-property-mixing.md` and is not
  in scope here.
- **No removal of the `Multiply(..., where=)` route.** It stays valid and stays
  documented; this feature is about having the concept in the right place, not
  about taking anything away.
