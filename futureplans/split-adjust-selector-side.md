# `split=` adjustments on a bare trait are rejected as source-side

**Status:** open, unscheduled. Reproduced on `4e52277` (unrelated to the branch
that filed this note).
**Where:** `_bind_adjust_masks`, `src/summer4/flows/actualize.py:276`;
`docs/summer2/10-derived-outputs-stratified.ipynb` (third code cell).

## What is wrong today

`pixi run -e docs docs-strict` fails on `main`. The summer2 page
`10-derived-outputs-stratified.ipynb` builds an `incidence` flow that splits
into a `clinical` property and then adjusts each branch:

```python
TransitionFlow(
    "incidence", ..., split={clinical: {name: 0.25 for name in clinical.traits}},
    adjust=[Multiply(YOUNG_SPLIT[c] / 0.25, where=age["young"] & clinical[c]) ...],
)
```

`clinical` is stratified only under `state["I"]`, so it is absent on the
source compartment of every `incidence` edge. `_bind_adjust_masks` defaults a
bare `Trait` to the source side, finds the property absent there, and raises:

```
ValueError: Adjustment where=And(left=Trait(property='age', name='young'),
right=Trait(property='clinical', name='asymptomatic')) can never apply on flow
'incidence': property 'clinical' is absent on the source of every edge.
Use Dest(...)/Source(...) or split=.
```

The property the adjustment names is exactly the one `split=` introduces on the
destination, so the adjustment is well-formed from the modeller's point of view
and the error's own advice ("or `split=`") is already satisfied.

## Why it hurts

`docs-strict` is a required check when a branch touches documented behaviour
(`AGENTS.md` § Documentation), and it is red before any change is made, so it
cannot distinguish a branch's own breakage from this one. It also makes a
documented summer2 page non-executable.

## Done when

`_bind_adjust_masks` resolves a bare trait against the post-`split=`
destination when the property is introduced by that flow's `split=` (rather
than defaulting to source and raising), **or** the notebook is corrected to say
`Dest(clinical[c])` and the error message stops suggesting `split=` for a case
that does not work. Either way `pixi run -e docs docs-strict` passes on `main`.
