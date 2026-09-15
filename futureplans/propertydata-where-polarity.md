# `PropertyData.where` has mask polarity under a keep-shaped name

## What is wrong today

{meth}`PropertyData.where` (`src/summer4/jax/propertydata.py:112`) *replaces*
the compartments its selector matches:

```python
def where(self, sel: Selector, other: Any) -> PropertyData:
    """Replace matching compartments with ``other`` (scalar or PropertyData)."""
    mask = self.pmap.mask(sel)
    ...
    return self._with_data(jnp.where(mask, other_data, self.data))
```

That is `pandas.Series.mask` semantics, not `pandas.Series.where` semantics, and
not `numpy.where` argument order either. To sum infectious prevalence a user
must write the double negative:

```python
pd_y.where(~state["I"], 0.0).sum_over(age)     # correct
pd_y.where(state["I"], 0.0).sum_over(age)      # sums S + E + R
```

## Why it hurts

The wrong form is **silently wrong**, not an error. Writing a force of
infection with `where(state["I"], 0.0)` produces a force of infection driven by
every non-infectious compartment. The model compiles, integrates, and produces
a plausible-looking epidemic that saturates almost immediately, and every
downstream conclusion is wrong. This happened while writing
`docs/case-studies/age-stratified-seirs.ipynb` and cost a full round of
identifiability analysis: a calibration that looked like a genuine flat
likelihood ridge was an artefact of the polarity, and the ridge disappeared once
it was fixed.

The name is the whole problem. Every other reduction on `PropertyData` names
what it keeps (`select`, `sum_over`, `partition`), so `where` reads as the same
family.

## What a fix looks like

Any of, in rough order of preference:

1. Add `keep(sel, other=0.0)` as the affirmative spelling, implemented as
   `where(~sel, other)`, and use it throughout the docs. Keep `where` for
   compatibility.
2. Lead the docstring with the polarity and the double-negative idiom, rather
   than stating it in passing — currently the one-line summary is accurate but
   easy to skim past.
3. Rename to `mask` and deprecate `where`. Closest to pandas, but a breaking
   change for a pre-1.0 nicety.

Option 1 plus option 2 is cheap and removes the trap without breaking anything.
A test asserting `keep(sel) == where(~sel)` over a ragged map would pin it.
