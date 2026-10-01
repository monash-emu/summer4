---
name: keep-absent-rows
overview: PropertyData.keep(sel) must replace every row sel does not positively match, including rows where sel's property is absent.
todos:
  - id: fix
    content: PropertyData.keep masks on pmap.mask(sel) instead of delegating to where(~sel)
    status: completed
  - id: audit
    content: Audit every keep / where(~sel) / mask use in src for the same absent-row leak
    status: completed
  - id: tests
    content: Ragged-map regression tests in tests/test_state_reductions.py (keep, Reduce) and tests/test_epi_foi.py (ForceOfInfection infectious= and denominator=)
    status: completed
  - id: notebook
    content: Section 7 of examples/notebooks/09-epi-models.ipynb — an SIR beside a 5,000-person register with no state axis matches the plain SIR
    status: completed
  - id: docs
    content: SEIRS case study uses keep; ragged-stratification guide shows keep vs where(~sel)
    status: completed
isProject: false
---

# `keep` leaks rows where the selector's property is absent

## Bug

`PropertyData.keep(sel, other)` was `self.where(~sel, other)`. Selectors are
Kleene three-valued: on a compartment where `sel`'s property is absent, both
`sel` and `~sel` evaluate to *unknown*, and `mask` keeps only *true*. So
`where(~sel)` replaced only the rows where `sel` is false and **kept** the
unknown ones, contradicting `keep`'s own docstring ("replace the rest") and
`Reduce`'s ("`where` means KEEP").

Every consumer of `keep` inherited the leak on a ragged map:

- `Reduce(sum_over=..., where=sel)` (`src/summer4/flows/compiled.py`).
- `ForceOfInfection`'s `infectious=` pool and `denominator=` pool
  (`src/summer4/epi/infection.py`).

Reproduction: `kind` splits the map into a population with `state` and a
programme with `programme`, then `pop` stratifies everything. With
`y = [S=99990, I=10, waiting=1, active=0]`,
`Reduce(sum_over=pop, where=programme["active"])` gave 100000 (expected 0) and
`Reduce(sum_over=pop, where=state["I"])` gave 11 (expected 10).
`pmap.select(...)` was always correct. Rectangular maps were unaffected, because
there no selector is ever unknown.

## Fix

`keep` builds its own mask, `jnp.where(pmap.mask(sel), data, other)`, so every
row that is not Kleene-true is replaced. The `where` docstring no longer
recommends `where(~sel, other)` as a spelling of keep.

## Audit

Everything else in `src/` selects with positive `select` / `mask` and was
already correct:

- `src/summer4/results/` (`eval._select_idx`, `Output.select`) gathers
  `pmap.select(sel)`.
- `apply_compartment_weights` multiplies where `mask(selector)` is true;
  unmatched and absent rows keep weight 1, as documented.
- Adjustment `where=` (`flows/actualize.py`) uses `EdgeMap.mask`, which keeps
  only true.
- `PropertyMap.stratify(where=)` splits only rows where `where` is true.

The only `where(~sel)` uses outside `keep` were user-level code in
`tests/test_state_reductions.py` and the SEIRS case study, both on rectangular
maps. Both now use `keep`.
