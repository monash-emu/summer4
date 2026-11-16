---
name: property-hashable
description: Store Property traits as a tuple so a list-built Property is hashable and works as a split= key; reject string traits and duplicate split keys.
---

# Hashable `Property` from any sequence

## Context

A tb_acf user following the 0.3.0a3 release note wrote `split={focus: ...}` and got
`unhashable type: 'list'`: their `focus` was `Property("focus", [...])`. `Property` is a frozen
dataclass annotated `traits: tuple[str, ...]`, but the public constructor never converted the
argument, so a list was stored and the generated `__hash__` failed. `Property._mangled` already
converted with `tuple(...)`.

## Design

- `Property.__post_init__` and `Property._mangled` store `traits` through `_as_traits`, which
  returns `tuple(traits)`. Hashing stays by value, `(name, traits)`, consistent with the
  dataclass `__eq__` and with the rule that properties are identified by name, not by object
  identity. An `id`-based hash was rejected: it would disagree with `__eq__`, and a rebuilt or
  copied property would stop matching.
- `_as_traits` raises `TypeError` for a bare `str`, which used to be accepted and iterate as one
  trait per character.
- `_normalize_split` raises when two keys resolve to the same property name (a `Property` and
  its name string, or two properties sharing a name), instead of keeping the last entry.
  `split=` is the only public API keyed by `Property`.

## Deliverables

1. `src/summer4/properties.py`, `src/summer4/flows/join.py`.
2. `tests/test_property_hashable.py`: a list is stored as a tuple; list- and tuple-built
   properties are equal and hash alike; any iterable works; string traits are rejected (public
   and mangled); a list-built property as a `split=` key matches the name key; a duplicate
   split key raises.
3. `examples/notebooks/37-param-split.ipynb` builds `focus` from a list, as tb_acf does.
4. CHANGELOG `[Unreleased]` → Fixed. No version bump here; it ships in the next release with
   the multi-model calibration helpers (`feat/multi-model`).
