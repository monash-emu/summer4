"""Shared Hypothesis strategies for taxonomy / flow property tests."""

from __future__ import annotations

from collections.abc import Sequence

from hypothesis import strategies as st

from summer4 import Property, PropertyMap
from summer4.selectors import Selector

_TRAIT_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _unique_names(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}{i}" for i in range(count))


@st.composite
def property_specs(draw: st.DrawFn) -> list[Property]:
    n_props = draw(st.integers(min_value=2, max_value=4))
    props: list[Property] = []
    for i in range(n_props):
        n_traits = draw(st.integers(min_value=2, max_value=4))
        props.append(Property(f"p{i}", _unique_names(f"{_TRAIT_ALPHABET[i]}", n_traits)))
    return props


def _build_map(props: Sequence[Property], partial_flags: Sequence[bool]) -> PropertyMap:
    pm = PropertyMap.from_property(props[0])
    for i, prop in enumerate(props[1:], start=1):
        where: Selector | None = None
        if partial_flags[i] and i >= 1:
            parent = props[i - 1]
            where = parent[parent.traits[0]]
        pm = pm.stratify(prop, where=where)
    return pm


@st.composite
def built_maps(
    draw: st.DrawFn,
) -> tuple[list[Property], list[bool], PropertyMap]:
    props = draw(property_specs())
    flags = [False, *[draw(st.booleans()) for _ in props[1:]]]
    return props, flags, _build_map(props, flags)
