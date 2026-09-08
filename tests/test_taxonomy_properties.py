"""Hypothesis property tests for the compartment taxonomy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from hypothesis import given, settings
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


@settings(max_examples=60, deadline=2000)
@given(built_maps())
def test_no_duplicate_compartments(
    payload: tuple[list[Property], list[bool], PropertyMap],
) -> None:
    _props, _flags, pm = payload
    unique = np.unique(pm.codes, axis=0)
    assert unique.shape[0] == pm.size


@settings(max_examples=60, deadline=2000)
@given(built_maps())
def test_rebuild_is_deterministic(
    payload: tuple[list[Property], list[bool], PropertyMap],
) -> None:
    props, flags, pm = payload
    again = _build_map(props, flags)
    assert pm == again
    np.testing.assert_array_equal(pm.codes, again.codes)


@settings(max_examples=60, deadline=2000)
@given(built_maps())
def test_select_and_is_intersection(
    payload: tuple[list[Property], list[bool], PropertyMap],
) -> None:
    props, _flags, pm = payload
    a = props[0][props[0].traits[0]]
    b = props[1][props[1].traits[0]]
    both = set(pm.select(a & b).tolist())
    left = set(pm.select(a).tolist())
    right = set(pm.select(b).tolist())
    assert both == left & right


@settings(max_examples=60, deadline=2000)
@given(built_maps())
def test_select_or_is_union(
    payload: tuple[list[Property], list[bool], PropertyMap],
) -> None:
    props, _flags, pm = payload
    a = props[0][props[0].traits[0]]
    b = props[1][props[1].traits[0]]
    either = set(pm.select(a | b).tolist())
    left = set(pm.select(a).tolist())
    right = set(pm.select(b).tolist())
    assert either == left | right


@settings(max_examples=60, deadline=2000)
@given(built_maps())
def test_partition_is_disjoint_cover_of_present(
    payload: tuple[list[Property], list[bool], PropertyMap],
) -> None:
    props, _flags, pm = payload
    for prop in props:
        parts = pm.partition(prop)
        indices = [int(i) for idx in parts.values() for i in idx]
        assert len(indices) == len(set(indices))
        assert set(indices) == set(pm.select(prop.present()).tolist())


@settings(max_examples=40, deadline=2000)
@given(built_maps())
def test_not_excludes_unknown(
    payload: tuple[list[Property], list[bool], PropertyMap],
) -> None:
    props, _flags, pm = payload
    for prop in props:
        trait = prop[prop.traits[0]]
        true_set = set(pm.select(trait).tolist())
        false_set = set(pm.select(~trait).tolist())
        unknown = set(pm.select(prop.absent()).tolist())
        assert true_set.isdisjoint(false_set)
        assert true_set.isdisjoint(unknown)
        assert false_set.isdisjoint(unknown)
        assert true_set | false_set | unknown == set(range(pm.size))
