"""Hypothesis property tests for the compartment taxonomy."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings

from summer4 import Property, PropertyMap
from tests.helpers.strategies import _build_map, built_maps


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
def test_equal_maps_have_equal_hashes(
    payload: tuple[list[Property], list[bool], PropertyMap],
) -> None:
    props, flags, pm = payload
    again = _build_map(props, flags)
    assert pm == again
    assert hash(pm) == hash(again)
    assert {pm: "a"}[again] == "a"


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
