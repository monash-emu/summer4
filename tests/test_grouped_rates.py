"""GroupedRate arithmetic, matrix product, and multi-property alignment."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from summer4 import (
    EntryFlow,
    Everything,
    ExitFlow,
    FlowModel,
    GroupedRate,
    Property,
    PropertyMap,
    derived_refs,
)


class _DeathParams(NamedTuple):
    death_rate: float


def test_grouped_scalar_arithmetic_preserves_grouping() -> None:
    age = Property("age", ("young", "old"))
    data = np.array([1.0, 2.0])
    g = GroupedRate(data, (age,))
    scaled = g * 3.0
    assert scaled.properties == (age,)
    np.testing.assert_allclose(np.asarray(scaled.data), [3.0, 6.0])
    scaled_l = 3.0 * g
    assert scaled_l.properties == (age,)
    np.testing.assert_allclose(np.asarray(scaled_l.data), [3.0, 6.0])
    other = GroupedRate(np.array([2.0, 4.0]), (age,))
    quot = g / other
    assert quot.properties == (age,)
    np.testing.assert_allclose(np.asarray(quot.data), [0.5, 0.5])


def test_mismatched_groupings_raise_naming_both() -> None:
    age = Property("age", ("young", "old"))
    loc = Property("location", ("north", "south"))
    left = GroupedRate(np.array([1.0, 2.0]), (age,))
    right = GroupedRate(np.array([1.0, 2.0]), (loc,))
    with pytest.raises(ValueError, match=r"age.*location|location.*age"):
        _ = left * right


def test_matrix_product_against_hand_computation() -> None:
    pytest.importorskip("jax")
    age = Property("age", ("young", "old", "senior"))
    g = GroupedRate(np.array([1.0, 2.0, 3.0]), (age,))
    m = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.8, 0.1],
            [0.0, 0.2, 0.8],
        ]
    )
    expected = m @ np.array([1.0, 2.0, 3.0])
    product = m @ g
    assert product.properties == (age,)
    np.testing.assert_allclose(np.asarray(product.data), expected)


def test_two_property_alignment_on_ragged_map() -> None:
    pytest.importorskip("jax")
    state = Property("state", ("S", "I"))
    age = Property("age", ("young", "old"))
    loc = Property("location", ("north", "south"))
    # loc only on I — S rows lack location.
    pmap = PropertyMap.from_property(state).stratify(age).stratify(loc, where=state["I"])
    groups = pmap.group_by(age, loc)
    # One value per realised (age, location) combination.
    values = np.arange(len(groups), dtype=np.float64) + 1.0
    grouped = GroupedRate(values, (age, loc))
    dest_idx = pmap.select(state["I"])
    from summer4.flows.compiled import _align_grouped_rate

    aligned = np.asarray(_align_grouped_rate(grouped, dest_idx, pmap))
    expected = []
    key_to_val = {tuple(t.code for t in traits): values[i] for i, traits in enumerate(groups)}
    for row in dest_idx:
        codes = tuple(int(pmap.codes[row, pmap.column_index(p)]) for p in (age, loc))
        expected.append(key_to_val[codes])
    np.testing.assert_allclose(aligned, expected)

    # Rows lacking location cannot align.
    s_idx = pmap.select(state["S"])
    with pytest.raises(ValueError, match="location|age"):
        _align_grouped_rate(grouped, s_idx, pmap)


def test_flow_ref_sum_over_still_works() -> None:
    """Regression: FlowRef.sum_over remains the producer of GroupedRate."""
    state = Property("state", ("S", "I"))
    loc = Property("location", ("north", "south"))
    pm = PropertyMap.from_property(state).stratify(loc)
    model = FlowModel(pm)
    refs = derived_refs(_DeathParams)
    death = model.add_flow(ExitFlow("death", Everything(), refs.death_rate))
    model.add_flow(EntryFlow("birth", state["S"], death.sum_over(loc)))
    y = np.array([100.0, 300.0, 10.0, 30.0])
    dy = np.asarray(model.compile().vector_field(0.0, y, {"death_rate": 0.1}))
    north_deaths = 0.1 * (100.0 + 10.0)
    south_deaths = 0.1 * (300.0 + 30.0)
    s_north, s_south = pm.select(state["S"])
    np.testing.assert_allclose(dy[s_north], -0.1 * 100.0 + north_deaths)
    np.testing.assert_allclose(dy[s_south], -0.1 * 300.0 + south_deaths)


def test_binop_preserves_grouped_rate_through_eval() -> None:
    """FlowRef.sum_over * scalar stays GroupedRate through BinOp evaluation."""
    pytest.importorskip("jax")
    state = Property("state", ("S", "I"))
    loc = Property("location", ("north", "south"))
    pm = PropertyMap.from_property(state).stratify(loc)
    model = FlowModel(pm)
    refs = derived_refs(_DeathParams)
    death = model.add_flow(ExitFlow("death", Everything(), refs.death_rate))
    # Scale the per-location death total before using it as a birth rate.
    model.add_flow(EntryFlow("birth", state["S"], death.sum_over(loc) * 2.0))
    y = np.array([100.0, 300.0, 10.0, 30.0])
    dy = np.asarray(model.compile().vector_field(0.0, y, {"death_rate": 0.1}))
    north_deaths = 0.1 * (100.0 + 10.0)
    south_deaths = 0.1 * (300.0 + 30.0)
    s_north, s_south = pm.select(state["S"])
    np.testing.assert_allclose(dy[s_north], -0.1 * 100.0 + 2.0 * north_deaths)
    np.testing.assert_allclose(dy[s_south], -0.1 * 300.0 + 2.0 * south_deaths)
