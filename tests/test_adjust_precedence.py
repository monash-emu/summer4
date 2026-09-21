"""Adjustment precedence, Source/Dest masks, and overwrite-overlap checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from summer4 import (
    Dest,
    Everything,
    FlowModel,
    Multiply,
    Overwrite,
    Property,
    PropertyMap,
    Source,
    TraitMatrix,
    Transform,
    TransitionFlow,
)
from summer4.flows import EntryFlow, ExitFlow
from summer4.flows.rates import Adjustment, canonical_adjustments


def _sir_severity() -> tuple[Property, Property, PropertyMap]:
    state = Property("state", ("S", "I", "R"))
    severity = Property("severity", ("mild", "severe"))
    pm = PropertyMap.from_property(state).stratify(severity, where=state["I"])
    return state, severity, pm


def _sir_age() -> tuple[Property, Property, PropertyMap]:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("0-4", "5-9", "10+"))
    return state, age, PropertyMap.from_property(state).stratify(age)


def test_default_levels_sort() -> None:
    chain: tuple[Adjustment, ...] = (
        Transform(lambda prev: prev),
        Multiply(2.0),
        Overwrite(1.0),
        Multiply(3.0),
    )
    ordered = canonical_adjustments(chain)
    assert [type(a).__name__ for a in ordered] == [
        "Overwrite",
        "Multiply",
        "Multiply",
        "Transform",
    ]
    assert ordered[1].value == Multiply(2.0).value
    assert ordered[2].value == Multiply(3.0).value


def test_case3_overwrite_before_multiply() -> None:
    state, age, pm = _sir_age()
    strain = Property("strain", ("a", "b"))
    pm = pm.stratify(strain)
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            1.0,
            adjust=[
                Multiply(2.0, where=age["10+"]),
                Overwrite(0.5, where=strain["b"]),
            ],
        )
    )
    cm = model.compile()
    flow = cm.flows["inf"]
    y = np.ones(pm.size)
    dy = np.asarray(cm.vector_field(0.0, y, {}))
    age_old = flow.edge_map.mask(Source(age["10+"]))
    strain_b = flow.edge_map.mask(Source(strain["b"]))
    expected = np.ones(flow.n_edges, dtype=np.float64)
    expected = np.where(strain_b, 0.5, expected)
    expected = np.where(age_old, expected * 2.0, expected)
    for i in range(flow.n_edges):
        assert dy[int(flow.dest_idx[i])] == pytest.approx(expected[i] * float(flow.weight[i]))

    model2 = FlowModel(pm)
    model2.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            1.0,
            adjust=[
                Multiply(2.0, where=age["10+"]),
                Overwrite(0.5, where=strain["b"], precedence=2),
            ],
        )
    )
    cm2 = model2.compile()
    flow2 = cm2.flows["inf"]
    dy2 = np.asarray(cm2.vector_field(0.0, y, {}))
    age_old2 = flow2.edge_map.mask(Source(age["10+"]))
    strain_b2 = flow2.edge_map.mask(Source(strain["b"]))
    expected2 = np.ones(flow2.n_edges, dtype=np.float64)
    expected2 = np.where(age_old2, expected2 * 2.0, expected2)
    expected2 = np.where(strain_b2, 0.5, expected2)
    for i in range(flow2.n_edges):
        assert dy2[int(flow2.dest_idx[i])] == pytest.approx(expected2[i] * float(flow2.weight[i]))


def test_case4_same_level_overwrite_overlap_raises() -> None:
    state, age, pm = _sir_age()
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            0.2,
            adjust=[
                Overwrite(0.0, where=age["0-4"]),
                Overwrite(0.5, where=age["0-4"]),
            ],
        )
    )
    with pytest.raises(ValueError, match="Overlapping Overwrite") as exc:
        model.compile()
    msg = str(exc.value)
    assert "->" in msg or "=" in msg

    model_ok = FlowModel(pm)
    model_ok.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            0.2,
            adjust=[
                Overwrite(0.0, where=age["0-4"]),
                Overwrite(0.1, where=age["5-9"]),
                Overwrite(0.2, where=age["10+"]),
            ],
        )
    )
    model_ok.compile()  # disjoint — no raise


@settings(max_examples=20, deadline=None)
@given(st.permutations([0, 1, 2]))
def test_multiply_order_irrelevant(order: list[int]) -> None:
    state, age, pm = _sir_age()
    wheres = [age["0-4"], age["5-9"], age["10+"]]
    factors = [2.0, 3.0, 5.0]
    adjusts = [Multiply(factors[i], where=wheres[i]) for i in order]
    model = FlowModel(pm)
    model.add_flow(TransitionFlow("inf", state["S"], state["I"], 1.0, adjust=adjusts))
    cm = model.compile()
    y = np.ones(pm.size)
    dy = np.asarray(cm.vector_field(0.0, y, {}))
    for trait, factor in zip(age.traits, factors, strict=True):
        dest = pm.select(state["I"] & age[trait])
        np.testing.assert_allclose(dy[dest], factor)


def test_transforms_keep_declaration_order() -> None:
    seen: list[str] = []

    def first(prev: object) -> object:
        seen.append("first")
        return prev

    def second(prev: object) -> object:
        seen.append("second")
        return prev

    state = Property("state", ("S", "I"))
    pm = PropertyMap.from_property(state)
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            1.0,
            adjust=[Transform(first), Multiply(2.0), Transform(second)],
        )
    )
    model.compile().vector_field(0.0, np.array([1.0, 0.0]), {})
    assert seen == ["first", "second"]


def test_bare_where_mask_identical_to_legacy() -> None:
    state, age, pm = _sir_age()
    where = age["0-4"]
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow("inf", state["S"], state["I"], 0.2, adjust=[Multiply(2.0, where=where)])
    )
    model.add_flow(ExitFlow("death", Everything(), 0.1, adjust=[Multiply(2.0, where=where)]))
    model.add_flow(EntryFlow("birth", state["S"], 1.0, adjust=[Multiply(2.0, where=where)]))
    cm = model.compile()
    for name, gather in (
        ("inf", cm.flows["inf"].src_idx),
        ("death", cm.flows["death"].src_idx),
        ("birth", cm.flows["birth"].dest_idx),
    ):
        flow = cm.flows[name]
        assert len(flow.adjust_masks) == 1
        mask = flow.adjust_masks[0]
        assert mask is not None
        legacy = np.isin(gather, pm.select(where))
        np.testing.assert_array_equal(mask, legacy)


def test_dest_where_fires_on_dest_only_property() -> None:
    state, severity, pm = _sir_severity()
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            1.0,
            adjust=[Multiply(2.0, where=Dest(severity["severe"]))],
        )
    )
    cm = model.compile()
    flow = cm.flows["inf"]
    y = np.ones(pm.size)
    dy = np.asarray(cm.vector_field(0.0, y, {}))
    # Dest-only severity splits infection 50/50; severe edges get rate 2.
    severe_dest = set(pm.select(state["I"] & severity["severe"]).tolist())
    mild_dest = set(pm.select(state["I"] & severity["mild"]).tolist())
    for dest_i, weight in zip(flow.dest_idx, flow.weight, strict=True):
        d = int(dest_i)
        if d in severe_dest:
            assert dy[d] == pytest.approx(2.0 * float(weight))
        elif d in mild_dest:
            assert dy[d] == pytest.approx(1.0 * float(weight))


def test_dead_where_raises() -> None:
    state, severity, pm = _sir_severity()
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            1.0,
            adjust=[Multiply(2.0, where=severity["severe"])],
        )
    )
    with pytest.raises(ValueError, match=r"Dest\("):
        model.compile()

    exit_model = FlowModel(pm)
    exit_model.add_flow(
        ExitFlow("death", state["S"], 0.1, adjust=[Multiply(2.0, where=Dest(severity["severe"]))])
    )
    with pytest.raises(ValueError, match="without a destination"):
        exit_model.compile()

    entry_model = FlowModel(pm)
    entry_model.add_flow(
        EntryFlow("birth", state["S"], 1.0, adjust=[Multiply(2.0, where=Source(state["S"]))])
    )
    with pytest.raises(ValueError, match="without a source"):
        entry_model.compile()


def test_absent_selector_is_not_dead() -> None:
    state, severity, pm = _sir_severity()
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            1.0,
            adjust=[Multiply(2.0, where=severity.absent())],
        )
    )
    cm = model.compile()
    mask = cm.flows["inf"].adjust_masks[0]
    assert mask is not None
    assert bool(mask.all())


def test_bare_split_trait_matches_dest() -> None:
    """A property introduced by split= is read on the destination."""
    state, severity, pm = _sir_severity()
    split = {severity: {"mild": 0.25, "severe": 0.75}}

    def compiled(where: object) -> tuple[Any, Any]:
        model = FlowModel(pm)
        model.add_flow(
            TransitionFlow(
                "inf",
                state["S"],
                state["I"],
                1.0,
                split=split,
                adjust=[Multiply(3.0, where=where)],
            )
        )
        cm = model.compile()
        return cm, np.asarray(cm.vector_field(0.0, np.ones(pm.size), {}))

    bare_cm, bare_dy = compiled(severity["severe"])
    dest_cm, dest_dy = compiled(Dest(severity["severe"]))
    bare_mask = bare_cm.flows["inf"].adjust_masks[0]
    dest_mask = dest_cm.flows["inf"].adjust_masks[0]
    assert bare_mask is not None and dest_mask is not None
    np.testing.assert_array_equal(bare_mask, dest_mask)
    np.testing.assert_allclose(bare_dy, dest_dy)
    severe = set(pm.select(state["I"] & severity["severe"]).tolist())
    mild = set(pm.select(state["I"] & severity["mild"]).tolist())
    for dest_i, weight in zip(
        bare_cm.flows["inf"].dest_idx, bare_cm.flows["inf"].weight, strict=True
    ):
        d = int(dest_i)
        if d in severe:
            assert bare_dy[d] == pytest.approx(3.0 * float(weight))
        elif d in mild:
            assert bare_dy[d] == pytest.approx(float(weight))


def test_bare_split_trait_keeps_other_properties_on_the_source() -> None:
    """``age & clinical`` uses source age when pairing moves age across the edge."""
    state = Property("state", ("S", "I"))
    loc = Property("loc", ("north", "south"))
    clinical = Property("clinical", ("mild", "severe"))
    pm = PropertyMap.from_property(state).stratify(loc).stratify(clinical, where=state["I"])
    # dest × source: north sources land in the south, and the reverse.
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    where = loc["north"] & clinical["severe"]
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            1.0,
            pairing=TraitMatrix(loc, matrix),
            split={clinical: {"mild": 0.25, "severe": 0.75}},
            adjust=[Multiply(3.0, where=where)],
        )
    )
    explicit = FlowModel(pm)
    explicit.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            1.0,
            pairing=TraitMatrix(loc, matrix),
            split={clinical: {"mild": 0.25, "severe": 0.75}},
            adjust=[Multiply(3.0, where=Source(loc["north"]) & Dest(clinical["severe"]))],
        )
    )
    cm = model.compile()
    other = explicit.compile()
    bare_mask = cm.flows["inf"].adjust_masks[0]
    explicit_mask = other.flows["inf"].adjust_masks[0]
    assert bare_mask is not None and explicit_mask is not None
    np.testing.assert_array_equal(bare_mask, explicit_mask)
    assert bool(bare_mask.any())
    y = np.zeros(pm.size)
    y[pm.select(state["S"] & loc["north"])] = 1.0
    dy = np.asarray(cm.vector_field(0.0, y, {}))
    south_severe = int(pm.select(state["I"] & loc["south"] & clinical["severe"])[0])
    north_severe = int(pm.select(state["I"] & loc["north"] & clinical["severe"])[0])
    assert dy[south_severe] == pytest.approx(3.0 * 0.75)
    assert dy[north_severe] == pytest.approx(0.0)


def test_explicit_source_of_split_property_still_raises() -> None:
    state, severity, pm = _sir_severity()
    model = FlowModel(pm)
    model.add_flow(
        TransitionFlow(
            "inf",
            state["S"],
            state["I"],
            1.0,
            split={severity: {"mild": 0.5, "severe": 0.5}},
            adjust=[Multiply(2.0, where=Source(severity["severe"]))],
        )
    )
    with pytest.raises(ValueError, match="absent on the source"):
        model.compile()
