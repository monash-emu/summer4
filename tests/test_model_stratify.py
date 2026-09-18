"""FlowModel.stratify, copy, update_flow, and adjust_flow."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.helpers.strategies import built_maps

from summer4 import (
    Everything,
    FlowModel,
    Multiply,
    Property,
    PropertyMap,
    TraitChain,
    TransitionFlow,
    euler,
)
from summer4.flows import EntryFlow, ExitFlow
from summer4.flows.compiled import _align_rate
from summer4.flows.rates import Const


def _sir() -> tuple[Property, PropertyMap]:
    state = Property("state", ("S", "I", "R"))
    return state, PropertyMap.from_property(state)


@settings(max_examples=40, deadline=None)
@given(built_maps(), st.booleans())
def test_stratify_commutes_with_flow_declaration(
    payload: tuple[list[Property], list[bool], PropertyMap],
    use_where: bool,
) -> None:
    props, _flags, pmap = payload
    existing = {p.name for p in pmap.properties}
    q = Property("q", ("q0", "q1"))
    assert q.name not in existing
    where = None
    if use_where:
        first = props[0]
        where = first[first.traits[0]]

    first = props[0]
    second = props[1]
    adjust = [Multiply(2.0, where=second[second.traits[0]])]
    exit_flow = ExitFlow("exit", Everything(), 0.1, adjust=adjust)
    entry_flow = EntryFlow("entry", first[first.traits[0]], 1.0, adjust=adjust)
    transition = TransitionFlow(
        "move",
        first[first.traits[0]],
        first[first.traits[1]],
        0.2,
        adjust=adjust,
    )

    def build(m: FlowModel) -> None:
        m.add_flow(exit_flow)
        m.add_flow(entry_flow)
        m.add_flow(transition)

    a = FlowModel(pmap)
    build(a)
    a.stratify(q, where)

    b = FlowModel(pmap.stratify(q, where))
    build(b)

    try:
        ca = a.compile()
    except Exception as exc_a:  # noqa: BLE001 — both builds must fail the same way
        with pytest.raises(type(exc_a)):
            b.compile()
        return
    try:
        cb = b.compile()
    except Exception as exc_b:  # noqa: BLE001
        raise AssertionError(
            f"map-first build failed after stratify-in-place succeeded: {exc_b}"
        ) from exc_b

    assert ca == cb
    for name in ca.order:
        np.testing.assert_array_equal(ca.edges(name).table.codes, cb.edges(name).table.codes)


def test_stratify_mutates_in_place_and_keeps_flows() -> None:
    state, pmap = _sir()
    age = Property("age", ("young", "old"))
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("inf", state["S"], state["I"], 0.1))
    model.set_initial_population({state["S"]: 1000.0, state["I"]: 1.0})
    flows_before = model.flows
    pop_before = model._initial_population
    model.stratify(age)
    assert model.pmap is not pmap
    assert "age" in {p.name for p in model.pmap.properties}
    assert model.flows is flows_before
    assert model._initial_population is pop_before


def test_compiled_before_stratify_is_unchanged() -> None:
    state, pmap = _sir()
    age = Property("age", ("young", "old"))
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("inf", state["S"], state["I"], 0.1))
    compiled = model.compile()
    assert compiled.pmap.size == 3
    model.stratify(age)
    assert compiled.pmap.size == 3
    assert model.pmap.size == 6
    assert compiled.pmap is not model.pmap


def test_copy_branches_independently() -> None:
    state, pmap = _sir()
    age = Property("age", ("young", "old"))
    strain = Property("strain", ("a", "b"))
    core = FlowModel(pmap)
    core.add_flow(TransitionFlow("inf", state["S"], state["I"], 0.1))
    left = core.copy()
    right = core.copy()
    left.stratify(age)
    right.stratify(strain)
    left.add_flow(ExitFlow("death", Everything(), 0.01))
    assert core.pmap.size == 3
    assert len(core.flows) == 1
    assert left.pmap.size == 6
    assert len(left.flows) == 2
    assert right.pmap.size == 6
    assert len(right.flows) == 1
    assert "age" in {p.name for p in left.pmap.properties}
    assert "strain" in {p.name for p in right.pmap.properties}


def test_dest_only_equal_split_and_split_override() -> None:
    state, pmap = _sir()
    severity = Property("severity", ("mild", "severe"))
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], 1.0))
    model.stratify(severity, where=state["I"])
    cm = model.compile()
    flow = cm.flows["infection"]
    np.testing.assert_allclose(flow.weight, 0.5)
    assert flow.n_edges == 2

    model.update_flow("infection", split={severity: {"mild": 0.25, "severe": 0.75}})
    cm2 = model.compile()
    flow2 = cm2.flows["infection"]
    # Order follows dest compartment order on the map.
    mild_i = int(
        np.flatnonzero(cm2.pmap.codes[flow2.dest_idx, cm2.pmap.column_index(severity)] == 0)[0]
    )
    severe_i = int(
        np.flatnonzero(cm2.pmap.codes[flow2.dest_idx, cm2.pmap.column_index(severity)] == 1)[0]
    )
    assert flow2.weight[mild_i] == pytest.approx(0.25)
    assert flow2.weight[severe_i] == pytest.approx(0.75)


def test_update_flow_errors() -> None:
    state, pmap = _sir()
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("inf", state["S"], state["I"], 0.1))
    with pytest.raises(KeyError, match="Unknown flow"):
        model.update_flow("missing", rate=0.2)
    with pytest.raises(TypeError, match="Unknown field"):
        model.update_flow("inf", not_a_field=1)
    with pytest.raises(ValueError, match="names cannot be changed"):
        model.update_flow("inf", name="other")


def test_update_flow_replace_all_kinds() -> None:
    state, pmap = _sir()
    age = Property("age", ("young", "old"))
    pmap = pmap.stratify(age)
    transition = TransitionFlow(
        "inf",
        state["S"],
        state["I"],
        0.1,
        adjust=[Multiply(2.0)],
        pairing=TraitChain(age, (("young", "young"), ("old", "old"))),
        split=None,
    )
    exit_f = ExitFlow("death", Everything(), 0.01, adjust=[0.5])
    entry = EntryFlow(
        "birth", state["S"], 1.0, adjust=[Multiply(1.0)], split={age: {"young": 0.6, "old": 0.4}}
    )

    for flow in (transition, exit_f, entry):
        replaced = dataclasses.replace(flow, rate=Const(0.2), adjust=flow.adjust)
        assert isinstance(replaced, type(flow))
        assert replaced.rate == Const(0.2)
        assert replaced.adjust == flow.adjust
        assert replaced.name == flow.name


def test_flowref_sum_over_survives_stratify() -> None:
    state, pmap = _sir()
    age = Property("age", ("young", "old"))
    model = FlowModel(pmap)
    death = model.add_flow(ExitFlow("death", Everything(), 0.1))
    model.add_flow(EntryFlow("birth", state["S"], death.sum()))
    model.stratify(age)
    cm = model.compile()
    y = np.ones(cm.pmap.size)
    dy = np.asarray(cm.vector_field(0.0, y, {}))
    assert dy.shape == (cm.pmap.size,)


def test_initial_population_splits_evenly_over_new_property() -> None:
    state, pmap = _sir()
    age = Property("age", ("young", "old"))
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("inf", state["S"], state["I"], 0.0))
    model.set_initial_population({state["S"]: 100.0, state["I"]: 10.0})
    model.stratify(age)
    cm = model.compile()
    y0 = np.asarray(cm.initial_state({}).data)
    s_idx = cm.pmap.select(state["S"])
    i_idx = cm.pmap.select(state["I"])
    np.testing.assert_allclose(y0[s_idx], 50.0)
    np.testing.assert_allclose(y0[i_idx], 5.0)


def test_erlang_latent_stages() -> None:
    state = Property("state", ("S", "E", "I", "R"))
    pop = Property("pop", ("all",))
    pmap = PropertyMap.from_property(state).stratify(pop)
    sigma = 0.5
    stage = Property("stage", ("e1", "e2", "e3"))

    def build_map_first() -> FlowModel:
        pm = pmap.stratify(stage, where=state["E"])
        m = FlowModel(pm)
        m.add_flow(TransitionFlow("infection", state["S"], state["E"] & stage["e1"], 0.1))
        m.add_flow(
            TransitionFlow(
                "progression",
                state["E"] & stage["e3"],
                state["I"],
                sigma,
                adjust=[3.0],
            )
        )
        m.add_flow(
            TransitionFlow(
                "latent_stages",
                state["E"],
                state["E"],
                sigma,
                pairing=TraitChain(stage, (("e1", "e2"), ("e2", "e3"))),
                adjust=[3.0],
            )
        )
        m.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.2))
        return m

    core = FlowModel(pmap)
    core.add_flow(TransitionFlow("infection", state["S"], state["E"], 0.1))
    core.add_flow(TransitionFlow("progression", state["E"], state["I"], sigma))
    core.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.2))
    erlang = core.copy()
    erlang.stratify(stage, where=state["E"])
    erlang.update_flow("infection", dest=state["E"] & stage["e1"])
    erlang.update_flow("progression", source=state["E"] & stage["e3"])
    erlang.adjust_flow("progression", 3.0)
    erlang.add_flow(
        TransitionFlow(
            "latent_stages",
            state["E"],
            state["E"],
            sigma,
            pairing=TraitChain(stage, (("e1", "e2"), ("e2", "e3"))),
            adjust=[3.0],
        )
    )

    map_first = build_map_first()
    ca = erlang.compile()
    cb = map_first.compile()
    assert ca == cb

    y0_a = np.zeros(ca.pmap.size)
    y0_a[ca.pmap.select(state["S"])] = 999.0
    y0_a[ca.pmap.select(state["E"] & stage["e1"])] = 1.0
    y0_b = np.zeros(cb.pmap.size)
    y0_b[cb.pmap.select(state["S"])] = 999.0
    y0_b[cb.pmap.select(state["E"] & stage["e1"])] = 1.0
    ya = np.asarray(euler(ca.vector_field, 0.0, y0_a, {}, dt=0.1, steps=20))
    yb = np.asarray(euler(cb.vector_field, 0.0, y0_b, {}, dt=0.1, steps=20))
    np.testing.assert_allclose(ya, yb)


def test_align_rate_error_mentions_parent_row() -> None:
    state, pmap = _sir()
    age = Property("age", ("young", "old"))
    stratified = pmap.stratify(age)
    parent_rate = np.arange(pmap.size, dtype=np.float64)
    with pytest.raises(ValueError, match="parent_row") as exc:
        _align_rate(
            parent_rate,
            gather_idx=np.arange(stratified.size, dtype=np.int32),
            n_edges=stratified.size,
            pmap=stratified,
            pair_src_codes=None,
            pair_dest_codes=None,
            pair_n_traits=None,
        )
    assert "stratified from" in str(exc.value)


def test_adjust_flow_requires_args() -> None:
    state, pmap = _sir()
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("inf", state["S"], state["I"], 0.1))
    with pytest.raises(ValueError, match="at least one"):
        model.adjust_flow("inf")
