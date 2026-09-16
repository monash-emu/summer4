"""EpiModel frontend digest-equality and escape hatches."""

from __future__ import annotations

import numpy as np
import pytest

from summer4 import (
    FlowModel,
    Param,
    Property,
    PropertyMap,
    TransitionFlow,
)
from summer4.epi import EpiModel, ForceOfInfection, MixingMatrix


def test_epimodel_digest_equals_declarative() -> None:
    pytest.importorskip("jax")
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age)
    K = np.eye(2)

    epi = EpiModel(pmap, infectious=state["I"])
    epi.set_mixing_matrix(age, K, check_reciprocal=False)
    epi.add_infectiousness_adjustments(age, {"young": 0.8, "old": 1.2}, normalize=None)
    epi.add_infection_frequency_flow("infection", state["S"], state["I"], Param("beta"))
    epi.add_transition_flow("recovery", state["I"], state["R"], 0.1)
    cm_epi = epi.compile()

    decl = FlowModel(pmap)
    foi = ForceOfInfection(
        "infection",
        infectious=state["I"],
        group_by=age,
        mixing=MixingMatrix(age, K, check_reciprocal=False),
        kind="frequency",
        contact_rate=Param("beta"),
        infectiousness={age["young"]: 0.8, age["old"]: 1.2},
        normalize_infectiousness=None,
    )
    decl.add_flow(TransitionFlow("infection", state["S"], state["I"], foi))
    decl.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.1))
    cm_decl = decl.compile()
    assert cm_epi == cm_decl


def test_epimodel_escape_hatches() -> None:
    state = Property("state", ("S", "I"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age)
    epi = EpiModel(pmap, infectious=state["I"])
    epi.set_mixing_matrix(age, np.eye(2), check_reciprocal=False)
    epi.add_infection_frequency_flow("infection", state["S"], state["I"], Param("beta"))
    assert epi.foi("infection").name == "infection"
    assert epi.flow_model is epi._flow_model
    epi.flow_model.add_flow(TransitionFlow("recovery", state["I"], state["S"], 0.1))
    cm = epi.compile()
    assert "recovery" in cm.order


def test_epimodel_no_infection_matches_flowmodel() -> None:
    state = Property("state", ("S", "I"))
    pmap = PropertyMap.from_property(state)
    epi = EpiModel(pmap)
    epi.add_transition_flow("recovery", state["I"], state["S"], 0.1)
    bare = FlowModel(pmap)
    bare.add_flow(TransitionFlow("recovery", state["I"], state["S"], 0.1))
    assert epi.compile() == bare.compile()


def test_epimodel_duplicate_flow_raises() -> None:
    state = Property("state", ("S", "I"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age)
    epi = EpiModel(pmap, infectious=state["I"])
    epi.set_mixing_matrix(age, np.eye(2), check_reciprocal=False)
    epi.add_infection_frequency_flow("infection", state["S"], state["I"], 0.3)
    with pytest.raises(ValueError, match="Duplicate"):
        epi.add_infection_frequency_flow("infection", state["S"], state["I"], 0.3)


def test_param_is_field_ref() -> None:
    from summer4.flows.rates import FieldRef

    p = Param("beta")
    assert isinstance(p, FieldRef)
    assert p.path == ("beta",)
