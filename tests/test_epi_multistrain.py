"""Multi-strain / multi-disease FOI non-interference."""

from __future__ import annotations

import numpy as np
import pytest

from summer4 import (
    Compartments,
    FlowModel,
    Property,
    PropertyMap,
    SavePlan,
    SaveRequest,
    TransitionFlow,
)
from summer4.epi import ForceOfInfection, MixingMatrix
from summer4.results.plan import GroupedOutput


def test_two_diseases_non_interference() -> None:
    """Headline gate: each disease alone matches the joint run with the other off."""
    pytest.importorskip("jax")
    state = Property("state", ("S", "I_a", "I_b", "R"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age)
    K = np.eye(2)

    def build(beta_a: float, beta_b: float) -> object:
        m = FlowModel(pmap)
        foi_a = ForceOfInfection(
            "inf_a",
            infectious=state["I_a"],
            group_by=age,
            mixing=MixingMatrix(age, K, check_reciprocal=False),
            kind="frequency",
            contact_rate=beta_a,
        )
        foi_b = ForceOfInfection(
            "inf_b",
            infectious=state["I_b"],
            group_by=age,
            mixing=MixingMatrix(age, K, check_reciprocal=False),
            kind="frequency",
            contact_rate=beta_b,
        )
        m.add_flow(TransitionFlow("inf_a", state["S"], state["I_a"], foi_a))
        m.add_flow(TransitionFlow("inf_b", state["S"], state["I_b"], foi_b))
        m.add_flow(TransitionFlow("rec_a", state["I_a"], state["R"], 0.1))
        m.add_flow(TransitionFlow("rec_b", state["I_b"], state["R"], 0.1))
        return m.compile()

    # S_young, S_old, Ia_y, Ia_o, Ib_y, Ib_o, R_y, R_o
    y0 = np.array([800.0, 700.0, 20.0, 10.0, 15.0, 5.0, 0.0, 0.0])
    plan = SavePlan(
        requests={"comp": SaveRequest(Compartments())},
        ts=np.linspace(0.0, 20.0, 21),
    )
    solo_a = build(0.4, 0.0).run({}, y0, t0=0.0, t1=20.0, dt=0.1, save=plan, solver="euler")
    solo_b = build(0.0, 0.35).run({}, y0, t0=0.0, t1=20.0, dt=0.1, save=plan, solver="euler")
    joint_a = build(0.4, 0.0).run({}, y0, t0=0.0, t1=20.0, dt=0.1, save=plan, solver="euler")
    joint_b = build(0.0, 0.35).run({}, y0, t0=0.0, t1=20.0, dt=0.1, save=plan, solver="euler")
    np.testing.assert_array_equal(
        np.asarray(solo_a["comp"].values.data),
        np.asarray(joint_a["comp"].values.data),
    )
    np.testing.assert_array_equal(
        np.asarray(solo_b["comp"].values.data),
        np.asarray(joint_b["comp"].values.data),
    )


def test_per_trait_builds_one_foi_per_strain() -> None:
    state = Property("state", ("S", "I"))
    strain = Property("strain", ("alpha", "beta"))
    age = Property("age", ("young", "old"))
    fois = ForceOfInfection.per_trait(
        strain,
        name_prefix="inf",
        infectious=state["I"],
        group_by=age,
        contact_rate=0.3,
    )
    assert len(fois) == 2
    assert [f.name for f in fois] == ["inf_alpha", "inf_beta"]
    # Selectors differ by strain trait.
    assert fois[0].infectious != fois[1].infectious


def test_each_foi_saves_own_trace() -> None:
    pytest.importorskip("jax")
    state = Property("state", ("S", "I"))
    strain = Property("strain", ("alpha", "beta"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age).stratify(strain, where=state["I"])
    m = FlowModel(pmap)
    for foi in ForceOfInfection.per_trait(
        strain,
        infectious=state["I"],
        group_by=age,
        mixing=MixingMatrix(age, np.eye(2), check_reciprocal=False),
        contact_rate=0.3,
    ):
        # Dest: I of that strain — use infectious selector as dest approximation.
        m.add_flow(TransitionFlow(foi.name, state["S"], foi.infectious, foi))
    cm = m.compile()
    assert "infection_alpha" in cm.capture_meta
    assert "infection_beta" in cm.capture_meta
    y0 = np.ones(pmap.size)
    plan = SavePlan(
        requests={
            "a": SaveRequest(GroupedOutput("infection_alpha")),
            "b": SaveRequest(GroupedOutput("infection_beta")),
        },
        ts=np.array([0.0, 1.0]),
    )
    res = cm.run({}, y0, t0=0.0, t1=1.0, dt=0.1, save=plan, solver="euler")
    assert res["a"].dims == ("time", "age")
    assert res["b"].dims == ("time", "age")
