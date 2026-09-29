"""Regression: a plan-level ``SavePlan.ts`` is the default save grid for every solver."""

from __future__ import annotations

import numpy as np
import pytest

from summer4 import (
    Compartments,
    CompiledModel,
    FlowModel,
    Property,
    PropertyMap,
    SavePlan,
    SaveRequest,
    TransitionFlow,
)
from summer4.solvers import SolveSpec


def _model() -> CompiledModel:
    state = Property("state", ("S", "I"))
    model = FlowModel(PropertyMap.from_property(state))
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], 0.1))
    return model.compile()


@pytest.mark.parametrize("solver", ["euler", "tsit5"])
def test_plan_level_ts_sets_the_save_grid(solver: str) -> None:
    cm = _model()
    ts = np.linspace(0.0, 40.0, 81)
    plan = SavePlan(requests={"c": SaveRequest(Compartments())}, ts=ts)
    res = cm.run({}, np.array([1.0, 0.0]), t0=0.0, t1=40.0, dt=0.1, save=plan, solver=solver)
    np.testing.assert_allclose(np.asarray(res["c"].times.values), ts)
    np.testing.assert_allclose(np.asarray(res.times.values), ts)
    assert np.asarray(res["c"].values.data).shape[0] == 81


def test_request_ts_still_overrides_the_plan_ts() -> None:
    cm = _model()
    plan = SavePlan(
        requests={
            "plan": SaveRequest(Compartments()),
            "own": SaveRequest(Compartments(), ts=np.array([0.0, 10.0, 20.0])),
        },
        ts=np.linspace(0.0, 40.0, 5),
    )
    res = cm.run({}, np.array([1.0, 0.0]), t0=0.0, t1=40.0, dt=0.1, save=plan, solver="euler")
    np.testing.assert_allclose(np.asarray(res["plan"].times.values), np.linspace(0.0, 40.0, 5))
    np.testing.assert_allclose(np.asarray(res["own"].times.values), [0.0, 10.0, 20.0])


def test_save_ts_falls_back_to_the_step_grid() -> None:
    spec = SolveSpec.window(t0=0.0, t1=1.0, dt=0.25)
    np.testing.assert_allclose(spec.save_ts(SavePlan()), [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(spec.save_ts(SavePlan(ts=np.array([0.0, 1.0]))), [0.0, 1.0])
