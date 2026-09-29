"""Regression: a plan-level ``SavePlan(ts=...)`` is the save grid for every backend.

Step 29 (``feat/solve-composable``) grouped requests on ``SolveSpec.default_ts()``
(every step), dropping ``SavePlan.ts``: an Euler run with ``dt=0.1`` over 40 days
saved 401 rows instead of the 81 requested.
"""

from __future__ import annotations

import numpy as np
import pytest

from summer4 import (
    Compartments,
    FlowMass,
    FlowModel,
    Property,
    PropertyMap,
    SavePlan,
    SaveRequest,
    TransitionFlow,
)
from summer4.solvers import SolveSpec


def _model() -> tuple[object, Property]:
    state = Property("state", ("S", "I"))
    model = FlowModel(PropertyMap.from_property(state))
    model.add_flow(TransitionFlow("infect", state["S"], state["I"], 0.1))
    return model.compile(), state


@pytest.mark.parametrize("solver", ["euler", "tsit5"])
def test_plan_ts_is_the_save_grid(solver: str) -> None:
    cm, _state = _model()
    ts = np.linspace(0.0, 40.0, 81)
    plan = SavePlan(
        requests={
            "comp": SaveRequest(Compartments()),
            "flow": SaveRequest(FlowMass(flow="infect")),
        },
        ts=ts,
    )
    res = cm.run(  # type: ignore[attr-defined]
        {}, np.array([1.0, 0.0]), t0=0.0, t1=40.0, dt=0.1, save=plan, solver=solver
    )
    for key in ("comp", "flow"):
        np.testing.assert_allclose(np.asarray(res[key].times.values), ts)
        assert np.asarray(res[key].values.data).shape[0] == ts.size
    np.testing.assert_allclose(np.asarray(res.times.values), ts)
    # S decays at rate 0.1: the saved values sit on the requested times.
    s_saved = np.asarray(res["comp"].values.data)[:, 0]
    np.testing.assert_allclose(s_saved, np.exp(-0.1 * ts), rtol=2e-2)


def test_request_ts_still_overrides_plan_ts() -> None:
    cm, _state = _model()
    plan = SavePlan(
        requests={
            "coarse": SaveRequest(Compartments()),
            "fine": SaveRequest(Compartments(), ts=np.linspace(0.0, 10.0, 101)),
        },
        ts=np.linspace(0.0, 10.0, 11),
    )
    res = cm.run(  # type: ignore[attr-defined]
        {}, np.array([1.0, 0.0]), t0=0.0, t1=10.0, dt=0.1, save=plan, solver="euler"
    )
    assert np.asarray(res["coarse"].times.values).size == 11
    assert np.asarray(res["fine"].times.values).size == 101


def test_save_ts_falls_back_to_the_step_grid() -> None:
    spec = SolveSpec.window(t0=0.0, t1=1.0, dt=0.25)
    no_ts = SavePlan(requests={"comp": SaveRequest(Compartments())})
    np.testing.assert_allclose(spec.save_ts(no_ts), spec.default_ts())
    with_ts = SavePlan(requests={"comp": SaveRequest(Compartments())}, ts=np.array([0.0, 1.0]))
    np.testing.assert_allclose(spec.save_ts(with_ts), [0.0, 1.0])
