"""Step 40: any rate expression is saved during the solve (``SavedExpr``)."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from summer4 import (
    CompiledModel,
    FlowModel,
    FlowRef,
    GroupedOutput,
    OutputSet,
    Param,
    Property,
    PropertyMap,
    Reduce,
    SavedExpr,
    SavePlan,
    SaveRequest,
    TableInterp,
    Time,
    TransitionFlow,
    exp,
)
from summer4.epi import ForceOfInfection
from summer4.flows.rates import ArrayConst, Capture

STATE = Property("state", ("S", "I", "R"))
AGE = Property("age", ("young", "old"))
PMAP = PropertyMap.from_property(STATE).stratify(AGE)
Y0 = np.array([90.0, 80.0, 10.0, 20.0, 0.0, 0.0])
PARAMS = {"b": 0.3, "g": 0.1, "k": 2.0}
EULER: dict[str, Any] = {"t0": 0.0, "t1": 5.0, "dt": 1.0, "solver": "euler"}
BETA = Param("b") * exp(-0.1 * Time())


def _sir(rate: Any = BETA) -> CompiledModel:
    model = FlowModel(PMAP)
    model.add_flow(TransitionFlow("inf", STATE["S"], STATE["I"], rate))
    model.add_flow(TransitionFlow("rec", STATE["I"], STATE["R"], Param("g")))
    return model.compile()


def _beta(t: np.ndarray) -> np.ndarray:
    return 0.3 * np.exp(-0.1 * t)


@pytest.mark.parametrize("solver", ["euler", "tsit5"])
def test_seasonal_rate_is_saved_at_save_times(solver: str) -> None:
    outputs = OutputSet()
    outputs["beta"] = BETA
    named = _sir().run(PARAMS, Y0, outputs=outputs, **{**EULER, "solver": solver})
    t = np.asarray(named["beta"].times.values)
    assert named["beta"].dims == ("time",)
    np.testing.assert_allclose(np.asarray(named["beta"].array()), _beta(t), rtol=1e-12)


def test_saved_expression_under_jit() -> None:
    outputs = OutputSet()
    outputs["beta"] = BETA
    cm = _sir()

    @jax.jit
    def beta_series(b: Any) -> Any:
        return cm.run({**PARAMS, "b": b}, Y0, outputs=outputs, **EULER)["beta"].array()

    t = np.arange(6.0)
    np.testing.assert_allclose(np.asarray(beta_series(0.3)), _beta(t), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(beta_series(0.6)), 2 * _beta(t), rtol=1e-12)


def test_raw_saved_expr_in_a_save_plan() -> None:
    plan = SavePlan(
        requests={
            "beta": SaveRequest(SavedExpr(BETA)),
            "infectious": SaveRequest(SavedExpr(Reduce(where=STATE["I"]))),
            "b": SaveRequest(SavedExpr(Param("b"))),
        }
    )
    result = _sir().run(PARAMS, Y0, save=plan, **EULER)
    comps = result.final_state.data
    assert np.isclose(float(result["infectious"].array()[-1]), float(comps[2] + comps[3]))
    np.testing.assert_allclose(np.asarray(result["b"].array()), np.full(6, 0.3))


def test_step_stage_part_of_a_mixed_expression_is_lifted() -> None:
    outputs = OutputSet()
    outputs.flow("inc", "inf")
    outputs["scaled"] = Param("k") * Time() * outputs.ref("inc")
    outputs["per_g"] = outputs.ref("inc") / Param("g")
    plan = outputs.plan()
    kinds = {key: type(req.what).__name__ for key, req in plan.requests.items()}
    assert kinds == {"inc": "FlowMass", "scaled": "SavedExpr"}
    saved = plan.requests["scaled"].what
    assert isinstance(saved, SavedExpr) and saved.expr == Param("k") * Time()
    named = _sir().run(PARAMS, Y0, outputs=outputs, **EULER)
    t = np.asarray(named["inc"].times.values)
    inc = np.asarray(named["inc"].array())
    np.testing.assert_allclose(np.asarray(named["scaled"].array()), 2.0 * t * inc)
    np.testing.assert_allclose(np.asarray(named["per_g"].array()), inc / 0.1)


def test_a_parameter_only_output_raises() -> None:
    outputs = OutputSet()
    with pytest.raises(ValueError, match="not a time"):
        outputs["c"] = Param("b") * 2.0


def test_equal_expressions_share_one_save() -> None:
    outputs = OutputSet()
    outputs["a"] = Param("b") * exp(-0.1 * Time())
    outputs["b"] = Param("b") * exp(-0.1 * Time())
    assert len(outputs.plan().requests) == 1
    assert SavedExpr(BETA) == SavedExpr(Param("b") * exp(-0.1 * Time()))
    assert hash(SavedExpr(BETA)) == hash(SavedExpr(Param("b") * exp(-0.1 * Time())))
    assert SavedExpr(BETA) != SavedExpr(BETA, over=AGE)


def test_flow_reads_cover_saved_expressions() -> None:
    plan = SavePlan(requests={"total_inf": SaveRequest(SavedExpr(FlowRef("inf").sum_by()))})
    assert plan.flow_reads() == frozenset({"inf"})
    cm = _sir()
    reduced = cm.run(PARAMS, Y0, save=plan, **EULER)
    full = cm.run(PARAMS, Y0, **EULER)
    np.testing.assert_allclose(
        np.asarray(reduced["total_inf"].array()),
        np.asarray(full["inf"].array()).sum(axis=-1),
        rtol=1e-12,
    )


# --- grouped values -------------------------------------------------------------------------


def _foi_model(foi: Any) -> CompiledModel:
    model = FlowModel(PMAP)
    model.add_flow(TransitionFlow("inf", STATE["S"], STATE["I"], foi))
    return model.compile()


def test_foi_by_age_saved_by_expression_equals_the_capture_route() -> None:
    foi = ForceOfInfection("foi", infectious=STATE["I"], contact_rate=Param("b"), group_by=AGE)
    cm = _foi_model(Capture("foi_by_age", foi))
    outputs = OutputSet()
    outputs["saved"] = foi
    outputs["given"] = SavedExpr(foi, over=AGE)
    outputs["captured"] = GroupedOutput("foi_by_age")
    named = cm.run(PARAMS, Y0, outputs=outputs, **EULER)
    assert named["saved"].dims == ("time", "age")
    captured = np.asarray(named["captured"].array())
    assert captured.shape == (6, 2)
    np.testing.assert_allclose(np.asarray(named["saved"].array()), captured, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(named["given"].array()), captured, rtol=1e-12)
    young = np.asarray(named["saved"].select(AGE["young"]).sum_by().array())
    np.testing.assert_allclose(young, captured[:, 0], rtol=1e-12)


def test_table_interp_over_is_inferred() -> None:
    table = TableInterp(
        "linear",
        ArrayConst(np.array([0.0, 10.0])),
        ArrayConst(np.array([[1.0, 2.0], [3.0, 6.0]])),
        over=AGE,
    )
    outputs = OutputSet()
    outputs["table"] = table * Param("g")
    named = _sir().run(PARAMS, Y0, outputs=outputs, **EULER)
    t = np.asarray(named["table"].times.values)
    want = 0.1 * np.stack([1.0 + 0.2 * t, 2.0 + 0.4 * t], axis=1)
    np.testing.assert_allclose(np.asarray(named["table"].array()), want, rtol=1e-12)


def test_wrong_over_and_per_compartment_values_raise() -> None:
    foi = ForceOfInfection("foi", infectious=STATE["I"], contact_rate=Param("b"), group_by=AGE)
    cm = _foi_model(foi)
    plan = SavePlan(requests={"x": SaveRequest(SavedExpr(foi, over=STATE))})
    with pytest.raises(TypeError, match="grouped by"):
        cm.run(PARAMS, Y0, save=plan, **EULER)
    edges = SavePlan(requests={"x": SaveRequest(SavedExpr(FlowRef("inf") * 1.0))})
    with pytest.raises(TypeError, match="Compartments or FlowMass"):
        cm.run(PARAMS, Y0, save=edges, **EULER)


def test_saved_expression_shares_the_vector_field_memo() -> None:
    """A saved expression that is also a flow's rate is one evaluation per save, not two."""
    calls: list[int] = []

    def counted(t: Any) -> Any:
        calls.append(1)
        return 0.3 * np.exp(0.0) + 0.0 * t

    from summer4 import defer

    node = defer(counted)(Time())
    cm = _sir(node)
    plan = SavePlan(requests={"node": SaveRequest(SavedExpr(node))})
    ctx = cm.observe(0.0, Y0, PARAMS, keep=plan.flow_reads())
    before = len(calls)
    assert ctx.evaluate is not None
    ctx.evaluate(node)
    assert len(calls) == before  # memo hit: the flow already evaluated this node object
