"""OutputSet: a named DAG over save leaves, and Result.to_frame."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import (
    Compartments,
    FlowMass,
    FlowModel,
    Output,
    OutputSet,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    Result,
    SavePlan,
    SaveRequest,
    Target,
    TargetSet,
    TimeAxis,
    TransitionFlow,
    tanh,
)
from summer4.results.outputset import OutputExpr


def _axis(times: np.ndarray) -> TimeAxis:
    return TimeAxis(values=times, epoch=None, kind="explicit")


def _scalar_result(name: str, values: np.ndarray) -> Result:
    times = np.arange(values.shape[0], dtype=np.float64)
    axis = _axis(times)
    output = Output(times=axis, values=values, dims=("time",))
    return Result(times=axis, outputs={name: output})


def test_plan_saves_leaves_only_and_dedups_specs() -> None:
    state = Property("state", ("S", "I"))
    outputs = OutputSet()
    outputs["by_age"] = Compartments(where=state["I"])
    outputs["infected"] = Compartments(where=state["I"]).total()
    outputs["population"] = Compartments().total()
    outputs["prevalence"] = outputs.ref("infected") / outputs.ref("population")
    plan = outputs.plan()
    assert set(plan.requests) == {"by_age", "population"}
    assert plan.requests["by_age"].what == Compartments(where=state["I"])
    assert "prevalence" not in plan.requests
    assert "infected" not in plan.requests


def test_several_leaves_in_one_expression_get_suffixed_keys() -> None:
    state = Property("state", ("S", "I"))
    outputs = OutputSet()
    outputs["ratio"] = Compartments(where=state["I"]).total() / Compartments().total()
    assert set(outputs.plan().requests) == {"ratio", "ratio__2"}


def test_plan_keeps_existing_times_and_rejects_a_different_quantity() -> None:
    state = Property("state", ("S", "I"))
    outputs = OutputSet()
    outputs["I"] = Compartments(where=state["I"])
    base = SavePlan(
        requests={"I": SaveRequest(Compartments(where=state["I"]), ts=np.array([0.0, 2.0]))}
    )
    merged = outputs.plan(base)
    np.testing.assert_array_equal(merged.requests["I"].ts, np.array([0.0, 2.0]))

    clash = SavePlan(requests={"I": SaveRequest(FlowMass("death"))})
    with pytest.raises(ValueError, match="already"):
        outputs.plan(clash)


def test_plan_merges_with_target_times() -> None:
    state = Property("state", ("S", "I"))
    outputs = OutputSet()
    outputs["I"] = Compartments(where=state["I"])
    targets = TargetSet(targets=(Target(key="I", times=np.array([0.0, 4.0]), values=np.zeros(2)),))
    plan = targets.plan(outputs.plan(SavePlan()))
    np.testing.assert_array_equal(plan.requests["I"].ts, np.array([0.0, 4.0]))
    again = outputs.plan(plan)
    np.testing.assert_array_equal(again.requests["I"].ts, np.array([0.0, 4.0]))


def test_cycle_and_missing_ref_are_named() -> None:
    outputs = OutputSet()
    outputs["a"] = outputs.ref("b")
    outputs["b"] = outputs.ref("a") + 1.0
    with pytest.raises(ValueError, match="a -> b -> a"):
        outputs.plan()

    missing = OutputSet()
    missing["a"] = missing.ref("nope")
    with pytest.raises(KeyError, match="nope"):
        missing.evaluate(_scalar_result("a", np.ones(2)), None)


def test_forward_ref_and_parameter_scale_under_jit_and_grad() -> None:
    outputs = OutputSet()
    outputs["scaled"] = 2.0 * outputs.ref("pop") * Param("scale")
    outputs["warped"] = outputs.ref("pop") * tanh(Param("s"))
    outputs["pop"] = Compartments()
    values = np.array([4.0, 4.0, 4.0])
    result = _scalar_result("pop", values)

    named = outputs.evaluate(result, {"scale": 2.0, "s": 0.0})
    np.testing.assert_allclose(np.asarray(named["scaled"].values), 16.0)
    np.testing.assert_allclose(np.asarray(named["warped"].values), 0.0, atol=1e-6)
    assert list(named.keys()) == ["scaled", "warped", "pop"]

    def loss(scale: Any) -> Any:
        out = outputs.evaluate(result, {"scale": scale, "s": 0.0})
        return jnp.sum(jnp.asarray(out["scaled"].values))

    # 2 * pop * scale, pop is 4 three times: value 48 at scale 2, grad 24.
    assert float(jax.jit(loss)(jnp.array(2.0))) == pytest.approx(48.0)
    assert float(jax.grad(loss)(jnp.array(2.0))) == pytest.approx(24.0)


def test_assignment_rejects_a_bare_number() -> None:
    outputs = OutputSet()
    with pytest.raises(TypeError, match="save spec"):
        outputs["x"] = 1.0  # type: ignore[assignment]


def test_chain_on_a_spec_is_an_expression() -> None:
    expr = Compartments().total()
    assert isinstance(expr, OutputExpr)
    assert expr.kind == "call"
    assert expr.method == "total"


def _aged_model() -> tuple[Property, Property, Any, PropertyData]:
    state = Property("state", ("S", "I"))
    age = Property("age", ("child", "adult"))
    pmap = PropertyMap.from_property(state).stratify(age)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], 0.2))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.ones(pmap.size))
    y0 = y0.at[state["S"] & age["child"]].set(50.0)
    y0 = y0.at[state["I"] & age["child"]].set(5.0)
    return state, age, cm, y0


def test_evaluate_matches_manual_output_algebra() -> None:
    state, _age, cm, y0 = _aged_model()
    outputs = OutputSet()
    outputs["by_age"] = Compartments(where=state["I"])
    outputs["infected"] = Compartments(where=state["I"]).total()
    outputs["population"] = Compartments().total()
    outputs["incidence"] = FlowMass("infection").total().midpoint()
    outputs["prevalence"] = outputs.ref("infected") / outputs.ref("population")
    outputs["cum"] = outputs.ref("incidence").cumulative(start=2.0)
    outputs["scaled"] = outputs.ref("incidence") * Param("scale")

    plan = outputs.plan(SavePlan())
    assert set(plan.requests) == {"by_age", "population", "incidence"}
    res = cm.run({}, y0, t0=0.0, steps=5, dt=1.0, save=plan)
    # The solve stores the raw leaf under the first name that used the spec.
    assert np.asarray(res["population"].values.data).ndim == 2
    assert np.asarray(res["incidence"].values.data).ndim == 2

    named = outputs.evaluate(res, {"scale": 3.0})
    infected = res["by_age"].total()
    population = res["population"].total()
    raw_total = res["incidence"].total()
    np.testing.assert_allclose(np.asarray(named["infected"].values), np.asarray(infected.values))
    np.testing.assert_allclose(
        np.asarray(named["population"].values), np.asarray(population.values)
    )
    assert np.asarray(named["population"].values).ndim == 1
    raw = np.asarray(raw_total.values)
    mid = np.asarray(named["incidence"].values)
    assert mid[0] == pytest.approx(float(raw[0]))
    np.testing.assert_allclose(mid[1:], 0.5 * (raw[1:] + raw[:-1]))
    np.testing.assert_allclose(
        np.asarray(named["prevalence"].values),
        np.asarray(infected.values) / np.asarray(population.values),
    )
    cum = np.asarray(named["cum"].values)
    assert cum[1] == 0.0
    assert cum[2] == pytest.approx(mid[2])
    np.testing.assert_allclose(np.asarray(named["scaled"].values), 3.0 * mid)

    wide = named.to_frame(shape="wide", backend="pandas")
    assert list(wide.columns)[:4] == [
        "by_age[state=I_age=child]",
        "by_age[state=I_age=adult]",
        "infected",
        "population",
    ]
    np.testing.assert_allclose(
        wide["prevalence"].to_numpy(), np.asarray(named["prevalence"].values)
    )

    long = named.to_frame(names=["by_age"], shape="long", backend="polars")
    assert long.height == len(np.asarray(named["by_age"].times.values)) * 2
    assert "age" in long.columns
    assert set(long["age"].unique().to_list()) == {"child", "adult"}


def test_wide_frame_rejects_mixed_times() -> None:
    outputs = OutputSet()
    outputs["pop"] = Compartments()
    outputs["early"] = outputs.ref("pop").at_times(np.array([0.0, 1.0]))
    result = _scalar_result("pop", np.arange(4, dtype=np.float64))
    named = outputs.evaluate(result, None)
    with pytest.raises(ValueError, match="same times"):
        named.to_frame(shape="wide")
    long = named.to_frame(shape="long", backend="pandas")
    assert set(long["output"]) == {"pop", "early"}
    assert len(long) == 4 + 2
