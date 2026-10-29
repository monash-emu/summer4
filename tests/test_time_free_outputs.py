"""Time-free outputs broadcast over time, and the lazy escape to plain arrays.

plans/time-free-outputs.plan.md; composability findings CX39 and CX40.
"""

from __future__ import annotations

import math
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
    Split,
    Target,
    TargetSet,
    TimeAxis,
    TransitionFlow,
    defer,
)
from summer4.epi import BayesianModel, ForceOfInfection, Uniform
from summer4.epi.calibration import NormalLikelihood
from summer4.flows.compiled import CompiledModel

STATE = Property("state", ("S", "I", "R"))
AGE = Property("age", ("child", "adult"))
RUN = dict(t0=0.0, t1=70.0, dt=1.0, solver="euler")
PARAMS = {"contact_rate": 0.2, "recovery_time": 20.0}


def _axis(n: int, t0: float = 0.0) -> TimeAxis:
    return TimeAxis(values=t0 + np.arange(n, dtype=np.float64), epoch=None, kind="explicit")


def _series(values: np.ndarray, t0: float = 0.0) -> Output:
    return Output(times=_axis(values.shape[0], t0), values=jnp.asarray(values), dims=("time",))


def _trapezoid(y: np.ndarray, t: np.ndarray) -> Any:
    return np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(t)[(...,) + (None,) * (y.ndim - 1)], axis=0)


def _model() -> CompiledModel:
    model = FlowModel(PropertyMap.from_property(STATE).stratify(AGE))
    foi = ForceOfInfection("foi", infectious=STATE["I"], contact_rate=Param("contact_rate"))
    model.add_flow(TransitionFlow("infection", STATE["S"], STATE["I"], foi))
    model.add_flow(TransitionFlow("recovery", STATE["I"], STATE["R"], 1.0 / Param("recovery_time")))
    model.set_initial_population(
        {AGE["child"]: 1000.0, AGE["adult"]: 1500.0},
        [Split(STATE, {"S": 0.9, "I": 0.1, "R": 0.0})],
    )
    return model.compile()


def _solve(outputs: OutputSet, params: dict[str, float] | None = None) -> Result:
    cm = _model()
    params = PARAMS if params is None else params
    result = cm.run(params, save=outputs.plan(), **RUN)
    return outputs.evaluate(result, params)


# --- 1. time-free outputs broadcast over time -----------------------------------------


def test_eager_series_over_its_integral_matches_hand_values() -> None:
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])
    rate = _series(y)
    share = rate / rate.integrate()
    total = _trapezoid(y, np.arange(6.0))
    np.testing.assert_allclose(np.asarray(share.values), y / total)
    assert share.dims == ("time",)
    np.testing.assert_array_equal(np.asarray(share.times.values), np.arange(6.0))
    # Either order: the timed side's axis wins.
    flipped = rate.integrate() / rate
    np.testing.assert_array_equal(np.asarray(flipped.times.values), np.arange(6.0))
    np.testing.assert_allclose(np.asarray(flipped.values)[1:], total / y[1:])


def test_series_minus_value_at_a_time() -> None:
    y = np.array([3.0, 5.0, 8.0, 13.0])
    rate = _series(y)
    np.testing.assert_allclose(np.asarray((rate - rate.at(2.0)).values), y - 8.0)


def test_two_series_on_different_grids_still_raise() -> None:
    a = _series(np.ones(4))
    b = _series(np.ones(4), t0=1.0)
    with pytest.raises(ValueError, match="time axes differ"):
        a + b


def test_two_time_free_outputs_combine_whatever_their_anchors() -> None:
    rate = _series(np.array([1.0, 2.0, 3.0]))
    ratio = rate.integrate() / rate.at(0.0)
    assert ratio.dims == ()
    np.testing.assert_allclose(float(ratio.values), 4.0)


def test_lazy_share_of_whole_period_total_matches_eager() -> None:
    outputs = OutputSet()
    outputs["rate"] = FlowMass("infection").sum_by()
    outputs["total"] = outputs.ref("rate").integrate()
    outputs["share"] = outputs.ref("rate") / outputs.ref("total")
    outputs["inline"] = FlowMass("infection").sum_by() / FlowMass("infection").sum_by().integrate()
    named = _solve(outputs)
    rate = np.asarray(named["rate"].values)
    t = np.asarray(named["rate"].times.values)
    expected = rate / _trapezoid(rate, t)
    np.testing.assert_allclose(np.asarray(named["share"].values), expected, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(named["inline"].values), expected, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(named["share"].times.values), t)
    np.testing.assert_allclose(float(_trapezoid(expected, t)), 1.0, rtol=1e-6)


def test_lazy_at_on_refs_and_save_specs() -> None:
    outputs = OutputSet()
    outputs["prev"] = Compartments(where=STATE["I"]).sum_by()
    outputs["at10"] = Compartments(where=STATE["I"]).sum_by().at(10.0)
    outputs["change"] = outputs.ref("prev") - outputs.ref("prev").at(10.0)
    named = _solve(outputs)
    prev = np.asarray(named["prev"].values)
    assert named["at10"].dims == ()
    np.testing.assert_allclose(float(named["at10"].values), prev[10])
    np.testing.assert_allclose(np.asarray(named["change"].values), prev - prev[10])


def test_per_age_over_all_age_total_keeps_the_map() -> None:
    outputs = OutputSet()
    outputs["by_age"] = FlowMass("infection").sum_by(AGE, side="dest")
    outputs["share_of_all"] = outputs.ref("by_age") / outputs.ref("by_age").integrate().sum_by()
    outputs["attack_share"] = (
        outputs.ref("by_age").integrate() / outputs.ref("by_age").integrate().sum_by()
    )
    named = _solve(outputs)
    by_age = np.asarray(named["by_age"].values.data)
    t = np.asarray(named["by_age"].times.values)
    total = _trapezoid(by_age.sum(axis=1), t)
    share = named["share_of_all"]
    assert share.dims == ("time", "group")
    assert isinstance(share.values, PropertyData)
    np.testing.assert_allclose(np.asarray(share.values.data), by_age / total, rtol=1e-6)
    attack = named["attack_share"]
    assert attack.dims == ("group",)
    assert isinstance(attack.values, PropertyData)
    np.testing.assert_allclose(float(np.sum(np.asarray(attack.values.data))), 1.0, rtol=1e-6)


def test_at_times_on_a_time_free_output_raises() -> None:
    total = _series(np.array([1.0, 2.0, 3.0])).integrate()
    with pytest.raises(ValueError, match="reduced over time"):
        total.at_times([1.0])


def test_target_on_a_time_free_output_names_the_key() -> None:
    outputs = OutputSet()
    outputs["rate"] = FlowMass("infection").sum_by()
    outputs["total"] = outputs.ref("rate").integrate()
    named = _solve(outputs)
    targets = TargetSet((Target(key="total", times=[70.0], values=[1.0]),))
    with pytest.raises(ValueError, match="'total' names a time-free output"):
        targets.residuals(named)


# --- 3. array(), scalar(), with_values() ----------------------------------------------


def test_eager_array_scalar_and_with_values() -> None:
    y = np.array([1.0, 2.0, 3.0])
    rate = _series(y)
    np.testing.assert_allclose(np.asarray(rate.array()), y)
    total = rate.integrate()
    np.testing.assert_allclose(float(total.scalar()), 4.0)
    with pytest.raises(ValueError, match="0-d output"):
        rate.scalar()
    back = rate.with_values(rate.array() / total.scalar())
    np.testing.assert_allclose(np.asarray(back.values), np.asarray((rate / total).values))
    assert back.times is rate.times
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        rate.with_values(jnp.ones(2))
    with pytest.raises(TypeError, match="not an Output"):
        rate.with_values(rate)


def test_array_unwraps_property_data_and_with_values_restores_the_map() -> None:
    pmap = PropertyMap.from_property(AGE)
    data = jnp.arange(6.0).reshape(3, 2)
    out = Output(times=_axis(3), values=PropertyData(pmap, data), dims=("time", "group"))
    raw = out.array()
    assert not isinstance(raw, PropertyData)
    back = out.with_values(raw * 2.0)
    assert isinstance(back.values, PropertyData)
    assert back.values.pmap == pmap
    np.testing.assert_allclose(np.asarray(back.values.data), np.asarray(data) * 2.0)


def test_lazy_arrays_compose_with_param_numbers_and_ufuncs() -> None:
    outputs = OutputSet()
    outputs["rate"] = FlowMass("infection").sum_by()
    outputs["total"] = outputs.ref("rate").integrate().scalar()
    outputs["log_total_scaled"] = np.log(outputs.ref("total")) * Param("k") + 1.0
    outputs["share"] = outputs.ref("rate").with_values(
        outputs.ref("rate").array() / outputs.ref("total")
    )
    outputs["share_direct"] = outputs.ref("rate") / outputs.ref("rate").integrate()
    named = _solve(outputs, {**PARAMS, "k": 2.0})
    assert set(named.arrays) == {"total", "log_total_scaled"}
    assert list(named.keys()) == ["rate", "share", "share_direct", "total", "log_total_scaled"]
    assert "total" in named
    rate = np.asarray(named["rate"].values)
    total = _trapezoid(rate, np.asarray(named["rate"].times.values))
    np.testing.assert_allclose(float(named["total"]), total, rtol=1e-6)
    np.testing.assert_allclose(float(named["log_total_scaled"]), math.log(total) * 2 + 1, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(named["share"].values), np.asarray(named["share_direct"].values), rtol=1e-6
    )
    with pytest.raises(TypeError, match="plain array"):
        named.to_frame(["total"])
    named.to_frame(["rate", "share"])  # outputs only still format


def test_lazy_scalar_on_a_series_raises_at_evaluate() -> None:
    outputs = OutputSet()
    outputs["bad"] = FlowMass("infection").sum_by().scalar()
    with pytest.raises(ValueError, match="0-d output"):
        _solve(outputs)


def test_lazy_with_values_checks_shape_at_evaluate() -> None:
    outputs = OutputSet()
    outputs["rate"] = FlowMass("infection").sum_by()
    outputs["bad"] = outputs.ref("rate").with_values(outputs.ref("rate").at(5.0).array())
    with pytest.raises(ValueError, match="with_values"):
        _solve(outputs)


def test_post_ops_on_an_array_node_raise() -> None:
    outputs = OutputSet()
    outputs["rate"] = FlowMass("infection").sum_by()
    outputs["bad"] = outputs.ref("rate").array().integrate()
    with pytest.raises(TypeError, match="applies to an Output"):
        _solve(outputs)


def test_defer_over_arrays_returns_an_array() -> None:
    outputs = OutputSet()
    outputs["prev"] = Compartments(where=STATE["I"]).sum_by()
    outputs["peak"] = defer(jnp.max)(outputs.ref("prev").array())
    named = _solve(outputs)
    np.testing.assert_allclose(float(named["peak"]), np.max(np.asarray(named["prev"].values)))


def test_target_on_an_array_output_points_at_with_values() -> None:
    outputs = OutputSet()
    outputs["rate"] = FlowMass("infection").sum_by().array()
    named = _solve(outputs)
    targets = TargetSet((Target(key="rate", times=[7.0], values=[1.0]),))
    with pytest.raises(TypeError, match="with_values"):
        targets.residuals(named)


def test_calibration_target_on_a_with_values_output() -> None:
    obs_times = np.arange(7.0, 70.0, 7.0)
    observed = np.full(obs_times.shape, 0.01)
    outputs = OutputSet()
    outputs["rate"] = FlowMass("infection").sum_by()
    outputs["share"] = outputs.ref("rate").with_values(
        outputs.ref("rate").array() / outputs.ref("rate").integrate().scalar()
    )
    target = Target(
        key="share", times=obs_times, values=observed, likelihood=NormalLikelihood(0.005)
    )
    bm = BayesianModel(
        _model(),
        {},
        [Uniform("contact_rate", 0.01, 0.5), Uniform("recovery_time", 10.0, 50.0)],
        TargetSet((target,)),
        outputs=outputs,
        run_kwargs=RUN,
    )
    ll = float(bm.log_likelihood(PARAMS))

    named = _solve(outputs)
    share = np.asarray(named["share"].values)
    idx = obs_times.astype(int)  # daily grid from day 0
    pred = share[idx]
    sd = 0.005
    dens = -0.5 * ((observed - pred) / sd) ** 2 - math.log(sd) - 0.5 * math.log(2 * math.pi)
    np.testing.assert_allclose(ll, float(np.mean(dens)), rtol=1e-5)


# --- jit and jaxpr size ---------------------------------------------------------------


def _share_outputs() -> OutputSet:
    outputs = OutputSet()
    outputs["rate"] = FlowMass("infection").sum_by()
    outputs["share"] = outputs.ref("rate") / outputs.ref("rate").integrate()
    outputs["change"] = outputs.ref("rate") - outputs.ref("rate").at(10.0)
    outputs["total"] = outputs.ref("rate").integrate().scalar()
    outputs["share_arr"] = outputs.ref("rate").with_values(
        outputs.ref("rate").array() / outputs.ref("total")
    )
    return outputs


def test_everything_evaluates_under_jit() -> None:
    outputs = _share_outputs()
    cm = _model()

    def run(params: dict[str, Any]) -> Result:
        return outputs.evaluate(cm.run(params, save=outputs.plan(), **RUN), params)

    eager = run(PARAMS)
    jitted = jax.jit(run)({k: jnp.asarray(v) for k, v in PARAMS.items()})
    for name in ("share", "change", "share_arr"):
        np.testing.assert_allclose(
            np.asarray(jitted[name].values), np.asarray(eager[name].values), rtol=1e-6
        )
    np.testing.assert_allclose(float(jitted["total"]), float(eager["total"]), rtol=1e-6)


def _eqn_count(n: int, build: Any) -> int:
    def fn(y: Any) -> Any:
        return build(_series_traced(y, n))

    return len(jax.make_jaxpr(fn)(jnp.ones(n)).jaxpr.eqns)


def _series_traced(y: Any, n: int) -> Output:
    return Output(times=_axis(n), values=y, dims=("time",))


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda r: (r / r.integrate()).values, id="share"),
        pytest.param(lambda r: (r - r.at(3.0)).values, id="change"),
        pytest.param(lambda r: r.with_values(r.array() / r.integrate().scalar()).values, id="arr"),
    ],
)
def test_broadcast_and_array_add_no_per_time_equations(build: Any) -> None:
    assert _eqn_count(50, build) == _eqn_count(500, build)


def test_array_and_with_values_add_no_equations() -> None:
    plain = _eqn_count(50, lambda r: r.values * 2.0)
    via_array = _eqn_count(50, lambda r: r.with_values(r.array() * 2.0).values)
    assert via_array == plain
