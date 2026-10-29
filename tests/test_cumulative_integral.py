"""``cumulative_integral`` and ``cumsum``: the two operations ``cumulative()`` conflated."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import (
    FlowMass,
    FlowModel,
    OutputSet,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    TransitionFlow,
)
from summer4.epi import ForceOfInfection
from summer4.results.output import Output
from summer4.time import TimeAxis

STATE = Property("state", ("S", "I", "R"))
# Non-uniform save grid: the integral must use each interval's own width.
TIMES = np.array([0.0, 0.5, 2.0, 2.5, 6.0, 10.0])
RATE = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 0.5])


def _axis(times: np.ndarray = TIMES) -> TimeAxis:
    return TimeAxis(values=times, epoch=None, kind="explicit")


def _rate(values: np.ndarray = RATE, times: np.ndarray = TIMES) -> Output:
    return Output(times=_axis(times), values=jnp.asarray(values), dims=("time",))


def _hand_trapezoid(t: np.ndarray, f: np.ndarray, i0: int = 0, i1: int | None = None) -> np.ndarray:
    i1 = t.size - 1 if i1 is None else i1
    out = np.zeros_like(f)
    for i in range(i0 + 1, i1 + 1):
        out[i] = out[i - 1] + 0.5 * (f[i - 1] + f[i]) * (t[i] - t[i - 1])
    return out


def _sir() -> tuple[object, np.ndarray]:
    model = FlowModel(PropertyMap.from_property(STATE))
    model.add_flow(
        TransitionFlow(
            "infection",
            STATE["S"],
            STATE["I"],
            ForceOfInfection("foi", infectious=STATE["I"], contact_rate=Param("beta")),
        )
    )
    model.add_flow(TransitionFlow("recovery", STATE["I"], STATE["R"], 0.1))
    return model.compile(), np.array([990.0, 10.0, 0.0])


def test_matches_hand_trapezoid_on_a_non_uniform_grid() -> None:
    got = _rate().cumulative_integral()
    assert got.dims == ("time",)
    np.testing.assert_array_equal(np.asarray(got.times.values), TIMES)
    np.testing.assert_allclose(np.asarray(got.values), _hand_trapezoid(TIMES, RATE), rtol=1e-6)
    assert float(got.values[0]) == 0.0


def test_window_starts_at_zero_and_zeros_outside() -> None:
    got = _rate().cumulative_integral(start=0.5, end=6.0)
    want = _hand_trapezoid(TIMES, RATE, i0=1, i1=4)
    np.testing.assert_allclose(np.asarray(got.values), want, rtol=1e-6)
    assert float(got.values[1]) == 0.0
    assert float(got.values[-1]) == 0.0
    from_two = _rate().cumulative_integral(start=2.0)
    np.testing.assert_allclose(
        np.asarray(from_two.values), _hand_trapezoid(TIMES, RATE, i0=2), rtol=1e-6
    )
    with pytest.raises(ValueError, match="cumulative_integral start=1.0 .* not a save time"):
        _rate().cumulative_integral(start=1.0)
    with pytest.raises(ValueError, match="end is before start"):
        _rate().cumulative_integral(start=6.0, end=2.0)
    with pytest.raises(ValueError, match="trapezoid"):
        _rate().cumulative_integral(method="simpson")  # type: ignore[arg-type]


def test_equals_integrate_intervals_cumsum_shifted_by_one_point() -> None:
    series = _rate()
    running = np.asarray(series.integrate_intervals().cumsum().values)
    np.testing.assert_allclose(
        np.asarray(series.cumulative_integral().values), np.concatenate([[0.0], running]), rtol=1e-6
    )
    np.testing.assert_allclose(
        float(series.cumulative_integral().values[-1]),
        float(series.integrate().values),
        rtol=1e-6,
    )


def test_keeps_the_property_map_and_integrates_each_column() -> None:
    pmap = PropertyMap.from_property(STATE)
    data = np.stack([RATE, 2.0 * RATE, np.zeros_like(RATE)], axis=1)
    series = Output(
        times=_axis(), values=PropertyData(pmap, jnp.asarray(data)), dims=("time", "compartment")
    )
    got = series.cumulative_integral()
    assert isinstance(got.values, PropertyData)
    assert got.values.pmap == pmap
    want = _hand_trapezoid(TIMES, RATE)
    np.testing.assert_allclose(
        np.asarray(got.values.data), np.stack([want, 2.0 * want, 0.0 * want], axis=1), rtol=1e-6
    )


def test_single_save_time_is_zero() -> None:
    got = _rate(np.array([7.0]), np.array([3.0])).cumulative_integral()
    np.testing.assert_array_equal(np.asarray(got.values), [0.0])


def test_under_jit_and_grad() -> None:
    def run(scale: jax.Array) -> jax.Array:
        series = Output(times=_axis(), values=jnp.asarray(RATE) * scale, dims=("time",))
        return series.cumulative_integral(start=0.5).values

    want = _hand_trapezoid(TIMES, RATE, i0=1)
    np.testing.assert_allclose(np.asarray(jax.jit(run)(2.0)), 2.0 * want, rtol=1e-6)
    grad = jax.grad(lambda s: run(s)[-1])(1.0)
    assert float(grad) == pytest.approx(want[-1], rel=1e-6)


def _jaxpr_eqns(n: int) -> int:
    times = np.linspace(0.0, 10.0, n)

    def run(values: jax.Array) -> jax.Array:
        series = Output(times=_axis(times), values=values, dims=("time",))
        return series.cumulative_integral(start=float(times[1])).values

    return len(jax.make_jaxpr(run)(jnp.ones(n)).jaxpr.eqns)


def test_jaxpr_does_not_grow_with_trajectory_length() -> None:
    assert _jaxpr_eqns(10) == _jaxpr_eqns(2000)


def test_counts_s_depletion_on_a_save_spacing_of_two() -> None:
    cm, y0 = _sir()
    res = cm.run({"beta": 0.3}, y0, t0=0.0, t1=60.0, dt=2.0, solver="tsit5", rtol=1e-8, atol=1e-8)
    s = np.asarray(res["compartments"].select(STATE["S"]).sum_by().values)
    rate = res["infection"].sum_by()
    integral = np.asarray(rate.cumulative_integral().values)
    # Trapezoid error on a 2-day grid is about 1% while the epidemic grows.
    np.testing.assert_allclose(integral, s[0] - s, rtol=1.5e-2, atol=0.5)
    assert integral[-1] == pytest.approx(s[0] - s[-1], rel=1e-3)
    # A plain running sum of the rate is out by the save spacing (here about half).
    assert float(rate.cumsum().values[-1]) < 0.6 * (s[0] - s[-1])


@pytest.mark.parametrize("spacing", [0.5, 1.0, 5.0])
def test_final_count_is_independent_of_save_spacing(spacing: float) -> None:
    cm, y0 = _sir()
    res = cm.run(
        {"beta": 0.3}, y0, t0=0.0, t1=60.0, dt=spacing, solver="tsit5", rtol=1e-8, atol=1e-8
    )
    s = np.asarray(res["compartments"].select(STATE["S"]).sum_by().values)
    rate = res["infection"].sum_by()
    true = s[0] - s[-1]
    assert float(rate.cumulative_integral().values[-1]) == pytest.approx(true, rel=2e-3)
    # The running sum of a rate scales with 1 / spacing.
    assert float(rate.cumsum().values[-1]) == pytest.approx(true / spacing, rel=1e-2)


def test_cumsum_is_a_running_sum() -> None:
    series = _rate()
    np.testing.assert_allclose(np.asarray(series.cumsum().values), np.cumsum(RATE))


def test_output_set_evaluates_named_outputs() -> None:
    cm, y0 = _sir()
    outputs = OutputSet()
    outputs["incidence"] = FlowMass("infection").sum_by()
    outputs["cum_infections"] = outputs.ref("incidence").cumulative_integral(start=4.0)
    outputs["infections"] = outputs.ref("incidence").integrate_intervals()
    outputs["running"] = outputs.ref("infections").cumsum()
    outputs["from_leaf"] = FlowMass("infection").sum_by().cumulative_integral()
    res = cm.run({"beta": 0.3}, y0, t0=0.0, t1=20.0, dt=2.0, save=outputs.plan(SavePlan()))
    named = outputs.evaluate(res, {})
    rate = res["incidence"].sum_by()
    np.testing.assert_allclose(
        np.asarray(named["cum_infections"].values),
        np.asarray(rate.cumulative_integral(start=4.0).values),
    )
    np.testing.assert_allclose(
        np.asarray(named["from_leaf"].values)[1:], np.asarray(named["running"].values), rtol=1e-6
    )
