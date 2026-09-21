"""Name-aligned Output arithmetic, windowed cumulative, midpoint, multi-flow mass."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import (
    ExitFlow,
    FlowMass,
    FlowModel,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    TransitionFlow,
)
from summer4.results.output import Output
from summer4.time import TimeAxis


def _times(n: int = 4) -> TimeAxis:
    return TimeAxis(values=np.arange(n, dtype=np.float64), epoch=None, kind="explicit")


def _series(values: object, dims: tuple[str, ...], times: TimeAxis | None = None) -> Output:
    arr = np.asarray(values, dtype=np.float64)
    axis = times if times is not None else _times(arr.shape[0])
    return Output(times=axis, values=jnp.asarray(arr), dims=dims)


def test_per_age_over_total_keeps_the_map() -> None:
    age = Property("age", ("0", "5"))
    pmap = PropertyMap.from_property(age)
    times = _times(3)
    per = Output(
        times=times,
        values=PropertyData(pmap, jnp.array([[2.0, 6.0], [4.0, 8.0], [1.0, 3.0]])),
        dims=("time", "group"),
    )
    total = _series([8.0, 12.0, 4.0], ("time",), times)
    ratio = per / total
    assert ratio.dims == ("time", "group")
    assert isinstance(ratio.values, PropertyData)
    assert ratio.values.pmap == pmap
    np.testing.assert_allclose(
        np.asarray(ratio.values.data),
        [[0.25, 0.75], [1.0 / 3.0, 2.0 / 3.0], [0.25, 0.75]],
        rtol=1e-5,
    )


def test_same_map_addition_and_reflected_scalar() -> None:
    times = _times(2)
    left = _series([[1.0, 2.0], [3.0, 4.0]], ("time", "group"), times)
    right = _series([[10.0, 20.0], [30.0, 40.0]], ("time", "group"), times)
    added = left + right
    np.testing.assert_allclose(np.asarray(added.values), [[11.0, 22.0], [33.0, 44.0]])
    scaled = 2.0 / left
    np.testing.assert_allclose(np.asarray(scaled.values), [[2.0, 1.0], [2.0 / 3.0, 0.5]], rtol=1e-5)


def test_time_and_dim_mismatches_name_both_sides() -> None:
    left = _series([1.0, 2.0], ("time",))
    other_times = Output(
        times=TimeAxis(values=np.array([0.0, 5.0]), epoch=None, kind="explicit"),
        values=jnp.array([1.0, 2.0]),
        dims=("time",),
    )
    with pytest.raises(ValueError, match="time axes differ"):
        _ = left + other_times
    strain = _series([[1.0, 2.0], [3.0, 4.0]], ("time", "strain"))
    age = _series([[1.0, 2.0], [3.0, 4.0]], ("time", "age"))
    with pytest.raises(ValueError, match="not broadcastable"):
        _ = strain + age


def test_jit_grad_of_named_division() -> None:
    times = _times(2)
    total = _series([8.0, 12.0], ("time",), times)

    def loss(scale: object) -> object:
        per = Output(
            times=times,
            values=jnp.array([[2.0, 6.0], [4.0, 8.0]]) * scale,
            dims=("time", "group"),
        )
        return jnp.sum((per / total).values)

    value = jax.jit(loss)(1.0)
    grad = jax.grad(loss)(1.0)
    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0


def test_cumulative_window_zeros_outside_and_requires_a_save_time() -> None:
    series = _series([1.0, 2.0, 3.0, 4.0], ("time",))
    window = series.cumulative(start=1.0, end=2.0)
    np.testing.assert_allclose(np.asarray(window.values), [0.0, 2.0, 5.0, 0.0])
    from_start = series.cumulative(start=2.0)
    np.testing.assert_allclose(np.asarray(from_start.values), [0.0, 0.0, 3.0, 7.0])
    np.testing.assert_allclose(np.asarray(series.cumulative().values), [1.0, 3.0, 6.0, 10.0])
    with pytest.raises(ValueError, match="not a save time"):
        series.cumulative(start=1.5)
    with pytest.raises(ValueError, match="end is before start"):
        series.cumulative(start=3.0, end=1.0)


def test_cumulative_under_jit() -> None:
    times = _times(4)

    def run(scale: object) -> object:
        series = Output(times=times, values=jnp.arange(1.0, 5.0) * scale, dims=("time",))
        return series.cumulative(start=1.0).values

    np.testing.assert_allclose(np.asarray(jax.jit(run)(1.0)), [0.0, 2.0, 5.0, 9.0], rtol=1e-5)


def test_midpoint_matches_summer2_convention() -> None:
    series = _series([1.0, 3.0, 7.0, 8.0], ("time",))
    mid = series.midpoint()
    np.testing.assert_allclose(np.asarray(mid.values), [1.0, 2.0, 5.0, 7.5])


def test_rolling_matches_clipped_window_and_jaxpr_is_flat_in_t() -> None:
    raw = np.arange(12, dtype=np.float64)
    series = _series(raw, ("time",))
    rolled = series.rolling(4, how="mean", center=True, min_periods=3)
    expected = np.empty_like(raw)
    for i in range(raw.size):
        left = i - 2
        right = left + 4
        lo = max(0, left)
        hi = min(raw.size, right)
        if hi - lo < 3:
            expected[i] = np.nan
        else:
            expected[i] = raw[lo:hi].mean()
    np.testing.assert_allclose(np.asarray(rolled.values), expected, rtol=1e-5, equal_nan=True)

    def eqns(n: int) -> int:
        ts = np.linspace(0.0, 1.0, n)
        vals = jnp.linspace(0.0, 1.0, n)

        def run(v: object) -> object:
            out = Output(
                times=TimeAxis(values=ts, epoch=None, kind="explicit"),
                values=v,
                dims=("time",),
            )
            return out.rolling(5, how="sum", min_periods=1).values

        return len(jax.make_jaxpr(run)(vals).jaxpr.eqns)

    assert eqns(8) == eqns(32)


def test_multi_flow_mass_sums_after_the_same_reduction() -> None:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("0", "5"))
    pmap = PropertyMap.from_property(state).stratify(age)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], 0.2))
    model.add_flow(ExitFlow("death", state["I"], 0.05))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.ones(pmap.size))
    summed = SavePlan(
        requests={
            "out": SaveRequest(FlowMass(flow=("infection", "death"), sum_over=(age, "source")))
        }
    )
    separate = SavePlan(
        requests={
            "infection": SaveRequest(FlowMass(flow="infection")),
            "death": SaveRequest(FlowMass(flow="death")),
        }
    )
    both = cm.run({}, y0, t0=0.0, steps=2, dt=1.0, save=summed)
    parts = cm.run({}, y0, t0=0.0, steps=2, dt=1.0, save=separate)
    manual = parts["infection"].sum_over(age, side="source") + parts["death"].sum_over(
        age, side="source"
    )
    np.testing.assert_allclose(
        np.asarray(both["out"].values.data),
        np.asarray(manual.values.data),
        rtol=1e-5,
    )
    assert FlowMass(flow=("infection",)) == FlowMass(flow="infection")
    one = SavePlan(requests={"x": SaveRequest(FlowMass(flow="infection"))})
    also = SavePlan(requests={"x": SaveRequest(FlowMass(flow=("infection",)))})
    pair = SavePlan(requests={"x": SaveRequest(FlowMass(flow=("infection", "death")))})
    assert one == also
    assert one != pair


def test_multi_flow_mass_rejects_different_selected_dims() -> None:
    state = Property("state", ("S", "I"))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], 0.2))
    model.add_flow(ExitFlow("death", state["I"], 0.05))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([10.0, 1.0]))
    plan = SavePlan(requests={"out": SaveRequest(FlowMass(flow=("infection", "death")))})
    with pytest.raises(ValueError, match="do not share selected dims"):
        cm.run({}, y0, t0=0.0, steps=1, dt=1.0, save=plan)


def test_flowmass_rejects_an_empty_or_repeated_name() -> None:
    with pytest.raises(ValueError, match="at least one flow"):
        FlowMass(flow=())
    with pytest.raises(ValueError, match="repeats"):
        FlowMass(flow=("infection", "infection"))
