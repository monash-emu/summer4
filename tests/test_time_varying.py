"""Tests for Time() and time-varying rate expressions."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import (
    Compartments,
    EntryFlow,
    Epoch,
    ExitFlow,
    FlowModel,
    Multiply,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    Time,
    Transform,
    derived_refs,
)
from summer4.flows.rates import _rate_bytes
from summer4.timevarying import gaussian_pulse, linear, sigmoidal, step


def test_time_equality_and_hash() -> None:
    assert Time() == Time()
    assert hash(Time()) == hash(Time())


def test_time_ramp_matches_analytic() -> None:
    """Exit rate Time()*0.1 integrates ẏ = -0.1 t y → y(t) = y0 exp(-0.05 t²)."""
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(ExitFlow("out", state["Y"], Time() * 0.1))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([1.0]))
    t1 = 5.0
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([t1]))
    res = cm.run({}, y0, t0=0.0, t1=t1, dt=0.01, save=plan, solver="tsit5", rtol=1e-8, atol=1e-10)
    got = float(np.asarray(res["y"].values.data)[-1, 0])
    expected = float(np.exp(-0.05 * t1 * t1))
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_time_inside_transform_and_multiply() -> None:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)

    def scale_by_t(prev: Any, t_val: Any) -> Any:
        return prev * t_val

    model = FlowModel(pmap)
    model.add_flow(
        ExitFlow(
            "out",
            state["Y"],
            0.1,
            adjust=(Transform(scale_by_t, Time()),),
        )
    )
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([1.0]))
    dy = np.asarray(cm.vector_field(2.0, y0, {}).data).reshape(-1)
    # rate = 0.1 * t = 0.2 → ẏ = -0.2
    np.testing.assert_allclose(dy, [-0.2], rtol=1e-6)

    model2 = FlowModel(pmap)
    model2.add_flow(
        ExitFlow(
            "out",
            state["Y"],
            0.1,
            adjust=(Multiply(Time()),),
        )
    )
    cm2 = model2.compile()
    dy2 = np.asarray(cm2.vector_field(3.0, y0, {}).data).reshape(-1)
    np.testing.assert_allclose(dy2, [-0.3], rtol=1e-6)


def test_time_jit_cache_hit_across_rebuilds() -> None:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)

    def build() -> Any:
        model = FlowModel(pmap)
        model.add_flow(ExitFlow("out", state["Y"], Time() * 0.1))
        return model.compile()

    a = build()
    b = build()
    assert a == b
    assert hash(a) == hash(b)
    assert a._digest == b._digest

    y0 = PropertyData.wrap(pmap, np.array([1.0]))
    plan = SavePlan(requests={"y": SaveRequest(Compartments())})
    traces = {"n": 0}

    def counting_derived(params: object, *, y: object, t: object) -> object:
        del params, y, t
        traces["n"] += 1
        return {}

    # Digest equality is the cache key; separately-built Time() must match.
    model_c = FlowModel(pmap)
    model_c.add_flow(ExitFlow("out", state["Y"], Time() * 0.1))
    # No derived_fn on a/b; rebuild with same structure still hashes equal.
    c = model_c.compile()
    assert c._digest == a._digest
    _ = a.run({}, y0, t0=0.0, steps=2, dt=1.0, save=plan)
    _ = b.run({}, y0, t0=0.0, steps=2, dt=1.0, save=plan)


def test_time_jaxpr_independent_of_save_count() -> None:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(ExitFlow("out", state["Y"], Time() * 0.1))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([1.0]))

    def loss_for(n_saves: int) -> Any:
        ts = np.linspace(0.0, 4.0, n_saves)
        plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=ts)

        def loss(_: float) -> Any:
            res = cm.run({}, y0, t0=0.0, t1=4.0, dt=0.1, save=plan, solver="tsit5")
            return jnp.sum(jnp.asarray(res["y"].values.data))

        return loss

    jp_small = jax.make_jaxpr(loss_for(5))(0.0)
    jp_large = jax.make_jaxpr(loss_for(40))(0.0)
    assert len(jp_small.jaxpr.eqns) == len(jp_large.jaxpr.eqns)


def _rate_at(expr: object, t: float, *, y0: float = 1.0, params: object | None = None) -> float:
    """Evaluate a rate expression as an absolute entry flow at time ``t``."""
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], expr))
    cm = model.compile()
    y = PropertyData.wrap(pmap, np.array([y0]))
    return float(np.asarray(cm.vector_field(t, y, params if params is not None else {}).data)[0])


def test_linear_matches_knots_and_midpoints() -> None:
    expr = linear(Time(), (0.0, 2.0, 4.0), (1.0, 3.0, 5.0))
    np.testing.assert_allclose(_rate_at(expr, 0.0), 1.0)
    np.testing.assert_allclose(_rate_at(expr, 2.0), 3.0)
    np.testing.assert_allclose(_rate_at(expr, 4.0), 5.0)
    np.testing.assert_allclose(_rate_at(expr, 1.0), 2.0)
    np.testing.assert_allclose(_rate_at(expr, 3.0), 4.0)
    # clamp outside range
    np.testing.assert_allclose(_rate_at(expr, -1.0), 1.0)
    np.testing.assert_allclose(_rate_at(expr, 10.0), 5.0)


def test_step_right_continuous_at_breakpoint() -> None:
    expr = step(Time(), (1.0, 3.0), (10.0, 20.0, 30.0))
    np.testing.assert_allclose(_rate_at(expr, 0.5), 10.0)
    np.testing.assert_allclose(_rate_at(expr, 1.0), 20.0)  # right-continuous
    np.testing.assert_allclose(_rate_at(expr, 2.0), 20.0)
    np.testing.assert_allclose(_rate_at(expr, 3.0), 30.0)
    np.testing.assert_allclose(_rate_at(expr, 4.0), 30.0)


def test_sigmoidal_at_knots_and_linear_sharpness() -> None:
    expr = sigmoidal(Time(), (0.0, 1.0), (0.0, 1.0), sharpness=1.0)
    np.testing.assert_allclose(_rate_at(expr, 0.0), 0.0, atol=1e-6)
    np.testing.assert_allclose(_rate_at(expr, 1.0), 1.0, atol=1e-6)
    # sharpness=1 → normalized sigmoid ≈ linear at midpoint
    np.testing.assert_allclose(_rate_at(expr, 0.5), 0.5, atol=1e-5)


def test_gaussian_pulse_peak() -> None:
    expr = gaussian_pulse(Time(), centre=10.0, width=2.0, height=5.0)
    np.testing.assert_allclose(_rate_at(expr, 10.0), 5.0)
    half = 5.0 * np.exp(-0.5 * ((12.0 - 10.0) / 2.0) ** 2)
    np.testing.assert_allclose(_rate_at(expr, 12.0), half)


def test_gaussian_pulse_width_param_matches_literal() -> None:
    """width is as_rate'd: Param / derived_refs match a literal, and scale the shoulders."""

    class Width(NamedTuple):
        width: float

    refs = derived_refs(Width)
    literal = gaussian_pulse(Time(), centre=10.0, width=2.0, height=5.0)
    via_refs = gaussian_pulse(Time(), centre=10.0, width=refs.width, height=5.0)
    via_param = gaussian_pulse(Time(), centre=10.0, width=Param("width"), height=5.0)
    params_nt = Width(width=2.0)
    params_dict = {"width": 2.0}
    for t in (8.0, 10.0, 12.0):
        want = _rate_at(literal, t)
        np.testing.assert_allclose(_rate_at(via_refs, t, params=params_nt), want)
        np.testing.assert_allclose(_rate_at(via_param, t, params=params_dict), want)
    # Wider pulse decays less far from the centre (peak itself is width-invariant).
    narrow = _rate_at(via_param, 12.0, params={"width": 2.0})
    wide = _rate_at(via_param, 12.0, params={"width": 4.0})
    assert wide > narrow


def test_interp_fieldref_knot_calibrates() -> None:
    class Knots(NamedTuple):
        lo: float
        hi: float

    refs = derived_refs(Knots)
    expr = linear(Time(), (0.0, 1.0), (refs.lo, refs.hi))
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], expr))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([0.0]))
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([1.0]))

    def loss(hi: Any) -> Any:
        params = Knots(lo=0.0, hi=hi)
        res = cm.run(params, y0, t0=0.0, t1=1.0, dt=0.1, save=plan, solver="euler")
        return jnp.sum(jnp.asarray(res["y"].values.data))

    g = float(jax.grad(loss)(jnp.asarray(2.0)))
    assert np.isfinite(g)
    eps = 1e-3
    fd = (float(loss(2.0 + eps)) - float(loss(2.0 - eps))) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-2, atol=1e-3)


def test_interp_digest_covers_breakpoints() -> None:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)

    def build(bps: tuple[float, ...]) -> Any:
        model = FlowModel(pmap)
        model.add_flow(EntryFlow("in", state["Y"], linear(Time(), bps, (1.0, 2.0))))
        return model.compile()

    a = build((0.0, 1.0))
    b = build((0.0, 1.0))
    c = build((0.0, 2.0))
    assert a._digest == b._digest
    assert a._digest != c._digest


def test_interp_jaxpr_independent_of_breakpoints_times_trajectory() -> None:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    bps = tuple(float(x) for x in range(6))
    vals = tuple(float(x) for x in range(6))
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], linear(Time(), bps, vals)))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([0.0]))

    def loss_for(n_saves: int) -> Any:
        ts = np.linspace(0.0, 5.0, n_saves)
        plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=ts)

        def loss(_: float) -> Any:
            res = cm.run({}, y0, t0=0.0, t1=5.0, dt=0.2, save=plan, solver="tsit5")
            return jnp.sum(jnp.asarray(res["y"].values.data))

        return loss

    jp_a = jax.make_jaxpr(loss_for(4))(0.0)
    jp_b = jax.make_jaxpr(loss_for(20))(0.0)
    assert len(jp_a.jaxpr.eqns) == len(jp_b.jaxpr.eqns)


def test_data_from_series_round_trip() -> None:
    pd = pytest.importorskip("pandas")
    from summer4.data import Data

    epoch = Epoch(date(2020, 1, 1))
    dates = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-01-21"])
    series = pd.Series([1.0, 3.0, 5.0], index=dates)
    data = Data.from_series(series, epoch)
    expr = data.interp(kind="linear")
    for t, v in zip(data.times, data.values, strict=True):
        np.testing.assert_allclose(_rate_at(expr, float(t)), float(v))


def test_data_interp_clamps_outside_range() -> None:
    from summer4.data import Data

    data = Data(times=np.array([0.0, 2.0]), values=np.array([1.0, 5.0]))
    expr = data.interp()
    np.testing.assert_allclose(_rate_at(expr, -5.0), 1.0)
    np.testing.assert_allclose(_rate_at(expr, 10.0), 5.0)


def test_data_from_csv_matches_from_series(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    from summer4.data import Data

    epoch = Epoch(date(2020, 1, 1))
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2020-02-01", "2020-02-15", "2020-03-01"]),
            "value": [0.2, 0.5, 0.1],
        }
    )
    path = tmp_path / "series.csv"
    frame.to_csv(path, index=False)
    from_csv = Data.from_csv(path, epoch)
    from_series = Data.from_series(frame.set_index("time")["value"], epoch)
    np.testing.assert_allclose(from_csv.times, from_series.times)
    np.testing.assert_allclose(from_csv.values, from_series.values)
    assert _rate_bytes(from_csv.interp()) == _rate_bytes(from_series.interp())


def test_data_import_without_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    """``summer4.data`` imports cleanly; ``from_series`` names the pandas extra."""
    import summer4.data as data_mod

    def boom() -> Any:
        raise ImportError(
            "summer4.data requires pandas. Install with: pip install 'summer4[pandas]'."
        )

    monkeypatch.setattr(data_mod, "_require_pandas", boom)
    with pytest.raises(ImportError, match="pandas"):
        data_mod.Data.from_series(None, None)  # type: ignore[arg-type]


def test_parametric_breakpoint_midpoints() -> None:
    class Times(NamedTuple):
        t0: float
        t1: float

    refs = derived_refs(Times)
    expr = linear(Time(), (refs.t0, refs.t1), (0.0, 1.0))
    params = Times(t0=0.0, t1=2.0)
    np.testing.assert_allclose(_rate_at(expr, 0.0, params=params), 0.0)
    np.testing.assert_allclose(_rate_at(expr, 1.0, params=params), 0.5)
    np.testing.assert_allclose(_rate_at(expr, 2.0, params=params), 1.0)


def test_interp_fieldref_breakpoint_calibrates() -> None:
    class KnotX(NamedTuple):
        t_mid: float

    refs = derived_refs(KnotX)
    expr = linear(Time(), (0.0, refs.t_mid, 2.0), (0.0, 1.0, 0.0))
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], expr))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([0.0]))
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([1.0]))

    def loss(t_mid: Any) -> Any:
        params = KnotX(t_mid=t_mid)
        res = cm.run(params, y0, t0=0.0, t1=1.0, dt=0.1, save=plan, solver="euler")
        return jnp.sum(jnp.asarray(res["y"].values.data))

    g = float(jax.grad(loss)(jnp.asarray(1.0)))
    assert np.isfinite(g)
    eps = 1e-3
    fd = (float(loss(1.0 + eps)) - float(loss(1.0 - eps))) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-2, atol=1e-3)


def test_mixed_const_fieldref_breakpoints() -> None:
    class T1(NamedTuple):
        t1: float

    refs = derived_refs(T1)
    expr = linear(Time(), (0.0, refs.t1, 4.0), (1.0, 3.0, 5.0))
    params = T1(t1=2.0)
    np.testing.assert_allclose(_rate_at(expr, 1.0, params=params), 2.0)
    np.testing.assert_allclose(_rate_at(expr, 3.0, params=params), 4.0)


def test_step_with_parametric_breakpoint() -> None:
    class Cut(NamedTuple):
        t_cut: float

    refs = derived_refs(Cut)
    expr = step(Time(), (refs.t_cut,), (0.0, 1.0))
    params = Cut(t_cut=5.0)
    np.testing.assert_allclose(_rate_at(expr, 4.0, params=params), 0.0)
    np.testing.assert_allclose(_rate_at(expr, 5.0, params=params), 1.0)


def test_parametric_breakpoint_digest_and_equality() -> None:
    class A(NamedTuple):
        t: float

    class B(NamedTuple):
        u: float

    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)

    def build(refs_t: Any) -> Any:
        model = FlowModel(pmap)
        model.add_flow(
            EntryFlow("in", state["Y"], linear(Time(), (0.0, refs_t, 2.0), (0.0, 1.0, 0.0)))
        )
        return model.compile()

    a_refs = derived_refs(A)
    b_refs = derived_refs(B)
    ca = build(a_refs.t)
    ca2 = build(derived_refs(A).t)
    cb = build(b_refs.u)
    assert ca._digest == ca2._digest
    assert ca._digest != cb._digest

    # Const breakpoint value also enters the digest.
    c0 = FlowModel(pmap)
    c0.add_flow(EntryFlow("in", state["Y"], linear(Time(), (0.0, 1.0), (0.0, 1.0))))
    c1 = FlowModel(pmap)
    c1.add_flow(EntryFlow("in", state["Y"], linear(Time(), (0.0, 1.5), (0.0, 1.0))))
    assert c0.compile()._digest != c1.compile()._digest


def test_all_const_breakpoints_must_increase() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        linear(Time(), (0.0, 1.0, 1.0), (0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        step(Time(), (2.0, 1.0), (0.0, 1.0, 2.0))


def test_fieldref_breakpoints_skip_host_order_check() -> None:
    class T(NamedTuple):
        t0: float
        t1: float

    refs = derived_refs(T)
    # Construct succeeds even though we cannot know order yet.
    expr = linear(Time(), (refs.t0, refs.t1), (0.0, 1.0))
    assert len(expr.breakpoints) == 2


def test_jit_vector_field_matches_eager_parametric_xy() -> None:
    """Parametric x+y linear rate evaluates identically under jax.jit."""

    class Knots(NamedTuple):
        t_mid: float
        height: float

    refs = derived_refs(Knots)
    expr = linear(Time(), (0.0, refs.t_mid, 10.0), (0.0, refs.height, 0.0))
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], expr))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([0.0]))
    params = Knots(t_mid=5.0, height=2.0)

    eager = np.asarray(cm.vector_field(5.0, y0, params).data)
    jitted = np.asarray(jax.jit(cm.vector_field)(5.0, y0, params).data)
    np.testing.assert_allclose(jitted, eager, rtol=1e-6)
    np.testing.assert_allclose(jitted.reshape(-1), [2.0], rtol=1e-5)


def test_jit_value_and_grad_through_parametric_xy() -> None:
    """jax.jit(value_and_grad) through both breakpoint and value FieldRefs."""

    class Knots(NamedTuple):
        t_mid: float
        height: float

    refs = derived_refs(Knots)
    expr = linear(Time(), (0.0, refs.t_mid, 4.0), (0.0, refs.height, 0.0))
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], expr))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([0.0]))
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([2.0]))

    def loss(params: Knots) -> Any:
        res = cm.run(params, y0, t0=0.0, t1=2.0, dt=0.25, save=plan, solver="euler")
        return jnp.sum(jnp.asarray(res["y"].values.data))

    base = Knots(t_mid=2.0, height=1.5)
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    val, grads = value_and_grad(base)
    assert np.isfinite(float(val))
    assert np.isfinite(float(grads.t_mid))
    assert np.isfinite(float(grads.height))

    eps = 1e-3
    fd_t = (
        float(loss(Knots(t_mid=2.0 + eps, height=1.5)))
        - float(loss(Knots(t_mid=2.0 - eps, height=1.5)))
    ) / (2 * eps)
    fd_h = (
        float(loss(Knots(t_mid=2.0, height=1.5 + eps)))
        - float(loss(Knots(t_mid=2.0, height=1.5 - eps)))
    ) / (2 * eps)
    np.testing.assert_allclose(float(grads.t_mid), fd_t, rtol=2e-2, atol=1e-3)
    np.testing.assert_allclose(float(grads.height), fd_h, rtol=2e-2, atol=1e-3)

    # Second jit call must not retrace (same digest / concrete shapes).
    val2, grads2 = value_and_grad(Knots(t_mid=2.1, height=1.4))
    assert np.isfinite(float(val2))
    assert np.isfinite(float(grads2.t_mid))


def test_jit_sigmoidal_and_gaussian_pulse_parametric() -> None:
    """Sigmoidal (parametric x+y) and gaussian_pulse evaluate and differentiate under jit."""

    class SigKnots(NamedTuple):
        t1: float
        hi: float

    refs = derived_refs(SigKnots)
    sig = sigmoidal(Time(), (0.0, refs.t1), (0.0, refs.hi), sharpness=1.0)
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], sig))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([0.0]))
    params = SigKnots(t1=1.0, hi=2.0)

    dy = np.asarray(jax.jit(cm.vector_field)(0.5, y0, params).data)
    np.testing.assert_allclose(dy.reshape(-1), [1.0], atol=1e-4)

    class Pulse(NamedTuple):
        centre: float
        height: float

    prefs = derived_refs(Pulse)
    pulse = gaussian_pulse(Time(), centre=prefs.centre, width=1.0, height=prefs.height)
    pm = FlowModel(pmap)
    pm.add_flow(EntryFlow("in", state["Y"], pulse))
    pcm = pm.compile()
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([5.0]))

    def loss(p: Pulse) -> Any:
        res = pcm.run(p, y0, t0=0.0, t1=5.0, dt=0.5, save=plan, solver="euler")
        return jnp.sum(jnp.asarray(res["y"].values.data))

    vg = jax.jit(jax.value_and_grad(loss))
    val, g = vg(Pulse(centre=2.5, height=3.0))
    assert np.isfinite(float(val))
    assert np.isfinite(float(g.centre))
    assert np.isfinite(float(g.height))
    # Peak height scales mass roughly linearly for a fixed centre on the grid.
    assert float(g.height) > 0.0


def test_jit_run_parametric_interp_tsit5() -> None:
    """Adaptive solver path with parametric Interp stays jittable."""

    class Knots(NamedTuple):
        t_mid: float
        height: float

    refs = derived_refs(Knots)
    expr = linear(Time(), (0.0, refs.t_mid, 10.0), (0.0, refs.height, 0.0))
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], expr))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([0.0]))
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([0.0, 5.0, 10.0]))

    def run(params: Knots) -> Any:
        res = cm.run(
            params,
            y0,
            t0=0.0,
            t1=10.0,
            dt=0.5,
            save=plan,
            solver="tsit5",
            rtol=1e-6,
            atol=1e-8,
        )
        return jnp.sum(jnp.asarray(res["y"].values.data))

    jrun = jax.jit(run)
    a = float(jrun(Knots(t_mid=5.0, height=1.0)))
    b = float(jrun(Knots(t_mid=5.0, height=2.0)))
    assert np.isfinite(a) and np.isfinite(b)
    assert b > a  # taller peak → more cumulative entry mass
