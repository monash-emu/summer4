"""Unary and extended binary rate operators, and the Output seam they share."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from summer4 import (  # noqa: E402
    EntryFlow,
    Everything,
    ExitFlow,
    FlowModel,
    GroupedRate,
    Output,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    Time,
    TransitionFlow,
    UnaryOp,
    clip,
    eval_closed,
    exp,
    floor,
    log,
    maximum,
    sqrt,
    tanh,
)
from summer4.flows.algebra import apply_binary, apply_unary  # noqa: E402
from summer4.flows.rates import (  # noqa: E402
    ArrayConst,
    FlowRef,
    _field_paths,
    _flow_refs,
    _rate_bytes,
)
from summer4.flows.stages import build_hoist_table, rate_stage  # noqa: E402
from summer4.time import TimeAxis  # noqa: E402


def _triang(t: np.ndarray, peak: float, height: float, width: float) -> np.ndarray:
    """tb_macro ``get_triang_vals``: ``clip(h * (1 - abs(t - p) / w), 0)``."""
    return np.maximum(height * (1.0 - np.abs(t - peak) / width), 0.0)


def _tanh_scaleup(
    t: np.ndarray,
    shape: float,
    inflection: float,
    start: float,
    end: float,
) -> np.ndarray:
    """tb_macro ``tanh_based_scaleup`` from ``start`` to ``end``."""
    return start + (end - start) * (np.tanh(shape * (t - inflection)) + 1.0) / 2.0


def _trace(values: np.ndarray) -> Output:
    times = np.linspace(0.0, 1.0, values.shape[0])
    return Output(
        times=TimeAxis(values=times, epoch=None, kind="explicit"),
        values=values,
        dims=("time",),
    )


def test_each_op_matches_jnp_on_scalars_arrays_and_grouped() -> None:
    age = Property("age", ("young", "old"))
    scalar = 1.5
    array = np.array([0.25, 1.0, 4.0])
    grouped = GroupedRate(data=np.array([1.0, 4.0]), properties=(age,))

    unary = {
        "neg": (lambda x: -x, -scalar),
        "exp": (jnp.exp, scalar),
        "log": (jnp.log, scalar),
        "abs": (jnp.abs, -scalar),
        "tanh": (jnp.tanh, scalar),
        "sqrt": (jnp.sqrt, scalar),
        "floor": (jnp.floor, scalar),
    }
    for op, (fn, sample) in unary.items():
        np.testing.assert_allclose(apply_unary(op, sample), fn(sample))
        np.testing.assert_allclose(apply_unary(op, array), fn(array))
        grouped_out = apply_unary(op, grouped)
        assert isinstance(grouped_out, GroupedRate)
        assert grouped_out.properties == (age,)
        np.testing.assert_allclose(np.asarray(grouped_out.data), fn(np.array([1.0, 4.0])))

    binary = {
        "pow": jnp.power,
        "maximum": jnp.maximum,
        "minimum": jnp.minimum,
    }
    other = np.array([2.0, 0.5, 0.5])
    other_g = GroupedRate(data=np.array([2.0, 0.5]), properties=(age,))
    for op, fn in binary.items():
        np.testing.assert_allclose(apply_binary(op, scalar, 2.0), fn(scalar, 2.0))
        np.testing.assert_allclose(apply_binary(op, array, other), fn(array, other))
        grouped_out = apply_binary(op, grouped, other_g)
        assert isinstance(grouped_out, GroupedRate)
        assert grouped_out.properties == (age,)
        np.testing.assert_allclose(
            np.asarray(grouped_out.data), fn(np.array([1.0, 4.0]), np.array([2.0, 0.5]))
        )


def test_constructors_build_nodes_and_apply_values() -> None:
    assert exp(Param("x")) == UnaryOp("exp", Param("x"))
    assert isinstance(Param("x") ** 2, type(Param("x") * 2))
    assert (-Param("x")).op == "neg"
    assert abs(Param("x")).op == "abs"
    from summer4.flows.rates import BinOp, Const

    node = 2 ** Param("p")
    assert isinstance(node, BinOp)
    assert node.op == "pow"
    assert node.left == Const(2.0)

    age = Property("age", ("young", "old"))
    grouped = GroupedRate(data=np.array([1.0, 9.0]), properties=(age,))
    rooted = sqrt(grouped)
    assert isinstance(rooted, GroupedRate)
    np.testing.assert_allclose(np.asarray(rooted.data), [1.0, 3.0])
    capped = maximum(grouped, 2.0)
    assert isinstance(capped, GroupedRate)
    np.testing.assert_allclose(np.asarray(capped.data), [2.0, 9.0])


def test_clip_is_maximum_minimum() -> None:
    expr = clip(Param("x"), 0, Param("hi"))
    from summer4.flows.rates import BinOp

    assert isinstance(expr, BinOp)
    assert expr.op == "minimum"
    assert isinstance(expr.left, BinOp)
    assert expr.left.op == "maximum"
    with pytest.raises(ValueError, match="lower bound"):
        clip(Param("x"))


def test_five_sites_see_unary_children() -> None:
    expr = clip(Param("h") * (1 - abs(Time() - Param("p")) / Param("w")), Param("lo"))
    assert _field_paths(expr) == {("h",), ("p",), ("w",), ("lo",)}
    assert _flow_refs(abs(FlowRef("seed"))) == {"seed"}
    assert _rate_bytes(exp(Param("x"))).startswith(b"unary:")
    assert _rate_bytes(log(Param("x"))).startswith(b"unary:")
    assert _rate_bytes(exp(Param("x"))) != _rate_bytes(log(Param("x")))
    assert _rate_bytes(Param("x") ** Param("p")).startswith(b"binop:")
    assert _rate_bytes(Param("x") ** Param("p")) != _rate_bytes(Param("x") * Param("p"))


def test_digest_distinguishes_exp_and_log() -> None:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)

    def digest(rate: object) -> bytes:
        model = FlowModel(pmap)
        model.add_flow(ExitFlow("out", state["Y"], rate))
        return model.compile()._digest

    assert digest(exp(Param("x"))) != digest(log(Param("x")))


def test_stage_and_hoist_of_parameter_only_unary() -> None:
    inner = Param("a") * Param("b")
    expr = exp(inner)
    assert rate_stage(expr, params_are_static=True) == "run"
    assert rate_stage(expr, params_are_static=False) == "step"
    assert rate_stage(exp(Time()), params_are_static=True) == "step"
    table = build_hoist_table([expr], params_are_static=True)
    assert len(table.entries) == 1

    # The parameter product inside a step-stage exp is still hoisted.
    stepped = exp(inner + Time())
    stepped_table = build_hoist_table([stepped], params_are_static=True)
    assert any(entry.node == inner for entry in stepped_table.entries)

    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(ExitFlow("out", state["Y"], stepped))
    y = np.array([1.0])
    params = {"a": 0.2, "b": 0.5}
    on = model.compile(hoist=True).vector_field(1.5, y, params)
    off = model.compile(hoist=False).vector_field(1.5, y, params)
    np.testing.assert_allclose(np.asarray(on), np.asarray(off))


def test_gradients_through_log_pow_maximum_and_floor() -> None:
    def log_loss(x: object) -> object:
        return eval_closed(log(Param("x")), {"x": x})

    np.testing.assert_allclose(jax.grad(log_loss)(2.0), 0.5)

    def pow_loss(p: object) -> object:
        return eval_closed(Param("x") ** Param("p"), {"x": 4.0, "p": p})

    p = 0.5
    np.testing.assert_allclose(jax.grad(pow_loss)(p), (4.0**p) * np.log(4.0))

    def max_loss(x: object) -> object:
        return eval_closed(maximum(Param("x"), 0.5), {"x": x})

    np.testing.assert_allclose(jax.grad(max_loss)(1.0), 1.0)
    np.testing.assert_allclose(jax.grad(max_loss)(0.0), 0.0)

    def floor_loss(x: object) -> object:
        # floor's own derivative is 0, so the sum's gradient is 1 and grad does not raise.
        return eval_closed(floor(Param("x")) + Param("x"), {"x": x})

    np.testing.assert_allclose(jax.grad(floor_loss)(1.7), 1.0)


def test_eval_closed_rejects_time_and_state() -> None:
    with pytest.raises(ValueError, match="parameter-only"):
        eval_closed(Time(), {})
    with pytest.raises(ValueError, match="parameter-only"):
        eval_closed(clip(Time(), 0), {})


def test_triangular_seed_and_tanh_scaleup_at_50_times() -> None:
    state = Property("state", ("S", "I"))
    pmap = PropertyMap.from_property(state)
    seed = clip(
        Param("height") * (1 - abs(Time() - Param("peak")) / Param("width")),
        0,
    )
    scale = (
        Param("start")
        + (Param("end") - Param("start"))
        * (tanh(Param("shape") * (Time() - Param("inflection"))) + 1)
        / 2
    )
    params = {
        "height": 0.05,
        "peak": 20.0,
        "width": 8.0,
        "shape": 0.3,
        "inflection": 18.0,
        "start": 0.1,
        "end": 1.0,
    }
    y = np.array([1.0, 0.0])
    times = np.linspace(0.0, 40.0, 50)

    def _series(rate: object) -> np.ndarray:
        lone = FlowModel(pmap)
        lone.add_flow(TransitionFlow("only", state["S"], state["I"], rate, absolute=True))
        cm = lone.compile()
        return np.array([float(np.asarray(cm.vector_field(float(t), y, params))[1]) for t in times])

    np.testing.assert_allclose(
        _series(seed),
        _triang(times, params["peak"], params["height"], params["width"]),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _series(scale),
        _tanh_scaleup(
            times,
            params["shape"],
            params["inflection"],
            params["start"],
            params["end"],
        ),
        rtol=1e-5,
        atol=1e-6,
    )


def test_grouped_pow_and_maximum() -> None:
    state = Property("state", ("S", "I"))
    loc = Property("location", ("north", "south"))
    pm = PropertyMap.from_property(state).stratify(loc)
    y = np.array([100.0, 300.0, 10.0, 30.0])
    params = {"death_rate": 0.1, "exp": 2.0, "floor": 5.0}
    north = 0.1 * (100.0 + 10.0)
    south = 0.1 * (300.0 + 30.0)
    s_north, s_south = pm.select(state["S"])

    powered = FlowModel(pm)
    death = powered.add_flow(ExitFlow("death", Everything(), Param("death_rate")))
    powered.add_flow(EntryFlow("birth", state["S"], death.sum_over(loc) ** Param("exp")))
    dy = np.asarray(powered.compile().vector_field(0.0, y, params))
    np.testing.assert_allclose(dy[s_north], -0.1 * 100.0 + north**2)
    np.testing.assert_allclose(dy[s_south], -0.1 * 300.0 + south**2)

    floored = FlowModel(pm)
    death = floored.add_flow(ExitFlow("death", Everything(), Param("death_rate")))
    floored.add_flow(EntryFlow("birth", state["S"], maximum(death.sum_over(loc), Param("floor"))))
    dy_floor = np.asarray(floored.compile().vector_field(0.0, y, params))
    np.testing.assert_allclose(dy_floor[s_north], -0.1 * 100.0 + max(north, 5.0))
    np.testing.assert_allclose(dy_floor[s_south], -0.1 * 300.0 + max(south, 5.0))


def test_param_transform_scales_a_trace() -> None:
    """The same ``tanh(Param)`` can scale a rate and, via eval_closed, an Output."""
    expr = tanh(Param("se"))
    params = {"se": 0.4}
    factor = eval_closed(expr, params)
    np.testing.assert_allclose(factor, np.tanh(0.4), rtol=1e-5, atol=1e-6)

    state = Property("state", ("S", "I"))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("detect", state["S"], state["I"], expr, absolute=True))
    dy = np.asarray(model.compile().vector_field(0.0, np.array([1.0, 0.0]), params))
    np.testing.assert_allclose(dy[1], factor)

    values = np.array([1.0, 2.0, 3.0, 4.0])
    scaled = _trace(values) * factor
    np.testing.assert_allclose(np.asarray(scaled.values), values * np.tanh(0.4), rtol=1e-5)
    reflected = factor * _trace(values)
    np.testing.assert_allclose(np.asarray(reflected.values), values * np.tanh(0.4), rtol=1e-5)

    with pytest.raises(TypeError, match="eval_closed"):
        _ = _trace(values) * expr
    with pytest.raises(TypeError, match="name-aligned"):
        _ = _trace(values) / _trace(values)

    out = tanh(_trace(values))
    np.testing.assert_allclose(np.asarray(out.values), np.tanh(values))
    clamped = clip(_trace(np.array([-1.0, 0.2, 2.0])), 0, 1)
    np.testing.assert_allclose(np.asarray(clamped.values), [0.0, 0.2, 1.0])


def test_trace_param_scale_is_differentiable() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    base = _trace(values)

    def loss(se: object) -> object:
        factor = eval_closed(tanh(Param("se")), {"se": se})
        return jnp.sum((base * factor).values)

    se = 0.3
    expected = float(values.sum()) * (1.0 - np.tanh(se) ** 2)
    np.testing.assert_allclose(jax.grad(loss)(se), expected)


def test_array_const_and_propertydata_use_the_same_kernel() -> None:
    arr = np.array([1.0, 4.0, 9.0])
    np.testing.assert_allclose(eval_closed(sqrt(ArrayConst(arr)), {}), np.sqrt(arr))

    state = Property("state", ("S", "I"))
    pmap = PropertyMap.from_property(state)
    pdata = PropertyData.wrap(pmap, np.array([[1.0, 4.0], [9.0, 16.0]]))
    traced = Output(
        times=TimeAxis(values=np.array([0.0, 1.0]), epoch=None, kind="explicit"),
        values=pdata,
        dims=("time", "compartment"),
    )
    scaled = traced * 0.5
    assert isinstance(scaled.values, PropertyData)
    assert scaled.values.pmap == pmap
    np.testing.assert_allclose(np.asarray(scaled.values.data), np.asarray(pdata.data) * 0.5)


def test_mismatched_grouped_rate_still_raises() -> None:
    age = Property("age", ("young", "old"))
    sex = Property("sex", ("f", "m"))
    left = GroupedRate(data=np.array([1.0, 2.0]), properties=(age,))
    right = GroupedRate(data=np.array([3.0, 4.0]), properties=(sex,))
    with pytest.raises(ValueError, match="groupings differ"):
        maximum(left, right)
