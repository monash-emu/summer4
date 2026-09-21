"""NumPy dispatch on rate expressions and the evaluated wrappers."""

from __future__ import annotations

import functools

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from summer4 import (  # noqa: E402
    BinOp,
    EntryFlow,
    FlowModel,
    GroupedRate,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    Time,
    Trace,
    UnaryOp,
    clip,
    exp,
)
from summer4.flows.rates import ArrayConst, Const, FieldRef, _rate_bytes  # noqa: E402
from summer4.flows.stages import rate_stage  # noqa: E402
from summer4.time import TimeAxis  # noqa: E402

# Captured from main before this step. ``np.multiply`` must stay byte-identical
# to ``*``; these literals are the regression net for that.
_GOLDEN_MUL = "62696e6f703a6d756c6669656c64282778272c29636f6e73740000000000000040"
_GOLDEN_EXP = "756e6172793a6578706669656c64282778272c29"
_GOLDEN_ADD = "62696e6f703a6164646669656c64282778272c296669656c64282778272c29"


def _trace(values: np.ndarray) -> Trace:
    times = np.linspace(0.0, 1.0, values.shape[0])
    return Trace(
        times=TimeAxis(values=times, epoch=None, kind="explicit"),
        values=values,
        dims=("time",),
    )


def _entry(expr: object) -> tuple[object, PropertyMap]:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], expr))
    return model.compile(), pmap


def test_digest_goldens_are_unchanged() -> None:
    assert _rate_bytes(Param("x") * 2).hex() == _GOLDEN_MUL
    assert _rate_bytes(exp(Param("x"))).hex() == _GOLDEN_EXP
    assert _rate_bytes(Param("x") + Param("x")).hex() == _GOLDEN_ADD


def test_ufunc_aliases_match_operators() -> None:
    left = Param("x")
    right = Param("y")
    pairs = (
        (np.multiply(left, right), left * right),
        (np.subtract(left, right), left - right),
        (np.divide(left, right), left / right),
        (np.true_divide(left, right), left / right),
        (np.power(left, right), left**right),
        (np.negative(left), -left),
        (np.absolute(left), abs(left)),
        (np.fabs(left), abs(left)),
    )
    for via_numpy, via_op in pairs:
        assert via_numpy == via_op
        assert _rate_bytes(via_numpy) == _rate_bytes(via_op)


def test_sin_compiles_to_the_numpy_seasonal_rate() -> None:
    expr = 0.1 * (1.0 + 0.5 * np.sin(Param("phase")))
    assert isinstance(np.sin(Param("phase")), UnaryOp)
    assert np.sin(Param("phase")).op == "sin"
    compiled, pmap = _entry(expr)
    y = PropertyData.wrap(pmap, np.array([1.0]))
    phase = 0.4
    dy = float(np.asarray(compiled.vector_field(0.0, y, {"phase": phase}).data)[0])
    expected = 0.1 * (1.0 + 0.5 * np.sin(phase))
    np.testing.assert_allclose(dy, expected, rtol=1e-5, atol=1e-5)


def test_staging_follows_the_argument_not_the_op() -> None:
    assert rate_stage(np.sin(Param("p")), params_are_static=True) == "run"
    assert rate_stage(np.sin(Time()), params_are_static=True) == "step"


def test_sin_models_share_a_jit_cache_entry_and_cos_does_not() -> None:
    sin_a, pmap = _entry(np.sin(Param("phase")))
    sin_b, _ = _entry(np.sin(Param("phase")))
    cos, _ = _entry(np.cos(Param("phase")))
    assert sin_a == sin_b
    assert sin_a != cos
    y = PropertyData.wrap(pmap, np.array([1.0]))
    traces = {"n": 0}

    @functools.partial(jax.jit, static_argnums=0)
    def _vf(model: object, state: PropertyData, params: dict[str, float]) -> object:
        traces["n"] += 1
        return model.vector_field(0.0, state, params)  # type: ignore[attr-defined]

    _vf(sin_a, y, {"phase": 0.2})
    _vf(sin_b, y, {"phase": 1.7})
    assert traces["n"] == 1
    _vf(cos, y, {"phase": 0.2})
    assert traces["n"] == 2


def test_numpy_ufunc_preserves_each_wrapper() -> None:
    age = Property("age", ("young", "old"))
    grouped = GroupedRate(data=np.array([1.0, 4.0]), properties=(age,))
    got_grouped = np.exp(grouped)
    assert isinstance(got_grouped, GroupedRate)
    assert got_grouped.properties == (age,)

    pmap = PropertyMap.from_property(age)
    data = PropertyData.wrap(pmap, np.array([1.0, 2.0]))
    got_data = np.exp(data)
    assert isinstance(got_data, PropertyData)
    assert got_data.pmap == pmap

    trace = _trace(np.array([1.0, 2.0]))
    got_trace = np.exp(trace)
    assert isinstance(got_trace, Trace)
    assert got_trace.dims == ("time",)

    node = np.exp(Param("x"))
    assert isinstance(node, UnaryOp)
    assert node.op == "exp"


def test_mixed_symbolic_and_evaluated_names_eval_closed() -> None:
    trace = _trace(np.array([1.0, 2.0]))
    with pytest.raises(TypeError, match="eval_closed"):
        np.multiply(trace, Param("x"))


def test_reductions_in_place_masks_and_matmul_are_refused() -> None:
    with pytest.raises(TypeError):
        np.add.reduce(Param("x"))
    with pytest.raises(TypeError):
        np.exp(Param("x"), out=np.array(0.0))
    with pytest.raises(ValueError, match="greater"):
        np.greater(Param("x"), 1.0)
    with pytest.raises(ValueError, match="matmul"):
        np.matmul(Param("x"), Param("y"))


def test_numpy_scalar_interop_is_unchanged() -> None:
    assert np.float64(2.0) * Param("x") == BinOp("mul", Const(2.0), FieldRef(("x",)))
    assert Param("x") * np.float64(2.0) == BinOp("mul", FieldRef(("x",)), Const(2.0))
    assert np.int32(3) * Param("x") == BinOp("mul", Const(3.0), FieldRef(("x",)))
    assert 2.0 * Param("x") == BinOp("mul", Const(2.0), FieldRef(("x",)))
    assert _rate_bytes(exp(Param("x"))).hex() == _GOLDEN_EXP


def test_ndarray_times_rate_is_one_node() -> None:
    node = np.array([1.0, 2.0]) * Param("x")
    assert isinstance(node, BinOp)
    assert node.op == "mul"
    assert isinstance(node.left, ArrayConst)
    np.testing.assert_allclose(node.left.value, [1.0, 2.0])
    assert node.right == FieldRef(("x",))
    assert not isinstance(node, np.ndarray)


def test_numpy_clip_matches_summer4_clip() -> None:
    expr = Param("x")
    assert np.clip(expr, 0.0, 1.0) == clip(expr, 0.0, 1.0)
    assert np.clip(expr, 0.0, None) == clip(expr, 0.0, None)
    assert np.clip(expr, None, 1.0) == clip(expr, None, 1.0)
    with pytest.raises(ValueError, match="clip requires"):
        np.clip(expr, None, None)


def test_unknown_op_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="nope"):
        UnaryOp("nope", Param("x"))


def test_grouped_matmul_still_uses_the_operator() -> None:
    age = Property("age", ("young", "old"))
    grouped = GroupedRate(data=np.array([1.0, 4.0]), properties=(age,))
    matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
    product = grouped @ matrix
    assert isinstance(product, GroupedRate)
    assert product.properties == (age,)
    np.testing.assert_allclose(np.asarray(product.data), [2.0, 12.0])
