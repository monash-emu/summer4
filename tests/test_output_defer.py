"""defer(f)(outputs.ref(...), Param(...)): an ordinary function of solved outputs."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import gamma as jax_gamma
from scipy.stats import gamma as scipy_gamma

from summer4 import (
    Compartments,
    Defer,
    FlowModel,
    Output,
    OutputSet,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    Result,
    Time,
    TimeAxis,
    TransitionFlow,
    defer,
    eval_closed,
)
from summer4.results.outputset import OutputExpr

KERNEL_WEEKS = 26
GAMMA_SHAPE = 4.0


def _axis(n: int) -> TimeAxis:
    return TimeAxis(values=np.arange(n, dtype=np.float64), epoch=None, kind="explicit")


def _series_result(name: str, values: np.ndarray) -> Result:
    axis = _axis(values.shape[0])
    return Result(times=axis, outputs={name: Output(times=axis, values=values, dims=("time",))})


def _raw(series: Any) -> Any:
    """The data of an Output as a JAX array, whichever form its values take."""
    values = series.values.data if isinstance(series.values, PropertyData) else series.values
    return jnp.asarray(values)


def _gamma_kernel(delay_days: Any, n: int) -> Any:
    """Weekly bins of a gamma distribution with mean ``delay_days``, summing to 1."""
    edges = jnp.arange(n + 1, dtype=jnp.float32)
    cdf = jax_gamma.cdf(edges, GAMMA_SHAPE, scale=delay_days / 7.0 / GAMMA_SHAPE)
    weights = jnp.diff(cdf)
    return weights / jnp.sum(weights)


def _numpy_gamma_kernel(delay_days: float, n: int) -> np.ndarray:
    """The same kernel in NumPy and SciPy, for the reference convolution."""
    edges = np.arange(n + 1, dtype=np.float64)
    cdf = scipy_gamma.cdf(edges, GAMMA_SHAPE, scale=delay_days / 7.0 / GAMMA_SHAPE)
    weights = np.diff(cdf)
    return weights / weights.sum()


def delayed(series: Any, delay: Any) -> Any:
    """Causal gamma-delay convolution of a weekly series; the same length as the input."""
    x = _raw(series)
    kernel = _gamma_kernel(delay, KERNEL_WEEKS)
    return jnp.convolve(x, kernel, mode="full")[: x.shape[0]]


def _incidence(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    return 100.0 + 40.0 * np.sin(2.0 * np.pi * t / 17.0) + 5.0 * t / n


def test_gamma_delay_convolution_matches_numpy() -> None:
    raw = _incidence(60)
    outputs = OutputSet()
    outputs["inc"] = Compartments()
    outputs["notif"] = defer(delayed)(outputs.ref("inc"), Param("delay"))

    named = outputs.evaluate(_series_result("inc", raw), {"delay": 10.0})
    expected = np.convolve(raw, _numpy_gamma_kernel(10.0, KERNEL_WEEKS), mode="full")[:60]
    got = np.asarray(named["notif"].values)
    assert got.shape == raw.shape
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)


def test_longer_delay_moves_the_peak_later_lower_and_keeps_the_area() -> None:
    # A pulse well away from both ends, so the kernel's tail never leaves the series.
    t = np.arange(104, dtype=np.float64)
    pulse = 1000.0 * np.exp(-0.5 * ((t - 30.0) / 4.0) ** 2)
    outputs = OutputSet()
    outputs["inc"] = Compartments()
    outputs["notif"] = defer(delayed)(outputs.ref("inc"), Param("delay"))
    result = _series_result("inc", pulse)

    peaks: list[tuple[int, float]] = []
    for delay in (3.0, 21.0, 42.0):
        notif = np.asarray(outputs.evaluate(result, {"delay": delay})["notif"].values)
        peaks.append((int(np.argmax(notif)), float(notif.max())))
        assert notif.sum() == pytest.approx(pulse.sum(), rel=1e-4)
    (early_day, early_height), (mid_day, mid_height), (late_day, late_height) = peaks
    assert early_day < mid_day < late_day
    assert early_height > mid_height > late_height


def test_return_contract_keeps_an_output_as_is() -> None:
    raw = _incidence(12)

    def doubled(series: Any) -> Output:
        return series._with(values=2.0 * _raw(series))

    outputs = OutputSet()
    outputs["inc"] = Compartments()
    outputs["twice"] = defer(doubled)(outputs.ref("inc"))
    named = outputs.evaluate(_series_result("inc", raw), None)
    np.testing.assert_allclose(np.asarray(named["twice"].values), 2.0 * raw, rtol=1e-6)
    assert named["twice"].dims == ("time",)


def test_same_shape_array_on_property_data_keeps_the_property_map() -> None:
    state = Property("state", ("S", "I"))
    pmap = PropertyMap.from_property(state)
    raw = np.arange(6, dtype=np.float64).reshape(3, 2)
    axis = _axis(3)
    series = Output(times=axis, values=PropertyData.wrap(pmap, raw), dims=("time", "state"))

    def times_three(x: Any) -> Any:
        return 3.0 * _raw(x)

    outputs = OutputSet()
    outputs["pop"] = Compartments()
    outputs["scaled"] = defer(times_three)(outputs.ref("pop"))
    result = Result(times=axis, outputs={"pop": series})
    named = outputs.evaluate(result, None)
    scaled = named["scaled"].values
    assert isinstance(scaled, PropertyData)
    assert scaled.pmap == pmap
    np.testing.assert_allclose(np.asarray(scaled.data), 3.0 * raw)


def test_output_passed_as_keyword_is_saved_by_plan_and_evaluates() -> None:
    state = Property("state", ("S", "I"))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("recovery", state["I"], state["S"], Param("gamma")))
    compiled = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([9.0, 1.0]))
    infected = Compartments(where=state["I"]).total()

    def scaled_by(x: Any, factor: Any) -> Any:
        return _raw(x) * factor

    outputs = OutputSet()
    outputs["scaled"] = defer(scaled_by)(x=infected, factor=Param("factor"))
    plan = outputs.plan()
    assert "scaled" in plan.requests
    assert plan.requests["scaled"].what == Compartments(where=state["I"])

    params = {"gamma": 0.3, "factor": 2.5}
    solved = compiled.run(params, y0, t0=0.0, steps=4, dt=1.0, save=plan)
    named = outputs.evaluate(solved, params)
    raw_total = np.asarray(solved["scaled"].total().values)
    np.testing.assert_allclose(np.asarray(named["scaled"].values), 2.5 * raw_total, rtol=1e-5)


def test_forward_reference_resolves() -> None:
    raw = _incidence(8)

    def double(series: Any) -> Any:
        return 2.0 * _raw(series)

    outputs = OutputSet()
    outputs["b"] = defer(double)(outputs.ref("a"))
    outputs["a"] = Compartments()
    named = outputs.evaluate(_series_result("a", raw), None)
    assert list(named.keys()) == ["b", "a"]
    np.testing.assert_allclose(np.asarray(named["b"].values), 2.0 * raw, rtol=1e-6)


def test_grad_in_delay_is_finite_nonzero_and_jits() -> None:
    raw = _incidence(60)
    outputs = OutputSet()
    outputs["inc"] = Compartments()
    outputs["notif"] = defer(delayed)(outputs.ref("inc"), Param("delay"))
    result = _series_result("inc", raw)

    def loss(delay: Any) -> Any:
        return jnp.sum(jnp.asarray(outputs.evaluate(result, {"delay": delay})["notif"].values))

    value = float(jax.jit(loss)(jnp.array(10.0)))
    expected = float(np.sum(np.convolve(raw, _numpy_gamma_kernel(10.0, KERNEL_WEEKS))[:60]))
    assert value == pytest.approx(expected, rel=1e-4)
    grad = float(jax.jit(jax.grad(loss))(jnp.array(10.0)))
    assert np.isfinite(grad)
    assert grad != 0.0


def test_wrong_shape_raises_type_error_naming_the_function() -> None:
    def bad_shape(series: Any, delay: Any) -> Any:
        del series, delay
        return jnp.ones(3)

    outputs = OutputSet()
    outputs["inc"] = Compartments()
    outputs["bad"] = defer(bad_shape)(outputs.ref("inc"), Param("delay"))
    with pytest.raises(TypeError, match="bad_shape") as info:
        outputs.evaluate(_series_result("inc", _incidence(5)), {"delay": 2.0})
    assert "(3,)" in str(info.value)
    assert "(5,)" in str(info.value)


def test_mixed_time_and_output_arguments_raise_value_error_at_evaluate() -> None:
    def add_time(t: Any, series: Any) -> Any:
        return jnp.asarray(t) + _raw(series)

    outputs = OutputSet()
    outputs["inc"] = Compartments()
    outputs["mixed"] = defer(add_time)(Time(), outputs.ref("inc"))
    with pytest.raises(ValueError, match="depends on time"):
        outputs.evaluate(_series_result("inc", _incidence(5)), {"delay": 2.0})


def test_array_return_without_an_output_argument_raises_type_error() -> None:
    # No public spelling reaches this branch: a leaf, a ref, or any expression built on one
    # evaluates to an Output. A parameter-only OutputExpr, built directly, does.
    def constant_array(value: Any) -> Any:
        return jnp.ones(3) * value

    outputs = OutputSet()
    outputs["inc"] = Compartments()
    outputs["flat"] = defer(constant_array)(OutputExpr(kind="param", param=Param("scale")))
    with pytest.raises(TypeError, match="constant_array") as info:
        outputs.evaluate(_series_result("inc", _incidence(5)), {"scale": 2.0})
    assert "no Output argument" in str(info.value)


def test_all_parameter_defer_still_builds_a_rate_node() -> None:
    def product(a: Any, b: Any) -> Any:
        return jnp.asarray(a) * jnp.asarray(b)

    node = defer(product)(Param("a"), Param("b"))
    assert float(eval_closed(node, {"a": 2.0, "b": 3.0})) == pytest.approx(6.0)


def test_direct_defer_node_still_rejects_an_output() -> None:
    outputs = OutputSet()
    with pytest.raises(TypeError, match="as a flow rate"):
        Defer(lambda value: value, outputs.ref("inc"))
