"""NumPy ufuncs, array functions and named helpers on post-solve output expressions.

Every test here asserts evaluated values against NumPy on the raw series, never a
node class or a ``kind`` string.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import Compartments, Output, OutputSet, Param, Result, TimeAxis
from summer4.flows.rates import clip, exp, maximum, minimum, tanh

# Strictly positive so ``np.log`` is defined on every point.
VALUES = np.array([0.5, 1.0, 2.0, 4.0])


def _axis(times: np.ndarray) -> TimeAxis:
    return TimeAxis(values=times, epoch=None, kind="explicit")


def _scalar_result(name: str, values: np.ndarray) -> Result:
    times = np.arange(values.shape[0], dtype=np.float64)
    axis = _axis(times)
    output = Output(times=axis, values=values, dims=("time",))
    return Result(times=axis, outputs={name: output})


def _evaluate(build: Callable[[Any], Any], params: object = None) -> np.ndarray:
    """Build one expression on the output ``pop`` and return its evaluated series."""
    outputs = OutputSet()
    outputs["pop"] = Compartments()
    outputs["expr"] = build(outputs.ref("pop"))
    named = outputs.evaluate(_scalar_result("pop", VALUES), params)
    return np.asarray(named["expr"].values)


@pytest.mark.parametrize(
    ("build", "reference"),
    [
        pytest.param(lambda ref: np.exp(ref), lambda v: np.exp(v), id="np.exp"),
        pytest.param(lambda ref: np.sin(ref), lambda v: np.sin(v), id="np.sin"),
        pytest.param(lambda ref: np.log(ref), lambda v: np.log(v), id="np.log"),
        pytest.param(
            lambda ref: np.maximum(ref, 2.5),
            lambda v: np.maximum(v, 2.5),
            id="np.maximum",
        ),
        pytest.param(
            lambda ref: np.minimum(ref, 2.5),
            lambda v: np.minimum(v, 2.5),
            id="np.minimum",
        ),
        pytest.param(
            lambda ref: np.clip(ref, 1.0, 3.0),
            lambda v: np.clip(v, 1.0, 3.0),
            id="np.clip",
        ),
        pytest.param(lambda ref: tanh(ref), lambda v: np.tanh(v), id="tanh"),
        pytest.param(lambda ref: exp(ref), lambda v: np.exp(v), id="exp"),
        pytest.param(lambda ref: maximum(ref, 2.5), lambda v: np.maximum(v, 2.5), id="maximum"),
        pytest.param(lambda ref: minimum(ref, 2.5), lambda v: np.minimum(v, 2.5), id="minimum"),
        pytest.param(
            lambda ref: clip(ref, 1.0, 3.0),
            lambda v: np.clip(v, 1.0, 3.0),
            id="clip",
        ),
    ],
)
def test_function_of_output_evaluates_like_numpy(
    build: Callable[[Any], Any], reference: Callable[[np.ndarray], np.ndarray]
) -> None:
    np.testing.assert_allclose(_evaluate(build), reference(VALUES), rtol=1e-6)


@pytest.mark.parametrize(
    ("build", "factor"),
    [
        pytest.param(lambda ref: np.array(2.0) * ref, 2.0, id="ndarray-left"),
        pytest.param(lambda ref: Param("s") * ref, 3.0, id="param-left"),
        pytest.param(lambda ref: ref * Param("s"), 3.0, id="param-right"),
        pytest.param(lambda ref: np.multiply(Param("s"), ref), 3.0, id="np.multiply-param-left"),
    ],
)
def test_scale_by_array_or_parameter_on_either_side(
    build: Callable[[Any], Any], factor: float
) -> None:
    np.testing.assert_allclose(_evaluate(build, {"s": 3.0}), factor * VALUES, rtol=1e-6)


def test_comparison_ufunc_is_refused() -> None:
    outputs = OutputSet()
    outputs["pop"] = Compartments()
    with pytest.raises(ValueError, match="cannot be a rate"):
        np.greater(outputs.ref("pop"), 0.0)


def test_exp_of_output_under_jit_and_grad() -> None:
    outputs = OutputSet()
    outputs["pop"] = Compartments()
    outputs["scaled"] = np.exp(outputs.ref("pop")) * Param("scale")
    result = _scalar_result("pop", np.array([4.0, 4.0, 4.0]))

    def loss(scale: Any) -> Any:
        out = outputs.evaluate(result, {"scale": scale})
        return jnp.sum(jnp.asarray(out["scaled"].values))

    # exp(4) * scale, summed over three points: value 6 * exp(4) at scale 2,
    # gradient 3 * exp(4).
    expected = 3.0 * float(np.exp(4.0))
    assert float(jax.jit(loss)(jnp.array(2.0))) == pytest.approx(2.0 * expected, rel=1e-5)
    assert float(jax.grad(loss)(jnp.array(2.0))) == pytest.approx(expected, rel=1e-5)
