"""Custom ``RateOps`` nodes must carry a value-keyed digest.

``CompiledModel.__hash__`` and ``__eq__`` are both its ``_digest``, and the
model is a static argument to ``jax.jit``. A custom node whose digest ignores
its fields therefore makes two different models share one compiled program, and
the second silently returns the first's numerics. These tests pin the two
places that now prevent it: ``register_rate_eval`` refuses a class without
``__rate_bytes__``, and ``_rate_bytes`` raises rather than guessing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from summer4 import ExitFlow, FlowModel, Property, PropertyMap  # noqa: E402
from summer4.flows.rates import (  # noqa: E402
    RateOps,
    _rate_bytes,
    as_rate,
    register_rate_eval,
)


@dataclass(frozen=True, slots=True)
class Scaled(RateOps):
    """``k`` times an inner rate — the write-up's reproduction node, done right."""

    inner: RateOps
    k: float

    def __rate_bytes__(self) -> bytes:
        return b"scaled" + np.float64(self.k).tobytes() + _rate_bytes(self.inner)


@register_rate_eval(Scaled)
def _eval_scaled(
    expr: Scaled,
    *,
    eval_child: Callable[[RateOps], Any],
    **_: Any,
) -> Any:
    return expr.k * eval_child(expr.inner)


def _compile_with(k: float) -> Any:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(ExitFlow("out", state["Y"], Scaled(as_rate(0.1), k=k)))
    return model.compile()


def test_custom_node_fields_reach_the_digest() -> None:
    a = _compile_with(1.0)
    b = _compile_with(7.0)

    assert a._digest != b._digest
    assert a != b
    assert hash(a) != hash(b)


def test_custom_node_fields_key_the_jit_cache() -> None:
    a = _compile_with(1.0)
    b = _compile_with(7.0)
    y = np.array([10.0])

    # Unjitted, the two models already differ; the bug was that jit did not.
    np.testing.assert_allclose(np.asarray(a.vector_field(0.0, y, {})), [-1.0])
    np.testing.assert_allclose(np.asarray(b.vector_field(0.0, y, {})), [-7.0])

    def field(model: Any, y_arr: Any) -> Any:
        return model.vector_field(0.0, y_arr, {})

    jitted = jax.jit(field, static_argnums=0)
    np.testing.assert_allclose(np.asarray(jitted(a, y)), [-1.0])
    np.testing.assert_allclose(np.asarray(jitted(b, y)), [-7.0])


def test_registration_rejects_a_node_without_rate_bytes() -> None:
    @dataclass(frozen=True, slots=True)
    class Unkeyed(RateOps):
        inner: RateOps
        k: float

    with pytest.raises(TypeError, match=r"Unkeyed.*__rate_bytes__"):
        register_rate_eval(Unkeyed)(lambda expr, **_: expr.k)

    from summer4.flows.rates import _RATE_EVALUATORS

    assert Unkeyed not in _RATE_EVALUATORS


def test_rate_bytes_raises_for_a_node_without_the_dunder() -> None:
    @dataclass(frozen=True, slots=True)
    class Unkeyed(RateOps):
        k: float

    with pytest.raises(TypeError, match=r"Unkeyed.*__rate_bytes__"):
        _rate_bytes(Unkeyed(k=1.0))


def test_force_of_infection_still_registers() -> None:
    from summer4.epi import ForceOfInfection
    from summer4.flows.rates import _RATE_EVALUATORS

    assert ForceOfInfection in _RATE_EVALUATORS
    assert callable(getattr(ForceOfInfection, "__rate_bytes__", None))
