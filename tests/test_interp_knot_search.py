"""Knot search in ``Interp`` / ``TableInterp`` evaluation is loop-free and exact.

``jnp.searchsorted``'s default method lowers to an XLA ``while`` loop, which cost one loop of
small kernels per interpolated rate in every vector-field call (and in every recomputation of
a reverse-mode solve). ``_knot_index`` and ``_linear_interp`` must give the same answers and
gradients as ``jnp.searchsorted`` / ``jnp.interp`` without one.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import ExitFlow, FlowModel, Property, PropertyMap, Time
from summer4.data import Data
from summer4.flows.compiled import _COMPARE_ALL_MAX_KNOTS, _knot_index, _linear_interp


def _primitives(jaxpr: Any) -> set[str]:
    """Every primitive name in ``jaxpr``, nested sub-jaxprs included."""
    core = jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr
    names: set[str] = set()
    for eqn in core.eqns:
        names.add(eqn.primitive.name)
        for value in eqn.params.values():
            for sub in value if isinstance(value, (list, tuple)) else (value,):
                if hasattr(sub, "eqns") or hasattr(sub, "jaxpr"):
                    names |= _primitives(sub)
    return names


@pytest.mark.parametrize("n_knots", [1, 2, 7, 152, _COMPARE_ALL_MAX_KNOTS + 5])
@pytest.mark.parametrize("side", ["left", "right"])
def test_knot_index_matches_numpy(n_knots: int, side: str) -> None:
    rng = np.random.default_rng(n_knots)
    xs = np.sort(rng.integers(0, max(2, n_knots // 2), size=n_knots).astype(float))
    queries = np.concatenate(
        [xs, xs + 0.25, [xs[0] - 1.0, xs[-1] + 1.0], rng.uniform(-1, n_knots, 50)]
    )
    got = np.asarray(jax.jit(lambda q: _knot_index(jnp.asarray(xs), q, side))(queries))
    np.testing.assert_array_equal(got, np.searchsorted(xs, queries, side=side))


@pytest.mark.parametrize("n_knots", [2, 29, _COMPARE_ALL_MAX_KNOTS + 5])
def test_linear_interp_matches_jnp_interp(n_knots: int) -> None:
    rng = np.random.default_rng(1)
    xs = np.sort(rng.uniform(0.0, 10.0, n_knots))
    xs[n_knots // 2] = xs[n_knots // 2 - 1]  # a repeated knot
    vals = rng.normal(size=n_knots)
    queries = np.concatenate([xs, [-1.0, 11.0], rng.uniform(-0.5, 10.5, 40)])
    ours = jax.vmap(lambda q: _linear_interp(q, jnp.asarray(xs), jnp.asarray(vals)))(queries)
    ref = jnp.interp(queries, xs, vals)
    eps = float(np.finfo(np.asarray(ref).dtype).eps)
    np.testing.assert_allclose(np.asarray(ours), np.asarray(ref), rtol=4 * eps, atol=4 * eps)

    def grads(fn: Any) -> Any:
        return jax.vmap(jax.grad(fn, argnums=(0, 1)), in_axes=(0, None))(
            jnp.asarray(queries), jnp.asarray(vals)
        )

    g_ours = grads(lambda q, v: _linear_interp(q, jnp.asarray(xs), v))
    g_ref = grads(lambda q, v: jnp.interp(q, jnp.asarray(xs), v))
    for a, b in zip(g_ours, g_ref, strict=True):
        assert bool(jnp.all(jnp.isfinite(a)))
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=4 * eps, atol=4 * eps)


@pytest.mark.parametrize("kind", ["linear", "sigmoidal", "step"])
def test_interp_vector_field_has_no_while_loop(kind: str) -> None:
    times = np.linspace(1950.0, 2100.0, 152)
    values = np.column_stack([np.linspace(0.1, 0.2, times.size), np.linspace(0.3, 0.4, times.size)])
    age = Property("age", ("0", "15"))
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state).stratify(age)
    model = FlowModel(pmap)
    table = Data.table(times, values, over=age).interp(kind, sharpness=16.0)
    scalar = Data(times, values[:, 0]).interp("linear")
    model.add_flow(ExitFlow("die", state["Y"], table * scalar * (1.0 + 0.0 * Time())))
    cm = model.compile()
    y = np.ones(pmap.size)

    def field(t: Any) -> Any:
        return jnp.asarray(cm.vector_field(t, y, {}))

    # ``fori_loop`` with static bounds traces to ``scan``; XLA lowers both to a ``while``.
    loops = {"while", "scan"}
    assert not loops & _primitives(jax.make_jaxpr(field)(2000.5))
    grad_jaxpr = jax.make_jaxpr(jax.grad(lambda t: jnp.sum(field(t))))(2000.5)
    assert not loops & _primitives(grad_jaxpr)
