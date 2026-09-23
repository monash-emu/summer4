"""Fused vs per-flow compartment-update scatters produce the same dy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
from benchmarks.tb_scale import T0, build_tb_scale_model

from summer4 import (
    EntryFlow,
    Everything,
    ExitFlow,
    FlowModel,
    Property,
    PropertyMap,
    TransitionFlow,
)


def _sir_builder() -> FlowModel:
    state = Property("state", ("S", "I", "R"))
    pm = PropertyMap.from_property(state)
    model = FlowModel(pm)
    model.add_flow(TransitionFlow("inf", state["S"], state["I"], 0.1))
    death = model.add_flow(ExitFlow("death", Everything(), 0.01))
    model.add_flow(EntryFlow("birth", state["S"], death.sum()))
    model.add_flow(TransitionFlow("rec", state["I"], state["R"], 0.05))
    return model


def test_fuse_compartment_updates_matches_per_flow_dy() -> None:
    model = _sir_builder()
    fused = model.compile(fuse_compartment_updates=True)
    looped = model.compile(fuse_compartment_updates=False)
    assert fused.fuse_compartment_updates is True
    assert looped.fuse_compartment_updates is False
    assert fused._digest != looped._digest

    y = np.array([900.0, 80.0, 20.0])
    dy_f = np.asarray(fused.vector_field(0.0, y, {}))
    dy_l = np.asarray(looped.vector_field(0.0, y, {}))
    np.testing.assert_allclose(dy_f, dy_l, rtol=0.0, atol=0.0)

    ctx_f = fused.observe(0.0, y, {})
    ctx_l = looped.observe(0.0, y, {})
    assert set(ctx_f.flows) == set(ctx_l.flows)
    for name in ctx_f.flows:
        np.testing.assert_allclose(
            np.asarray(ctx_f.flows[name]),
            np.asarray(ctx_l.flows[name]),
            rtol=0.0,
            atol=0.0,
        )


def test_fuse_compartment_updates_jit_and_grad() -> None:
    model = _sir_builder()
    fused = model.compile(fuse_compartment_updates=True)
    looped = model.compile(fuse_compartment_updates=False)
    y = jnp.asarray([900.0, 80.0, 20.0])

    def loss(cm: Any, state: Any) -> Any:
        dy = cm.vector_field(0.0, state, {})
        return jnp.sum(dy**2)

    jf = jax.jit(lambda s: loss(fused, s))
    jl = jax.jit(lambda s: loss(looped, s))
    # Fused vs sequential scatter-add changes float32 reduction order.
    np.testing.assert_allclose(float(jf(y)), float(jl(y)), rtol=1e-5, atol=0.0)

    gf = jax.grad(lambda s: loss(fused, s))(y)
    gl = jax.grad(lambda s: loss(looped, s))(y)
    np.testing.assert_allclose(np.asarray(gf), np.asarray(gl), rtol=1e-5, atol=0.0)


def test_fuse_reduces_vf_scatter_add_count() -> None:
    model = _sir_builder()
    y = np.array([900.0, 80.0, 20.0])
    fused = model.compile(fuse_compartment_updates=True)
    looped = model.compile(fuse_compartment_updates=False)

    def count_scatter_add(cm: Any) -> int:
        jaxpr = jax.make_jaxpr(cm.vector_field)(0.0, y, {})
        return sum(1 for e in jaxpr.jaxpr.eqns if e.primitive.name == "scatter-add")

    n_fused = count_scatter_add(fused)
    n_looped = count_scatter_add(looped)
    # Per-flow: transition×2 + exit + entry scatters; fused collapses to 1.
    # FlowRef Reduce may still emit scatters for segment_sum.
    assert n_looped > n_fused
    assert n_fused >= 1


def _as_array(dy: Any) -> np.ndarray:
    data = dy.data if hasattr(dy, "data") else dy
    return np.asarray(data)


def test_tb_scale_fuse_matches_looped_dy() -> None:
    fused_m = build_tb_scale_model(fuse_compartment_updates=True)
    looped_m = build_tb_scale_model(fuse_compartment_updates=False)
    y0 = fused_m.compiled.initial_state(fused_m.compiled.prepare(fused_m.params))
    prepared = fused_m.compiled.prepare(fused_m.params)
    dy_f = _as_array(fused_m.compiled.vector_field(T0, y0, prepared))
    dy_l = _as_array(looped_m.compiled.vector_field(T0, y0, prepared))
    np.testing.assert_allclose(dy_f, dy_l, rtol=1e-10, atol=1e-10)
