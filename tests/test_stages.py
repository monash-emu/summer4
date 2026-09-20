"""Run-start prepare_fn stage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from tests.helpers.jaxpr import loop_body_primitives, outside_loop_primitives  # noqa: E402

from summer4 import (  # noqa: E402
    Compartments,
    EntryFlow,
    FlowModel,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
)
from summer4.flows.stages import Prepared  # noqa: E402


def _entry_model() -> tuple[FlowModel, PropertyMap, PropertyData]:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], Param("beta")))
    y0 = PropertyData.wrap(pmap, np.array([0.0]))
    return model, pmap, y0


def _run_final(cm: Any, params: object, y0: Any, *, solver: str) -> float:
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([1.0]))
    res = cm.run(params, y0, t0=0.0, t1=1.0, dt=0.25, save=plan, solver=solver)
    return float(np.asarray(res["y"].values.data)[-1, 0])


@pytest.mark.parametrize("solver", ["euler", "tsit5"])
def test_prepare_fn_feeds_rates(solver: str) -> None:
    model, _pmap, y0 = _entry_model()

    def prepare_fn(p: dict[str, float]) -> dict[str, float]:
        return {**p, "beta": p["r0"] / p["period"]}

    cm = model.compile(prepare_fn=prepare_fn)
    got = _run_final(cm, {"r0": 2.0, "period": 4.0}, y0, solver=solver)

    bare_model, _pmap2, y0b = _entry_model()
    cm_bare = bare_model.compile()
    expected = _run_final(cm_bare, {"beta": 0.5}, y0b, solver=solver)
    rtol = 0.0 if solver == "euler" else 1e-9
    np.testing.assert_allclose(got, expected, rtol=rtol)


@pytest.mark.parametrize("solver", ["euler", "tsit5"])
def test_prepare_fn_runs_at_run_start(solver: str) -> None:
    model, pmap, y0 = _entry_model()

    def prepare_fn(p: dict[str, Any]) -> dict[str, Any]:
        # cumsum is distinctive in the jaxpr and must sit outside the loop body.
        stacked = jnp.cumsum(jnp.asarray([p["r0"], p["period"]]))
        return {**p, "beta": stacked[0] / stacked[1]}

    cm = model.compile(prepare_fn=prepare_fn)
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([1.0]))

    def loss(params: dict[str, float]) -> Any:
        res = cm.run(params, y0, t0=0.0, t1=1.0, dt=0.25, save=plan, solver=solver)
        return jnp.sum(jnp.asarray(res["y"].values.data))

    params = {"r0": 2.0, "period": 4.0}
    closed = jax.make_jaxpr(loss)(params)
    assert "cumsum" not in loop_body_primitives(closed)
    assert "cumsum" in outside_loop_primitives(closed)


def test_raw_params_still_work() -> None:
    model, _pmap, y0 = _entry_model()
    cm = model.compile(prepare_fn=lambda p: {**p, "beta": p["r0"] / p["period"]})
    params = {"r0": 2.0, "period": 4.0}
    prepared = cm.prepare(params)
    t = 0.0
    dy_raw = cm.observe(t, y0, params).dy
    dy_prep = cm.observe(t, y0, prepared).dy
    np.testing.assert_allclose(np.asarray(dy_raw.data), np.asarray(dy_prep.data))
    vf_raw = cm.vector_field(t, y0, params)
    vf_prep = cm.vector_field(t, y0, prepared)
    np.testing.assert_allclose(np.asarray(vf_raw.data), np.asarray(vf_prep.data))
    assert isinstance(prepared, Prepared)


def test_prepare_boxes_python_floats() -> None:
    """Plain float leaves become floating arrays; ints stay Python ints."""
    import equinox as eqx

    model, _pmap, _y0 = _entry_model()
    cm = model.compile()
    prepared = cm.prepare({"beta": 0.1, "n_steps": 3})
    beta = prepared.params["beta"]
    assert eqx.is_array(beta)
    assert jnp.issubdtype(beta.dtype, jnp.floating)
    np.testing.assert_allclose(np.asarray(beta), 0.1)
    assert prepared.params["n_steps"] == 3
    assert isinstance(prepared.params["n_steps"], int)


def test_prepare_boxes_prepared_input() -> None:
    """Manually built Prepared with floats is re-boxed on prepare()."""
    import equinox as eqx

    model, _pmap, _y0 = _entry_model()
    cm = model.compile()
    raw = Prepared({"beta": 0.25}, ())
    prepared = cm.prepare(raw)
    assert eqx.is_array(prepared.params["beta"])
    assert jnp.issubdtype(prepared.params["beta"].dtype, jnp.floating)


def test_digest_differs_with_prepare_fn() -> None:
    model, _pmap, _y0 = _entry_model()
    cm0 = model.compile()
    cm1 = model.compile(prepare_fn=lambda p: p)
    assert cm0 != cm1
