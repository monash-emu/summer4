"""Stage classification and rate-subtree hoisting."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from tests.helpers.jaxpr import loop_body_primitives  # noqa: E402

from summer4 import (  # noqa: E402
    Compartments,
    EntryFlow,
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
    TransitionFlow,
)
from summer4.epi import ForceOfInfection, MixingMatrix  # noqa: E402
from summer4.flows.rates import (  # noqa: E402
    ArrayConst,
    BinOp,
    Capture,
    Const,
    FieldRef,
    FlowRef,
    GaussianPulse,
    Interp,
    RateOps,
    Reduce,
    derived_refs,
)
from summer4.flows.stages import build_hoist_table, rate_stage  # noqa: E402
from summer4.timevarying import gaussian_pulse, linear  # noqa: E402


def test_rate_stage_classification_table() -> None:
    assert rate_stage(Const(1.0), params_are_static=True) == "run"
    assert rate_stage(ArrayConst(np.array([1.0])), params_are_static=True) == "run"
    assert rate_stage(FieldRef(("a",)), params_are_static=True) == "run"
    assert rate_stage(FieldRef(("a",)), params_are_static=False) == "step"
    assert rate_stage(Time(), params_are_static=True) == "step"
    assert rate_stage(FlowRef("f"), params_are_static=True) == "step"
    assert rate_stage(Reduce(sum_over="age"), params_are_static=True) == "step"
    assert rate_stage(Capture("c", Const(1.0)), params_are_static=True) == "step"

    run_binop = BinOp("mul", Const(2.0), FieldRef(("a",)))
    assert rate_stage(run_binop, params_are_static=True) == "run"
    assert rate_stage(run_binop, params_are_static=False) == "step"
    step_binop = BinOp("add", Const(1.0), Time())
    assert rate_stage(step_binop, params_are_static=True) == "step"

    interp_run = Interp(
        "linear",
        (Const(0.0), Const(1.0)),
        (FieldRef(("y0",)), FieldRef(("y1",))),
        Const(0.5),
    )
    assert rate_stage(interp_run, params_are_static=True) == "run"
    interp_step = linear(Time(), (0.0, 1.0), (0.0, 1.0))
    assert rate_stage(interp_step, params_are_static=True) == "step"

    pulse = GaussianPulse(Time(), Const(1.0), Const(1.0), FieldRef(("h",)))
    assert rate_stage(pulse, params_are_static=True) == "step"
    pulse_run = GaussianPulse(Const(0.0), FieldRef(("c",)), Const(1.0), Const(1.0))
    assert rate_stage(pulse_run, params_are_static=True) == "run"

    class _Custom(RateOps):
        pass

    assert rate_stage(_Custom(), params_are_static=True) == "step"

    class _Staged(RateOps):
        def __rate_stage__(self) -> str:
            return "run"

    assert rate_stage(_Staged(), params_are_static=True) == "run"


def _equiv_model() -> tuple[FlowModel, PropertyMap, PropertyData, Any]:
    class P(NamedTuple):
        a: float
        b: float
        t0: float
        t1: float
        y0: float
        y1: float
        centre: float

    refs = derived_refs(P)
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    rate = refs.a * refs.b + 1.0
    interp = linear(Time(), (refs.t0, refs.t1, 10.0), (refs.y0, refs.y1, 0.0))
    pulse = gaussian_pulse(Time(), centre=refs.centre, width=1.0, height=1.0)
    model.add_flow(
        EntryFlow(
            "in",
            state["Y"],
            rate + interp + pulse,
            adjust=(
                Multiply(refs.a),
                Transform(lambda prev, x: prev + x, refs.b),
            ),
        )
    )
    y0 = PropertyData.wrap(pmap, np.array([0.0]))
    params = P(a=2.0, b=3.0, t0=0.0, t1=4.0, y0=1.0, y1=0.5, centre=2.0)
    return model, pmap, y0, params


def _final_y(cm: Any, params: object, y0: Any, *, solver: str) -> np.ndarray:
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([4.0]))
    res = cm.run(params, y0, t0=0.0, t1=4.0, dt=0.25, save=plan, solver=solver)
    return np.asarray(res["y"].values.data)


@pytest.mark.parametrize("solver,rtol", [("euler", 1e-12), ("tsit5", 1e-9)])
def test_hoist_equivalence(solver: str, rtol: float) -> None:
    model, _pmap, y0, params = _equiv_model()
    cm_on = model.compile(hoist=True)
    cm_off = model.compile(hoist=False)
    np.testing.assert_allclose(
        _final_y(cm_on, params, y0, solver=solver),
        _final_y(cm_off, params, y0, solver=solver),
        rtol=rtol,
    )

    def loss(cm: Any, p: Any) -> Any:
        plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([4.0]))
        res = cm.run(p, y0, t0=0.0, t1=4.0, dt=0.25, save=plan, solver="euler")
        return jnp.sum(jnp.asarray(res["y"].values.data))

    g_on = jax.grad(lambda p: loss(cm_on, p))(params)
    g_off = jax.grad(lambda p: loss(cm_off, p))(params)
    for a, b in zip(jax.tree_util.tree_leaves(g_on), jax.tree_util.tree_leaves(g_off), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5)


def test_knot_count_does_not_grow_loop_body() -> None:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)

    def make(k: int, *, hoist: bool) -> tuple[Any, Any]:
        KnotsNT = NamedTuple("KnotsNT", [(f"v{i}", float) for i in range(k)])  # noqa: N806
        refs = derived_refs(KnotsNT)
        bps = tuple(float(i) for i in range(k))
        vals = tuple(getattr(refs, f"v{i}") for i in range(k))
        expr = linear(Time(), bps, vals)
        model = FlowModel(pmap)
        model.add_flow(EntryFlow("in", state["Y"], expr))
        params = KnotsNT(**{f"v{i}": float(i % 3) for i in range(k)})
        return model.compile(hoist=hoist), params

    y0 = PropertyData.wrap(pmap, np.array([0.0]))
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([1.0]))

    def body_count(k: int, *, hoist: bool) -> int:
        cm, params = make(k, hoist=hoist)

        def loss(p: Any) -> Any:
            res = cm.run(p, y0, t0=0.0, t1=1.0, dt=0.25, save=plan, solver="euler")
            return jnp.sum(jnp.asarray(res["y"].values.data))

        closed = jax.make_jaxpr(loss)(params)
        return int(sum(loop_body_primitives(closed).values()))

    on4 = body_count(4, hoist=True)
    on40 = body_count(40, hoist=True)
    assert on4 == on40

    off4 = body_count(4, hoist=False)
    off40 = body_count(40, hoist=False)
    assert off40 > off4


def test_epi_unaffected_by_hoist() -> None:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age)
    K = np.eye(2)
    infectious = state["I"]
    mixing = MixingMatrix(age, K, check_reciprocal=False)

    def build(*, hoist: bool) -> Any:
        m = FlowModel(pmap)
        m.add_flow(
            TransitionFlow(
                "infection",
                state["S"],
                state["I"],
                ForceOfInfection(
                    "infection",
                    infectious=infectious,
                    group_by=mixing.prop,
                    mixing=mixing,
                    kind="frequency",
                    contact_rate=Param("beta"),
                ),
            )
        )
        m.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.1))
        return m.compile(hoist=hoist)

    cm_on = build(hoist=True)
    cm_off = build(hoist=False)
    data = np.zeros(pmap.size)
    y0 = PropertyData.wrap(pmap, data)
    y0 = y0.at[state["S"]].set(990.0)
    y0 = y0.at[state["I"]].set(10.0)
    params = {"beta": 0.3}
    plan = SavePlan(requests={"y": SaveRequest(Compartments())}, ts=np.array([5.0]))
    r_on = cm_on.run(params, y0, t0=0.0, t1=5.0, dt=0.5, save=plan, solver="euler")
    r_off = cm_off.run(params, y0, t0=0.0, t1=5.0, save=plan, dt=0.5, solver="euler")
    np.testing.assert_allclose(
        np.asarray(r_on["y"].values.data),
        np.asarray(r_off["y"].values.data),
        rtol=1e-12,
    )


def test_hoist_digest_differs() -> None:
    state = Property("state", ("Y",))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(EntryFlow("in", state["Y"], Param("a") * Param("b") + 1.0))
    assert model.compile(hoist=True) != model.compile(hoist=False)


def test_build_hoist_table_slots_binop() -> None:
    expr = Param("a") * Param("b") + 1.0
    table = build_hoist_table([expr], params_are_static=True)
    assert len(table.entries) == 1
    assert table.entries[0].part == "value"
