"""Phase 4 — flow outputs, polarity queries, incidence, computed values."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import (
    ComputedValue,
    Dest,
    Everything,
    ExitFlow,
    FlowMass,
    FlowModel,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    Source,
    TraitChain,
    TransitionFlow,
)
from summer4.flows.edges import edge_labels, rewrite_edge_selector, sum_over_edge
from summer4.jax.propertydata import PropertyData as PD
from summer4.results.trace import Trace
from summer4.time import TimeAxis


def _sir_age() -> tuple[Property, Property, PropertyMap]:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("0-4", "5-9", "10+"))
    return state, age, PropertyMap.from_property(state).stratify(age)


def _ageing_model() -> tuple[Property, Property, object, PropertyData]:
    state, age, pmap = _sir_age()
    model = FlowModel(pmap)
    model.add_flow(
        TransitionFlow(
            "ageing",
            age.present(),
            age.present(),
            0.2,
            pairing=TraitChain(age, (("0-4", "5-9"), ("5-9", "10+"))),
        )
    )
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.1))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.ones(pmap.size))
    return state, age, cm, y0


def test_sum_over_source_vs_dest_ageing() -> None:
    """Headline gate: source and dest aggregations differ on a TraitChain flow."""
    _state, age, cm, y0 = _ageing_model()
    plan = SavePlan(requests={"ageing": SaveRequest(FlowMass(flow="ageing"))})
    res = cm.run({}, y0, t0=0.0, steps=1, dt=1.0, save=plan)
    trace = res["ageing"]
    by_src = trace.sum_over(age, side="source")
    by_dest = trace.sum_over(age, side="dest")
    src = np.asarray(by_src.values.data)
    dest = np.asarray(by_dest.values.data)
    assert src.shape[-1] == 3
    assert dest.shape[-1] == 3
    # Edges leave 0-4 and 5-9; none leave 10+.
    assert float(src[0, 2]) == 0.0
    # Edges enter 5-9 and 10+; none enter 0-4.
    assert float(dest[0, 0]) == 0.0
    assert not np.allclose(src, dest)

    # Hand expectation at t0: mass = rate * scale * y_src = 0.2 * 1 * 1 per edge.
    # Three states × two chain steps = 6 edges; three leave age 0-4, three leave 5-9.
    np.testing.assert_allclose(src[0], [3 * 0.2, 3 * 0.2, 0.0])
    np.testing.assert_allclose(dest[0], [0.0, 3 * 0.2, 3 * 0.2])


def test_sum_over_requires_side_on_edge() -> None:
    _state, age, cm, y0 = _ageing_model()
    plan = SavePlan(requests={"ageing": SaveRequest(FlowMass(flow="ageing"))})
    res = cm.run({}, y0, t0=0.0, steps=1, dt=1.0, save=plan)
    with pytest.raises(ValueError, match="side='source' or side='dest'"):
        res["ageing"].sum_over(age)


def test_select_source_and_dest() -> None:
    state, age, cm, y0 = _ageing_model()
    plan = SavePlan(requests={"ageing": SaveRequest(FlowMass(flow="ageing"))})
    res = cm.run({}, y0, t0=0.0, steps=1, dt=1.0, save=plan)
    selected = res["ageing"].select(Source(age["0-4"]) & Dest(~state["R"]))
    # Three states leave 0-4; Dest(~R) drops the R→R ageing edge → 2 edges.
    assert selected.values.pmap.size == 2


def test_dest_everything_on_exit_is_unknown() -> None:
    state, _age, pmap = _sir_age()
    model = FlowModel(pmap)
    model.add_flow(ExitFlow("death", state["I"], 0.05))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.1))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.ones(pmap.size))
    plan = SavePlan(
        requests={
            "death": SaveRequest(FlowMass(flow="death")),
            "recovery": SaveRequest(FlowMass(flow="recovery")),
        }
    )
    res = cm.run({}, y0, t0=0.0, steps=1, dt=1.0, save=plan)
    # Exit flow: Dest(Everything()) is Kleene-unknown → selects nothing.
    assert res["death"].select(Dest(Everything())).values.pmap.size == 0
    # Same request on a transition flow selects all edges.
    assert (
        res["recovery"].select(Dest(Everything())).values.pmap.size
        == res["recovery"].values.pmap.size
    )


def test_select_moves_mask() -> None:
    _state, age, cm, y0 = _ageing_model()
    plan = SavePlan(requests={"ageing": SaveRequest(FlowMass(flow="ageing"))})
    res = cm.run({}, y0, t0=0.0, steps=1, dt=1.0, save=plan)
    selected = res["ageing"].select(cm.edges("ageing").moves_mask(age))
    assert selected.values.pmap.size == res["ageing"].values.pmap.size
    assert selected.values.pmap.size == 6


def test_select_source_dest_present() -> None:
    _state, age, cm, y0 = _ageing_model()
    plan = SavePlan(requests={"ageing": SaveRequest(FlowMass(flow="ageing"))})
    res = cm.run({}, y0, t0=0.0, steps=1, dt=1.0, save=plan)
    selected = res["ageing"].select(Source(age.present()) & Dest(age.present()))
    assert selected.values.pmap.size == res["ageing"].values.pmap.size


def test_incidence_trapezoid_second_order() -> None:
    """Halving dt should roughly quarter the trapezoid error vs analytic integral."""
    times_coarse = np.linspace(0.0, 1.0, 5)
    times_fine = np.linspace(0.0, 1.0, 9)
    # Rate f(t) = t^2 → ∫_0^1 t^2 dt = 1/3 (linear f is exact under trapezoid).
    analytic = 1.0 / 3.0

    def _integral(ts: np.ndarray) -> float:
        vals = ts**2
        trace = Trace(
            times=TimeAxis(values=ts, epoch=None, kind="explicit"),
            values=vals,
            dims=("time",),
        )
        return float(np.asarray(trace.integrate(method="trapezoid").values))

    err_c = abs(_integral(times_coarse) - analytic)
    err_f = abs(_integral(times_fine) - analytic)
    assert err_f < err_c
    assert err_c / err_f == pytest.approx(4.0, rel=0.15)


def test_simpson_beats_trapezoid_and_raises() -> None:
    ts = np.linspace(0.0, 1.0, 5)  # odd count, uniform
    vals = ts**2  # ∫_0^1 t^2 dt = 1/3
    analytic = 1.0 / 3.0
    trace = Trace(
        times=TimeAxis(values=ts, epoch=None, kind="explicit"),
        values=vals,
        dims=("time",),
    )
    trap = float(np.asarray(trace.integrate(method="trapezoid").values))
    simp = float(np.asarray(trace.integrate(method="simpson").values))
    assert abs(simp - analytic) < abs(trap - analytic)

    even = Trace(
        times=TimeAxis(values=np.linspace(0.0, 1.0, 4), epoch=None, kind="explicit"),
        values=np.linspace(0.0, 1.0, 4) ** 2,
        dims=("time",),
    )
    with pytest.raises(ValueError, match="odd number"):
        even.integrate(method="simpson")

    uneven = Trace(
        times=TimeAxis(values=np.array([0.0, 0.2, 0.5, 0.8, 1.0]), epoch=None, kind="explicit"),
        values=np.array([0.0, 0.04, 0.25, 0.64, 1.0]),
        dims=("time",),
    )
    with pytest.raises(ValueError, match="uniform"):
        uneven.integrate(method="simpson")


def test_where_with_sum_over_uses_filtered_mass() -> None:
    _state, age, cm, y0 = _ageing_model()
    plan = SavePlan(
        requests={
            "young": SaveRequest(
                FlowMass(flow="ageing", where=Source(age["0-4"]), sum_over=(age, "source"))
            )
        }
    )
    res = cm.run({}, y0, t0=0.0, steps=1, dt=1.0, save=plan)
    data = np.asarray(res["young"].values.data)
    # Only edges leaving 0-4 contribute → [3*0.2, 0, 0]
    np.testing.assert_allclose(data[0], [3 * 0.2, 0.0, 0.0])

    # Same via post-hoc query
    full = SavePlan(requests={"ageing": SaveRequest(FlowMass(flow="ageing"))})
    res2 = cm.run({}, y0, t0=0.0, steps=1, dt=1.0, save=full)
    post = res2["ageing"].select(Source(age["0-4"])).sum_over(age, side="source")
    np.testing.assert_allclose(np.asarray(post.values.data)[0], data[0])


class _BetaDerived(NamedTuple):
    beta: float


def test_computed_value_validates_before_solve() -> None:
    state, age, pmap = _sir_age()

    def derived_fn(params: object, *, y: object, t: object) -> _BetaDerived:
        del y, t
        assert isinstance(params, dict)
        return _BetaDerived(beta=float(params["beta"]))

    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], 0.3))
    cm = model.compile(derived_fn=derived_fn)
    y0 = PropertyData.wrap(pmap, np.ones(pmap.size))
    bad = SavePlan(requests={"x": SaveRequest(ComputedValue(path=("nope",)))})
    with pytest.raises(ValueError, match="Available paths"):
        cm.run({"beta": 0.3}, y0, t0=0.0, steps=1, dt=1.0, save=bad)

    good = SavePlan(requests={"beta": SaveRequest(ComputedValue(path=("beta",)))})
    res = cm.run({"beta": 0.3}, y0, t0=0.0, steps=2, dt=1.0, save=good)
    np.testing.assert_allclose(np.asarray(res["beta"].values), 0.3)


def test_observe_prunes_flows() -> None:
    _state, _age, cm, y0 = _ageing_model()
    plan = SavePlan(requests={"ageing": SaveRequest(FlowMass(flow="ageing"))})
    ctx = cm.observe(0.0, y0, {}, keep=plan.flow_reads())
    assert set(ctx.flows) == {"ageing"}

    everything = cm.describe(SavePlan())
    sparse = cm.describe(plan)
    assert sparse.total_nbytes < everything.total_nbytes
    assert {o.key for o in sparse.outputs} == {"ageing"}


def test_jit_grad_sum_over_dest_at_times() -> None:
    _state, age, cm, y0 = _ageing_model()
    plan = SavePlan(requests={"ageing": SaveRequest(FlowMass(flow="ageing"))})
    targets = np.array([0.5, 1.0])

    def loss(scale: object) -> object:
        y = y0 * scale
        res = cm.run({}, y, t0=0.0, steps=4, dt=0.25, save=plan)
        return jnp.sum(res["ageing"].sum_over(age, side="dest").at_times(targets).values.data)

    val = jax.jit(loss)(1.0)
    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(val))
    assert np.isfinite(float(g))


def test_incidence_jaxpr_independent_of_t() -> None:
    def _op_count(n: int) -> int:
        ts = np.linspace(0.0, 1.0, n)
        vals = jnp.asarray(ts)

        def run(v: object) -> object:
            trace = Trace(
                times=TimeAxis(values=ts, epoch=None, kind="explicit"),
                values=v,
                dims=("time",),
            )
            return trace.incidence(method="trapezoid").values

        jaxpr = jax.make_jaxpr(run)(vals)
        return len(jaxpr.jaxpr.eqns)

    # Host-side dt; traced body should not grow with T.
    assert _op_count(9) == _op_count(17)


def test_to_frame_edge_labels() -> None:
    _state, _age, cm, y0 = _ageing_model()
    plan = SavePlan(requests={"ageing": SaveRequest(FlowMass(flow="ageing"))})
    res = cm.run({}, y0, t0=0.0, steps=1, dt=1.0, save=plan)
    frame = res["ageing"].to_frame()
    expected = edge_labels(res["ageing"].values.pmap)
    cols = [c for c in frame.columns if c != "time"]
    assert cols == list(expected)
    assert any("->" in c for c in cols)


def test_rewrite_edge_selector_matches_edgemap() -> None:
    _state, age, cm, y0 = _ageing_model()
    emap = cm.edges("ageing")
    sel = Source(age["0-4"]) & Dest(age["5-9"])
    assert np.array_equal(
        emap.table.select(rewrite_edge_selector(emap.table, sel)),
        emap.select(sel),
    )


def test_sum_over_edge_helper() -> None:
    _state, age, cm, y0 = _ageing_model()
    emap = cm.edges("ageing")
    mass = np.ones(emap.n_edges)
    pd = sum_over_edge(mass, emap.table, age, "source")
    assert isinstance(pd, PD)
    np.testing.assert_allclose(np.asarray(pd.data), [3.0, 3.0, 0.0])
