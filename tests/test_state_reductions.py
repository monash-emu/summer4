"""State reductions via Reduce and PropertyData.keep polarity."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from summer4 import (
    Compartments,
    EntryFlow,
    FlowModel,
    Property,
    PropertyData,
    PropertyMap,
    Reduce,
    SavePlan,
    SaveRequest,
    TransitionFlow,
    derived_refs,
)
from summer4.timevarying import linear


class _BetaParams(NamedTuple):
    beta: float


class _FoiParams(NamedTuple):
    foi: object


def test_keep_equals_where_not_sel() -> None:
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age)
    data = jnp.arange(pmap.size, dtype=jnp.float64) + 1.0
    pd = PropertyData(pmap, data)
    kept = pd.keep(state["I"], 0.0)
    via_where = pd.where(~state["I"], 0.0)
    np.testing.assert_allclose(np.asarray(kept.data), np.asarray(via_where.data))
    wrong = pd.where(state["I"], 0.0)
    assert not np.allclose(np.asarray(kept.data), np.asarray(wrong.data))


def test_reduce_where_matches_keep_sum_over() -> None:
    """Reduce(where=sel) equals pd.keep(sel).sum_over numerically."""
    pytest.importorskip("jax")
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age)
    y = np.arange(pmap.size, dtype=np.float64) + 1.0
    pd = PropertyData.wrap(pmap, y)
    expected = np.asarray(pd.keep(state["I"], 0.0).sum_over(age).data)

    model = FlowModel(pmap)
    model.add_flow(EntryFlow("probe", state["S"], Reduce(sum_over=age, where=state["I"])))
    ctx = model.compile().observe(0.0, y, {})
    mass = np.asarray(ctx.flows["probe"])
    s_idx = pmap.select(state["S"])
    age_codes = pmap.codes[s_idx, pmap.column_index(age)]
    np.testing.assert_allclose(mass, expected[age_codes])


def test_reduce_foi_matches_derived_fn_trajectory() -> None:
    """Headline gate: Reduce-built SIR FOI matches hand-written derived_fn."""
    pytest.importorskip("jax")
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age)
    refs = derived_refs(_BetaParams)

    def build_reduce() -> object:
        m = FlowModel(pmap)
        foi = refs.beta * (Reduce(sum_over=age, where=state["I"]) / Reduce(sum_over=age))
        m.add_flow(TransitionFlow("infection", state["S"], state["I"], foi))
        m.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.1))
        return m.compile()

    def build_derived() -> object:
        m = FlowModel(pmap)

        def derived_fn(params: object, *, y: object, t: object) -> _FoiParams:
            pd_y = PropertyData(pmap, y)
            i_by = pd_y.where(~state["I"], 0.0).sum_over(age).data
            n_by = pd_y.sum_over(age).data
            beta = params["beta"]  # type: ignore[index]
            foi = beta * i_by / n_by
            return _FoiParams(foi=pd_y.broadcast_over(age, foi))

        foi_refs = derived_refs(_FoiParams)
        m.add_flow(TransitionFlow("infection", state["S"], state["I"], foi_refs.foi))
        m.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.1))
        return m.compile(derived_fn=derived_fn)

    y0 = np.array([900.0, 800.0, 50.0, 40.0, 50.0, 60.0])
    params = {"beta": 0.4}
    plan = SavePlan(
        requests={"comp": SaveRequest(Compartments())},
        ts=np.linspace(0.0, 40.0, 41),
    )
    r_reduce = build_reduce().run(params, y0, t0=0.0, t1=40.0, dt=0.1, save=plan, solver="euler")
    r_derived = build_derived().run(params, y0, t0=0.0, t1=40.0, dt=0.1, save=plan, solver="euler")
    np.testing.assert_allclose(
        np.asarray(r_reduce["comp"].values.data),
        np.asarray(r_derived["comp"].values.data),
        rtol=1e-5,
        atol=1e-5,
    )


def test_reduce_jaxpr_one_segment_sum_per_reduce() -> None:
    jax = pytest.importorskip("jax")
    state = Property("state", ("S", "I"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age)
    model = FlowModel(pmap)
    model.add_flow(
        TransitionFlow(
            "infection",
            state["S"],
            state["I"],
            Reduce(sum_over=age, where=state["I"]) / Reduce(sum_over=age),
        )
    )
    cm = model.compile()
    y = np.ones(pmap.size)
    jaxpr = jax.make_jaxpr(cm.vector_field)(0.0, y, {})
    text = str(jaxpr)
    # JAX lowers segment_sum to scatter; require one scatter family op per Reduce.
    n_scatter = text.count("scatter")
    assert n_scatter >= 2, f"expected ≥2 scatter ops for two Reduce nodes, got {n_scatter}"


def test_reduce_ragged_map_skips_absent_property() -> None:
    pytest.importorskip("jax")
    state = Property("state", ("S", "I"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age, where=state["I"])
    assert pmap.size == 3  # S + I-young + I-old
    y = np.array([100.0, 10.0, 30.0])
    pd = PropertyData.wrap(pmap, y)
    expected = np.asarray(pd.sum_over(age).data)

    model = FlowModel(pmap)
    model.add_flow(EntryFlow("probe", state["S"], Reduce(sum_over=age)))
    with pytest.raises(ValueError, match="age"):
        model.compile().observe(0.0, y, {})

    model2 = FlowModel(pmap)
    model2.add_flow(EntryFlow("probe", state["I"], Reduce(sum_over=age)))
    mass = np.asarray(model2.compile().observe(0.0, y, {}).flows["probe"])
    i_idx = pmap.select(state["I"])
    age_codes = pmap.codes[i_idx, pmap.column_index(age)]
    np.testing.assert_allclose(mass, expected[age_codes])
    np.testing.assert_allclose(expected, [10.0, 30.0])


def test_reduce_composes_with_interp() -> None:
    pytest.importorskip("jax")
    from summer4 import Time

    state = Property("state", ("S", "I"))
    age = Property("age", ("young", "old"))
    pmap = PropertyMap.from_property(state).stratify(age)
    model = FlowModel(pmap)
    scale = linear(Time(), [0.0, 10.0], [1.0, 2.0])
    foi = scale * (Reduce(sum_over=age, where=state["I"]) / Reduce(sum_over=age))
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], foi))
    y = np.array([100.0, 200.0, 10.0, 20.0])
    dy0 = np.asarray(model.compile().vector_field(0.0, y, {}))
    dy10 = np.asarray(model.compile().vector_field(10.0, y, {}))
    np.testing.assert_allclose(dy10, 2.0 * dy0)
