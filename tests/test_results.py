"""Tests for summer4.results and CompiledModel.run."""

from __future__ import annotations

from datetime import date
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import (
    EVERYTHING,
    Compartments,
    Epoch,
    FlowMass,
    FlowModel,
    Property,
    PropertyData,
    PropertyMap,
    SaveFn,
    SavePlan,
    SaveRequest,
    State,
    TransitionFlow,
    derived_refs,
)
from summer4.jax.propertydata import PropertyData as PD
from summer4.results.plan import PlanDescription


def _sir_age() -> tuple[Property, Property, PropertyMap]:
    state = Property("state", ("S", "I", "R"))
    age = Property("age", ("0-4", "5-9", "10+"))
    return state, age, PropertyMap.from_property(state).stratify(age)


def _compiled_sir() -> tuple[Property, Property, PropertyMap, object, PD]:
    state, age, pmap = _sir_age()
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], 0.2))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.1))
    cm = model.compile()
    y0 = np.zeros(pmap.size)
    y0[pmap.select(state["S"])] = 999.0
    y0[pmap.select(state["I"])] = 1.0
    return state, age, pmap, cm, PropertyData.wrap(pmap, y0)


def test_describe_shapes_without_solve() -> None:
    _state, _age, _pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    desc = cm.describe(plan, y0=y0, n_saves=11)
    assert isinstance(desc, PlanDescription)
    assert desc.n_saves == 11
    assert desc.outputs[0].shape == (11, 9)


class _DescribeRates(NamedTuple):
    infection: float
    recovery: float


def test_describe_with_params_indexing_derived_fn() -> None:
    """describe(params=...) sizes a plan when derived_fn indexes params."""
    state, _age, pmap = _sir_age()

    def derived_fn(params: object, *, y: object, t: object) -> _DescribeRates:
        del y, t
        assert isinstance(params, dict)
        return _DescribeRates(
            infection=params["infection"],
            recovery=params["recovery"],
        )

    refs = derived_refs(_DescribeRates)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], refs.infection))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], refs.recovery))
    cm = model.compile(derived_fn=derived_fn)
    y0 = np.zeros(pmap.size)
    y0[pmap.select(state["S"])] = 999.0
    y0[pmap.select(state["I"])] = 1.0
    y0_pd = PropertyData.wrap(pmap, y0)
    params = {"infection": 0.2, "recovery": 0.1}
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})

    desc = cm.describe(plan, params=params, y0=y0_pd, n_saves=11)
    assert isinstance(desc, PlanDescription)
    assert desc.n_saves == 11
    assert desc.outputs[0].shape == (11, pmap.size)

    res = cm.run(params, y0_pd, t0=0.0, steps=10, dt=1.0, save=plan)
    actual = int(np.asarray(res["compartments"].values.data).nbytes)
    assert desc.total_nbytes == actual


def test_describe_without_params_still_works_without_derived_fn() -> None:
    _state, _age, _pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    desc = cm.describe(plan, y0=y0, n_saves=5)
    assert desc.outputs[0].shape == (5, 9)


def test_euler_fast_path_matches_final_euler() -> None:
    from summer4.flows.compiled import euler

    _state, _age, _pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run({}, y0, t0=0.0, steps=10, dt=1.0, save=plan)
    final_run = np.asarray(res["compartments"].values.data)[-1]
    final_euler = np.asarray(euler(cm.vector_field, 0.0, y0, {}, dt=1.0, steps=10).data)
    np.testing.assert_allclose(final_run, final_euler, rtol=1e-5)


def test_query_select_sum_between_dates() -> None:
    state, age, pmap, cm, y0 = _compiled_sir()
    epoch = Epoch(date(2020, 1, 1))
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run({}, y0, t0=0.0, steps=30, dt=1.0, save=plan, epoch=epoch)
    selected = res["compartments"].select(state["I"])
    assert selected.values.pmap.size == len(age.traits)
    assert selected.values.pmap.size < pmap.size
    infected = selected.sum_over(age)
    windowed = infected.between(date(2020, 1, 5), date(2020, 1, 15))
    assert np.asarray(windowed.times.values).size == 11
    total_i = selected.total()
    assert float(np.asarray(total_i.at(date(2020, 1, 10)).values)) > 0


def test_select_gathers_not_masks() -> None:
    """select shrinks the map; to_pandas columns match selected compartments only."""
    pytest.importorskip("pandas")
    state, _age, pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run({}, y0, t0=0.0, steps=5, dt=1.0, save=plan)
    full = res["compartments"]
    selected = full.select(state["I"])
    idx = pmap.select(state["I"])
    assert selected.values.pmap.size == idx.size
    np.testing.assert_allclose(
        np.asarray(selected.values.data),
        np.asarray(full.values.data)[..., idx],
        rtol=1e-5,
    )
    pdf = selected.to_pandas()
    assert list(pdf.columns) == list(selected.values.pmap.labels())
    assert all("state=I" in c for c in pdf.columns)
    assert not any("state=S" in c or "state=R" in c for c in pdf.columns)


def test_select_then_partition_is_restricted() -> None:
    """partition after select must only see rows on the sub-map."""
    state, age, pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run({}, y0, t0=0.0, steps=3, dt=1.0, save=plan)
    selected = res["compartments"].select(state["I"])
    assert selected.values.pmap.size == len(age.traits)
    parts = selected.partition(age)
    assert set(parts) == {age[t] for t in age.traits}
    for _trait, tr in parts.items():
        # One column per age band on the I-only sub-map (not S/I/R × age).
        assert np.asarray(tr.values).shape[-1] == 1
    # Full-map partition still has three disease states per age — the old mask
    # bug would have used those wider index arrays after select.
    for _trait, idx in pmap.partition(age).items():
        assert idx.size == len(state.traits)
    for _trait, idx in selected.values.pmap.partition(age).items():
        assert idx.size == 1
        assert "state=I" in selected.values.pmap.labels()[int(idx[0])]


def test_compartments_where_in_saveplan_stays_map_aware() -> None:
    state, age, pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(
        requests={
            "infected": SaveRequest(Compartments(where=state["I"])),
            "infected_by_age": SaveRequest(Compartments(where=state["I"], sum_over=age)),
        }
    )
    res = cm.run({}, y0, t0=0.0, steps=4, dt=1.0, save=plan)
    infected = res["infected"]
    assert isinstance(infected.values, PropertyData)
    assert infected.values.pmap.size == len(age.traits)
    assert infected.values.pmap.size < pmap.size
    # Still queryable after a where=-restricted save.
    narrowed = infected.select(age["0-4"])
    assert narrowed.values.pmap.size == 1
    by_age = infected.sum_over(age)
    assert by_age.dims[-1] == "group"
    assert np.asarray(res["infected_by_age"].values.data).shape[-1] == len(age.traits)


def test_jit_select_gather_shrinks_under_jit() -> None:
    """Trace.select gather must stay usable as the default jax.jit target path."""
    state, age, pmap, cm, y0 = _compiled_sir()

    def loss(scale: object) -> Any:
        y = PropertyData(pmap, y0.data * scale)
        plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
        res = cm.run({}, y, t0=0.0, steps=5, dt=1.0, save=plan)
        selected = res["compartments"].select(state["I"])
        # Shape is concrete under jit (static sub-map size).
        assert selected.values.pmap.size == len(age.traits)
        pred = selected.total().at_times(np.array([5.0]))
        return jnp.sum(jnp.asarray(pred.values) ** 2)

    jitted = jax.jit(loss)
    val = float(jitted(1.0))
    g = float(jax.grad(loss)(1.0))
    assert np.isfinite(val) and np.isfinite(g)
    # Eager and jitted agree.
    np.testing.assert_allclose(val, float(loss(1.0)), rtol=1e-5)


def test_jit_saveplan_where_gather() -> None:
    """Compartments(where=) save path gathers under jax.jit, not mask-zero."""
    state, age, pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(requests={"infected": SaveRequest(Compartments(where=state["I"]))})

    def run(scale: object) -> Any:
        y = PropertyData(pmap, y0.data * scale)
        res = cm.run({}, y, t0=0.0, steps=4, dt=1.0, save=plan)
        data = res["infected"].values.data
        assert res["infected"].values.pmap.size == len(age.traits)
        return jnp.sum(data)

    jitted = jax.jit(run)
    out = jitted(1.0)
    assert np.isfinite(float(out))
    # describe reports the shrunk compartment axis
    desc = cm.describe(plan, y0=y0, n_saves=5)
    assert desc.outputs[0].shape == (5, len(age.traits))


def test_resample_me_matches_host_reference() -> None:
    state, _age, _pmap, cm, y0 = _compiled_sir()
    epoch = Epoch(date(2020, 1, 1))
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run({}, y0, t0=0.0, steps=59, dt=1.0, save=plan, epoch=epoch)
    total = res["compartments"].select(state["I"]).total()
    monthly = total.resample("ME", how="sum")

    # Host reference with stdlib datetime only (not pandas).
    values = np.asarray(total.values)
    times = np.asarray(total.times.values)
    dates = epoch.from_model(times).astype("datetime64[D]")
    keys = [(str(d)[:7]) for d in dates]  # YYYY-MM
    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    expected = np.array([values[ix].sum() for ix in groups.values()])
    np.testing.assert_allclose(np.asarray(monthly.values), expected, rtol=1e-5)


def test_rolling_mean_matches_numpy() -> None:
    state, _age, _pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run({}, y0, t0=0.0, steps=20, dt=1.0, save=plan)
    total = res["compartments"].select(state["I"]).total()
    rolled = total.rolling(7, how="mean", min_periods=1)
    raw = np.asarray(total.values, dtype=np.float64)
    expected = np.empty_like(raw)
    for i in range(raw.size):
        lo = max(0, i - 6)
        expected[i] = raw[lo : i + 1].mean()
    np.testing.assert_allclose(np.asarray(rolled.values), expected, rtol=1e-5)


def test_cumulative_matches_cumsum() -> None:
    state, _age, _pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run({}, y0, t0=0.0, steps=10, dt=1.0, save=plan)
    total = res["compartments"].select(state["I"]).total()
    np.testing.assert_allclose(
        np.asarray(total.cumulative().values),
        np.cumsum(np.asarray(total.values)),
        rtol=1e-5,
    )


def test_at_times_interpolates_and_on_grid_noop() -> None:
    state, _age, _pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run({}, y0, t0=0.0, steps=10, dt=1.0, save=plan)
    total = res["compartments"].select(state["I"]).total()
    on_grid = total.at_times(np.array([0.0, 5.0, 10.0]))
    np.testing.assert_allclose(
        np.asarray(on_grid.values),
        np.asarray(total.values)[[0, 5, 10]],
        rtol=1e-5,
    )
    mid = total.at_times(np.array([0.5]))
    expected = 0.5 * float(np.asarray(total.values).reshape(-1)[0]) + 0.5 * float(
        np.asarray(total.values).reshape(-1)[1]
    )
    np.testing.assert_allclose(float(np.asarray(mid.values).reshape(-1)[0]), expected, rtol=1e-5)


def test_jit_grad_loss_ending_in_at_times() -> None:
    state, _age, pmap, cm, y0 = _compiled_sir()

    class Params(NamedTuple):
        scale: float

    def loss(scale: object) -> Any:
        # Scale initial I
        data = y0.data * 0 + y0.data
        data = data.at[pmap.select(state["I"])].set(
            y0.data[pmap.select(state["I"])] * scale  # type: ignore[operator]
        )
        y = PropertyData(pmap, data)
        plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
        res = cm.run({}, y, t0=0.0, steps=5, dt=1.0, save=plan)
        pred = res["compartments"].select(state["I"]).total().at_times(np.array([5.0]))
        return jnp.sum(jnp.asarray(pred.values) ** 2)

    jitted = jax.jit(loss)
    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(jitted(1.0)))
    assert np.isfinite(float(g))


def test_state_ledgers_round_trip_jit() -> None:
    _state, _age, pmap, cm, y0 = _compiled_sir()
    st = State(compartments=y0, ledgers={})

    def step(s: State) -> State:
        return cm.vector_field(0.0, s, {})  # type: ignore[return-value]

    out = jax.jit(step)(st)
    assert isinstance(out, State)
    assert out.ledgers == {}


def test_saveplan_hash_differs_on_ts() -> None:
    a = SavePlan(requests={"c": SaveRequest(Compartments())}, ts=np.array([0.0, 1.0]))
    b = SavePlan(requests={"c": SaveRequest(Compartments())}, ts=np.array([0.0, 2.0]))
    assert hash(a) != hash(b)


def test_flow_mass_in_plan() -> None:
    _state, _age, _pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(
        requests={
            "infection": SaveRequest(FlowMass(flow="infection")),
        }
    )
    res = cm.run({}, y0, t0=0.0, steps=5, dt=1.0, save=plan)
    assert res["infection"].dims == ("time", "edge")
    assert isinstance(res["infection"].values, PropertyData)
    assert np.asarray(res["infection"].values.data).ndim == 2


def test_everything_expands() -> None:
    _state, _age, _pmap, cm, y0 = _compiled_sir()
    expanded = cm.expand(EVERYTHING)
    assert "compartments" in expanded.requests
    assert "infection" in expanded.requests
    assert "recovery" in expanded.requests


def test_result_pytree_no_params() -> None:
    _state, _age, _pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run({}, y0, t0=0.0, steps=3, dt=1.0, save=plan)
    leaves, _ = jax.tree_util.tree_flatten(res)
    assert all(not isinstance(x, dict) for x in leaves)
    assert not hasattr(res, "params")
    assert not hasattr(res, "model")


def test_to_frame_polars() -> None:
    _state, _age, _pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run({}, y0, t0=0.0, steps=3, dt=1.0, save=plan)
    frame = res["compartments"].to_frame()
    assert "time" in frame.columns


def test_to_pandas_optional() -> None:
    pytest.importorskip("pandas")
    _state, _age, _pmap, cm, y0 = _compiled_sir()
    epoch = Epoch(date(2020, 1, 1))
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run({}, y0, t0=0.0, steps=3, dt=1.0, save=plan, epoch=epoch)
    pdf = res["compartments"].to_pandas()
    assert pdf.index.name == "time"


def test_savefn_escape_hatch() -> None:
    _state, _age, _pmap, cm, y0 = _compiled_sir()

    def fn(ctx: object) -> object:
        import jax.numpy as jnp

        y = ctx.y  # type: ignore[attr-defined]
        data = y.data if isinstance(y, PropertyData) else y
        return jnp.sum(data)

    plan = SavePlan(requests={"pop": SaveRequest(SaveFn(fn=fn))})
    res = cm.run({}, y0, t0=0.0, steps=3, dt=1.0, save=plan)
    assert res["pop"].dims == ("time",)
    np.testing.assert_allclose(np.asarray(res["pop"].values), 3000.0, rtol=1e-4)


def test_axis_rule_under_vmap_dims() -> None:
    """vmap over draws keeps time at -2."""
    state, _age, pmap, cm, y0 = _compiled_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})

    def run_one(scale: object) -> object:
        data = y0.data * scale
        y = PropertyData(pmap, data)
        res = cm.run({}, y, t0=0.0, steps=4, dt=1.0, save=plan)
        return res["compartments"].values.data

    scales = jnp.array([1.0, 1.1, 0.9])
    stacked = jax.vmap(run_one)(scales)
    # (draw, time, compartment)
    assert tuple(stacked.shape) == (3, 5, pmap.size)
