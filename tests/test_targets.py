"""Phase 5 — sparse calibration targets."""

from __future__ import annotations

from datetime import date
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from summer4 import (
    Compartments,
    Epoch,
    FlowModel,
    Output,
    Property,
    PropertyData,
    PropertyMap,
    Result,
    SavePlan,
    SaveRequest,
    Target,
    TargetSet,
    TimeAxis,
    TransitionFlow,
    derived_refs,
)
from summer4.results.groups import group_requests


class _Rates(NamedTuple):
    infection: float
    recovery: float


def _sir_model() -> tuple[Property, PropertyMap, Any, PropertyData]:
    state = Property("state", ("S", "I", "R"))
    pmap = PropertyMap.from_property(state)
    refs = derived_refs(_Rates)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], refs.infection))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], refs.recovery))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([999.0, 1.0, 0.0]))
    return state, pmap, cm, y0


def test_from_series_round_trips_dates() -> None:
    pd = pytest.importorskip("pandas")
    epoch = Epoch(date(2020, 1, 1))
    idx = pd.DatetimeIndex([date(2020, 1, 1), date(2020, 1, 11), date(2020, 2, 1)])
    series = pd.Series([1.0, 2.0, 3.0], index=idx)
    target = Target.from_series("I", series, epoch)
    np.testing.assert_allclose(target.times, np.array([0.0, 10.0, 31.0]))
    np.testing.assert_allclose(target.values, np.array([1.0, 2.0, 3.0]))


def test_contribute_merges_dedupes_idempotent() -> None:
    state, _pmap, _cm, _y0 = _sir_model()
    qty = Compartments(where=state["I"])
    base = SavePlan(requests={"I": SaveRequest(qty, ts=np.array([0.0, 5.0]))})
    t1 = Target(key="I", times=np.array([5.0, 10.0]), values=np.array([1.0, 2.0]))
    t2 = Target(key="I", times=np.array([10.0, 0.0]), values=np.array([2.0, 0.0]))
    once = t1.contribute(base)
    twice = t1.contribute(once)
    np.testing.assert_array_equal(once.requests["I"].ts, twice.requests["I"].ts)
    np.testing.assert_array_equal(once.requests["I"].ts, np.array([0.0, 5.0, 10.0]))
    both = t2.contribute(once)
    np.testing.assert_array_equal(both.requests["I"].ts, np.array([0.0, 5.0, 10.0]))


def test_two_targets_equal_times_one_save_group() -> None:
    state, _pmap, _cm, _y0 = _sir_model()
    ts = np.array([1.0, 3.0, 7.0])
    targets = TargetSet(
        targets=(
            Target(
                key="I",
                times=ts,
                values=np.ones(3),
                quantity=Compartments(where=state["I"]),
            ),
            Target(
                key="R",
                times=ts.copy(),
                values=np.zeros(3),
                quantity=Compartments(where=state["R"]),
            ),
        )
    )
    plan = targets.plan(SavePlan())
    groups = group_requests(plan, default_ts=np.array([0.0]))
    assert len(groups) == 1
    assert set(groups[0].keys) == {"I", "R"}
    np.testing.assert_array_equal(groups[0].ts, ts)


def test_absent_key_without_quantity_raises() -> None:
    target = Target(key="missing", times=np.array([0.0]), values=np.array([1.0]))
    base = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    with pytest.raises(KeyError, match="missing"):
        target.contribute(base)


def test_off_grid_observation_diffrax_and_euler() -> None:
    state, _pmap, cm, y0 = _sir_model()
    params = _Rates(infection=0.3, recovery=0.1)
    # Off the unit grid: 0.5, 2.5, 4.5
    obs_t = np.array([0.5, 2.5, 4.5])
    qty = Compartments(where=state["I"])
    targets = TargetSet(targets=(Target(key="I", times=obs_t, values=np.zeros(3), quantity=qty),))
    sparse = targets.plan(SavePlan())

    truth = cm.run(
        params,
        y0,
        t0=0.0,
        t1=5.0,
        dt=0.1,
        save=sparse,
        solver="dopri5",
        rtol=1e-8,
        atol=1e-10,
    )
    truth_vals = np.asarray(truth["I"].at_times(obs_t).values.data).reshape(-1)

    # Same source: TargetSet drives both backends.
    observed = TargetSet(targets=(Target(key="I", times=obs_t, values=truth_vals, quantity=qty),))
    plan = observed.plan(SavePlan())

    dopri = cm.run(
        params,
        y0,
        t0=0.0,
        t1=5.0,
        dt=0.1,
        save=plan,
        solver="dopri5",
        rtol=1e-8,
        atol=1e-10,
    )
    pred_d = np.asarray(dopri["I"].at_times(obs_t).values.data).reshape(-1)
    np.testing.assert_allclose(pred_d, truth_vals, rtol=1e-6)

    euler = cm.run(params, y0, t0=0.0, steps=5000, dt=0.001, save=plan, solver="euler")
    pred_e = np.asarray(euler["I"].at_times(obs_t).values.data).reshape(-1)
    np.testing.assert_allclose(pred_e, truth_vals, rtol=1e-3, atol=1e-3)


def test_describe_sparse_smaller_than_dense() -> None:
    state = Property("state", ("S", "I", "R"))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], 0.3))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], 0.1))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([999.0, 1.0, 0.0]))
    obs_t = np.linspace(0.0, 100.0, 40)
    qty = Compartments(where=state["I"])
    targets = TargetSet(targets=(Target(key="I", times=obs_t, values=np.ones(40), quantity=qty),))
    sparse = targets.plan(SavePlan())
    dense = SavePlan(requests={"I": SaveRequest(qty)})
    sparse_desc = cm.describe(sparse, y0=y0, t0=0.0, dt=1.0, steps=100)
    dense_desc = cm.describe(dense, y0=y0, t0=0.0, dt=1.0, steps=100)
    assert sparse_desc.total_nbytes < dense_desc.total_nbytes
    assert sparse_desc.outputs[0].shape[0] == 40
    assert dense_desc.outputs[0].shape[0] == 101


def test_sparse_and_dense_agree_at_observation_times() -> None:
    state, _pmap, cm, y0 = _sir_model()
    params = _Rates(infection=0.25, recovery=0.1)
    obs_t = np.array([0.0, 10.0, 25.0, 50.0, 80.0, 100.0])
    qty = Compartments(where=state["I"])
    targets = TargetSet(targets=(Target(key="I", times=obs_t, values=np.zeros(6), quantity=qty),))
    sparse = targets.plan(SavePlan())
    dense = SavePlan(requests={"I": SaveRequest(qty)})
    s_res = cm.run(params, y0, t0=0.0, t1=100.0, dt=1.0, save=sparse, solver="dopri5")
    d_res = cm.run(params, y0, t0=0.0, t1=100.0, dt=1.0, save=dense, solver="dopri5")
    s_at = np.asarray(s_res["I"].at_times(obs_t).values.data).reshape(-1)
    d_at = np.asarray(d_res["I"].at_times(obs_t).values.data).reshape(-1)
    np.testing.assert_allclose(s_at, d_at, rtol=1e-5, atol=1e-5)


def test_optax_fit_recovers_known_parameters() -> None:
    """Phase gate: ~40 sparse observations recover infection rate within 2%."""
    state, _pmap, cm, y0 = _sir_model()
    true = _Rates(infection=0.35, recovery=0.1)
    rng = np.random.default_rng(0)
    obs_t = np.sort(rng.uniform(0.0, 80.0, size=40))
    qty = Compartments(where=state["I"])

    truth_plan = TargetSet(
        targets=(Target(key="I", times=obs_t, values=np.zeros(40), quantity=qty),)
    ).plan(SavePlan())
    truth = cm.run(
        true,
        y0,
        t0=0.0,
        t1=80.0,
        dt=0.5,
        save=truth_plan,
        solver="dopri5",
        rtol=1e-7,
        atol=1e-9,
    )
    obs_vals = np.asarray(truth["I"].at_times(obs_t).values.data).reshape(-1)
    targets = TargetSet(targets=(Target(key="I", times=obs_t, values=obs_vals, quantity=qty),))
    plan = targets.plan(SavePlan())

    def loss(infection: Any) -> Any:
        params = _Rates(infection=infection, recovery=true.recovery)
        res = cm.run(
            params,
            y0,
            t0=0.0,
            t1=80.0,
            dt=0.5,
            save=plan,
            solver="dopri5",
            rtol=1e-6,
            atol=1e-8,
        )
        resid = targets.residuals(res)["I"]
        return jnp.sum(jnp.asarray(resid) ** 2)

    opt = optax.adam(0.05)
    infection = jnp.asarray(0.15)
    state_opt = opt.init(infection)
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    update = jax.jit(opt.update)

    for _ in range(80):
        value, grad = value_and_grad(infection)
        updates, state_opt = update(grad, state_opt, infection)
        infection = optax.apply_updates(infection, updates)
        del value

    recovered = float(infection)
    assert abs(recovered - true.infection) / true.infection < 0.02


def test_jaxpr_gather_scales_with_targets_not_trajectory() -> None:
    state, _pmap, cm, y0 = _sir_model()
    params = _Rates(infection=0.2, recovery=0.1)
    qty = Compartments(where=state["I"])

    def make_loss(n_targets: int, t1: float) -> Any:
        times = np.linspace(0.0, t1, n_targets)
        values = np.ones(n_targets)
        targets = TargetSet(targets=(Target(key="I", times=times, values=values, quantity=qty),))
        plan = targets.plan(SavePlan())

        def loss(scale: Any) -> Any:
            p = _Rates(infection=params.infection * scale, recovery=params.recovery)
            res = cm.run(p, y0, t0=0.0, dt=1.0, save=plan, solver="euler", steps=int(t1))
            resid = targets.residuals(res)["I"]
            return jnp.sum(jnp.asarray(resid) ** 2)

        return loss

    short = make_loss(5, 20.0)
    long = make_loss(5, 200.0)
    more_targets = make_loss(20, 20.0)

    # jaxpr size must not grow with trajectory length for the same target count.
    jp_short = jax.make_jaxpr(short)(1.0)
    jp_long = jax.make_jaxpr(long)(1.0)
    jp_more = jax.make_jaxpr(more_targets)(1.0)
    short_len = len(str(jp_short.jaxpr))
    long_len = len(str(jp_long.jaxpr))
    more_len = len(str(jp_more.jaxpr))
    # Trajectory length must not dominate; allow modest constant overhead.
    assert long_len < short_len * 2.5
    # More targets may grow the gather unrolling.
    assert more_len > short_len * 0.9


def test_from_series_rejects_non_datetime_index() -> None:
    pd = pytest.importorskip("pandas")
    epoch = Epoch(date(2020, 1, 1))
    series = pd.Series([1.0, 2.0], index=[0.0, 1.0])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        Target.from_series("I", series, epoch)


def test_gather_and_residuals() -> None:
    state, _pmap, cm, y0 = _sir_model()
    params = _Rates(infection=0.3, recovery=0.1)
    obs_t = np.array([0.0, 5.0, 10.0])
    qty = Compartments(where=state["I"])
    plan = SavePlan(requests={"I": SaveRequest(qty, ts=obs_t)})
    res = cm.run(params, y0, t0=0.0, t1=10.0, dt=1.0, save=plan, solver="dopri5")
    pred = np.asarray(res["I"].values.data).reshape(-1)
    targets = TargetSet(targets=(Target(key="I", times=obs_t, values=pred, quantity=qty),))
    gathered = targets.gather(res)
    assert "I" in gathered
    resid = targets.residuals(res)["I"]
    np.testing.assert_allclose(np.asarray(resid), 0.0, atol=1e-6)


def test_residuals_reduce_sum_and_mean_match_aggregate() -> None:
    """A stratified save can meet a one-column observation."""
    times = np.array([0.0, 1.0, 2.0])
    axis = TimeAxis(values=times, epoch=None, kind="explicit")
    pred = np.array([[1.0, 3.0], [2.0, 4.0], [0.0, 5.0]])
    result = Result(
        times=axis,
        outputs={"I": Output(times=axis, values=pred, dims=("time", "compartment"))},
    )
    summed = Target(key="I", times=times, values=np.array([4.0, 6.0, 5.0]), reduce="sum")
    resid = TargetSet(targets=(summed,)).residuals(result)["I"]
    np.testing.assert_allclose(np.asarray(resid), 0.0, atol=1e-6)

    averaged = Target(key="I", times=times, values=np.array([2.0, 3.0, 2.5]), reduce="mean")
    resid_mean = TargetSet(targets=(averaged,)).residuals(result)["I"]
    np.testing.assert_allclose(np.asarray(resid_mean), 0.0, atol=1e-6)

    bare = Target(key="I", times=times, values=np.array([4.0, 6.0, 5.0]))
    with pytest.raises(ValueError, match="incompatible"):
        TargetSet(targets=(bare,)).residuals(result)


def test_residuals_reduce_callable_and_rejects_unknown() -> None:
    times = np.array([0.0, 1.0])
    axis = TimeAxis(values=times, epoch=None, kind="explicit")
    pred = np.array([[1.0, 10.0], [2.0, 20.0]])
    result = Result(
        times=axis,
        outputs={"I": Output(times=axis, values=pred, dims=("time", "compartment"))},
    )

    def first_column(values: Any) -> Any:
        return values[:, 0]

    target = Target(key="I", times=times, values=np.array([1.0, 2.0]), reduce=first_column)
    resid = TargetSet(targets=(target,)).residuals(result)["I"]
    np.testing.assert_allclose(np.asarray(resid), 0.0, atol=1e-6)
    with pytest.raises(ValueError, match="Target.reduce"):
        Target(key="I", times=times, values=times, reduce="max")


def test_residuals_reduce_is_traceable() -> None:
    times = np.array([0.0, 1.0, 2.0])
    axis = TimeAxis(values=times, epoch=None, kind="explicit")
    pred = np.array([[1.0, 3.0], [2.0, 4.0], [0.0, 5.0]])
    result = Result(
        times=axis,
        outputs={"I": Output(times=axis, values=pred, dims=("time", "compartment"))},
    )
    targets = TargetSet(
        targets=(Target(key="I", times=times, values=np.array([4.0, 6.0, 5.0]), reduce="sum"),)
    )

    def loss(shift: Any) -> Any:
        resid = targets.residuals(result)["I"]
        return jnp.sum(jnp.asarray(resid) ** 2) + shift * 0.0

    assert float(jax.jit(loss)(jnp.array(1.0))) == 0.0
