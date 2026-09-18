"""Tests for the solver seam: Euler, diffrax, save groups, SolverInfo."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest
from tests.helpers.loss import analytic_sir_model, infected_loss

from summer4 import (
    Compartments,
    PropertyData,
    SavePlan,
    SaveRequest,
)
from summer4.results.groups import group_requests


def _simple_sir() -> tuple[Any, PropertyData]:
    cm, y0, _state = analytic_sir_model()
    return cm, y0


def test_euler_dopri5_agree_on_analytic_sir() -> None:
    cm, y0 = _simple_sir()
    ts = np.array([0.0, 1.0], dtype=np.float64)
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments(), ts=ts)})
    dt = 1e-4
    steps = 10_000
    euler = cm.run({}, y0, t0=0.0, steps=steps, dt=dt, save=plan, solver="euler")
    dopri = cm.run(
        {},
        y0,
        t0=0.0,
        t1=1.0,
        dt=0.1,
        save=plan,
        solver="dopri5",
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(euler["compartments"].values.data)[-1],
        np.asarray(dopri["compartments"].values.data)[-1],
        rtol=1e-4,
        atol=1e-4,
    )


def test_grad_agrees_across_backends() -> None:
    """Phase acceptance: one loss source, both backends, grads agree."""
    loss_euler = infected_loss("euler", dt=1e-4)
    loss_dopri = infected_loss("dopri5", dt=1e-4)
    g_e = jax.grad(loss_euler)(1.0)
    g_d = jax.grad(loss_dopri)(1.0)
    np.testing.assert_allclose(np.asarray(g_e), np.asarray(g_d), rtol=1e-3)


def test_off_grid_save_time_interpolated() -> None:
    cm, y0 = _simple_sir()
    # Analytic S(t) = 999 * exp(-0.3 t) for this linear infection rate.
    t_off = 3.37
    plan = SavePlan(
        requests={"compartments": SaveRequest(Compartments(), ts=np.array([0.0, t_off, 10.0]))}
    )
    res = cm.run(
        {},
        y0,
        t0=0.0,
        t1=10.0,
        dt=1.0,
        save=plan,
        solver="dopri5",
        rtol=1e-8,
        atol=1e-10,
    )
    times = np.asarray(res["compartments"].times.values)
    assert times.tolist() == pytest.approx([0.0, t_off, 10.0])
    s_saved = float(np.asarray(res["compartments"].values.data)[1, 0])
    s_analytic = 999.0 * np.exp(-0.3 * t_off)
    np.testing.assert_allclose(s_saved, s_analytic, rtol=1e-5)


def test_solver_info_populated() -> None:
    cm, y0 = _simple_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run(
        {},
        y0,
        t0=0.0,
        t1=5.0,
        dt=0.1,
        save=plan,
        solver="dopri5",
        rtol=1e-4,
        atol=1e-6,
    )
    assert res.solver is not None
    assert int(res.solver.result_code) == 0
    assert isinstance(res.solver.message, str) and len(res.solver.message) > 0
    accepted = int(res.solver.num_accepted_steps)
    rejected = int(res.solver.num_rejected_steps)
    total = int(res.solver.num_steps)
    assert accepted + rejected == total


def test_jit_run_flattens_traced_solver_stats() -> None:
    cm, y0 = _simple_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})

    @jax.jit
    def run_fn(y: Any) -> Any:
        return cm.run(
            {},
            y,
            t0=0.0,
            steps=20,
            dt=0.5,
            save=plan,
            solver="dopri5",
            rtol=1e-4,
            atol=1e-6,
        )

    result = run_fn(y0)
    leaves, aux = jax.tree_util.tree_flatten(result)
    rebuilt = jax.tree_util.tree_unflatten(aux, leaves)
    assert rebuilt.solver is not None
    # Stats remain tracers / arrays through flatten — not stuck in static aux.
    assert any(hasattr(leaf, "dtype") for leaf in leaves)


def test_equal_ts_arrays_one_group() -> None:
    ts_a = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ts_b = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    assert ts_a is not ts_b
    plan = SavePlan(
        requests={
            "a": SaveRequest(Compartments(), ts=ts_a),
            "b": SaveRequest(Compartments(), ts=ts_b),
        }
    )
    groups = group_requests(plan, default_ts=np.array([0.0, 5.0]))
    assert len(groups) == 1
    assert groups[0].keys == ("a", "b")


def test_per_request_ts_honoured_both_backends() -> None:
    cm, y0 = _simple_sir()
    ts_fine = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    ts_coarse = np.array([0.0, 1.0], dtype=np.float64)
    plan = SavePlan(
        requests={
            "fine": SaveRequest(Compartments(), ts=ts_fine),
            "coarse": SaveRequest(Compartments(), ts=ts_coarse),
        }
    )
    for solver, kwargs in (
        ("euler", {"steps": 100, "dt": 0.01}),
        ("dopri5", {"t1": 1.0, "dt": 0.1, "rtol": 1e-6, "atol": 1e-8}),
    ):
        res = cm.run({}, y0, t0=0.0, save=plan, solver=solver, **kwargs)
        np.testing.assert_allclose(np.asarray(res["fine"].times.values), ts_fine)
        np.testing.assert_allclose(np.asarray(res["coarse"].times.values), ts_coarse)
        assert np.asarray(res["fine"].values.data).shape[0] == 3
        assert np.asarray(res["coarse"].values.data).shape[0] == 2


def test_dense_evaluate_and_raises_without() -> None:
    cm, y0 = _simple_sir()
    dense_plan = SavePlan(
        requests={"compartments": SaveRequest(Compartments())},
        dense=True,
    )
    res = cm.run(
        {},
        y0,
        t0=0.0,
        t1=2.0,
        dt=0.1,
        save=dense_plan,
        solver="dopri5",
        rtol=1e-6,
        atol=1e-8,
        max_steps=512,
    )
    at0 = res.evaluate(0.0)
    np.testing.assert_allclose(
        np.asarray(at0.data),
        np.asarray(res["compartments"].values.data)[0],
        rtol=1e-4,
    )
    plain = SavePlan(requests={"compartments": SaveRequest(Compartments())}, dense=False)
    res_plain = cm.run({}, y0, t0=0.0, steps=5, dt=0.2, save=plain, solver="euler")
    with pytest.raises(ValueError, match="dense"):
        res_plain.evaluate(0.0)


def test_unknown_solver_lists_known_names() -> None:
    cm, y0 = _simple_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    with pytest.raises(ValueError, match="dopri5"):
        cm.run({}, y0, t0=0.0, steps=2, dt=1.0, save=plan, solver="not-a-solver")


def test_diffrax_solver_instance_accepted() -> None:
    diffrax = pytest.importorskip("diffrax")
    cm, y0 = _simple_sir()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())})
    res = cm.run(
        {},
        y0,
        t0=0.0,
        t1=1.0,
        dt=0.1,
        save=plan,
        solver=diffrax.Tsit5(),
        rtol=1e-5,
        atol=1e-7,
    )
    assert res.solver is not None
    assert res.solver.solver == "tsit5"


def test_describe_sizes_mixed_ts() -> None:
    cm, y0 = _simple_sir()
    plan = SavePlan(
        requests={
            "daily": SaveRequest(Compartments(), ts=np.linspace(0, 10, 11)),
            "ends": SaveRequest(Compartments(), ts=np.array([0.0, 10.0])),
        }
    )
    desc = cm.describe(plan, y0=y0, t0=0.0)
    by_key = {o.key: o for o in desc.outputs}
    assert by_key["daily"].shape[0] == 11
    assert by_key["ends"].shape[0] == 2
