"""CP1 (roadmap step 29): the solve takes the caller's diffrax objects."""

from __future__ import annotations

from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import (
    Compartments,
    FlowModel,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    TransitionFlow,
)
from summer4.results.plan import EVERYTHING
from summer4.solvers import (
    Diffrax,
    Euler,
    SolveOutput,
    SolverBackend,
    SolveSpec,
    resolve_solver,
)

PARAMS = {"infection": 0.35, "recovery": 0.1}


def _model() -> tuple[Any, PropertyData]:
    state = Property("state", ("S", "I", "R"))
    model = FlowModel(PropertyMap.from_property(state))
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], Param("infection")))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], Param("recovery")))
    cm = model.compile()
    return cm, PropertyData.wrap(cm.pmap, np.array([999.0, 1.0, 0.0]))


def _values(result: Any, key: str = "I") -> np.ndarray:
    raw = result[key].values
    return np.asarray(getattr(raw, "data", raw))


def _plan(ts: np.ndarray | None = None) -> SavePlan:
    state = Property("state", ("S", "I", "R"))
    return SavePlan(requests={"I": SaveRequest(Compartments(where=state["I"]), ts=ts)})


# --- CX1: the caller's stepsize controller ---------------------------------


def test_diffrax_backend_equals_rtol_sugar() -> None:
    cm, y0 = _model()
    window = {"t0": 0.0, "t1": 60.0, "dt": 0.5, "save": _plan()}
    sugar = cm.run(PARAMS, y0, solver="tsit5", rtol=1e-6, atol=1e-8, **window)
    backend = Diffrax(
        diffrax.Tsit5(), stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8)
    )
    explicit = cm.run(PARAMS, y0, solver=backend, **window)
    np.testing.assert_array_equal(_values(sugar), _values(explicit))
    assert int(sugar.solver.num_steps) == int(explicit.solver.num_steps)
    assert explicit.solver.solver == "tsit5"


def test_callers_controller_options_reach_diffrax() -> None:
    cm, y0 = _model()
    window = {"t0": 0.0, "t1": 60.0, "dt": 0.5, "save": _plan()}
    loose = cm.run(
        PARAMS,
        y0,
        solver=Diffrax(diffrax.Tsit5(), stepsize_controller=diffrax.PIDController(1e-6, 1e-8)),
        **window,
    )
    capped = cm.run(
        PARAMS,
        y0,
        solver=Diffrax(
            diffrax.Tsit5(),
            stepsize_controller=diffrax.PIDController(1e-6, 1e-8, dtmax=0.25),
        ),
        **window,
    )
    # dtmax=0.25 over 60 days forces at least 240 steps.
    assert int(capped.solver.num_steps) >= 240 > int(loose.solver.num_steps)
    np.testing.assert_allclose(_values(capped), _values(loose), rtol=1e-4, atol=1e-3)


def test_default_diffrax_backend_uses_a_constant_step() -> None:
    cm, y0 = _model()
    res = cm.run(PARAMS, y0, t0=0.0, t1=10.0, dt=0.5, save=_plan(), solver=diffrax.Dopri5())
    assert int(res.solver.num_steps) == 20
    assert res.solver.solver == "dopri5"


# --- CX2: adjoint and event -------------------------------------------------


def _peak_loss(backend: Any) -> Any:
    cm, y0 = _model()

    def loss(beta: Any) -> Any:
        res = cm.run(
            {"infection": beta, "recovery": 0.1},
            y0,
            t0=0.0,
            t1=40.0,
            dt=0.5,
            save=_plan(np.linspace(0.0, 40.0, 9)),
            solver=backend,
        )
        return jnp.sum(res["I"].values.data)

    return loss


def test_adjoint_choice_gives_the_same_gradient() -> None:
    controller = diffrax.PIDController(rtol=1e-7, atol=1e-9)
    default = jax.grad(_peak_loss(Diffrax(diffrax.Tsit5(), stepsize_controller=controller)))
    direct = jax.grad(
        _peak_loss(
            Diffrax(
                diffrax.Tsit5(), stepsize_controller=controller, adjoint=diffrax.DirectAdjoint()
            )
        )
    )
    g_default, g_direct = float(default(0.35)), float(direct(0.35))
    assert np.isfinite(g_default) and g_default != 0.0
    np.testing.assert_allclose(g_direct, g_default, rtol=1e-4)


def test_event_stops_the_run_where_it_fires() -> None:
    cm, y0 = _model()

    def infectious_over_300(t: Any, y: Any, args: Any, **kwargs: Any) -> Any:
        return y[1] > 300.0

    backend = Diffrax(diffrax.Tsit5(), event=diffrax.Event(infectious_over_300))
    ts = np.arange(0.0, 81.0, 1.0)
    res = cm.run(PARAMS, y0, t0=0.0, t1=80.0, dt=0.1, save=_plan(ts), solver=backend)

    t_end = float(res.final_time)
    assert 0.0 < t_end < 80.0
    assert bool(res.solver.ok) and bool(res.solver.event)
    assert float(np.asarray(res.final_state.data)[1]) > 300.0
    saved = _values(res).reshape(-1)
    assert np.all(np.isfinite(saved[ts <= t_end]))
    assert np.all(np.isinf(saved[ts > t_end]))


def test_no_event_reports_no_event() -> None:
    cm, y0 = _model()
    res = cm.run(PARAMS, y0, t0=0.0, t1=5.0, dt=0.5, save=_plan(), solver="dopri5")
    assert bool(res.solver.ok) and not bool(res.solver.event)


# --- CX3: public backends, resolve_solver, SolveSpec ------------------------


def test_resolve_solver_sugar() -> None:
    assert isinstance(resolve_solver("euler"), Euler)
    named = resolve_solver("dopri5", rtol=1e-5)
    assert isinstance(named, Diffrax)
    assert isinstance(named.solver, diffrax.Dopri5)
    assert isinstance(named.stepsize_controller, diffrax.PIDController)
    bare = resolve_solver(diffrax.Heun())
    assert isinstance(bare, Diffrax) and bare.stepsize_controller is None
    own = Diffrax("tsit5")
    assert resolve_solver(own) is own
    assert isinstance(own, SolverBackend) and isinstance(Euler(), SolverBackend)


@pytest.mark.parametrize(
    ("solver", "kwargs", "message"),
    [
        ("euler", {"rtol": 1e-6}, "not euler"),
        (Euler(), {"atol": 1e-6}, "set its stepsize_controller"),
        (Diffrax("tsit5"), {"rtol": 1e-6}, "set its stepsize_controller"),
        ("rk4", {}, "Unknown solver"),
        (42, {}, "diffrax solver instance"),
    ],
)
def test_resolve_solver_errors(solver: Any, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_solver(solver, **kwargs)


def test_solve_spec_window() -> None:
    spec = SolveSpec.window(t0=1.0, t1=3.0, dt=0.5)
    assert spec.steps == 4 and spec.end == 3.0
    np.testing.assert_array_equal(spec.default_ts(), [1.0, 1.5, 2.0, 2.5, 3.0])
    assert SolveSpec.window(t0=0.0, steps=10, dt=0.1).end == pytest.approx(1.0)
    with pytest.raises(ValueError, match="exactly one"):
        SolveSpec.window(t0=0.0, t1=1.0, steps=10, dt=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        SolveSpec.window(t0=0.0, dt=0.1)
    with pytest.raises(ValueError, match="dt must be > 0"):
        SolveSpec.window(t0=0.0, t1=1.0, dt=0.0)
    with pytest.raises(ValueError, match="t1 must be >= t0"):
        SolveSpec.window(t0=2.0, t1=1.0, dt=0.1)


def test_run_uses_a_user_written_backend() -> None:
    cm, y0 = _model()
    seen: list[SolveSpec] = []

    class Recording:
        name = "recording"

        def solve(
            self, model: Any, *, y0: object, params: object, spec: SolveSpec, plan: Any
        ) -> SolveOutput:
            seen.append(spec)
            return Euler().solve(model, y0=y0, params=params, spec=spec, plan=plan)

    window = {"t0": 0.0, "t1": 10.0, "dt": 0.5, "save": _plan()}
    mine = cm.run(PARAMS, y0, solver=Recording(), **window)
    euler = cm.run(PARAMS, y0, solver="euler", **window)
    np.testing.assert_array_equal(_values(mine), _values(euler))
    assert seen == [SolveSpec.window(t0=0.0, t1=10.0, dt=0.5)]


# --- CX4: run is a documented composition ----------------------------------


@pytest.mark.parametrize("solver", ["euler", "dopri5"])
def test_run_equals_its_documented_expansion(solver: str) -> None:
    cm, y0 = _model()
    short = cm.run(PARAMS, y0, t0=0.0, t1=20.0, dt=0.5, save=None, solver=solver)

    prepared = cm.prepare(PARAMS)
    plan = cm.expand(EVERYTHING)
    spec = SolveSpec.window(t0=0.0, t1=20.0, dt=0.5, dense=plan.dense)
    backend = resolve_solver(solver)
    out = backend.solve(cm, y0=y0, params=prepared, spec=spec, plan=plan)
    long = cm.assemble_result(plan, out, spec=spec)

    assert list(short.outputs) == list(long.outputs)
    for key in short.outputs:
        np.testing.assert_array_equal(_values(short, key), _values(long, key))
    np.testing.assert_array_equal(short.final_state.data, long.final_state.data)


# --- CX5: continue a run from its end state --------------------------------


def test_euler_run_continues_exactly_from_final_state() -> None:
    cm, y0 = _model()
    whole = cm.run(PARAMS, y0, t0=0.0, t1=80.0, dt=0.5, save=_plan(), solver="euler")
    first = cm.run(PARAMS, y0, t0=0.0, t1=40.0, dt=0.5, save=_plan(), solver="euler")
    second = cm.run(
        PARAMS,
        first.final_state,
        t0=float(first.final_time),
        t1=80.0,
        dt=0.5,
        save=_plan(),
        solver="euler",
    )
    assert float(first.final_time) == 40.0 and float(second.final_time) == 80.0
    np.testing.assert_allclose(second.final_state.data, whole.final_state.data, rtol=1e-6)
    np.testing.assert_allclose(_values(second), _values(whole)[80:], rtol=1e-6)


def test_diffrax_run_continues_from_final_state() -> None:
    cm, y0 = _model()
    kw = {"dt": 0.5, "save": _plan(), "solver": "dopri5", "rtol": 1e-8, "atol": 1e-10}
    whole = cm.run(PARAMS, y0, t0=0.0, t1=80.0, **kw)
    first = cm.run(PARAMS, y0, t0=0.0, t1=40.0, **kw)
    second = cm.run(PARAMS, first.final_state, t0=float(first.final_time), t1=80.0, **kw)
    assert float(first.final_time) == pytest.approx(40.0)
    np.testing.assert_allclose(second.final_state.data, whole.final_state.data, rtol=1e-5)


@pytest.mark.parametrize(
    "ts",
    [np.array([0.0, 10.0, 20.0]), np.array([0.0, 2.5, 7.25])],
    ids=["scan-subgrid", "interpolated"],
)
def test_euler_final_state_is_at_the_end_even_when_saves_stop_early(ts: np.ndarray) -> None:
    cm, y0 = _model()
    early = cm.run(PARAMS, y0, t0=0.0, t1=60.0, dt=0.5, save=_plan(ts), solver="euler")
    full = cm.run(PARAMS, y0, t0=0.0, t1=60.0, dt=0.5, save=_plan(), solver="euler")
    assert float(early.final_time) == 60.0
    np.testing.assert_allclose(early.final_state.data, full.final_state.data, rtol=1e-6)


def test_final_state_survives_jit() -> None:
    cm, y0 = _model()

    @jax.jit
    def run(beta: Any) -> Any:
        return cm.run(
            {"infection": beta, "recovery": 0.1},
            y0,
            t0=0.0,
            t1=10.0,
            dt=0.5,
            save=_plan(),
            solver="dopri5",
            rtol=1e-6,
            atol=1e-8,
        )

    res = run(0.35)
    assert isinstance(res.final_state, PropertyData)
    assert res.final_state.pmap == cm.pmap
    assert float(res.final_time) == pytest.approx(10.0)
    total = float(jnp.sum(res.final_state.data))
    assert total == pytest.approx(1000.0, rel=1e-5)
