"""The default solver: adaptive Tsit5, overridable per process and per call (plan step R2)."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from summer4 import (
    Compartments,
    EntryFlow,
    ExitFlow,
    FlowModel,
    FlowRef,
    Param,
    Property,
    PropertyMap,
    Reduce,
    SavePlan,
    SaveRequest,
    runtime_info,
)
from summer4.solvers import Diffrax, Euler, default_solver, resolve_solver, set_default_solver
from summer4.solvers import resolve as resolve_module


@pytest.fixture(autouse=True)
def _clean_default() -> Iterator[None]:
    """Each test starts with no setter default and no cached environment default."""
    set_default_solver(None)
    resolve_module._env_default = resolve_module._UNSET
    yield
    set_default_solver(None)
    resolve_module._env_default = resolve_module._UNSET


def _decay() -> tuple[Any, dict[str, float]]:
    s = Property("s", ("a",))
    model = FlowModel(PropertyMap.from_property(s))
    model.add_flow(ExitFlow("out", s["a"], Param("k")))
    model.set_initial_population({s["a"]: 100.0})
    return model.compile(), {"k": 0.3}


def _values(result: Any) -> np.ndarray:
    return np.asarray(result["compartments"].values.data)


def _plan(t1: float, n: int) -> SavePlan:
    return SavePlan(
        requests={"compartments": SaveRequest(Compartments())}, ts=np.linspace(0.0, t1, n)
    )


def test_default_equals_explicit_default_solver() -> None:
    compiled, params = _decay()
    kw: dict[str, Any] = {"t0": 0.0, "t1": 10.0, "dt": 0.5, "save": _plan(10.0, 11)}
    a = compiled.run(params, **kw)
    b = compiled.run(params, solver=default_solver(), **kw)
    np.testing.assert_array_equal(_values(a), _values(b))
    assert a.solver is not None and b.solver is not None
    assert a.solver.solver == b.solver.solver


def test_default_solver_is_tsit5_pid() -> None:
    backend = default_solver()
    assert isinstance(backend, Diffrax)
    assert type(backend.solver).__name__ == "Tsit5"
    controller = backend.stepsize_controller
    assert controller.rtol == 1e-6 and controller.atol == 1e-9
    assert default_solver(rtol=1e-4, atol=1e-7).stepsize_controller.rtol == 1e-4


def test_matches_analytic_exponential_decay() -> None:
    compiled, params = _decay()
    result = compiled.run(params, t0=0.0, t1=10.0, dt=0.5, save=_plan(10.0, 11))
    exact = 100.0 * np.exp(-0.3 * 10.0)
    assert abs(_values(result)[-1, 0] - exact) / exact < 1e-6


def test_rtol_atol_override_default_tolerances() -> None:
    compiled, params = _decay()
    kw: dict[str, Any] = {"t0": 0.0, "t1": 10.0, "dt": 0.5, "save": _plan(10.0, 11)}
    exact = 100.0 * np.exp(-3.0)
    loose = compiled.run(params, rtol=1e-2, atol=1e-2, **kw)
    tight = compiled.run(params, rtol=1e-10, atol=1e-12, **kw)
    assert loose.solver is not None and tight.solver is not None
    assert loose.solver.num_steps < tight.solver.num_steps
    assert abs(_values(tight)[-1, 0] - exact) < abs(_values(loose)[-1, 0] - exact)
    backend = resolve_solver(None, rtol=1e-3)
    assert backend.stepsize_controller.rtol == 1e-3  # type: ignore[attr-defined]
    assert backend.stepsize_controller.atol == 1e-9  # type: ignore[attr-defined]


def _lotka_volterra() -> tuple[Any, dict[str, float]]:
    """Guide 09, copied."""
    species = Property("species", ("prey", "predator"))
    model = FlowModel(PropertyMap.from_property(species))
    model.add_flow(
        EntryFlow("prey_birth", species["prey"], Param("alpha") * Reduce(where=species["prey"]))
    )
    model.add_flow(ExitFlow("prey_death", species["prey"], Param("mu")))
    model.add_flow(
        ExitFlow("predation", species["prey"], Param("a") * Reduce(where=species["predator"]))
    )
    model.add_flow(
        EntryFlow("predator_birth", species["predator"], Param("e") * FlowRef("predation").sum_by())
    )
    model.add_flow(ExitFlow("predator_death", species["predator"], Param("d")))
    model.set_initial_population({species["prey"]: 40.0, species["predator"]: 15.0})
    return model.compile(), {"alpha": 0.6, "mu": 0.1, "a": 0.02, "e": 0.5, "d": 0.25}


def _prey_peaks(solver: Any) -> np.ndarray:
    compiled, params = _lotka_volterra()
    kw: dict[str, Any] = {"t0": 0.0, "t1": 100.0, "dt": 0.1, "save": _plan(100.0, 4001)}
    if solver is not None:
        kw["solver"] = solver
    prey = _values(compiled.run(params, **kw))[:, 0]
    is_peak = (prey[1:-1] > prey[:-2]) & (prey[1:-1] >= prey[2:])
    return prey[1:-1][is_peak]


def test_lotka_volterra_orbit_stays_closed() -> None:
    peaks = _prey_peaks(None)
    assert peaks.size >= 4
    assert np.max(np.abs(np.diff(peaks)) / peaks[:-1]) < 1e-4
    euler = _prey_peaks("euler")
    assert abs(euler[-1] - euler[0]) / euler[0] > 1e-2  # Euler spirals, so the test can tell.


def test_set_default_solver_euler_and_back() -> None:
    compiled, params = _decay()
    kw: dict[str, Any] = {"t0": 0.0, "t1": 10.0, "dt": 0.5, "save": _plan(10.0, 11)}
    assert isinstance(resolve_solver(None), Diffrax)
    set_default_solver("euler")
    assert isinstance(resolve_solver(None), Euler)
    assert "euler" in runtime_info().default_solver
    euler_value = _values(compiled.run(params, **kw))[-1, 0]
    assert euler_value == _values(compiled.run(params, solver="euler", **kw))[-1, 0]
    assert abs(euler_value - 100.0 * np.exp(-3.0)) > 1e-3  # Euler's error, not Tsit5's.
    with pytest.raises(ValueError, match="euler"):
        resolve_solver(None, rtol=1e-6)
    set_default_solver(None)
    assert isinstance(resolve_solver(None), Diffrax)
    assert runtime_info().default_solver.startswith("tsit5")


def test_set_default_solver_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown solver"):
        set_default_solver("rk99")


def test_set_default_solver_accepts_backend_object() -> None:
    set_default_solver(Euler())
    assert isinstance(resolve_solver(None), Euler)


def test_explicit_solver_wins_over_setter() -> None:
    set_default_solver("euler")
    assert isinstance(resolve_solver("tsit5"), Diffrax)
    set_default_solver("tsit5")
    assert isinstance(resolve_solver("euler"), Euler)


def _child(code: str, solver_env: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "SUMMER4_SOLVER": solver_env}
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )


def test_environment_variable_selects_euler() -> None:
    code = (
        "from summer4.solvers import resolve_solver, set_default_solver, Euler, Diffrax\n"
        "from summer4 import runtime_info\n"
        "assert isinstance(resolve_solver(None), Euler)\n"
        "assert runtime_info().default_solver == 'euler (SUMMER4_SOLVER)'\n"
        "assert isinstance(resolve_solver('tsit5'), Diffrax)\n"  # explicit wins
        "set_default_solver('dopri5')\n"  # setter wins over the environment
        "assert isinstance(resolve_solver(None), Diffrax)\n"
    )
    proc = _child(code, "euler")
    assert proc.returncode == 0, proc.stderr


def test_environment_variable_unknown_name_raises() -> None:
    proc = _child("from summer4.solvers import resolve_solver\nresolve_solver(None)\n", "rk99")
    assert proc.returncode != 0
    assert "SUMMER4_SOLVER" in proc.stderr and "Unknown solver" in proc.stderr


def test_works_under_jit_and_grad() -> None:
    compiled, _ = _decay()
    plan = _plan(10.0, 11)

    def final(k: Any) -> Any:
        res = compiled.run({"k": k}, t0=0.0, t1=10.0, dt=0.5, save=plan)
        return res["compartments"].values.data[-1, 0]

    k0 = jnp.asarray(0.3)
    value = jax.jit(final)(k0)
    grad = jax.jit(jax.grad(final))(k0)
    assert np.isfinite(float(value)) and np.isfinite(float(grad))
    h = 1e-5
    fd = (float(final(k0 + h)) - float(final(k0 - h))) / (2 * h)
    assert abs(float(grad) - fd) / abs(fd) < 1e-4
    exact = -10.0 * 100.0 * np.exp(-3.0)
    assert abs(float(grad) - exact) / abs(exact) < 1e-4


def test_jaxpr_size_does_not_grow_with_saves() -> None:
    compiled, params = _decay()

    def eqns(n_saves: int) -> int:
        plan = _plan(10.0, n_saves)

        def fn(k: Any) -> Any:
            res = compiled.run({"k": k}, t0=0.0, t1=10.0, dt=0.5, save=plan)
            return res["compartments"].values.data

        make: Callable[[Any], Any] = jax.make_jaxpr(fn)
        return len(make(params["k"]).jaxpr.eqns)

    assert eqns(50) == eqns(500)


@pytest.mark.slow
def test_tb_scale_does_not_hit_max_steps() -> None:
    from benchmarks.tb_scale import SAVE_TIMES, T0, T1, build_tb_scale_model

    tb = build_tb_scale_model()
    plan = SavePlan(requests={"compartments": SaveRequest(Compartments())}, ts=SAVE_TIMES)
    result = tb.compiled.run(tb.params, save=plan, t0=T0, t1=T1, dt=1.0)
    assert result.solver is not None
    assert bool(result.solver.ok)
    assert result.solver.num_steps is not None and result.solver.max_steps is not None
    assert result.solver.num_steps < 0.5 * result.solver.max_steps
    assert np.all(np.isfinite(_values(result)))
