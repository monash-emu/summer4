"""``laplace_metric(hessian=...)`` on an ODE posterior solved with summer4's default solver.

The default solver's diffrax adjoint (``RecursiveCheckpointAdjoint``) is a ``custom_vjp``, so
``jax.hessian`` (forward over reverse) cannot differentiate it. ``"auto"`` falls back to central
differences of the reverse-mode gradient; ``"exact"`` says why it failed and what to do.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import diffrax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
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
from summer4.epi.calibration import NumpyroPosterior
from summer4.epi.calibration import workflow as wf
from summer4.solvers import Diffrax, default_solver

_Z = {"beta": np.log(0.35), "gamma": np.log(0.1)}


def _sir_solve() -> Callable[[Any, Any, Any], Any]:
    state = Property("state", ("S", "I", "R"))
    fm = FlowModel(PropertyMap.from_property(state))
    fm.add_flow(TransitionFlow("infection", state["S"], state["I"], Param("infection")))
    fm.add_flow(TransitionFlow("recovery", state["I"], state["R"], Param("recovery")))
    cm = fm.compile()
    y0 = PropertyData.wrap(cm.pmap, np.array([999.0, 1.0, 0.0]))
    plan = SavePlan(
        requests={"I": SaveRequest(Compartments(where=state["I"]), ts=np.linspace(0, 40, 9))}
    )

    def solve(beta: Any, gamma: Any, solver: Any) -> Any:
        result = cm.run(
            {"infection": beta, "recovery": gamma},
            y0,
            t0=0.0,
            t1=40.0,
            dt=0.5,
            save=plan,
            solver=solver,
        )
        raw = result["I"].values
        return jnp.ravel(getattr(raw, "data", raw))

    return solve


def _direct_adjoint() -> Diffrax:
    base = default_solver()
    return Diffrax(
        base.solver,
        stepsize_controller=base.stepsize_controller,
        adjoint=diffrax.DirectAdjoint(),
    )


@pytest.fixture(scope="module")
def sir() -> Callable[[Any], NumpyroPosterior]:
    solve = _sir_solve()
    obs = np.asarray(solve(0.35, 0.1, default_solver()))

    def make_post(solver: Any) -> NumpyroPosterior:
        def model() -> None:
            beta = numpyro.sample("beta", dist.LogNormal(np.log(0.3), 0.5))
            gamma = numpyro.sample("gamma", dist.LogNormal(np.log(0.1), 0.5))
            numpyro.sample("y", dist.Normal(solve(beta, gamma, solver), 10.0), obs=obs)

        return NumpyroPosterior(model)

    return make_post


def test_auto_falls_back_to_fd_and_matches_the_direct_adjoint_hessian(
    sir: Callable[[Any], NumpyroPosterior],
) -> None:
    metric = wf.laplace_metric(sir(default_solver()), _Z)
    exact = wf.laplace_metric(sir(_direct_adjoint()), _Z, hessian="exact")
    assert metric.hessian == "fd"
    assert exact.hessian == "exact"
    assert metric.sites == exact.sites == ("beta", "gamma")
    np.testing.assert_allclose(metric.covariance, exact.covariance, rtol=1e-5)
    np.testing.assert_allclose(metric.eigenvalues, exact.eigenvalues, rtol=1e-5)
    assert metric.clipped == exact.clipped


def test_exact_on_the_default_solver_names_both_fixes(
    sir: Callable[[Any], NumpyroPosterior],
) -> None:
    with pytest.raises(ValueError, match="hessian='fd'") as info:
        wf.laplace_metric(sir(default_solver()), _Z, hessian="exact")
    assert "DirectAdjoint" in str(info.value)
    assert isinstance(info.value.__cause__, TypeError)
