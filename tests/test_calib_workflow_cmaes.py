"""Step 26 — CMA-ES gradient-free backend on the multi-start driver."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import numpy as np
import pytest

from summer4 import (
    Compartments,
    FlowModel,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    Target,
    TargetSet,
    TransitionFlow,
    derived_refs,
)
from summer4.epi.calibration import (
    BayesianModel,
    NormalLikelihood,
    Uniform,
)
from summer4.epi.calibration import workflow as wf

pytest.importorskip("evosax")


class _Rates(NamedTuple):
    infection: float
    recovery: float


def _sir_bm(*, infection: float = 0.35, recovery: float = 0.1) -> BayesianModel:
    state = Property("state", ("S", "I", "R"))
    pmap = PropertyMap.from_property(state)
    refs = derived_refs(_Rates)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], refs.infection))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], refs.recovery))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([999.0, 1.0, 0.0]))
    times = np.array([0.0, 20.0, 40.0, 60.0])
    qty = Compartments(where=state["I"])
    params = {"infection": infection, "recovery": recovery}
    truth = cm.run(
        params,
        y0,
        t0=0.0,
        t1=80.0,
        dt=1.0,
        save=SavePlan(requests={"I": SaveRequest(qty, ts=times)}),
        solver="euler",
    )
    raw = truth["I"].at_times(times).values
    values = np.asarray(raw.data if hasattr(raw, "data") else raw).reshape(-1)
    targets = TargetSet(
        targets=(
            Target(
                key="I",
                times=times,
                values=values,
                quantity=qty,
                likelihood=NormalLikelihood(sd=5.0),
            ),
        )
    )
    return BayesianModel(
        cm,
        {"recovery": recovery},
        priors=(Uniform("infection", 0.05, 1.0),),
        targets=targets,
        y0=y0,
        run_kwargs={"t0": 0.0, "t1": 80.0, "dt": 1.0, "solver": "euler"},
    )


def test_cmaes_recovers_near_truth() -> None:
    bm = _sir_bm()
    starts = wf.evaluate(bm, wf.lhs(bm, 16, seed=1), batch_size=8).best(3)
    result = wf.optimize(
        bm,
        starts,
        method=wf.CMAES(sigma0=0.3, population=8),
        tuning=wf.AutoTune(patience=3, rtol=1e-4, max_restarts=0),
        chunk_steps=5,
        max_steps=40,
        seed=0,
    )
    fitted = np.asarray(result.candidates.params["infection"])
    assert np.all(np.abs(fitted - 0.35) < 0.05)


def test_cmaes_with_stop_gradient_potential() -> None:
    """CMA-ES still converges when gradients are unavailable."""
    bm = _sir_bm()
    starts = wf.evaluate(bm, wf.lhs(bm, 12, seed=2), batch_size=6).best(2)
    raw_potential = bm.potential_fn

    def blocked(z: Any) -> Any:
        return raw_potential(jax.tree.map(jax.lax.stop_gradient, z))

    # Swap potential via a thin BayesianModel-like shim
    class _Shim:
        potential_fn = staticmethod(blocked)
        prior_names = bm.prior_names
        constrain = bm.constrain
        log_density = bm.log_density

    result = wf.optimize(
        _Shim(),
        starts,
        method=wf.CMAES(sigma0=0.3, population=8),
        tuning=wf.AutoTune(patience=3, rtol=1e-4, max_restarts=0),
        chunk_steps=5,
        max_steps=40,
        seed=1,
    )
    fitted = np.asarray(result.candidates.params["infection"])
    assert np.all(np.abs(fitted - 0.35) < 0.08)


def test_cmaes_and_optax_both_descend() -> None:
    bm = _sir_bm()
    starts = wf.evaluate(bm, wf.lhs(bm, 16, seed=0), batch_size=8).best(2)
    optax_res = wf.optimize(
        bm,
        starts,
        method=wf.Optax(learning_rate=0.05, plateau=False),
        tuning=wf.AutoTune(lr_grid=(0.05,), probe_steps=4, probe_starts=1, patience=2),
        chunk_steps=20,
        max_steps=60,
        seed=0,
    )
    cma_res = wf.optimize(
        bm,
        starts,
        method=wf.CMAES(sigma0=0.3, population=8),
        tuning=wf.AutoTune(patience=2, rtol=1e-4, max_restarts=0),
        chunk_steps=5,
        max_steps=30,
        seed=0,
    )
    assert float(optax_res.loss_trace[-1].mean()) < float(optax_res.loss_trace[0].mean())
    assert float(cma_res.loss_trace[-1].mean()) < float(cma_res.loss_trace[0].mean())
