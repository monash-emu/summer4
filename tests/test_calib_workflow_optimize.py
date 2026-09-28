"""Step 25 — multi-start Optax optimisation with AutoTune."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

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


class _Rates(NamedTuple):
    infection: float
    recovery: float


def _sir_bm(
    *,
    infection: float = 0.35,
    recovery: float = 0.1,
) -> BayesianModel:
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


def test_potential_fn_matches_neg_log_density() -> None:
    bm = _sir_bm()
    z = {k: v[0] for k, v in wf.lhs(bm, 2, seed=0).z.items()}
    np.testing.assert_allclose(float(bm.potential_fn(z)), -float(bm.log_density(z)), rtol=1e-5)


def test_optimize_recovers_near_find_map() -> None:
    bm = _sir_bm()
    starts = wf.evaluate(bm, wf.lhs(bm, 16, seed=1), batch_size=8).best(3)
    result = wf.optimize(
        bm,
        starts,
        method=wf.Optax(learning_rate=0.05),
        tuning=wf.AutoTune(
            probe_steps=8, probe_starts=2, patience=3, rtol=1e-5, lr_grid=(1e-2, 5e-2)
        ),
        chunk_steps=25,
        max_steps=150,
        seed=0,
    )
    for i in range(len(starts)):
        z0 = {k: jnp.asarray(starts.z[k][i]) for k in starts.sites}
        mapped = bm.find_map(z0, steps=200, seed=0).best_params
        fitted = float(result.candidates.params["infection"][i])
        np.testing.assert_allclose(fitted, float(mapped["infection"]), rtol=0.05, atol=0.02)


def test_converged_starts_are_frozen() -> None:
    bm = _sir_bm()
    starts = wf.evaluate(bm, wf.lhs(bm, 8, seed=2), batch_size=4).best(2)
    result = wf.optimize(
        bm,
        starts,
        method=wf.Optax(learning_rate=0.05, plateau=False),
        tuning=wf.AutoTune(
            probe_steps=5,
            probe_starts=1,
            patience=1,
            rtol=1e-2,
            lr_grid=(0.05,),
            max_restarts=0,
        ),
        chunk_steps=30,
        max_steps=120,
        seed=0,
    )
    assert bool(np.any(result.converged))
    fitted = np.asarray(result.candidates.params["infection"])
    assert np.all(np.isfinite(fitted))


def test_restart_from_reserve() -> None:
    """A failed start is replaced from ``reserve`` on the next chunk."""
    bm = _sir_bm()
    design = wf.evaluate(bm, wf.lhs(bm, 12, seed=3), batch_size=6)
    starts = design.best(2)
    reserve = design.best(6)
    run = wf.OptimizeRun(
        bm,
        starts,
        method=wf.Optax(learning_rate=0.05, plateau=False),
        tuning=wf.AutoTune(patience=2, rtol=1e-4, max_restarts=2),
        reserve=reserve,
        chunk_steps=15,
        seed=0,
    )
    # Poison start 0: a non-finite best loss marks it failed after the next chunk.
    run.states = {**run.states, "best_loss": run.states["best_loss"].at[0].set(jnp.nan)}
    run.extend(15)
    assert int(run.restarts[0]) == 1
    assert int(run.restarts[1]) == 0
    np.testing.assert_allclose(
        np.asarray(run.states["z"]["infection"][0]),
        np.asarray(reserve.z["infection"][0]),
    )
    run.extend(30)
    assert np.isfinite(float(run.states["best_loss"][0]))


def test_lr_probe_picks_finite_rate() -> None:
    bm = _sir_bm()
    starts = wf.evaluate(bm, wf.lhs(bm, 8, seed=4), batch_size=4).best(3)
    result = wf.optimize(
        bm,
        starts,
        method=wf.Optax(learning_rate=0.05),
        tuning=wf.AutoTune(
            lr_grid=(1e-3, 1e-2, 1e-1),
            probe_steps=10,
            probe_starts=2,
            patience=2,
            rtol=1e-4,
        ),
        chunk_steps=20,
        max_steps=60,
        seed=0,
    )
    assert np.isfinite(result.learning_rate)
    assert result.learning_rate in (1e-3, 1e-2, 1e-1)


def test_chunk_jaxpr_independent_of_n_starts() -> None:
    bm = _sir_bm()
    method = wf.Optax(learning_rate=0.05, plateau=False)
    tuning = wf.AutoTune(lr_grid=(0.05,))
    run4 = wf.OptimizeRun(bm, wf.lhs(bm, 4, seed=0), method=method, tuning=tuning, chunk_steps=20)
    run8 = wf.OptimizeRun(bm, wf.lhs(bm, 8, seed=0), method=method, tuning=tuning, chunk_steps=20)
    jp4 = jax.make_jaxpr(run4.chunk_program)(run4.states)
    jp8 = jax.make_jaxpr(run8.chunk_program)(run8.states)
    assert len(jp4.jaxpr.eqns) == len(jp8.jaxpr.eqns)
    # max_steps is only a host loop around this program
    assert "max_steps" not in str(jp4.jaxpr)
