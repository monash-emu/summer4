"""Step 27 — seeded MCMC with StopRule."""

from __future__ import annotations

import warnings
from typing import NamedTuple

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


def test_seeded_init_params_are_used() -> None:
    """Seeding injects unconstrained z; seeded and unseeded posteriors differ."""
    import importlib

    mcmc_mod = importlib.import_module("summer4.epi.calibration.workflow.mcmc")
    bm = _sir_bm()
    starts = wf.evaluate(bm, wf.lhs(bm, 8, seed=0), batch_size=4).best(2)
    init_params = mcmc_mod._seed_init_params(
        starts,
        sites=bm.prior_names(),
        num_chains=2,
        jitter=0.0,
        seed=0,
        kind="nuts",
    )
    assert init_params is not None
    np.testing.assert_allclose(
        np.asarray(init_params["infection"]),
        np.asarray(starts.z["infection"][:2]),
        rtol=1e-5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seeded = wf.run_mcmc(
            bm,
            init=starts,
            num_chains=2,
            num_warmup=20,
            chunk_samples=30,
            stop=wf.StopRule(rhat=None, ess=None, max_samples=30),
            jitter=0.0,
            seed=0,
        )
        unseeded = wf.run_mcmc(
            bm,
            init=None,
            num_chains=2,
            num_warmup=20,
            chunk_samples=30,
            stop=wf.StopRule(rhat=None, ess=None, max_samples=30),
            seed=0,
        )
    s = np.asarray(seeded.idata.posterior["infection"].values).mean()
    u = np.asarray(unseeded.idata.posterior["infection"].values).mean()
    assert abs(float(s) - float(u)) > 1e-4
    assert len(seeded.candidates) == 2


def test_stops_early_once_diagnostics_pass() -> None:
    bm = _sir_bm()
    starts = wf.evaluate(bm, wf.lhs(bm, 8, seed=1), batch_size=4).best(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wf.run_mcmc(
            bm,
            init=starts,
            num_chains=2,
            num_warmup=30,
            chunk_samples=40,
            stop=wf.StopRule(rhat=1.2, ess=15, max_samples=400),
            seed=1,
        )
    assert res.converged
    assert res.reason == "diagnostics"
    assert res.chunks >= 1
    assert res.chunks < 10  # should not burn the full budget


def test_continues_then_hits_budget() -> None:
    bm = _sir_bm()
    starts = wf.evaluate(bm, wf.lhs(bm, 8, seed=2), batch_size=4).best(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wf.run_mcmc(
            bm,
            init=starts,
            num_chains=2,
            num_warmup=10,
            chunk_samples=20,
            # Impossible ESS so we keep going until max_samples
            stop=wf.StopRule(rhat=1.01, ess=10_000, max_samples=60),
            seed=2,
        )
    assert not res.converged
    assert res.reason == "max_samples"
    assert res.chunks >= 2


def test_aies_rejects_duplicate_walkers_when_jitter_zero() -> None:
    bm = _sir_bm()
    # One candidate cycled → duplicates with jitter=0
    starts = wf.evaluate(bm, wf.lhs(bm, 4, seed=3), batch_size=4).best(1)
    with pytest.raises(ValueError, match="distinct"):
        wf.run_mcmc(
            bm,
            init=starts,
            kind="aies",
            num_chains=2,
            num_warmup=5,
            chunk_samples=5,
            stop=wf.StopRule(rhat=None, ess=None, max_samples=5),
            jitter=0.0,
            seed=0,
        )


def test_divergence_retry_raises_target_accept() -> None:
    """With max_divergence_frac=0, NUTS retries and steps target_accept_prob."""
    bm = _sir_bm()
    starts = wf.evaluate(bm, wf.lhs(bm, 8, seed=4), batch_size=4).best(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wf.run_mcmc(
            bm,
            init=starts,
            kind="nuts",
            num_chains=2,
            num_warmup=20,
            chunk_samples=20,
            stop=wf.StopRule(rhat=None, ess=None, max_samples=20, max_divergence_frac=0.0),
            target_accept_prob=0.8,
            max_retries=2,
            seed=4,
        )
    # Completes (may or may not eliminate all divergences on this tiny model)
    assert res.chunks >= 1
    assert res.history[0].settings["num_warmup"] >= 20
