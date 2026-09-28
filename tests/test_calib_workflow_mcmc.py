"""Step 27 — seeded chunked MCMC: init_params, sample_until, StopRule, run_mcmc."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, NamedTuple

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


def _starts(bm: BayesianModel, n: int, seed: int) -> wf.Candidates:
    return wf.evaluate(bm, wf.lhs(bm, 8, seed=seed), batch_size=4).best(n)


def _nuts_mcmc(bm: BayesianModel, *, warmup: int, chunk: int, chains: int = 2) -> Any:
    from numpyro.infer import MCMC, NUTS

    return MCMC(
        NUTS(bm.numpyro_model()),
        num_warmup=warmup,
        num_samples=chunk,
        num_chains=chains,
        chain_method="vectorized",
        progress_bar=False,
    )


def test_init_params_cycles_rows_and_jitters() -> None:
    bm = _sir_bm()
    starts = _starts(bm, 2, seed=0)
    exact = starts.init_params(5, jitter=0.0)
    z = np.asarray(starts.z["infection"])
    np.testing.assert_array_equal(exact["infection"], z[[0, 1, 0, 1, 0]])

    jittered = starts.init_params(5, jitter=0.05, seed=3)
    assert jittered["infection"].shape == (5,)
    assert not np.allclose(jittered["infection"], exact["infection"])
    assert np.max(np.abs(jittered["infection"] - exact["infection"])) < 0.5
    np.testing.assert_array_equal(
        jittered["infection"], starts.init_params(5, jitter=0.05, seed=3)["infection"]
    )


def test_init_params_rejects_empty_and_zero_chains() -> None:
    bm = _sir_bm()
    starts = _starts(bm, 2, seed=0)
    with pytest.raises(ValueError, match="at least one"):
        starts.take(np.array([], dtype=int)).init_params(2)
    with pytest.raises(ValueError, match=">= 1"):
        starts.init_params(0)


def test_stop_rule_order_of_checks() -> None:
    row = {
        "samples": 100,
        "rhat_max": 1.2,
        "ess_bulk_min": 50.0,
        "divergence_frac": 0.1,
        "seconds": 5.0,
    }
    assert wf.StopRule(rhat=1.3, ess=40)(row) == (True, "diagnostics")
    assert wf.StopRule(rhat=1.1, ess=40, max_samples=1000)(row) is None
    assert wf.StopRule(rhat=1.1, ess=40, max_samples=100)(row) == (False, "max_samples")
    assert wf.StopRule(rhat=1.1, max_samples=1000, max_seconds=5.0)(row) == (
        False,
        "max_seconds",
    )
    # Divergences outrank a passing diagnostic.
    assert wf.StopRule(rhat=1.3, ess=40, max_divergence_frac=0.05)(row) == (
        False,
        "divergences",
    )
    assert wf.StopRule(rhat=None, ess=None)(row) == (True, "diagnostics")


def test_sample_until_passes_init_params_to_numpyro() -> None:
    from numpyro.infer import MCMC

    bm = _sir_bm()
    init = _starts(bm, 2, seed=0).init_params(2, jitter=0.0)
    seen: list[Any] = []

    class RecordingMCMC(MCMC):  # type: ignore[misc]
        def run(self, rng_key: Any, *args: Any, **kwargs: Any) -> None:
            seen.append(kwargs.get("init_params"))
            super().run(rng_key, *args, **kwargs)

    from numpyro.infer import NUTS

    mcmc = RecordingMCMC(
        NUTS(bm.numpyro_model()), num_warmup=10, num_samples=10, num_chains=2, progress_bar=False
    )
    run = wf.sample_until(mcmc, wf.StopRule(rhat=None, ess=None), init_params=init)
    assert run.mcmc is mcmc
    assert run.chunks == 1
    np.testing.assert_array_equal(np.asarray(seen[0]["infection"]), init["infection"])


def test_stops_on_first_passing_chunk() -> None:
    bm = _sir_bm()
    init = _starts(bm, 2, seed=1).init_params(2, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        run = wf.sample_until(
            _nuts_mcmc(bm, warmup=30, chunk=40),
            wf.StopRule(rhat=1.2, ess=15, max_samples=400),
            init_params=init,
            seed=1,
        )
    assert run.converged
    assert run.reason == "diagnostics"
    assert 1 <= run.chunks < 10
    progress = run.progress
    assert list(progress.index) == list(range(1, run.chunks + 1))
    assert list(progress["samples"]) == [40 * k for k in range(1, run.chunks + 1)]
    assert list(progress["decision"].dropna()) == ["diagnostics"]
    assert progress["decision"].last_valid_index() == run.chunks
    last = progress.iloc[-1]
    assert last["rhat_max"] <= 1.2
    assert last["ess_bulk_min"] >= 15
    assert last["rhat_max"] == pytest.approx(float(run.diagnostics["rhat"].max()))
    assert np.asarray(run.idata.posterior["infection"].values).shape == (2, 40 * run.chunks)


def test_budget_stop_warns_and_extend_resumes() -> None:
    bm = _sir_bm()
    init = _starts(bm, 2, seed=2).init_params(2, seed=2)
    mcmc = _nuts_mcmc(bm, warmup=10, chunk=20)
    with pytest.warns(RuntimeWarning, match="max_samples") as caught:
        run = wf.sample_until(
            mcmc, wf.StopRule(rhat=1.01, ess=10_000, max_samples=60), init_params=init, seed=2
        )
    assert caught[0].filename == __file__  # points at the caller, not summer4 internals
    assert not run.converged
    assert run.reason == "max_samples"
    assert run.chunks == 3
    before = np.array(run.samples["infection"])

    resumed = run.extend(wf.StopRule(rhat=None, ess=None))
    assert resumed is run
    assert run.mcmc is mcmc
    assert run.chunks == 4
    assert run.converged and run.reason == "diagnostics"
    after = run.samples["infection"]
    assert after.shape == (2, 80)
    np.testing.assert_array_equal(after[:, :60], before)
    assert run.progress["decision"].dropna().to_dict() == {3: "max_samples", 4: "diagnostics"}


def test_custom_stop_callable() -> None:
    bm = _sir_bm()

    def two_chunks(row: Mapping[str, Any]) -> wf.Decision | None:
        return (True, "two chunks") if row["samples"] >= 40 else None

    run = wf.sample_until(_nuts_mcmc(bm, warmup=10, chunk=20), two_chunks)
    assert run.chunks == 2
    assert run.reason == "two chunks"


def test_divergence_stop_ends_run_unconverged() -> None:
    bm = _sir_bm()
    with pytest.warns(RuntimeWarning, match="divergences"):
        run = wf.sample_until(
            _nuts_mcmc(bm, warmup=10, chunk=20),
            wf.StopRule(rhat=None, ess=None, max_divergence_frac=-1.0),
        )
    assert run.chunks == 1
    assert run.reason == "divergences"
    assert run.diverging is not None and run.diverging.shape == (2, 20)


def test_ensemble_kernel_runs_and_rejects_duplicate_walkers() -> None:
    from numpyro.infer import AIES, MCMC

    bm = _sir_bm()
    starts = _starts(bm, 1, seed=3)
    mcmc = MCMC(
        AIES(bm.numpyro_model()),
        num_warmup=10,
        num_samples=10,
        num_chains=4,
        chain_method="vectorized",
        progress_bar=False,
    )
    with pytest.raises(ValueError, match="distinct walkers"):
        wf.sample_until(mcmc, wf.StopRule(), init_params=starts.init_params(4, jitter=0.0))
    run = wf.sample_until(
        mcmc, wf.StopRule(rhat=None, ess=None), init_params=starts.init_params(4, jitter=0.05)
    )
    assert run.chunks == 1
    assert run.diverging is None  # AIES records no divergences
    assert run.samples["infection"].shape == (4, 10)


def test_run_mcmc_equals_its_documented_expansion() -> None:
    bm = _sir_bm()
    starts = _starts(bm, 2, seed=4)
    stop = wf.StopRule(rhat=None, ess=None)
    short = wf.run_mcmc(
        bm, starts, num_chains=2, num_warmup=15, chunk_samples=15, stop=stop, seed=4
    )
    long = wf.sample_until(
        _nuts_mcmc(bm, warmup=15, chunk=15),
        stop,
        init_params=starts.init_params(2, jitter=0.05, seed=4),
        seed=4,
    )
    np.testing.assert_array_equal(short.samples["infection"], long.samples["infection"])


def test_candidates_from_run_are_last_draws() -> None:
    bm = _sir_bm()
    run = wf.sample_until(_nuts_mcmc(bm, warmup=10, chunk=10), wf.StopRule(rhat=None, ess=None))
    cands = run.candidates(bm)
    assert len(cands) == 2
    np.testing.assert_allclose(
        cands.params["infection"], run.samples["infection"][:, -1], rtol=1e-6
    )
    assert cands.history[0].stage == "sample_until"
