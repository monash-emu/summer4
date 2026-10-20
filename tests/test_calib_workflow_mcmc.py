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


def test_replay_reproduces_live_decisions() -> None:
    bm = _sir_bm()
    init = _starts(bm, 2, seed=1).init_params(2, seed=1)
    run = wf.sample_until(
        _nuts_mcmc(bm, warmup=30, chunk=40),
        wf.StopRule(rhat=1.2, ess=15, max_samples=400),
        init_params=init,
        seed=1,
    )
    replayed = wf.replay(
        wf.StopRule(rhat=1.2, ess=15, max_samples=400),
        run.samples,
        40,
        diverging=run.diverging,
    )
    live = run.progress
    assert list(replayed.index) == list(live.index)
    np.testing.assert_allclose(replayed["rhat_max"], live["rhat_max"])
    np.testing.assert_allclose(replayed["ess_bulk_min"], live["ess_bulk_min"])
    assert replayed["decision"].iloc[-1] == live["decision"].iloc[-1] == "diagnostics"
    assert replayed["seconds"].isna().all()

    # A stricter rule on the same draws, without sampling again.
    strict = wf.replay(wf.StopRule(rhat=1.0, ess=10_000, max_samples=10_000), run.samples, 40)
    assert len(strict) == run.chunks
    assert strict["decision"].isna().all()


def test_sample_until_rejects_init_params_after_warmup() -> None:
    bm = _sir_bm()
    mcmc = _nuts_mcmc(bm, warmup=10, chunk=10)
    from jax import random

    mcmc.warmup(random.PRNGKey(0))
    init = _starts(bm, 2, seed=0).init_params(2)
    with pytest.raises(ValueError, match="already warmed up"):
        wf.sample_until(mcmc, wf.StopRule(rhat=None, ess=None), init_params=init)


def _make_nuts(bm: BayesianModel, *, chunk: int = 20, chains: int = 4) -> Any:
    return lambda n: _nuts_mcmc(bm, warmup=n, chunk=chunk, chains=chains)


def test_warmup_until_doubles_until_the_check_passes() -> None:
    bm = _sir_bm()
    seen: list[int] = []

    def passes_at_100(row: Mapping[str, Any]) -> tuple[str, ...]:
        seen.append(int(row["num_warmup"]))
        return () if row["num_warmup"] >= 100 else ("too short",)

    warm = wf.warmup_until(_make_nuts(bm), 25, passes_at_100, seed=0)
    assert seen == [25, 50, 100]
    assert warm.converged and warm.reason == "passed"
    assert warm.num_warmup == 100 and warm.mcmc.num_warmup == 100
    assert warm.mcmc.post_warmup_state is not None
    assert list(warm.progress["failed"].fillna("")) == ["too short", "too short", ""]
    assert warm.samples["infection"].shape == (4, 100)
    assert 0.5 < warm.progress["accept_mean"].iloc[-1] <= 1.0
    assert warm.progress["target_accept_prob"].iloc[-1] == pytest.approx(0.8)

    run = wf.sample_until(warm.mcmc, wf.StopRule(rhat=None, ess=None), seed=0)
    assert run.mcmc is warm.mcmc
    assert run.samples["infection"].shape == (4, 20)


def test_warmup_until_carries_chain_positions_between_rounds() -> None:
    from numpyro.infer import MCMC, NUTS

    bm = _sir_bm()
    inits: list[Any] = []

    class RecordingMCMC(MCMC):  # type: ignore[misc]
        def warmup(self, rng_key: Any, *args: Any, **kwargs: Any) -> None:
            inits.append(kwargs.get("init_params"))
            super().warmup(rng_key, *args, **kwargs)

    def make(n: int) -> Any:
        return RecordingMCMC(
            NUTS(bm.numpyro_model()), num_warmup=n, num_samples=10, num_chains=2, progress_bar=False
        )

    def record_then_fail_once(row: Mapping[str, Any]) -> tuple[str, ...]:
        return ("first round",) if row["round"] == 1 else ()

    start = _starts(bm, 2, seed=0).init_params(2, jitter=0.0)
    warm = wf.warmup_until(make, 20, record_then_fail_once, init_params=start, seed=0)
    assert warm.rounds == 2
    np.testing.assert_array_equal(np.asarray(inits[0]["infection"]), start["infection"])
    assert not np.allclose(np.asarray(inits[1]["infection"]), start["infection"])


def test_warmup_until_stops_at_max_warmup_and_warns() -> None:
    bm = _sir_bm()
    with pytest.warns(RuntimeWarning, match="max_warmup=100") as caught:
        warm = wf.warmup_until(_make_nuts(bm), 25, lambda row: ("never",), max_warmup=100)
    assert caught[0].filename == __file__
    assert not warm.converged and warm.reason == "max_warmup"
    assert list(warm.progress["num_warmup"]) == [25, 50, 100]
    assert warm.mcmc.post_warmup_state is not None  # still usable for sampling


def test_warmup_until_rejects_a_bad_factory() -> None:
    bm = _sir_bm()
    with pytest.raises(ValueError, match="num_warmup=25"):
        wf.warmup_until(lambda n: _nuts_mcmc(bm, warmup=10, chunk=10), 25)
    with pytest.raises(ValueError, match="growth"):
        wf.warmup_until(_make_nuts(bm), 25, growth=1.0)


def test_warmup_rule_criteria() -> None:
    good = {
        "num_warmup": 200,
        "rhat_max": 1.01,
        "step_size_ratio": 1.5,
        "accept_mean": 0.78,
        "target_accept_prob": 0.8,
        "divergence_frac": 0.0,
        "treedepth_frac": 0.0,
    }
    rule = wf.WarmupRule()
    assert rule(good) == ()
    assert rule({**good, "num_warmup": 50}) == ("min_warmup",)
    assert rule({**good, "rhat_max": float("nan")}) == ("rhat",)
    assert rule({**good, "step_size_ratio": 5.0}) == ("step_size_ratio",)
    assert rule({**good, "accept_mean": 0.6}) == ("accept_prob",)
    assert rule({**good, "divergence_frac": 0.05}) == ("divergences",)
    assert rule({**good, "treedepth_frac": 0.2}) == ("treedepth",)
    # Kernels without HMC statistics are checked on R-hat alone.
    ensemble_row = {
        **good,
        "step_size_ratio": None,
        "accept_mean": None,
        "target_accept_prob": None,
        "divergence_frac": None,
        "treedepth_frac": None,
    }
    assert rule(ensemble_row) == ()
    assert wf.WarmupRule(rhat=None, min_warmup=0)({**ensemble_row, "rhat_max": 9.0}) == ()


def test_warmup_until_with_an_ensemble_kernel_checks_rhat_only() -> None:
    from numpyro.infer import AIES, MCMC

    bm = _sir_bm()

    def make(n: int) -> Any:
        return MCMC(
            AIES(bm.numpyro_model()),
            num_warmup=n,
            num_samples=10,
            num_chains=4,
            chain_method="vectorized",
            progress_bar=False,
        )

    warm = wf.warmup_until(
        make,
        20,
        wf.WarmupRule(min_warmup=0, rhat=100.0),
        init_params=_starts(bm, 2, seed=0).init_params(4, jitter=0.05),
    )
    row = warm.progress.iloc[-1]
    assert warm.converged
    assert row["step_size_ratio"] is None or np.isnan(row["step_size_ratio"])
    assert row["accept_mean"] is None or np.isnan(row["accept_mean"])


def test_run_mcmc_with_warmup_equals_its_documented_expansion() -> None:
    bm = _sir_bm()
    starts = _starts(bm, 2, seed=5)
    rule = wf.WarmupRule(min_warmup=40, rhat=1.2)
    stop = wf.StopRule(rhat=None, ess=None)
    short = wf.run_mcmc(
        bm,
        starts,
        num_chains=2,
        num_warmup=20,
        chunk_samples=15,
        stop=stop,
        warmup=rule,
        seed=5,
    )
    warm = wf.warmup_until(
        lambda n: _nuts_mcmc(bm, warmup=n, chunk=15),
        20,
        rule,
        init_params=starts.init_params(2, jitter=0.05, seed=5),
        seed=5,
    )
    long = wf.sample_until(warm.mcmc, stop, seed=5)
    assert short.mcmc.num_warmup == warm.num_warmup
    np.testing.assert_array_equal(short.samples["infection"], long.samples["infection"])


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


# --- step 47: extra fields, worst-element diagnostics, steerable warm-up (CX36-CX38) ---


def _after(samples: int) -> wf.Stop:
    def stop(row: Mapping[str, Any]) -> wf.Decision | None:
        return (True, "enough") if int(row["samples"]) >= samples else None

    return stop


def test_extra_fields_are_kept_across_chunks_and_extend() -> None:
    bm = _sir_bm()
    run = wf.sample_until(
        _nuts_mcmc(bm, warmup=30, chunk=10),
        _after(20),
        init_params=_starts(bm, 2, seed=0).init_params(2),
        extra_fields=("num_steps", "accept_prob"),
    )
    assert run.chunks == 2
    assert set(run.extra) == {"num_steps", "accept_prob"}
    assert run.extra["num_steps"].shape == (2, 20)
    run.extend(_after(30))
    assert run.extra["num_steps"].shape == (2, 30)
    assert run.extra["accept_prob"].shape == (2, 30)
    assert run.diverging is not None and run.diverging.shape == (2, 30)
    assert np.all(run.extra["num_steps"] >= 1)
    assert np.all((run.extra["accept_prob"] >= 0) & (run.extra["accept_prob"] <= 1))
    stats = run.idata.sample_stats
    assert {"num_steps", "accept_prob", "diverging"} <= set(stats.data_vars)


def test_without_extra_fields_only_divergences_are_kept() -> None:
    bm = _sir_bm()
    run = wf.sample_until(
        _nuts_mcmc(bm, warmup=20, chunk=10),
        _after(10),
        init_params=_starts(bm, 2, seed=0).init_params(2),
    )
    assert run.extra == {} and run.diverging is not None


def _one_bad_element(seed: int = 0) -> dict[str, np.ndarray]:
    """Four chains; element 4 of the plate site has chain means spread apart."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(4, 400, 5))
    z[:, :, 4] += 0.3 * np.array([-1.5, -0.5, 0.5, 1.5])[:, None]
    return {"z": z, "s": rng.normal(size=(4, 400))}


def test_worst_element_rhat_is_reported_and_per_element_stop_uses_it() -> None:
    samples = _one_bad_element()
    rows = wf.replay(wf.StopRule(rhat=1.05, ess=None), samples, 400)
    row = rows.iloc[-1]
    # The mean over z's five elements hides the one bad element; the max does not.
    assert row["rhat_max"] < 1.05 < row["rhat_element_max"]
    assert row["ess_bulk_element_min"] < row["ess_bulk_min"]
    assert row["decision"] == "diagnostics"
    strict = wf.replay(wf.StopRule(rhat=1.05, ess=None, per_element=True), samples, 400)
    assert strict["decision"].isna().all()


def test_diagnostics_table_has_worst_element_columns() -> None:
    bm = _sir_bm()
    run = wf.sample_until(
        _nuts_mcmc(bm, warmup=20, chunk=20),
        _after(20),
        init_params=_starts(bm, 2, seed=0).init_params(2),
    )
    diag = run.diagnostics
    assert list(diag.columns) == ["rhat", "ess_bulk", "rhat_element_max", "ess_bulk_element_min"]
    # A scalar site has one element, so mean and worst agree.
    assert diag.loc["infection", "rhat"] == pytest.approx(diag.loc["infection", "rhat_element_max"])


def test_warmup_until_hands_a_metric_to_the_factory_and_reports_each_round() -> None:
    from numpyro.infer import MCMC, NUTS

    bm = _sir_bm()
    calls: list[Any] = []

    def make(n: int, inverse_mass_matrix: Any = None) -> Any:
        calls.append(inverse_mass_matrix)
        return MCMC(
            NUTS(bm.numpyro_model(), inverse_mass_matrix=inverse_mass_matrix),
            num_warmup=n,
            num_samples=10,
            num_chains=2,
            chain_method="vectorized",
            progress_bar=False,
        )

    seen: list[Mapping[str, Any]] = []
    start = {("infection",): np.array([0.5])}
    warm = wf.warmup_until(
        make,
        40,
        lambda row: ("first",) if row["round"] == 1 else (),
        init_params=_starts(bm, 2, seed=0).init_params(2),
        inverse_mass_matrix=start,
        carry_metric=True,
        on_round=seen.append,
    )
    assert [r["round"] for r in seen] == [1, 2] == list(warm.progress.index)
    assert all(int(r["grad_evals"]) >= r["num_warmup"] * 2 for r in seen)
    assert calls[0] is start
    # Round 2 starts from round 1's adapted metric, averaged over the two chains.
    assert set(calls[1]) == {("infection",)}
    assert calls[1][("infection",)].shape == (1,)
    assert not np.allclose(calls[1][("infection",)], 0.5)


def test_a_fixed_metric_reaches_the_kernel() -> None:
    from numpyro.infer import MCMC, NUTS

    bm = _sir_bm()
    metric = {("infection",): np.array([0.25])}

    def make(n: int, inverse_mass_matrix: Any = None) -> Any:
        kernel = NUTS(
            bm.numpyro_model(), inverse_mass_matrix=inverse_mass_matrix, adapt_mass_matrix=False
        )
        return MCMC(kernel, num_warmup=n, num_samples=10, num_chains=1, progress_bar=False)

    warm = wf.warmup_until(make, 30, lambda row: (), inverse_mass_matrix=metric)
    used = warm.mcmc.post_warmup_state.adapt_state.inverse_mass_matrix
    np.testing.assert_allclose(np.asarray(used[("infection",)]), [0.25])


def test_warmup_until_abort_stops_doubling() -> None:
    from numpyro.infer import MCMC, NUTS

    bm = _sir_bm()

    def make(n: int) -> Any:
        kernel = NUTS(bm.numpyro_model(), max_tree_depth=1)  # every trajectory at the cap
        return MCMC(kernel, num_warmup=n, num_samples=10, num_chains=2, progress_bar=False)

    with pytest.warns(RuntimeWarning, match="aborted"):
        warm = wf.warmup_until(
            make,
            20,
            lambda row: ("always",),
            init_params=_starts(bm, 2, seed=0).init_params(2),
            abort=lambda row: float(row["treedepth_frac"]) > 0.5,
        )
    assert warm.rounds == 1 and not warm.converged and warm.reason == "aborted"
    assert warm.progress["treedepth_frac"].iloc[0] == pytest.approx(1.0)
