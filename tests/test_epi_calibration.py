"""WP10 §10.1–§10.3 — priors, likelihoods, BayesianModel, posterior runs."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from summer4 import (
    Compartments,
    FlowModel,
    OutputSet,
    Param,
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
    Beta,
    Gamma,
    LogNormal,
    NegativeBinomial,
    NormalLikelihood,
    NormalPrior,
    Poisson,
    Scenario,
    TruncatedNormal,
    Uniform,
    prior_sites,
    priors_from_frame,
)
from summer4.epi.calibration.likelihoods import AggregateHow, resolve_scale


class _Rates(NamedTuple):
    infection: float
    recovery: float


def _sir_fixture(
    *,
    infection: float = 0.35,
    recovery: float = 0.1,
    times: np.ndarray | None = None,
    sd: float = 5.0,
) -> tuple[Any, Any, TargetSet, Any, np.ndarray]:
    """Compiled SIR, y0, targets at synthetic truth, and fixed-param dict."""
    state = Property("state", ("S", "I", "R"))
    pmap = PropertyMap.from_property(state)
    refs = derived_refs(_Rates)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], refs.infection))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], refs.recovery))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([999.0, 1.0, 0.0]))
    if times is None:
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
                likelihood=NormalLikelihood(sd=sd),
            ),
        )
    )
    return cm, y0, targets, params, values


def _sir_run(
    *,
    infection: float = 0.3,
    recovery: float = 0.1,
    times: np.ndarray | None = None,
) -> tuple[Any, TargetSet, Any]:
    if times is None:
        times = np.array([0.0, 10.0, 20.0, 40.0])
    cm, y0, targets, params, _values = _sir_fixture(
        infection=infection, recovery=recovery, times=times, sd=1.0
    )
    result = cm.run(
        params,
        y0,
        t0=0.0,
        t1=float(np.max(times)),
        dt=1.0,
        save=targets.plan(SavePlan()),
        solver="euler",
    )
    return cm, targets, result


def test_prior_to_numpyro_and_bounds() -> None:
    cases: list[tuple[Any, tuple[float | None, float | None]]] = [
        (Uniform("u", 0.0, 2.0), (0.0, 2.0)),
        (NormalPrior("n", 0.0, 1.0), (None, None)),
        (LogNormal("ln", 0.0, 0.5), (0.0, None)),
        (TruncatedNormal("tn", 0.0, 1.0, low=-1.0, high=1.0), (-1.0, 1.0)),
        (Beta("b", 2.0, 5.0), (0.0, 1.0)),
        (Gamma("g", 2.0, rate=0.5), (0.0, None)),
    ]
    for prior, expected_bounds in cases:
        assert prior.bounds() == expected_bounds
        d = prior.to_numpyro()
        x = 0.5 if expected_bounds[0] == 0.0 or expected_bounds[0] is None else 0.0
        if isinstance(prior, Uniform):
            x = 1.0
        if isinstance(prior, TruncatedNormal):
            x = 0.0
        assert np.isfinite(float(d.log_prob(x)))


def test_prior_transforms_round_trip_via_numpyro() -> None:
    """Constraint bijector forward/inverse recovers the constrained value."""
    from numpyro.distributions.transforms import biject_to

    prior = Uniform("u", 1.0, 3.0)
    d = prior.to_numpyro()
    transform = biject_to(d.support)
    constrained = jnp.asarray(2.0)
    unconstrained = transform.inv(constrained)
    recovered = transform(unconstrained)
    np.testing.assert_allclose(float(recovered), 2.0, rtol=1e-5)


def test_priors_from_frame_mapping_and_pandas() -> None:
    table = {
        "name": ["beta", "sigma"],
        "dist": ["uniform", "Normal"],
        "p1": [0.0, 0.0],
        "p2": [1.0, 2.0],
    }
    built = priors_from_frame(table, "name", "dist", "p1", "p2")
    assert len(built) == 2
    assert isinstance(built[0], Uniform)
    assert built[0].name == "beta"
    assert built[0].lo == 0.0 and built[0].hi == 1.0
    assert isinstance(built[1], NormalPrior)
    assert built[1].scale == 2.0

    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(table)
    again = priors_from_frame(df, "name", "dist", "p1", "p2")
    assert again[0].name == built[0].name
    assert again[1].scale == built[1].scale


def test_likelihood_log_prob_matches_numpyro() -> None:
    observed = jnp.asarray([1.0, 2.0, 3.0])
    predicted = jnp.asarray([1.1, 1.9, 3.2])
    params: dict[str, float] = {}

    normal = NormalLikelihood(sd=0.5)
    expected = jnp.mean(dist.Normal(predicted, 0.5).log_prob(observed))
    np.testing.assert_allclose(
        float(normal.log_prob(observed, predicted, params)),
        float(expected),
        rtol=1e-5,
    )

    pois = Poisson()
    expected_p = jnp.mean(dist.Poisson(predicted).log_prob(observed))
    np.testing.assert_allclose(
        float(pois.log_prob(observed, predicted, params)),
        float(expected_p),
        rtol=1e-5,
    )

    nb = NegativeBinomial(dispersion=4.0)
    expected_nb = jnp.mean(dist.NegativeBinomial2(predicted, 4.0).log_prob(observed))
    np.testing.assert_allclose(
        float(nb.log_prob(observed, predicted, params)),
        float(expected_nb),
        rtol=1e-5,
    )

    summed = NormalLikelihood(sd=0.5, aggregate="sum")
    expected_sum = jnp.sum(dist.Normal(predicted, 0.5).log_prob(observed))
    np.testing.assert_allclose(
        float(summed.log_prob(observed, predicted, params)),
        float(expected_sum),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        float(summed.log_prob(observed, predicted, params)),
        float(normal.log_prob(observed, predicted, params)) * 3.0,
        rtol=1e-5,
    )


def test_normal_from_tolerance() -> None:
    data = np.array([100.0, 110.0, 90.0])
    lik = NormalLikelihood.from_tolerance(data, tol_pct=10.0)
    mean = float(np.mean(np.abs(data)))
    assert lik.sd == pytest.approx((10.0 / 100.0) * mean / 1.96)
    assert lik.aggregate == AggregateHow.MEAN


def test_hierarchical_sd_resolves_from_params() -> None:
    prior = Uniform("mixing_dist_sd", 5.0, 20.0)
    lik = NormalLikelihood(sd=prior)
    assert prior_sites(lik) == (prior,)
    params = {"mixing_dist_sd": 12.0}
    assert float(resolve_scale(lik.sd, params)) == 12.0
    lik_param = NormalLikelihood(sd=Param("sigma"))
    assert float(resolve_scale(lik_param.sd, {"sigma": 0.25})) == 0.25


def test_targetset_log_likelihood_matches_hand_mean() -> None:
    _cm, targets, result = _sir_run()
    params: dict[str, float] = {}
    ll = targets.log_likelihood(result, params)
    raw = result["I"].at_times(targets.targets[0].times).values
    pred = np.asarray(raw.data if hasattr(raw, "data") else raw).reshape(-1)
    obs = targets.targets[0].values
    expected = float(jnp.mean(dist.Normal(pred, 1.0).log_prob(obs)))
    np.testing.assert_allclose(float(ll), expected, rtol=1e-4)


def test_targetset_log_likelihood_jittable() -> None:
    _cm, targets, result = _sir_run()
    params: dict[str, float] = {}

    def loss(params_in: Any) -> Any:
        return targets.log_likelihood(result, params_in)

    jitted = jax.jit(loss)
    a = float(jitted(params))
    b = float(loss(params))
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_log_likelihood_requires_likelihood() -> None:
    _cm, good_targets, result = _sir_run()
    bare = TargetSet(
        targets=(
            Target(
                key="I",
                times=good_targets.targets[0].times,
                values=good_targets.targets[0].values,
                quantity=good_targets.targets[0].quantity,
            ),
        )
    )
    with pytest.raises(ValueError, match="no likelihood"):
        bare.log_likelihood(result, {})


def test_bayesian_model_rejects_preprocess() -> None:
    cm, y0, targets, params, _ = _sir_fixture()
    with pytest.raises(TypeError, match="prepare_fn"):
        BayesianModel(
            cm,
            params,
            priors=(Uniform("infection", 0.05, 0.8),),
            targets=targets,
            y0=y0,
            preprocess=lambda p: p,
            run_kwargs=dict(t0=0.0, t1=80.0, dt=1.0, solver="euler"),
        )


def test_hierarchical_sd_is_a_sample_site() -> None:
    cm, y0, _targets, params, values = _sir_fixture()
    times = np.array([0.0, 20.0, 40.0, 60.0])
    sd_prior = Uniform("obs_sd", 5.0, 20.0)
    state = Property("state", ("S", "I", "R"))
    targets = TargetSet(
        targets=(
            Target(
                key="I",
                times=times,
                values=values,
                quantity=Compartments(where=state["I"]),
                likelihood=NormalLikelihood(sd=sd_prior),
            ),
        )
    )
    bm = BayesianModel(
        cm,
        params,
        priors=(Uniform("infection", 0.05, 0.8),),
        targets=targets,
        y0=y0,
        run_kwargs=dict(t0=0.0, t1=80.0, dt=1.0, solver="euler"),
    )
    assert bm.prior_names() == ("infection", "obs_sd")


def test_log_density_jittable_and_map_near_truth() -> None:
    cm, y0, targets, params, _ = _sir_fixture(infection=0.35)
    bm = BayesianModel(
        cm,
        params,
        priors=(Uniform("infection", 0.05, 0.8),),
        targets=targets,
        y0=y0,
        run_kwargs=dict(t0=0.0, t1=80.0, dt=1.0, solver="euler"),
    )
    _pot, _post, z = bm._ensure_potential()
    jitted = jax.jit(bm.log_density)
    np.testing.assert_allclose(float(jitted(z)), float(bm.log_density(z)), rtol=1e-5)
    mapped = bm.find_map(steps=80, seed=0)
    assert abs(float(mapped["infection"]) - 0.35) < 0.05


def test_solver_failure_penalises_log_density() -> None:
    cm, y0, targets, params, _ = _sir_fixture()
    healthy = BayesianModel(
        cm,
        params,
        priors=(Uniform("infection", 0.05, 0.8),),
        targets=targets,
        y0=y0,
        run_kwargs=dict(t0=0.0, t1=80.0, dt=1.0, solver="euler"),
    )
    failed = BayesianModel(
        cm,
        params,
        priors=(Uniform("infection", 0.05, 0.8),),
        targets=targets,
        y0=y0,
        run_kwargs=dict(
            t0=0.0,
            t1=80.0,
            dt=1.0,
            solver="dopri5",
            max_steps=2,
            rtol=1e-6,
            atol=1e-6,
        ),
    )
    z_h = healthy._ensure_potential()[2]
    z_f = failed._ensure_potential()[2]
    assert float(failed.log_density(z_f)) < float(healthy.log_density(z_h)) - 1e20
    assert not bool(failed._run(failed.merge_params({"infection": 0.35})).solver.ok)


@pytest.mark.slow
def test_nuts_recovers_infection_in_95_interval() -> None:
    cm, y0, targets, params, _ = _sir_fixture(infection=0.35)
    bm = BayesianModel(
        cm,
        params,
        priors=(Uniform("infection", 0.05, 0.8),),
        targets=targets,
        y0=y0,
        run_kwargs=dict(t0=0.0, t1=80.0, dt=1.0, solver="euler"),
    )
    idata = bm.sample(
        kind="nuts",
        num_warmup=80,
        num_samples=80,
        num_chains=1,
        seed=0,
        progress_bar=False,
    )
    samples = np.asarray(idata.posterior["infection"]).reshape(-1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    assert lo <= 0.35 <= hi


@pytest.mark.slow
def test_aies_recovers_infection_in_95_interval() -> None:
    cm, y0, targets, params, _ = _sir_fixture(infection=0.35)
    bm = BayesianModel(
        cm,
        params,
        priors=(Uniform("infection", 0.05, 0.8),),
        targets=targets,
        y0=y0,
        run_kwargs=dict(t0=0.0, t1=80.0, dt=1.0, solver="euler"),
    )
    idata = bm.sample(
        kind="aies",
        num_warmup=40,
        num_samples=40,
        num_chains=4,
        seed=1,
        progress_bar=False,
    )
    samples = np.asarray(idata.posterior["infection"]).reshape(-1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    assert lo <= 0.35 <= hi


@pytest.mark.slow
def test_sa_recovers_infection_in_95_interval() -> None:
    cm, y0, targets, params, _ = _sir_fixture(infection=0.35)
    bm = BayesianModel(
        cm,
        params,
        priors=(Uniform("infection", 0.05, 0.8),),
        targets=targets,
        y0=y0,
        run_kwargs=dict(t0=0.0, t1=80.0, dt=1.0, solver="euler"),
    )
    # SA wants a long warmup relative to NUTS; 500 is enough on this SIR.
    idata = bm.sample(
        kind="sa",
        num_warmup=500,
        num_samples=100,
        num_chains=1,
        seed=2,
        progress_bar=False,
    )
    samples = np.asarray(idata.posterior["infection"]).reshape(-1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    assert lo <= 0.35 <= hi


def _sir_with_outputs(
    *,
    infection: float = 0.35,
) -> tuple[Any, Any, TargetSet, dict[str, float], OutputSet]:
    cm, y0, targets, params, _ = _sir_fixture(infection=infection)
    state = Property("state", ("S", "I", "R"))
    outputs = OutputSet()
    outputs["I"] = Compartments(where=state["I"]).total()
    outputs["cum_I"] = outputs.ref("I").cumulative()
    return cm, y0, targets, params, outputs


def test_posterior_runs_quantiles_match_looped_draws() -> None:
    cm, y0, targets, params, outputs = _sir_with_outputs()
    bm = BayesianModel(
        cm,
        params,
        priors=(Uniform("infection", 0.05, 0.8),),
        targets=targets,
        outputs=outputs,
        y0=y0,
        run_kwargs=dict(t0=0.0, t1=80.0, dt=1.0, solver="euler"),
    )
    infections = np.linspace(0.2, 0.5, 20)
    draws = {"infection": infections}
    runs = bm.posterior_runs(
        draws,
        n=None,
        scenarios={"baseline": None},
        batch_size=8,
    )
    assert runs.samples("baseline", "I").shape == (20, 81)
    q = (0.1, 0.5, 0.9)
    got = runs.quantiles(q=q)["baseline"]
    stacked = []
    save = outputs.plan(SavePlan())
    for inf in infections:
        merged = bm.merge_params({"infection": float(inf)})
        result = cm.run(merged, y0, save=save, t0=0.0, t1=80.0, dt=1.0, solver="euler")
        scored = outputs.evaluate(result, merged)
        vals = scored["I"].values
        data = vals.data if hasattr(vals, "data") else vals
        stacked.append(np.asarray(data).reshape(-1))
    explicit = np.stack(stacked, axis=0)
    for qi, label in zip(q, [str(float(v)) for v in q], strict=True):
        expected = np.quantile(explicit, qi, axis=0)
        np.testing.assert_allclose(
            got[("I", label)].to_numpy(),
            expected,
            rtol=1e-5,
            atol=1e-5,
        )


def test_posterior_runs_differences_schema_and_scenario_params() -> None:
    cm, y0, targets, params, outputs = _sir_with_outputs()
    bm = BayesianModel(
        cm,
        params,
        priors=(Uniform("infection", 0.05, 0.8),),
        targets=targets,
        outputs=outputs,
        y0=y0,
        run_kwargs=dict(t0=0.0, t1=80.0, dt=1.0, solver="euler"),
    )
    draws = {"infection": np.linspace(0.25, 0.45, 12)}
    runs = bm.posterior_runs(
        draws,
        n=None,
        scenarios={
            "baseline": None,
            "stronger": Scenario(params={"infection": 0.6}),
        },
        batch_size=4,
    )
    diffs = runs.differences(
        ref="baseline",
        outputs={"I_averted": "I"},
        at=40.0,
        relative=True,
        q=(0.25, 0.5, 0.75),
    )
    assert "baseline" not in diffs
    frame = diffs["stronger"]
    assert list(frame.index) == ["0.25", "0.5", "0.75"]
    assert "I_averted" in frame.columns
    assert "I_averted_relative" in frame.columns
    # Scenario pins infection=0.6; baseline draws stay in [0.25, 0.45] — trajectories differ.
    assert float(np.abs(frame.loc["0.5", "I_averted"])) > 1.0
    base = runs.samples("baseline", "I")
    scen = runs.samples("stronger", "I")
    assert not np.allclose(base, scen)


def test_posterior_runs_accepts_plain_site_dict_only() -> None:
    """Candidates-compatible path: Mapping site arrays, not only InferenceData."""
    cm, y0, targets, params, outputs = _sir_with_outputs()
    bm = BayesianModel(
        cm,
        params,
        priors=(Uniform("infection", 0.05, 0.8),),
        targets=targets,
        outputs=outputs,
        y0=y0,
        run_kwargs=dict(t0=0.0, t1=80.0, dt=1.0, solver="euler"),
    )
    draws = {"infection": np.array([0.3, 0.35, 0.4])}
    runs = bm.posterior_runs(draws, n=2, seed=0, batch_size=2)
    assert runs.n == 2
    assert runs.samples("baseline", "I").shape[0] == 2
