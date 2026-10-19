"""Step 46: Posterior protocol, NumpyroPosterior, and workflow equivalence."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from scipy import stats

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
    Gamma,
    LogNormal,
    NormalLikelihood,
    NormalPrior,
    NumpyroPosterior,
    TruncatedNormal,
    Uniform,
)
from summer4.epi.calibration import (
    workflow as wf,
)
from summer4.epi.calibration.priors import NumpyroPrior, Prior


class _Rates(NamedTuple):
    infection: float
    recovery: float


def _sir_bm_three_priors() -> BayesianModel:
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
    truth = cm.run(
        {"infection": 0.35, "recovery": 0.12, "log_scale": 0.0},
        y0,
        t0=0.0,
        t1=80.0,
        dt=1.0,
        save=SavePlan(requests={"I": SaveRequest(qty, ts=times)}),
        solver="euler",
    )
    values = np.asarray(truth["I"].at_times(times).values.data).reshape(-1)
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
        {"log_scale": 0.0},
        priors=(
            Uniform("infection", 0.05, 0.8),
            LogNormal("recovery", -2.0, 0.5),
            NormalPrior("log_scale", 0.0, 1.0),
        ),
        targets=targets,
        y0=y0,
        run_kwargs={"t0": 0.0, "t1": 80.0, "dt": 1.0, "solver": "euler"},
    )


def _sir_bm_gamma_truncated() -> BayesianModel:
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
    truth = cm.run(
        {"infection": 0.35, "recovery": 0.1},
        y0,
        t0=0.0,
        t1=80.0,
        dt=1.0,
        save=SavePlan(requests={"I": SaveRequest(qty, ts=times)}),
        solver="euler",
    )
    values = np.asarray(truth["I"].at_times(times).values.data).reshape(-1)
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
        {},
        priors=(
            Gamma("infection", concentration=2.0, rate=4.0),
            TruncatedNormal("recovery", loc=0.1, scale=0.05, low=0.01, high=0.5),
        ),
        targets=targets,
        y0=y0,
        run_kwargs={"t0": 0.0, "t1": 80.0, "dt": 1.0, "solver": "euler"},
    )


def _random_z(bm: BayesianModel, rng: np.random.Generator, n: int) -> list[dict[str, Any]]:
    sites = bm.prior_names()
    out: list[dict[str, Any]] = []
    for _ in range(n):
        out.append({name: float(rng.normal()) for name in sites})
    return out


def test_numpyro_posterior_equivalent_to_bayesian_model() -> None:
    bm = _sir_bm_three_priors()
    post = NumpyroPosterior(bm.numpyro_model())
    assert post.prior_names() == bm.prior_names()
    rng = np.random.default_rng(42)
    with jax.experimental.enable_x64():
        for z in _random_z(bm, rng, 32):
            np.testing.assert_allclose(
                float(post.log_density(z)), float(bm.log_density(z)), atol=1e-10
            )
            np.testing.assert_allclose(
                float(post.potential_fn(z)), float(bm.potential_fn(z)), atol=1e-10
            )
            c_bm = bm.constrain(z)
            c_post = post.constrain(z)
            u_bm = bm.unconstrain(c_bm)
            u_post = post.unconstrain(c_post)
            for name in bm.prior_names():
                np.testing.assert_allclose(c_bm[name], c_post[name], atol=1e-10)
                np.testing.assert_allclose(u_bm[name], u_post[name], atol=1e-10)
            batched = {k: jnp.asarray([v, v]) for k, v in z.items()}
            c_b = post.constrain(batched)
            u_b = post.unconstrain({k: jnp.asarray([c_bm[k], c_bm[k]]) for k in c_bm})
            for name in bm.prior_names():
                np.testing.assert_allclose(c_b[name][0], c_b[name][1], atol=1e-10)
                np.testing.assert_allclose(u_b[name][0], u_b[name][1], atol=1e-10)

        lhs_bm = wf.lhs(bm, 16, seed=3)
        lhs_post = wf.lhs(post, 16, seed=3)
        for name in bm.prior_names():
            np.testing.assert_allclose(lhs_bm.z[name], lhs_post.z[name], atol=1e-10)
            np.testing.assert_allclose(lhs_bm.params[name], lhs_post.params[name], atol=1e-10)

        start = wf.Candidates.from_z(bm, {k: v[:2] for k, v in lhs_bm.z.items()})
        opt_bm = wf.optimize(
            bm,
            start,
            method=wf.Optax(optax.adam(0.05)),
            max_steps=40,
            seed=0,
            tuning=wf.AutoTune(lr_grid=(0.05,), patience=10**6),
        )
        opt_post = wf.optimize(
            post,
            start,
            method=wf.Optax(optax.adam(0.05)),
            max_steps=40,
            seed=0,
            tuning=wf.AutoTune(lr_grid=(0.05,), patience=10**6),
        )
        for name in bm.prior_names():
            np.testing.assert_allclose(
                opt_bm.best_params[name], opt_post.best_params[name], atol=1e-10
            )
        np.testing.assert_allclose(opt_bm.loss_trace, opt_post.loss_trace, atol=1e-10)

    bm2 = _sir_bm_gamma_truncated()
    post2 = NumpyroPosterior(bm2.numpyro_model(), site_priors=bm2.site_priors())
    with jax.experimental.enable_x64():
        z0 = bm2.init_point(1)
        np.testing.assert_allclose(
            float(post2.log_density(z0)), float(bm2.log_density(z0)), atol=1e-10
        )
        lhs_bm2 = wf.lhs(bm2, 8, seed=1)
        lhs_post2 = wf.lhs(post2, 8, seed=1)
        for name in bm2.prior_names():
            np.testing.assert_allclose(lhs_bm2.params[name], lhs_post2.params[name], atol=1e-10)


def _hierarchical_numpyro_model(centred: bool = False) -> Any:
    def model() -> None:
        mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
        tau = numpyro.sample("tau", dist.HalfNormal(1.0))
        with numpyro.plate("k", 5):
            if centred:
                numpyro.sample("theta", dist.Normal(mu, tau))
            else:
                z = numpyro.sample("z", dist.Normal(0.0, 1.0))
                numpyro.deterministic("theta", mu + tau * z)
        numpyro.sample("y", dist.Normal(mu, 1.0), obs=jnp.zeros(()))

    return model


def test_plate_noncentred_lhs_and_workflow() -> None:
    model = _hierarchical_numpyro_model(centred=False)
    post = NumpyroPosterior(model)
    assert post.prior_names() == ("mu", "tau", "z")
    assert post.site_shapes()["z"] == (5,)
    strat_names = tuple(p.name for p in post.site_priors())
    assert strat_names == ("mu", "tau")

    lhs_a = wf.lhs(post, 4, seed=10)
    lhs_b = wf.lhs(post, 4, seed=11)
    assert lhs_a.params["z"].shape == (4, 5)
    assert not np.allclose(lhs_a.params["z"], lhs_b.params["z"])
    assert lhs_a.history[0].settings["lhs_sites"] == ["mu", "tau"]
    assert lhs_a.history[0].settings["predictive_sites"] == ["z"]

    scored = wf.evaluate(post, lhs_a, batch_size=2)
    assert scored.log_density is not None
    start = wf.Candidates.from_z(post, {k: v[:2] for k, v in scored.z.items()})
    wf.optimize(
        post,
        start,
        method=wf.Optax(optax.adam(0.1)),
        max_steps=5,
        seed=0,
        tuning=wf.AutoTune(lr_grid=(0.1,), patience=10**6),
    )
    init = scored.init_params(2, seed=0)
    mcmc = MCMC(NUTS(post.numpyro_model()), num_warmup=10, num_samples=10, num_chains=2)
    run = wf.sample_until(
        mcmc,
        wf.StopRule(rhat=None, ess=None, max_samples=10),
        init_params=init,
        seed=0,
    )
    _ = run.candidates(post)

    centred = NumpyroPosterior(_hierarchical_numpyro_model(centred=True))
    assert tuple(p.name for p in centred.site_priors()) == ("mu", "tau")


def test_numpyro_prior_icdf_and_protocol() -> None:
    n = NumpyroPrior("x", dist.Normal(1.0, 2.0))
    u = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(n.icdf(u), stats.norm(loc=1.0, scale=2.0).ppf(u), rtol=1e-6)
    assert n.bounds() == (None, None)
    uni = NumpyroPrior("u", dist.Uniform(0.0, 2.0))
    np.testing.assert_allclose(uni.icdf(0.25), 0.5)
    beta = NumpyroPrior("b", dist.Beta(2.0, 2.0))
    assert beta.has_icdf is False
    assert isinstance(n, Prior)


def test_bayesian_model_log_likelihood() -> None:
    bm = _sir_bm_three_priors()
    draws = {"infection": 0.3, "recovery": 0.15, "log_scale": 0.1}
    ll = float(bm.log_likelihood(draws))
    assert ll == float(bm.log_likelihood(draws))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        bm.numpyro_model()()
    assert "likelihood" in tr
    with handlers.condition(data=draws):
        tr2 = handlers.trace(bm.numpyro_model()).get_trace()
    assert tr2["likelihood"]["infer"].get("is_auxiliary")
    z = bm.unconstrain(draws)
    ld_before = float(bm.log_density(z))
    z_bad = {**z, "infection": z["infection"] + 50.0}
    assert float(bm.log_density(z_bad)) < ld_before

    cm = bm.compiled
    y0 = PropertyData.wrap(
        PropertyMap.from_property(Property("state", ("S", "I", "R"))),
        np.array([999.0, 1.0, 0.0]),
    )
    failed = BayesianModel(
        cm,
        {"log_scale": 0.0},
        priors=bm.priors,
        targets=bm.targets,
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
    assert float(failed.log_likelihood(draws)) < -1.0e29

    def joint() -> None:
        inf = numpyro.sample("infection", dist.Uniform(0.05, 0.8))
        numpyro.factor(
            "main",
            bm.log_likelihood({"infection": inf, "recovery": 0.12, "log_scale": 0.0}),
        )

    with handlers.trace() as tr2, handlers.seed(rng_seed=0):
        joint()
    assert "main" in tr2


def test_posterior_isinstance() -> None:
    from summer4.epi.calibration.posterior import Posterior

    bm = _sir_bm_three_priors()
    assert isinstance(bm, Posterior)
    assert isinstance(NumpyroPosterior(bm.numpyro_model()), Posterior)
