"""Step 46: Posterior protocol, NumpyroPosterior, and workflow equivalence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from scipy import stats
from tests.gate_custom_posterior import (
    LHS_JOINT_SEED,
    LHS_M_JOINT,
    LHS_M_MAIN,
    LHS_MAIN_SEED,
    MCMC_CHAINS,
    MCMC_JOINT_SEED,
    MCMC_MAIN_SEED,
    MCMC_SAMPLES_PER_CHAIN,
    MCMC_WARMUP_MIN,
    MCMC_WARMUP_START,
    N_OPT_STARTS,
    OPT_CHUNK,
    OPT_LR,
    OPT_MAX_STEPS,
    OPT_SEED,
    build_gate_models,
    recovery_from_joint_params,
)

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
            c_b_post = post.constrain(batched)
            c_b_bm = bm.constrain(batched)
            u_b_post = post.unconstrain(c_b_post)
            u_b_bm = bm.unconstrain(c_b_bm)
            for name in bm.prior_names():
                np.testing.assert_allclose(c_b_bm[name], c_b_post[name], atol=1e-10)
                np.testing.assert_allclose(u_b_bm[name], u_b_post[name], atol=1e-10)

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
        c_bm2 = bm2.constrain(z0)
        c_post2 = post2.constrain(z0)
        u_bm2 = bm2.unconstrain(c_bm2)
        u_post2 = post2.unconstrain(c_post2)
        for name in bm2.prior_names():
            np.testing.assert_allclose(c_bm2[name], c_post2[name], atol=1e-10)
            np.testing.assert_allclose(u_bm2[name], u_post2[name], atol=1e-10)
        lhs_bm2 = wf.lhs(bm2, 8, seed=1)
        lhs_post2 = wf.lhs(post2, 8, seed=1)
        for name in bm2.prior_names():
            np.testing.assert_allclose(lhs_bm2.z[name], lhs_post2.z[name], atol=1e-10)
            np.testing.assert_allclose(lhs_bm2.params[name], lhs_post2.params[name], atol=1e-10)
        start2 = wf.Candidates.from_z(bm2, {k: v[:2] for k, v in lhs_bm2.z.items()})
        opt_bm2 = wf.optimize(
            bm2,
            start2,
            method=wf.Optax(optax.adam(0.05)),
            max_steps=40,
            seed=0,
            tuning=wf.AutoTune(lr_grid=(0.05,), patience=10**6),
        )
        opt_post2 = wf.optimize(
            post2,
            start2,
            method=wf.Optax(optax.adam(0.05)),
            max_steps=40,
            seed=0,
            tuning=wf.AutoTune(lr_grid=(0.05,), patience=10**6),
        )
        for name in bm2.prior_names():
            np.testing.assert_allclose(
                opt_bm2.best_params[name], opt_post2.best_params[name], atol=1e-10
            )
        np.testing.assert_allclose(opt_bm2.loss_trace, opt_post2.loss_trace, atol=1e-10)


def _hierarchical_numpyro_model(centred: bool = False, *, tau_scale: float = 1.0) -> Any:
    def model() -> None:
        mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
        tau = numpyro.sample("tau", dist.HalfNormal(tau_scale))
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

    m_lhs = 4
    mu_vals = np.asarray(lhs_a.params["mu"], float)
    tau_vals = np.asarray(lhs_a.params["tau"], float)
    assert len(np.unique(mu_vals)) == m_lhs
    assert len(np.unique(tau_vals)) == m_lhs

    scored = wf.evaluate(post, lhs_a, batch_size=2)
    assert scored.log_density is not None
    assert np.all(np.isfinite(np.asarray(scored.log_density)))
    if scored.ok is not None:
        assert np.all(np.asarray(scored.ok))

    start = wf.Candidates.from_z(post, {k: v[:2] for k, v in scored.z.items()})
    opt_run = wf.optimize(
        post,
        start,
        method=wf.Optax(optax.adam(0.1)),
        max_steps=5,
        seed=0,
        tuning=wf.AutoTune(lr_grid=(0.1,), patience=10**6),
    )
    assert np.all(np.isfinite(opt_run.loss_trace))

    init = scored.init_params(2, seed=0)
    mcmc = MCMC(NUTS(post.numpyro_model()), num_warmup=10, num_samples=10, num_chains=2)
    run = wf.sample_until(
        mcmc,
        wf.StopRule(rhat=None, ess=None, max_samples=10),
        init_params=init,
        seed=0,
    )
    cand = run.candidates(post)
    assert cand.sites == post.prior_names()

    centred_post = NumpyroPosterior(_hierarchical_numpyro_model(centred=True, tau_scale=0.05))
    assert tuple(p.name for p in centred_post.site_priors()) == ("mu", "tau")
    lhs_c = wf.lhs(centred_post, m_lhs, seed=10)
    assert lhs_c.history[0].settings["predictive_sites"] == ["theta"]
    mu_c = np.asarray(lhs_c.params["mu"], float)
    theta_c = np.asarray(lhs_c.params["theta"], float)
    for i in range(m_lhs):
        np.testing.assert_allclose(theta_c[i].mean(), mu_c[i], atol=0.15)


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
    with handlers.trace() as tr, handlers.seed(rng_seed=0), handlers.condition(data=draws):
        bm.numpyro_model()()
    assert "likelihood" in tr
    traced_ll = float(tr["likelihood"]["fn"].log_factor)
    np.testing.assert_allclose(ll, traced_ll, rtol=0, atol=1e-9)
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


def _gate_optimize_method() -> wf.Optax:
    return wf.Optax(optax.adam(OPT_LR))


_GATE_NOTEBOOK = Path("examples/notebooks/28-calibration-custom-posterior.ipynb")
_GATE_SYNC_CONSTANTS = (
    "DATA_SEED",
    "K_OUTBREAKS",
    "N_DUR",
    "TRUE_INFECTION",
    "TRUE_MU",
    "TRUE_TAU",
    "MU_PRIOR_LOC",
    "MU_PRIOR_SCALE",
    "TAU_PRIOR_SCALE",
    "NB_DISPERSION",
    "POP_SIZE",
    "OPT_SEED",
    "LHS_MAIN_SEED",
    "LHS_JOINT_SEED",
    "MCMC_MAIN_SEED",
    "MCMC_JOINT_SEED",
    "MCMC_SAMPLES_PER_CHAIN",
    "MCMC_CHAINS",
    "MCMC_WARMUP_START",
    "MCMC_WARMUP_MIN",
    "OPT_MAX_STEPS",
    "OPT_CHUNK",
    "OPT_LR",
    "LHS_M_MAIN",
    "LHS_M_JOINT",
    "N_OPT_STARTS",
)
_ASSIGN_RE = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=\s*(.+)$")


def _gate_notebook_code_source() -> str:
    nb = json.loads(_GATE_NOTEBOOK.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            parts.append("".join(cell["source"]))
    return "\n".join(parts)


def _gate_notebook_assignments(source: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGN_RE.match(stripped)
        if match is None:
            continue
        name, rhs = match.group(1), match.group(2).strip()
        result[name] = rhs
    return result


def _eval_gate_notebook_rhs(rhs: str) -> object:
    return eval(rhs, {"__builtins__": {}}, {"np": np, "float": float})


def test_gate_notebook_constants_match_helper() -> None:
    import tests.gate_custom_posterior as gate_constants

    assigns = _gate_notebook_assignments(_gate_notebook_code_source())
    for name in _GATE_SYNC_CONSTANTS:
        assert name in assigns, f"missing {name} assignment in notebook 28"
        notebook_val = _eval_gate_notebook_rhs(assigns[name])
        helper_val = getattr(gate_constants, name)
        assert (
            notebook_val == helper_val
        ), f"{name}: notebook {notebook_val!r} != helper {helper_val!r}"


def test_gate_main_only_ridge_multistart() -> None:
    gate = build_gate_models()
    bm, truth = gate.bm, gate.truth
    design = wf.evaluate(bm, wf.lhs(bm, LHS_M_MAIN, seed=LHS_MAIN_SEED), batch_size=LHS_M_MAIN)
    starts = design.best(N_OPT_STARTS)
    run = wf.optimize(
        bm,
        starts,
        method=_gate_optimize_method(),
        chunk_steps=OPT_CHUNK,
        max_steps=OPT_MAX_STEPS,
        seed=OPT_SEED,
    )
    ends = run.candidates
    recovery = np.asarray(ends.params["recovery"], float)
    # Same optimiser budget collapses main-only ends to one wrong ridge point.
    assert np.ptp(recovery) < 0.01
    assert abs(float(recovery.mean()) - truth["recovery"]) > 0.15


def test_gate_joint_recovery_concentration() -> None:
    gate = build_gate_models()
    bm, joint, truth = gate.bm, gate.joint, gate.truth
    method = _gate_optimize_method()
    design_main = wf.evaluate(bm, wf.lhs(bm, LHS_M_MAIN, seed=LHS_MAIN_SEED), batch_size=LHS_M_MAIN)
    main_ends = wf.optimize(
        bm,
        design_main.best(N_OPT_STARTS),
        method=method,
        chunk_steps=OPT_CHUNK,
        max_steps=OPT_MAX_STEPS,
        seed=OPT_SEED,
    ).candidates
    rec_main = np.asarray(main_ends.params["recovery"], float)

    design_joint = wf.evaluate(
        joint, wf.lhs(joint, LHS_M_JOINT, seed=LHS_JOINT_SEED), batch_size=16
    )
    joint_ends = wf.optimize(
        joint,
        design_joint.best(N_OPT_STARTS),
        method=method,
        chunk_steps=OPT_CHUNK,
        max_steps=OPT_MAX_STEPS,
        seed=OPT_SEED,
    ).candidates
    rec_joint = recovery_from_joint_params(joint_ends.params)

    assert np.all(np.abs(rec_joint - truth["recovery"]) <= 0.05)
    assert abs(float(rec_main.mean()) - truth["recovery"]) > abs(
        float(rec_joint.mean()) - truth["recovery"]
    )

    warmup_rule = wf.WarmupRule(min_warmup=MCMC_WARMUP_MIN, rhat=1.15)
    stop = wf.StopRule(rhat=None, ess=None, max_samples=MCMC_SAMPLES_PER_CHAIN)

    def make_main_mcmc(num_warmup: int) -> MCMC:
        return MCMC(
            NUTS(bm.numpyro_model(), target_accept_prob=0.9),
            num_warmup=num_warmup,
            num_samples=MCMC_SAMPLES_PER_CHAIN,
            num_chains=MCMC_CHAINS,
        )

    def make_joint_mcmc(num_warmup: int) -> MCMC:
        return MCMC(
            NUTS(joint.numpyro_model(), target_accept_prob=0.9),
            num_warmup=num_warmup,
            num_samples=MCMC_SAMPLES_PER_CHAIN,
            num_chains=MCMC_CHAINS,
        )

    init_main = design_main.init_params(MCMC_CHAINS, seed=MCMC_MAIN_SEED)
    warm_main = wf.warmup_until(
        make_main_mcmc,
        MCMC_WARMUP_START,
        warmup_rule,
        init_params=init_main,
        seed=MCMC_MAIN_SEED,
    )
    run_main = wf.sample_until(warm_main.mcmc, stop, seed=MCMC_MAIN_SEED)
    rec_main_draws = np.asarray(run_main.samples["recovery"]).reshape(-1)

    init_joint = design_joint.init_params(MCMC_CHAINS, seed=MCMC_JOINT_SEED)
    warm_joint = wf.warmup_until(
        make_joint_mcmc,
        MCMC_WARMUP_START,
        warmup_rule,
        init_params=init_joint,
        seed=MCMC_JOINT_SEED,
    )
    run_joint = wf.sample_until(warm_joint.mcmc, stop, seed=MCMC_JOINT_SEED)
    mu = run_joint.samples["mu"].reshape(-1)
    tau = run_joint.samples["tau"].reshape(-1)
    z_main = run_joint.samples["z_main"].reshape(-1)
    rec_joint_draws = np.exp(mu + tau * z_main)

    width_main = np.quantile(rec_main_draws, 0.9) - np.quantile(rec_main_draws, 0.1)
    width_joint = np.quantile(rec_joint_draws, 0.9) - np.quantile(rec_joint_draws, 0.1)
    assert width_joint < 0.3 * width_main
    assert (
        np.quantile(rec_joint_draws, 0.1) <= truth["recovery"] <= np.quantile(rec_joint_draws, 0.9)
    )


def test_gate_bm_numpyro_posterior_log_density() -> None:
    gate = build_gate_models()
    bm = gate.bm
    post = NumpyroPosterior(bm.numpyro_model())
    rng = np.random.default_rng(99)
    with jax.experimental.enable_x64():
        for z in _random_z(bm, rng, 16):
            np.testing.assert_allclose(
                float(post.log_density(z)), float(bm.log_density(z)), atol=1e-10
            )


def test_numpyro_posterior_site_priors_rejects_non_scalar_and_duplicates() -> None:
    model = _hierarchical_numpyro_model(centred=False)
    dup = (NumpyroPrior("mu", dist.Normal(0.0, 1.0)), NumpyroPrior("mu", dist.Normal(0.0, 1.0)))
    with pytest.raises(ValueError, match="duplicate"):
        NumpyroPosterior(model, site_priors=dup)
    with pytest.raises(ValueError, match="not scalar"):
        NumpyroPosterior(model, site_priors=(NumpyroPrior("z", dist.Normal(0.0, 1.0)),))


def test_numpyro_posterior_auto_site_priors_bounded_model_runs() -> None:
    runs = {"n": 0}

    def model() -> None:
        runs["n"] += 1
        numpyro.sample("a", dist.Normal(0.0, 1.0))
        numpyro.sample("b", dist.Uniform(0.0, 1.0))

    NumpyroPosterior(model)
    assert runs["n"] <= 3
    runs2 = {"n": 0}

    def model2() -> None:
        runs2["n"] += 1
        numpyro.sample("a", dist.Normal(0.0, 1.0))

    NumpyroPosterior(model2, site_priors=(NumpyroPrior("a", dist.Normal(0.0, 1.0)),))
    assert runs2["n"] == 1


def _plate_only_model() -> Any:
    def model() -> None:
        with numpyro.plate("k", 3):
            numpyro.sample("z", dist.Normal(0.0, 1.0))

    return model


def test_prior_draws_with_empty_site_priors() -> None:
    post = NumpyroPosterior(_plate_only_model(), site_priors=())
    assert post.site_priors() == ()
    draws = wf.prior_draws(post, 6, seed=1)
    assert draws.history[0].settings["predictive_sites"] == ["z"]
    assert draws.params["z"].shape == (6, 3)
    assert draws.sites == ("z",)


def test_lhs_with_empty_site_priors_raises() -> None:
    post = NumpyroPosterior(_plate_only_model(), site_priors=())
    with pytest.raises(ValueError, match="prior_draws"):
        wf.lhs(post, 4, seed=0)
