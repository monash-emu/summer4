"""Step 50: multi-model calibration helpers (``CX46``, ``CX47``, ``CX48``)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive
from tests import gate_multi_model as gate_mod
from tests.gate_multi_model import (
    LHS_M,
    LHS_SEED,
    MCMC_CHAINS,
    MCMC_SAMPLES_PER_CHAIN,
    MCMC_SEED,
    MCMC_TARGET_ACCEPT,
    MCMC_WARMUP_MIN,
    MCMC_WARMUP_START,
    MODELS,
    N_OPT_STARTS,
    OPT_CHUNK,
    OPT_LR,
    OPT_MAX_STEPS,
    OPT_SEED,
    build_gate,
)

from summer4 import (
    Compartments,
    FlowModel,
    Param,
    Property,
    PropertyData,
    PropertyMap,
    SavePlan,
    SaveRequest,
    Target,
    TargetSet,
    TransitionFlow,
)
from summer4.epi.calibration import (
    BayesianModel,
    NegativeBinomial,
    NumpyroPosterior,
    Uniform,
)
from summer4.epi.calibration import workflow as wf

_TIMES = np.array([0.0, 2.0, 4.0])


def _enable_x64_ctx() -> Any:
    return jax.enable_x64(True) if hasattr(jax, "enable_x64") else jax.experimental.enable_x64()


def _tiny_bm(*, prior_recovery: bool = False, observed_shift: float = 0.0) -> BayesianModel:
    """A tiny euler SIR over four days; ``recovery`` is a prior site or fixed at 0.25."""
    state = Property("state", ("S", "I", "R"))
    pmap = PropertyMap.from_property(state)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], Param("infection")))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], Param("recovery")))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([990.0, 10.0, 0.0]))
    infectious = Compartments(where=state["I"])
    kwargs = {"t0": 0.0, "t1": 4.0, "dt": 0.5, "solver": "euler"}
    truth = cm.run(
        {"infection": 0.0004, "recovery": 0.3},
        y0,
        save=SavePlan(requests={"I": SaveRequest(infectious, ts=_TIMES)}),
        **kwargs,
    )
    values = np.asarray(truth["I"].values.data).reshape(-1) + observed_shift
    priors = [Uniform("infection", 0.0001, 0.002)]
    fixed: dict[str, float] = {}
    if prior_recovery:
        priors.append(Uniform("recovery", 0.02, 1.0))
    else:
        fixed["recovery"] = 0.25
    return BayesianModel(
        cm,
        fixed,
        priors=tuple(priors),
        targets=TargetSet(
            targets=(
                Target(
                    key="I",
                    times=_TIMES,
                    values=values,
                    quantity=infectious,
                    likelihood=NegativeBinomial(dispersion=20.0),
                ),
            )
        ),
        y0=y0,
        run_kwargs=kwargs,
    )


def _raises_text(bm: BayesianModel, params: Mapping[str, Any]) -> str:
    with pytest.raises(ValueError) as info:
        bm.check_params(params)
    return str(info.value)


# --- T1 ---------------------------------------------------------------------------------


def test_check_params_names() -> None:
    bm = _tiny_bm(prior_recovery=True)
    bm_fixed = _tiny_bm()
    bm.check_params({"infection": 0.001, "recovery": 0.3})
    # An extra fixed name is fine, and so is leaving a fixed name out.
    bm_fixed.check_params({"infection": 0.001, "recovery": 0.3})
    bm_fixed.check_params({"infection": 0.001})

    msg = _raises_text(bm, {})
    assert msg == "missing prior sites: ['infection', 'recovery']"

    msg = _raises_text(bm_fixed, {"infection": 0.001, "zeta": 1.0, "alpha": 2.0})
    assert msg == (
        "unknown parameter names: ['alpha', 'zeta'] "
        "(not a prior site or a fixed_params name of this model)"
    )
    assert "local_params" not in msg

    both = _raises_text(bm, {"recovry": 0.3})
    assert both == (
        "missing prior sites: ['infection', 'recovery']; "
        "unknown parameter names: ['recovry'] "
        "(not a prior site or a fixed_params name of this model)"
    )

    dotted = _raises_text(bm_fixed, {"infection": 0.001, "a.recovery": 0.3})
    assert dotted.endswith(
        ". Prefixed joint-model site names? Map them with local_params(draws, prefix)."
    )
    # An unknown name without a dot does not trigger the hint.
    assert "local_params" not in _raises_text(bm, {"x": 1.0})


# --- T2 / T3 ----------------------------------------------------------------------------


def test_strict_log_likelihood_matches_and_catches_misspelling() -> None:
    bm = _tiny_bm()
    full = {"infection": 0.001, "recovery": 0.25}
    assert float(bm.log_likelihood(full, strict=True)) == float(bm.log_likelihood(full))
    typo = {"infection": 0.001, "recovry": 0.9}
    with pytest.raises(ValueError, match="unknown parameter names"):
        bm.log_likelihood(typo, strict=True)
    lenient = float(bm.log_likelihood(typo))
    assert np.isfinite(lenient)
    assert lenient == pytest.approx(float(bm.log_likelihood({"infection": 0.001})))
    assert lenient == pytest.approx(float(bm.log_likelihood(full)))


def test_strict_checks_work_under_jit() -> None:
    bm = _tiny_bm(prior_recovery=True)

    def ll(infection: Any, recovery: Any) -> Any:
        params = {"infection": infection, "recovery": recovery}
        bm.check_params(params)
        return bm.log_likelihood(params, strict=True)

    jitted = jax.jit(ll)
    expected = bm.log_likelihood({"infection": 0.001, "recovery": 0.3}, strict=True)
    np.testing.assert_allclose(float(jitted(0.001, 0.3)), float(expected), rtol=1e-5)


# --- T4 / T5 / T6 -----------------------------------------------------------------------


def _trace(fn: Callable[[], Any]) -> tuple[Any, Any]:
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        out = fn()
    return tr, out


def _site_kinds(tr: Mapping[str, Any]) -> dict[str, str]:
    return {name: site["type"] for name, site in tr.items()}


def test_sample_params_prefixes_sites() -> None:
    bm = _tiny_bm(prior_recovery=True)
    tr, out = _trace(lambda: bm.sample_params("a."))
    assert [n for n, s in tr.items() if s["type"] == "sample"] == ["a.infection", "a.recovery"]
    assert list(out) == list(bm.prior_names()) == ["infection", "recovery"]
    assert not any(s["type"] == "deterministic" for s in tr.values())


def test_sample_params_given_and_record() -> None:
    bm = _tiny_bm(prior_recovery=True)
    value = jnp.asarray(0.31)
    tr, out = _trace(lambda: bm.sample_params("a.", given={"recovery": value}))
    kinds = _site_kinds(tr)
    assert kinds["a.infection"] == "sample"
    assert kinds["a.recovery"] == "deterministic"
    assert float(tr["a.recovery"]["value"]) == pytest.approx(0.31)
    assert list(out) == ["infection", "recovery"]
    assert float(out["recovery"]) == pytest.approx(0.31)

    tr2, out2 = _trace(lambda: bm.sample_params("a.", given={"recovery": value}, record=False))
    assert "a.recovery" not in tr2
    assert list(out2) == ["infection", "recovery"]

    # A fixed-name override is returned (after the prior sites) and recorded.
    fixed = _tiny_bm()
    tr3, out3 = _trace(lambda: fixed.sample_params("b.", given={"recovery": value}))
    assert list(out3) == ["infection", "recovery"]
    assert _site_kinds(tr3) == {"b.infection": "sample", "b.recovery": "deterministic"}

    with pytest.raises(ValueError) as info:
        _trace(lambda: bm.sample_params("a.", given={"zeta": 1.0, "alpha": 2.0}))
    assert str(info.value) == (
        "given names are not parameters of this model: ['alpha', 'zeta'] "
        "(not a prior site or a fixed_params name)"
    )


def test_same_prefix_collides_loudly() -> None:
    bm_a = _tiny_bm()
    bm_b = _tiny_bm()

    def model() -> None:
        bm_a.sample_params("m.")
        bm_b.sample_params("m.")

    with pytest.raises(Exception, match="unique names"):
        _trace(model)


# --- joint helpers ----------------------------------------------------------------------


def _joint_model(bms: Mapping[str, BayesianModel], *, use_factor: bool = True) -> Any:
    def joint() -> None:
        mu = numpyro.sample("mu", dist.Normal(jnp.log(0.25), 1.0))
        tau = numpyro.sample("tau", dist.HalfNormal(0.3))
        for m, bm in bms.items():
            z = numpyro.sample(f"z_{m}", dist.Normal(0.0, 1.0))
            params = bm.sample_params(f"{m}.", given={"recovery": jnp.exp(mu + tau * z)})
            if use_factor:
                bm.factor(m, params)
            else:
                numpyro.factor(m, bm.log_likelihood(params, strict=True))

    return joint


# --- T7 ---------------------------------------------------------------------------------


def test_factor_equals_strict_numpyro_factor() -> None:
    bms = {"a": _tiny_bm(), "b": _tiny_bm(observed_shift=3.0)}
    post_factor = NumpyroPosterior(_joint_model(bms, use_factor=True))
    post_manual = NumpyroPosterior(_joint_model(bms, use_factor=False))
    assert post_factor.prior_names() == post_manual.prior_names()
    rng = np.random.default_rng(5)
    with _enable_x64_ctx():
        for _ in range(8):
            z = {name: float(rng.normal()) for name in post_factor.prior_names()}
            np.testing.assert_allclose(
                float(post_factor.log_density(z)), float(post_manual.log_density(z)), rtol=1e-10
            )


# --- T8 ---------------------------------------------------------------------------------


def _joint_draws(n: int = 8) -> tuple[dict[str, Any], dict[str, BayesianModel]]:
    bms = {"a": _tiny_bm(), "b": _tiny_bm(observed_shift=3.0)}
    samples = Predictive(_joint_model(bms), num_samples=n)(jax.random.PRNGKey(1))
    # Predictive also returns the factor sites (empty arrays); MCMC samples do not.
    return {k: np.asarray(v) for k, v in samples.items() if k not in bms}, bms


def test_local_params_round_trip_and_errors() -> None:
    flat, bms = _joint_draws(8)
    shaped = {k: v.reshape(2, 4, *v.shape[1:]) for k, v in flat.items()}
    local = bms["a"].local_params(shaped, "a.")
    assert list(local) == ["infection", "recovery"]  # priors, then the recorded fixed override
    assert local["infection"].shape == (2, 4)
    np.testing.assert_array_equal(local["infection"], shaped["a.infection"])
    np.testing.assert_array_equal(local["recovery"], shaped["a.recovery"])
    # Other models' sites, hyperparameters and plate offsets are ignored.
    other = bms["b"].local_params(shaped, "b.")
    np.testing.assert_array_equal(other["infection"], shaped["b.infection"])
    assert list(other) == ["infection", "recovery"]  # local names only, no prefixed sites

    broken = {k: v for k, v in shaped.items() if k != "a.infection"}
    with pytest.raises(ValueError) as info:
        bms["a"].local_params(broken, "a.")
    assert str(info.value) == (
        "draws lack sites for this model's priors: ['a.infection']. "
        "Sample them with sample_params(prefix), or record given values (record=True)."
    )
    # Fixed names absent from the draws are simply not returned.
    only_prior = {"a.infection": shaped["a.infection"]}
    assert list(bms["a"].local_params(only_prior, "a.")) == ["infection"]


# --- T9 / T10 / T11 ---------------------------------------------------------------------


def _idata(flat: Mapping[str, np.ndarray]) -> Any:
    import arviz as az

    n = next(iter(flat.values())).shape[0]
    posterior = {k: v.reshape(2, n // 2, *v.shape[1:]) for k, v in flat.items()}
    return az.from_dict({"posterior": posterior})


def test_posterior_runs_params_fn_matches_hand_built_mapping() -> None:
    flat, bms = _joint_draws(8)
    flat["offsets"] = np.random.default_rng(0).normal(size=(8, 3))  # a plate site
    idata = _idata(flat)
    bm = bms["a"]
    via_fn = bm.posterior_runs(
        idata, n=5, seed=2, params_fn=lambda d: bm.local_params(d, "a."), batch_size=4
    )
    hand = {"infection": flat["a.infection"], "recovery": flat["a.recovery"]}
    direct = bm.posterior_runs(hand, n=5, seed=2, batch_size=4)
    assert via_fn.n == direct.n == 5
    np.testing.assert_allclose(
        via_fn.samples("baseline", "I"), direct.samples("baseline", "I"), rtol=1e-5
    )
    assert set(via_fn.draws) == {"infection", "recovery"}


def test_posterior_runs_params_fn_checks_names_and_shapes() -> None:
    flat, bms = _joint_draws(8)
    idata = _idata(flat)
    bm = bms["a"]
    with pytest.raises(ValueError, match="missing prior sites"):
        bm.posterior_runs(idata, params_fn=lambda d: {"recovery": d["a.recovery"]})
    with pytest.raises(ValueError, match="unknown parameter names"):
        bm.posterior_runs(
            idata, params_fn=lambda d: {"infection": d["a.infection"], "recovry": d["a.recovery"]}
        )
    with pytest.raises(ValueError, match=re.escape("one scalar per draw; 'infection' has shape")):
        bm.posterior_runs(
            idata,
            n=None,
            params_fn=lambda d: {"infection": np.stack([d["a.infection"]] * 2, axis=1)},
        )


def test_posterior_runs_default_rejects_plate_sites() -> None:
    flat, bms = _joint_draws(8)
    flat["offsets"] = np.zeros((8, 3))
    with pytest.raises(ValueError, match="not scalar"):
        bms["a"].posterior_runs(_idata(flat))


# --- T12: gate --------------------------------------------------------------------------

_GATE_NOTEBOOK = Path("examples/notebooks/38-calibration-multi-model.ipynb")
_GATE_MODULE = Path("tests/gate_multi_model.py")


def _gate_notebook_code() -> str:
    nb = json.loads(_GATE_NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_gate_notebook_copies_settings_module() -> None:
    """The notebook's settings cell is the module body, verbatim."""
    body = _GATE_MODULE.read_text(encoding="utf-8").split("# --- constants", 1)[1]
    body = "# --- constants" + body
    assert body.strip() in _gate_notebook_code()


def test_gate_notebook_constants_match_module() -> None:
    code = _gate_notebook_code()
    for name in (
        "DATA_SEED",
        "TRUE_TAU",
        "NB_DISPERSION",
        "OPT_MAX_STEPS",
        "LHS_M",
        "MCMC_SAMPLES_PER_CHAIN",
        "MCMC_SEED",
    ):
        match = re.search(rf"^{name} = (.+)$", code, re.M)
        assert match is not None, name
        assert eval(match.group(1)) == getattr(gate_mod, name)  # noqa: S307


def _recovery_draws(run: Any) -> dict[str, np.ndarray]:
    return {m: np.asarray(run.samples[f"{m}.recovery"]).reshape(-1) for m in MODELS}


def _width(x: np.ndarray) -> float:
    return float(np.quantile(x, 0.9) - np.quantile(x, 0.1))


def test_gate_joint_pools_recovery() -> None:
    gate = build_gate()
    bm_a = gate.models["a"]
    joint = NumpyroPosterior(gate.joint_model)
    assert joint.prior_names() == (
        "mu",
        "tau",
        "z_a",
        "z_b",
        "log_recovery_cohort",
        "a.infection",
        "b.infection",
    )

    # Section 2: the lenient misspelled call used the fixed 0.25.
    truth_a = gate.truth["a.recovery"]
    lenient = float(bm_a.log_likelihood({"infection": 0.55, "recovry": truth_a}))
    correct = float(bm_a.log_likelihood({"infection": 0.55, "recovery": truth_a}))
    assert np.isfinite(lenient) and lenient != correct
    with pytest.raises(ValueError, match="recovry"):
        bm_a.log_likelihood({"infection": 0.55, "recovry": truth_a}, strict=True)

    warm_rule = wf.WarmupRule(min_warmup=MCMC_WARMUP_MIN, rhat=1.15)
    stop = wf.StopRule(rhat=None, ess=None, max_samples=MCMC_SAMPLES_PER_CHAIN)
    method = wf.Optax(optax.adam(OPT_LR))

    def fit(post: Any) -> Any:
        design = wf.evaluate(post, wf.lhs(post, LHS_M, seed=LHS_SEED), batch_size=16)
        starts = design.best(N_OPT_STARTS)
        opt = wf.optimize(
            post,
            starts,
            method=method,
            chunk_steps=OPT_CHUNK,
            max_steps=OPT_MAX_STEPS,
            seed=OPT_SEED,
        )

        def make_mcmc(num_warmup: int) -> MCMC:
            return MCMC(
                NUTS(post.numpyro_model(), target_accept_prob=MCMC_TARGET_ACCEPT),
                num_warmup=num_warmup,
                num_samples=MCMC_SAMPLES_PER_CHAIN,
                num_chains=MCMC_CHAINS,
            )

        warm = wf.warmup_until(
            make_mcmc,
            MCMC_WARMUP_START,
            warm_rule,
            init_params=opt.candidates.init_params(MCMC_CHAINS, seed=MCMC_SEED),
            seed=MCMC_SEED,
        )
        assert warm.converged, warm.progress
        return wf.sample_until(warm.mcmc, stop, seed=MCMC_SEED)

    run_joint = fit(joint)
    run_a = fit(bm_a)
    joint_rec = _recovery_draws(run_joint)
    for m in MODELS:
        lo, hi = np.quantile(joint_rec[m], [0.1, 0.9])
        assert lo <= gate.truth[f"{m}.recovery"] <= hi, m
    alone = np.asarray(run_a.samples["recovery"]).reshape(-1)
    assert _width(joint_rec["a"]) < 0.5 * _width(alone)
