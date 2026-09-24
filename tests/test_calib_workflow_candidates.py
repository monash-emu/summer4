"""Step 24 — Candidates, LHS design, batched evaluate, best, save/load."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

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
    Beta,
    Gamma,
    LogNormal,
    NormalLikelihood,
    NormalPrior,
    TruncatedNormal,
    Uniform,
)
from summer4.epi.calibration import (
    workflow as wf,
)


class _Rates(NamedTuple):
    infection: float
    recovery: float


def _sir_bm(
    *,
    infection: float = 0.35,
    recovery: float = 0.1,
    times: np.ndarray | None = None,
    sd: float = 5.0,
) -> tuple[BayesianModel, dict[str, float], Any]:
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
    bm = BayesianModel(
        cm,
        {"recovery": recovery},
        priors=(Uniform("infection", 0.05, 1.0),),
        targets=targets,
        y0=y0,
        run_kwargs={"t0": 0.0, "t1": 80.0, "dt": 1.0, "solver": "euler"},
    )
    return bm, params, y0


def test_prior_icdf_round_trips_numpyro_cdf() -> None:
    """icdf then numpyro cdf recovers the unit probability (where cdf exists)."""
    cases: list[tuple[Any, float]] = [
        (Uniform("u", 0.0, 2.0), 0.3),
        (NormalPrior("n", 0.0, 1.0), 0.7),
        (LogNormal("ln", 0.0, 0.5), 0.4),
        (TruncatedNormal("tn", 0.0, 1.0, low=-1.0, high=2.0), 0.55),
        (Beta("b", 2.0, 5.0), 0.25),
        (Gamma("g", 2.0, rate=0.5), 0.6),
    ]
    checked = 0
    for prior, u in cases:
        x = float(np.asarray(prior.icdf(u)).reshape(()))
        d = prior.to_numpyro()
        try:
            recovered = float(d.cdf(jnp.asarray(x)))
        except NotImplementedError:
            assert np.isfinite(float(d.log_prob(jnp.asarray(x))))
            continue
        np.testing.assert_allclose(recovered, u, atol=1e-5, rtol=1e-4)
        checked += 1
    assert checked >= 3  # at least Normal / LogNormal / Beta usually implement cdf


def test_icdf_respects_bounds() -> None:
    u = np.linspace(0.01, 0.99, 50)
    prior = Uniform("u", 1.0, 3.0)
    x = prior.icdf(u)
    assert np.all(x >= 1.0) and np.all(x <= 3.0)
    beta = Beta("b", 2.0, 2.0)
    bx = beta.icdf(u)
    assert np.all(bx >= 0.0) and np.all(bx <= 1.0)


def test_lhs_hits_each_stratum_once() -> None:
    bm, _params, _y0 = _sir_bm()
    m = 32
    c = wf.lhs(bm, m, seed=7)
    assert len(c) == m
    # Reconstruct unit probs via the Uniform prior's linear map.
    prior = bm.site_priors()[0]
    assert isinstance(prior, Uniform)
    u = (np.asarray(c.params["infection"]) - prior.lo) / (prior.hi - prior.lo)
    strata = np.floor(u * m).astype(int)
    strata = np.clip(strata, 0, m - 1)
    assert sorted(strata.tolist()) == list(range(m))


def test_constrain_unconstrain_round_trip() -> None:
    bm, params, _y0 = _sir_bm()
    p = {"infection": jnp.asarray([0.2, 0.5, 0.8])}
    z = bm.unconstrain(p)
    recovered = bm.constrain(z)
    np.testing.assert_allclose(
        np.asarray(recovered["infection"]), np.asarray(p["infection"]), rtol=1e-5
    )


def test_evaluate_matches_loop() -> None:
    bm, _params, _y0 = _sir_bm()
    c = wf.lhs(bm, 8, seed=0)
    scored = wf.evaluate(bm, c, batch_size=3)
    assert scored.log_density is not None and scored.ok is not None
    expected = np.array(
        [float(bm.log_density({k: v[i] for k, v in c.z.items()})) for i in range(len(c))]
    )
    np.testing.assert_allclose(np.asarray(scored.log_density), expected, rtol=1e-5)


def test_evaluate_marks_failed_solve() -> None:
    """A Diffrax solve that exhausts max_steps yields ok=False."""
    bm, _params, _y0 = _sir_bm()
    bm_fail = BayesianModel(
        bm.compiled,
        bm.fixed_params,
        priors=bm.priors,
        targets=bm.targets,
        y0=bm.y0,
        run_kwargs={
            "t0": 0.0,
            "t1": 80.0,
            "dt": 1.0,
            "solver": "tsit5",
            "max_steps": 1,
        },
    )
    c = wf.lhs(bm_fail, 4, seed=1)
    scored = wf.evaluate(bm_fail, c, batch_size=2)
    assert scored.ok is not None and scored.log_density is not None
    assert not bool(np.all(np.asarray(scored.ok)))
    failed_ld = np.asarray(scored.log_density)[np.logical_not(np.asarray(scored.ok))]
    assert np.all(failed_ld < -1e29)


def test_evaluate_jaxpr_independent_of_m() -> None:
    bm, _params, _y0 = _sir_bm()
    c8 = wf.lhs(bm, 8, seed=0)
    c32 = wf.lhs(bm, 32, seed=0)
    # Warm potential.
    _ = bm.log_density({k: v[0] for k, v in c8.z.items()})

    def mapped(z: dict[str, Any]) -> Any:
        return jax.lax.map(bm.log_density, z, batch_size=4)

    z8 = {k: jnp.asarray(v) for k, v in c8.z.items()}
    z32 = {k: jnp.asarray(v) for k, v in c32.z.items()}
    jp8 = jax.make_jaxpr(mapped)(z8)
    jp32 = jax.make_jaxpr(mapped)(z32)
    assert len(jp8.jaxpr.eqns) == len(jp32.jaxpr.eqns)


def test_best_ordering_and_excludes_failed() -> None:
    bm, _params, _y0 = _sir_bm()
    c = wf.evaluate(bm, wf.lhs(bm, 16, seed=2), batch_size=8)
    top = c.best(4)
    assert len(top) == 4
    assert top.ok is not None and bool(np.all(np.asarray(top.ok)))
    ld = np.asarray(top.log_density)
    assert np.all(ld[:-1] >= ld[1:])


def test_save_load_round_trip(tmp_path: Path) -> None:
    bm, _params, _y0 = _sir_bm()
    scored = wf.evaluate(bm, wf.lhs(bm, 6, seed=3), batch_size=3)
    path = tmp_path / "cands.npz"
    scored.save(path)
    loaded = wf.Candidates.load(path)
    assert loaded.sites == scored.sites
    assert len(loaded) == len(scored)
    for name in scored.sites:
        np.testing.assert_array_equal(np.asarray(loaded.z[name]), np.asarray(scored.z[name]))
        np.testing.assert_array_equal(
            np.asarray(loaded.params[name]), np.asarray(scored.params[name])
        )
    np.testing.assert_array_equal(np.asarray(loaded.log_density), np.asarray(scored.log_density))
    np.testing.assert_array_equal(np.asarray(loaded.ok), np.asarray(scored.ok))
    assert len(loaded.history) == len(scored.history)


def test_prior_draws_length() -> None:
    bm, _params, _y0 = _sir_bm()
    c = wf.prior_draws(bm, 10, seed=0)
    assert len(c) == 10
    assert c.log_density is None
