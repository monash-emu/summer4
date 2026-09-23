"""WP10 §10.1 — priors, likelihoods, and TargetSet.log_likelihood."""

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
    Beta,
    Gamma,
    LogNormal,
    NegativeBinomial,
    NormalLikelihood,
    NormalPrior,
    Poisson,
    TruncatedNormal,
    Uniform,
    prior_sites,
    priors_from_frame,
)
from summer4.epi.calibration.likelihoods import AggregateHow, resolve_scale


class _Rates(NamedTuple):
    infection: float
    recovery: float


def _sir_run(
    *,
    infection: float = 0.3,
    recovery: float = 0.1,
    times: np.ndarray | None = None,
) -> tuple[Any, TargetSet, Any]:
    state = Property("state", ("S", "I", "R"))
    pmap = PropertyMap.from_property(state)
    refs = derived_refs(_Rates)
    model = FlowModel(pmap)
    model.add_flow(TransitionFlow("infection", state["S"], state["I"], refs.infection))
    model.add_flow(TransitionFlow("recovery", state["I"], state["R"], refs.recovery))
    cm = model.compile()
    y0 = PropertyData.wrap(pmap, np.array([999.0, 1.0, 0.0]))
    if times is None:
        times = np.array([0.0, 10.0, 20.0, 40.0])
    qty = Compartments(where=state["I"])
    params = _Rates(infection=infection, recovery=recovery)
    truth = cm.run(
        params,
        y0,
        t0=0.0,
        t1=40.0,
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
                likelihood=NormalLikelihood(sd=1.0),
            ),
        )
    )
    result = cm.run(
        params,
        y0,
        t0=0.0,
        t1=40.0,
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
        # Smoke: log_prob at a feasible point is finite.
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

    # Default aggregate is mean (estival).
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
    # Param-valued scale
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
