"""Design stages: Latin hypercube and independent prior draws."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from summer4.epi.calibration.posterior import Posterior
from summer4.epi.calibration.priors import Prior
from summer4.epi.calibration.workflow.candidates import Candidates, StageRecord


def _unit_lhs(m: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube in the unit cube: one hit per stratum per dimension."""
    u = np.empty((m, d), dtype=np.float64)
    for j in range(d):
        # stratum i gets a uniform draw in [i/m, (i+1)/m), then permute rows
        strata = (np.arange(m, dtype=np.float64) + rng.random(m)) / float(m)
        u[:, j] = strata[rng.permutation(m)]
    return u


def _stratified_site_priors(post: Posterior) -> tuple[Prior, ...]:
    """``site_priors()`` restricted to ``prior_names()`` order."""
    by_name = {p.name: p for p in post.site_priors()}
    return tuple(by_name[name] for name in post.prior_names() if name in by_name)


def _params_from_predictive_only(
    post: Posterior,
    m: int,
    *,
    seed: int,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Draw all sites with ``Predictive`` when nothing is stratifiable."""
    from jax import random
    from numpyro.infer import Predictive  # type: ignore[import-untyped]

    all_sites = tuple(post.prior_names())
    pred = Predictive(
        post.numpyro_model(),
        num_samples=int(m),
        return_sites=all_sites,
    )
    drawn = pred(random.PRNGKey(int(seed)))
    params = {name: np.asarray(drawn[name]) for name in all_sites}
    return params, (), all_sites


def _params_from_strat_unit(
    post: Posterior,
    u: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Push stratified unit columns through ``icdf``; Predictive fills the rest."""
    all_sites = tuple(post.prior_names())
    strat = _stratified_site_priors(post)
    if u.ndim != 2 or u.shape[1] != len(strat):
        raise ValueError(
            f"unit array shape {u.shape} does not match {len(strat)} stratified prior sites."
        )
    params: dict[str, Any] = {}
    lhs_names: list[str] = []
    for j, prior in enumerate(strat):
        params[prior.name] = prior.icdf(u[:, j])
        lhs_names.append(prior.name)
    predictive_names = tuple(n for n in all_sites if n not in params)
    if predictive_names:
        import jax.numpy as jnp
        from jax import random
        from numpyro.infer import Predictive

        posterior_samples = {k: jnp.asarray(v) for k, v in params.items()}
        pred = Predictive(
            post.numpyro_model(),
            posterior_samples=posterior_samples,
            batch_ndims=1,
            return_sites=predictive_names,
        )
        drawn = pred(random.PRNGKey(int(seed)))
        for name in predictive_names:
            params[name] = np.asarray(drawn[name])
    return params, tuple(lhs_names), predictive_names


def _candidates_from_unit(
    post: Posterior,
    u: np.ndarray,
    *,
    stage: str,
    settings: dict[str, Any],
    seed: int,
    seconds: float,
) -> Candidates:
    sites = tuple(post.prior_names())
    if not sites:
        raise ValueError(f"{stage} requires at least one prior site.")
    params, lhs_sites, predictive_sites = _params_from_strat_unit(post, u, seed=seed)
    full_settings = {
        **settings,
        "lhs_sites": list(lhs_sites),
        "predictive_sites": list(predictive_sites),
    }
    z = {k: np.asarray(v) for k, v in post.unconstrain(params).items()}
    params_arr = {k: np.asarray(v) for k, v in params.items()}
    return Candidates(
        sites=sites,
        z=z,
        params=params_arr,
        history=(StageRecord(stage=stage, settings=full_settings, seconds=seconds),),
    )


def lhs(post: Posterior, m: int, *, seed: int = 0) -> Candidates:
    """Latin-hypercube design of ``m`` points over stratifiable prior sites.

    Host-side: strata in the unit cube (numpy permutation + jitter; no
    ``scipy.qmc``), pushed through each stratified site's :meth:`~Prior.icdf`.
    Sites in :meth:`~Posterior.prior_names` that are not in
    :meth:`~Posterior.site_priors` are filled with
    ``numpyro.infer.Predictive`` conditional on the stratified row (same
    ``seed`` as the stage). Then :meth:`~Posterior.unconstrain` builds ``z``.

    When :meth:`~Posterior.site_priors` is empty, use :func:`prior_draws` instead
    (full Predictive draws for every site).
    """
    m_int = int(m)
    if m_int < 1:
        raise ValueError(f"lhs expects m >= 1, got {m_int}.")
    strat = _stratified_site_priors(post)
    if not strat:
        raise ValueError(
            "lhs requires at least one site in site_priors(); "
            "when site_priors() is empty, use prior_draws instead."
        )
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    u = _unit_lhs(m_int, len(strat), rng)
    return _candidates_from_unit(
        post,
        u,
        stage="lhs",
        settings={"m": m_int, "seed": int(seed)},
        seed=int(seed),
        seconds=time.perf_counter() - t0,
    )


def prior_draws(post: Posterior, m: int, *, seed: int = 0) -> Candidates:
    """``m`` independent prior draws (i.i.d. unit uniforms through ``icdf``).

    Uncovered sites use the same :class:`~numpyro.infer.Predictive` fallback as
    :func:`lhs`. When :meth:`~Posterior.site_priors` is empty, every site is
    drawn from a single unconditional Predictive call.
    """
    m_int = int(m)
    if m_int < 1:
        raise ValueError(f"prior_draws expects m >= 1, got {m_int}.")
    strat = _stratified_site_priors(post)
    t0 = time.perf_counter()
    if not strat:
        sites = tuple(post.prior_names())
        if not sites:
            raise ValueError("prior_draws requires at least one prior site.")
        params, lhs_sites, predictive_sites = _params_from_predictive_only(
            post, m_int, seed=int(seed)
        )
        full_settings = {
            "m": m_int,
            "seed": int(seed),
            "lhs_sites": list(lhs_sites),
            "predictive_sites": list(predictive_sites),
        }
        z = {k: np.asarray(v) for k, v in post.unconstrain(params).items()}
        params_arr = {k: np.asarray(v) for k, v in params.items()}
        return Candidates(
            sites=sites,
            z=z,
            params=params_arr,
            history=(
                StageRecord(
                    stage="prior_draws",
                    settings=full_settings,
                    seconds=time.perf_counter() - t0,
                ),
            ),
        )
    rng = np.random.default_rng(int(seed))
    u = rng.random((m_int, len(strat)))
    return _candidates_from_unit(
        post,
        u,
        stage="prior_draws",
        settings={"m": m_int, "seed": int(seed)},
        seed=int(seed),
        seconds=time.perf_counter() - t0,
    )


__all__ = ["lhs", "prior_draws"]
