"""Design stages: Latin hypercube and independent prior draws."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from summer4.epi.calibration.workflow.candidates import Candidates, StageRecord


def _unit_lhs(m: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube in the unit cube: one hit per stratum per dimension."""
    u = np.empty((m, d), dtype=np.float64)
    for j in range(d):
        # stratum i gets a uniform draw in [i/m, (i+1)/m), then permute rows
        strata = (np.arange(m, dtype=np.float64) + rng.random(m)) / float(m)
        u[:, j] = strata[rng.permutation(m)]
    return u


def _candidates_from_unit(
    bm: Any,
    u: np.ndarray,
    *,
    stage: str,
    settings: dict[str, Any],
    seconds: float,
) -> Candidates:
    sites = tuple(bm.prior_names())
    if u.ndim != 2 or u.shape[1] != len(sites):
        raise ValueError(
            f"{stage}: unit array shape {u.shape} does not match " f"{len(sites)} prior sites."
        )
    by_name = {p.name: p for p in bm.site_priors()}
    params: dict[str, Any] = {}
    for j, name in enumerate(sites):
        params[name] = by_name[name].icdf(u[:, j])
    z = {k: np.asarray(v) for k, v in bm.unconstrain(params).items()}
    params_arr = {k: np.asarray(v) for k, v in params.items()}
    return Candidates(
        sites=sites,
        z=z,
        params=params_arr,
        history=(StageRecord(stage=stage, settings=settings, seconds=seconds),),
    )


def lhs(bm: Any, m: int, *, seed: int = 0) -> Candidates:
    """Latin-hypercube design of ``m`` points over the model's priors.

    Host-side: strata in the unit cube (numpy permutation + jitter; no
    ``scipy.qmc``), pushed through each site's :meth:`~Prior.icdf`, then
    :meth:`~BayesianModel.unconstrain`. Includes hierarchical scale sites.
    """
    m_int = int(m)
    if m_int < 1:
        raise ValueError(f"lhs expects m >= 1, got {m_int}.")
    sites = tuple(bm.prior_names())
    if not sites:
        raise ValueError("lhs requires at least one prior site.")
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    u = _unit_lhs(m_int, len(sites), rng)
    return _candidates_from_unit(
        bm,
        u,
        stage="lhs",
        settings={"m": m_int, "seed": int(seed)},
        seconds=time.perf_counter() - t0,
    )


def prior_draws(bm: Any, m: int, *, seed: int = 0) -> Candidates:
    """``m`` independent prior draws (i.i.d. unit uniforms through ``icdf``)."""
    m_int = int(m)
    if m_int < 1:
        raise ValueError(f"prior_draws expects m >= 1, got {m_int}.")
    sites = tuple(bm.prior_names())
    if not sites:
        raise ValueError("prior_draws requires at least one prior site.")
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    u = rng.random((m_int, len(sites)))
    return _candidates_from_unit(
        bm,
        u,
        stage="prior_draws",
        settings={"m": m_int, "seed": int(seed)},
        seconds=time.perf_counter() - t0,
    )


__all__ = ["lhs", "prior_draws"]
