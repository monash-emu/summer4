"""Bayesian calibration assembly: priors and target likelihoods (WP10).

Sampling (``BayesianModel``), MAP, and posterior runs land in later WP10 steps.
``numpyro`` / ``optax`` stay behind the ``calibration`` extra.
"""

from __future__ import annotations

from summer4.epi.calibration import likelihoods as likelihoods
from summer4.epi.calibration import priors as priors
from summer4.epi.calibration.likelihoods import (
    AggregateHow,
    NegativeBinomial,
    Poisson,
    prior_sites,
    resolve_scale,
)
from summer4.epi.calibration.likelihoods import Normal as NormalLikelihood
from summer4.epi.calibration.priors import (
    Beta,
    Gamma,
    LogNormal,
    TruncatedNormal,
    Uniform,
    priors_from_frame,
)
from summer4.epi.calibration.priors import Normal as NormalPrior

__all__ = [
    "AggregateHow",
    "Beta",
    "Gamma",
    "LogNormal",
    "NegativeBinomial",
    "NormalLikelihood",
    "NormalPrior",
    "Poisson",
    "TruncatedNormal",
    "Uniform",
    "likelihoods",
    "prior_sites",
    "priors",
    "priors_from_frame",
    "resolve_scale",
]
