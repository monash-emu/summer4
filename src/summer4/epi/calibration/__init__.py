"""Bayesian calibration assembly: priors, likelihoods, sampling, posterior runs."""

from __future__ import annotations

from summer4.epi.calibration import likelihoods as likelihoods
from summer4.epi.calibration import priors as priors
from summer4.epi.calibration import workflow as workflow
from summer4.epi.calibration.likelihoods import (
    AggregateHow,
    NegativeBinomial,
    Poisson,
    prior_sites,
    resolve_scale,
)
from summer4.epi.calibration.likelihoods import Normal as NormalLikelihood
from summer4.epi.calibration.model import BayesianModel, SampleKind
from summer4.epi.calibration.posterior import NumpyroPosterior, Posterior
from summer4.epi.calibration.posterior_runs import PosteriorRuns, Scenario
from summer4.epi.calibration.priors import (
    Beta,
    Gamma,
    LogNormal,
    NumpyroPrior,
    TruncatedNormal,
    Uniform,
    priors_from_frame,
)
from summer4.epi.calibration.priors import Normal as NormalPrior

__all__ = [
    "AggregateHow",
    "BayesianModel",
    "Beta",
    "Gamma",
    "LogNormal",
    "NegativeBinomial",
    "NormalLikelihood",
    "NormalPrior",
    "NumpyroPosterior",
    "NumpyroPrior",
    "Poisson",
    "Posterior",
    "PosteriorRuns",
    "SampleKind",
    "Scenario",
    "TruncatedNormal",
    "Uniform",
    "likelihoods",
    "prior_sites",
    "priors",
    "priors_from_frame",
    "resolve_scale",
    "workflow",
]
