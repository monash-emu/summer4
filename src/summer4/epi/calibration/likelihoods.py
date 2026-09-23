"""Observation likelihoods attached to :class:`~summer4.results.targets.Target`.

Requires the ``calibration`` extra (``numpyro``) for :meth:`log_prob`. A
likelihood scale may be a float, a :func:`~summer4.flows.rates.Param` /
:class:`~summer4.flows.rates.FieldRef`, or a
:class:`~summer4.epi.calibration.priors.Prior` (hierarchical target sd).

Per-time log-densities default to a **mean** (estival's default when no time
weights are set). Pass ``aggregate="sum"`` for a sum instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from summer4.enums import coerce_strenum
from summer4.epi.calibration.priors import Prior
from summer4.flows.rates import FieldRef, _lookup_path


class AggregateHow(StrEnum):
    """How to collapse per-observation log-densities into a scalar."""

    MEAN = "mean"
    SUM = "sum"


AggregateHowArg = AggregateHow | Literal["mean", "sum"]


def _require_numpyro() -> Any:
    try:
        import numpyro.distributions as dist  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "Likelihoods require the calibration extra: pip install summer4[calibration]"
        ) from exc
    return dist


def resolve_scale(spec: Any, params: Any) -> Any:
    """Resolve a float, ``Param``/``FieldRef``, or prior-valued scale from ``params``."""
    if isinstance(spec, Prior):
        return _lookup_path(params, (spec.name,))
    if isinstance(spec, FieldRef):
        return _lookup_path(params, spec.path)
    if isinstance(spec, (int, float, np.floating)):
        return float(spec)
    # Traced JAX scalar / array already in hand.
    return spec


def _aggregate_ll(ll: Any, how: AggregateHowArg) -> Any:
    import jax.numpy as jnp

    mode = how if isinstance(how, AggregateHow) else _coerce_aggregate(how)
    if mode is AggregateHow.MEAN:
        return jnp.mean(ll)
    if mode is AggregateHow.SUM:
        return jnp.sum(ll)
    raise ValueError(f"Unknown likelihood aggregate {how!r}.")


def _coerce_aggregate(value: AggregateHowArg) -> AggregateHow:
    return coerce_strenum(AggregateHow, value, what="likelihood aggregate")


@dataclass(frozen=True, slots=True)
class Normal:
    """Independent normal observations with shared scale ``sd``.

    ``sd`` may be a float, a :func:`~summer4.flows.rates.Param`, or a
    :class:`~summer4.epi.calibration.priors.Prior` whose ``name`` is looked up
    in ``params`` at :meth:`log_prob` time (Kiribati hierarchical target sd).

    ``aggregate`` collapses per-time log-densities: ``"mean"`` (default, like
    estival) or ``"sum"``.
    """

    sd: Any
    aggregate: AggregateHowArg = AggregateHow.MEAN

    def __post_init__(self) -> None:
        object.__setattr__(self, "aggregate", _coerce_aggregate(self.aggregate))

    @classmethod
    def from_tolerance(
        cls,
        data: NDArray[Any] | list[float],
        tol_pct: float,
        *,
        aggregate: AggregateHowArg = AggregateHow.MEAN,
    ) -> Normal:
        """Build ``sd = (tol_pct/100) * mean(|data|) / 1.96``.

        Reproduces the usual ``get_normal_target`` rule: a ``tol_pct`` percent
        relative half-width of a Normal 95% interval around the mean.
        """
        arr = np.asarray(data, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError("Normal.from_tolerance requires non-empty data.")
        mean = float(np.mean(np.abs(arr)))
        if mean == 0.0:
            raise ValueError("Normal.from_tolerance: mean absolute value is zero.")
        sd = (float(tol_pct) / 100.0) * mean / 1.96
        return cls(sd=sd, aggregate=aggregate)

    def log_prob(self, observed: Any, predicted: Any, params: Any) -> Any:
        dist = _require_numpyro()
        import jax.numpy as jnp

        sd = resolve_scale(self.sd, params)
        obs = jnp.asarray(observed)
        pred = jnp.asarray(predicted)
        # validate_args=False: a failed solve can hand NaNs here; BayesianModel
        # replaces the scalar with -inf when result.solver.ok is false.
        return _aggregate_ll(
            dist.Normal(pred, sd, validate_args=False).log_prob(obs), self.aggregate
        )


@dataclass(frozen=True, slots=True)
class Poisson:
    """Independent Poisson observations with rate ``predicted``."""

    aggregate: AggregateHowArg = AggregateHow.MEAN

    def __post_init__(self) -> None:
        object.__setattr__(self, "aggregate", _coerce_aggregate(self.aggregate))

    def log_prob(self, observed: Any, predicted: Any, params: Any) -> Any:
        del params  # Poisson has no free scale.
        dist = _require_numpyro()
        import jax.numpy as jnp

        obs = jnp.asarray(observed)
        pred = jnp.asarray(predicted)
        return _aggregate_ll(dist.Poisson(pred, validate_args=False).log_prob(obs), self.aggregate)


@dataclass(frozen=True, slots=True)
class NegativeBinomial:
    """Negative binomial (numpyro ``NegativeBinomial2``) with concentration ``dispersion``.

    ``dispersion`` is numpyro's concentration (``var = mean + mean**2 / concentration``).
    It may be a float, a ``Param``, or a prior, same as :class:`Normal.sd`.
    Matches estival's ``NegativeBinomialTarget`` mean/variance shape.
    """

    dispersion: Any
    aggregate: AggregateHowArg = AggregateHow.MEAN

    def __post_init__(self) -> None:
        object.__setattr__(self, "aggregate", _coerce_aggregate(self.aggregate))

    def log_prob(self, observed: Any, predicted: Any, params: Any) -> Any:
        dist = _require_numpyro()
        import jax.numpy as jnp

        concentration = resolve_scale(self.dispersion, params)
        obs = jnp.asarray(observed)
        pred = jnp.asarray(predicted)
        return _aggregate_ll(
            dist.NegativeBinomial2(pred, concentration, validate_args=False).log_prob(obs),
            self.aggregate,
        )


def prior_sites(likelihood: Any) -> tuple[Prior, ...]:
    """Priors nested in a likelihood scale (empty if the scale is fixed)."""
    for attr in ("sd", "dispersion"):
        value = getattr(likelihood, attr, None)
        if isinstance(value, Prior):
            return (value,)
    return ()


__all__ = [
    "AggregateHow",
    "NegativeBinomial",
    "Normal",
    "Poisson",
    "prior_sites",
    "resolve_scale",
]
