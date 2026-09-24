"""Priors for Bayesian calibration (WP10).

Requires the ``calibration`` extra (``numpyro``). Each prior is a frozen
dataclass naming a parameter, with :meth:`to_numpyro` for sampling sites and
:meth:`bounds` for unconstrained transforms. :meth:`icdf` is host-side
(``scipy.stats``) for Latin-hypercube and prior designs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


def _require_numpyro() -> Any:
    try:
        import numpyro.distributions as dist  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "Priors require the calibration extra: pip install summer4[calibration]"
        ) from exc
    return dist


def _require_scipy_stats() -> Any:
    try:
        from scipy import stats  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "Prior.icdf requires scipy (calibration extra): pip install summer4[calibration]"
        ) from exc
    return stats


def _as_unit_array(u: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(u, dtype=np.float64)
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("icdf expects probabilities in [0, 1].")
    return arr


@runtime_checkable
class Prior(Protocol):
    """Named distribution that can become a numpyro sample site."""

    name: str

    def to_numpyro(self) -> Any:
        """Return a numpyro distribution (no sample site yet)."""
        ...

    def bounds(self) -> tuple[float | None, float | None]:
        """``(low, high)`` support bounds; ``None`` means unbounded on that side."""
        ...

    def icdf(self, u: np.ndarray | float) -> np.ndarray:
        """Inverse CDF (quantile function) on the host; ``u`` in ``[0, 1]``."""
        ...


@dataclass(frozen=True, slots=True)
class Uniform:
    """Continuous uniform on ``[lo, hi]``."""

    name: str
    lo: float
    hi: float

    def to_numpyro(self) -> Any:
        dist = _require_numpyro()
        return dist.Uniform(self.lo, self.hi, validate_args=True)

    def bounds(self) -> tuple[float | None, float | None]:
        return (float(self.lo), float(self.hi))

    def icdf(self, u: np.ndarray | float) -> np.ndarray:
        stats = _require_scipy_stats()
        return np.asarray(
            stats.uniform(loc=self.lo, scale=self.hi - self.lo).ppf(_as_unit_array(u)),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class Normal:
    """Univariate normal with location ``loc`` and scale ``scale``."""

    name: str
    loc: float
    scale: float

    def to_numpyro(self) -> Any:
        dist = _require_numpyro()
        return dist.Normal(self.loc, self.scale, validate_args=True)

    def bounds(self) -> tuple[float | None, float | None]:
        return (None, None)

    def icdf(self, u: np.ndarray | float) -> np.ndarray:
        stats = _require_scipy_stats()
        return np.asarray(
            stats.norm(loc=self.loc, scale=self.scale).ppf(_as_unit_array(u)),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class LogNormal:
    """Log-normal; ``loc`` / ``scale`` are on the log scale (numpyro convention)."""

    name: str
    loc: float
    scale: float

    def to_numpyro(self) -> Any:
        dist = _require_numpyro()
        return dist.LogNormal(self.loc, self.scale, validate_args=True)

    def bounds(self) -> tuple[float | None, float | None]:
        return (0.0, None)

    def icdf(self, u: np.ndarray | float) -> np.ndarray:
        stats = _require_scipy_stats()
        # scipy: shape=s (log-scale σ), scale=exp(μ)
        return np.asarray(
            stats.lognorm(s=self.scale, scale=float(np.exp(self.loc))).ppf(_as_unit_array(u)),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class TruncatedNormal:
    """Normal truncated to ``[low, high]`` (either end may be ``None``)."""

    name: str
    loc: float
    scale: float
    low: float | None = None
    high: float | None = None

    def to_numpyro(self) -> Any:
        dist = _require_numpyro()
        return dist.TruncatedNormal(
            self.loc, self.scale, low=self.low, high=self.high, validate_args=True
        )

    def bounds(self) -> tuple[float | None, float | None]:
        return (self.low, self.high)

    def icdf(self, u: np.ndarray | float) -> np.ndarray:
        stats = _require_scipy_stats()
        a = -np.inf if self.low is None else (self.low - self.loc) / self.scale
        b = np.inf if self.high is None else (self.high - self.loc) / self.scale
        return np.asarray(
            stats.truncnorm(a=a, b=b, loc=self.loc, scale=self.scale).ppf(_as_unit_array(u)),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class Beta:
    """Beta with concentration parameters (numpyro ``concentration1``, ``concentration0``)."""

    name: str
    concentration1: float
    concentration0: float

    def to_numpyro(self) -> Any:
        dist = _require_numpyro()
        return dist.Beta(self.concentration1, self.concentration0, validate_args=True)

    def bounds(self) -> tuple[float | None, float | None]:
        return (0.0, 1.0)

    def icdf(self, u: np.ndarray | float) -> np.ndarray:
        stats = _require_scipy_stats()
        return np.asarray(
            stats.beta(a=self.concentration1, b=self.concentration0).ppf(_as_unit_array(u)),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class Gamma:
    """Gamma with ``concentration`` (shape) and ``rate``."""

    name: str
    concentration: float
    rate: float = 1.0

    def to_numpyro(self) -> Any:
        dist = _require_numpyro()
        return dist.Gamma(self.concentration, rate=self.rate, validate_args=True)

    def bounds(self) -> tuple[float | None, float | None]:
        return (0.0, None)

    def icdf(self, u: np.ndarray | float) -> np.ndarray:
        stats = _require_scipy_stats()
        # scipy uses scale = 1/rate
        return np.asarray(
            stats.gamma(a=self.concentration, scale=1.0 / self.rate).ppf(_as_unit_array(u)),
            dtype=np.float64,
        )


_DIST_BUILDERS: dict[str, Any] = {
    "uniform": lambda name, p1, p2: Uniform(name, float(p1), float(p2)),
    "normal": lambda name, p1, p2: Normal(name, float(p1), float(p2)),
    "lognormal": lambda name, p1, p2: LogNormal(name, float(p1), float(p2)),
    "truncatednormal": lambda name, p1, p2: TruncatedNormal(
        name, float(p1), float(p2), low=0.0, high=None
    ),
    "beta": lambda name, p1, p2: Beta(name, float(p1), float(p2)),
    "gamma": lambda name, p1, p2: Gamma(name, float(p1), rate=float(p2)),
}


def priors_from_frame(
    df: Any,
    name_col: str,
    dist_col: str,
    p1_col: str,
    p2_col: str,
) -> tuple[Prior, ...]:
    """Build priors from a table (Kiribati ``parameters.xlsx`` constant sheet).

    ``df`` is anything with column labels and row iteration (pandas
    ``DataFrame``, polars, or a mapping of column name → sequence). Dist names
    are matched case-insensitively: ``uniform``, ``normal``, ``lognormal``,
    ``truncatednormal`` (truncated at 0 from below), ``beta``, ``gamma``.
    """
    columns = _frame_columns(df, (name_col, dist_col, p1_col, p2_col))
    names = columns[name_col]
    dists = columns[dist_col]
    p1s = columns[p1_col]
    p2s = columns[p2_col]
    if not (len(names) == len(dists) == len(p1s) == len(p2s)):
        raise ValueError("priors_from_frame columns must have equal length.")

    out: list[Prior] = []
    for name, dist_name, p1, p2 in zip(names, dists, p1s, p2s, strict=True):
        key = str(dist_name).strip().lower().replace(" ", "").replace("_", "")
        # accept log_normal / truncated_normal spellings after underscore strip
        builder = _DIST_BUILDERS.get(key)
        if builder is None:
            raise ValueError(
                f"Unknown prior distribution {dist_name!r} for parameter {name!r}. "
                f"Expected one of {sorted(_DIST_BUILDERS)}."
            )
        out.append(builder(str(name), p1, p2))
    return tuple(out)


def _frame_columns(df: Any, cols: Sequence[str]) -> dict[str, Sequence[Any]]:
    if isinstance(df, Mapping):
        missing = [c for c in cols if c not in df]
        if missing:
            raise KeyError(f"priors_from_frame missing columns: {missing}.")
        return {c: list(df[c]) for c in cols}

    # pandas / polars-like
    for attr in ("to_dict",):
        if hasattr(df, attr) and hasattr(df, "columns"):
            available = set(map(str, df.columns))
            missing = [c for c in cols if c not in available]
            if missing:
                raise KeyError(f"priors_from_frame missing columns: {missing}.")
            # Prefer column Series → list without requiring pandas types.
            return {c: list(np.asarray(df[c]).reshape(-1)) for c in cols}

    raise TypeError(
        "priors_from_frame expects a mapping or a frame with named columns, "
        f"got {type(df).__name__}."
    )


__all__ = [
    "Beta",
    "Gamma",
    "LogNormal",
    "Normal",
    "Prior",
    "TruncatedNormal",
    "Uniform",
    "priors_from_frame",
]
