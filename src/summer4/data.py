"""Dated observation series as interpolation knots on the model time axis.

Pandas is an optional extra. Importing this module never requires it;
:meth:`Data.from_series` and :meth:`Data.from_csv` raise a clear error if it
is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from summer4.flows.rates import Interp, Time
from summer4.time import Epoch
from summer4.timevarying import linear, sigmoidal, step

__all__ = ["Data"]


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "summer4.data requires pandas. Install with: pip install 'summer4[pandas]' "
            "(or the project's pandas optional extra)."
        ) from exc
    return pd


@dataclass(frozen=True, slots=True)
class Data:
    """Observed series on the model's numeric time axis.

    Built from calendar dates via an :class:`~summer4.time.Epoch` — the same
    conversion path as :meth:`~summer4.results.targets.Target.from_series`.
    """

    times: NDArray[np.float64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if times.shape != values.shape:
            raise ValueError(f"Data times shape {times.shape} != values shape {values.shape}.")
        if times.size < 1:
            raise ValueError("Data requires at least one observation.")
        if times.size >= 2 and np.any(np.diff(times) <= 0):
            raise ValueError("Data times must be strictly increasing.")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_series(cls, s: Any, epoch: Epoch) -> Data:
        """Build from a pandas Series with a ``DatetimeIndex``.

        Uses :meth:`Epoch.to_model` so dated times agree with
        :meth:`~summer4.results.targets.Target.from_series`.
        """
        pd = _require_pandas()
        if not isinstance(s.index, pd.DatetimeIndex):
            raise TypeError(
                f"Data.from_series expects a DatetimeIndex, got {type(s.index).__name__}."
            )
        times = epoch.to_model(s.index.to_numpy())
        values = np.asarray(s.to_numpy(), dtype=np.float64)
        return cls(times=times, values=values)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        epoch: Epoch,
        *,
        time_column: str = "time",
        value_column: str = "value",
    ) -> Data:
        """Load a dated CSV and map ``time_column`` through ``epoch``."""
        pd = _require_pandas()
        frame = pd.read_csv(path, parse_dates=[time_column])
        series = frame.set_index(time_column)[value_column]
        return cls.from_series(series, epoch)

    def interp(
        self,
        kind: Literal["linear", "sigmoidal", "step"] = "linear",
        *,
        sharpness: float = 1.0,
    ) -> Interp:
        """Return an :class:`~summer4.flows.rates.Interp` over this series.

        Outside the observed range the interpolator clamps to the end value
        (``jax.numpy.interp`` behaviour for ``linear``).

        For ``kind=\"step\"``, an extra leading plateau equal to the first
        observation is prepended so ``len(values) == len(breakpoints) + 1``
        as required by :func:`~summer4.timevarying.step`.
        """
        bps = tuple(float(t) for t in self.times)
        vals = tuple(float(v) for v in self.values)
        if kind == "linear":
            return linear(Time(), bps, vals)
        if kind == "sigmoidal":
            return sigmoidal(Time(), bps, vals, sharpness=sharpness)
        if kind == "step":
            return step(Time(), bps, (vals[0], *vals))
        raise ValueError(f"Unknown interp kind {kind!r}.")
