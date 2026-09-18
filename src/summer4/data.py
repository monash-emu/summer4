"""Dated observation series as interpolation knots on the model time axis.

Pandas is an optional extra. Importing this module never requires it;
:meth:`Data.from_series` and :meth:`Data.from_csv` raise a clear error if it
is missing. :meth:`Data.table` accepts a DataFrame when pandas is installed,
and a plain array otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from summer4.flows.rates import ArrayConst, Interp, RateOps, TableInterp, Time, as_rate
from summer4.properties import Property
from summer4.time import Epoch
from summer4.timevarying import linear, sigmoidal, step

__all__ = ["Data", "TableData"]


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

    @classmethod
    def table(
        cls,
        times: ArrayLike,
        values: ArrayLike | Any,
        *,
        over: Property,
        columns: Sequence[str] | None = None,
    ) -> TableData:
        """Build a column-per-trait table sharing one time axis.

        ``values`` is an array of shape ``(n_times, n_traits)`` in trait
        order, or a DataFrame whose columns are the trait names. ``columns``,
        when given, must equal ``over.traits`` in order; it selects those
        columns from a wider frame. A frame passed without ``columns`` must
        already have exactly those columns, in that order.
        """
        knot_times = np.asarray(times, dtype=np.float64).reshape(-1)
        matrix = _table_matrix(values, over, columns)
        if knot_times.size < 1:
            raise ValueError("Data.table requires at least one time.")
        if knot_times.shape[0] != matrix.shape[0]:
            raise ValueError(
                f"Data.table times length {knot_times.shape[0]} != "
                f"values rows {matrix.shape[0]}."
            )
        if knot_times.size >= 2 and bool(np.any(np.diff(knot_times) <= 0)):
            raise ValueError("Data.table times must be strictly increasing.")
        return TableData(times=knot_times, values=matrix, over=over)


def _frame_column_names(values: object) -> tuple[str, ...] | None:
    columns = getattr(values, "columns", None)
    to_numpy = getattr(values, "to_numpy", None)
    if columns is None or not callable(to_numpy):
        return None
    return tuple(str(column) for column in columns)


def _table_matrix(
    values: ArrayLike | Any,
    over: Property,
    columns: Sequence[str] | None,
) -> NDArray[np.float64]:
    names = _frame_column_names(values)
    if names is not None:
        frame: Any = values
        if columns is None:
            if names != over.traits:
                raise ValueError(
                    f"Data.table columns {names} do not match property {over.name!r} "
                    f"traits {over.traits}."
                )
            selected = over.traits
        else:
            selected = tuple(columns)
            if selected != over.traits:
                raise ValueError(
                    f"Data.table columns {selected} do not match property {over.name!r} "
                    f"traits {over.traits}."
                )
            missing = [name for name in selected if name not in names]
            if missing:
                raise ValueError(
                    f"Data.table is missing columns {missing} for property {over.name!r}."
                )
        matrix = np.asarray(frame[list(selected)].to_numpy(), dtype=np.float64)
    else:
        if columns is not None and tuple(columns) != over.traits:
            raise ValueError(
                f"Data.table columns {tuple(columns)} do not match property {over.name!r} "
                f"traits {over.traits}."
            )
        matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"Data.table values must have shape (n_times, n_traits), got {matrix.shape}."
        )
    if matrix.shape[1] != len(over.traits):
        raise ValueError(
            f"Data.table has {matrix.shape[1]} columns but property {over.name!r} "
            f"has traits {over.traits}."
        )
    return matrix


@dataclass(frozen=True, slots=True)
class TableData:
    """One observed series per trait, sharing :attr:`times`.

    Built by :meth:`Data.table`. Column ``i`` is ``over.traits[i]``.
    """

    times: NDArray[np.float64]
    values: NDArray[np.float64]
    over: Property

    def interp(
        self,
        kind: Literal["linear", "sigmoidal", "step"] = "linear",
        *,
        arg: RateOps | None = None,
        sharpness: float = 1.0,
    ) -> TableInterp:
        """Return a :class:`~summer4.flows.rates.TableInterp` over this table.

        ``arg`` defaults to :class:`~summer4.flows.rates.Time`. For
        ``kind="step"`` the first row is prepended so the table matches
        :meth:`Data.interp` — the value before the first time is the first
        observation, and the value changes at each later time.
        """
        if kind not in ("linear", "sigmoidal", "step"):
            raise ValueError(f"Unknown interp kind {kind!r}.")
        vals = self.values
        if kind == "step":
            vals = np.concatenate([vals[:1], vals], axis=0)
        return TableInterp(
            kind=kind,
            times=ArrayConst(self.times),
            values=ArrayConst(vals),
            over=self.over,
            arg=Time() if arg is None else as_rate(arg),
            sharpness=float(sharpness),
        )
