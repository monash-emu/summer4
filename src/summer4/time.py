"""Real-world and model time.

Dates never enter the JAX path: an :class:`Epoch` maps between host-side
``date`` / ``datetime64`` values and numeric model time. All index arithmetic
(:meth:`TimeAxis.locate`, :meth:`TimeAxis.grouping`, :meth:`TimeAxis.rolling`)
is host-side and static, computed from a concrete axis at trace time. Only
gathers, segment reductions, lerps and arithmetic are traced.

A ``date`` is never a traced value.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from summer4.enums import coerce_strenum

type When = float | date | datetime | np.datetime64
type CalendarRule = str | int


class ReduceHow(StrEnum):
    """Aggregation for time reductions (:meth:`TimeAxis.rolling`, Output resample).

    Prefer these members at call sites. Bare strings such as ``"sum"`` are still
    accepted and coerced.
    """

    SUM = "sum"
    MEAN = "mean"


class TimeAxisKind(StrEnum):
    """How a :class:`TimeAxis` was constructed.

    Prefer these members at call sites. Bare strings such as ``"grid"`` are still
    accepted and coerced.
    """

    GRID = "grid"
    EXPLICIT = "explicit"
    STEPS = "steps"


type ReduceHowArg = ReduceHow | str
type TimeAxisKindArg = TimeAxisKind | str


def _is_tracer(value: object) -> bool:
    """Return True when ``value`` looks like a JAX tracer."""
    return type(value).__name__ in {"DynamicJaxprTracer", "Tracer", "ShapedArray"} or (
        hasattr(value, "aval") and not isinstance(value, np.ndarray)
    )


def _require_concrete(values: object, *, op: str) -> NDArray[np.float64]:
    if _is_tracer(values):
        raise TypeError(
            f"time values are traced; `{op}` needs a concrete axis. "
            "Build TimeAxis on host data, or call this outside jit."
        )
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"TimeAxis.values must be 1-d, got shape {arr.shape}.")
    return arr


@dataclass(frozen=True, slots=True)
class Epoch:
    """Affine map between calendar dates and numeric model time.

    ``model_time = (date - ref_date) / unit`` (in seconds).
    """

    ref_date: date
    unit: timedelta = field(default_factory=lambda: timedelta(days=1))

    def __post_init__(self) -> None:
        if self.unit.total_seconds() <= 0:
            raise ValueError(f"Epoch.unit must be positive, got {self.unit!r}.")

    def to_model(self, dates: When | NDArray[Any] | list[Any]) -> NDArray[np.float64]:
        """Map calendar values to model time."""
        arr = np.asarray(dates)
        if arr.dtype == object or np.issubdtype(arr.dtype, np.datetime64):
            dt64 = np.asarray(arr, dtype="datetime64[ns]")
        else:
            # list of date/datetime
            dt64 = np.asarray(
                [np.datetime64(self._as_datetime(d), "ns") for d in np.atleast_1d(arr).tolist()],
                dtype="datetime64[ns]",
            )
        ref = np.datetime64(datetime.combine(self.ref_date, datetime.min.time()), "ns")
        delta_ns = (dt64.astype("datetime64[ns]") - ref).astype(np.int64)
        unit_ns = int(self.unit.total_seconds() * 1e9)
        out = delta_ns.astype(np.float64) / float(unit_ns)
        return np.asarray(out, dtype=np.float64)

    def from_model(self, values: NDArray[np.floating[Any]] | float) -> NDArray[np.datetime64]:
        """Map model time to ``datetime64[ns]``."""
        arr = np.asarray(values, dtype=np.float64)
        unit_ns = int(self.unit.total_seconds() * 1e9)
        delta_ns = np.rint(arr * unit_ns).astype(np.int64)
        ref = np.datetime64(datetime.combine(self.ref_date, datetime.min.time()), "ns")
        return (ref + delta_ns.astype("timedelta64[ns]")).astype("datetime64[ns]")

    @staticmethod
    def _as_datetime(value: When) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        if isinstance(value, np.datetime64):
            # ns -> python datetime via astype
            as_dt = value.astype("datetime64[us]").item()
            if isinstance(as_dt, datetime):
                return as_dt
            raise TypeError(f"Expected datetime from datetime64, got {type(as_dt).__name__}.")
        raise TypeError(f"Expected date/datetime/datetime64, got {type(value).__name__}.")


@dataclass(frozen=True, slots=True)
class TimeGrouping:
    """Static calendar or fixed-stride grouping plan for a :class:`TimeAxis`."""

    segment_ids: NDArray[np.int32]
    n_groups: int
    counts: NDArray[np.int32]
    starts: NDArray[np.float64]
    labels: tuple[str, ...]
    rule: CalendarRule

    def __hash__(self) -> int:
        return hash(
            (
                self.segment_ids.tobytes(),
                self.n_groups,
                self.counts.tobytes(),
                self.starts.tobytes(),
                self.labels,
                self.rule,
            )
        )


@dataclass(frozen=True, slots=True)
class RollingSpec:
    """Static rolling-window plan (cumsum-difference, O(n))."""

    window: int
    how: ReduceHowArg
    center: bool
    min_periods: int
    valid_counts: NDArray[np.int32]

    def __post_init__(self) -> None:
        resolved = coerce_strenum(ReduceHow, self.how, what="RollingSpec how")
        object.__setattr__(self, "how", resolved)

    def __hash__(self) -> int:
        how = self.how if isinstance(self.how, ReduceHow) else ReduceHow(self.how)
        return hash(
            (
                self.window,
                how.value,
                self.center,
                self.min_periods,
                self.valid_counts.tobytes(),
            )
        )


def _normalize_rule(rule: CalendarRule) -> CalendarRule:
    if isinstance(rule, int):
        if rule < 1:
            raise ValueError(f"Integer resample factor must be >= 1, got {rule}.")
        return rule
    allowed = {"D", "W", "ME", "QE", "YE"}
    if rule in allowed or (rule.startswith("W-") and len(rule) == 5):
        return rule
    raise ValueError(
        f"Unsupported calendar rule {rule!r}. Supported: 'D', 'W'/'W-<DAY>', "
        f"'ME', 'QE', 'YE', or an integer factor. For exotic offsets use "
        f"Output.to_pandas() and resample there."
    )


_WEEKDAY = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


def _period_key(dt: np.datetime64, rule: str) -> tuple[Hashable, ...]:
    """Return a hashable period identity for ``dt`` under ``rule``."""
    # Work in calendar days
    day = dt.astype("datetime64[D]")
    y, m, d = (int(x) for x in str(day).split("-"))
    if rule == "D":
        return (y, m, d)
    if rule == "ME":
        return (y, m)
    if rule == "QE":
        return (y, (m - 1) // 3)
    if rule == "YE":
        return (y,)
    if rule == "W" or rule.startswith("W-"):
        end = _WEEKDAY.get(rule[2:] if rule.startswith("W-") else "SUN", 5)
        # ISO: Monday=0 ... Sunday=6 for datetime.weekday()
        py = date(y, m, d)
        # Days until next `end` (inclusive week ending on that weekday)
        delta = (end - py.weekday()) % 7
        week_end = py + timedelta(days=delta)
        return (week_end.year, week_end.month, week_end.day)
    raise ValueError(rule)


def _build_calendar_grouping(
    values: NDArray[np.float64],
    epoch: Epoch,
    rule: str,
    origin: float | None,
) -> TimeGrouping:
    del origin  # reserved; calendar rules use the natural period boundaries
    dates = epoch.from_model(values)
    keys: list[tuple[Hashable, ...]] = [_period_key(d, rule) for d in dates]
    # Assign segment ids in order of first appearance
    key_to_id: dict[tuple[Hashable, ...], int] = {}
    segment_ids = np.empty(len(keys), dtype=np.int32)
    starts_list: list[float] = []
    labels_list: list[str] = []
    for i, key in enumerate(keys):
        if key not in key_to_id:
            key_to_id[key] = len(key_to_id)
            starts_list.append(float(values[i]))
            labels_list.append("-".join(str(p) for p in key))
        segment_ids[i] = key_to_id[key]
    n_groups = len(key_to_id)
    counts = np.bincount(segment_ids, minlength=n_groups).astype(np.int32)
    return TimeGrouping(
        segment_ids=segment_ids,
        n_groups=n_groups,
        counts=counts,
        starts=np.asarray(starts_list, dtype=np.float64),
        labels=tuple(labels_list),
        rule=rule,
    )


def _build_factor_grouping(values: NDArray[np.float64], factor: int) -> TimeGrouping:
    n = values.size
    n_groups = (n + factor - 1) // factor
    segment_ids = (np.arange(n, dtype=np.int32) // factor).astype(np.int32)
    # Drop incomplete trailing group? Keep it (partial allowed; min_periods elsewhere).
    counts = np.bincount(segment_ids, minlength=n_groups).astype(np.int32)
    starts = np.asarray([float(values[i * factor]) for i in range(n_groups)], dtype=np.float64)
    labels = tuple(str(i) for i in range(n_groups))
    return TimeGrouping(
        segment_ids=segment_ids,
        n_groups=n_groups,
        counts=counts,
        starts=starts,
        labels=labels,
        rule=factor,
    )


@dataclass(frozen=True, slots=True)
class TimeAxis:
    """Numeric model times with an optional calendar :class:`Epoch`.

    ``values`` is a pytree leaf (may be a tracer); ``epoch`` and ``kind`` are aux.
    """

    values: NDArray[np.float64] | Any
    epoch: Epoch | None = None
    kind: TimeAxisKindArg = TimeAxisKind.GRID
    _group_cache: dict[tuple[CalendarRule, float | None], TimeGrouping] = field(
        default_factory=dict, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", coerce_strenum(TimeAxisKind, self.kind, what="TimeAxis kind")
        )

    def _concrete(self, *, op: str) -> NDArray[np.float64]:
        return _require_concrete(self.values, op=op)

    def _to_model(self, when: When) -> float:
        if isinstance(when, (int, float, np.floating)):
            return float(when)
        if self.epoch is None:
            raise TypeError(f"Cannot locate {when!r} without an Epoch on this TimeAxis.")
        return float(self.epoch.to_model(when))

    def locate(self, when: When) -> int:
        """Return the index of the grid point nearest to ``when`` (host-side)."""
        values = self._concrete(op="locate")
        t = self._to_model(when)
        return int(np.argmin(np.abs(values - t)))

    def window(self, t0: When, t1: When) -> slice:
        """Return a half-open slice covering ``[t0, t1]`` on the axis (host-side)."""
        values = self._concrete(op="window")
        a = self._to_model(t0)
        b = self._to_model(t1)
        if b < a:
            a, b = b, a
        i0 = int(np.searchsorted(values, a, side="left"))
        i1 = int(np.searchsorted(values, b, side="right"))
        return slice(i0, i1)

    def weights_for(
        self, ts: NDArray[np.floating[Any]] | list[float]
    ) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
        """Return linear-interpolation index/weight pairs for times ``ts``.

        ``idx`` has shape ``(n, 2)``, ``w`` has shape ``(n, 2)`` with
        ``w[:, 0] + w[:, 1] == 1``. On-grid times are an exact gather
        (``w[i] = [1, 0]``, both indices equal).
        """
        values = self._concrete(op="weights_for")
        targets = np.asarray(ts, dtype=np.float64).reshape(-1)
        n = targets.size
        idx = np.empty((n, 2), dtype=np.int32)
        w = np.empty((n, 2), dtype=np.float64)
        if values.size == 0:
            raise ValueError("Cannot interpolate on an empty TimeAxis.")
        for i, t in enumerate(targets):
            if t <= values[0]:
                idx[i] = (0, 0)
                w[i] = (1.0, 0.0)
                continue
            if t >= values[-1]:
                last = values.size - 1
                idx[i] = (last, last)
                w[i] = (1.0, 0.0)
                continue
            right = int(np.searchsorted(values, t, side="left"))
            if values[right] == t:
                idx[i] = (right, right)
                w[i] = (1.0, 0.0)
                continue
            left = right - 1
            span = float(values[right] - values[left])
            if span == 0.0:
                idx[i] = (left, left)
                w[i] = (1.0, 0.0)
            else:
                alpha = float((t - values[left]) / span)
                idx[i] = (left, right)
                w[i] = (1.0 - alpha, alpha)
        return idx, w

    def grouping(self, rule: CalendarRule, *, origin: float | None = None) -> TimeGrouping:
        """Return a cached :class:`TimeGrouping` for ``rule``."""
        rule = _normalize_rule(rule)
        key = (rule, origin)
        cached = self._group_cache.get(key)
        if cached is not None:
            return cached
        values = self._concrete(op="grouping")
        if isinstance(rule, int):
            grouping = _build_factor_grouping(values, rule)
        else:
            if self.epoch is None:
                raise TypeError(f"Calendar rule {rule!r} requires TimeAxis.epoch.")
            grouping = _build_calendar_grouping(values, self.epoch, rule, origin)
        self._group_cache[key] = grouping
        return grouping

    def rolling(
        self,
        window: int,
        *,
        how: ReduceHowArg = ReduceHow.MEAN,
        center: bool = False,
        min_periods: int | None = None,
    ) -> RollingSpec:
        """Return a static :class:`RollingSpec`.

        Sum/mean apply as a cumulative-sum difference (O(n), one pass) rather
        than ``jnp.convolve``, so the plan stays a pair of prefix sums.
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}.")
        values = self._concrete(op="rolling")
        n = values.size
        mp = window if min_periods is None else int(min_periods)
        if mp < 1:
            raise ValueError(f"min_periods must be >= 1, got {mp}.")
        resolved_how = coerce_strenum(ReduceHow, how, what="TimeAxis.rolling how")
        counts = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if center:
                left = i - (window // 2)
                right = left + window
            else:
                left = i - window + 1
                right = i + 1
            lo = max(0, left)
            hi = min(n, right)
            counts[i] = hi - lo
        return RollingSpec(
            window=window,
            how=resolved_how,
            center=center,
            min_periods=mp,
            valid_counts=counts,
        )

    def as_dates(self) -> NDArray[np.datetime64]:
        """Return ``datetime64[ns]`` values; requires :attr:`epoch`."""
        if self.epoch is None:
            raise TypeError("as_dates() requires TimeAxis.epoch.")
        return self.epoch.from_model(self._concrete(op="as_dates"))

    def as_index(self) -> object:
        """Return a pandas Index / DatetimeIndex (lazy import)."""
        import pandas as pd  # type: ignore[import-untyped]

        values = self._concrete(op="as_index")
        if self.epoch is None:
            return pd.Index(values, name="time")
        return pd.DatetimeIndex(self.epoch.from_model(values), name="time")

    def tree_flatten(self) -> tuple[tuple[Any], tuple[Epoch | None, TimeAxisKindArg]]:
        return (self.values,), (self.epoch, self.kind)

    @classmethod
    def tree_unflatten(
        cls, aux: tuple[Epoch | None, TimeAxisKindArg], children: tuple[Any, ...]
    ) -> TimeAxis:
        (values,) = children
        epoch, kind = aux
        return cls(values=values, epoch=epoch, kind=kind)
