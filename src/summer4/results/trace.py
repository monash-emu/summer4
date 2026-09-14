"""Named timeseries traces and their query surface.

Axis rule: time is the second-to-last axis; the aligned (compartment or edge)
axis is last; anything to the left is free. Resolve axes through ``dims``, never
by hard-coded position — reductions move them.

All index arithmetic is host-side and static, computed from
:class:`~summer4.time.TimeAxis` at trace time. Only gathers, segment
reductions, lerps and arithmetic are traced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from summer4.jax.propertydata import PropertyData
from summer4.properties import Property, Trait
from summer4.propertymap import Groups, PropertyMap
from summer4.selectors import Selector
from summer4.time import CalendarRule, RollingSpec, TimeAxis, TimeGrouping, When


def _xp() -> Any:
    import jax.numpy as jnp

    return jnp


def _as_array(values: PropertyData | Any) -> Any:
    if isinstance(values, PropertyData):
        return values.data
    return values


def _time_axis_index(dims: tuple[str, ...]) -> int:
    try:
        return dims.index("time")
    except ValueError as exc:
        raise ValueError(f"Trace dims {dims} have no 'time' axis.") from exc


def _aligned_axis_index(dims: tuple[str, ...]) -> int | None:
    for name in ("compartment", "edge", "group"):
        if name in dims:
            return dims.index(name)
    return None


@dataclass(frozen=True, slots=True)
class Trace:
    """One named quantity over time — the unit the query surface operates on."""

    times: TimeAxis
    values: PropertyData | Any
    dims: tuple[str, ...]

    def _with(
        self,
        *,
        values: PropertyData | Any | None = None,
        dims: tuple[str, ...] | None = None,
        times: TimeAxis | None = None,
    ) -> Trace:
        return Trace(
            times=self.times if times is None else times,
            values=self.values if values is None else values,
            dims=self.dims if dims is None else dims,
        )

    def _pmap(self) -> PropertyMap | None:
        return self.values.pmap if isinstance(self.values, PropertyData) else None

    # --- compartment / edge selection -------------------------------------------------

    def select(self, sel: Selector) -> Trace:
        """Zero compartments/edges where ``sel`` is not Kleene-true (keeps the map)."""
        xp = _xp()
        pmap = self._pmap()
        if pmap is None:
            raise TypeError("select() requires PropertyData values.")
        mask = pmap.mask(sel)
        data = xp.where(mask, _as_array(self.values), 0)
        return self._with(values=PropertyData(pmap, data))

    def sum_over(self, prop: Property | str) -> Trace:
        """Sum the aligned axis by trait of ``prop`` (compartments only in Phase 2)."""
        pmap = self._pmap()
        if pmap is None:
            raise TypeError("sum_over() requires PropertyData values.")
        reduced = PropertyData(pmap, _as_array(self.values)).sum_over(prop)
        dims = tuple(d if d != "compartment" else "group" for d in self.dims)
        if "group" not in dims:
            dims = self.dims[:-1] + ("group",)
        return self._with(values=reduced, dims=dims)

    def total(self) -> Trace:
        """Sum over the aligned (last) axis."""
        xp = _xp()
        data = xp.sum(_as_array(self.values), axis=-1)
        dims = self.dims[:-1]
        return self._with(values=data, dims=dims)

    def partition(self, prop: Property | str) -> dict[Trait, Trace]:
        """Split into one Trace per trait of ``prop`` (static keys; values traced)."""
        pmap = self._pmap()
        if pmap is None:
            raise TypeError("partition() requires PropertyData values.")
        out: dict[Trait, Trace] = {}
        for trait, idx in pmap.partition(prop).items():
            data = _as_array(self.values)[..., idx]
            out[trait] = self._with(values=data, dims=self.dims)
        return out

    def group_by(self, *props: Property | str) -> Groups[Trace]:
        """Group by one or more properties (static keys; values traced)."""
        pmap = self._pmap()
        if pmap is None:
            raise TypeError("group_by() requires PropertyData values.")
        data: dict[tuple[Trait, ...], Trace] = {}
        for key, idx in pmap.group_by(*props).items():
            gathered = _as_array(self.values)[..., idx]
            data[key] = self._with(values=gathered, dims=self.dims)
        return Groups(_data=data)

    # --- time selection ---------------------------------------------------------------

    def between(self, t0: When, t1: When) -> Trace:
        """Slice the time axis to ``[t0, t1]`` (indices computed host-side)."""
        sl = self.times.window(t0, t1)
        t_ax = _time_axis_index(self.dims)
        data = _as_array(self.values)
        slicer: list[Any] = [slice(None)] * data.ndim
        slicer[t_ax] = sl
        new_vals = data[tuple(slicer)]
        new_times = TimeAxis(
            values=np.asarray(self.times.values)[sl],
            epoch=self.times.epoch,
            kind=self.times.kind,
        )
        if isinstance(self.values, PropertyData):
            new_vals = PropertyData(self.values.pmap, new_vals)
        return self._with(values=new_vals, times=new_times)

    def at(self, when: When) -> Trace:
        """Take the nearest time point (host-side locate, traced gather)."""
        i = self.times.locate(when)
        t_ax = _time_axis_index(self.dims)
        data = _as_array(self.values)
        slicer: list[Any] = [slice(None)] * data.ndim
        slicer[t_ax] = i
        new_vals = data[tuple(slicer)]
        new_times = TimeAxis(
            values=np.asarray([float(np.asarray(self.times.values)[i])]),
            epoch=self.times.epoch,
            kind="explicit",
        )
        dims = self.dims[:t_ax] + self.dims[t_ax + 1 :]
        if isinstance(self.values, PropertyData):
            new_vals = PropertyData(self.values.pmap, new_vals)
        return self._with(values=new_vals, times=new_times, dims=dims)

    def at_times(self, ts: Any) -> Trace:
        """Linearly interpolate onto ``ts`` (exact gather when on-grid)."""
        xp = _xp()
        targets = np.asarray(ts, dtype=np.float64)
        idx, w = self.times.weights_for(targets)
        t_ax = _time_axis_index(self.dims)
        data = _as_array(self.values)
        # Move time axis to 0 for gather, then restore
        data_t = xp.moveaxis(data, t_ax, 0)
        left = data_t[idx[:, 0]]
        right = data_t[idx[:, 1]]
        w0 = xp.asarray(w[:, 0]).reshape((-1,) + (1,) * (left.ndim - 1))
        w1 = xp.asarray(w[:, 1]).reshape((-1,) + (1,) * (left.ndim - 1))
        out = w0 * left + w1 * right
        out = xp.moveaxis(out, 0, t_ax)
        new_times = TimeAxis(values=targets, epoch=self.times.epoch, kind="explicit")
        if isinstance(self.values, PropertyData):
            out = PropertyData(self.values.pmap, out)
        return self._with(values=out, times=new_times)

    # --- reductions along time --------------------------------------------------------

    def reduce_by(self, grouping: TimeGrouping, how: Literal["sum", "mean"] = "sum") -> Trace:
        """Segment-reduce along time using a static :class:`TimeGrouping`."""
        xp = _xp()
        import jax

        t_ax = _time_axis_index(self.dims)
        data = _as_array(self.values)
        data_t = xp.moveaxis(data, t_ax, 0)
        flat = data_t.reshape((data_t.shape[0], -1))
        ids = np.asarray(grouping.segment_ids, dtype=np.int32)
        n = grouping.n_groups

        def _col(col: Any) -> Any:
            return jax.ops.segment_sum(col, ids, num_segments=n)

        summed = jax.vmap(_col, in_axes=1, out_axes=1)(flat)
        if how == "mean":
            counts = xp.asarray(grouping.counts, dtype=data_t.dtype).reshape((n, 1))
            summed = summed / xp.maximum(counts, 1)
        new_shape = (n,) + data_t.shape[1:]
        out = xp.reshape(summed, new_shape)
        out = xp.moveaxis(out, 0, t_ax)
        new_times = TimeAxis(values=grouping.starts, epoch=self.times.epoch, kind="explicit")
        if isinstance(self.values, PropertyData):
            out = PropertyData(self.values.pmap, out)
        return self._with(values=out, times=new_times)

    def resample(self, rule: CalendarRule, how: Literal["sum", "mean"] = "sum") -> Trace:
        """Sugar over :meth:`reduce_by` with a cached :class:`TimeGrouping`."""
        return self.reduce_by(self.times.grouping(rule), how=how)

    def rolling(
        self,
        window: int,
        *,
        how: Literal["sum", "mean"] = "mean",
        center: bool = False,
        min_periods: int | None = None,
    ) -> Trace:
        """Rolling window via cumsum difference (O(n)); see :class:`RollingSpec`."""
        spec = self.times.rolling(window, how=how, center=center, min_periods=min_periods)
        return self._apply_rolling(spec)

    def _apply_rolling(self, spec: RollingSpec) -> Trace:
        xp = _xp()
        t_ax = _time_axis_index(self.dims)
        data = _as_array(self.values)
        data_t = xp.moveaxis(data, t_ax, 0)
        n = data_t.shape[0]
        # Prefix sums along time; pad a leading zero
        zeros = xp.zeros((1,) + data_t.shape[1:], dtype=data_t.dtype)
        csum = xp.concatenate([zeros, xp.cumsum(data_t, axis=0)], axis=0)
        out_rows: list[Any] = []
        for i in range(n):
            if spec.center:
                left = i - (spec.window // 2)
                right = left + spec.window
            else:
                left = i - spec.window + 1
                right = i + 1
            lo = max(0, left)
            hi = min(n, right)
            count = hi - lo
            if count < spec.min_periods:
                out_rows.append(xp.full(data_t.shape[1:], xp.nan, dtype=data_t.dtype))
            else:
                total = csum[hi] - csum[lo]
                out_rows.append(total / count if spec.how == "mean" else total)
        out = xp.stack(out_rows, axis=0)
        out = xp.moveaxis(out, 0, t_ax)
        if isinstance(self.values, PropertyData):
            out = PropertyData(self.values.pmap, out)
        return self._with(values=out)

    def cumulative(self) -> Trace:
        """Cumulative sum along the time axis."""
        xp = _xp()
        t_ax = _time_axis_index(self.dims)
        data = _as_array(self.values)
        out = xp.cumsum(data, axis=t_ax)
        if isinstance(self.values, PropertyData):
            out = PropertyData(self.values.pmap, out)
        return self._with(values=out)

    # --- host-side export -------------------------------------------------------------

    def to_frame(self) -> object:
        """Return a polars DataFrame (lazy import)."""
        import polars as pl

        values = np.asarray(_as_array(self.values))
        times = np.asarray(self.times.values, dtype=np.float64)
        t_ax = _time_axis_index(self.dims)
        if values.ndim == 1 and t_ax == 0:
            return pl.DataFrame({"time": times, "value": values})
        # Flatten trailing axes into columns
        if t_ax != 0:
            values = np.moveaxis(values, t_ax, 0)
        n_t = values.shape[0]
        flat = values.reshape(n_t, -1)
        cols: dict[str, Any] = {"time": times[:n_t]}
        pmap = self._pmap()
        if pmap is not None and flat.shape[1] == pmap.size:
            labels = pmap.labels()
            for j, lab in enumerate(labels):
                cols[lab] = flat[:, j]
        else:
            for j in range(flat.shape[1]):
                cols[f"v{j}"] = flat[:, j]
        return pl.DataFrame(cols)

    def to_pandas(self) -> object:
        """Return a pandas DataFrame indexed by time / DatetimeIndex (lazy import)."""
        import pandas as pd  # type: ignore[import-untyped]

        frame = self.to_frame()
        pdf = frame.to_pandas()  # type: ignore[attr-defined]
        if self.times.epoch is not None:
            pdf.index = pd.DatetimeIndex(
                self.times.epoch.from_model(np.asarray(pdf["time"], dtype=np.float64)),
                name="time",
            )
            return pdf.drop(columns=["time"])
        pdf = pdf.set_index("time")
        pdf.index.name = "time"
        return pdf

    def plot(self, **kwargs: Any) -> Any:
        """Plot via matplotlib (lazy import). Host-side only."""
        import matplotlib.pyplot as plt

        frame = self.to_pandas()
        ax = frame.plot(**kwargs)  # type: ignore[attr-defined]
        plt.xlabel("time")
        return ax
