"""Named timeseries traces and their query surface.

Axis rule: time is the second-to-last axis; the aligned (compartment or edge)
axis is last; anything to the left is free. Resolve axes through ``dims``, never
by hard-coded position — reductions move them.

All index arithmetic is host-side and static, computed from
:class:`~summer4.time.TimeAxis` at trace time. Only gathers, segment
reductions, lerps and arithmetic are traced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from summer4.flows.edges import edge_labels, rewrite_edge_selector, sum_over_edge
from summer4.jax.propertydata import PropertyData
from summer4.properties import Property, Trait
from summer4.propertymap import Groups, PropertyMap
from summer4.selectors import Selector
from summer4.time import (
    CalendarRule,
    ReduceHow,
    ReduceHowArg,
    RollingSpec,
    TimeAxis,
    TimeAxisKind,
    TimeGrouping,
    When,
)

type QuadMethod = Literal["trapezoid", "simpson"]


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


def _is_edge_trace(dims: tuple[str, ...]) -> bool:
    return "edge" in dims


def _wrap_like(template: PropertyData | Any, data: Any) -> PropertyData | Any:
    if isinstance(template, PropertyData):
        return PropertyData(template.pmap, data)
    return data


def _column_labels(pmap: PropertyMap, dims: tuple[str, ...]) -> tuple[str, ...]:
    if _is_edge_trace(dims):
        return edge_labels(pmap)
    return pmap.labels()


def _require_uniform_odd(times: np.ndarray, *, op: str) -> float:
    """Return the common ``dt`` or raise for Simpson's rule."""
    if times.size < 3:
        raise ValueError(f"{op}(method='simpson') needs at least 3 time points, got {times.size}.")
    if times.size % 2 == 0:
        raise ValueError(
            f"{op}(method='simpson') requires an odd number of time points "
            f"(even number of intervals), got {times.size}."
        )
    dts = np.diff(times)
    dt0 = float(dts[0])
    if not np.allclose(dts, dt0, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{op}(method='simpson') requires a uniform time grid.")
    return dt0


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

    __array_priority__ = 1000

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        from summer4.flows.algebra import dispatch_ufunc

        return dispatch_ufunc(ufunc, method, inputs, kwargs, mode="trace")

    def __array_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        from summer4.flows.algebra import dispatch_array_function

        return dispatch_array_function(func, types, args, kwargs, mode="trace")

    def _map_unary(self, op: str) -> Trace:
        """Pointwise unary op, same kernel as a rate-tree :class:`UnaryOp`."""
        from summer4.flows.algebra import apply_unary

        return self._with(values=apply_unary(op, self.values))

    @staticmethod
    def _combine(left: object, right: object, op: str) -> Trace:
        """Binary op of a Trace with a scalar or array.

        The numeric kernel is :func:`summer4.flows.algebra.apply_binary`, shared with
        rate evaluation. An unevaluated rate expression (a parameter transform
        such as ``tanh(Param("s"))``) is rejected: evaluate it with
        :func:`summer4.flows.compiled.eval_closed` and combine with the array.
        Trace-to-Trace name alignment is not done here.
        """
        from summer4.flows.algebra import apply_binary
        from summer4.flows.rates import RateOps

        left_trace = isinstance(left, Trace)
        right_trace = isinstance(right, Trace)
        if left_trace and right_trace:
            raise TypeError(
                "Trace-to-Trace arithmetic needs name-aligned broadcasting, "
                "which is not implemented yet."
            )
        if not left_trace and not right_trace:
            raise TypeError("Trace._combine expects one Trace.")
        if isinstance(left, RateOps) or isinstance(right, RateOps):
            raise TypeError(
                "Cannot combine a Trace with an unevaluated rate expression. "
                "Evaluate parameter-only expressions with eval_closed(expr, params) "
                "and combine the Trace with that array."
            )
        owner = left if left_trace else right
        if not isinstance(owner, Trace):
            raise TypeError("Trace._combine expects one Trace.")
        values = apply_binary(
            op,
            left.values if isinstance(left, Trace) else left,
            right.values if isinstance(right, Trace) else right,
        )
        return owner._with(values=values)

    def __neg__(self) -> Trace:
        return self._map_unary("neg")

    def __abs__(self) -> Trace:
        return self._map_unary("abs")

    def __add__(self, other: object) -> Trace:
        return Trace._combine(self, other, "add")

    def __radd__(self, other: object) -> Trace:
        return Trace._combine(other, self, "add")

    def __sub__(self, other: object) -> Trace:
        return Trace._combine(self, other, "sub")

    def __rsub__(self, other: object) -> Trace:
        return Trace._combine(other, self, "sub")

    def __mul__(self, other: object) -> Trace:
        return Trace._combine(self, other, "mul")

    def __rmul__(self, other: object) -> Trace:
        return Trace._combine(other, self, "mul")

    def __truediv__(self, other: object) -> Trace:
        return Trace._combine(self, other, "div")

    def __rtruediv__(self, other: object) -> Trace:
        return Trace._combine(other, self, "div")

    def __pow__(self, other: object) -> Trace:
        return Trace._combine(self, other, "pow")

    def __rpow__(self, other: object) -> Trace:
        return Trace._combine(other, self, "pow")

    def _pmap(self) -> PropertyMap | None:
        return self.values.pmap if isinstance(self.values, PropertyData) else None

    # --- compartment / edge selection -------------------------------------------------

    def select(self, sel: Selector | np.ndarray) -> Trace:
        """Return a Trace restricted where ``sel`` is Kleene-true.

        On an edge trace (``\"edge\"`` in ``dims``), ``Source`` / ``Dest``
        selectors are rewritten against the edge table. A boolean mask of
        length ``pmap.size`` is also accepted (e.g. ``EdgeMap.moves_mask``).
        Index selection is host-side and static; only the gather is traced.
        """
        pmap = self._pmap()
        if pmap is None:
            raise TypeError("select() requires PropertyData values.")
        if isinstance(sel, np.ndarray):
            if sel.shape != (pmap.size,):
                raise ValueError(
                    f"Boolean mask shape {sel.shape} does not match aligned size {pmap.size}."
                )
            idx = np.flatnonzero(np.asarray(sel, dtype=np.bool_)).astype(np.int32, copy=False)
        elif _is_edge_trace(self.dims):
            idx = pmap.select(rewrite_edge_selector(pmap, sel))
        else:
            idx = pmap.select(sel)
        submap = pmap.take(idx)
        data = _as_array(self.values)[..., idx]
        return self._with(values=PropertyData(submap, data))

    def sum_over(
        self,
        prop: Property | str,
        side: Literal["source", "dest"] | None = None,
    ) -> Trace:
        """Sum the aligned axis by trait of ``prop``.

        On an edge trace, ``side`` is required (``\"source\"`` or ``\"dest\"``) —
        there is no safe default. On a compartment trace, ``side`` must be
        omitted.
        """
        pmap = self._pmap()
        if pmap is None:
            raise TypeError("sum_over() requires PropertyData values.")
        if _is_edge_trace(self.dims):
            if side is None:
                raise ValueError(
                    "sum_over() on a flow (edge) trace requires side='source' or "
                    "side='dest'; neither is a safe default."
                )
            reduced = sum_over_edge(_as_array(self.values), pmap, prop, side)
        else:
            if side is not None:
                raise ValueError("side= is only valid on edge traces.")
            reduced = PropertyData(pmap, _as_array(self.values)).sum_over(prop)
        dims = tuple("group" if d in ("compartment", "edge") else d for d in self.dims)
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
            kind=TimeAxisKind.EXPLICIT,
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
        data_t = xp.moveaxis(data, t_ax, 0)
        left = data_t[idx[:, 0]]
        right = data_t[idx[:, 1]]
        w0 = xp.asarray(w[:, 0]).reshape((-1,) + (1,) * (left.ndim - 1))
        w1 = xp.asarray(w[:, 1]).reshape((-1,) + (1,) * (left.ndim - 1))
        out = w0 * left + w1 * right
        out = xp.moveaxis(out, 0, t_ax)
        new_times = TimeAxis(values=targets, epoch=self.times.epoch, kind=TimeAxisKind.EXPLICIT)
        if isinstance(self.values, PropertyData):
            out = PropertyData(self.values.pmap, out)
        return self._with(values=out, times=new_times)

    # --- reductions along time --------------------------------------------------------

    def reduce_by(self, grouping: TimeGrouping, how: ReduceHowArg = ReduceHow.SUM) -> Trace:
        """Segment-reduce along time using a static :class:`TimeGrouping`."""
        from summer4.enums import coerce_strenum

        xp = _xp()
        import jax

        resolved = coerce_strenum(ReduceHow, how, what="Trace.reduce_by how")
        t_ax = _time_axis_index(self.dims)
        data = _as_array(self.values)
        data_t = xp.moveaxis(data, t_ax, 0)
        flat = data_t.reshape((data_t.shape[0], -1))
        ids = np.asarray(grouping.segment_ids, dtype=np.int32)
        n = grouping.n_groups

        def _col(col: Any) -> Any:
            return jax.ops.segment_sum(col, ids, num_segments=n)

        summed = jax.vmap(_col, in_axes=1, out_axes=1)(flat)
        if resolved is ReduceHow.MEAN:
            counts = xp.asarray(grouping.counts, dtype=data_t.dtype).reshape((n, 1))
            summed = summed / xp.maximum(counts, 1)
        new_shape = (n,) + data_t.shape[1:]
        out = xp.reshape(summed, new_shape)
        out = xp.moveaxis(out, 0, t_ax)
        new_times = TimeAxis(
            values=grouping.starts, epoch=self.times.epoch, kind=TimeAxisKind.EXPLICIT
        )
        if isinstance(self.values, PropertyData):
            out = PropertyData(self.values.pmap, out)
        return self._with(values=out, times=new_times)

    def resample(self, rule: CalendarRule, how: ReduceHowArg = ReduceHow.SUM) -> Trace:
        """Sugar over :meth:`reduce_by` with a cached :class:`TimeGrouping`."""
        return self.reduce_by(self.times.grouping(rule), how=how)

    def rolling(
        self,
        window: int,
        *,
        how: ReduceHowArg = ReduceHow.MEAN,
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
                out_rows.append(total / count if spec.how is ReduceHow.MEAN else total)
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

    def incidence(self, method: QuadMethod = "trapezoid") -> Trace:
        """Integrate instantaneous rates over each save interval.

        Returns ``T-1`` rows (trapezoid) or ``(T-1)/2`` Simpson panels, with
        times at each panel's right edge.

        **Quadrature error.** Trapezoid over a save interval is second-order
        accurate; summer2's accumulated incidence is exact for the solver's own
        quadrature. For calibration against case counts that bias is real and
        is **not** recoverable from a :class:`~summer4.results.result.Result`.
        Mitigations: save finer than you calibrate and :meth:`resample`; use
        ``method='simpson'``; eventually an opt-in accumulator in
        ``State.ledgers`` (not implemented).
        """
        xp = _xp()
        t_ax = _time_axis_index(self.dims)
        times = np.asarray(self.times.values, dtype=np.float64)
        data = _as_array(self.values)
        data_t = xp.moveaxis(data, t_ax, 0)

        if method == "trapezoid":
            if times.size < 2:
                raise ValueError("incidence() needs at least 2 time points.")
            dt = xp.asarray(np.diff(times)).reshape((-1,) + (1,) * (data_t.ndim - 1))
            panels = (data_t[:-1] + data_t[1:]) * (0.5 * dt)
            new_t = times[1:]
        elif method == "simpson":
            dt0 = _require_uniform_odd(times, op="incidence")
            y0 = data_t[:-2:2]
            y1 = data_t[1:-1:2]
            y2 = data_t[2::2]
            panels = (dt0 / 3.0) * (y0 + 4.0 * y1 + y2)
            new_t = times[2::2]
        else:
            raise ValueError(f"Unknown incidence method {method!r}.")

        out = xp.moveaxis(panels, 0, t_ax)
        new_times = TimeAxis(values=new_t, epoch=self.times.epoch, kind=TimeAxisKind.EXPLICIT)
        return self._with(values=_wrap_like(self.values, out), times=new_times)

    def integrate(self, method: QuadMethod = "trapezoid") -> Trace:
        """Integrate over the whole time window; drops the ``time`` axis.

        See :meth:`incidence` for the quadrature-error caveats. Simpson
        requires a uniform grid with an odd point count.
        """
        xp = _xp()
        t_ax = _time_axis_index(self.dims)
        times = np.asarray(self.times.values, dtype=np.float64)
        data = _as_array(self.values)
        data_t = xp.moveaxis(data, t_ax, 0)

        if method == "trapezoid":
            if times.size < 2:
                raise ValueError("integrate() needs at least 2 time points.")
            dt = xp.asarray(np.diff(times)).reshape((-1,) + (1,) * (data_t.ndim - 1))
            panels = (data_t[:-1] + data_t[1:]) * (0.5 * dt)
            total = xp.sum(panels, axis=0)
        elif method == "simpson":
            dt0 = _require_uniform_odd(times, op="integrate")
            weights = np.empty(times.size, dtype=np.float64)
            weights[0] = 1.0
            weights[-1] = 1.0
            weights[1:-1:2] = 4.0
            weights[2:-1:2] = 2.0
            w = xp.asarray(weights).reshape((-1,) + (1,) * (data_t.ndim - 1))
            total = (dt0 / 3.0) * xp.sum(w * data_t, axis=0)
        else:
            raise ValueError(f"Unknown integrate method {method!r}.")

        dims = self.dims[:t_ax] + self.dims[t_ax + 1 :]
        new_times = TimeAxis(
            values=np.asarray([float(times[-1])], dtype=np.float64),
            epoch=self.times.epoch,
            kind=TimeAxisKind.EXPLICIT,
        )
        return Trace(times=new_times, values=_wrap_like(self.values, total), dims=dims)

    # --- host-side export -------------------------------------------------------------

    def to_frame(self) -> object:
        """Return a polars DataFrame (lazy import)."""
        import polars as pl

        values = np.asarray(_as_array(self.values))
        times = np.asarray(self.times.values, dtype=np.float64)
        pmap = self._pmap()

        if "time" not in self.dims:
            flat = values.reshape(1, -1) if values.size else np.zeros((1, 0))
            cols: dict[str, Any] = {"time": times[:1]}
            if pmap is not None and flat.shape[1] == pmap.size:
                for j, lab in enumerate(_column_labels(pmap, self.dims)):
                    cols[lab] = flat[:, j]
            else:
                for j in range(flat.shape[1]):
                    cols[f"v{j}"] = flat[:, j]
            return pl.DataFrame(cols)

        t_ax = _time_axis_index(self.dims)
        if values.ndim == 1 and t_ax == 0:
            return pl.DataFrame({"time": times, "value": values})
        if t_ax != 0:
            values = np.moveaxis(values, t_ax, 0)
        n_t = values.shape[0]
        flat = values.reshape(n_t, -1)
        cols = {"time": times[:n_t]}
        if pmap is not None and flat.shape[1] == pmap.size:
            for j, lab in enumerate(_column_labels(pmap, self.dims)):
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
