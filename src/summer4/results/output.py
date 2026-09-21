"""Named outputs from a solve and their query surface.

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
    _is_tracer,
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
        raise ValueError(f"Output dims {dims} have no 'time' axis.") from exc


def _aligned_axis_index(dims: tuple[str, ...]) -> int | None:
    for name in ("compartment", "edge", "group"):
        if name in dims:
            return dims.index(name)
    return None


def _is_edge_output(dims: tuple[str, ...]) -> bool:
    return "edge" in dims


def _wrap_like(template: PropertyData | Any, data: Any) -> PropertyData | Any:
    if isinstance(template, PropertyData):
        return PropertyData(template.pmap, data)
    return data


def _column_labels(pmap: PropertyMap, dims: tuple[str, ...]) -> tuple[str, ...]:
    if _is_edge_output(dims):
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


_ALIGNED_DIMS = frozenset({"compartment", "edge", "group"})


def _describe_times(axis: TimeAxis) -> str:
    """Short host-side description of a time axis for error messages."""
    if _is_tracer(axis.values):
        return f"traced shape {getattr(axis.values, 'shape', None)} epoch={axis.epoch!r}"
    vals = np.asarray(axis.values, dtype=np.float64)
    if vals.size <= 6:
        shown: list[float | str] = vals.tolist()
    else:
        shown = vals[:3].tolist() + ["…"] + vals[-1:].tolist()
    return f"{shown} epoch={axis.epoch!r}"


def _require_same_times(left: TimeAxis, right: TimeAxis) -> None:
    """Raise when two outputs are not on the same save times.

    Concrete axes must match value-for-value. Traced axes (a ``Result`` passed
    into ``jit``) are compared by shape: the values are not a Python object
    the check can read.
    """
    if left.epoch != right.epoch:
        raise ValueError(
            "Output time axes differ: " f"{_describe_times(left)} vs {_describe_times(right)}."
        )
    lv, rv = left.values, right.values
    if not _is_tracer(lv) and not _is_tracer(rv):
        la = np.asarray(lv, dtype=np.float64)
        ra = np.asarray(rv, dtype=np.float64)
        if la.shape != ra.shape or not np.array_equal(la, ra):
            raise ValueError(
                "Output time axes differ: " f"{_describe_times(left)} vs {_describe_times(right)}."
            )
        return
    if getattr(lv, "shape", None) != getattr(rv, "shape", None):
        raise ValueError(
            "Output time axes differ: " f"{_describe_times(left)} vs {_describe_times(right)}."
        )


def _dim_names(dims: tuple[str, ...]) -> set[str]:
    if len(set(dims)) != len(dims):
        raise ValueError(f"Output dims {dims} repeat a name.")
    return set(dims)


def _broadcast_dim_order(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    """Dim order of an Output ∘ Output: the richer side, or the left if equal."""
    left_names = _dim_names(left)
    right_names = _dim_names(right)
    if left_names == right_names:
        return left
    if left_names < right_names:
        return right
    if right_names < left_names:
        return left
    raise ValueError(
        f"Output dims {left} and {right} are not broadcastable. "
        "One side's names must equal the other's, or be a subset "
        "(for example a per-age output divided by a total)."
    )


def _broadcast_named(arr: Any, src: tuple[str, ...], dst: tuple[str, ...]) -> Any:
    """Move ``arr`` onto ``dst`` dim order, inserting size-1 axes for missing names."""
    if arr.ndim != len(src):
        raise ValueError(f"Output values have ndim {arr.ndim} but dims {src} name {len(src)} axes.")
    if src == dst:
        return arr
    xp = _xp()
    missing = [name for name in dst if name not in src]
    expanded = arr
    for _ in missing:
        expanded = xp.expand_dims(expanded, axis=-1)
    order = list(src) + missing
    perm = tuple(order.index(name) for name in dst)
    if perm == tuple(range(len(dst))):
        return expanded
    return xp.transpose(expanded, perm)


def _window_bounds(n: int, window: int, center: bool) -> tuple[np.ndarray, np.ndarray]:
    """Host-side ``[lo, hi)`` indexes for each time of a rolling window."""
    idx = np.arange(n, dtype=np.int32)
    left = idx - (window // 2) if center else idx - window + 1
    right = left + window
    lo = np.clip(left, 0, n).astype(np.int32, copy=False)
    hi = np.clip(right, 0, n).astype(np.int32, copy=False)
    return lo, hi


def _shared_pmap(left: Output, right: Output) -> PropertyMap | None:
    """Map to keep on a combined output, or None when the values stay raw.

    Both maps must be equal. A single map is kept when the other output has
    no aligned axis, so a per-age series divided by a total stays selectable.
    """
    left_map = left.values.pmap if isinstance(left.values, PropertyData) else None
    right_map = right.values.pmap if isinstance(right.values, PropertyData) else None
    if left_map is None and right_map is None:
        return None
    if left_map is not None and right_map is not None:
        if left_map != right_map:
            raise ValueError(
                "Output operands do not share a PropertyMap "
                f"(dims {left.dims} and {right.dims})."
            )
        return left_map
    other = right if left_map is not None else left
    if _ALIGNED_DIMS.intersection(other.dims):
        return None
    return left_map if left_map is not None else right_map


def _exact_save_index(times: np.ndarray, when: When, epoch: Any, *, bound: str) -> int:
    """Index of ``when`` on ``times``. ``when`` must be a save time."""
    if isinstance(when, (int, float, np.floating)):
        t = float(when)
    else:
        if epoch is None:
            raise TypeError(
                f"cumulative {bound}={when!r} needs TimeAxis.epoch to become a model time."
            )
        t = float(np.asarray(epoch.to_model(when)).reshape(-1)[0])
    hits = np.flatnonzero(np.isclose(times, t, rtol=0.0, atol=0.0))
    if hits.size != 1:
        hits = np.flatnonzero(np.isclose(times, t, rtol=0.0, atol=1e-8))
    if hits.size != 1:
        raise ValueError(
            f"cumulative {bound}={when!r} (model time {t}) is not a save time. "
            f"Save times run from {float(times[0])} to {float(times[-1])} "
            f"({times.size} points)."
        )
    return int(hits[0])


@dataclass(frozen=True, slots=True)
class Output:
    """One named quantity from a solve — the unit the query surface operates on."""

    times: TimeAxis
    values: PropertyData | Any
    dims: tuple[str, ...]

    def _with(
        self,
        *,
        values: PropertyData | Any | None = None,
        dims: tuple[str, ...] | None = None,
        times: TimeAxis | None = None,
    ) -> Output:
        return Output(
            times=self.times if times is None else times,
            values=self.values if values is None else values,
            dims=self.dims if dims is None else dims,
        )

    __array_priority__ = 1000

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        from summer4.flows.algebra import dispatch_ufunc

        return dispatch_ufunc(ufunc, method, inputs, kwargs, mode="output")

    def __array_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        from summer4.flows.algebra import dispatch_array_function

        return dispatch_array_function(func, types, args, kwargs, mode="output")

    def _map_unary(self, op: str) -> Output:
        """Pointwise unary op, same kernel as a rate-tree :class:`UnaryOp`."""
        from summer4.flows.algebra import apply_unary

        return self._with(values=apply_unary(op, self.values))

    @staticmethod
    def _combine(left: object, right: object, op: str) -> Output:
        """Binary op of an Output with a scalar, an array, or another Output.

        Output ∘ Output requires the same time axis. Dim names broadcast when
        they are equal or one side's names are a subset of the other's (a
        per-age series divided by a total). Alignment is host-side; the
        arithmetic is traced through :func:`summer4.flows.algebra.apply_binary`.

        The result stays :class:`~summer4.jax.propertydata.PropertyData` when
        the operands share one map, or exactly one operand has a map and the
        other has no aligned axis (per-age ÷ total). Otherwise the values are
        a raw array.

        An unevaluated rate expression (a parameter transform such as
        ``tanh(Param("s"))``) is rejected: evaluate it with
        :func:`summer4.flows.compiled.eval_closed` and combine with the array.
        """
        from summer4.flows.algebra import apply_binary
        from summer4.flows.rates import RateOps

        left_is_output = isinstance(left, Output)
        right_is_output = isinstance(right, Output)
        if not left_is_output and not right_is_output:
            raise TypeError("Output._combine expects one Output.")
        if isinstance(left, RateOps) or isinstance(right, RateOps):
            raise TypeError(
                "Cannot combine an Output with an unevaluated rate expression. "
                "Evaluate parameter-only expressions with eval_closed(expr, params) "
                "and combine the Output with that array."
            )
        if left_is_output and right_is_output:
            assert isinstance(left, Output) and isinstance(right, Output)
            return Output._combine_outputs(left, right, op)
        owner = left if left_is_output else right
        if not isinstance(owner, Output):
            raise TypeError("Output._combine expects one Output.")
        values = apply_binary(
            op,
            left.values if isinstance(left, Output) else left,
            right.values if isinstance(right, Output) else right,
        )
        return owner._with(values=values)

    @staticmethod
    def _combine_outputs(left: Output, right: Output, op: str) -> Output:
        from summer4.flows.algebra import apply_binary

        _require_same_times(left.times, right.times)
        dims = _broadcast_dim_order(left.dims, right.dims)
        left_arr = _broadcast_named(_as_array(left.values), left.dims, dims)
        right_arr = _broadcast_named(_as_array(right.values), right.dims, dims)
        values = apply_binary(op, left_arr, right_arr)
        pmap = _shared_pmap(left, right)
        if (
            pmap is not None
            and dims
            and dims[-1] in _ALIGNED_DIMS
            and values.shape[-1] == pmap.size
        ):
            values = PropertyData(pmap, values)
        return Output(times=left.times, values=values, dims=dims)

    def __neg__(self) -> Output:
        return self._map_unary("neg")

    def __abs__(self) -> Output:
        return self._map_unary("abs")

    def __add__(self, other: object) -> Output:
        return Output._combine(self, other, "add")

    def __radd__(self, other: object) -> Output:
        return Output._combine(other, self, "add")

    def __sub__(self, other: object) -> Output:
        return Output._combine(self, other, "sub")

    def __rsub__(self, other: object) -> Output:
        return Output._combine(other, self, "sub")

    def __mul__(self, other: object) -> Output:
        return Output._combine(self, other, "mul")

    def __rmul__(self, other: object) -> Output:
        return Output._combine(other, self, "mul")

    def __truediv__(self, other: object) -> Output:
        return Output._combine(self, other, "div")

    def __rtruediv__(self, other: object) -> Output:
        return Output._combine(other, self, "div")

    def __pow__(self, other: object) -> Output:
        return Output._combine(self, other, "pow")

    def __rpow__(self, other: object) -> Output:
        return Output._combine(other, self, "pow")

    def _pmap(self) -> PropertyMap | None:
        return self.values.pmap if isinstance(self.values, PropertyData) else None

    # --- compartment / edge selection -------------------------------------------------

    def select(self, sel: Selector | np.ndarray) -> Output:
        """Return an Output restricted where ``sel`` is Kleene-true.

        On an edge output (``\"edge\"`` in ``dims``), ``Source`` / ``Dest``
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
        elif _is_edge_output(self.dims):
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
    ) -> Output:
        """Sum the aligned axis by trait of ``prop``.

        On an edge output, ``side`` is required (``\"source\"`` or ``\"dest\"``) —
        there is no safe default. On a compartment output, ``side`` must be
        omitted.
        """
        pmap = self._pmap()
        if pmap is None:
            raise TypeError("sum_over() requires PropertyData values.")
        if _is_edge_output(self.dims):
            if side is None:
                raise ValueError(
                    "sum_over() on a flow (edge) output requires side='source' or "
                    "side='dest'; neither is a safe default."
                )
            reduced = sum_over_edge(_as_array(self.values), pmap, prop, side)
        else:
            if side is not None:
                raise ValueError("side= is only valid on edge outputs.")
            reduced = PropertyData(pmap, _as_array(self.values)).sum_over(prop)
        dims = tuple("group" if d in ("compartment", "edge") else d for d in self.dims)
        if "group" not in dims:
            dims = self.dims[:-1] + ("group",)
        return self._with(values=reduced, dims=dims)

    def total(self) -> Output:
        """Sum over the aligned (last) axis."""
        xp = _xp()
        data = xp.sum(_as_array(self.values), axis=-1)
        dims = self.dims[:-1]
        return self._with(values=data, dims=dims)

    def partition(self, prop: Property | str) -> dict[Trait, Output]:
        """Split into one Output per trait of ``prop`` (static keys; values traced)."""
        pmap = self._pmap()
        if pmap is None:
            raise TypeError("partition() requires PropertyData values.")
        out: dict[Trait, Output] = {}
        for trait, idx in pmap.partition(prop).items():
            data = _as_array(self.values)[..., idx]
            out[trait] = self._with(values=data, dims=self.dims)
        return out

    def group_by(self, *props: Property | str) -> Groups[Output]:
        """Group by one or more properties (static keys; values traced)."""
        pmap = self._pmap()
        if pmap is None:
            raise TypeError("group_by() requires PropertyData values.")
        data: dict[tuple[Trait, ...], Output] = {}
        for key, idx in pmap.group_by(*props).items():
            gathered = _as_array(self.values)[..., idx]
            data[key] = self._with(values=gathered, dims=self.dims)
        return Groups(_data=data)

    # --- time selection ---------------------------------------------------------------

    def between(self, t0: When, t1: When) -> Output:
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

    def at(self, when: When) -> Output:
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

    def at_times(self, ts: Any) -> Output:
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

    def reduce_by(self, grouping: TimeGrouping, how: ReduceHowArg = ReduceHow.SUM) -> Output:
        """Segment-reduce along time using a static :class:`TimeGrouping`."""
        from summer4.enums import coerce_strenum

        xp = _xp()
        import jax

        resolved = coerce_strenum(ReduceHow, how, what="Output.reduce_by how")
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

    def resample(self, rule: CalendarRule, how: ReduceHowArg = ReduceHow.SUM) -> Output:
        """Sugar over :meth:`reduce_by` with a cached :class:`TimeGrouping`."""
        return self.reduce_by(self.times.grouping(rule), how=how)

    def rolling(
        self,
        window: int,
        *,
        how: ReduceHowArg = ReduceHow.MEAN,
        center: bool = False,
        min_periods: int | None = None,
    ) -> Output:
        """Rolling window along time.

        Sum and mean are a cumulative-sum difference. The window indexes are
        built on the host from the concrete time axis, so the traced program
        is one gather and does not grow with trajectory length. A window that
        has fewer than ``min_periods`` points is NaN.
        """
        spec = self.times.rolling(window, how=how, center=center, min_periods=min_periods)
        return self._apply_rolling(spec)

    def _apply_rolling(self, spec: RollingSpec) -> Output:
        xp = _xp()
        t_ax = _time_axis_index(self.dims)
        data = _as_array(self.values)
        n = int(self.times._concrete(op="rolling").size)
        data_t = xp.moveaxis(data, t_ax, 0)
        if int(data_t.shape[0]) != n:
            raise ValueError(f"Time axis length {data_t.shape[0]} does not match {n} save times.")
        lo, hi = _window_bounds(n, spec.window, bool(spec.center))
        zeros = xp.zeros((1,) + tuple(data_t.shape[1:]), dtype=data_t.dtype)
        csum = xp.concatenate([zeros, xp.cumsum(data_t, axis=0)], axis=0)
        # Concrete index vectors: one gather, not one op per time.
        totals = csum[hi] - csum[lo]
        counts_np = (hi - lo).astype(np.int32, copy=False)
        trailing = (1,) * (data_t.ndim - 1)
        counts = xp.asarray(counts_np).reshape((n,) + trailing)
        if spec.how is ReduceHow.MEAN:
            filled = totals / xp.maximum(counts, 1)
        elif spec.how is ReduceHow.SUM:
            filled = totals
        else:
            raise ValueError(f"Unknown rolling reduction {spec.how!r}.")
        valid = xp.asarray(counts_np >= int(spec.min_periods)).reshape((n,) + trailing)
        out = xp.where(valid, filled, xp.nan)
        out = xp.moveaxis(out, 0, t_ax)
        if isinstance(self.values, PropertyData):
            out = PropertyData(self.values.pmap, out)
        return self._with(values=out)

    def cumulative(self, *, start: When | None = None, end: When | None = None) -> Output:
        """Cumulative sum along time.

        With no bounds this is ``cumsum``. ``start`` and ``end`` must be save
        times on this output (summer2 ``request_cumulative_output(start_time=)``
        requires the same). Points before ``start`` and after ``end`` are zero.
        The running sum includes only that window, so the value at ``start`` is
        that point itself.
        """
        xp = _xp()
        t_ax = _time_axis_index(self.dims)
        data = _as_array(self.values)
        if start is None and end is None:
            out = xp.cumsum(data, axis=t_ax)
            return self._with(values=_wrap_like(self.values, out))
        times = self.times._concrete(op="cumulative")
        n = int(times.size)
        i0 = (
            0 if start is None else _exact_save_index(times, start, self.times.epoch, bound="start")
        )
        i1 = n - 1 if end is None else _exact_save_index(times, end, self.times.epoch, bound="end")
        if i1 < i0:
            raise ValueError(f"cumulative end is before start (save indexes {i0} and {i1}).")
        data_t = xp.moveaxis(data, t_ax, 0)
        if int(data_t.shape[0]) != n:
            raise ValueError(f"Time axis length {data_t.shape[0]} does not match {n} save times.")
        csum = xp.cumsum(data_t, axis=0)
        window = csum if i0 == 0 else csum - csum[i0 - 1]
        idx = xp.arange(n)
        mask = (idx >= i0) & (idx <= i1)
        mask = mask.reshape((n,) + (1,) * (data_t.ndim - 1))
        out = xp.where(mask, window, xp.zeros_like(window))
        out = xp.moveaxis(out, 0, t_ax)
        return self._with(values=_wrap_like(self.values, out))

    def midpoint(self) -> Output:
        """Summer2's default flow-output convention (``raw_results=False``).

        ``out[0]`` equals the first sample and later points are the average of
        the sample and the one before it: ``out[i] = (f[i] + f[i - 1]) / 2``.
        Use this when a number has to match summer2. It is a parity convention,
        not the solver's accumulated incidence — see :meth:`incidence`.
        """
        xp = _xp()
        t_ax = _time_axis_index(self.dims)
        data = _as_array(self.values)
        data_t = xp.moveaxis(data, t_ax, 0)
        prev = xp.concatenate([data_t[:1], data_t[:-1]], axis=0)
        out = xp.moveaxis((data_t + prev) / 2, 0, t_ax)
        return self._with(values=_wrap_like(self.values, out))

    def incidence(self, method: QuadMethod = "trapezoid") -> Output:
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

    def integrate(self, method: QuadMethod = "trapezoid") -> Output:
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
        return Output(times=new_times, values=_wrap_like(self.values, total), dims=dims)

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
