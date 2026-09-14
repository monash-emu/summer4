"""Compiled flow models and JAX vector fields."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from summer4.flows.actualize import (
    EntryEdges,
    ExitEdges,
    FlowEdges,
    TransitionEdges,
    actualize,
    topo_sort,
)
from summer4.flows.edges import EdgeMap
from summer4.flows.rates import (
    BinOp,
    Const,
    FieldRef,
    FlowRef,
    Overwrite,
    RateOps,
    Transform,
    _adjust_bytes,
    _adjust_field_paths,
    _field_paths,
    _lookup_path,
    _rate_bytes,
)
from summer4.flows.types import FlowLike
from summer4.properties import Property
from summer4.propertymap import PropertyMap

_NA: int = -1


class DerivedFn(Protocol):
    """``derived_fn(params, *, y, t)`` producing the derived-param struct."""

    def __call__(self, params: object, *, y: object, t: object) -> object: ...


@dataclass(frozen=True, slots=True)
class _SubmapRate:
    """Rate whose last axis is aligned to one property's traits."""

    data: object
    properties: tuple[Property, ...]


def _gather_idx(flow: FlowEdges) -> NDArray[np.int32]:
    match flow:
        case EntryEdges(dest_idx=idx):
            return idx
        case TransitionEdges(src_idx=idx) | ExitEdges(src_idx=idx):
            return idx


def _pair_info(
    flow: FlowEdges,
) -> tuple[NDArray[np.int32] | None, NDArray[np.int32] | None, int | None]:
    match flow:
        case TransitionEdges(pair_src_codes=src, pair_dest_codes=dest, pair_n_traits=n):
            return src, dest, n
        case _:
            return None, None, None


def _sum_mass_over(
    mass: Any,
    edge_idx: NDArray[np.int32],
    pmap: PropertyMap,
    prop: Property,
) -> Any:
    """Segment-sum edge mass by ``prop`` trait codes (skip absent)."""
    import jax
    import jax.numpy as jnp

    col_i = pmap.column_index(prop)
    codes = np.asarray(pmap.codes[edge_idx, col_i], dtype=np.int32)
    n_traits = len(prop.traits)
    valid = codes >= 0
    safe = np.where(valid, codes, 0)
    mass_j = jnp.asarray(mass)
    valid_j = jnp.asarray(valid)
    safe_j = jnp.asarray(safe)
    weighted = mass_j * valid_j
    n_edges = mass_j.shape[-1]

    def _row(row: Any) -> Any:
        return jax.ops.segment_sum(row, safe_j, num_segments=n_traits)

    if mass_j.ndim == 1:
        return _row(weighted)
    flat = jnp.reshape(weighted, (-1, n_edges))
    summed = jax.vmap(_row)(flat)
    return jnp.reshape(summed, mass_j.shape[:-1] + (n_traits,))


def _eval_rate(
    expr: RateOps,
    *,
    derived: Any,
    flow_values: Mapping[str, Any],
    flow_meta: Mapping[str, FlowEdges],
    pmap: PropertyMap,
) -> Any:
    import jax.numpy as jnp

    match expr:
        case Const(value=value):
            return value
        case FieldRef(path=path):
            return _lookup_path(derived, path)
        case FlowRef(name=name, reduce=reduce):
            if name not in flow_values:
                raise KeyError(f"Flow {name!r} has not been evaluated yet.")
            values = flow_values[name]
            if reduce == "sum":
                return jnp.sum(jnp.asarray(values))
            if isinstance(reduce, tuple) and reduce[0] == "sum_over":
                producer = flow_meta[name]
                edge_idx = _gather_idx(producer)
                prop = pmap.get_property(reduce[1])
                data = _sum_mass_over(values, edge_idx, pmap, prop)
                return _SubmapRate(data=data, properties=(prop,))
            return values
        case BinOp(op=op, left=left, right=right):
            left_v = _eval_rate(
                left,
                derived=derived,
                flow_values=flow_values,
                flow_meta=flow_meta,
                pmap=pmap,
            )
            right_v = _eval_rate(
                right,
                derived=derived,
                flow_values=flow_values,
                flow_meta=flow_meta,
                pmap=pmap,
            )
            if op == "add":
                return left_v + right_v
            if op == "sub":
                return left_v - right_v
            if op == "mul":
                return left_v * right_v
            return left_v / right_v
        case _:
            raise TypeError(f"Unsupported rate expression {type(expr).__name__}.")


def _as_array(value: Any) -> Any:
    import jax.numpy as jnp

    from summer4.jax.propertydata import PropertyData

    if isinstance(value, _SubmapRate):
        return value.data
    if isinstance(value, PropertyData):
        return value.data
    return jnp.asarray(value)


def _align_submap_rate(
    rate: _SubmapRate,
    gather_idx: NDArray[np.int32],
    pmap: PropertyMap,
) -> Any:
    if len(rate.properties) != 1:
        raise ValueError("sum_over alignment supports exactly one property.")
    prop = rate.properties[0]
    col_i = pmap.column_index(prop)
    codes = np.asarray(pmap.codes[gather_idx, col_i], dtype=np.int32)
    if np.any(codes == _NA):
        raise ValueError(
            f"Cannot align sum_over({prop.name!r}): some gather rows lack that property."
        )
    data = _as_array(rate)
    return data[..., codes]


def _align_rate(
    rate: Any,
    *,
    gather_idx: NDArray[np.int32],
    n_edges: int,
    pmap: PropertyMap,
    pair_src_codes: NDArray[np.int32] | None,
    pair_dest_codes: NDArray[np.int32] | None,
    pair_n_traits: int | None,
) -> Any:
    if isinstance(rate, _SubmapRate):
        return _align_submap_rate(rate, gather_idx, pmap)
    arr = _as_array(rate)
    shape = tuple(getattr(arr, "shape", ()))
    pmap_size = pmap.size
    if (
        pair_n_traits is not None
        and pair_src_codes is not None
        and pair_dest_codes is not None
        and len(shape) >= 2
        and shape[-2] == pair_n_traits
        and shape[-1] == pair_n_traits
    ):
        return arr[..., pair_dest_codes, pair_src_codes]
    if shape == () or shape == (1,):
        return arr
    if shape[-1] == pmap_size:
        return arr[..., gather_idx]
    if shape[-1] == n_edges:
        return arr
    raise ValueError(
        f"Rate last axis {shape[-1]} matches neither pmap.size {pmap_size} "
        f"nor n_edges {n_edges}."
    )


def _eval_aligned(
    expr: RateOps,
    *,
    flow: FlowEdges,
    derived: Any,
    flow_values: Mapping[str, Any],
    flow_meta: Mapping[str, FlowEdges],
    pmap: PropertyMap,
    gather_idx: NDArray[np.int32],
) -> Any:
    raw = _eval_rate(
        expr,
        derived=derived,
        flow_values=flow_values,
        flow_meta=flow_meta,
        pmap=pmap,
    )
    pair_src, pair_dest, pair_n = _pair_info(flow)
    return _align_rate(
        raw,
        gather_idx=gather_idx,
        n_edges=int(flow.weight.size),
        pmap=pmap,
        pair_src_codes=pair_src,
        pair_dest_codes=pair_dest,
        pair_n_traits=pair_n,
    )


def _apply_adjustments(
    rate: Any,
    flow: FlowEdges,
    *,
    derived: Any,
    flow_values: Mapping[str, Any],
    flow_meta: Mapping[str, FlowEdges],
    pmap: PropertyMap,
    gather_idx: NDArray[np.int32],
) -> Any:
    import jax.numpy as jnp

    prev: Any = rate
    for adj, mask in zip(flow.adjust, flow.adjust_masks, strict=True):
        if isinstance(adj, Transform):
            args = [
                _eval_aligned(
                    arg,
                    flow=flow,
                    derived=derived,
                    flow_values=flow_values,
                    flow_meta=flow_meta,
                    pmap=pmap,
                    gather_idx=gather_idx,
                )
                for arg in adj.args
            ]
            new: Any = adj.fn(prev, *args)
        else:
            value = _eval_aligned(
                adj.value,
                flow=flow,
                derived=derived,
                flow_values=flow_values,
                flow_meta=flow_meta,
                pmap=pmap,
                gather_idx=gather_idx,
            )
            new = value if isinstance(adj, Overwrite) else prev * value
        prev = jnp.where(jnp.asarray(mask), new, prev) if mask is not None else new
    return prev


def _scatter_add(target: Any, indices: NDArray[np.int32], values: Any) -> Any:
    return target.at[..., indices].add(values)


def _flow_paths(flow: FlowEdges) -> set[tuple[str, ...]]:
    paths = _field_paths(flow.rate)
    for adj in flow.adjust:
        paths |= _adjust_field_paths(adj)
    return paths


def _flow_digest(flow: FlowEdges, edge_map: EdgeMap) -> bytes:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(flow.name.encode())
    hasher.update(type(flow).__name__.encode())
    hasher.update(edge_map.table.codes.tobytes())
    hasher.update(np.ascontiguousarray(flow.weight).tobytes())
    hasher.update(np.ascontiguousarray(flow.scale).tobytes())
    hasher.update(_rate_bytes(flow.rate))
    hasher.update(b"abs1" if flow.absolute else b"abs0")
    for adj, mask in zip(flow.adjust, flow.adjust_masks, strict=True):
        hasher.update(_adjust_bytes(adj))
        hasher.update(b"nomask" if mask is None else np.ascontiguousarray(mask).tobytes())
    return hasher.digest()


def _model_digest(
    pmap: PropertyMap,
    order: tuple[str, ...],
    flows: Mapping[str, FlowEdges],
    edge_maps: Mapping[str, EdgeMap],
    derived_fn: DerivedFn | None,
    computed_paths: tuple[tuple[str, ...], ...],
) -> bytes:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(pmap.codes.tobytes())
    if pmap.parent_row is not None:
        hasher.update(pmap.parent_row.tobytes())
    for name in order:
        hasher.update(_flow_digest(flows[name], edge_maps[name]))
    hasher.update(b"df")
    hasher.update(str(id(derived_fn) if derived_fn is not None else 0).encode())
    hasher.update(repr(computed_paths).encode())
    return hasher.digest()


def _eval_derived(derived_fn: DerivedFn | None, params: object, y_arr: Any, t: object) -> Any:
    if derived_fn is None:
        return params
    return derived_fn(params, y=y_arr, t=t)


@dataclass(frozen=True, slots=True)
class SaveContext:
    """Snapshot of one field evaluation for save functions."""

    t: Any
    y: Any
    dy: Any
    derived: Any
    flows: Mapping[str, Any]


@dataclass(frozen=True, slots=True, eq=False)
class CompiledModel:
    """Static compiled flows over one :class:`PropertyMap`.

    Entirely static: pass as ``jax.jit(..., static_argnums=0)`` rather than
    registering a pytree. The digest covers float arrays (``weight``, ``scale``,
    adjust masks) as well as topology, so two models that differ only in split
    proportions do not collide in the jit cache.

    ``derived_fn``, :class:`Transform` callables, and ``SaveFn.fn`` hash by
    identity, so a lambda defined inside a loop retraces every call.
    """

    pmap: PropertyMap
    order: tuple[str, ...]
    flows: Mapping[str, FlowEdges]
    edge_maps: Mapping[str, EdgeMap]
    derived_fn: DerivedFn | None
    computed_paths: tuple[tuple[str, ...], ...]
    _digest: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_digest",
            _model_digest(
                self.pmap,
                self.order,
                self.flows,
                self.edge_maps,
                self.derived_fn,
                self.computed_paths,
            ),
        )

    def edges(self, name: str) -> EdgeMap:
        """Return the :class:`EdgeMap` for the named flow."""
        try:
            return self.edge_maps[name]
        except KeyError:
            raise KeyError(f"Unknown flow {name!r}. Known: {list(self.edge_maps)}") from None

    def observe(self, t: object, y: object, params: object) -> SaveContext:
        """Evaluate the field once and return ``dy`` plus per-flow masses."""
        import jax.numpy as jnp

        from summer4.jax.propertydata import PropertyData
        from summer4.jax.state import State, unpack_state

        y_arr, rebox = unpack_state(y, self.pmap)
        derived = _eval_derived(self.derived_fn, params, y_arr, t)
        dy: Any = jnp.zeros_like(y_arr)
        flow_values: dict[str, Any] = {}
        pmap = self.pmap
        for name in self.order:
            flow = self.flows[name]
            gather = _gather_idx(flow)
            rate: Any = _eval_aligned(
                flow.rate,
                flow=flow,
                derived=derived,
                flow_values=flow_values,
                flow_meta=self.flows,
                pmap=pmap,
                gather_idx=gather,
            )
            rate = rate * jnp.asarray(flow.scale)
            rate = _apply_adjustments(
                rate,
                flow,
                derived=derived,
                flow_values=flow_values,
                flow_meta=self.flows,
                pmap=pmap,
                gather_idx=gather,
            )
            weight = jnp.asarray(flow.weight)
            match flow:
                case EntryEdges(dest_idx=dest_idx):
                    mass = rate * weight
                    dy = _scatter_add(dy, dest_idx, mass)
                    flow_values[name] = mass
                case ExitEdges(src_idx=src_idx):
                    src_y = y_arr[..., src_idx]
                    contrib = rate if flow.absolute else rate * src_y
                    mass = contrib * weight
                    dy = _scatter_add(dy, src_idx, -mass)
                    flow_values[name] = mass
                case TransitionEdges(src_idx=src_idx, dest_idx=dest_idx):
                    src_y = y_arr[..., src_idx]
                    contrib = rate if flow.absolute else rate * src_y
                    mass = contrib * weight
                    dy = _scatter_add(dy, src_idx, -mass)
                    dy = _scatter_add(dy, dest_idx, mass)
                    flow_values[name] = mass
        y_boxed = rebox(y_arr) if not isinstance(y, (PropertyData, State)) else y
        if isinstance(y, State):
            y_out: Any = y
            dy_out: Any = State(
                compartments=PropertyData(y.compartments.pmap, dy),
                ledgers=y.ledgers,
            )
        elif isinstance(y, PropertyData):
            y_out = y
            dy_out = y._with_data(dy)
        else:
            y_out = y_boxed
            dy_out = dy
        return SaveContext(t=t, y=y_out, dy=dy_out, derived=derived, flows=flow_values)

    def vector_field(self, t: object, y: object, params: object) -> Any:
        """Return ``dy/dt`` for state ``y`` (JAX arrays, PropertyData, or State)."""
        return self.observe(t, y, params).dy

    def expand(self, plan: Any) -> Any:
        """Fill an empty (EVERYTHING) plan with compartments, flows, and computed paths."""
        from summer4.results.plan import (
            Compartments,
            ComputedValue,
            FlowMass,
            SavePlan,
            SaveRequest,
        )

        if not isinstance(plan, SavePlan):
            raise TypeError(f"Expected SavePlan, got {type(plan).__name__}.")
        if plan.requests:
            return plan
        requests: dict[str, SaveRequest] = {
            "compartments": SaveRequest(Compartments()),
        }
        for name in self.order:
            requests[name] = SaveRequest(FlowMass(flow=name))
        for path in self.computed_paths:
            key = ".".join(path)
            requests[key] = SaveRequest(ComputedValue(path=path))
        return SavePlan(
            requests=requests,
            ts=plan.ts,
            dense=plan.dense,
            solver_stats=plan.solver_stats,
        )

    def describe(
        self,
        plan: Any,
        *,
        t0: float = 0.0,
        y0: object | None = None,
        n_saves: int | None = None,
    ) -> Any:
        """Return output shapes and memory footprint without solving."""
        import jax

        from summer4.jax.propertydata import PropertyData
        from summer4.results.eval import build_save_fn
        from summer4.results.plan import OutputShape, PlanDescription, SavePlan

        expanded = self.expand(plan)
        if not isinstance(expanded, SavePlan):
            raise TypeError("expand must return SavePlan")
        if y0 is None:
            y0 = PropertyData.wrap(self.pmap, np.zeros(self.pmap.size))
        save_fn = build_save_fn(expanded, pmap=self.pmap, edge_maps=dict(self.edge_maps))

        def _one(t: object, y: object, params: object) -> Any:
            ctx = self.observe(t, y, params)
            return save_fn(ctx)

        try:
            shaped = jax.eval_shape(_one, t0, y0, None)
        except Exception as exc:
            raise ValueError(
                f"SavePlan failed shape inference (SaveFn must return statically "
                f"shaped arrays): {exc}"
            ) from exc

        if n_saves is not None:
            n = int(n_saves)
        elif expanded.ts is not None:
            n = int(np.asarray(expanded.ts).size)
        else:
            n = 1

        outputs: list[OutputShape] = []
        total = 0
        for key in expanded.requests:
            leaf = shaped[key]
            if hasattr(leaf, "data"):
                leaf = leaf.data
            shape = tuple(int(s) for s in getattr(leaf, "shape", ()))
            full_shape = (n, *shape)
            dtype = str(getattr(leaf, "dtype", "float64"))
            itemsize = np.dtype(dtype).itemsize if dtype else 8
            nbytes = int(np.prod(full_shape)) * int(itemsize)
            outputs.append(OutputShape(key=key, shape=full_shape, dtype=dtype, nbytes=nbytes))
            total += nbytes
        return PlanDescription(outputs=tuple(outputs), n_saves=n, total_nbytes=total)

    def run(
        self,
        params: object,
        y0: object,
        *,
        t0: float,
        t1: float | None = None,
        dt: float,
        steps: int | None = None,
        save: Any = None,
        solver: str = "euler",
        epoch: Any = None,
    ) -> Any:
        """Integrate and return a :class:`~summer4.results.Result`.

        Exactly one of ``t1`` / ``steps``. ``euler`` stays the low-level seam;
        Phase 3 swaps diffrax in behind ``solver=``.
        """
        from summer4.results.eval import dims_for_quantity
        from summer4.results.plan import EVERYTHING, SavePlan
        from summer4.results.result import Result, SolverInfo
        from summer4.results.trace import Trace
        from summer4.time import Epoch, TimeAxis

        if save is None:
            save = EVERYTHING
        if not isinstance(save, SavePlan):
            raise TypeError(f"save must be a SavePlan, got {type(save).__name__}.")
        if solver != "euler":
            raise ValueError(f"Unsupported solver {solver!r}; Phase 2 supports 'euler' only.")
        if (t1 is None) == (steps is None):
            raise ValueError("Provide exactly one of t1 or steps.")
        if steps is None:
            assert t1 is not None
            if dt <= 0:
                raise ValueError(f"dt must be > 0, got {dt}.")
            steps = int(round((float(t1) - float(t0)) / float(dt)))
            if steps < 0:
                raise ValueError("t1 must be >= t0.")
        n_steps = int(steps)

        expanded = self.expand(save)
        # Default save grid: include t0 and every step endpoint
        if expanded.ts is None:
            ts = t0 + dt * np.arange(n_steps + 1, dtype=np.float64)
        else:
            ts = np.asarray(expanded.ts, dtype=np.float64)

        times = TimeAxis(
            values=ts,
            epoch=epoch if isinstance(epoch, Epoch) else epoch,
            kind="grid",
        )

        saved = _euler_save(
            self,
            t0=t0,
            y0=y0,
            params=params,
            dt=float(dt),
            steps=n_steps,
            ts=ts,
            plan=expanded,
        )

        traces: dict[str, Trace] = {}
        for key, req in expanded.requests.items():
            dims = dims_for_quantity(req.what)
            raw = saved[key]
            from summer4.jax.propertydata import PropertyData
            from summer4.results.plan import Compartments

            if (
                isinstance(req.what, Compartments)
                and req.what.sum_over is None
                and req.what.where is None
            ):
                values: Any = PropertyData(self.pmap, raw)
            else:
                values = raw
            traces[key] = Trace(times=times, values=values, dims=dims)

        solver_info = (
            SolverInfo(solver="euler", num_steps=n_steps) if expanded.solver_stats else None
        )
        return Result(times=times, traces=traces, solver=solver_info)

    def __hash__(self) -> int:
        return hash(self._digest)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompiledModel):
            return NotImplemented
        return self._digest == other._digest


def _is_arithmetic_subgrid(ts: NDArray[np.float64], t0: float, dt: float) -> bool:
    """True when ``ts`` is an arithmetic subgrid of the Euler step grid including t0."""
    if ts.size == 0:
        return False
    if abs(float(ts[0]) - float(t0)) > 1e-9 * max(1.0, abs(t0)):
        return False
    steps = (ts - t0) / dt
    if not np.allclose(steps, np.round(steps), atol=1e-6):
        return False
    rounded = np.round(steps).astype(np.int64)
    if rounded.size >= 2:
        diffs = np.diff(rounded)
        if not np.all(diffs == diffs[0]) or diffs[0] <= 0:
            return False
    return True


def _euler_save(
    model: CompiledModel,
    *,
    t0: float,
    y0: object,
    params: object,
    dt: float,
    steps: int,
    ts: NDArray[np.float64],
    plan: Any,
) -> dict[str, Any]:
    """Integrate with Euler and evaluate the save plan on ``ts``."""
    import jax
    import jax.numpy as jnp

    from summer4.jax.propertydata import PropertyData
    from summer4.jax.state import State, unpack_state
    from summer4.results.eval import build_save_fn

    save_fn = build_save_fn(plan, pmap=model.pmap, edge_maps=dict(model.edge_maps))
    y_arr, rebox = unpack_state(y0, model.pmap)

    def observe_arr(t: Any, y: Any) -> Any:
        y_in = rebox(y)
        return model.observe(t, y_in, params)

    def snapshot(t: Any, y: Any) -> dict[str, Any]:
        ctx = observe_arr(t, y)
        raw = save_fn(ctx)
        out: dict[str, Any] = {}
        for k, v in raw.items():
            out[k] = v.data if isinstance(v, PropertyData) else jnp.asarray(v)
        return out

    # Shape template
    init_snap = snapshot(jnp.asarray(t0), y_arr)

    # Nested scan fast path when ts is an arithmetic subgrid of the step grid.
    if _is_arithmetic_subgrid(ts, t0, dt):
        stride = int(round((float(ts[1]) - float(ts[0])) / dt)) if ts.size > 1 else max(steps, 1)
        n_saves = int(ts.size)

        def outer_body(
            carry: tuple[Any, Any, Any], unused: Any
        ) -> tuple[tuple[Any, Any, Any], dict[str, Any]]:
            del unused
            t, y, k = carry
            snap = snapshot(t, y)

            def step(i: Any, ty: tuple[Any, Any]) -> tuple[Any, Any]:
                del i
                tt, yy = ty
                ctx = observe_arr(tt, yy)
                dy_val = ctx.dy
                if isinstance(dy_val, State):
                    dy_a = dy_val.compartments.data
                elif isinstance(dy_val, PropertyData):
                    dy_a = dy_val.data
                else:
                    dy_a = dy_val
                return tt + dt, yy + dt * dy_a

            do_step = k < (n_saves - 1)

            def do_stride(ty: tuple[Any, Any]) -> tuple[Any, Any]:
                result: tuple[Any, Any] = jax.lax.fori_loop(0, stride, step, ty)
                return result

            t2, y2 = jax.lax.cond(do_step, do_stride, lambda ty: ty, (t, y))
            return (t2, y2, k + 1), snap

        (_tf, _yf, _), snaps = jax.lax.scan(
            outer_body, (jnp.asarray(t0), y_arr, jnp.asarray(0)), xs=None, length=n_saves
        )
        return {k: snaps[k] for k in init_snap}

    # General path: stack all step states, interpolate onto ts, vmap observe.
    def body(carry: tuple[Any, Any], unused: Any) -> tuple[tuple[Any, Any], Any]:
        del unused
        t, y = carry
        ctx = observe_arr(t, y)
        dy_val = ctx.dy
        if isinstance(dy_val, State):
            dy_a = dy_val.compartments.data
        elif isinstance(dy_val, PropertyData):
            dy_a = dy_val.data
        else:
            dy_a = dy_val
        return (t + dt, y + dt * dy_a), y

    (_t_final, y_final), ys = jax.lax.scan(body, (jnp.asarray(t0), y_arr), xs=None, length=steps)
    ys_all = jnp.concatenate([ys, y_final[None, ...]], axis=0)
    step_ts = t0 + dt * np.arange(steps + 1, dtype=np.float64)

    from summer4.time import TimeAxis

    axis = TimeAxis(values=step_ts, kind="grid")
    idx, w = axis.weights_for(ts)
    idx_j = jnp.asarray(idx)
    w_j = jnp.asarray(w)
    y_left = ys_all[idx_j[:, 0]]
    y_right = ys_all[idx_j[:, 1]]
    y_at = w_j[:, 0:1] * y_left + w_j[:, 1:2] * y_right

    snaps = jax.vmap(snapshot)(jnp.asarray(ts), y_at)
    return {k: snaps[k] for k in init_snap}


class FlowModel:
    """Named flows over one :class:`PropertyMap`, compiled to a vector field."""

    def __init__(self, pmap: PropertyMap) -> None:
        self.pmap = pmap
        self.flows: list[FlowLike] = []

    def add_flow(self, flow: FlowLike) -> FlowRef:
        """Register a named flow and return a :class:`FlowRef` to it.

        Names must be unique. Use the returned ref in later rate expressions
        (``death.sum()``, ``death.sum_over(location)``).
        """
        if any(existing.name == flow.name for existing in self.flows):
            raise ValueError(f"Duplicate flow name {flow.name!r}.")
        self.flows.append(flow)
        return FlowRef(flow.name)

    def compile(
        self,
        *,
        derived_fn: DerivedFn | None = None,
        strict_pairing: bool = True,
    ) -> CompiledModel:
        """Actualize joins once and return a :class:`CompiledModel`."""
        if not self.flows:
            raise ValueError("FlowModel has no flows.")
        actualized = topo_sort(
            [actualize(flow, self.pmap, strict_pairing=strict_pairing) for flow in self.flows]
        )
        flows = {flow.name: flow for flow in actualized}
        order = tuple(flow.name for flow in actualized)
        edge_maps = {flow.name: flow.edge_map for flow in actualized}
        paths: set[tuple[str, ...]] = set()
        for flow in actualized:
            paths |= _flow_paths(flow)
        computed_paths = tuple(sorted(paths))
        return CompiledModel(
            pmap=self.pmap,
            order=order,
            flows=flows,
            edge_maps=edge_maps,
            derived_fn=derived_fn,
            computed_paths=computed_paths,
        )


def _euler_jax(
    vf: Callable[..., Any],
    t0: object,
    y0: object,
    params: object,
    dt: float,
    steps: int,
) -> Any:
    import jax
    import jax.numpy as jnp

    from summer4.jax.propertydata import PropertyData
    from summer4.jax.state import State, unpack_state

    y_arr, rebox = unpack_state(y0)
    t0_j: Any = jnp.asarray(t0)
    dt_j: Any = jnp.asarray(dt)

    def body(carry: tuple[Any, Any], unused: Any) -> tuple[tuple[Any, Any], None]:
        del unused
        t, y = carry
        y_in = rebox(y)
        dy = vf(t, y_in, params)
        if isinstance(dy, State):
            dy_arr = dy.compartments.data
        elif isinstance(dy, PropertyData):
            dy_arr = dy.data
        else:
            dy_arr = dy
        return (t + dt_j, y + dt_j * dy_arr), None

    (_t_final, y_final), _ = jax.lax.scan(body, (t0_j, y_arr), xs=None, length=int(steps))
    return rebox(y_final)


def euler(
    vf: Callable[..., Any],
    t0: object,
    y0: object,
    params: object,
    *,
    dt: float,
    steps: int,
) -> Any:
    """Forward Euler via ``lax.scan``: ``y <- y + dt * vf(t, y, params)``.

    Returns the final state only. For trajectories use :meth:`CompiledModel.run`.
    For a NumPy reference stepper see :func:`numpy_euler`.
    """
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}.")
    return _euler_jax(vf, t0, y0, params, dt, steps)


def numpy_euler(
    vf: Callable[..., Any],
    t0: object,
    y0: object,
    params: object,
    *,
    dt: float,
    steps: int,
) -> Any:
    """NumPy reference Euler loop. Used in tests; not the compiled JAX path."""
    from summer4.jax.propertydata import PropertyData
    from summer4.jax.state import State, unpack_state

    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}.")
    y: Any = y0
    t: Any = t0
    dt_f = float(dt)
    for _ in range(int(steps)):
        dy = vf(t, y, params)
        y_arr, rebox = unpack_state(y)
        if isinstance(dy, State):
            dy_data = dy.compartments.data
        elif isinstance(dy, PropertyData):
            dy_data = dy.data
        else:
            dy_data = dy
        y = rebox(np.asarray(y_arr) + dt_f * np.asarray(dy_data))
        t = np.asarray(t) + dt_f
    return y
