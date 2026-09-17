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
    GaussianPulse,
    Interp,
    Overwrite,
    RateOps,
    Reduce,
    Time,
    Transform,
    _adjust_bytes,
    _adjust_field_paths,
    _field_paths,
    _flow_rate_refs,
    _lookup_path,
    _rate_bytes,
    derived_return_schema,
    validate_computed_path,
)
from summer4.flows.stages import HoistTable, Prepared, PrepareFn
from summer4.flows.types import FlowLike
from summer4.properties import Property
from summer4.propertymap import PropertyMap

_NA: int = -1


class DerivedFn(Protocol):
    """``derived_fn(params, *, y, t)`` producing the derived-param struct."""

    def __call__(self, params: object, *, y: object, t: object) -> object: ...


@dataclass(frozen=True, slots=True)
class GroupedRate:
    """Rate whose last axis is aligned to one or more properties' traits.

    Arithmetic preserves ``properties``. Two ``GroupedRate`` values combine only
    when their groupings match; combining with a scalar keeps this grouping.
    Mismatched groupings raise rather than broadcasting silently.
    """

    data: object
    properties: tuple[Property, ...]
    __array_priority__ = 1000
    __array_ufunc__ = None  # force ``__rmatmul__`` / arithmetic over NumPy coercion

    def _require_same_grouping(self, other: GroupedRate) -> None:
        if self.properties != other.properties:
            left = ", ".join(p.name for p in self.properties) or "(none)"
            right = ", ".join(p.name for p in other.properties) or "(none)"
            raise ValueError(f"GroupedRate groupings differ: ({left}) vs ({right}).")

    def _combine(self, other: object, op: Any) -> GroupedRate:
        if isinstance(other, GroupedRate):
            self._require_same_grouping(other)
            return GroupedRate(op(self.data, other.data), self.properties)
        return GroupedRate(op(self.data, other), self.properties)

    def __add__(self, other: object) -> GroupedRate:
        return self._combine(other, lambda a, b: a + b)

    def __radd__(self, other: object) -> GroupedRate:
        return self._combine(other, lambda a, b: b + a)

    def __sub__(self, other: object) -> GroupedRate:
        return self._combine(other, lambda a, b: a - b)

    def __rsub__(self, other: object) -> GroupedRate:
        return self._combine(other, lambda a, b: b - a)

    def __mul__(self, other: object) -> GroupedRate:
        return self._combine(other, lambda a, b: a * b)

    def __rmul__(self, other: object) -> GroupedRate:
        return self._combine(other, lambda a, b: b * a)

    def __truediv__(self, other: object) -> GroupedRate:
        return self._combine(other, lambda a, b: a / b)

    def __rtruediv__(self, other: object) -> GroupedRate:
        return self._combine(other, lambda a, b: b / a)

    def __matmul__(self, other: object) -> GroupedRate:
        """``grouped @ M`` — right-multiply the last axis by a square matrix."""
        import jax.numpy as jnp

        data = jnp.asarray(self.data)
        mat = jnp.asarray(other)
        return GroupedRate(data @ mat, self.properties)

    def __rmatmul__(self, other: object) -> GroupedRate:
        """``M @ grouped`` — left-multiply the last axis by a square matrix."""
        import jax.numpy as jnp

        data = jnp.asarray(self.data)
        mat = jnp.asarray(other)
        return GroupedRate(mat @ data, self.properties)


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


def _norm_sigmoid(x: Any, sharpness: float) -> Any:
    """Normalized logistic on ``[0, 1]`` with summer2 curvature semantics.

    ``sharpness=1`` is linear-equivalent after normalization; larger values
    shrink the transition width toward a step at the segment midpoint. Ends
    map to 0 and 1 exactly.
    """
    import jax.numpy as jnp

    def uncorrected(u: Any) -> Any:
        return 1.0 / (1.0 + jnp.exp(sharpness * (0.5 - u)))

    offset = uncorrected(0.0)
    scale = 1.0 / (1.0 - (offset * 2.0))
    return (uncorrected(x) - offset) * scale


def _eval_interp(
    kind: str,
    breakpoints: Any,
    values: Any,
    x: Any,
    sharpness: float,
) -> Any:
    import jax.numpy as jnp

    xs = jnp.asarray(breakpoints)
    vals = jnp.asarray(values)
    x_arr = jnp.asarray(x)
    if kind == "linear":
        return jnp.interp(x_arr, xs, vals)
    if kind == "step":
        idx = jnp.searchsorted(xs, x_arr, side="right")
        return vals[idx]
    # sigmoidal — clamp outside the knot range, blend inside.
    lo = xs[0]
    hi = xs[-1]
    # Find left knot index in [0, n-2] for interior points.
    raw = jnp.searchsorted(xs, x_arr, side="right") - 1
    idx = jnp.clip(raw, 0, xs.shape[0] - 2)
    x0 = xs[idx]
    x1 = xs[idx + 1]
    y0 = vals[idx]
    y1 = vals[idx + 1]
    rel = (x_arr - x0) / (x1 - x0)
    blend = _norm_sigmoid(rel, sharpness)
    interior = y0 + blend * (y1 - y0)
    return jnp.where(x_arr <= lo, vals[0], jnp.where(x_arr >= hi, vals[-1], interior))


def _eval_rate(
    expr: RateOps,
    *,
    derived: Any,
    flow_values: Mapping[str, Any],
    flow_meta: Mapping[str, FlowEdges],
    pmap: PropertyMap,
    t: object,
    y_arr: Any,
    captures: dict[str, GroupedRate],
    hoisted: Prepared | None = None,
    table: HoistTable | None = None,
) -> Any:
    import jax.numpy as jnp

    from summer4.flows.rates import _RATE_EVALUATORS, ArrayConst, Capture
    from summer4.jax.propertydata import PropertyData

    if table is not None and hoisted is not None:
        value_slot = table.slot.get((id(expr), "value"))
        if value_slot is not None:
            return hoisted.hoisted[value_slot]

    def child(node: RateOps) -> Any:
        return _eval_rate(
            node,
            derived=derived,
            flow_values=flow_values,
            flow_meta=flow_meta,
            pmap=pmap,
            t=t,
            y_arr=y_arr,
            captures=captures,
            hoisted=hoisted,
            table=table,
        )

    match expr:
        case Const(value=value):
            return value
        case Time():
            return t
        case FieldRef(path=path):
            return _lookup_path(derived, path)
        case ArrayConst(value=value):
            return jnp.asarray(value)
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
                return GroupedRate(data=data, properties=(prop,))
            return values
        case Reduce(sum_over=sum_over, where=where):
            prop = pmap.get_property(sum_over.name if isinstance(sum_over, Property) else sum_over)
            pd = PropertyData(pmap, y_arr)
            if where is not None:
                pd = pd.keep(where, 0.0)
            reduced = pd.sum_over(prop)
            return GroupedRate(data=reduced.data, properties=(prop,))
        case Capture(name=name, inner=inner):
            value = child(inner)
            if isinstance(value, GroupedRate):
                captures[name] = value
            else:
                raise TypeError(
                    f"Capture({name!r}) inner must evaluate to GroupedRate, "
                    f"got {type(value).__name__}."
                )
            return value
        case BinOp(op=op, left=left, right=right):
            left_v = child(left)
            right_v = child(right)
            if op == "add":
                return left_v + right_v
            if op == "sub":
                return left_v - right_v
            if op == "mul":
                return left_v * right_v
            return left_v / right_v
        case Interp(
            kind=kind,
            breakpoints=breakpoints,
            values=values,
            arg=arg,
            sharpness=sharpness,
        ):
            bp_slot = table.slot.get((id(expr), "breakpoints")) if table is not None else None
            val_slot = table.slot.get((id(expr), "values")) if table is not None else None
            if bp_slot is not None and hoisted is not None:
                stacked_bps = hoisted.hoisted[bp_slot]
            else:
                stacked_bps = jnp.stack([jnp.asarray(child(bp)) for bp in breakpoints])
            if val_slot is not None and hoisted is not None:
                stacked = hoisted.hoisted[val_slot]
            else:
                stacked = jnp.stack([jnp.asarray(child(value)) for value in values])
            return _eval_interp(kind, stacked_bps, stacked, child(arg), sharpness)
        case GaussianPulse(arg=arg, centre=centre, width=width, height=height):
            x = child(arg)
            c = child(centre)
            w = child(width)
            h = child(height)
            return h * jnp.exp(-0.5 * ((x - c) / w) ** 2)
        case _:
            evaluator = _RATE_EVALUATORS.get(type(expr))
            if evaluator is not None:
                return evaluator(
                    expr,
                    eval_child=child,
                    derived=derived,
                    pmap=pmap,
                    t=t,
                    y_arr=y_arr,
                    captures=captures,
                )
            raise TypeError(f"Unsupported rate expression {type(expr).__name__}.")


def _as_array(value: Any) -> Any:
    import jax.numpy as jnp

    from summer4.jax.propertydata import PropertyData

    if isinstance(value, GroupedRate):
        return value.data
    if isinstance(value, PropertyData):
        return value.data
    return jnp.asarray(value)


def _group_codes_to_index(
    pmap: PropertyMap,
    gather_idx: NDArray[np.int32],
    properties: tuple[Property, ...],
) -> NDArray[np.int32]:
    """Map each gather row onto the last-axis index of a :class:`GroupedRate`."""
    if not properties:
        raise ValueError("GroupedRate alignment requires at least one property.")
    if len(properties) == 1:
        prop = properties[0]
        col_i = pmap.column_index(prop)
        codes = np.asarray(pmap.codes[gather_idx, col_i], dtype=np.int32)
        if np.any(codes == _NA):
            raise ValueError(
                f"Cannot align grouped rate over {prop.name!r}: "
                "some gather rows lack that property."
            )
        return codes

    groups = pmap.group_by(*properties)
    key_to_idx: dict[tuple[int, ...], int] = {}
    for i, traits in enumerate(groups):
        key_to_idx[tuple(t.code for t in traits)] = i
    cols = np.column_stack(
        [
            np.asarray(pmap.codes[gather_idx, pmap.column_index(p)], dtype=np.int32)
            for p in properties
        ]
    )
    if np.any(cols == _NA):
        names = ", ".join(p.name for p in properties)
        raise ValueError(
            f"Cannot align grouped rate over ({names}): some gather rows lack a property."
        )
    indices = np.empty(len(gather_idx), dtype=np.int32)
    for i, row in enumerate(cols):
        key = tuple(int(c) for c in row)
        try:
            indices[i] = key_to_idx[key]
        except KeyError as exc:
            names = ", ".join(p.name for p in properties)
            raise ValueError(
                f"Cannot align grouped rate over ({names}): "
                f"combination {key} is not present on the map."
            ) from exc
    return indices


def _align_grouped_rate(
    rate: GroupedRate,
    gather_idx: NDArray[np.int32],
    pmap: PropertyMap,
) -> Any:
    indices = _group_codes_to_index(pmap, gather_idx, rate.properties)
    return _as_array(rate)[..., indices]


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
    if isinstance(rate, GroupedRate):
        return _align_grouped_rate(rate, gather_idx, pmap)
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
    t: object,
    y_arr: Any,
    captures: dict[str, GroupedRate],
    hoisted: Prepared | None = None,
    table: HoistTable | None = None,
) -> Any:
    raw = _eval_rate(
        expr,
        derived=derived,
        flow_values=flow_values,
        flow_meta=flow_meta,
        pmap=pmap,
        t=t,
        y_arr=y_arr,
        captures=captures,
        hoisted=hoisted,
        table=table,
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
    t: object,
    y_arr: Any,
    captures: dict[str, GroupedRate],
    hoisted: Prepared | None = None,
    table: HoistTable | None = None,
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
                    t=t,
                    y_arr=y_arr,
                    captures=captures,
                    hoisted=hoisted,
                    table=table,
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
                t=t,
                y_arr=y_arr,
                captures=captures,
                hoisted=hoisted,
                table=table,
            )
            new = value if isinstance(adj, Overwrite) else prev * value
        prev = jnp.where(jnp.asarray(mask), new, prev) if mask is not None else new
    return prev


def _scatter_add(target: Any, indices: NDArray[np.int32], values: Any) -> Any:
    return target.at[..., indices].add(values)


def _collect_capture_meta(expr: RateOps) -> dict[str, tuple[Property, ...]]:
    """Walk a rate tree for nodes that expose ``__capture_meta__``."""
    from summer4.flows.rates import BinOp, GaussianPulse, Interp

    out: dict[str, tuple[Property, ...]] = {}
    meta_fn = getattr(expr, "__capture_meta__", None)
    if callable(meta_fn):
        name, props = meta_fn()
        out[name] = props
    match expr:
        case BinOp(left=left, right=right):
            out.update(_collect_capture_meta(left))
            out.update(_collect_capture_meta(right))
        case Interp(breakpoints=breakpoints, values=values, arg=arg):
            for bp in breakpoints:
                out.update(_collect_capture_meta(bp))
            for value in values:
                out.update(_collect_capture_meta(value))
            out.update(_collect_capture_meta(arg))
        case GaussianPulse(arg=arg, centre=centre, width=width, height=height):
            out.update(_collect_capture_meta(arg))
            out.update(_collect_capture_meta(centre))
            out.update(_collect_capture_meta(width))
            out.update(_collect_capture_meta(height))
        case _:
            custom = getattr(expr, "__capture_children__", None)
            if callable(custom):
                for child in custom():
                    out.update(_collect_capture_meta(child))
    return out


def _flow_paths(flow: FlowEdges) -> set[tuple[str, ...]]:
    paths = _field_paths(flow.rate)
    for adj in flow.adjust:
        paths |= _adjust_field_paths(adj)
    return paths


def _rate_flow_deps(flows: Mapping[str, FlowEdges]) -> frozenset[str]:
    """Flow names referenced by any rate / adjustment via ``FlowRef``."""
    names: set[str] = set()
    for flow in flows.values():
        names |= _flow_rate_refs(flow.rate, flow.adjust)
    return frozenset(names)


def _validate_plan_computed(plan: Any, derived_fn: DerivedFn | None) -> None:
    """Validate ``ComputedValue`` paths against ``derived_fn``'s return schema."""
    from summer4.results.plan import ComputedValue, SavePlan

    if not isinstance(plan, SavePlan):
        return
    schema = derived_return_schema(derived_fn)
    if schema is None:
        return
    for req in plan.requests.values():
        if isinstance(req.what, ComputedValue):
            validate_computed_path(req.what.path, schema)


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
    prepare_fn: PrepareFn | None = None,
    hoist: bool = True,
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
    hasher.update(b"pf")
    hasher.update(str(id(prepare_fn) if prepare_fn is not None else 0).encode())
    hasher.update(b"hoist1" if hoist else b"hoist0")
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
    captures: Mapping[str, GroupedRate] = field(default_factory=dict)


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
    prepare_fn: PrepareFn | None = None
    hoist_table: HoistTable | None = None
    hoist: bool = True
    capture_meta: Mapping[str, tuple[Property, ...]] = field(default_factory=dict)
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
                prepare_fn=self.prepare_fn,
                hoist=self.hoist,
            ),
        )

    def prepare(self, params: object) -> Prepared:
        """Run-start stage: apply ``prepare_fn`` and evaluate hoisted rate slots."""
        if isinstance(params, Prepared):
            return params
        p = params if self.prepare_fn is None else self.prepare_fn(params)
        table = self.hoist_table
        if table is None or not table.entries:
            return Prepared(p, ())
        import jax.numpy as jnp

        hoisted: list[Any] = []
        for entry in table.entries:
            if entry.part == "value":
                hoisted.append(
                    _eval_rate(
                        entry.node,
                        derived=p,
                        flow_values={},
                        flow_meta={},
                        pmap=self.pmap,
                        t=None,
                        y_arr=None,
                        captures={},
                    )
                )
            elif entry.part == "breakpoints":
                assert isinstance(entry.node, Interp)
                stacked = jnp.stack(
                    [
                        jnp.asarray(
                            _eval_rate(
                                bp,
                                derived=p,
                                flow_values={},
                                flow_meta={},
                                pmap=self.pmap,
                                t=None,
                                y_arr=None,
                                captures={},
                            )
                        )
                        for bp in entry.node.breakpoints
                    ]
                )
                hoisted.append(stacked)
            else:
                assert isinstance(entry.node, Interp)
                stacked = jnp.stack(
                    [
                        jnp.asarray(
                            _eval_rate(
                                val,
                                derived=p,
                                flow_values={},
                                flow_meta={},
                                pmap=self.pmap,
                                t=None,
                                y_arr=None,
                                captures={},
                            )
                        )
                        for val in entry.node.values
                    ]
                )
                hoisted.append(stacked)
        return Prepared(p, tuple(hoisted))

    def edges(self, name: str) -> EdgeMap:
        """Return the :class:`EdgeMap` for the named flow."""
        try:
            return self.edge_maps[name]
        except KeyError:
            raise KeyError(f"Unknown flow {name!r}. Known: {list(self.edge_maps)}") from None

    def observe(
        self,
        t: object,
        y: object,
        params: object,
        *,
        keep: frozenset[str] | None = None,
    ) -> SaveContext:
        """Evaluate the field once and return ``dy`` plus per-flow masses.

        ``keep`` restricts which flow masses appear in
        :attr:`SaveContext.flows` after the step. Rate expressions still see
        every intermediate mass during the loop; pruning happens at the end.
        Pass ``None`` to keep every flow (the default).
        """
        import jax.numpy as jnp

        from summer4.jax.propertydata import PropertyData
        from summer4.jax.state import State, unpack_state

        if not isinstance(params, Prepared):
            params = self.prepare(params)
        prepared = params
        y_arr, rebox = unpack_state(y, self.pmap)
        derived = _eval_derived(self.derived_fn, prepared.params, y_arr, t)
        dy: Any = jnp.zeros_like(y_arr)
        flow_values: dict[str, Any] = {}
        captures: dict[str, GroupedRate] = {}
        pmap = self.pmap
        table = self.hoist_table if self.hoist else None
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
                t=t,
                y_arr=y_arr,
                captures=captures,
                hoisted=prepared,
                table=table,
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
                t=t,
                y_arr=y_arr,
                captures=captures,
                hoisted=prepared,
                table=table,
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
        if keep is not None:
            retain = keep | _rate_flow_deps(self.flows)
            flow_values = {k: v for k, v in flow_values.items() if k in retain}
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
        return SaveContext(
            t=t, y=y_out, dy=dy_out, derived=derived, flows=flow_values, captures=captures
        )

    def vector_field(self, t: object, y: object, params: object) -> Any:
        """Return ``dy/dt`` for state ``y`` (JAX arrays, PropertyData, or State)."""
        return self.observe(t, y, params).dy

    def expand(self, plan: Any) -> Any:
        """Fill an empty (EVERYTHING) plan with compartments, flows, and computed paths."""
        from summer4.results.plan import (
            Compartments,
            ComputedValue,
            FlowMass,
            GroupedOutput,
            SavePlan,
            SaveRequest,
        )

        if not isinstance(plan, SavePlan):
            raise TypeError(f"Expected SavePlan, got {type(plan).__name__}.")
        if plan.requests:
            _validate_plan_computed(plan, self.derived_fn)
            return plan
        requests: dict[str, SaveRequest] = {
            "compartments": SaveRequest(Compartments()),
        }
        for name in self.order:
            requests[name] = SaveRequest(FlowMass(flow=name))
        for path in self.computed_paths:
            key = ".".join(path)
            requests[key] = SaveRequest(ComputedValue(path=path))
        for cap_name in self.capture_meta:
            requests[cap_name] = SaveRequest(GroupedOutput(name=cap_name))
        expanded = SavePlan(
            requests=requests,
            ts=plan.ts,
            dense=plan.dense,
            solver_stats=plan.solver_stats,
        )
        _validate_plan_computed(expanded, self.derived_fn)
        return expanded

    def describe(
        self,
        plan: Any,
        *,
        params: object | None = None,
        t0: float = 0.0,
        y0: object | None = None,
        n_saves: int | None = None,
        dt: float = 1.0,
        steps: int | None = None,
    ) -> Any:
        """Return output shapes and memory footprint without solving.

        Each output is sized from its own save group's ``ts`` (per-request times
        override the plan default).

        Pass ``params`` (or a pytree of ``jax.ShapeDtypeStruct`` stand-ins) when
        the model has a ``derived_fn`` that indexes them. ``None`` remains valid
        for models that ignore params.
        """
        import jax

        from summer4.jax.propertydata import PropertyData
        from summer4.results.eval import build_save_fn
        from summer4.results.groups import group_requests
        from summer4.results.plan import OutputShape, PlanDescription, SavePlan

        expanded = self.expand(plan)
        if not isinstance(expanded, SavePlan):
            raise TypeError("expand must return SavePlan")
        if y0 is None:
            y0 = PropertyData.wrap(self.pmap, np.zeros(self.pmap.size))
        save_fn = build_save_fn(expanded, pmap=self.pmap, edge_maps=dict(self.edge_maps))
        keep = expanded.flow_reads()

        def _one(t: object, y: object, params: object) -> Any:
            ctx = self.observe(t, y, params, keep=keep)
            return save_fn(ctx)

        try:
            shaped = jax.eval_shape(_one, t0, y0, params)
        except Exception as exc:
            raise ValueError(
                f"SavePlan failed shape inference (SaveFn must return statically "
                f"shaped arrays): {exc}"
            ) from exc

        if n_saves is not None:
            default_ts = t0 + float(dt) * np.arange(int(n_saves), dtype=np.float64)
        elif expanded.ts is not None:
            default_ts = np.asarray(expanded.ts, dtype=np.float64)
        elif steps is not None:
            default_ts = t0 + float(dt) * np.arange(int(steps) + 1, dtype=np.float64)
        else:
            default_ts = np.asarray([t0], dtype=np.float64)

        groups = group_requests(expanded, default_ts)
        key_to_n = {key: int(g.ts.size) for g in groups for key in g.keys}

        outputs: list[OutputShape] = []
        total = 0
        for key in expanded.requests:
            leaf = shaped[key]
            if hasattr(leaf, "data"):
                leaf = leaf.data
            shape = tuple(int(s) for s in getattr(leaf, "shape", ()))
            n = key_to_n[key]
            full_shape = (n, *shape)
            dtype = str(getattr(leaf, "dtype", "float64"))
            itemsize = np.dtype(dtype).itemsize if dtype else 8
            nbytes = int(np.prod(full_shape)) * int(itemsize)
            outputs.append(OutputShape(key=key, shape=full_shape, dtype=dtype, nbytes=nbytes))
            total += nbytes
        return PlanDescription(
            outputs=tuple(outputs),
            n_saves=int(default_ts.size),
            total_nbytes=total,
        )

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
        solver: str | Any = "euler",
        epoch: Any = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_steps: int | None = None,
    ) -> Any:
        """Integrate and return a :class:`~summer4.results.Result`.

        Exactly one of ``t1`` / ``steps``. ``solver`` is a name (``"euler"``,
        ``"heun"``, ``"tsit5"``, ``"dopri5"``) or a diffrax solver instance.
        """
        from summer4.results.eval import dims_for_quantity, values_for
        from summer4.results.groups import group_requests
        from summer4.results.plan import EVERYTHING, SavePlan
        from summer4.results.result import Result
        from summer4.results.trace import Trace
        from summer4.solvers.base import SolveSpec
        from summer4.solvers.diffrax_backend import KNOWN_SOLVER_NAMES, diffrax_solve
        from summer4.solvers.euler_backend import euler_solve
        from summer4.time import Epoch, TimeAxis

        if save is None:
            save = EVERYTHING
        if not isinstance(save, SavePlan):
            raise TypeError(f"save must be a SavePlan, got {type(save).__name__}.")
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
            default_ts = t0 + dt * np.arange(n_steps + 1, dtype=np.float64)
        else:
            default_ts = np.asarray(expanded.ts, dtype=np.float64)

        groups = group_requests(expanded, default_ts)
        key_to_group = {key: g for g in groups for key in g.keys}

        times = TimeAxis(
            values=default_ts,
            epoch=epoch if isinstance(epoch, Epoch) else epoch,
            kind="grid",
        )

        spec = SolveSpec(
            t0=float(t0),
            t1=float(t1) if t1 is not None else None,
            steps=n_steps,
            dt=float(dt),
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            dense=bool(expanded.dense),
        )

        use_euler = solver == "euler" or (isinstance(solver, str) and solver.lower() == "euler")
        if use_euler:
            if rtol is not None or atol is not None:
                raise ValueError("rtol/atol apply to adaptive diffrax solvers, not euler.")
            out = euler_solve(
                self,
                y0=y0,
                params=params,
                spec=spec,
                groups=groups,
                plan=expanded,
                solver_stats=bool(expanded.solver_stats),
            )
        else:
            if isinstance(solver, str) and solver.lower() not in KNOWN_SOLVER_NAMES:
                known = ", ".join(repr(n) for n in KNOWN_SOLVER_NAMES)
                raise ValueError(f"Unknown solver {solver!r}. Known names: {known}.")
            out = diffrax_solve(
                self,
                y0=y0,
                params=params,
                spec=spec,
                groups=groups,
                plan=expanded,
                solver=solver,
                solver_stats=bool(expanded.solver_stats),
            )

        traces: dict[str, Trace] = {}
        for key, req in expanded.requests.items():
            from summer4.results.plan import GroupedOutput

            dims: tuple[str, ...]
            if isinstance(req.what, GroupedOutput) and req.what.name in self.capture_meta:
                prop = self.capture_meta[req.what.name][0]
                dims = ("time", prop.name)
            else:
                dims = dims_for_quantity(req.what)
            raw = out.saved[key]
            group = key_to_group[key]
            trace_times = TimeAxis(
                values=group.ts,
                epoch=epoch if isinstance(epoch, Epoch) else epoch,
                kind="grid",
            )
            values = values_for(req, raw, self)
            traces[key] = Trace(times=trace_times, values=values, dims=dims)

        return Result(
            times=times,
            traces=traces,
            solver=out.stats,
            dense=out.dense,
            _state_pmap=self.pmap if out.dense is not None else None,
        )

    def __hash__(self) -> int:
        return hash(self._digest)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompiledModel):
            return NotImplemented
        return self._digest == other._digest


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
        prepare_fn: PrepareFn | None = None,
        hoist: bool = True,
        strict_pairing: bool = True,
    ) -> CompiledModel:
        """Actualize joins once and return a :class:`CompiledModel`."""
        from summer4.flows.stages import HoistTable, build_hoist_table, roots_of

        if not self.flows:
            raise ValueError("FlowModel has no flows.")
        actualized = topo_sort(
            [actualize(flow, self.pmap, strict_pairing=strict_pairing) for flow in self.flows]
        )
        flows = {flow.name: flow for flow in actualized}
        order = tuple(flow.name for flow in actualized)
        edge_maps = {flow.name: flow.edge_map for flow in actualized}
        paths: set[tuple[str, ...]] = set()
        capture_meta: dict[str, tuple[Property, ...]] = {}
        for flow in actualized:
            paths |= _flow_paths(flow)
            capture_meta.update(_collect_capture_meta(flow.rate))
            for adj in flow.adjust:
                if hasattr(adj, "value"):
                    capture_meta.update(_collect_capture_meta(adj.value))
                elif hasattr(adj, "args"):
                    for arg in adj.args:
                        capture_meta.update(_collect_capture_meta(arg))
        computed_paths = tuple(sorted(paths))
        if hoist:
            hoist_table = build_hoist_table(
                roots_of(flows, order),
                params_are_static=derived_fn is None,
            )
        else:
            hoist_table = HoistTable(entries=(), slot={})
        return CompiledModel(
            pmap=self.pmap,
            order=order,
            flows=flows,
            edge_maps=edge_maps,
            derived_fn=derived_fn,
            computed_paths=computed_paths,
            prepare_fn=prepare_fn,
            hoist_table=hoist_table,
            hoist=hoist,
            capture_meta=capture_meta,
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
