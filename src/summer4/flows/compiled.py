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


@dataclass(frozen=True, slots=True, eq=False)
class CompiledModel:
    """Static compiled flows over one :class:`PropertyMap`.

    Entirely static: pass as ``jax.jit(..., static_argnums=0)`` rather than
    registering a pytree. The digest covers float arrays (``weight``, ``scale``,
    adjust masks) as well as topology, so two models that differ only in split
    proportions do not collide in the jit cache.

    ``derived_fn`` and :class:`Transform` callables hash by identity, so a lambda
    defined inside a loop retraces every call.
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

    def vector_field(self, t: object, y: object, params: object) -> Any:
        """Return ``dy/dt`` for state ``y`` (JAX arrays or :class:`PropertyData`)."""
        import jax.numpy as jnp

        from summer4.jax.propertydata import PropertyData

        y_pd: PropertyData | None = y if isinstance(y, PropertyData) else None
        y_arr: Any = y_pd.data if y_pd is not None else jnp.asarray(y)
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
        if y_pd is not None:
            return y_pd._with_data(dy)
        return dy

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

    y_pd = y0 if isinstance(y0, PropertyData) else None
    y_arr: Any = y_pd.data if y_pd is not None else jnp.asarray(y0)
    t0_j: Any = jnp.asarray(t0)
    dt_j: Any = jnp.asarray(dt)

    def body(carry: tuple[Any, Any], unused: Any) -> tuple[tuple[Any, Any], None]:
        del unused
        t, y = carry
        y_in = y_pd._with_data(y) if y_pd is not None else y
        dy = vf(t, y_in, params)
        dy_arr = dy.data if isinstance(dy, PropertyData) else dy
        return (t + dt_j, y + dt_j * dy_arr), None

    (_t_final, y_final), _ = jax.lax.scan(body, (t0_j, y_arr), xs=None, length=int(steps))
    if y_pd is not None:
        return y_pd._with_data(y_final)
    return y_final


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

    Returns the final state only. For a NumPy reference stepper see
    :func:`numpy_euler`.
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

    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}.")
    y: Any = y0
    t: Any = t0
    dt_f = float(dt)
    for _ in range(int(steps)):
        dy = vf(t, y, params)
        if isinstance(y, PropertyData):
            dy_data = dy.data if isinstance(dy, PropertyData) else dy
            y = y._with_data(np.asarray(y.data) + dt_f * np.asarray(dy_data))
        else:
            y = np.asarray(y) + dt_f * np.asarray(dy)
        t = np.asarray(t) + dt_f
    return y
