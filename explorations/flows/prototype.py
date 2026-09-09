"""Draft flows: query join, lazy rates, and a basic vector field.

This module is not part of summer4. It exists so the explore-flows branch
can prove the join / rate / vector-field story before a public API is frozen.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from summer4 import Everything, Property, PropertyMap, Selector, Trait
from summer4.selectors import Absent, And, IsIn, Not, Nothing, Or, Present, SelectorOps

_NA: int = -1
_SPLIT_TOL: float = 1e-9

type SplitSpec = Mapping[Property | str, Mapping[str, float]]
type NormalizedSplit = dict[str, dict[str, float]]
type BackendName = Literal["numpy", "jax"]
type DerivedFn = Callable[..., Any]
type FlowLike = TransitionFlow | ExitFlow | EntryFlow
type FlowReduce = Literal["identity", "sum"] | tuple[Literal["sum_over"], str]

_NamedTupleT = TypeVar("_NamedTupleT", bound=NamedTuple)


# ---------------------------------------------------------------------------
# Selector introspection
# ---------------------------------------------------------------------------


def selector_properties(sel: Selector) -> frozenset[str]:
    """Return property names mentioned in ``sel`` (bound names)."""
    match sel:
        case Trait() as trait:
            return frozenset({trait.property})
        case IsIn(property=name) | Present(property=name) | Absent(property=name):
            return frozenset({name})
        case Everything() | Nothing():
            return frozenset()
        case And(left=left, right=right) | Or(left=left, right=right):
            return selector_properties(left) | selector_properties(right)
        case Not(inner=inner):
            return selector_properties(inner)
        case _:
            raise TypeError(f"Unsupported selector {type(sel).__name__}.")


def _property_present(pmap: PropertyMap, indices: NDArray[np.int32], col: int) -> bool:
    if indices.size == 0:
        return False
    return bool(np.any(pmap.codes[indices, col] != _NA))


def _pack_keys(codes: NDArray[np.int16], cols: tuple[int, ...]) -> NDArray[np.int64]:
    """Pack selected code columns into one int64 key per row."""
    n = codes.shape[0]
    if not cols:
        return np.zeros(n, dtype=np.int64)
    packed = np.zeros(n, dtype=np.int64)
    for col in cols:
        packed = packed * 65536 + (codes[:, col].astype(np.int64) + 1)
    return packed


def _normalize_split(split: SplitSpec | None) -> NormalizedSplit | None:
    if split is None:
        return None
    out: NormalizedSplit = {}
    for key, proportions in split.items():
        name = key.name if isinstance(key, Property) else key
        out[name] = {str(trait): float(value) for trait, value in proportions.items()}
    return out


def _validate_split(
    pmap: PropertyMap,
    dest_idx: NDArray[np.int32],
    dest_only: tuple[Property, ...],
    split: NormalizedSplit | None,
) -> NormalizedSplit:
    if not split:
        return {}
    dest_only_by_name = {prop.name: prop for prop in dest_only}
    resolved: NormalizedSplit = {}
    for name, proportions in split.items():
        if name not in dest_only_by_name:
            raise ValueError(
                f"split key {name!r} is not a dest-only property. "
                f"Dest-only: {sorted(dest_only_by_name)}."
            )
        prop = dest_only_by_name[name]
        col = pmap.codes[dest_idx, pmap._prop_index[prop.name]]
        appearing = {prop.traits[int(code)] for code in col if code != _NA}
        named = set(proportions)
        if named != appearing:
            raise ValueError(
                f"split[{name!r}] must name every trait that appears on dest "
                f"{sorted(appearing)}; got {sorted(named)}."
            )
        if any(value < 0.0 for value in proportions.values()):
            raise ValueError(f"split[{name!r}] proportions must be >= 0.")
        total = sum(proportions.values())
        if abs(total - 1.0) > _SPLIT_TOL:
            raise ValueError(f"split[{name!r}] proportions must sum to 1, got {total}.")
        resolved[name] = dict(proportions)
    return resolved


def _group_weights(
    pmap: PropertyMap,
    dest_members: NDArray[np.int32],
    dest_only: tuple[Property, ...],
    split: NormalizedSplit,
) -> NDArray[np.float64]:
    """Return dest weights that sum to 1 for one free-key group."""
    n = int(dest_members.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)
    for prop in dest_only:
        proportions = split.get(prop.name)
        if proportions is None:
            continue
        col = pmap.codes[dest_members, pmap._prop_index[prop.name]]
        for i, code in enumerate(col):
            if code == _NA:
                continue
            weights[i] *= proportions[prop.traits[int(code)]]
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Dest-split weights summed to 0; check split proportions.")
    return weights / total


@dataclass(frozen=True, slots=True)
class EdgeArrays:
    """Concrete source/dest index pairs plus conservation weights."""

    src_idx: NDArray[np.int32]
    dest_idx: NDArray[np.int32]
    weight: NDArray[np.float64]
    scale: NDArray[np.float64]

    @property
    def n_edges(self) -> int:
        return int(self.src_idx.size)


def identity_join(
    pmap: PropertyMap,
    source: Selector,
    dest: Selector,
    *,
    extra_bound: frozenset[str] = frozenset(),
    split: SplitSpec | None = None,
    scale: float = 1.0,
) -> EdgeArrays:
    """Pair source and dest compartments that agree on every free property."""
    src_idx = pmap.select(source)
    dest_idx = pmap.select(dest)
    if src_idx.size == 0:
        raise ValueError("Source selector matched no compartments.")
    if dest_idx.size == 0:
        raise ValueError("Destination selector matched no compartments.")

    bound = selector_properties(source) | selector_properties(dest) | extra_bound
    free_cols: list[int] = []
    dest_only: list[Property] = []
    for col, prop in enumerate(pmap.properties):
        if prop.name in bound:
            continue
        src_has = _property_present(pmap, src_idx, col)
        dest_has = _property_present(pmap, dest_idx, col)
        if src_has and dest_has:
            free_cols.append(col)
        elif dest_has and not src_has:
            dest_only.append(prop)

    dest_only_t = tuple(dest_only)
    split_n = _validate_split(pmap, dest_idx, dest_only_t, _normalize_split(split))

    src_keys = _pack_keys(pmap.codes[src_idx], tuple(free_cols))
    dest_keys = _pack_keys(pmap.codes[dest_idx], tuple(free_cols))
    dest_groups: dict[int, list[int]] = defaultdict(list)
    for key, gi in zip(dest_keys.tolist(), dest_idx.tolist(), strict=True):
        dest_groups[int(key)].append(int(gi))

    src_out: list[int] = []
    dest_out: list[int] = []
    weight_out: list[float] = []
    unique_src, inverse = np.unique(src_keys, return_inverse=True)
    labels = pmap.labels()
    for group_i, key in enumerate(unique_src.tolist()):
        members = dest_groups.get(int(key))
        if members is None:
            missing = src_idx[inverse == group_i]
            label = labels[int(missing[0])]
            raise ValueError(
                f"No destination match for source {label!r} on free properties "
                f"{[pmap.properties[c].name for c in free_cols]}."
            )
        dest_members = np.asarray(members, dtype=np.int32)
        weights = _group_weights(pmap, dest_members, dest_only_t, split_n)
        src_members = src_idx[inverse == group_i]
        for src in src_members.tolist():
            for dest_i, weight in zip(dest_members.tolist(), weights.tolist(), strict=True):
                src_out.append(int(src))
                dest_out.append(int(dest_i))
                weight_out.append(float(weight))

    n = len(src_out)
    return EdgeArrays(
        src_idx=np.asarray(src_out, dtype=np.int32),
        dest_idx=np.asarray(dest_out, dtype=np.int32),
        weight=np.asarray(weight_out, dtype=np.float64),
        scale=np.full(n, float(scale), dtype=np.float64),
    )


def _concat_edges(parts: Sequence[EdgeArrays]) -> EdgeArrays:
    if not parts:
        raise ValueError("Pairing produced no edges.")
    return EdgeArrays(
        src_idx=np.concatenate([part.src_idx for part in parts]),
        dest_idx=np.concatenate([part.dest_idx for part in parts]),
        weight=np.concatenate([part.weight for part in parts]),
        scale=np.concatenate([part.scale for part in parts]),
    )


@dataclass(frozen=True, slots=True)
class TraitChain:
    """Explicit source→dest trait pairs on one property (ageing).

    One named flow can carry the whole chain. Optional ``rates`` are per-pair
    multipliers (band width ``1/5``, ``1/10``, …) stored on the edge ``scale``,
    the same place ``TraitMatrix`` puts nonzero entries. The flow ``rate`` still
    applies on top. The last trait is omitted from ``pairs`` so it has no
    outgoing edge.
    """

    property: Property
    pairs: tuple[tuple[str, str], ...]
    rates: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("TraitChain requires at least one trait pair.")
        known = set(self.property.traits)
        for src, dest in self.pairs:
            if src not in known or dest not in known:
                raise KeyError(
                    f"Unknown trait in chain {(src, dest)} for "
                    f"{self.property.name!r}. Known: {list(self.property.traits)}"
                )
        if self.rates is not None and len(self.rates) != len(self.pairs):
            raise ValueError(
                f"TraitChain.rates length {len(self.rates)} does not match "
                f"{len(self.pairs)} pairs."
            )


@dataclass(frozen=True, slots=True)
class TraitMatrix:
    """Dense or sparse transition matrix over one property (dest × source)."""

    property: Property
    matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        n = len(self.property.traits)
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (n, n):
            raise ValueError(
                f"TraitMatrix for {self.property.name!r} must have shape "
                f"({n}, {n}) dest×source, got {matrix.shape}."
            )
        object.__setattr__(self, "matrix", matrix)


def _join_with_pairing(
    pmap: PropertyMap,
    source: Selector,
    dest: Selector,
    pairing: TraitChain | TraitMatrix | None,
    split: SplitSpec | None,
) -> EdgeArrays:
    if pairing is None:
        return identity_join(pmap, source, dest, split=split)
    extra = frozenset({pairing.property.name})
    if isinstance(pairing, TraitChain):
        parts = [
            identity_join(
                pmap,
                source & pairing.property[src],
                dest & pairing.property[dest_name],
                extra_bound=extra,
                split=split,
                scale=1.0 if pairing.rates is None else float(pairing.rates[index]),
            )
            for index, (src, dest_name) in enumerate(pairing.pairs)
        ]
        return _concat_edges(parts)
    parts = []
    traits = pairing.property.traits
    for dest_code, dest_name in enumerate(traits):
        for src_code, src_name in enumerate(traits):
            value = float(pairing.matrix[dest_code, src_code])
            if value == 0.0:
                continue
            parts.append(
                identity_join(
                    pmap,
                    source & pairing.property[src_name],
                    dest & pairing.property[dest_name],
                    extra_bound=extra,
                    split=split,
                    scale=value,
                )
            )
    return _concat_edges(parts)


# ---------------------------------------------------------------------------
# Rate expressions
# ---------------------------------------------------------------------------


class RateOps:
    """Arithmetic mixin for rate expression nodes."""

    def __add__(self, other: object) -> BinOp:
        return BinOp("add", as_rate(self), as_rate(other))

    def __radd__(self, other: object) -> BinOp:
        return BinOp("add", as_rate(other), as_rate(self))

    def __sub__(self, other: object) -> BinOp:
        return BinOp("sub", as_rate(self), as_rate(other))

    def __rsub__(self, other: object) -> BinOp:
        return BinOp("sub", as_rate(other), as_rate(self))

    def __mul__(self, other: object) -> BinOp:
        return BinOp("mul", as_rate(self), as_rate(other))

    def __rmul__(self, other: object) -> BinOp:
        return BinOp("mul", as_rate(other), as_rate(self))

    def __truediv__(self, other: object) -> BinOp:
        return BinOp("div", as_rate(self), as_rate(other))

    def __rtruediv__(self, other: object) -> BinOp:
        return BinOp("div", as_rate(other), as_rate(self))


@dataclass(frozen=True, slots=True)
class Const(RateOps):
    """Literal scalar rate."""

    value: float


@dataclass(frozen=True, slots=True)
class FieldRef(RateOps):
    """Lazy path into the runtime derived-param struct."""

    path: tuple[str, ...]

    def __getattr__(self, name: str) -> FieldRef:
        if name.startswith("_"):
            raise AttributeError(name)
        return FieldRef((*self.path, name))


@dataclass(frozen=True, slots=True)
class FlowRef(RateOps):
    """Reference to another flow's already-computed contribution."""

    name: str
    reduce: FlowReduce = "identity"

    def sum(self) -> FlowRef:
        """Return this flow's contribution reduced to a scalar."""
        return FlowRef(self.name, reduce="sum")

    def sum_over(self, prop: Property | str) -> FlowRef:
        """Reduce this flow's contribution onto one property's traits."""
        name = prop.name if isinstance(prop, Property) else prop
        return FlowRef(self.name, reduce=("sum_over", name))


@dataclass(frozen=True, slots=True)
class BinOp(RateOps):
    """Binary arithmetic on two rate expressions."""

    op: Literal["add", "sub", "mul", "div"]
    left: RateOps
    right: RateOps


@dataclass(frozen=True, slots=True)
class _SubmapRate:
    """Rate whose last axis is aligned to one or more property trait maps."""

    data: Any
    properties: tuple[Property, ...]


def _is_namedtuple_class(schema: type) -> bool:
    return (
        isinstance(schema, type)
        and issubclass(schema, tuple)
        and isinstance(getattr(schema, "_fields", None), tuple)
    )


def derived_refs(schema: type[_NamedTupleT]) -> _NamedTupleT:
    """Build a NamedTuple of :class:`FieldRef`s named after ``schema`` fields.

    The return is annotated as ``schema`` so IDEs complete only those fields.
    Runtime values are ``FieldRef`` paths, not the schema's declared types.
    """
    if not _is_namedtuple_class(schema):
        raise TypeError(f"derived_refs expects a NamedTuple class, got {schema!r}.")
    refs = (FieldRef((name,)) for name in schema._fields)
    return schema(*refs)  # type: ignore[arg-type]


def as_rate(value: object) -> RateOps:
    """Coerce a scalar or rate node into a :class:`RateOps` expression."""
    if isinstance(value, RateOps):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return Const(float(value))
    raise TypeError(f"Cannot use {type(value).__name__} as a flow rate.")


def _flow_refs(expr: RateOps) -> set[str]:
    match expr:
        case FlowRef(name=name):
            return {name}
        case BinOp(left=left, right=right):
            return _flow_refs(left) | _flow_refs(right)
        case _:
            return set()


def _lookup_path(root: Any, path: tuple[str, ...]) -> Any:
    current = root
    for name in path:
        if isinstance(current, Mapping):
            try:
                current = current[name]
            except KeyError as exc:
                raise KeyError(f"Derived struct has no field {path!r}.") from exc
        else:
            try:
                current = getattr(current, name)
            except AttributeError as exc:
                raise AttributeError(f"Derived struct has no field {path!r}.") from exc
    return current


def _sum_mass_over(
    mass: Any,
    edge_idx: NDArray[np.int32],
    pmap: PropertyMap,
    prop: Property,
    xp: Any,
) -> Any:
    """Segment-sum edge mass by ``prop`` trait codes (skip absent)."""
    col_i = pmap._prop_index[prop.name]
    codes = np.asarray(pmap.codes[edge_idx, col_i], dtype=np.int32)
    n_traits = len(prop.traits)
    valid = codes >= 0
    safe = np.where(valid, codes, 0)
    if xp is np:
        mass_arr = np.asarray(mass)
        leading = mass_arr.shape[:-1]
        out = np.zeros(leading + (n_traits,), dtype=mass_arr.dtype)
        weighted = mass_arr * valid
        if mass_arr.ndim == 1:
            np.add.at(out, safe, weighted)
            return out
        flat_m = weighted.reshape(-1, mass_arr.shape[-1])
        flat_o = out.reshape(-1, n_traits)
        for row_m, row_o in zip(flat_m, flat_o, strict=True):
            np.add.at(row_o, safe, row_m)
        return out
    import jax
    import jax.numpy as jnp

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
    flow_meta: Mapping[str, ActualizedFlow],
    pmap: PropertyMap,
    xp: Any,
) -> Any:
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
                return xp.sum(values)
            if isinstance(reduce, tuple) and reduce[0] == "sum_over":
                producer = flow_meta[name]
                edge_idx = producer.src_idx if producer.kind != "entry" else producer.dest_idx
                if edge_idx is None:
                    raise RuntimeError(f"Flow {name!r} is missing indices for sum_over.")
                prop = pmap.get_property(reduce[1])
                data = _sum_mass_over(values, edge_idx, pmap, prop, xp)
                return _SubmapRate(data=data, properties=(prop,))
            return values
        case BinOp(op=op, left=left, right=right):
            left_v = _eval_rate(
                left,
                derived=derived,
                flow_values=flow_values,
                flow_meta=flow_meta,
                pmap=pmap,
                xp=xp,
            )
            right_v = _eval_rate(
                right,
                derived=derived,
                flow_values=flow_values,
                flow_meta=flow_meta,
                pmap=pmap,
                xp=xp,
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


# ---------------------------------------------------------------------------
# Flow types
# ---------------------------------------------------------------------------


def _as_selector(sel: Selector) -> Selector:
    if not isinstance(sel, SelectorOps):
        raise TypeError(f"Expected a Selector, got {type(sel).__name__}.")
    return sel


@dataclass(frozen=True, slots=True)
class TransitionFlow:
    """Population moving from source compartments to destination compartments."""

    name: str
    source: Selector
    dest: Selector
    rate: RateOps
    pairing: TraitChain | TraitMatrix | None = None
    split: NormalizedSplit | None = None
    absolute: bool = False

    def __init__(
        self,
        name: str,
        source: Selector,
        dest: Selector,
        rate: object,
        *,
        pairing: TraitChain | TraitMatrix | None = None,
        split: SplitSpec | None = None,
        absolute: bool = False,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source", _as_selector(source))
        object.__setattr__(self, "dest", _as_selector(dest))
        object.__setattr__(self, "rate", as_rate(rate))
        object.__setattr__(self, "pairing", pairing)
        object.__setattr__(self, "split", _normalize_split(split))
        object.__setattr__(self, "absolute", absolute)


@dataclass(frozen=True, slots=True)
class ExitFlow:
    """Population leaving source compartments to outside the system."""

    name: str
    source: Selector
    rate: RateOps
    absolute: bool = False

    def __init__(
        self,
        name: str,
        source: Selector,
        rate: object,
        *,
        absolute: bool = False,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source", _as_selector(source))
        object.__setattr__(self, "rate", as_rate(rate))
        object.__setattr__(self, "absolute", absolute)


@dataclass(frozen=True, slots=True)
class EntryFlow:
    """Population entering destination compartments from outside the system."""

    name: str
    dest: Selector
    rate: RateOps
    split: NormalizedSplit | None = None

    def __init__(
        self,
        name: str,
        dest: Selector,
        rate: object,
        *,
        split: SplitSpec | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dest", _as_selector(dest))
        object.__setattr__(self, "rate", as_rate(rate))
        object.__setattr__(self, "split", _normalize_split(split))


@dataclass(frozen=True, slots=True)
class ActualizedFlow:
    """A flow resolved against one :class:`PropertyMap`."""

    name: str
    kind: Literal["transition", "exit", "entry"]
    src_idx: NDArray[np.int32] | None
    dest_idx: NDArray[np.int32] | None
    weight: NDArray[np.float64]
    scale: NDArray[np.float64]
    rate: RateOps
    absolute: bool

    @property
    def n_edges(self) -> int:
        return int(self.weight.size)


def _sum_over_property_name(expr: RateOps) -> str | None:
    """Return the property name if ``expr`` is a ``sum_over`` flow ref."""
    match expr:
        case FlowRef(reduce=("sum_over", name)):
            return name
        case _:
            return None


def _entry_edges(
    pmap: PropertyMap,
    dest: Selector,
    split: NormalizedSplit | None,
    group_by: str | None = None,
) -> EdgeArrays:
    dest_idx = pmap.select(dest)
    if dest_idx.size == 0:
        raise ValueError("Destination selector matched no compartments.")
    dest_only = tuple(
        prop
        for col, prop in enumerate(pmap.properties)
        if prop.name != group_by and _property_present(pmap, dest_idx, col)
    )
    split_n = _validate_split(pmap, dest_idx, dest_only, split)
    n = int(dest_idx.size)
    if group_by is None:
        weights = _group_weights(pmap, dest_idx, dest_only, split_n)
    else:
        prop = pmap.get_property(group_by)
        col = np.asarray(pmap.codes[dest_idx, pmap._prop_index[prop.name]], dtype=np.int32)
        if np.any(col == _NA):
            raise ValueError(f"Cannot sum_over({prop.name!r}): some dest rows lack that property.")
        weights = np.empty(n, dtype=np.float64)
        for code in np.unique(col).tolist():
            mask = col == int(code)
            members = dest_idx[mask]
            weights[mask] = _group_weights(pmap, members, dest_only, split_n)
    return EdgeArrays(
        src_idx=np.full(n, -1, dtype=np.int32),
        dest_idx=dest_idx,
        weight=weights,
        scale=np.ones(n, dtype=np.float64),
    )


def actualize(flow: FlowLike, pmap: PropertyMap) -> ActualizedFlow:
    """Resolve ``flow`` to index arrays against ``pmap``."""
    if isinstance(flow, TransitionFlow):
        edges = _join_with_pairing(pmap, flow.source, flow.dest, flow.pairing, flow.split)
        return ActualizedFlow(
            name=flow.name,
            kind="transition",
            src_idx=edges.src_idx,
            dest_idx=edges.dest_idx,
            weight=edges.weight,
            scale=edges.scale,
            rate=flow.rate,
            absolute=flow.absolute,
        )
    if isinstance(flow, ExitFlow):
        src_idx = pmap.select(flow.source)
        if src_idx.size == 0:
            raise ValueError("Source selector matched no compartments.")
        n = int(src_idx.size)
        return ActualizedFlow(
            name=flow.name,
            kind="exit",
            src_idx=src_idx,
            dest_idx=None,
            weight=np.ones(n, dtype=np.float64),
            scale=np.ones(n, dtype=np.float64),
            rate=flow.rate,
            absolute=flow.absolute,
        )
    edges = _entry_edges(pmap, flow.dest, flow.split, group_by=_sum_over_property_name(flow.rate))
    return ActualizedFlow(
        name=flow.name,
        kind="entry",
        src_idx=None,
        dest_idx=edges.dest_idx,
        weight=edges.weight,
        scale=edges.scale,
        rate=flow.rate,
        absolute=True,
    )


def _topo_sort(flows: Sequence[ActualizedFlow]) -> list[ActualizedFlow]:
    by_name = {flow.name: flow for flow in flows}
    pending = {flow.name: _flow_refs(flow.rate) for flow in flows}
    for name, deps in pending.items():
        unknown = deps - by_name.keys()
        if unknown:
            raise KeyError(f"Flow {name!r} references unknown flow(s) {sorted(unknown)}.")
    ordered: list[ActualizedFlow] = []
    remaining = set(by_name)
    while remaining:
        ready = [name for name in remaining if pending[name].isdisjoint(remaining)]
        if not ready:
            raise ValueError(f"Cyclic flow-rate references among {sorted(remaining)}.")
        ready.sort()
        for name in ready:
            remaining.remove(name)
            ordered.append(by_name[name])
    return ordered


def _as_array(value: Any, xp: Any) -> Any:
    if isinstance(value, _SubmapRate):
        return value.data
    if hasattr(value, "data") and hasattr(value, "pmap"):
        return value.data
    return xp.asarray(value)


def _align_submap_rate(
    rate: _SubmapRate,
    gather_idx: NDArray[np.int32],
    pmap: PropertyMap,
    xp: Any,
) -> Any:
    if len(rate.properties) != 1:
        raise ValueError("sum_over alignment supports exactly one property.")
    prop = rate.properties[0]
    col_i = pmap._prop_index[prop.name]
    codes = np.asarray(pmap.codes[gather_idx, col_i], dtype=np.int32)
    if np.any(codes == _NA):
        raise ValueError(
            f"Cannot align sum_over({prop.name!r}): some gather rows lack that property."
        )
    data = _as_array(rate, xp)
    return data[..., codes]


def _align_rate(
    rate: Any,
    *,
    gather_idx: NDArray[np.int32],
    n_edges: int,
    pmap: PropertyMap,
    xp: Any,
) -> Any:
    if isinstance(rate, _SubmapRate):
        return _align_submap_rate(rate, gather_idx, pmap, xp)
    arr = _as_array(rate, xp)
    shape = getattr(arr, "shape", ())
    pmap_size = pmap.size
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


def _backend_module(backend: BackendName) -> Any:
    if backend == "numpy":
        return np
    import jax.numpy as jnp

    return jnp


def _is_property_data(value: object) -> bool:
    return type(value).__name__ == "PropertyData" and hasattr(value, "data")


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
        backend: BackendName = "numpy",
    ) -> Callable[[Any, Any, Any], Any]:
        """Actualize joins once and return ``vf(t, y, params)``."""
        if not self.flows:
            raise ValueError("FlowModel has no flows.")
        actualized = _topo_sort([actualize(flow, self.pmap) for flow in self.flows])
        flow_meta = {flow.name: flow for flow in actualized}
        pmap = self.pmap
        xp = _backend_module(backend)

        def vector_field(t: Any, y: Any, params: Any) -> Any:
            wrapped = _is_property_data(y)
            y_arr = y.data if wrapped else xp.asarray(y)
            derived = params if derived_fn is None else derived_fn(params, y=y_arr, t=t)
            dy = xp.zeros_like(y_arr)
            flow_values: dict[str, Any] = {}
            for flow in actualized:
                gather = flow.src_idx if flow.kind != "entry" else flow.dest_idx
                if gather is None:
                    raise RuntimeError(f"Flow {flow.name!r} is missing gather indices.")
                raw = _eval_rate(
                    flow.rate,
                    derived=derived,
                    flow_values=flow_values,
                    flow_meta=flow_meta,
                    pmap=pmap,
                    xp=xp,
                )
                rate = _align_rate(
                    raw,
                    gather_idx=gather,
                    n_edges=int(flow.weight.size),
                    pmap=pmap,
                    xp=xp,
                )
                scale = xp.asarray(flow.scale)
                weight = xp.asarray(flow.weight)
                if flow.kind == "entry":
                    mass = rate * scale * weight
                    dy = _scatter_add(dy, flow.dest_idx, mass, xp)
                    flow_values[flow.name] = mass
                    continue
                src_idx = cast(NDArray[np.int32], flow.src_idx)
                src_y = y_arr[..., src_idx]
                contrib = rate * scale if flow.absolute else rate * scale * src_y
                mass = contrib * weight
                dy = _scatter_add(dy, src_idx, -mass, xp)
                if flow.kind == "transition":
                    dy = _scatter_add(dy, flow.dest_idx, mass, xp)
                flow_values[flow.name] = mass
            if wrapped:
                return y._with_data(dy)
            return dy

        return vector_field


def _scatter_add(target: Any, indices: NDArray[np.int32] | None, values: Any, xp: Any) -> Any:
    if indices is None:
        raise RuntimeError("Cannot scatter to a missing index array.")
    if xp is np:
        out = np.array(target, copy=True, dtype=np.result_type(target, values))
        np.add.at(out, (..., indices) if out.ndim > 1 else indices, values)
        return out
    return target.at[..., indices].add(values)
