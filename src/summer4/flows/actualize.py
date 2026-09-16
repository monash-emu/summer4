"""Resolve declared flows against a :class:`PropertyMap` into typed edge arrays."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from summer4.flows.edges import EdgeMap, EdgeRoles
from summer4.flows.join import (
    _group_weights,
    _property_present,
    _validate_split,
    join_with_pairing,
    selector_values,
)
from summer4.flows.rates import Adjustment, FlowRef, RateOps, _flow_rate_refs
from summer4.flows.types import ExitFlow, FlowLike, TransitionFlow
from summer4.properties import Property
from summer4.propertymap import PropertyMap
from summer4.selectors import Selector

_NA: int = -1


def _freeze_int32(values: NDArray[np.int32]) -> NDArray[np.int32]:
    array = np.ascontiguousarray(values, dtype=np.int32)
    array.flags.writeable = False
    return array


def _freeze_float64(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.ascontiguousarray(values, dtype=np.float64)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class TransitionEdges:
    """Resolved transition: both endpoints present."""

    name: str
    src_idx: NDArray[np.int32]
    dest_idx: NDArray[np.int32]
    weight: NDArray[np.float64]
    scale: NDArray[np.float64]
    rate: RateOps
    absolute: bool
    adjust: tuple[Adjustment, ...]
    adjust_masks: tuple[NDArray[np.bool_] | None, ...]
    edge_map: EdgeMap
    pair_src_codes: NDArray[np.int32] | None = None
    pair_dest_codes: NDArray[np.int32] | None = None
    pair_n_traits: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "src_idx", _freeze_int32(self.src_idx))
        object.__setattr__(self, "dest_idx", _freeze_int32(self.dest_idx))
        object.__setattr__(self, "weight", _freeze_float64(self.weight))
        object.__setattr__(self, "scale", _freeze_float64(self.scale))
        if self.pair_src_codes is not None:
            object.__setattr__(self, "pair_src_codes", _freeze_int32(self.pair_src_codes))
        if self.pair_dest_codes is not None:
            object.__setattr__(self, "pair_dest_codes", _freeze_int32(self.pair_dest_codes))

    @property
    def n_edges(self) -> int:
        return int(self.weight.size)


@dataclass(frozen=True, slots=True)
class ExitEdges:
    """Resolved exit: source only; destination columns are absent."""

    name: str
    src_idx: NDArray[np.int32]
    weight: NDArray[np.float64]
    scale: NDArray[np.float64]
    rate: RateOps
    absolute: bool
    adjust: tuple[Adjustment, ...]
    adjust_masks: tuple[NDArray[np.bool_] | None, ...]
    edge_map: EdgeMap

    def __post_init__(self) -> None:
        object.__setattr__(self, "src_idx", _freeze_int32(self.src_idx))
        object.__setattr__(self, "weight", _freeze_float64(self.weight))
        object.__setattr__(self, "scale", _freeze_float64(self.scale))

    @property
    def n_edges(self) -> int:
        return int(self.weight.size)


@dataclass(frozen=True, slots=True)
class EntryEdges:
    """Resolved entry: destination only; source columns are absent."""

    name: str
    dest_idx: NDArray[np.int32]
    weight: NDArray[np.float64]
    scale: NDArray[np.float64]
    rate: RateOps
    absolute: bool
    adjust: tuple[Adjustment, ...]
    adjust_masks: tuple[NDArray[np.bool_] | None, ...]
    edge_map: EdgeMap

    def __post_init__(self) -> None:
        object.__setattr__(self, "dest_idx", _freeze_int32(self.dest_idx))
        object.__setattr__(self, "weight", _freeze_float64(self.weight))
        object.__setattr__(self, "scale", _freeze_float64(self.scale))

    @property
    def n_edges(self) -> int:
        return int(self.weight.size)


type FlowEdges = TransitionEdges | ExitEdges | EntryEdges


def _pairing_codes(
    pmap: PropertyMap,
    pairing: object,
    src_idx: NDArray[np.int32],
    dest_idx: NDArray[np.int32],
) -> tuple[NDArray[np.int32] | None, NDArray[np.int32] | None, int | None]:
    from summer4.flows.join import TraitMatrix

    if not isinstance(pairing, TraitMatrix):
        return None, None, None
    col = pmap.column_index(pairing.property)
    src_codes = np.asarray(pmap.codes[src_idx, col], dtype=np.int32)
    dest_codes = np.asarray(pmap.codes[dest_idx, col], dtype=np.int32)
    return src_codes, dest_codes, len(pairing.property.traits)


def _bind_adjust_masks(
    adjust: tuple[Adjustment, ...],
    gather_idx: NDArray[np.int32],
    pmap: PropertyMap,
) -> tuple[NDArray[np.bool_] | None, ...]:
    masks: list[NDArray[np.bool_] | None] = []
    for adj in adjust:
        if adj.where is None:
            masks.append(None)
            continue
        selected = pmap.select(adj.where)
        mask = np.isin(gather_idx, selected)
        mask.flags.writeable = False
        masks.append(mask)
    return tuple(masks)


def _sum_over_property_name(expr: RateOps) -> str | None:
    """Property name for entry-weight grouping, if the rate is trait-aligned."""
    from summer4.flows.rates import BinOp, Capture, Reduce

    match expr:
        case FlowRef(reduce=("sum_over", name)):
            return name
        case Reduce(sum_over=sum_over):
            return sum_over.name if isinstance(sum_over, Property) else sum_over
        case Capture(inner=inner):
            return _sum_over_property_name(inner)
        case BinOp(left=left, right=right):
            # Prefer a Reduce / sum_over on either side (FOI = contact * grouped).
            return _sum_over_property_name(left) or _sum_over_property_name(right)
        case _:
            custom = getattr(expr, "group_by", None)
            if isinstance(custom, Property):
                return custom.name
            return None


def _side_roles(
    pmap: PropertyMap,
    idx: NDArray[np.int32],
    bound: frozenset[str],
    *,
    source: bool,
) -> EdgeRoles:
    present: set[str] = set()
    for prop in pmap.properties:
        if _property_present(pmap, idx, pmap.column_index(prop)):
            present.add(prop.name)
    leftover = present - bound
    if source:
        return EdgeRoles(
            bound=bound,
            free=frozenset(),
            dest_only=frozenset(),
            src_only=frozenset(leftover),
            paired=frozenset(),
        )
    return EdgeRoles(
        bound=bound,
        free=frozenset(),
        dest_only=frozenset(leftover),
        src_only=frozenset(),
        paired=frozenset(),
    )


def _entry_indices(
    pmap: PropertyMap,
    dest: Selector,
    split: dict[str, dict[str, float]] | None,
    group_by: str | None,
) -> tuple[NDArray[np.int32], NDArray[np.float64], tuple[Property, ...]]:
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
        col = np.asarray(pmap.column(prop)[dest_idx], dtype=np.int32)
        if np.any(col == _NA):
            raise ValueError(f"Cannot sum_over({prop.name!r}): some dest rows lack that property.")
        weights = np.empty(n, dtype=np.float64)
        unique, inverse = np.unique(col, return_inverse=True)
        for group_i in range(int(unique.size)):
            mask = inverse == group_i
            members = dest_idx[mask]
            weights[mask] = _group_weights(pmap, members, dest_only, split_n)
    return dest_idx, weights, dest_only


def actualize(flow: FlowLike, pmap: PropertyMap, *, strict_pairing: bool = True) -> FlowEdges:
    """Resolve ``flow`` to typed edge arrays against ``pmap``."""
    if isinstance(flow, TransitionFlow):
        edges = join_with_pairing(
            pmap,
            flow.source,
            flow.dest,
            flow.pairing,
            flow.split,
            strict_pairing=strict_pairing,
        )
        src_codes, dest_codes, n_traits = _pairing_codes(
            pmap, flow.pairing, edges.src_idx, edges.dest_idx
        )
        return TransitionEdges(
            name=flow.name,
            src_idx=edges.src_idx,
            dest_idx=edges.dest_idx,
            weight=edges.weight,
            scale=edges.scale,
            rate=flow.rate,
            absolute=flow.absolute,
            adjust=flow.adjust,
            adjust_masks=_bind_adjust_masks(flow.adjust, edges.src_idx, pmap),
            edge_map=edges.edge_map,
            pair_src_codes=src_codes,
            pair_dest_codes=dest_codes,
            pair_n_traits=n_traits,
        )
    if isinstance(flow, ExitFlow):
        src_idx = pmap.select(flow.source)
        if src_idx.size == 0:
            raise ValueError("Source selector matched no compartments.")
        n = int(src_idx.size)
        roles = _side_roles(pmap, src_idx, selector_values(flow.source), source=True)
        edge_map = EdgeMap.from_indices(pmap, src_idx, None, roles)
        return ExitEdges(
            name=flow.name,
            src_idx=src_idx,
            weight=np.ones(n, dtype=np.float64),
            scale=np.ones(n, dtype=np.float64),
            rate=flow.rate,
            absolute=flow.absolute,
            adjust=flow.adjust,
            adjust_masks=_bind_adjust_masks(flow.adjust, src_idx, pmap),
            edge_map=edge_map,
        )
    dest_idx, weights, _dest_only = _entry_indices(
        pmap, flow.dest, flow.split, group_by=_sum_over_property_name(flow.rate)
    )
    roles = _side_roles(pmap, dest_idx, selector_values(flow.dest), source=False)
    edge_map = EdgeMap.from_indices(pmap, None, dest_idx, roles)
    return EntryEdges(
        name=flow.name,
        dest_idx=dest_idx,
        weight=weights,
        scale=np.ones(int(dest_idx.size), dtype=np.float64),
        rate=flow.rate,
        absolute=True,
        adjust=flow.adjust,
        adjust_masks=_bind_adjust_masks(flow.adjust, dest_idx, pmap),
        edge_map=edge_map,
    )


def topo_sort(flows: Sequence[FlowEdges]) -> list[FlowEdges]:
    """Return flows in an order that satisfies :class:`FlowRef` dependencies."""
    by_name = {flow.name: flow for flow in flows}
    pending = {flow.name: _flow_rate_refs(flow.rate, flow.adjust) for flow in flows}
    for name, deps in pending.items():
        unknown = deps - by_name.keys()
        if unknown:
            raise KeyError(f"Flow {name!r} references unknown flow(s) {sorted(unknown)}.")
    ordered: list[FlowEdges] = []
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
