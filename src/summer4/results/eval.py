"""Evaluate SavePlan quantities against a SaveContext."""

from __future__ import annotations

from typing import Any

import numpy as np

from summer4.flows.edges import EdgeMap, rewrite_edge_selector, sum_over_edge
from summer4.flows.rates import _lookup_path
from summer4.jax.propertydata import PropertyData
from summer4.propertymap import PropertyMap
from summer4.results.plan import (
    Compartments,
    ComputedValue,
    FlowMass,
    GroupedOutput,
    SaveFn,
    SavePlan,
    flow_names,
)
from summer4.selectors import Selector


def _select_idx(
    pmap: PropertyMap,
    where: Selector | np.ndarray,
    *,
    edge: bool = False,
) -> np.ndarray:
    if isinstance(where, np.ndarray):
        arr = np.asarray(where)
        if arr.dtype == np.bool_ or arr.dtype == bool:
            return np.flatnonzero(arr).astype(np.int32, copy=False)
        return arr.astype(np.int32, copy=False)
    sel = rewrite_edge_selector(pmap, where) if edge else where
    return pmap.select(sel)


def _apply_where(
    data: Any,
    pmap: PropertyMap,
    where: Selector | np.ndarray | None,
    *,
    edge: bool = False,
) -> Any:
    if where is None:
        return data
    idx = _select_idx(pmap, where, edge=edge)
    return data[..., idx]


def _flow_piece(
    name: str,
    ctx: Any,
    edge_maps: dict[str, EdgeMap],
    where: Selector | np.ndarray | None,
    sum_over: tuple[Any, Any] | None,
) -> Any:
    """One flow's mass after ``where`` / ``sum_over``."""
    if name not in ctx.flows:
        raise KeyError(f"Flow {name!r} not in SaveContext.flows.")
    mass = ctx.flows[name]
    table = edge_maps[name].table
    if where is not None:
        idx = _select_idx(table, where, edge=True)
        table = table.take(idx)
        mass = mass[..., idx]
    if sum_over is None:
        return PropertyData(table, mass)
    prop, side = sum_over
    return sum_over_edge(mass, table, prop, side)


def _sum_named_flows(
    names: tuple[str, ...],
    ctx: Any,
    edge_maps: dict[str, EdgeMap],
    where: Selector | np.ndarray | None,
    sum_over: tuple[Any, Any] | None,
) -> Any:
    """Sum flows after the same filter. Selected maps must be equal."""
    acc = _flow_piece(names[0], ctx, edge_maps, where, sum_over)
    for name in names[1:]:
        piece = _flow_piece(name, ctx, edge_maps, where, sum_over)
        if (
            isinstance(acc, PropertyData)
            and isinstance(piece, PropertyData)
            and acc.pmap != piece.pmap
        ):
            raise ValueError(
                f"FlowMass flows {names[0]!r} and {name!r} do not share selected "
                "dims after where/sum_over."
            )
        acc = acc + piece
    return acc


def eval_quantity(
    what: Compartments | FlowMass | ComputedValue | SaveFn | GroupedOutput,
    ctx: Any,
    *,
    pmap: PropertyMap,
    edge_maps: dict[str, EdgeMap],
) -> Any:
    """Evaluate one Quantity against an observe() SaveContext."""
    import jax.numpy as jnp

    match what:
        case Compartments(where=where, sum_over=sum_over):
            y = ctx.y
            from summer4.jax.state import State

            if isinstance(y, State):
                y = y.compartments
            data = y.data if isinstance(y, PropertyData) else jnp.asarray(y)
            active = pmap
            if where is not None:
                idx = _select_idx(pmap, where, edge=False)
                active = pmap.take(idx)
                data = data[..., idx]
            pd = PropertyData(active, data)
            return pd.sum_over(sum_over) if sum_over is not None else pd
        case FlowMass(flow=flow, where=where, sum_over=sum_over):
            return _sum_named_flows(flow_names(flow), ctx, edge_maps, where, sum_over)
        case ComputedValue(path=path):
            return _lookup_path(ctx.derived, path)
        case GroupedOutput(name=name):
            if name not in ctx.captures:
                known = ", ".join(sorted(ctx.captures)) or "(none)"
                raise KeyError(f"GroupedOutput {name!r} not captured. Known: {known}.")
            captured = ctx.captures[name]
            prop = captured.properties[0]
            return PropertyData(PropertyMap.from_property(prop), captured.data)
        case SaveFn(fn=fn):
            return fn(ctx)
        case _:
            raise TypeError(f"Unknown quantity {type(what).__name__}.")


def values_for(req: Any, raw: Any, model: Any) -> Any:
    """Re-attach :class:`PropertyMap`s to solver-saved bare arrays.

    Both Euler and diffrax backends strip ``PropertyData.data`` when stacking
    saves; this is the single place that rebuilds the maps so the backends
    cannot disagree about an output's ``values``.
    """
    what = req.what if hasattr(req, "what") else req
    match what:
        case Compartments(where=where, sum_over=sum_over):
            if sum_over is not None:
                return PropertyData(PropertyMap.from_property(sum_over), raw)
            if where is None:
                return PropertyData(model.pmap, raw)
            idx = _select_idx(model.pmap, where, edge=False)
            return PropertyData(model.pmap.take(idx), raw)
        case FlowMass(flow=flow, where=where, sum_over=sum_over):
            # Every named flow shares this layout; evaluation already rejected
            # a mismatch, so the first flow's map is the saved array's map.
            emap = model.edge_maps[flow_names(flow)[0]]
            table = emap.table
            if where is not None:
                idx = _select_idx(table, where, edge=True)
                table = table.take(idx)
            if sum_over is not None:
                prop, _side = sum_over
                return PropertyData(PropertyMap.from_property(prop), raw)
            return PropertyData(table, raw)
        case GroupedOutput(name=name):
            meta = getattr(model, "capture_meta", {})
            props = meta.get(name)
            if props is None:
                return raw
            return PropertyData(PropertyMap.from_property(props[0]), raw)
        case _:
            return raw


def build_save_fn(
    plan: SavePlan,
    *,
    pmap: PropertyMap,
    edge_maps: dict[str, EdgeMap],
) -> Any:
    """Return ``fn(ctx) -> dict[str, array]`` for the expanded plan's requests."""

    def save_fn(ctx: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, req in plan.requests.items():
            out[key] = eval_quantity(req.what, ctx, pmap=pmap, edge_maps=edge_maps)
        return out

    return save_fn


def dims_for_quantity(
    what: Compartments | FlowMass | ComputedValue | SaveFn | GroupedOutput,
) -> tuple[str, ...]:
    match what:
        case Compartments(sum_over=sum_over):
            if sum_over is not None:
                return ("time", "group")
            return ("time", "compartment")
        case FlowMass(sum_over=sum_over):
            if sum_over is not None:
                return ("time", "group")
            return ("time", "edge")
        case GroupedOutput(name=name):
            # Property name filled in by callers that know capture_meta; default
            # keeps a stable second axis label when meta is unavailable.
            return ("time", name)
        case ComputedValue() | SaveFn():
            return ("time",)
        case _:
            return ("time",)
