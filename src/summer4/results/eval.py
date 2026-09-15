"""Evaluate SavePlan quantities against a SaveContext."""

from __future__ import annotations

from typing import Any

import numpy as np

from summer4.flows.edges import EdgeMap, rewrite_edge_selector, sum_over_edge
from summer4.flows.rates import _lookup_path
from summer4.jax.propertydata import PropertyData
from summer4.propertymap import PropertyMap
from summer4.results.plan import Compartments, ComputedValue, FlowMass, SaveFn, SavePlan
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


def eval_quantity(
    what: Compartments | FlowMass | ComputedValue | SaveFn,
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
            if flow not in ctx.flows:
                raise KeyError(f"Flow {flow!r} not in SaveContext.flows.")
            mass = ctx.flows[flow]
            emap = edge_maps[flow]
            table = emap.table
            if where is not None:
                idx = _select_idx(table, where, edge=True)
                table = table.take(idx)
                mass = mass[..., idx]
            if sum_over is None:
                return PropertyData(table, mass)
            prop, side = sum_over
            return sum_over_edge(mass, table, prop, side)
        case ComputedValue(path=path):
            return _lookup_path(ctx.derived, path)
        case SaveFn(fn=fn):
            return fn(ctx)
        case _:
            raise TypeError(f"Unknown quantity {type(what).__name__}.")


def values_for(req: Any, raw: Any, model: Any) -> Any:
    """Re-attach :class:`PropertyMap`s to solver-saved bare arrays.

    Both Euler and diffrax backends strip ``PropertyData.data`` when stacking
    saves; this is the single place that rebuilds the maps so the backends
    cannot disagree about a trace's ``values``.
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
            emap = model.edge_maps[flow]
            table = emap.table
            if where is not None:
                idx = _select_idx(table, where, edge=True)
                table = table.take(idx)
            if sum_over is not None:
                prop, _side = sum_over
                return PropertyData(PropertyMap.from_property(prop), raw)
            return PropertyData(table, raw)
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


def dims_for_quantity(what: Compartments | FlowMass | ComputedValue | SaveFn) -> tuple[str, ...]:
    match what:
        case Compartments(sum_over=sum_over):
            if sum_over is not None:
                return ("time", "group")
            return ("time", "compartment")
        case FlowMass(sum_over=sum_over):
            if sum_over is not None:
                return ("time", "group")
            return ("time", "edge")
        case ComputedValue() | SaveFn():
            return ("time",)
        case _:
            return ("time",)
