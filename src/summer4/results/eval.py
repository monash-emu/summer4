"""Evaluate SavePlan quantities against a SaveContext."""

from __future__ import annotations

from typing import Any

import numpy as np

from summer4.flows.edges import EdgeMap
from summer4.flows.rates import _lookup_path
from summer4.jax.propertydata import PropertyData
from summer4.propertymap import PropertyMap
from summer4.results.plan import Compartments, ComputedValue, FlowMass, SaveFn, SavePlan
from summer4.selectors import Selector


def _apply_where(data: Any, pmap: PropertyMap, where: Selector | np.ndarray | None) -> Any:
    if where is None:
        return data
    if isinstance(where, np.ndarray):
        return data[..., where]
    idx = pmap.select(where)
    return data[..., idx]


def _sum_over_edge(
    mass: Any,
    edge_map: EdgeMap,
    prop: Any,
    side: str,
) -> Any:
    import jax
    import jax.numpy as jnp

    from summer4.properties import Property

    if not isinstance(prop, Property):
        prop = edge_map.table.get_property(prop)
    suffix = "@source" if side == "source" else "@dest"
    mangled = edge_map.table.get_property(f"{prop.name}{suffix}")
    col_i = edge_map.table.column_index(mangled)
    codes = np.asarray(edge_map.table.codes[:, col_i], dtype=np.int32)
    n_traits = len(prop.traits)
    valid = codes >= 0
    safe = np.where(valid, codes, 0).astype(np.int32)
    mass_j = jnp.asarray(mass)
    weighted = mass_j * jnp.asarray(valid)

    def _row(row: Any) -> Any:
        return jax.ops.segment_sum(row, safe, num_segments=n_traits)

    if mass_j.ndim == 1:
        return _row(weighted)
    flat = jnp.reshape(weighted, (-1, mass_j.shape[-1]))
    summed = jax.vmap(_row)(flat)
    return jnp.reshape(summed, mass_j.shape[:-1] + (n_traits,))


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
            if sum_over is not None:
                pd = PropertyData(pmap, data)
                if where is not None:
                    mask = pmap.mask(where)
                    pd = PropertyData(pmap, jnp.where(mask, pd.data, 0))
                return pd.sum_over(sum_over)
            data = _apply_where(data, pmap, where)
            if where is None:
                return PropertyData(pmap, data)
            return data
        case FlowMass(flow=flow, where=where, sum_over=sum_over):
            if flow not in ctx.flows:
                raise KeyError(f"Flow {flow!r} not in SaveContext.flows.")
            mass = ctx.flows[flow]
            emap = edge_maps[flow]
            mass = _apply_where(mass, emap.table, where)
            if sum_over is None:
                return mass
            prop, side = sum_over
            return _sum_over_edge(ctx.flows[flow], emap, prop, side)
        case ComputedValue(path=path):
            return _lookup_path(ctx.derived, path)
        case SaveFn(fn=fn):
            return fn(ctx)
        case _:
            raise TypeError(f"Unknown quantity {type(what).__name__}.")


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
