"""Pointwise operators shared by rate trees and output traces.

:class:`~summer4.flows.rates.UnaryOp` and :class:`~summer4.flows.rates.BinOp`
are the symbolic form. Evaluating them, and applying the same operation to an
already-evaluated value (a JAX array, a :class:`~summer4.flows.compiled.GroupedRate`,
or the :class:`~summer4.jax.propertydata.PropertyData` inside a
:class:`~summer4.results.trace.Trace`), both go through :func:`apply_unary` and
:func:`apply_binary`.

Wrapper policy stays with the wrapper. A ``GroupedRate`` combines only with the
same grouping; a ``PropertyData`` only with an equal map. Neither broadcasts
across names. Name-aligned ``Trace``-to-``Trace`` broadcasting is a later step;
what this module provides is the numeric kernel that step will call after the
indexes are aligned, and the kernel a parameter-only rate expression uses when
:func:`~summer4.flows.compiled.eval_closed` turns it into an array a ``Trace``
can scale by.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

UNARY_OPS: dict[str, Callable[[Any], Any]] = {
    "neg": jnp.negative,
    "exp": jnp.exp,
    "log": jnp.log,
    "abs": jnp.abs,
    "tanh": jnp.tanh,
    "sqrt": jnp.sqrt,
    "floor": jnp.floor,
}

BINARY_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "add": jnp.add,
    "sub": jnp.subtract,
    "mul": jnp.multiply,
    "div": jnp.divide,
    "pow": jnp.power,
    "maximum": jnp.maximum,
    "minimum": jnp.minimum,
}


def apply_unary(op: str, value: Any) -> Any:
    """Apply ``op`` pointwise, preserving a ``GroupedRate`` or ``PropertyData``."""
    fn = UNARY_OPS.get(op)
    if fn is None:
        known = ", ".join(sorted(UNARY_OPS))
        raise ValueError(f"Unknown unary op {op!r}. Known ops: {known}.")
    return _map_unary(fn, value)


def apply_binary(op: str, left: Any, right: Any) -> Any:
    """Apply ``op`` pointwise, preserving a ``GroupedRate`` or ``PropertyData``."""
    fn = BINARY_OPS.get(op)
    if fn is None:
        known = ", ".join(sorted(BINARY_OPS))
        raise ValueError(f"Unknown binary op {op!r}. Known ops: {known}.")
    return _map_binary(fn, left, right)


def _map_unary(fn: Callable[[Any], Any], value: Any) -> Any:
    from summer4.flows.compiled import GroupedRate
    from summer4.jax.propertydata import PropertyData

    if isinstance(value, GroupedRate):
        return GroupedRate(data=fn(value.data), properties=value.properties)
    if isinstance(value, PropertyData):
        return PropertyData(value.pmap, fn(value.data))
    return fn(value)


def _map_binary(fn: Callable[[Any, Any], Any], left: Any, right: Any) -> Any:
    from summer4.flows.compiled import GroupedRate
    from summer4.jax.propertydata import PropertyData, _require_same_map

    left_grouped = isinstance(left, GroupedRate)
    right_grouped = isinstance(right, GroupedRate)
    left_pdata = isinstance(left, PropertyData)
    right_pdata = isinstance(right, PropertyData)
    if (left_grouped or right_grouped) and (left_pdata or right_pdata):
        raise TypeError("Cannot combine GroupedRate with PropertyData.")
    if left_grouped and right_grouped:
        left._require_same_grouping(right)
        return GroupedRate(data=fn(left.data, right.data), properties=left.properties)
    if left_grouped:
        return GroupedRate(data=fn(left.data, right), properties=left.properties)
    if right_grouped:
        return GroupedRate(data=fn(left, right.data), properties=right.properties)
    if left_pdata and right_pdata:
        _require_same_map(left.pmap, right.pmap)
        return PropertyData(left.pmap, fn(left.data, right.data))
    if left_pdata:
        return PropertyData(left.pmap, fn(left.data, right))
    if right_pdata:
        return PropertyData(right.pmap, fn(left, right.data))
    return fn(left, right)
