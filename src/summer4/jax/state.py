"""Solver state: compartments plus reserved ledger slots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from jax.tree_util import register_pytree_node_class

from summer4.jax.propertydata import PropertyData
from summer4.propertymap import PropertyMap


@register_pytree_node_class
@dataclass(frozen=True, slots=True)
class State:
    """ODE state: compartment densities plus optional named ledger arrays.

    Phase 2 populates only :attr:`compartments`. Ledgers are the extension
    point for cumulative quantities that feed back into rates (vaccination
    coverage, etc.) — those must be state, not post-run reductions.
    """

    compartments: PropertyData
    ledgers: Mapping[str, Any] = field(default_factory=dict)

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[PropertyMap, tuple[str, ...]]]:
        ledger_keys = tuple(self.ledgers.keys())
        ledger_vals = tuple(self.ledgers[k] for k in ledger_keys)
        return (self.compartments.data, *ledger_vals), (self.compartments.pmap, ledger_keys)

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[PropertyMap, tuple[str, ...]],
        children: tuple[Any, ...],
    ) -> State:
        pmap, ledger_keys = aux
        data, *ledger_vals = children
        ledgers = dict(zip(ledger_keys, ledger_vals, strict=True))
        return cls(compartments=PropertyData(pmap, data), ledgers=ledgers)

    @classmethod
    def wrap(
        cls, pmap: PropertyMap, data: Any, *, ledgers: Mapping[str, Any] | None = None
    ) -> State:
        """Build a :class:`State` from host or device compartment values."""
        return cls(compartments=PropertyData.wrap(pmap, data), ledgers=dict(ledgers or {}))


def unpack_state(
    y: State | PropertyData | Any,
    pmap: PropertyMap | None = None,
) -> tuple[Any, Callable[[Any], Any]]:
    """Return ``(y_arr, rebox)`` for ``State``, ``PropertyData``, or a bare array.

    ``rebox(dy_arr)`` rebuilds a value with the same outer type as ``y``.
    """
    if isinstance(y, State):
        y_arr = y.compartments.data

        def rebox(data: Any) -> State:
            return State(
                compartments=PropertyData(y.compartments.pmap, data),
                ledgers=y.ledgers,
            )

        return y_arr, rebox

    if isinstance(y, PropertyData):
        y_arr = y.data

        def rebox_pd(data: Any) -> PropertyData:
            return PropertyData(y.pmap, data)

        return y_arr, rebox_pd

    import jax.numpy as jnp

    y_arr = jnp.asarray(y)
    if pmap is not None:

        def rebox_arr(data: Any) -> Any:
            return data

        return y_arr, rebox_arr

    return y_arr, lambda data: data
