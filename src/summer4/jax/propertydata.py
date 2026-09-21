"""Slim PropertyData: a PropertyMap paired with a JAX array as a pytree.

``__init__`` does not check ``data.shape`` because JAX unflatten passes tracers;
validate with :meth:`PropertyData.check` / :meth:`wrap`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from summer4.properties import Property, Trait
from summer4.propertymap import PropertyMap
from summer4.selectors import Selector


def _resolve_property(pmap: PropertyMap, prop: Property | str) -> Property:
    if isinstance(prop, str):
        return pmap.get_property(prop)
    existing = pmap.get_property(prop.name)
    if existing.traits != prop.traits:
        raise ValueError(
            f"Property {prop.name!r} is registered with traits {existing.traits}, "
            f"not {prop.traits}."
        )
    return existing


def _require_same_map(left: PropertyMap, right: PropertyMap) -> None:
    if left is right:
        return
    if left != right:
        raise ValueError("PropertyData operands must share an equal PropertyMap.")


class _AtOp:
    """Scatter helper returned by ``PropertyData.at[sel]``."""

    def __init__(self, owner: PropertyData, sel: Selector) -> None:
        self._owner = owner
        self._idx = owner.pmap.select(sel)

    def set(self, value: Any) -> PropertyData:
        return self._owner._with_data(self._owner.data.at[..., self._idx].set(value))

    def add(self, value: Any) -> PropertyData:
        return self._owner._with_data(self._owner.data.at[..., self._idx].add(value))

    def mul(self, value: Any) -> PropertyData:
        return self._owner._with_data(self._owner.data.at[..., self._idx].mul(value))


class _AtHelper:
    def __init__(self, owner: PropertyData) -> None:
        self._owner = owner

    def __getitem__(self, sel: Selector) -> _AtOp:
        return _AtOp(self._owner, sel)


@register_pytree_node_class
@dataclass(frozen=True, slots=True)
class PropertyData:
    """JAX array whose last axis is aligned with a :class:`PropertyMap`."""

    pmap: PropertyMap
    data: Any

    @classmethod
    def wrap(cls, pmap: PropertyMap, data: Any) -> PropertyData:
        """Build from host or device values and check the last axis."""
        return cls(pmap, jnp.asarray(data)).check()

    def check(self) -> PropertyData:
        """Raise if the last axis is not ``pmap.size``. Skip for tracers."""
        data = self.data
        size = getattr(data, "shape", None)
        if size is None or not size:
            return self
        if not isinstance(size[-1], int):
            return self
        if size[-1] != self.pmap.size:
            raise ValueError(
                f"data last axis {size[-1]} does not match pmap.size {self.pmap.size}."
            )
        return self

    def _with_data(self, data: Any) -> PropertyData:
        return PropertyData(self.pmap, data)

    def tree_flatten(self) -> tuple[tuple[Any], PropertyMap]:
        return (self.data,), self.pmap

    @classmethod
    def tree_unflatten(cls, aux: PropertyMap, children: tuple[Any, ...]) -> PropertyData:
        (data,) = children
        return cls(aux, data)

    @property
    def at(self) -> _AtHelper:
        return _AtHelper(self)

    def __getitem__(self, sel: Selector) -> Any:
        """Gather values where ``sel`` is Kleene-true."""
        return self.data[..., self.pmap.select(sel)]

    def where(self, sel: Selector, other: Any) -> PropertyData:
        """Replace matching compartments with ``other`` (scalar or PropertyData).

        Polarity note: this *replaces* what ``sel`` matches (pandas
        ``Series.mask`` semantics). To *keep* matches and zero the rest, use
        :meth:`keep` or write ``where(~sel, other)``.
        """
        mask = self.pmap.mask(sel)
        other_data = other.data if isinstance(other, PropertyData) else other
        if isinstance(other, PropertyData):
            _require_same_map(self.pmap, other.pmap)
        return self._with_data(jnp.where(mask, other_data, self.data))

    def keep(self, sel: Selector, other: Any = 0.0) -> PropertyData:
        """Keep matching compartments; replace the rest with ``other``.

        Equivalent to ``where(~sel, other)``. Prefer this spelling when summing
        a subset (e.g. infectious prevalence) so the polarity matches
        ``select`` / ``sum_over`` / :class:`~summer4.flows.rates.Reduce`.
        """
        return self.where(~sel, other)

    def partition(self, prop: Property | str) -> dict[Trait, Any]:
        """Gather values for each trait of ``prop``."""
        return {trait: self.data[..., idx] for trait, idx in self.pmap.partition(prop).items()}

    def sum_over(self, prop: Property | str) -> PropertyData:
        """Sum compartments by trait of ``prop``.

        The result map is ``PropertyMap.from_property(prop)``. Compartments
        where ``prop`` is absent do not contribute.
        """
        resolved = _resolve_property(self.pmap, prop)
        col_i = self.pmap.column_index(resolved)
        col = np.asarray(self.pmap.codes[:, col_i], dtype=np.int32)
        n_traits = len(resolved.traits)
        valid = col >= 0
        safe_ids = np.where(valid, col, 0).astype(np.int32)
        weighted = self.data * valid
        n = self.pmap.size

        def _row(row: Any) -> Any:
            return jax.ops.segment_sum(row, safe_ids, num_segments=n_traits)

        flat = jnp.reshape(weighted, (-1, n))
        summed = jax.vmap(_row)(flat)
        out_shape = self.data.shape[:-1] + (n_traits,)
        reduced = jnp.reshape(summed, out_shape)
        return PropertyData(PropertyMap.from_property(resolved), reduced)

    def broadcast_over(self, prop: Property | str, values: Any) -> PropertyData:
        """Expand per-trait ``values`` onto this map along ``prop``.

        Absent compartments receive 0.
        """
        resolved = _resolve_property(self.pmap, prop)
        if isinstance(values, PropertyData):
            _require_same_map(values.pmap, PropertyMap.from_property(resolved))
            values = values.data
        col_i = self.pmap.column_index(resolved)
        col = np.asarray(self.pmap.codes[:, col_i], dtype=np.int32)
        n_traits = len(resolved.traits)
        pad = jnp.zeros(values.shape[:-1] + (1,), dtype=values.dtype)
        padded = jnp.concatenate([values, pad], axis=-1)
        safe = np.where(col >= 0, col, n_traits).astype(np.int32)
        return self._with_data(padded[..., safe])

    def _binary(self, other: Any, op: Any) -> PropertyData:
        if isinstance(other, PropertyData):
            _require_same_map(self.pmap, other.pmap)
            return self._with_data(op(self.data, other.data))
        return self._with_data(op(self.data, other))

    def __add__(self, other: Any) -> PropertyData:
        return self._binary(other, jnp.add)

    def __radd__(self, other: Any) -> PropertyData:
        return self._with_data(jnp.add(other, self.data))

    def __sub__(self, other: Any) -> PropertyData:
        return self._binary(other, jnp.subtract)

    def __rsub__(self, other: Any) -> PropertyData:
        return self._with_data(jnp.subtract(other, self.data))

    def __mul__(self, other: Any) -> PropertyData:
        return self._binary(other, jnp.multiply)

    def __rmul__(self, other: Any) -> PropertyData:
        return self._with_data(jnp.multiply(other, self.data))

    def __truediv__(self, other: Any) -> PropertyData:
        return self._binary(other, jnp.divide)

    def __rtruediv__(self, other: Any) -> PropertyData:
        return self._with_data(jnp.divide(other, self.data))

    def __neg__(self) -> PropertyData:
        return self._with_data(jnp.negative(self.data))

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        from summer4.flows.algebra import dispatch_ufunc

        return dispatch_ufunc(ufunc, method, inputs, kwargs, mode="value")

    def __array_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        from summer4.flows.algebra import dispatch_array_function

        return dispatch_array_function(func, types, args, kwargs, mode="value")
