"""Spike PropertyData: a PropertyMap paired with a JAX array as a pytree.

This module is not part of summer4. It exists so the explore-datatypes branch
can measure pytree dispatch cost and a small algebra before any public API
is frozen.

Two static-aux strategies are compared:

* ``variant="digest"`` — hash/eq a blake2 digest of the code table so
  structurally equal maps share a treedef.
* ``variant="identity"`` — hash/eq by ``id(pmap)``. Dispatch is cheaper;
  a new equal map retraces.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from summer4 import Property, PropertyMap, Selector, Trait

VARIANT_DIGEST = "digest"
VARIANT_IDENTITY = "identity"
_NA = -1


def _digest_bytes(pmap: PropertyMap) -> bytes:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(pmap.codes.tobytes())
    if pmap.parent_row is not None:
        hasher.update(b"|pr|")
        hasher.update(pmap.parent_row.tobytes())
    else:
        hasher.update(b"|pr|none")
    return hasher.digest()


class DigestStatic:
    """Hashable PropertyMap stand-in: structural digest (variant A)."""

    __slots__ = ("digest", "pmap")

    def __init__(self, pmap: PropertyMap) -> None:
        self.pmap = pmap
        self.digest = _digest_bytes(pmap)

    def __hash__(self) -> int:
        return hash((self.pmap.properties, self.pmap.history, self.digest))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DigestStatic):
            return NotImplemented
        return (
            self.digest == other.digest
            and self.pmap.properties == other.pmap.properties
            and self.pmap.history == other.pmap.history
        )


class IdentityStatic:
    """Hashable PropertyMap stand-in: object identity (variant B)."""

    __slots__ = ("pmap",)

    def __init__(self, pmap: PropertyMap) -> None:
        self.pmap = pmap

    def __hash__(self) -> int:
        return id(self.pmap)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IdentityStatic):
            return NotImplemented
        return self.pmap is other.pmap


def _wrap_static(pmap: PropertyMap, variant: str) -> DigestStatic | IdentityStatic:
    if variant == VARIANT_DIGEST:
        return DigestStatic(pmap)
    if variant == VARIANT_IDENTITY:
        return IdentityStatic(pmap)
    raise ValueError(
        f"Unknown variant {variant!r}; use {VARIANT_DIGEST!r} or {VARIANT_IDENTITY!r}."
    )


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
    """JAX array whose last axis is aligned with a :class:`PropertyMap`.

    ``__init__`` does not check ``data.shape``. JAX ``tree_map`` / ``unflatten``
    pass tracers and sentinels; validate with :meth:`check` at the Python edge.
    """

    pmap: PropertyMap
    data: Any
    variant: str = VARIANT_DIGEST
    _static: DigestStatic | IdentityStatic | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._static is None:
            object.__setattr__(self, "_static", _wrap_static(self.pmap, self.variant))

    @classmethod
    def wrap(
        cls,
        pmap: PropertyMap,
        data: Any,
        *,
        variant: str = VARIANT_DIGEST,
    ) -> PropertyData:
        """Build from host or device values and optionally check the last axis."""
        return cls(pmap, jnp.asarray(data), variant=variant).check()

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
        return PropertyData(self.pmap, data, variant=self.variant, _static=self._static)

    def tree_flatten(self) -> tuple[tuple[Any], DigestStatic | IdentityStatic]:
        static = self._static
        if static is None:  # pragma: no cover - post_init always sets this
            raise RuntimeError("PropertyData is missing its static wrapper.")
        return (self.data,), static

    @classmethod
    def tree_unflatten(
        cls,
        aux: DigestStatic | IdentityStatic,
        children: tuple[Any, ...],
    ) -> PropertyData:
        (data,) = children
        variant = VARIANT_DIGEST if isinstance(aux, DigestStatic) else VARIANT_IDENTITY
        return cls(aux.pmap, data, variant=variant, _static=aux)

    @property
    def at(self) -> _AtHelper:
        return _AtHelper(self)

    def __getitem__(self, sel: Selector) -> Any:
        """Gather values where ``sel`` is Kleene-true."""
        return self.data[..., self.pmap.select(sel)]

    def subset(self, sel: Selector) -> PropertyData:
        """Return a PropertyData over the rows matching ``sel``."""
        idx = self.pmap.select(sel)
        new_map = PropertyMap(properties=self.pmap.properties, codes=self.pmap.codes[idx])
        return PropertyData(new_map, self.data[..., idx], variant=self.variant)

    def where(self, sel: Selector, other: Any) -> PropertyData:
        """Replace matching compartments with ``other`` (scalar or PropertyData)."""
        mask = self.pmap.mask(sel)
        other_data = other.data if isinstance(other, PropertyData) else other
        if isinstance(other, PropertyData):
            _require_same_map(self.pmap, other.pmap)
        return self._with_data(jnp.where(mask, other_data, self.data))

    def partition(self, prop: Property | str) -> dict[Trait, Any]:
        """Gather values for each trait of ``prop``."""
        return {trait: self.data[..., idx] for trait, idx in self.pmap.partition(prop).items()}

    def sum_over(self, prop: Property | str) -> PropertyData:
        """Sum compartments by trait of ``prop``.

        The result map is ``PropertyMap.from_property(prop)``. Compartments
        where ``prop`` is absent do not contribute.
        """
        resolved = _resolve_property(self.pmap, prop)
        col_i = next(i for i, p in enumerate(self.pmap.properties) if p.name == resolved.name)
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
        return PropertyData(PropertyMap.from_property(resolved), reduced, variant=self.variant)

    def broadcast_over(self, prop: Property | str, values: Any) -> PropertyData:
        """Expand per-trait ``values`` onto this map along ``prop``.

        ``values`` is a ``(…, n_traits)`` array or a single-property
        :class:`PropertyData`. Absent compartments receive 0.
        """
        resolved = _resolve_property(self.pmap, prop)
        if isinstance(values, PropertyData):
            _require_same_map(values.pmap, PropertyMap.from_property(resolved))
            values = values.data
        col_i = next(i for i, p in enumerate(self.pmap.properties) if p.name == resolved.name)
        col = np.asarray(self.pmap.codes[:, col_i], dtype=np.int32)
        n_traits = len(resolved.traits)
        pad = jnp.zeros(values.shape[:-1] + (1,), dtype=values.dtype)
        padded = jnp.concatenate([values, pad], axis=-1)
        safe = np.where(col >= 0, col, n_traits).astype(np.int32)
        return self._with_data(padded[..., safe])

    def normalise_within(self, prop: Property | str) -> PropertyData:
        """Divide each compartment by the sum of its ``prop`` group."""
        totals = self.sum_over(prop)
        denom = self.broadcast_over(prop, totals)
        return self._with_data(self.data / denom.data)

    def sum_over_onehot(self, prop: Property | str) -> PropertyData:
        """Reference ``sum_over`` via a static one-hot matmul."""
        resolved = _resolve_property(self.pmap, prop)
        col_i = next(i for i, p in enumerate(self.pmap.properties) if p.name == resolved.name)
        col = np.asarray(self.pmap.codes[:, col_i], dtype=np.int32)
        n_traits = len(resolved.traits)
        onehot = (col[:, None] == np.arange(n_traits)).astype(np.float32)
        reduced = self.data @ onehot
        return PropertyData(PropertyMap.from_property(resolved), reduced, variant=self.variant)

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
