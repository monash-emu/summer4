"""Immutable integer-coded compartment taxonomy."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from summer4.properties import Property, Trait
from summer4.selectors import (
    Absent,
    And,
    Dest,
    Everything,
    IsIn,
    Not,
    Nothing,
    Or,
    Present,
    Selector,
    SelectorOps,
    Source,
)

_NA: int = -1
_TRUE: np.int8 = np.int8(1)
_FALSE: np.int8 = np.int8(-1)
_UNKNOWN: np.int8 = np.int8(0)


def _freeze_int16(values: NDArray[np.int16]) -> NDArray[np.int16]:
    array = np.ascontiguousarray(values, dtype=np.int16)
    array.flags.writeable = False
    return array


def _freeze_int32(values: NDArray[np.int32]) -> NDArray[np.int32]:
    array = np.ascontiguousarray(values, dtype=np.int32)
    array.flags.writeable = False
    return array


def _as_int8(values: np.ndarray) -> NDArray[np.int8]:
    return values.astype(np.int8, copy=False)


def _digest_bytes(codes: NDArray[np.int16], parent_row: NDArray[np.int32] | None) -> bytes:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(codes.tobytes())
    if parent_row is not None:
        hasher.update(b"|pr|")
        hasher.update(parent_row.tobytes())
    else:
        hasher.update(b"|pr|none")
    return hasher.digest()


@dataclass(frozen=True, slots=True)
class Groups[V](Mapping[tuple[Trait, ...], V]):
    """Ordered mapping from realised trait combinations to values.

    Returned by :meth:`PropertyMap.group_by`. Keys are always tuples of
    :class:`~summer4.properties.Trait` (length equals the number of grouping
    properties). Empty combinations are omitted.
    """

    _data: dict[tuple[Trait, ...], V]

    def __getitem__(self, key: tuple[Trait, ...]) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[tuple[Trait, ...]]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Groups({self._data!r})"


@dataclass(frozen=True, slots=True)
class Stratification:
    """The application of a :class:`Property` to a :class:`PropertyMap`."""

    property: Property
    where: Selector | None = None

    def apply(self, pmap: PropertyMap) -> PropertyMap:
        """Return a new map with this stratification applied."""
        return pmap._apply_stratification(self)


@dataclass(frozen=True, slots=True, eq=False)
class PropertyMap:
    """Frozen compartment table: one row per compartment, one column per property.

    Codes are ``int16``. ``-1`` means the property does not apply to that
    compartment. All other values are trait codes from the column's
    :class:`Property`.
    """

    properties: tuple[Property, ...]
    codes: NDArray[np.int16]
    history: tuple[Stratification, ...] = ()
    parent_row: NDArray[np.int32] | None = None
    _cache: dict[Selector, NDArray[np.int8]] = field(init=False, repr=False, compare=False)
    _prop_index: dict[str, int] = field(init=False, repr=False, compare=False)
    _digest: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = [prop.name for prop in self.properties]
        if len(set(names)) != len(names):
            raise ValueError(f"Property names must be unique, got {names}.")
        codes = _freeze_int16(self.codes)
        if codes.ndim != 2:
            raise ValueError(f"codes must be 2-D, got shape {codes.shape}.")
        if codes.shape[1] != len(self.properties):
            raise ValueError(
                f"codes has {codes.shape[1]} columns but {len(self.properties)} properties."
            )
        object.__setattr__(self, "codes", codes)
        if self.parent_row is not None:
            object.__setattr__(self, "parent_row", _freeze_int32(self.parent_row))
            if self.parent_row.shape != (codes.shape[0],):
                raise ValueError(
                    f"parent_row length {self.parent_row.shape[0]} "
                    f"does not match {codes.shape[0]} compartments."
                )
        object.__setattr__(self, "_cache", {})
        object.__setattr__(
            self,
            "_prop_index",
            {prop.name: i for i, prop in enumerate(self.properties)},
        )
        object.__setattr__(self, "_digest", _digest_bytes(codes, self.parent_row))

    @classmethod
    def from_property(cls, prop: Property) -> PropertyMap:
        """Bootstrap a map whose compartments are ``prop``'s traits."""
        n = len(prop.traits)
        codes = np.arange(n, dtype=np.int16).reshape(n, 1)
        return cls(properties=(prop,), codes=codes)

    @classmethod
    def from_properties(cls, props: Sequence[Property]) -> PropertyMap:
        """Bootstrap a fully-crossed map from one or more properties."""
        if not props:
            raise ValueError("from_properties requires at least one property.")
        pmap = cls.from_property(props[0])
        for prop in props[1:]:
            pmap = pmap.stratify(prop)
        return pmap

    @property
    def size(self) -> int:
        """Number of compartments (rows)."""
        return int(self.codes.shape[0])

    @property
    def n_properties(self) -> int:
        """Number of properties (columns)."""
        return int(self.codes.shape[1])

    def __len__(self) -> int:
        """Number of compartments (same as :attr:`size`)."""
        return self.size

    def get_property(self, name: str) -> Property:
        """Return the registered property called ``name``."""
        try:
            return self.properties[self._prop_index[name]]
        except KeyError:
            raise KeyError(f"Unknown property {name!r}. Known: {list(self._prop_index)}") from None

    def column_index(self, prop: Property | str) -> int:
        """Return the column index of ``prop`` on this map."""
        return self._prop_index[self._resolve_property(prop).name]

    def column(self, prop: Property | str) -> NDArray[np.int16]:
        """Return the ``int16`` code column for ``prop``."""
        return self.codes[:, self.column_index(prop)]

    def label(self, row: int) -> str:
        """Return a human-readable label for compartment ``row``."""
        parts = [
            f"{prop.name}={prop.traits[int(code)]}"
            for prop, code in zip(self.properties, self.codes[row], strict=True)
            if code != _NA
        ]
        return "_".join(parts)

    def copy(self) -> PropertyMap:
        """Return a map with the same table and an empty query cache."""
        return PropertyMap(
            properties=self.properties,
            codes=self.codes,
            history=self.history,
            parent_row=self.parent_row,
        )

    def stratify(self, prop: Property, where: Selector | None = None) -> PropertyMap:
        """Return a new map with ``prop`` applied to matching compartments."""
        return self._apply_stratification(Stratification(prop, where))

    def _apply_stratification(self, strat: Stratification) -> PropertyMap:
        prop = strat.property
        if prop.name in self._prop_index:
            raise ValueError(f"Property {prop.name!r} is already on this map.")
        if strat.where is None:
            matched = np.ones(self.size, dtype=bool)
        else:
            self._validate_selector(strat.where)
            matched = self.kleene(strat.where) == _TRUE

        k = len(prop.traits)
        reps = np.where(matched, k, 1).astype(np.int32, copy=False)
        row_src = np.repeat(np.arange(self.size, dtype=np.int32), reps)
        gathered = self.codes[row_src]
        new_col = np.full(row_src.shape[0], _NA, dtype=np.int16)
        n_matched = int(matched.sum())
        if n_matched:
            new_col[matched[row_src]] = np.tile(np.arange(k, dtype=np.int16), n_matched)
        new_codes = np.concatenate([gathered, new_col.reshape(-1, 1)], axis=1)
        return PropertyMap(
            properties=(*self.properties, prop),
            codes=new_codes,
            history=(*self.history, strat),
            parent_row=row_src,
        )

    def mask(self, sel: Selector) -> NDArray[np.bool_]:
        """Return a boolean mask of compartments where ``sel`` is true."""
        self._validate_selector(sel)
        return np.asarray(self.kleene(sel) == _TRUE, dtype=np.bool_)

    def select(self, sel: Selector) -> NDArray[np.int32]:
        """Return integer indices of compartments where ``sel`` is true."""
        return np.flatnonzero(self.mask(sel)).astype(np.int32, copy=False)

    def select_one(self, sel: Selector) -> int:
        """Return the single index matching ``sel``, or raise."""
        indices = self.select(sel)
        if indices.size != 1:
            raise ValueError(f"Expected exactly one compartment for {sel!r}, found {indices.size}.")
        return int(indices[0])

    def partition(self, prop: Property | str) -> dict[Trait, NDArray[np.int32]]:
        """Return per-trait index arrays covering ``prop.present()``.

        Includes empty groups. Keys are unwrapped :class:`~summer4.properties.Trait`
        values (single property).
        """
        resolved = self._resolve_property(prop)
        col = self.column(resolved)
        return {
            resolved.trait(name): np.flatnonzero(col == code).astype(np.int32, copy=False)
            for code, name in enumerate(resolved.traits)
        }

    def group_by(self, *props: Property | str) -> Groups[NDArray[np.int32]]:
        """Return groups for each existing combination of ``props``.

        Compartments missing any of the requested properties are omitted.
        Groups are ordered lexicographically by trait code. Empty combinations
        are not included. Keys are always ``tuple[Trait, ...]``.
        """
        if not props:
            raise ValueError("group_by requires at least one property.")
        resolved = [self._resolve_property(prop) for prop in props]
        columns = np.column_stack([self.column(p) for p in resolved])
        present = np.all(columns != _NA, axis=1)
        if not np.any(present):
            return Groups(_data={})
        present_idx = np.flatnonzero(present).astype(np.int32, copy=False)
        present_codes = columns[present]
        unique, inverse = np.unique(present_codes, axis=0, return_inverse=True)
        order = np.lexsort(unique.T[::-1])
        data: dict[tuple[Trait, ...], NDArray[np.int32]] = {}
        for group_pos in order:
            key_codes = unique[group_pos]
            traits = tuple(
                resolved[i].trait(resolved[i].traits[int(code)]) for i, code in enumerate(key_codes)
            )
            members = present_idx[inverse == group_pos]
            data[traits] = members
        return Groups(_data=data)

    def labels(self) -> tuple[str, ...]:
        """Return a human-readable label for every compartment."""
        return tuple(self.label(i) for i in range(self.size))

    def to_dicts(self) -> list[dict[str, str]]:
        """Return each compartment as ``{property: trait}``, omitting absent properties."""
        return [self._as_dict(i) for i in range(self.size)]

    def to_frame(self) -> object:
        """Return a polars DataFrame with one column per property.

        Absent properties are null. Polars is imported lazily so the taxonomy
        package stays NumPy-only at import time.
        """
        import polars as pl

        columns: dict[str, list[str | None]] = {prop.name: [] for prop in self.properties}
        for row in range(self.size):
            for prop, code in zip(self.properties, self.codes[row], strict=True):
                columns[prop.name].append(None if code == _NA else prop.traits[int(code)])
        return pl.DataFrame(columns)

    def _as_dict(self, row: int) -> dict[str, str]:
        return {
            prop.name: prop.traits[int(code)]
            for prop, code in zip(self.properties, self.codes[row], strict=True)
            if code != _NA
        }

    def _resolve_property(self, prop: Property | str) -> Property:
        if isinstance(prop, str):
            return self.get_property(prop)
        existing = self.get_property(prop.name)
        if existing.traits != prop.traits:
            raise ValueError(
                f"Property {prop.name!r} is registered with traits {existing.traits}, "
                f"not {prop.traits}."
            )
        return existing

    def _validate_selector(self, sel: Selector) -> None:
        if not isinstance(sel, SelectorOps):
            raise TypeError(f"Expected a Selector, got {type(sel).__name__}.")
        match sel:
            case And(left=left, right=right):
                self._validate_selector(left)
                self._validate_selector(right)
            case Or(left=left, right=right):
                self._validate_selector(left)
                self._validate_selector(right)
            case Not(inner=inner):
                self._validate_selector(inner)
            case Everything() | Nothing():
                return
            case Present(property=name) | Absent(property=name):
                self.get_property(name)
            case IsIn(property=name, names=names):
                prop = self.get_property(name)
                unknown = [trait for trait in names if trait not in prop.traits]
                if unknown:
                    raise KeyError(
                        f"Unknown trait(s) {unknown} for property {name!r}. "
                        f"Known: {list(prop.traits)}"
                    )
            case Trait() as trait:
                prop = self.get_property(trait.property)
                if trait.name not in prop.traits or prop._index[trait.name] != trait.code:
                    raise ValueError(
                        f"Trait {trait!r} does not match registered property {prop.name!r}."
                    )
            case Source() | Dest():
                raise TypeError("Source()/Dest() select flow edges, not compartments")
            case _:
                raise TypeError(f"Unsupported selector {type(sel).__name__}.")

    def kleene(self, sel: Selector) -> NDArray[np.int8]:
        """Evaluate ``sel`` to a cached Kleene ``int8`` array over compartments."""
        cached = self._cache.get(sel)
        if cached is not None:
            return cached
        result = np.ascontiguousarray(self._evaluate(sel), dtype=np.int8)
        result.flags.writeable = False
        self._cache[sel] = result
        return result

    def _evaluate(self, sel: Selector) -> NDArray[np.int8]:
        match sel:
            case And(left=left, right=right):
                return _as_int8(np.minimum(self.kleene(left), self.kleene(right)))
            case Or(left=left, right=right):
                return _as_int8(np.maximum(self.kleene(left), self.kleene(right)))
            case Not(inner=inner):
                return np.negative(self.kleene(inner), dtype=np.int8)
            case Everything():
                return np.full(self.size, _TRUE, dtype=np.int8)
            case Nothing():
                return np.full(self.size, _FALSE, dtype=np.int8)
            case Present(property=name):
                return _as_int8(np.where(self.column(name) != _NA, _TRUE, _FALSE))
            case Absent(property=name):
                return _as_int8(np.where(self.column(name) == _NA, _TRUE, _FALSE))
            case IsIn(property=name, names=names):
                prop = self.get_property(name)
                codes = np.array([prop._index[trait] for trait in names], dtype=np.int16)
                col = self.column(name)
                present = col != _NA
                hit = np.isin(col, codes)
                return _as_int8(np.where(present, np.where(hit, _TRUE, _FALSE), _UNKNOWN))
            case Trait() as trait:
                col = self.column(trait.property)
                present = col != _NA
                hit = np.where(col == trait.code, _TRUE, _FALSE)
                return _as_int8(np.where(present, hit, _UNKNOWN))
            case Source() | Dest():
                raise TypeError("Source()/Dest() select flow edges, not compartments")
            case _:
                raise TypeError(f"Unsupported selector {type(sel).__name__}.")

    def __hash__(self) -> int:
        return hash((self.properties, self.history, self._digest))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PropertyMap):
            return NotImplemented
        parent_equal = (self.parent_row is None and other.parent_row is None) or (
            self.parent_row is not None
            and other.parent_row is not None
            and np.array_equal(self.parent_row, other.parent_row)
        )
        return (
            self.properties == other.properties
            and self.history == other.history
            and np.array_equal(self.codes, other.codes)
            and parent_equal
        )

    def __repr__(self) -> str:
        names = tuple(prop.name for prop in self.properties)
        lines = [f"PropertyMap(n={self.size}, properties={names})"]
        max_rows = 20
        for i in range(min(self.size, max_rows)):
            lines.append(f"  {i:>4}  {self.label(i)}")
        if self.size > max_rows:
            lines.append(f"  ... ({self.size - max_rows} more)")
        return "\n".join(lines)
