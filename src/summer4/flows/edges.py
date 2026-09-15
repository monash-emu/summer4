"""Flow edge maps: doubled compartment tables with ``@source`` / ``@dest`` columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from summer4.properties import Property, Trait
from summer4.propertymap import PropertyMap
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
_MARKER_TRAIT: str = "yes"


def _freeze_int16(values: NDArray[np.int16]) -> NDArray[np.int16]:
    array = np.ascontiguousarray(values, dtype=np.int16)
    array.flags.writeable = False
    return array


def _freeze_int32(values: NDArray[np.int32]) -> NDArray[np.int32]:
    array = np.ascontiguousarray(values, dtype=np.int32)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class EdgeRoles:
    """How the join classified each property. Provenance, not truth."""

    bound: frozenset[str]
    free: frozenset[str]
    dest_only: frozenset[str]
    src_only: frozenset[str]
    paired: frozenset[str]


def _rename(sel: Selector, suffix: str) -> Selector:
    """Rewrite compartment property names to mangled ``{name}{suffix}`` columns."""
    match sel:
        case Trait() as trait:
            return Trait(property=f"{trait.property}{suffix}", name=trait.name, code=trait.code)
        case IsIn(property=name, names=names):
            return IsIn(property=f"{name}{suffix}", names=names)
        case Present(property=name):
            return Present(property=f"{name}{suffix}")
        case Absent(property=name):
            return Absent(property=f"{name}{suffix}")
        case And(left=left, right=right):
            return And(_rename(left, suffix), _rename(right, suffix))
        case Or(left=left, right=right):
            return Or(_rename(left, suffix), _rename(right, suffix))
        case Not(inner=inner):
            return Not(_rename(inner, suffix))
        case Everything() | Nothing():
            return sel
        case Source() | Dest():
            raise TypeError("Source()/Dest() cannot be nested inside Source()/Dest().")
        case _:
            raise TypeError(f"Unsupported selector {type(sel).__name__}.")


def _gate(gate: Trait | None, rewritten: Selector) -> Selector:
    if gate is None:
        return rewritten
    return gate & rewritten


def _marker_gates(table: PropertyMap) -> tuple[Trait | None, Trait | None]:
    """Return ``(src_gate, dest_gate)`` derived from marker columns on ``table``.

    A gate is needed exactly when its marker column is ``-1`` throughout — the
    same condition :meth:`EdgeMap.from_indices` uses at construction.
    """
    src_marker = table.get_property("@source")
    dest_marker = table.get_property("@dest")
    src_col = np.asarray(table.column("@source"))
    dest_col = np.asarray(table.column("@dest"))
    src_gate = src_marker.trait(_MARKER_TRAIT) if np.all(src_col == _NA) else None
    dest_gate = dest_marker.trait(_MARKER_TRAIT) if np.all(dest_col == _NA) else None
    return src_gate, dest_gate


def rewrite_edge_selector(table: PropertyMap, sel: Selector) -> Selector:
    """Lower ``Source``/``Dest`` against an edge ``table`` alone (no ``EdgeMap``).

    Gates are derived from the ``@source`` / ``@dest`` marker columns so a
    :class:`~summer4.results.trace.Trace` can select edges without holding an
    unhashable :class:`EdgeMap` in pytree aux.
    """
    src_gate, dest_gate = _marker_gates(table)

    def _rewrite(node: Selector) -> Selector:
        match node:
            case Source(inner=inner):
                return _gate(src_gate, _rename(inner, "@source"))
            case Dest(inner=inner):
                return _gate(dest_gate, _rename(inner, "@dest"))
            case And(left=left, right=right):
                return And(_rewrite(left), _rewrite(right))
            case Or(left=left, right=right):
                return Or(_rewrite(left), _rewrite(right))
            case Not(inner=inner):
                return Not(_rewrite(inner))
            case Everything() | Nothing():
                return node
            case _:
                raise TypeError(
                    f"{type(node).__name__} is a compartment selector; on a flow it must "
                    "be wrapped in Source(...) or Dest(...)."
                )

    return _rewrite(sel)


def _compartment_props_from_table(table: PropertyMap) -> tuple[Property, ...]:
    """Recover demangled compartment properties from an edge table's columns."""
    seen: dict[str, Property] = {}
    for prop in table.properties:
        name = prop.name
        if name in ("@source", "@dest"):
            continue
        if name.endswith("@source"):
            base = name[: -len("@source")]
        elif name.endswith("@dest"):
            base = name[: -len("@dest")]
        else:
            continue
        if base not in seen:
            seen[base] = Property._mangled(base, prop.traits)
    return tuple(seen.values())


def _side_label_from_table(table: PropertyMap, side: str, row: int) -> str:
    suffix = f"@{side}"
    parts: list[str] = []
    for prop in _compartment_props_from_table(table):
        code = int(table.column(f"{prop.name}{suffix}")[row])
        if code == _NA:
            continue
        parts.append(f"{prop.name}={prop.traits[int(code)]}")
    return "_".join(parts)


def edge_labels(table: PropertyMap) -> tuple[str, ...]:
    """Demangled edge labels such as ``state=S_age=0-4 -> state=I_age=0-4``."""
    src_col = np.asarray(table.column("@source"))
    dest_col = np.asarray(table.column("@dest"))
    out: list[str] = []
    for row in range(table.size):
        src = _side_label_from_table(table, "source", row)
        dest = _side_label_from_table(table, "dest", row)
        has_src = int(src_col[row]) != _NA
        has_dest = int(dest_col[row]) != _NA
        if not has_src:
            out.append(f"-> {dest}")
        elif not has_dest:
            out.append(f"{src} ->")
        else:
            out.append(f"{src} -> {dest}")
    return tuple(out)


def sum_over_edge(
    mass: object,
    table: PropertyMap,
    prop: Property | str,
    side: str,
) -> Any:
    """Sum per-edge ``mass`` by trait of ``prop`` on ``side`` (``source`` or ``dest``).

    Returns a :class:`~summer4.jax.propertydata.PropertyData` over
    :meth:`PropertyMap.from_property` so group columns are labelled by trait name.
    """
    import jax
    import jax.numpy as jnp

    from summer4.jax.propertydata import PropertyData

    if side not in ("source", "dest"):
        raise ValueError(f"side must be 'source' or 'dest', got {side!r}.")
    if isinstance(prop, Property):
        resolved = prop
    else:
        mangled_lookup = table.get_property(f"{prop}@{side}")
        resolved = Property(prop, mangled_lookup.traits)

    mangled = table.get_property(f"{resolved.name}@{side}")
    col_i = table.column_index(mangled)
    codes = np.asarray(table.codes[:, col_i], dtype=np.int32)
    n_traits = len(resolved.traits)
    valid = codes >= 0
    safe = np.where(valid, codes, 0).astype(np.int32)
    mass_j = jnp.asarray(mass)
    weighted = mass_j * jnp.asarray(valid)

    def _row(row: Any) -> Any:
        return jax.ops.segment_sum(row, safe, num_segments=n_traits)

    if mass_j.ndim == 1:
        reduced: Any = _row(weighted)
    else:
        flat = jnp.reshape(weighted, (-1, mass_j.shape[-1]))
        summed = jax.vmap(_row)(flat)
        reduced = jnp.reshape(summed, mass_j.shape[:-1] + (n_traits,))
    return PropertyData(PropertyMap.from_property(resolved), reduced)


@dataclass(frozen=True, slots=True)
class EdgeMap:
    """Edges over a compartment map, stored as a doubled :class:`PropertyMap` table.

    Rows are edges, not compartments. ``table`` columns are ``{p}@source``,
    ``{p}@dest`` for each compartment property ``p``, plus marker properties
    ``@source`` and ``@dest`` (trait ``yes``, code 0 when that endpoint exists,
    ``-1`` when it does not).
    """

    pmap: PropertyMap
    table: PropertyMap
    src_idx: NDArray[np.int32] | None
    dest_idx: NDArray[np.int32] | None
    roles: EdgeRoles
    _src_gate: Trait | None
    _dest_gate: Trait | None

    def __post_init__(self) -> None:
        if self.src_idx is not None:
            object.__setattr__(self, "src_idx", _freeze_int32(self.src_idx))
        if self.dest_idx is not None:
            object.__setattr__(self, "dest_idx", _freeze_int32(self.dest_idx))

    @classmethod
    def from_indices(
        cls,
        pmap: PropertyMap,
        src_idx: NDArray[np.int32] | None,
        dest_idx: NDArray[np.int32] | None,
        roles: EdgeRoles,
    ) -> EdgeMap:
        """Build an edge map from parallel source and/or destination indices."""
        if src_idx is None and dest_idx is None:
            raise ValueError("EdgeMap requires at least one of src_idx or dest_idx.")
        if src_idx is not None and dest_idx is not None and src_idx.size != dest_idx.size:
            raise ValueError(
                f"src_idx length {src_idx.size} does not match dest_idx length {dest_idx.size}."
            )
        n = int(src_idx.size if src_idx is not None else dest_idx.size)  # type: ignore[union-attr]
        n_props = pmap.n_properties
        src_block = np.full((n, n_props), _NA, dtype=np.int16)
        dest_block = np.full((n, n_props), _NA, dtype=np.int16)
        if src_idx is not None:
            src_block = np.asarray(pmap.codes[src_idx], dtype=np.int16)
        if dest_idx is not None:
            dest_block = np.asarray(pmap.codes[dest_idx], dtype=np.int16)

        properties: list[Property] = []
        columns: list[NDArray[np.int16]] = []
        for i, prop in enumerate(pmap.properties):
            properties.append(Property._mangled(f"{prop.name}@source", prop.traits))
            properties.append(Property._mangled(f"{prop.name}@dest", prop.traits))
            columns.append(src_block[:, i])
            columns.append(dest_block[:, i])
        src_marker = Property._mangled("@source", (_MARKER_TRAIT,))
        dest_marker = Property._mangled("@dest", (_MARKER_TRAIT,))
        properties.append(src_marker)
        properties.append(dest_marker)
        src_codes = np.full(n, 0 if src_idx is not None else _NA, dtype=np.int16)
        dest_codes = np.full(n, 0 if dest_idx is not None else _NA, dtype=np.int16)
        columns.append(src_codes)
        columns.append(dest_codes)
        table = PropertyMap(
            properties=tuple(properties),
            codes=_freeze_int16(
                np.column_stack(columns) if columns else np.zeros((n, 0), np.int16)
            ),
        )
        src_gate = None if src_idx is not None else src_marker.trait(_MARKER_TRAIT)
        dest_gate = None if dest_idx is not None else dest_marker.trait(_MARKER_TRAIT)
        return cls(
            pmap=pmap,
            table=table,
            src_idx=src_idx,
            dest_idx=dest_idx,
            roles=roles,
            _src_gate=src_gate,
            _dest_gate=dest_gate,
        )

    @property
    def n_edges(self) -> int:
        return self.table.size

    def rewrite(self, sel: Selector) -> Selector:
        """Lower ``Source``/``Dest`` to mangled compartment selectors on ``table``."""
        return rewrite_edge_selector(self.table, sel)

    def _validate_edge_selector(self, sel: Selector) -> None:
        if not isinstance(sel, SelectorOps):
            raise TypeError(f"Expected a Selector, got {type(sel).__name__}.")
        match sel:
            case Source(inner=inner) | Dest(inner=inner):
                self.pmap._validate_selector(inner)
            case And(left=left, right=right) | Or(left=left, right=right):
                self._validate_edge_selector(left)
                self._validate_edge_selector(right)
            case Not(inner=inner):
                self._validate_edge_selector(inner)
            case Everything() | Nothing():
                return
            case _:
                raise TypeError(
                    f"{type(sel).__name__} is a compartment selector; on a flow it must "
                    "be wrapped in Source(...) or Dest(...)."
                )

    def kleene(self, sel: Selector) -> NDArray[np.int8]:
        """Evaluate ``sel`` to a Kleene ``int8`` array over edges."""
        self._validate_edge_selector(sel)
        return self.table.kleene(self.rewrite(sel))

    def mask(self, sel: Selector | NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Return a boolean mask of edges where ``sel`` is Kleene-true."""
        if isinstance(sel, np.ndarray):
            if sel.shape != (self.table.size,):
                raise ValueError(
                    f"Boolean mask shape {sel.shape} does not match n_edges {self.table.size}."
                )
            return np.asarray(sel, dtype=np.bool_)
        return np.asarray(self.kleene(sel) == np.int8(1), dtype=np.bool_)

    def select(self, sel: Selector | NDArray[np.bool_]) -> NDArray[np.int32]:
        """Return integer indices of edges where ``sel`` is true (or ``sel`` is a mask)."""
        return np.flatnonzero(self.mask(sel)).astype(np.int32, copy=False)

    def moves_mask(self, prop: Property | str) -> NDArray[np.bool_]:
        """True where ``prop`` is present on both ends and the codes differ."""
        name = prop.name if isinstance(prop, Property) else prop
        self.pmap.get_property(name)
        src_col = self.table.column(f"{name}@source")
        dest_col = self.table.column(f"{name}@dest")
        both = (src_col != _NA) & (dest_col != _NA)
        return np.asarray((src_col != dest_col) & both, dtype=np.bool_)

    def labels(self) -> tuple[str, ...]:
        """Demangled labels such as ``state=S_age=0-4 -> state=I_age=0-4``."""
        return edge_labels(self.table)
